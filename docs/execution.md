# Resident buffers and asynchronous execution

Use typed buffers when a workload performs several operations on the same data.
Metal shared storage lets the GPU reuse those values between dispatches. A batch
encodes dependent operations into one command buffer, with an explicit buffer
barrier between dispatches. Precision comes from the buffer element type.

```cpp
#include <limbforge/engine.hpp>
#include <limbforge/mpfr_bridge.hpp>
#include <vector>

using namespace limbforge;
using F = Float<384>;
Engine gpu;
std::vector<F> host(65536, from_decimal<384>("1"));
std::vector<F> factors(host.size(), from_decimal<384>("1.01"));
auto x = gpu.make_buffer<F>(host.size());
auto y = gpu.make_buffer<F>(host.size());
x.upload(host.data(), host.size());
y.upload(factors.data(), factors.size());

auto batch = gpu.batch();
for (int step = 0; step < 16; ++step)
    batch.run(Operation::mul, x, y, x);
auto submission = batch.submit();
// The CPU can do independent work here.
auto timing = submission.wait();
x.download(host.data(), host.size());
```

`Buffer<Complex<12>>` selects 384-bit complex arithmetic. Real buffers accept
`add`, `sub`, `mul`, `div`, `square`, and `sqrt`; complex buffers accept `complex_add`,
`complex_mul`, and `complex_div`. Operand and output counts must match. Buffers
must belong to the engine creating the batch. These checks throw before submission.
In-place pointwise arithmetic is supported, including several dependent
operations in one batch.

## Unary operations and reductions

```cpp
auto total = gpu.make_buffer<F>(1);
auto batch = gpu.batch();
batch.run(Operation::square, x, x);
batch.tree_sum(x, total);
batch.run(Operation::sqrt, total, total);
auto timing = batch.submit().wait();
F norm;
total.download(&norm, 1);
```

The unary overload takes one input and one output of matching size. Negative
square-root inputs produce `invalid`. `tree_sum` accepts an arbitrary input size,
including zero, and an output of size one. It allocates up to two scratch buffers
and encodes all levels into the batch; the submission retains the scratch storage
until completion. For inputs larger than two, real reductions and complex
reductions through 384 bits use complete power-of-two threadgroups to compute several
levels in shared memory; the remaining group roots stay on the GPU. Wider
complex reductions use the global-memory tree. Each
adjacent pair rounds at each level. See the
[numerical contract](numerics.md) for the exact order and status behavior.

## Threadgroup tuning

`Engine gpu(EngineOptions{64})` requests 64 threads per group. Zero selects the
default policy: ordinarily 128; one SIMD group for recurrences and complex
multiplication/division at 384 bits or above. Explicit sizes must be powers of two
from 32 through 1024, and a multiple of the pipeline's SIMD width. The engine
clamps to the pipeline's legal maximum, rounding down to a complete SIMD group.
`gpu.pipeline_info(bits,op)` reports SIMD width, maximum threads, and the actual
selected group size for pointwise arithmetic. It compiles the pipeline on first use.

Cooperative reductions cap real groups at 128 threads and complex groups at 64,
and choose a complete power-of-two group within the pipeline limit.
`EngineOptions{0,false}` selects the global-memory reduction fallback for
comparison or investigation; it preserves the same numerical tree. The default
`cooperative_reductions=true` enables the measured policy rather than forcing
cooperation at every precision.

Workgroup choices can improve device throughput while having little effect on
end-to-end wall time. Use `tune_limbforge` or `benchmark_limbforge --threads N`
for measurements on the target device rather than assuming one size always wins.

## Ownership

- `mapped()` exposes shared storage. Only use the returned pointer while the
  buffer is idle, and finish all accesses before submitting work that uses it.
- `upload`, `download`, and new mapping requests throw while an unwaited
  submission owns the buffer. The caller must also stop using previously
  obtained pointers during this period; pointer writes cannot be intercepted.
- `ready()` polls completion. Call `wait()` before using the buffers again, even
  when `ready()` is true. Repeated waits return the same timing.
- Submission destruction waits safely and releases buffer ownership. Keep the
  submission object when asynchronous execution is intended.
- Batches and submissions retain their resources. They may outlive the original
  engine or buffer handles. An unsubmitted batch is discarded when destroyed.
- A batch is submitted once. A pending submission excludes the same buffers from
  another batch; submissions using independent buffers may be outstanding together.

Use one engine per calling host thread. Host mapping and submission of the same
buffer require coordination by the caller. Buffer handles share storage when
copied; copying a handle does not duplicate its values.

## Timing and the simpler API

`Timing::gpu_seconds` uses Metal command-buffer timestamps.
`Timing::wall_seconds` for a resident submission starts at batch construction and
ends when the first `wait()` completes. It includes encoding, submission, and any
CPU work before that wait. Uploads before construction and downloads after the
wait are separate costs. First-use shader compilation may occur during encoding;
warm each precision/operation before timing steady-state execution.

`Engine::run(bits, op, a, b, out, count)` remains a convenient synchronous host
array API. Its wall timing includes input/output copies, encoding, submission,
and waiting, but excludes pipeline compilation. Use resident execution to amortize
copies and command submission across a chain of arithmetic operations.
`run_unary(bits,op,a,out,count)` provides the corresponding synchronous interface
for square and square root.
