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

## Resident units

The separate units `Numerics` (polynomials, jets, norms; `numerics.hpp`) and `Transcendentals`
(`transcendental.hpp`) also encode into an engine's batch, on buffers of that engine. Construct them
from the engine so that their pipelines are compiled on its Metal device (on single-GPU Macs the
default constructor selects the same device; a mismatch throws `invalid_argument`):

```cpp
Engine gpu;
Numerics nm(gpu);
Transcendentals tr(gpu);
using C = Complex<7>;                                   // 224 bits
auto a = gpu.make_buffer<C>(n), b = gpu.make_buffer<C>(n), z = gpu.make_buffer<C>(n);
auto e = gpu.make_buffer<C>(n), v = gpu.make_buffer<C>(n), coeffs = gpu.make_buffer<C>(terms);
auto norm = gpu.make_buffer<Float<224>>(1);
auto info = gpu.make_buffer<NormInfo>(1);
// ... upload a, b, coeffs ...
auto batch = gpu.batch();
batch.run(Operation::complex_mul, a, b, z);
auto exp_pass = tr.run(batch, Function::complex_exp, z, e);    // GPU pass + resident retry list
nm.poly_eval(batch, Polynomial{n, terms, n, true}, coeffs, e, v);
nm.norm_inf(batch, Segments{1, n, true}, v, norm, info);
auto done = batch.submit();
done.wait();                                            // also runs the host retries of exp_pass
// exp_pass.report(): retried, resolved per rung, provisional_reads
```

- Every resident unit call is bit-identical to the corresponding host-array call: the same pipelines
  and dispatch sequence run on the batch's encoder. Ownership is the engine's: foreign buffers throw
  `invalid_argument`, buffers of an unwaited submission `logic_error`, and buffer sizes and shapes are
  checked before encoding (buffers may be larger than required). `Buffer<NormInfo>` holds per-segment
  summaries; `Buffer<std::int32_t>` holds `powi` exponents.
- Multi-pass reductions (the norm tree levels) are ordinary dispatches of the batch, separated by
  buffer barriers. Their scratch (summary words, block roots) is allocated per call, initialised on
  the host while encoding, and retained by the submission until it is waited.
- The batch, its submission and transcendental tickets may outlive the unit objects and the engine.
  Pipelines and tables are retained by the batch. Unit pipeline caches are thread-safe as the
  engine's: a unit may encode into batches of several threads' engines (host-array calls of one unit
  still belong to one thread).

**Transcendental passes are final only at `wait()`.** The GPU certifies each element and appends the
undecided ones to a resident compacted list. `Submission::wait()` (or the submission's destructor)
then runs the host retry ladder for those elements and patches `out` *before* it releases the
buffers, so after `wait()` the output equals the host-array call bit for bit, and before it the
output cannot be mapped, downloaded or used by another batch (it is busy). Within the encoding
batch, `out` is provisional:

- writing it again (any engine or unit operation, or a second pass) throws `logic_error`;
- reading it (as `poly_eval` above) is allowed, but such an operation sees the GPU's provisional
  values for undecided elements. After `wait()` the ticket's `report()` gives `retried` and
  `provisional_reads`; when both are nonzero, recompute the dependent results (or split the batch at
  the pass). Random full-precision arguments are essentially never undecided; short dyadic arguments
  near the documented hard cases are (`docs/numerics.md`, "Transcendental functions").
- The pass snapshots its inputs inside the batch before evaluating (a blit copy), so `out` may alias
  the input and later operations may overwrite the input; the snapshot and the retry list
  (`4(count+1)` bytes) are released at `wait()`. `powi` has no retries and needs no snapshot.
- `ticket.resolved()` is false and `ticket.report()` throws until the submission is waited; a batch
  discarded without submission leaves its tickets unresolved. Host retries that fail make `wait()`
  throw.

Library-internal hook: `src/engine_internal.hpp` (ObjC++, not installed) exposes the engine's device
and queue, the batch's compute encoder (with the barrier), in-batch copies, ownership and scratch,
Metal object retention, provisional outputs and completion steps to library units.

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

## Compiling pipelines ahead of use

Each precision compiles one shader library (about 1 s) and each operation/mode its own pipeline on
first use. Most pipelines take milliseconds, but fused vector recurrences and other kernels that inline
the exact complex `fma` several times can take tens of seconds for a new width. The OS shader cache
normally makes later runs fast. To keep this latency out of a solver's first iteration, request the
pipelines up front:

```cpp
Prewarm request;
request.bits = {224, 256};
request.operations = {Operation::complex_mul, Operation::complex_fma};
request.vector_shapes = {shape};          // VectorRecurrence options that matter: matrix/affine/all_steps/reverse/tangent/fused
request.complex_dots = {SegmentedDot{0, 0, true}};
auto ready = gpu.prewarm_async(request);  // compile on a background thread
// ... CPU-side preparation; the Engine stays usable ...
ready.get();                              // rethrows compilation or request errors
```

`Engine::prewarm` does the same in the calling thread. The pipeline cache is thread-safe; compilation
runs outside its lock, so concurrent work on the same Engine is not blocked.
