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

The separate units `Numerics` (polynomials, jets, norms; `numerics.hpp`), `Transcendentals`
(`transcendental.hpp`) and `Linalg` (dense products; `linalg.hpp`, below) also encode into an engine's
batch, on buffers of that engine. Construct them
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
auto exp_pass = tr.run(batch, Function::complex_exp, z, e);    // GPU pass + GPU retry rungs
nm.poly_eval(batch, Polynomial{n, terms, n, true}, coeffs, e, v);
nm.norm_inf(batch, Segments{1, n, true}, v, norm, info);
auto done = batch.submit();
done.wait();                                            // also runs the (rare) host final step of exp_pass
// exp_pass.report(): retried, resolved per GPU rung and by the host, in_batch_reads, provisional_reads
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

**Transcendental passes are final inside the batch when the GPU decides every element** (round 44). The
first pass certifies each element and appends the undecided ones to a resident compacted list; up to three
retry rungs, encoded right after it in the same batch, re-evaluate the listed elements at more words
(`docs/numerics.md`, "Transcendental functions") and compact the still undecided ones for the next rung.
Every dispatch is separated from the next by the batch's buffer barrier, so operations encoded later in
the batch read the final values of all elements that a GPU rung resolved. Elements still undecided after
the last GPU rung are finished by the host ladder in `Submission::wait()` (or the submission's destructor), which patches `out` before it releases the buffers.
Random inputs essentially never reach that step. Hard inputs whose decision needs more than the 35-word
cap of the GPU rungs (short dyadics near a rounding midpoint need about `2N` words) do: in the test sets none
up to 352 bits, one per width at 384–544 bits, most of the ~100 hard retries per width from 576 bits (the
third rung runs on the host from 512 bits, the second and third at 992–1024;
`benchmarks/results/round44_transcendental_tests_per_width.txt`).
After `wait()` the output equals the host-array call bit for bit, and before it the output cannot be
mapped, downloaded or used by another batch (it is busy). Within the encoding batch:

- reading `out` (as `poly_eval` above) gives final values unless the pass left elements for the host;
  then those elements hold the RN value of the last GPU rung's approximation (provisional). After
  `wait()` the ticket's `report()` gives `retried`, the per-rung counts `resolved[0..2]` (GPU rungs of
  `rung_words[0..2]` words) and `resolved[3]` (host), `host_retried()`, `in_batch_reads` (a later
  operation read `out`) and `provisional_reads` (it did so while `host_retried() > 0`: recompute the
  dependent results, or split the batch at the pass);
- writing `out` again (any engine or unit operation, or a second pass) throws `logic_error`, because the
  host final step may still patch it;
- the levels read the inputs directly, without a snapshot: they run before any later operation of the
  batch, a level writes only elements it decided (an input aliased by `out` stays intact for undecided
  elements), and the last GPU level saves the operands of the elements it leaves for the host. Each rung
  is an indirect dispatch sized on the GPU from the previous level's counter (a one-thread kernel writes
  the arguments), so a pass without retries costs two tiny dispatches per rung; a rung with work adds 1.2–5 ms
  of GPU time (one GPU thread per element at up to 35 words), more than the host ladder needs for a few
  elements, which is the price of final values inside the batch. Scratch per pass:
  two `4·count`-byte lists with a 64-byte header (counters, dispatch arguments), and a `count`-element
  operand buffer whose pages are touched only by elements left for the host; all released at `wait()`.
  `powi` has no retries.
- `ticket.resolved()` is false and `ticket.report()` throws until the submission is waited; a batch
  discarded without submission leaves its tickets unresolved. A failing host final step makes `wait()`
  throw.
- `transcendental_force_level(level)` (namespace `detail`, a test hook) makes the levels below `level`
  treat every element as undecided, to exercise a rung or the host final step.

**Dense products (`Linalg`, `linalg.hpp`).** `Linalg la(gpu)` encodes `syrk` and `gemm` (also the
`subtract` updates) on `Buffer<Float<bits>>` operands; band analysis, member lists and the modulus count
are GPU passes of the batch that size the product dispatches indirectly. They follow the transcendental
pattern: the call returns a `LinalgTicket`, and outputs of lines that the host path computes with exact
host dot products (rare: spreads above `max_spread`, more than `max_bands` bands) are written at
`wait()`, so C is provisional in its batch (no second write; reads are flagged in
`ticket.provisional_reads()` and matter only when `ticket.report().fallback_outputs > 0`).

```cpp
Linalg la(gpu);
auto batch = gpu.batch();
auto t1 = la.gemm(batch, true, J, R, m, n, k, C1);         // C1 = J^T R
auto t2 = la.syrk(batch, C1, m, n, G, true);               // G = C1^T C1 (lower); reads C1 (flagged)
auto t3 = la.syrk(batch, J, k, m, H, true, true);           // H = RN(H - J^T J), lower triangle
batch.submit().wait();
```

The calls of one batch share the `Linalg`'s per-batch workspace (digit planes, residues, accumulators);
batches pending at the same time use separate workspaces. Workspace sizes are bounded independently of
the data by `LinalgOptions::resident_bands`; a call that needs more band rows computes on the host at
`wait()` (`ticket.report().resident_overflow`). Contract, bounds and memory: `docs/numerics.md`,
"Resident products".

`cholesky`, `trsm`, `cholesky_solve`, `factor_qr`, `QRFactor::solve` and `apply_q` take `Buffer` operands
too, but run synchronously outside any batch (host panels sit between dependent GPU updates): the call
owns the idle buffers like a submission until it returns, uses them on the GPU in place, and returns
the host-array results bit for bit.

Library-internal hook: `src/engine_internal.hpp` (ObjC++, not installed) exposes the engine's device
and queue, the batch's command buffer (the batch identity for per-batch workspaces) and compute encoder
(with the barrier), in-batch copies, ownership and scratch, Metal object retention, provisional outputs
and completion steps to library units.

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
the exact complex `fma` can take several seconds for a new width (fused vector recurrences 0.3–4 s since round 46). The OS shader cache
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
