# Batched products and resident solver building blocks

These APIs implement the second round in [`tips.md`](../tips.md#9-second-round-what-we-would-like-next-8-oct-2026).
They are general numerical operations: no QSC or Baxter dependency enters the library.
Include `limbforge/batched_linalg.hpp` and construct `BatchedLinalg numerical(engine)`.
One instance belongs to one host encoding thread; its Metal libraries and pipelines are cached.
Widths are 64–1024 bits in increments of 32. First use compiles a pipeline; prepare it before timing.

## Products and power moments

`gemm(batch, shape, A, B, C)` supports real and complex row-major buffers. `StridedGemm`
contains `count,m,n,k,stride_a,stride_b,stride_c`; strides count elements. Zero input stride
broadcasts a matrix, including a Fourier phase matrix shared across many vectors. Output
matrices must not overlap, and output storage must differ from input storage. Padding is
preserved. Empty dots produce zero, or retain the accumulator in accumulation mode.

```cpp
using Value = limbforge::Complex<11>; // 352 bits
limbforge::StridedGemm s{count, 4, 4, 4, 16, 16, 16};
auto batch = engine.batch();
numerical.gemm(batch, s, A, B, C);
numerical.product3(batch, s, s, A, B, D, ABC, true); // -(A B) D
batch.submit().wait();
```

`product3` accepts general compatible shapes, creates a resident intermediate, and encodes
two GEMMs. It does not perform a host readback. Both shapes must disable accumulation.
The final argument negates the final result; the intermediate is rounded normally.

`PowerMoments{count,steps,nmax,ncols,n0,accumulate,fused}` computes

```
out[b,n,c] (+)= sum_k (E[b,k] * y[b,k]^(n+n0)) * W[b,k,c]
```

for **n = 0 through nmax inclusive**. E/y are `[count][steps]`, W is
`[count][steps][ncols]`, and out is `[count][nmax+1][ncols]`.
`power_moments(batch, shape, E, y, W, out)` generates the left matrix on device
and reuses the tiled GEMM. It removes the host Vandermonde construction and upload,
but currently still allocates that table in device memory. Negative n0 uses a reciprocal;
a zero denominator propagates arithmetic status. Runtime-width synchronous host forms
of both `gemm(bits,complex,shape,A,B,C)` and `power_moments(...)` return `Timing`.

Products visit k in ascending order. By default, complex multiplication uses the existing
four separately rounded real products, followed by rounded real add/subtract, and each
dot update uses a separate rounded addition. `fused=true` uses fma/cfma for each update.
Accumulation starts from the supplied output; `negative=true` negates the completed
GEMM result, including its initial accumulator. Power generation uses composed integer
exponentiation by squaring at n0, then repeated composed multiplication by y; E is
multiplied afterwards. Thus the result has a deterministic **sequence of roundings**,
not the single-round exact-dot contract of `Linalg::gemm`.

The initial optimization is an 8×4 output tile with cooperative shared-memory loads and
rolled k loops. A specialized 4×4 path packs two matrices per SIMD group so that every
lane produces an output. It does not implement a three-product complex algorithm or TensorOps
complex GEMM. Large-shape throughput is still to be measured.

## Normal equations, damping trials and reusable factors

Two contracts are explicit:

* `normal_equations(batch,J,g,rows,cols,A,rhs,fused=false)` returns JᵀJ and Jᵀg
  in one dispatch, with ascending-row composed or fused dot updates. Results are final
  on device and can feed a factorization in the same batch.
* `normal_equations_exact(linalg,batch,J,g,rows,cols,A,rhs)` packs `[J g]` on device
  and calls one exact augmented SYRK. Its `LinalgTicket` resolves at `wait()`. A completion
  step extracts the **repaired** exact products after any host fallback; when no fallback
  occurs, extraction remains on the GPU and the host copy is skipped. Wait before feeding
  those outputs to a new resident operation. The new APIs reject an input that still needs
  host completion, rather than consuming a provisional value.

```cpp
auto batch = engine.batch();
numerical.normal_equations(batch, J, g, rows, cols, A, rhs);
numerical.cholesky_trials(batch, {cols, trials}, A, diagonal, mu, L, status);
numerical.cholesky_solve(batch, L, status, trials, cols, rhs, 1, X);
auto submission = batch.submit();
submission.wait();
```

`cholesky_trials` factors `A + mu[t]*diag(D)` for all trials. A is a shared row-major
`[n][n]` matrix whose lower triangle is used, D is `[n]`, mu is `[count]`, L is
`[count][n][n]`, and status is `[count]` of `uint32_t`. Columns run in ascending order:
round sqrt of the pivot, round each division, then subtract separately rounded products
from the trailing lower triangle. The diagonal damping update is a separately rounded
multiply and add. This differs from the blocked exact-update `Linalg::cholesky` sequence.

Status is zero on success, otherwise the first unusable pivot plus one (one-based).
Remaining lower factor columns carry `invalid`; the upper triangle is canonical zero.
The solve returns canonical zero with `invalid` for a failed trial. It supports a shared
B `[n][nrhs]` and writes X `[count][n][nrhs]`, with ascending forward and backward dot
indices and separately rounded multiply/subtract/divide. Keep L/status resident for
several solves or use them in subsequent batches after waiting. Inputs and X must differ.

No host panel or intermediate readback occurs. Factorization currently uses scalar-column
pivot/column/update passes, with shrinking update grids; it is a resident baseline rather
than a claim to outperform the existing blocked Cholesky. Each solve uses one dispatch,
with one thread owning an entire trial/RHS column. Large-factor performance and stability
comparisons belong to the subsequent full audit.

## Recurrence with polynomial sources

`polynomial_recurrence(batch,shape,start,cp,cq,y,Ep,Eq,out)` evaluates sources and applies
`v <- v + p*(q^T v)` for each four-component trajectory without allocating p/q tables.

* cp/cq: `[coefficient_sets][4][terms]` in **ascending degree**.
* coefficient_sets: one shared set or `ceil(lanes/lanes_per_weight)` sets.
* y: `[steps][weight_groups]`; Ep/Eq: `[steps][4][weight_groups]`.
* start: component-major `[4][lanes]`; out: the same, or `[steps+1][4][lanes]`.

The sources are `p_a = Ep_a*Horner(cp_a,y)` and `q_a = Eq_a*Horner(cq_a,y)`.
Horner runs from the highest degree down, source scaling uses composed multiplication,
and the rank-one dot visits components 0,1,2,3 before updating any component.
`fused` selects cfma for Horner and recurrence updates. `reverse` applies steps in reverse
coefficient order; `all_steps` writes states in **application order**, starting with the
input state. Adjacent lanes may share weights; the final group may be incomplete.
Zero terms means the zero polynomial. Zero steps copies the initial state.

This variant covers the rank-one base trajectory. The existing `Engine::vector_recurrence`
retains its matrix, affine and tangent variants. Explicit source tables generated by
ascending powers may round differently from Horner, so integrations must compare the
documented sequences before accepting solver results. This branch builds on `3d0eb0c`: the
existing Engine recurrence supports fused mode through 1024 bits, and the new kernels reuse
its rolled exact complex-FMA primitive to keep shader compilation manageable.

## Bridges, external MPC storage and completion

`mpfr_bridge.hpp` adds runtime-width scalar and array overloads of `from_mpfr`, `to_mpfr`,
`from_mpc` and `to_mpc`. `bridge_element_bytes(bits,complex)` gives the element size.
Byte buffers need sufficient capacity; unaligned byte storage is supported via memcpy.
Rounding, statuses, array worker selection and error behavior follow the typed bridge.
The output is LimbForge's ordinary packed representation, not an MPFR structure.

With `mpc_staging.hpp`, call `describe_inline_mpc(bits,allocation,bytes,pointers,count)`
then `import_inline_complex(batch,allocation,bytes,records,count,out)`. The GPU reads
the original significands and writes ordinary resident Complex values. **No host limb
repacking occurs**, but this is not an alias between the incompatible MPC/LimbForge layouts.
Small exponent/sign/status metadata is copied. Precisions must already equal bits;
otherwise use the ordinary bridge. This includes both odd 32-bit limb counts (e.g. 352-bit
values in six 64-bit limbs) and even counts (e.g. 640 bits in ten limbs).

The allocation must own complete, page-aligned pages, be shared with Metal, and remain
alive and immutable until completion. Each significand must be aligned and within that
allocation. Little-endian 64-bit GMP limbs are required. NaN/Inf and out-of-range exponents
become the usual arithmetic status. MPC structures and the descriptor array need not
remain alive after encoding, because their metadata is copied.

`wait_all_async(std::vector<Submission>)` moves already-submitted work to a waiter thread
and returns `std::future<std::vector<Timing>>`. The CPU can do independent work, poll the
future, or wait on it. The helper drains every submission even if one fails, then rethrows
the first error. It preserves the usual rule: buffers become available after submission
completion and host repair. It is not a GPU event that bypasses those completion steps.

## Offline calibration and scope of this audit

`dispatch_policy.hpp` supplies a `BreakEvenTable`: record measured CPU/GPU wall times,
then look up an exact operation, precision, shape, worker count and fused-mode key.
The profile string identifies the machine/library/build configuration; `resident` separates
resident timing from host-array timing. A missing key or a near tie returns `unknown`.
There is no interpolation, universal cutoff, or timing on an application call. The table
recommends a backend; application code owns the CPU implementation and selection.
Use separate operation keys for further parameters, such as n0, layout and accumulation.

The `section9_limbforge` executable prepares CSV samples for complex GEMM and power moments
(composed mode, n0=0). CPU MPFR operands, outputs, scratch and the worker pool are allocated
before timing. GPU wall includes temporary allocation, staging, encoding, submission,
waiting and readback; it excludes shader compilation and bridge conversion. GPU device time
is also reported. All outputs of a warm sample are compared against the matching independent
MPFR sequence outside timing. Three repetitions are a smoke calibration, not a robust
performance study; record new profiles after changes or for a resident caller.

```sh
cmake --build build --target test_limbforge_section9 section9_limbforge -j4
ctest --test-dir build -R '^section9_smoke$' --output-on-failure
./build/section9_limbforge --operation gemm --bits 352 --count 32 --m 4 --n 4 --k 4 --workers 18 --repeats 3
./build/section9_limbforge --operation power --bits 352 --count 2 --m 4 --n 4 --k 7 --workers 18 --repeats 3
```

Recorded smoke measurements live in `benchmarks/results/section9_*_smoke.csv`.
They deliberately cover small cases; CPU is faster there. They are not cutoffs for the
400–900 batch workloads in §9.1, and the other kernels remain uncalibrated. Existing
benchmark headline claims are unchanged and do not describe these new kernels.

The light check covers 352-bit complex tiled GEMM tails/broadcast/padding and accumulation,
host GEMM, real 4×4 products with an odd batch and empty dots, triple products, negative
power moments, resident normal equations, successful
and failed damping trials, solves, asynchronous completion, and forced exact-dot fallback.
It also covers 352- and 704-bit composed/fused polynomial recurrences with reverse/all_steps and
sharing tails, and a 224-bit runtime bridge/inline MPC import. The existing threshold GPU
smoke passed separately. No full width sweep, large-factor audit, adversarial cancellation
suite, QSC column comparison, or end-to-end Δ/iteration measurement was run in this round.
The numbers in §9.0 are supplied integration observations and were not independently rerun.
