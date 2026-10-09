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

The authoritative rounding sequences, including power generation and composed/fused updates,
are in [Numerics: batched products](numerics.md#batched-products-and-polynomial-source-recurrences).
These differ from the single-round exact-dot contract of `Linalg::gemm`.

The initial optimization is an 8×4 output tile with cooperative shared-memory loads and
rolled k loops. A specialized 4×4 path packs two matrices per SIMD group so that every
lane produces an output. It does not implement a three-product complex algorithm or TensorOps
complex GEMM. Provisional large-shape measurements are in [the follow-up](section9-next.md);
idle-host confirmation and the full numerical audit remain pending.

## Normal equations, damping trials and reusable factors

Use `normal_equations` for immediately final, sequential products, or `normal_equations_exact`
for single-round exact dots. The [numerical contracts](numerics.md#batched-products-and-polynomial-source-recurrences)
and [completion rules](execution.md#batched-numerical-operations) explain which can be chained in one batch.

For independent calls at the measured n=944/K=1100, 352/384/448-bit profiles,
prefer the exact form: it is faster and has smaller measured global error in these
fixtures. See [timing and accuracy evidence](../benchmarks/experiments/section9_normal_equations.md).
The sequential form remains useful for immediate chaining as in the example below;
exact outputs must be waited and repaired before a dependent read.

```cpp
auto batch = engine.batch();
numerical.normal_equations(batch, J, g, rows, cols, A, rhs);
numerical.cholesky_trials(batch, {cols, trials}, A, diagonal, mu, L, status);
numerical.cholesky_solve(batch, L, status, trials, cols, rhs, 1, X);
auto submission = batch.submit();
submission.wait();
```

`cholesky_trials` factors a family of diagonally damped matrices; `cholesky_solve` supports
shared multiple right-hand sides. Their [scalar-column sequence and failure statuses](numerics.md#batched-products-and-polynomial-source-recurrences)
differ from `Linalg` factorization/solve contracts. Keep factors and statuses resident for reuse.
Layouts and aliasing rules are in [Execution](execution.md#batched-numerical-operations).

No host panel or intermediate readback occurs. Factorization currently uses scalar-column
pivot/column/update passes, with shrinking update grids; it is a resident baseline rather
than a claim to outperform the existing blocked Cholesky. Each solve uses one dispatch,
with one thread owning an entire trial/RHS column. Large-factor performance and stability
comparisons belong to the subsequent full audit.

## Recurrence with polynomial sources

`polynomial_recurrence(batch,shape,start,cp,cq,y,Ep,Eq,out)` evaluates sources and applies
`v <- v + p*(q^T v)` for each four-component trajectory without allocating p/q tables.

Coefficient ordering, Horner and recurrence rounding, sharing tails and empty inputs are
specified in [Numerics](numerics.md#batched-products-and-polynomial-source-recurrences).

The additive `PolynomialEvaluation::shared_sources` overload evaluates each shared group's
sources once. `polynomial_sources` also exposes p/q buffers for other device consumers.
Both are available since 1.3.0; validation and measured performance are recorded in [the experiment record](../benchmarks/experiments/section9_shared_sources.md).
Existing calls retain per-lane evaluation.

This variant covers the rank-one base trajectory. The existing `Engine::vector_recurrence`
retains its matrix, affine and tangent variants. Explicit source tables generated by
ascending powers may round differently from Horner, so integrations must compare the
documented sequences before accepting solver results. This branch builds on `3d0eb0c`: the
existing Engine recurrence supports fused mode through 1024 bits, and the new kernels reuse
its rolled exact complex-FMA primitive to keep shader compilation manageable.

## Bridges, external MPC storage and completion

Runtime-width scalar/array MPFR/MPC conversions follow the [typed bridge contract](numerics.md#mpfr-and-mpc-bridge).
External MPC imports read original significands on the GPU, with metadata copied on the host.
Allocation requirements, lifetime and `wait_all_async` completion are specified in
[Execution](execution.md#batched-numerical-operations). This import writes ordinary resident values;
it does not alias the incompatible MPC and WolfNum layouts.

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

### Follow-up reference runner

`test_limbforge_section9_audit` is a separate, opt-in target; it is excluded from the default
build and CTest so `section9_smoke` remains light. It reuses the smoke references and adds
real/complex tiled and 4×4 GEMM in both modes, cancellation/status inputs, padding and
broadcasts; real/complex triple products and accumulated empty dots; real/complex powers
at positive, zero and negative n0 across a row-tile boundary, with both modes and accumulation;
composed/fused normal equations with valid inputs and statuses; late/zero/error Cholesky pivots and multiple RHS;
polynomial modes, shared coefficients, sharing tails, empty terms/steps and statuses;
inline MPC full-width significands, NaN/Inf, zero and exponent boundaries.
Every width from 64 to 1024 bits in steps of 32 is compiled into the runner.
The subsequent 1.0.1 audit results are recorded below; the earlier harness preparation
and light checks alone did not establish all-width correctness.

```sh
cmake --build build --target test_limbforge_section9_audit -j4
./build/test_limbforge_section9_audit --bits 352
# Later full audit, explicitly requested rather than part of a light check:
./build/test_limbforge_section9_audit --all-widths --dense
MTL_SHADER_VALIDATION=1 ./build/test_limbforge_section9_audit --all-widths --dense
```

`--dense` adds repeated GEMM runs with 4,097 small matrices (65,552 output entries per run)
and 257 tiled matrices. It also adds 65x65 dense damping trials with three RHS,
513-lane polynomial recurrences and 65,537 inline complex imports. Solver-size factors
and consumer integration still require their separate checks.

`--spread-only` checks broad exponent gaps and cancellation. `--gemm-only` isolates
products. `--normal-only` and `--power-only` isolate those contracts. `--keep-going` reports failed widths and returns nonzero if any fail.
`LIMBFORGE_AUDIT_TRACE=1` diagnoses a failing dot with separate engine and CPU
primitive replays; MPFR remains the acceptance oracle and the mismatch is rethrown.

Follow-up light validation on 2026-10-08: the unchanged smoke test and expanded cases at
352 and 1024 bits passed on the M5 Max. Logs are `benchmarks/results/section9_p0_*`.
No full sweep, dense stress, shader-validation or performance runs were made in that round.


The subsequent full baseline audit passed 26/26 CTest suites, 19/19 GPU suites
under shader validation and all 31 small-shape widths. The serialized dense sweep
nevertheless exposed complex GEMM errors at 11 widths. See
[the diagnostic record](../benchmarks/experiments/section9_gemm_dense.md) and
[the private-storage correction](gpu-codegen.md#13-dense-section-9-complex-gemm-and-private-significands).
Release **1.0.1** corrects the dense complex GEMM issue. The rebuilt library passes
26/26 CTest suites and 19/19 GPU suites under shader validation. All 31 widths pass
the complete expanded dense audit and broad-exponent GEMM checks, normally and
under shader validation. Separate sweeps cover valid/status normal-equation RHS
and valid negative powers (three batches, one fully valid), multi-panel shapes and
a four-row tail. Raw logs and run settings are in `benchmarks/results/section9_p0_*`.
The power shapes are rows/steps/columns 10/5/3, 33/17/17 and 12/4/4.
Solver-size factorization, performance recalibration and end-to-end integration remain pending.
No new speed claim is made by this correctness release.
