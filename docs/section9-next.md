# Section 9 follow-up: review and next tasks

*Review of commit `65c4373` ("Add section 9 batched products, polynomial sources and resident damping trials",
`tips.md` §9, `docs/section9.md`) on top of `3d0eb0c`. Written 2026-10-08 for the agents continuing §9 work.
Read `docs/optimization-plan.md` §3 (ground rules) and `docs/gpu-codegen.md` (all rules) before starting.*

## State

`65c4373` delivers every §9 item as a new unit (`batched_linalg.hpp/.mm/.metal`, `dispatch_policy.hpp`,
`mpc_staging.hpp`, runtime-width overloads in `mpfr_bridge.hpp`, `Internal::require_final`):

| §9 item | API |
|---|---|
| 9.1 power moments / strided batched GEMM | `BatchedLinalg::power_moments`, `gemm` (real/complex, strides, broadcast, accumulate, negate; composed or fused sequential dots) |
| 9.2 batched small products | 4×4 path inside `gemm`, `product3` (A·B·D, optional negation) |
| 9.3 resident normal equations, damping trials | `normal_equations` (sequential), `normal_equations_exact` (augmented exact `syrk`), `cholesky_trials`, `cholesky_solve` |
| 9.4 recurrence with on-device sources | `polynomial_recurrence` (rank-one; Horner per step per lane) |
| 9.5 infrastructure | runtime-width bridge, `describe_inline_mpc` + `import_inline_complex`, `wait_all_async`, `BreakEvenTable` |

Aliasing, statuses and provisional inputs are handled carefully. The lead agent reported
all 25 CTest suites passing after `65c4373`, including `section9_smoke`; this predates the two version/package checks
added in `ed41426`. Passing that suite does not establish all-width coverage of the new kernels.
The commit's own testing is a light smoke run (one 0.6 s test,
essentially 352 bits plus one 224/704-bit case each), and its docs say so.

### Compatibility baseline

The public API is versioned at **1.0.0** (`ed41426`), including the §9 entry points and their documented sequences.
Read `docs/versioning.md` before implementation. Preserve existing names, defaults, layouts, statuses and ownership
rules within 1.x. A different arithmetic sequence must be an explicit new option/API with its own reference, or wait
for a major release. In particular, P1.2/P1.3 and P2.1 must not silently change or rename existing entry points.
These are general numerical APIs; solver-specific integration remains in the consumer (P3).

## Measurements at QSC shapes (lead agent, 2026-10-08)

**Historical, pre-1.0.1 measurements:** these describe the original backend, which later
failed dense correctness checks. Remeasure the corrected backend before using these ratios.

Apple M5 Max, 352 bits, host load average 25–37 (CPU ratios optimistic for the GPU; recheck on an idle host).
`section9_limbforge` checks every GPU output against an independent MPFR sequence.

| Workload | GPU wall | Comparison | Ratio |
|---|---:|---|---:|
| `power_moments` complex, 128 batches, nmax+1 = 60, steps 160, ncols 16 | 0.115 s | 18-worker MPFR 0.73 s | 6.3× |
| same, 520 batches | 0.48 s | 18-worker MPFR 2.88 s | 6.0× |
| complex `gemm` 66×130 · 130×3800 (Fourier shape) | 0.17 s | 18-worker MPFR 1.36 s | 7.9× |
| `cholesky_trials`, 8 trials, n = 200 | 0.034 s | 8 × `Linalg::cholesky` 0.17 s | 4.9× |
| same, n = 400 | 0.094 s | 0.29 s | 3.1× |
| same, n = 944 | 1.54 s | 1.23 s | **0.8×** |

Commands:

```sh
./build/section9_limbforge --operation power --bits 352 --count 520 --m 60 --n 16 --k 160 --workers 18 --repeats 3
./build/section9_limbforge --operation gemm --bits 352 --count 1 --m 66 --n 3800 --k 130 --workers 18 --repeats 3
```

The `cholesky_trials` timing used a scratch program: A = JᵀJ from `Linalg::syrk` with J of size (n + n/6) × n,
D = diag(A), μ_t = 10⁻³·2⁻ᵗ, compared with eight host-loop `Linalg::cholesky` calls on A + μ_t D.
Turn it into a committed benchmark (task P1.2).

## Tasks

Each task: one round, an independent reference for any new contract, all-width GPU tests before benchmarks,
a row in `docs/optimizations.md`, rejected attempts kept under `benchmarks/experiments/`.

### P0: correctness and records (do first)

**P0.1: All-width verification of the §9 kernels.**
- **Harness prepared:** `test_limbforge_section9_audit --all-widths` selects every width. `--bits B` selects one width; `--dense` repeats larger real/complex GEMM batches three times. The target is excluded from default builds and CTest to keep routine checks light; build and run it explicitly (see `docs/section9.md`). Preparation and focused runs do not complete P0.1; full-width, dense, shader-validation and large-factor checks remain required.
- **Full baseline audit:** 26/26 CTest suites, 19/19 shader-validation GPU suites,
  and all 31 small-shape widths pass. The serialized dense sweep fails complex GEMM
  at 11 widths. **Correction accepted in 1.0.1:** all 31 dense GEMM/wide-exponent widths
  pass normally and under validation; the full expanded audit passes in both modes.
  Rebuilt 26/26 library suites and 19/19 shader-validation GPU suites pass. Additional
  valid/status normal-equation and negative-power sweeps also pass at all widths.
  This completes the recorded dense/all-width checks; solver-size factors and consumer
  histories still need their P1/P3 checks.
  [Diagnostic record](../benchmarks/experiments/section9_gemm_dense.md).
- **Earlier focused checks, 2026-10-08:** `section9_smoke` and expanded reference cases at 352 and 1024 bits passed on the M5 Max. Logs: `benchmarks/results/section9_p0_*`. No full sweep, dense stress, shader validation or new benchmark was run.
- **Kernels:** MPFR (or MPC) replays of each documented sequence at all 31 widths (64–1024) for:
  - `gemm`, real and complex: the tiled path and the 4×4 path, tails, broadcast, padding, accumulate, negate, composed and fused;
  - `product3`;
  - `power_moments`: n0 positive, zero and negative (reciprocal, zero denominator), accumulate;
  - `normal_equations`, composed and fused;
  - `cholesky_trials` and its solve: success, failing trials, statuses;
  - `polynomial_recurrence`, composed and fused, reverse, all_steps, sharing tails;
  - `import_inline_complex`: odd and even word counts, NaN/Inf, exponent limits.
- **Inputs:** cancellation-heavy cases, statuses, and large dense batches (cross-thread corruption).
- **Shader validation:** run the GPU suites again under `MTL_SHADER_VALIDATION=1`.
- **Why widths matter:** several GPU-only miscompiles in this library appeared only at particular widths (`gpu-codegen.md` §3, §6–§12), and the new kernels instantiate `cfma_rolled`, `fma`, `cdiv` and Horner loops in new contexts.

**P0.2: Records.**
- **Recorded:** section 9 implementation/P0 rows are in `docs/optimizations.md`, the plan status table is updated, and the contracts now live in `docs/numerics.md` and `docs/execution.md`; `docs/section9.md` links to them as a usage guide. The full-width outcomes,
  dense correction and rejected patches are now recorded with the 1.0.1 release; no new performance claim is accepted by this round.
- Add a round row to `docs/optimizations.md` and update the status table in `docs/optimization-plan.md`.
- Fold the contracts in `docs/section9.md` into `docs/numerics.md` (sequences) and `docs/execution.md` (resident and async rules), keeping a single source of truth.

### P1: performance at real sizes

**P1.1: `power_moments` without the full device power table.**
- **Delivered in 1.1.0:** explicit `PowerStorage::compact` host/resident overloads use an
  eight-row panel plus unscaled checkpoints, preserving the sequence. Scratch falls by
  6.7–11.2× at the measured large shapes. Most timings are slower (warm central checks
  +9–24%), so existing calls retain the full-table default. This solves the auxiliary-memory
  problem; a faster fused tile-loader remains a possible future experiment.
  [Validation and measurements](../benchmarks/experiments/section9_compact_power.md).
- **Problem:** the kernel materialises `count × (nmax+1) × steps` complex powers in device memory.
- **Size of the table:**

  | Shape | Table size |
  |---|---|
  | Measured: 520 × 60 × 160 at 352 bits | ≈ 0.56 GB |
  | QSC upper range: 900 × 101 × 250 at 352–448 bits | ≈ 2.5–3 GB |

- **Change:** generate the powers inside the GEMM tile loader, preserving the existing sequence: start from `powi(y, n0)`, advance by repeated multiplication by y, and multiply each resulting power by E.
- **Tile boundaries:** `powi(y, n0 + tile_start)` generally rounds differently from advancing the original sequence. Replay the prefix, or benchmark a smaller checkpoint table containing unscaled powers generated by that same sequence. A different sequence needs an explicit new option/API and its own reference.
- **Measure:** 400–900 batches, steps 100–250, nmax 40–100, ncols 16, at 352/384/448 bits, against the current kernel and 18-worker MPFR.

**P1.2: Blocked `cholesky_trials`.**
- **Sequential block candidate rejected:** `f15ad1f`, based on 1.2.0/main
  `02bca53`, passes all 31 factor/solve widths in both modes, 26 normal suites and
  all 19 GPU-validation suites. All six three-repeat comparisons pass their independent
  CPU/GPU references, but block panels lose to scalar at n=200/400/944 and take
  5.9–11.2x longer than the exact Linalg loop for factor plus three-RHS solve.
  The historical small-matrix advantage is not reproduced; retain production defaults.
  Busy-host library results do not establish consumer timing or a matrix-size cutoff.
  The patch, full grid and raw records are preserved in
  [the rejection record](../benchmarks/experiments/section9_blocked_trials.md).
  The 3x target remains open; the new exact-contract batched design is P2.2.
- **Historical problem statement:** the column-by-column passes (≈ 3n small dispatches with shrinking grids) lose to eight sequential `Linalg::cholesky` calls at n = 944 (1.54 s vs 1.23 s).
- **Change:** use a blocked algorithm, with per-trial panels and trailing updates as batched GEMM over all trials.
- **Target:** ≥ 3× over 8 × `Linalg::cholesky` at n ≈ 1000, and keep the n ≤ 400 advantage.
- **Contract:** expose a different sequence through an explicit new option/API and give it an MPFR replay. Preserve the existing `cholesky_trials` sequence and default. Compare numerical contracts alongside timings against `Linalg::cholesky`.
- **Benchmark:** commit one for the trials (shape above), including `cholesky_solve` with several right-hand sides (the chord steps).

**P1.3: GEMM kernel.**
- **Prepared comparison:** opt-in `section9_gemm_limbforge` compares composed/fused
  complex GEMM, exact real embedding and a separately documented three-real-GEMM
  Gauss sequence. All 31 CPU exact-reference checks and focused 352/1024 GPU cases
  pass normally and under validation. 352-bit Fourier host/resident and 10,000/100,000 resident 4x4 timings
  now pass every output check. Three private 8x4 tile configurations pass their all 31
  normal/shader GEMM sweeps; register-blocked 8x8 and a named-accumulator correction
  fail dense 64-bit shader validation. Remaining tile gates/timing and native-product acceptance remain pending; see `benchmarks/experiments/section9_gemm_comparison.md`.
- **Historical baseline:** the pre-1.0.1 composed tiled GEMM measured 7.9× MPFR at
  the Fourier shape; remeasure the corrected backend before using that ratio.
- **Compare it against:**
  - `Linalg::gemm` (exact, one rounding per entry; complex through the real embedding);
  - a 3-multiplication complex product, in either path, with its own documented sequence;
  - register blocking and a larger tile (try 8×8 / 16×8 outputs per SIMD group, TK sweep), within the 32 KB threadgroup limit at 1024-bit complex.
- **Shapes:** both §9.2 shapes, plus the 4×4 batches at 1e4–1e5 matrices.

**P1.4: Normal equations.**
- **Measured:** all six 352/384/448-bit host/resident comparisons at n=944, K=1100
  complete three checked repetitions. Exact augmented normals have 4.39–9.76x lower
  median GPU wall time than composed sequential normals and smaller measured error.
  Plain exact SYRK plus GEMM has the lowest median in these fixtures. Independent
  all-width CPU references and focused physical-GPU/shader checks pass beforehand.
- **Usage guidance:** recommend `normal_equations_exact` for independent calls at
  these measured profiles. Preserve sequential defaults and immediate within-batch
  chaining; exact outputs require wait-time finalization. Bounded input exponents
  and busy-host load limit these results; consumer acceptance remains P3.
  [Timing, accuracy and raw evidence](../benchmarks/experiments/section9_normal_equations.md).

**P1.5: `polynomial_recurrence` sources.**
- **Delivered in 1.3.0:** independent `polynomial_sources` API, explicit shared-source
  recurrence and separate polynomial prewarm requests. All 31 recurrence/source widths
  pass normally and under shader validation, with 26/26 normal and 19/19 GPU-validation
  suites passing. Final API/cache/lifetime, linked fixtures and package checks pass.
  Correctness-checked 352/384/448-bit composed/fused measurements at sharing 1/16/60
  show 22–63x lower wall time than the old GPU path. CPU ratios are provisional on the
  busy host; consumer convergence/idle timing remain P3 work. Existing production calls
  retain per-lane evaluation; rejected adapters and measurements are preserved in
  [the experiment record](../benchmarks/experiments/section9_shared_sources.md).
- **Problem:** every lane re-evaluates 8 Horner polynomials of degree ≈ Nc per step, and lanes sharing a weight group repeat identical work. Horner dominates, at roughly 60× the arithmetic of the recurrence update itself.
- **Option to measure:** a device source pass that writes p and q per (step, weight group, component) into batch scratch. That scratch is small, e.g. 160 × 65 × 8 complex, and is not the host table the consumer wants to avoid. Follow it with the existing `vector_recurrence`, which also brings the matrix, affine and tangent modes.
- **Contract:** keep the documented Horner and recurrence order, so results stay identical. Preserve incomplete weight-sharing groups: `polynomial_recurrence` permits sharing tails, which an adapter to `vector_recurrence` must handle explicitly.

**P1.6 delivered in 1.2.0:** reusable per-call slots cache `product3` and exact-normal scratch.
Normal and shader-validation smoke checks pass, including two exact repairs in one batch,
busy release, abandoned batches, engine ownership and unit destruction. No new numerical
kernel or timing claim. Combined API checks also pass in both modes; busy storage
survives release and is freed only after its owning work is finished.

**P1.6 scope:** warm calls reuse intermediate Metal buffers. Host encoding still
allocates bookkeeping, and concurrent live calls need distinct slots. Workspace
queries and release follow the existing Linalg ownership semantics.

### P2: API cohesion

**P2.1: Two contracts, similar names.**
- **Delivered in 1.2.0:** explicit exact GEMM/SYRK, sequential batched
  products/normals/trials/solves, and blocked Cholesky/solve aliases. Old names and
  defaults remain supported. The Numerics table compares accuracy, speed-evidence
  limits and residency/finality. Build and version/package checks pass; focused
  cancellation/physical-GPU checks pass normally and under shader validation.
  Combined API integration checks pass normally and under shader validation.
  Linalg products round an exact dot once; blocked factorizations have multiple
  rounded panel/update steps. BatchedLinalg uses sequential composed/fused dots
  and a different Cholesky sequence.
- Make the contract visible through documentation and additive names or a contract enum.
- Clarify same-named methods with different meanings (`cholesky_solve`); preserve existing entry points through 1.x.
- Add a comparison table (contract, accuracy, speed, residency) to `docs/numerics.md`.

**P2.2: Exact-contract damping trials.** The isolated `70d87d4` baseline adds
synchronous `cholesky_trials_blocked`, parallel host panels and shared resident
exact SYRK submissions. All31 normal/shader references, full26 normal suites,
all19 distinct shader-validation suites and four post-WolfNum linked fixtures pass.
Numerical source hashes survive the rebase; this API remains outside production.
The first checked 352-bit n=944/count8/three-RHS measurement takes0.685s for batched
factorization versus0.619s for the exact Linalg loop, missing the3x target and losing
about11%. Component accounting identifies serialized trailing-update waits despite
cheaper parallel panels. Preserve the baseline patch, raw gates, timing/load and
memory in [the baseline record](../benchmarks/experiments/section9_exact_trials_baseline.md).
Improve scheduling/update throughput, then complete352/384/448 and200/400 comparisons
before release acceptance. The target remains bit-identical factors per trial with
the same block; existing sequential calls and defaults stay unchanged. Consumer
Newton histories and idle timings still require P3 checks.

A separate look-ahead schedule at `05f22fa` now passes all 31 widths in both modes
and the full 26/19 regressions, including failures during overlapped updates.
Its first checked n944/352 median is 0.528 s versus the exact loop's 0.612 s,
with overlapping ranges; the 3x target remains unmet. See [the retained experiment](../benchmarks/experiments/section9_exact_trials_lookahead.md).

**P2.3: Break-even table.** Partial: 50 source-recurrence keys and 24 normal-equation
keys cover measured host/resident contracts and clock profiles; 14 GEMM keys cover
the checked 352-bit Fourier host/resident and resident 4x4 measurements. Three factor-plus-solve
keys retain the explicit unreleased-trial scope. Exported factories load every key;
unmeasured widths/profiles remain unknown. Terminal/hash/idle/overwrite guards pass.
No table installs a default or establishes idle-consumer policy. See
[offline calibration](calibration.md). Other GEMM widths/clock modes and accepted idle-host profiles remain pending.

### P3: end-to-end (consumer)

**P3.1: qscmx integration.**
- **Switch over:**
  - the adjoint descent tables to `power_moments`;
  - the per-column 4×4 post-processing to batched products plus `lu4`;
  - the Fourier residual to the GEMM;
  - the damping trials to `cholesky_trials`;
  - optionally the base pass to `polynomial_recurrence`.
- **Check:** Δ agrees to 10⁻²⁵ at g = 0.1, 0.2, 0.5, and the Newton histories match.
- **Measure:** wall time per Jacobian against the current ≈ 7 s (Nc = 59, Nsh = 160, nPts = 65, 352 bits), on an idle host.

### P3 handoff checklist from `qsccpp/LIMBFORGE_NEXT.md` (2026-10-08)

Source: the consumer's `LIMBFORGE_NEXT.md`; implementation/verification detail lives in
`qsccpp/LIMBFORGE_STATUS.md`. Keep universal numerical APIs in WolfNum and solver
switches in qscmx. The following records each handoff item, without treating a local
candidate or an unconverged diagnostic as accepted production behavior.

| Item | Status / acceptance still needed |
|---|---|
| Catch GPU/bridge errors and fall back to CPU; warn once | Implemented locally; focused helper check passes, including transactional output preservation and worker exception propagation |
| One shared Engine for algebra units | Implemented locally; matching 1.2.0 headers/archive checked at startup; pipeline caches remain separate |
| Algebra prewarm | Delivered in 1.2.0 and wired locally in qscmx: futures overlap CPU setup and both are drained before solver construction; startup failure disables GPU helpers for CPU fallback. Focused normal/shader checks and a real main startup eval pass; consumer records are in `runs/limbforge/prewarm_1_2`. First-Jacobian timing and converged-history acceptance remain pending |
| Keep `adj_prep` CPU-only; select stage 2 after the glue | Preserved; do not move the glue or invoke GPU work inside stage 1 |
| Relative MPFR adjoint check in units of `2^-prec` | Implemented; finite, nonempty checks require <=65536 normalized units; report finite-difference amplification separately |
| Converged CPU/GPU g=.1/.2/.5 checks | Pending: componentwise Delta agreement <=1e-25, equal iteration counts and matching residual histories; save inputs and both logs |
| Idle-host Jacobian timings | Pending: all load averages <4 before each solve; 352 bits, Nc=23/31/59, with the large case Nsh=160 and nPts=65 |
| Fold `J^T G` into exact augmented SYRK | Local candidate behind explicit `QSC_GPU_NORMAL=1`; default off, CPU fallback and `QSC_GPU_SYRK` gate retained; full converged-history and timing acceptance pending |
| Fourier residual -> complex GEMM | Pending; measure 66x130 times 130x3800 on the corrected backend (P1.3) |
| Per-column 4x4 products and LU4 | Isolated product3 and LU4 candidates batch tangent products and descent/gluing inverses, with explicit off-default switches, CPU fallback and matching final CPU replays. GPU/consumer gates are queued; original integration and full acceptance remain pending. See [products](../benchmarks/experiments/section9_products4_consumer.md) and [LU4](../benchmarks/experiments/section9_lu4_consumer.md) |
| Remaining damping trials | Pending; preserve the first exact factor path, use batched trials only after P1.2/P2.2 numerical/performance acceptance; keep the new switch off until current-backend acceptance; historical small-matrix ratios are insufficient |
| Optional base polynomial recurrence | Pending; measure shared Horner sources (P1.5) before switching the solver |
| Large-shape memory | Pending consumer measurement: host W remains; compact 1.1.0 power storage is available explicitly, with a latency trade-off; chunk count if needed |
| >1024-bit fallback and precision rounding | Preserved; helpers refuse unsupported widths after rounding up to a multiple of 32 |
| Deliverable in qscmx | `LIMBFORGE_STATUS.md` and verification logs exist; final accepted timing table and complete switch list remain pending |

For each new consumer switch, keep CPU fallback and a `QSC_GPU` sub-switch, then repeat
both converged-history verification and idle-host measurements. Time stage 1, stage 2,
MPC/MPFR packing and unpacking, SYRK, `J^T G`, Cholesky and total separately. An ordinary
arithmetic `Engine::Prewarm` does not prepare algebra pipelines and must not stand in
for the algebra preparation API.

The combined-normal candidate computes its RHS with a once-rounded exact dot instead
of the CPU's sequential MPFR FMA. Its focused helper checks pass normally and under
shader validation. A saved-seed g=.2 diagnostic at Nc=15/NQ=19/Nsh=100/nPts=21 agrees in
Delta to 2.83e-118, has the same five history values at all 17 serialized digits and
passes adjoint checks, but both runs stall at 5.34e-14 above gtol=1e-25. This is agreement
evidence, **not converged acceptance**. Select this comparison explicitly with the
consumer runner's `--normal-equations`; ordinary comparisons disable the candidate.

The original saved-seed .1/.5 diagnostics also agree but stall. A reused-seed .2 case
diverges and stalls; its own saved seed agrees but still fails convergence acceptance.
Full g=.1/.2/.5 production checks remain open. Fourier 7.9x and old trial ratios are
historical pre-1.0.1 measurements and must be remeasured before any speed claim.

### Verified WolfNum 1.3.1 consumer replay (2026-10-09)

The current `build-wolfnum-hp` consumer uses matching 1.3.1 headers/archive at
`78f7da6`, with ten inline limbs. Source/archive/binary hashes, raw serialized
outputs and log markers have been independently checked against the existing
consumer records; no new solver run or timing was performed by this review.

The J=3 X²Y fixture at g=.202, Nc=21/NQ=25/Nsh=300/nPts=27 and 374 bits passes
its input convergence gate (gtol=1e-22): both residuals are 5.57752179966997436e-25,
three-entry histories match, and maximum component Delta difference is
7.10245197342712e-125. Adjoint checks pass and all three existing GPU paths run
without fallback. The combined exact-normal candidate is explicitly disabled.
This is evidence for that fixture; it does not complete the requested g=.1/.2/.5
checks, new switches or idle-host timings. The same comparison command also
contains the old J=2 g=.2 diagnostic, which still stalls at 5.34e-14 and therefore
returns overall exit 2. Preserve that unsuccessful acceptance result.

[Archived raw records and independent check](../benchmarks/experiments/section9_consumer_replay.md).

An isolated batched Fourier Jacobian adapter now compiles and reproduces the J3
default and split-CPU Delta/history/residual exactly. It gathers both directions
from all columns into one exact real-embedding GEMM; GPU, converged-history and
idle gates are pending. Existing consumer sources/jobs are untouched.
[Prepared adapter and scope](../benchmarks/experiments/section9_fourier_consumer.md).
