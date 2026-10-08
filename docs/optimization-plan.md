# LimbForge optimisation plan (rounds 13+)

*Written for coding agents that will implement rounds independently. Self-contained; read fully before starting a round.*

## Context

LimbForge (`/Users/k0959535/Dropbox/IntegrabilityProjects/2026 4d SoV/LimbForge`) is a Metal (Apple GPU) library for batched
fixed-precision real/complex floating-point arithmetic, 64–1024 bits in 32-bit limbs, round-to-nearest-even, validated
bit-for-bit against MPFR. Twelve optimisation rounds are done (below). Two consumers now drive priorities:

- **qscmx** (twisted Quantum Spectral Curve solver, `2026 Near N=4/qsccpp/include/mx.hpp`): 2·10⁴–1.4·10⁵ independent
  lanes × 40–200 steps of a 4-component complex vector recurrence, plus segmented dot products, small dense algebra and
  normal equations, at **224/256 bits**. Requests are in `LimbForge/tips.md` (ordered by payoff) — this plan absorbs them.
- **BSolver4D** (Baxter equation, `BSolver4D/cpp/baxter.hpp`): the existing scalar 4-term `recurrence` kernel at **384 bits**,
  ~128 trajectories × 600 dependent steps — *latency-bound*, the GPU is mostly idle (`benchmarks/m5_max_baxter.txt`).

The API is a development preview: **breaking changes to structs, layouts and API are allowed** (owner's decision).
Correctness is non-negotiable: every accepted change passes MPFR comparison on CPU *and physical GPU*.

---

## 1. Current state (as of `f558789`, rounds 1–12 pushed to `origin/main`)

| Round | Status | What |
|---|---|---|
| 1 | accepted | exponent-boundary rounding fix |
| 2 | accepted | typed resident `Buffer<T>`, `CommandBatch`, async `Submission` |
| 3 | accepted | exact `N+2`-limb addition workspace for gaps ≤ 32 |
| 3t | **rejected (GPU-only fail)** | known product top bit in `mul` (skip `highest` scan) |
| 4 | accepted | reused integer reciprocal in Knuth division |
| 5t | **rejected (GPU-only fail)** | Comba / product-scanning multiply (3 variants) |
| 6 | accepted | threadgroup-size sweep; `EngineOptions::threads_per_threadgroup`; heavy complex ops use one SIMD group |
| 7 | probe only | word-plane (SoA) layout harness `benchmarks/layout.mm`; AoS retained (mixed results) |
| 8 | accepted (restricted) | symmetric `square`, selected on GPU **only at 384 bits** |
| 8t | **rejected (GPU-only fail)** | full-width square (fails 992 bits); square inside `cdiv` (fails 384 bits) |
| 9 | accepted | correctly rounded bit-restoring `sqrt` (slow baseline) |
| 10–12 | accepted | fixed-order resident `tree_sum` (global + cooperative threadgroup variants, policy-selected) |

Key code: `include/limbforge/core.hpp` (all arithmetic, shared CPU/Metal: `pack`, `aligned_add`, `add`, `mul`, `square`,
`divide_word`, `div`, `sqrt`, `cadd/cmul/cdiv`), `src/kernels.metal` (`arithmetic`, `complex_arithmetic`, `recurrence`,
`*tree_sum*`), `src/engine.mm` (runtime compile per precision with `MP_BITS`, function-constant specialisation, `group_size`
policy, `CommandBatch::encode`, `encode_tree_sum`), `include/limbforge/engine.hpp` (public API), `mpfr_bridge.hpp`.
Harnesses: `benchmarks/{benchmark,tune,layout.mm,square,reduction}.cpp`, `benchmarks/run_round.py`, `benchmarks/compare.py`.
Tests: `tests/test_arithmetic.cpp` (all 31 precisions × 4096 cases), `test_buffers.cpp`, `test_reduction.cpp`, `reference.hpp`.

---

## 2. Findings that shape this plan (read before touching kernels)

**F1 — One unexplained GPU-only failure class blocks most multiply-level speedups.** Comba (3 variants), the known-top-bit
`pack`, full-width square and square-in-`cdiv` all pass CPU/MPFR and fail on the GPU. In every failing log the first
reported case is `case=1`; case 0 is trivially `status=invalid` (`i%101==0`), so **case 1 is the first real product — the
kernels are wholly wrong above a size threshold, not wrong on edge cases**. Thresholds: Comba fails from N=25 (800 bits)
and passes N≤24; full square fails at N=31; hybrid Comba fails in *complex* kernels already at 128 bits (more inlined code
→ same cliff earlier). Schoolbook `mul` passes at N=32. This pattern indicates a compiler codegen/register-allocation cliff
(large private arrays, dynamic indices, unroll limits), or latent UB exposed by different codegen — not arithmetic errors.
Fixing or characterising this is the single biggest enabler (Phase A2).

**F2 — Published device times are inflated by GPU clock ramp-up.** The same 384-bit `mul` kernel at 65,536 values measures
0.45 ms device time in `benchmark.cpp` (GPU runs after idle CPU/MPFR phases) but 0.106 ms in `tune.cpp`
(`round6_tuning.csv`). Per-step cost inside `mul_chain_resident` (2.44 ms / 16 = 0.15 ms) confirms it. Every optimisation
decision so far was made with noisy, cold-clock numbers; the "384 bits slower per limb than 1024" anomaly is likely this.
New metrology comes first (A1).

**F3 — Apple GPUs have no native 64-bit integer ALU.** Every `dword` multiply-add/compare in `mul`, `div`, `aligned_add`
is emulated with several 32-bit ops. Rewriting inner loops in 32-bit form (`a*b` + `mulhi(a,b)`, explicit carries) is a
plausible 1.3–2× ALU win — a hypothesis to measure, not a promise.

**F4 — Dynamic indexing into private arrays spills to scratch memory.** The `recurrence` kernel comment already notes it.
Suspects: `div`'s `u[j+i]` sliding window, `sqrt`'s bit loop, `pack`/`extract` with runtime shifts, `aligned_add`'s
`shifted_limb`. Compile-time indices (fully unrolled / template-generated straight-line code) keep data in registers.

**F5 — Wall time is dominated by host overhead at ≤ 65k values.** e.g. 1024-bit add: 0.22 ms device vs 1.21 ms wall
(host-array path: 3 memcpys of 9 MB + commit + `waitUntilCompleted`). Resident paths cut this but per-submission latency
~0.2–0.3 ms remains. Small batches (256 values) are pure latency.

**F6 — Consumer shapes.** qscmx: many lanes, short-ish chains, 224/256 bits, needs fused primitives and fixed reduction
order (bitwise reproducibility for finite-difference Jacobians). BSolver: few lanes, long chains, 384 bits → needs either
more lanes per submission (consumer-side batching of points) or intra-number parallelism (Phase E).

---

## 3. Ground rules for every round (all agents)

1. **Start** from current `origin/main`. Work in a git worktree on branch
   `round-NN-<topic>`; one change per round; rebase before merge. Shared hot files (`core.hpp`, `kernels.metal`,
   `engine.mm`) — coordinate: only one agent edits `core.hpp` arithmetic at a time (Track T1 owns it).
2. **Correctness gate** (must all pass before any benchmark is recorded):
   `cmake --build build -j4 && ctest --test-dir build --output-on-failure`, then the GPU suites again with
   `MTL_SHADER_VALIDATION=1`. CPU agreement alone is never sufficient (see F1).
3. **New primitives** need a written rounding contract in `docs/numerics.md` and an *independent* MPFR/MPC reference in
   `tests/reference.hpp` (never compare a kernel against itself on CPU only).
4. **Measure** with `python3 benchmarks/run_round.py <fresh-label>` (never overwrite results) plus the new warm-clock kernel
   sweep (A1). Report device-warm, device-cold and wall separately. Repeat any ≤ 15% claim with an interleaved A/B recheck.
5. **Record** every round in the `docs/optimizations.md` table, including rejections; save rejected code as
   `benchmarks/experiments/<name>.patch` with base revision and failure log, and a line in `benchmarks/experiments/README.md`.
6. Match the repo's dense one-liner C++ style.

---

## 4. Roadmap

Effort: S ≤ 1 day, M 2–4 days, L ≥ 1 week of agent work. Payoff estimates are hypotheses to be measured.

### Phase A — Foundations (do first; unblock everything)

**A1. Warm-clock kernel metrology sweep** — S/M, Track T1
- New `benchmarks/kernel_sweep.mm` (+ CMake target): for each of the 31 precisions × each op (real add/sub/mul/div/
  square/sqrt, complex add/mul/div, recurrence step), allocate resident buffers, run a ~200 ms GPU warm-up spin, then encode
  R=32 independent dispatches (separate outputs) in one command buffer; report device time per dispatch, ns/value,
  effective GB/s, limb-products/s, and `pipeline_info` (`max_threads` is a register-pressure proxy). Check one output set
  against MPFR. Optional: `MTLCounterSampleBuffer` per-dispatch timestamps where supported.
- Add `--gpu-warm` to `benchmark.cpp` and record both cold and warm GPU columns; fix README claims to say which.
- Output: `benchmarks/results/round13_sweep.csv` — the baseline "roofline" table every later round compares against.

**A2. Root-cause the GPU-only failures (F1)** — M/L (time-box 4 days), Track T1
- Build a probe (`tests/gpu_codegen_probe.mm`) that compiles an alternative core from a source string through the existing
  runtime-compile path and writes **raw unrounded products** (2N+1 words) to a debug buffer; diff against CPU per limb.
  Apply `benchmarks/experiments/comba_full.patch` / `square_full.patch` / `known_product_top_bit.patch` in the probe.
- Bisect: by N; by column/limb of first divergence; product-only vs `pack`; kernel inlined alone vs inside `cdiv`.
- Test hypotheses in order:
  H1 private-array size/unroll cliff → try `#pragma clang loop unroll(full)` / `unroll(disable)`, splitting `product` into
  fixed-size register blocks, template-generated straight-line code (`std::index_sequence`-style recursion, all indices
  compile-time), or staging `product` in threadgroup memory;
  H2 64-bit carry idiom (`low+=t; c+=low<old`) miscompiled → 32-bit-only formulation via `mulhi`;
  H3 compiler option sensitivity → `MTLCompileOptions.optimizationLevel` (Size vs Default), `languageVersion` latest;
  H4 latent UB in shared code exposed by codegen → build CPU tests with `-fsanitize=address,undefined` and
  `-ftrivial-auto-var-init=pattern` across all 31 precisions; review shift-by-32 guards and uninitialised `z` in kernels.
- Deliverable: `docs/gpu-codegen.md` with root cause (or best characterisation) and **coding rules** for kernels; a minimal
  standalone reproducer suitable for Apple Feedback Assistant; re-run the four rejected patches under the workaround.
- Exit: either a workaround under which Comba passes all 31 precisions on GPU, or a documented "do not" list plus the
  safe-pattern template used by B-rounds.

**A3. GPU test hardening** — S, Track T1
- Per-precision dense GPU tests for every op (today only 1024-bit complex is dense); report *count* of mismatches and the
  first 3 cases, not only the first; `--bits`/`--op` filters for fast iteration; a "first real case" sanity check at each N.

### Phase B — Core arithmetic kernels (benefit every consumer) — Track T1, after A1 (+A2 where noted)

> **Round 13 outcome** ([gpu-codegen.md](gpu-codegen.md)): A1 and A2 are done. B1 is dropped (32-bit
> `mulhi` products are not faster). B2 is not pursued (rolled Comba is slower; unrolled Comba hits a
> compiler store bug). A new top item, **B0**, removes the register-array cliff.

- **B0. Remove runtime-indexed small register arrays** — S/M, highest payoff. Apply rule 1 of
  `gpu-codegen.md` to `mul`, `square`, `div` (`u`, `twice`), `aligned_add` workspaces, `sqrt`, and `pack`:
  pad scratch to ≥ 33 words or fully unroll, chosen per width by warm `kernel_sweep`. The probe shows
  3–4.7× faster 288–480-bit multiplication (384 bits: 104 → 35 µs). Validate in real, complex, chain,
  and recurrence kernels at all 31 precisions.

- **B1. 32-bit-native multiply-accumulate** (F3) — M. Rewrite schoolbook inner loop with `a*b`/`mulhi` and explicit 32-bit
  carries on Metal (keep CPU path or use the same code). Measure per precision with A1.
- **B2. Product-scanning (Comba) multiply** — M, needs A2. Accumulators in registers, one store per column.
- **B3. Specialised `pack` for products** — S, needs A2. Top bit is 64N−1 or 64N−2: no `highest` scan, 1-bit shift
  select, compile-time `extract` (retry `known_product_top_bit.patch` under A2 rules).
- **B4. Square everywhere + `|b|²` in `cdiv`** — S, needs A2 (retry `square_full.patch`). Also use square in tree/norm code.
- **B5. Short (high-half) product with exact fallback** — M. Compute only columns ≥ N−2 plus a rigorous error bound on the
  omitted low part; if the rounding decision is certain, round; otherwise fall back to the full product (rare, ~2⁻³⁰).
  Expect ~1.6× fewer limb products. Requires a written proof sketch in `docs/numerics.md` and adversarial near-halfway
  fixtures (existing fixtures 8–39 in `test_arithmetic.cpp` are a template).
- **B6. Addition fast paths** — S. Same-sign small-gap path without the full aligned workspace; compile-time-shift variants
  for gap 0; avoid `shifted_limb` dynamic indexing (F4).
- **B7. Division** — M/L. (a) remove dynamic indexing in the Knuth loop (template-unrolled window); (b) evaluate
  reciprocal-based division: compute an (N+1)-limb reciprocal of the divisor once, quotient = high part of a·recip,
  correct with the exact remainder (same final `2r ≥ d` rounding test). `cdiv` shares one reciprocal for both components.
- **B8. Faster `sqrt`** — M, low priority for consumers. Limb-wise (Zimmermann Karatsuba-sqrt or Newton on reciprocal sqrt
  with a `double` seed), certified by the existing exact residual test.

### Phase C — Runtime, memory layout, host overhead — Track T2 (can run in parallel with B)

- **C1 status (round 28):** stride/period broadcast for `b`/`c` done; offsets and gather lists remain.
- **C1. Operand descriptors / broadcast & gather (tips #4)** — S/M. Every `run` operand becomes
  `{buffer, offset, index_mode: plain | i/stride | i%period}`; pass in `Params`; generalises `states_per_weight`.
  Do this early: D-phase kernels reuse it.
- **C2. Host-array path** — S. Zero-copy `newBufferWithBytesNoCopy` for page-aligned host memory (provide an aligned
  allocator), parallel memcpy for large transfers, buffer reuse; MTLSharedEvent spin-wait option for low-latency waits.
- **C3. Concurrent encoder with hazard tracking** — S/M. `MTLDispatchTypeConcurrent` encoder; insert a barrier only when a
  dispatch reads/writes a buffer written earlier in the batch, so independent ops overlap.
- **C4. Completion callbacks (tips #8)** — S. `Submission::on_complete(std::function<void(Timing)>)` via
  `addCompletedHandler`; document thread-safety.
- **C5. Small-batch CPU path and break-even table (tips #8)** — M. Run `core.hpp` on a CPU thread pool below a measured
  per-(op, bits) threshold; publish `docs/break-even.md` (vs 18-thread MPFR) and an API `Engine::recommend_gpu(op,bits,count)`.
- **C6. Layout v2 (breaking)** — M/L, after B so compute is cheaper. Re-run the round-7 probe with A1 metrology.
  Candidates: limb-plane SoA inside `Buffer<T>` with transposing upload/download (CPU multithreaded or GPU kernel);
  compact header (exponent 32-bit + sign/status packed in one word → `bits/8+8` bytes); even-word padding to match
  64-bit MPFR limbs (synergy with D7). Decide by warm device time *and* end-to-end consumer wall time.
- **C7. Pipeline prewarm / binary archives** — S. `Engine::prewarm(bits, ops)` (async) and optional `MTLBinaryArchive`
  cache to remove first-use compile latency from consumer runs.

### Phase D — Consumer primitives (tips.md) — Track T3 (D1–D4), Track T4 (D5–D6)

Each item: written contract → MPFR/MPC reference → CPU implementation in `core.hpp` → GPU kernel → tests at 224, 256,
384 bits (plus all-precision smoke) → benchmark vs 18-thread MPFR at qscmx sizes.

- **D1. Faster MPFR/MPC bridge (tips #7)** — S/M, CPU only, start immediately. Direct limb copy when the MPFR precision
  ≤ target bits: on arm64 a 64-bit MPFR limb is two little-endian 32-bit words, so the top `bits/32` words of the
  left-aligned MPFR significand map by `memcpy`; LimbForge exponent = MPFR exponent − 1; handle zero/NaN/Inf/precision >
  bits via the existing slow path. Array versions for `mpfr_t[]` and `mpc_t[]` (`Complex<N>` ↔ `mpc_t`), multithreaded.
  Optional fixed-limb layout compatible with `mpfr_custom_init` (ties to C6 padding). Test: round-trip at all 31 precisions,
  including 224 (odd word count) and specials.
- **D2. Fused multiply-add (tips #2)** — M/L.
  Real `fma(a,b,c)=RN(a·b+c)`, `fms` — single rounding; reference `mpfr_fma`/`mpfr_fms`. Implementation: exact 2N-limb
  product, addend aligned in a bounded workspace; an addend entirely below the window becomes a signed sticky bit (MPFR
  technique), so huge exponent gaps stay exact without wide buffers.
  Complex `cfma(a,b,c)=a·b±c`: preferred contract `re=RN(a.re·b.re − a.im·b.im + c.re)`,
  `im=RN(a.re·b.im + a.im·b.re + c.im)` (one rounding per component); reference: exact products at 2·bits precision
  + `mpfr_sum`. If the exact 3-term sum is too costly, fall back to the documented contract `RN(RN(fmma)+c)` with
  `mpfr_fmma` — decide by measurement, document either way. Ops: `Operation::fma, fms, complex_fma, complex_fms` (ternary
  `run` overload, works with C1 operand descriptors and in-place).
- **D3 status (round 27):** sequential-fused `segmented_dot` (separate/shared table, real/complex, host and resident). Pairwise order and SIMD-group-per-segment variants remain; batched GEMV/GEMM is covered by `linalg.hpp`.
- **D3. Segmented dot products and batched GEMV (tips #3)** — M. `segmented_dot(a,b,K,out)`,
  `segmented_dot_shared(T,x,K,out)` (shared table broadcast via C1), real and complex. **Fixed order contract:**
  `s₀=RN(a₀b₀)`, `s_k=fma(a_k,b_k,s_{k−1})` sequentially (bitwise reproducible regardless of thread configuration); optional
  `pairwise` mode with a fixed adjacent-pair tree (reuse round-10/11 tree code) for long segments. One thread per segment
  for K ≤ ~128; SIMD-group-per-segment variant for low segment counts. Fourier transforms = batched GEMM → D6 kernel.
- **D4 status (round 21):** implemented as `Engine::vector_recurrence` (host-array API; composed default, fused option; tangent from a stored base chain). Resident version done (round 24). Next: faster fused three-term sum; ≥ 1024-bit compile; qscmx integration.
- **D4. Vector recurrence (tips #1, highest consumer value)** — L, needs D2 (and C1 for sharing).
  `vector_recurrence(bits, v0, p, q, r, out, lanes, steps, lanes_per_weight, flags)`; per step
  `s=q_k·v` (fixed order via D2/D3), `v_i ← cfma(p_{k,i}, s, v_i) (+ r_{k,i})`. Flags: affine source `r`; general 4×4
  `M_k` instead of `1+p qᵀ`; write `v` every step (layout `out[k*lanes+lane]`). **Also evaluate a fused tangent mode**
  (base `v` and tangent `dv` advanced together: `ds=q·dv+dq·v`, `dv ← dv + p·ds + dp·s`) which removes the stored base
  chain and the separate `r` pass — confirm the `dp/dq` data shape with the qscmx owner before building it.
  Kernel design: one thread per lane; `v` (and `dv`) in registers — check spills with `pipeline_info` at N=7,8 (4 complex
  ≈ 88 words each); shared `p,q` staged per step in threadgroup memory for the `lanes_per_weight` group. Test: MPFR chains
  (same contract) at 224/256 bits, 150 steps, random data with |p qᵀ|≈1 plus cancellation-heavy cases; then qscmx end-to-end.
- **D5 status (round 29):** `Engine::lu4` (host-array) done; resident version and general small-n shapes remain.
- **D5. Batched 4×4 complex LU / solve / inverse / det (tips #5)** — M. One thread per matrix. Specify a deterministic
  pivot rule that needs no rounding (e.g. max of `max(|re|,|im|)` by exact exponent/limb compare, lowest index on ties) so
  the MPFR reference reproduces it bitwise. Ops `lu4`, `solve4(nrhs)`, `inv4`, `det4` with per-matrix status (singular).
- **D6. Dense real GEMM / SYRK / Cholesky (tips #6)** — L. Tiled GEMM: each thread owns one output computed with the
  D3 sequential-fma order over k (reproducible); A/B tiles in threadgroup memory. `syrk` (`JᵀJ`, lower triangle) first —
  it dominates for J ≈ 2n×n. Cholesky: start on CPU (n³/3), move to a GPU blocked right-looking version if profiling shows
  it dominates. QR optional later. Sizes 200–1000, 256 bits.
  **Status (round 20-D6): SYRK/GEMM done through L1c, with a stronger contract than sequential fma.**
  `include/limbforge/linalg.hpp` (`Linalg::syrk`, `Linalg::gemm`, host arrays) rounds every output once from
  the exact dot product. Contract and algorithm: `docs/numerics.md`, "Dense products". At n = 1000 (K = 2000,
  256 bits, QSC-like columns) SYRK takes 18.5 ms device and 22 ms wall, against 529 ms for a one-thread-per-output
  composed limb kernel and an estimated 17 s for 18-thread MPFR (`benchmarks/results/round20_linalg.csv`).
  The CPU MPFR Cholesky now dominates the normal equations (2.2 s serial at n = 400; n^3/6 multiply-adds).
  **Status (round 23-D6b): blocked Cholesky and triangular solves done.** `Linalg::cholesky` / `trsm` /
  `cholesky_solve` (multiple right-hand sides, per-call first failing pivot), on update GEMMs `RN(C - op(A) B)` with one
  rounding per entry (`syrk`/`gemm` with `subtract`, host `exact_dot_add`). Fixed documented rounding sequence (block 32),
  deterministic, not correctly rounded overall; `docs/numerics.md`, "Cholesky factorization and triangular solves".
  QSC-like, 256 bits: n = 400 in 37 ms, n = 1000 in 163 ms (151× serial MPFR, 16× 18-thread MPFR); SYRK + Cholesky +
  solve n = 1000 in 0.27 s (`benchmarks/results/round23_cholesky.csv`). The host panel (in-block exact dots) is now the
  critical path at n = 1000 (~110 ms on a loaded host); trailing updates overlap it.
  Next steps:
  - shorten the critical path: fewer dispatches per small-K update GEMM (moduli for K <= 64, fused digit/product
    passes), or a cooperative (SIMD-group per row) GPU panel; the one-thread-per-row panel was slower
    (`benchmarks/experiments/cholesky_gpu_panel.patch`);
  - resident `Buffer` operands in a `CommandBatch` (design note in `numerics.md`), so that J, A and L stay on the GPU;
  - QR (S5) and factor handles.
- **D7. Transcendentals at buffer level (tips #8)** — L, lowest priority: complex `exp`, `log`, integer `powi` with explicit
  accuracy contracts (faithful or correctly rounded via Ziv-style retry).

### Phase E — Latency-bound workloads: cooperative intra-number arithmetic — Track T1, after B

For BSolver (≈128 lanes × 600 steps) the GPU runs ~128 threads. Two complementary steps:
- **E0 (consumer-side, S):** round 13 measured ~165 µs per 384-bit recurrence step independent of lane
  count up to 4,096 lanes (saturation near 16,384). Recommend BSolver batch all `u` points / Newton finite-difference evaluations into one
  `recurrence` call (more trajectories per submission) — cheapest win; document in the BSolver integration notes.
- **E1 status (rounds 17, 22):** cooperative G = 16/32 recurrence is in `Engine::recurrence` for ≤ 1,024 trajectories (BSolver `baxter_batch`: 6× with warm clocks, 2× when the call follows idle CPU work — GPU clock ramp-up now dominates; consider keeping the GPU busy or batching more work per call). Open: G = 4/8 under shader validation; a cooperative vector recurrence only if small-lane D4 shapes appear.
- **E1 (L, research):** one number per 8/16/32 SIMD lanes: limb-parallel multiply with `simd_shuffle` broadcasts, carry
  resolution with `simd_prefix_exclusive_sum`/ballot loops, normalisation via ballot + `clz`. Prototype `mul` + `add` in a
  `recurrence_coop` kernel at 384 bits; compare per-step latency against the thread-per-lane kernel; division via a
  cooperative reciprocal only if the prototype wins. Keep the thread-per-lane kernel as the default for large lane counts.

---

## Status summary (2026-10-07, after round 29)

| Item | Status | Where |
|---|---|---|
| A1 metrology, A2 codegen diagnosis | done (13) | `kernel_sweep`, `gpu_codegen_probe`, `docs/gpu-codegen.md` |
| B0 register-array padding | done (14) | `scratch()` in `core.hpp` |
| B1/B2 32-bit and Comba products | dropped (13) | not faster / compiler store bug |
| D1 MPFR/MPC bridge | done (13-D1) | `mpfr_bridge.hpp` |
| D2 fused fma / complex fma | done (15, 19, 25, 26) | `Operation::fma…complex_fms` |
| D3 segmented dots | done (27) | `segmented_dot` |
| D4 vector recurrence (+tangent, resident) | done (21, 24) | `vector_recurrence` |
| D5 batched 4×4 LU/solve/inverse/det | done (29, resident 34) | `Engine::lu4`, `CommandBatch::lu4` |
| D6 SYRK/GEMM, Cholesky, triangular solves | done (20-D6, 23-D6b), host-array | `linalg.hpp` |
| C1 broadcast operands | done (28) | `Broadcast` |
| C7 pipeline prewarm | done (33) | `Engine::prewarm_async` |
| E1 cooperative recurrence | done for G = 16/32 (17, 22) | `cooperative.metal` |
| L1 residue GEMM | done (16, 18, 20) | `linalg.hpp` |

Next, by consumer value: qscmx/BSolver integration and end-to-end timing on an idle host (also the
pending GitHub speed figures); resident
`linalg`; S1 polynomial jets, S4 norms/status summaries, S5 QR; G = 4/8 cooperative kernels under
shader validation; ≥ 1024-bit `vector_recurrence`; D7 transcendentals.

## 5. Order, tracks and dependencies

| Sprint | T1 kernels (owns `core.hpp`) | T2 runtime (`engine.*`) | T3 consumer prims | T4 linear algebra |
|---|---|---|---|---|
| 0 | commit this plan (`docs/optimization-plan.md`) and `tips.md` | — | — | — |
| 1 | A1 ✓, A2 ✓ (round 13), A3; B0 ✓ (round 14) | C1 operand descriptors | D1 bridge ✓ (13-D1); D2 fma ✓ (round 15) | — |
| 2 | B3/B4 (under gpu-codegen rules); B0 for `sqrt`/`pack` | C2, C3, C4 | D2 tuning; D3 segmented dot | — |
| 3 | B5 short product, B6 | C7 prewarm | D3 segmented dot, D4 vector recurrence | D5 batched 4×4 |
| 4 | B7 division | C5 CPU path + break-even table | D4 tangent mode, qscmx integration | D6 SYRK/GEMM |
| 5 | E1 prototype; B8 | C6 layout v2 | — | D6 Cholesky |
| later | — | — | D7 transcendentals | QR |

Hard dependencies: B2/B3/B4 ← A2; every B/C perf decision ← A1; D3, D4 ← D2; D4 sharing, D3 shared table ← C1;
D6 ← D3 microkernel; C6 ← B (re-measure once compute is cheaper). T3 may add kernels to `core.hpp` only as new functions
(no edits to existing arithmetic) to avoid conflicts with T1.

---

## 6. Verification (end-to-end)

- Per round: `ctest --test-dir build --output-on-failure`; `MTL_SHADER_VALIDATION=1 ./build/test_limbforge` (and the
  other GPU tests); `python3 benchmarks/run_round.py roundNN-<topic>`; `./build/kernel_sweep` (A1) compared with
  `round13_sweep.csv`; `python3 benchmarks/compare.py`.
- New primitives: lane-by-lane comparison against the MPFR/MPC reference implementing the documented contract, at 224,
  256, 384 bits plus an all-precision smoke run; dense 65,536-lane GPU runs repeated 3× (detects cross-thread corruption).
- Consumers:
  qscmx — converged Δ at g = 0.1, 0.2, 0.5 agrees with the CPU path to 10⁻²⁵; wall time per Newton step vs the 18-thread
  CPU engine at Nc = 23, 31, 61.
  BSolver — `cmake -DLIMBFORGE_BUILD_BAXTER_EXAMPLE=ON` then `ctest -R baxter_batch` (80-digit threshold) and
  `./build/baxter_batch 16 600` timings vs `benchmarks/m5_max_baxter.txt`.
- Docs updated each round: `docs/optimizations.md` (table row), `docs/numerics.md` (contracts),
  `docs/performance.md`/README (only warm/cold-labelled, MPFR-validated numbers).

---

## 7. Additional APIs motivated by BSolver and QSC (2026-10-07)

**Design proposal, not implemented functionality.** This extends C/D/E rather than replacing them.
LimbForge remains a standalone numerical library: model equations, gluing rules, branch selection,
Newton/LM acceptance and truncation choices belong to consumers. API names below are sketches.
Priorities are inferred from inspected consumer code, not measured speedup claims.

### Public API design rule: general operations, specialised implementations

BSolver and QSC supply demanding workloads and integration tests. The public API must express
numerical operations that another application can use without knowing either solver.

- **Separate three layers.** Core arithmetic supplies numbers, rounding and status; numerical operations
  supply polynomials, series, reductions, recurrences and factorizations; consumer adapters compose
  these into model equations. Keep adapters and solver fixtures in examples/integration code. Public
  headers must not import solver types or expose coupling, twists, gluing, Baxter or QSC options.
- **Make mathematical shape explicit.** Polynomial degree, batch count, recurrence order, vector
  dimension, jet order, matrix dimensions, RHS count and series truncation are caller-supplied shapes.
  A four-state or 4x4 kernel is an optimised specialisation of a documented operation. Describe the
  supported shapes and backend limits explicitly; unsupported shapes fail before submission or use
  an explicitly selected CPU backend. Do not imply arbitrary dimensions are already supported.
- **Use common views and execution rules.** Reuse typed resident buffers, checked offsets/strides,
  broadcasting, workspace queries and asynchronous submissions across operation families. Publish
  coefficient ordering, transpose versus conjugate-transpose, aliasing, ownership and output layouts.
  Provide a straightforward host-array entry point where practical, backed by the same contracts.
- **Keep numerical choices explicit.** Precision, accumulation order, fused/composed arithmetic,
  rank thresholds and backend choice must have documented semantics. Diagnostics describe numerical
  quantities and failing indices; consumers decide convergence, acceptance and precision retries.
  For example, `scaled_residual` takes residual and scale arrays supplied by the caller, rather than
  assuming a Baxter equation or a QSC gluing condition.
- **Keep plans independent of physical parameters.** Library caches key on operation, shape, precision,
  layout and explicit data-generation identifiers. Consumers invalidate their model-dependent tables
  when their own parameters change. A library plan must not interpret those parameters.
- **Validate reuse without speculative abstraction.** Before stabilising a family, demonstrate it in
  a solver fixture and an independent example. Start with a small concrete API and measured kernels;
  add general-purpose composition through buffers and plans rather than a new GPU expression language.

| API family | Independent example demonstrating the same contract |
|---|---|
| Polynomial values and jets | Batched response-polynomial evaluation and local Taylor expansions |
| Scalar/vector recurrences | Time-dependent linear state updates and transfer-matrix chains |
| Truncated series / Toeplitz operations | Formal power-series arithmetic and finite causal convolution |
| Norms and status summaries | Batched linear-system residual diagnostics |
| Reusable QR / LU / Cholesky | Multiple-RHS least squares and dense linear systems |
| Casts and execution plans | Repeated numerical pipelines at an explicitly chosen precision |

These examples are planned acceptance fixtures, not claims of implemented features. The supported
shape subset can grow independently of specialised fast paths, while each public contract stays useful
outside the motivating solvers. Prioritise the intersection of demonstrated consumer value and reuse.

### Source evidence

| Consumer code | Repeated operation or missing capability |
|---|---|
| BSolver4D `cpp/numeric.hpp`: `eval`, `polynomialJet`, `productJet` | Polynomial evaluation and Taylor coefficients through second order |
| BSolver4D `cpp/baxter.hpp`: `largeAllJets`, `lattice`, `derivativeGrid`, `jetLattice` | Shared asymptotic seeds; selected trajectory outputs; recurrence jets in both shift directions |
| BSolver4D `cpp/baxter.hpp`: `omega`, `newton`, `recurrenceResidual` | Same matrix with four RHS; central differences; maximum and scaled residual reductions |
| qsccpp `include/series.hpp`: `ser_mul`, `ser_inv`, Laurent `LS` | Truncated convolution and inversion with explicit known coefficient ranges |
| qsccpp `include/mx.hpp`: `large_tri`, `cons`, `toe_mul`, `mul_toe` | Lower-triangular products/solves and Toeplitz structure |
| qsccpp `include/mx.hpp`: `QR`, `makeQRD`, `project`, `solve` | Reused QR; many RHS for the constraint chain rule; real/imaginary Jacobian columns |
| qsccpp `include/mx.hpp`: `glue_fit`, `glue_modes`, damping trials in `solve` | Residual norms, separate unfitted-mode diagnostics, repeated same-shape evaluations |

Paths refer to the local consumer checkouts; their equations are evidence for demand, not library dependencies.

### S1. Polynomial evaluation with jets — first additional compute API

Sketch: `poly_eval(coeffs, points, out)` and `poly_eval_jet<2>(coeffs, points, jets)`.
Accept shared coefficient sets, batched points and C1 views. Coefficients use ascending powers.
Jets mean Taylor coefficients `(f, f', f''/2!)`, matching BSolver, with a compile-time order cap of
two initially. Use simultaneous Horner evaluation, updating higher jet orders first so each update
uses the previous state. This avoids separate value/derivative passes and explicit power tables.

- First kernel: real and complex Horner, shared coefficients, optional jets. No transcendental dependency.
- Inverse-power evaluation is explicit `points = 1/u`; jets are with respect to that argument.
  A consumer must apply the chain rule and its exponential/power prefactor to obtain derivatives in `u`.
- Offer a documented composed arithmetic mode matching existing consumer operations and an explicit
  fused mode after D2 is certified. Each has its own reference; fusion can change output bits.
- QSC already precomputes power tables: benchmark Horner against D3 shared-table dots before choosing
  either for that workload. Keep both available when the table is reused across many Jacobian columns.
- Gate: degrees 0, 1, 23, 31, 61 and 100; empty polynomial = zero; cancellation, complex points,
  and analytic jet identities, against independent MPFR/MPC implementations.

### S2. Recurrence outputs and jets — extend the existing recurrence APIs

Sketch: `RecurrencePlan::run(seeds, weights, output_selection)` and
`run_jets<2>(seed_jets, weight_jets, output_selection)`. This adds capabilities to the scalar ring
recurrence and D4 vector recurrence; it is not a second competing recurrence abstraction.

- Select final state, final ring, an explicit ordered list of step indices, or full trajectory.
  Define step zero as the seed state and every later index as the state after that many updates.
  A selected ring must preserve the documented logical order, independently of physical ring rotation.
- Iterate forward or reverse through coefficient views, preserving sequential arithmetic within each
  trajectory. Outputs remain resident, so matching/solves/reductions can follow in the same batch.
- Normalised Taylor jets use truncated product convolution. Differentiate a division using the
  triangular coefficient relation with one shared reciprocal of the zeroth-order denominator;
  report a zero denominator per trajectory. This covers BSolver's `jetLattice` pole calculations.
- Bound registers and scratch: separate order-0/1/2 kernels and an explicit workspace requirement;
  benchmark fused jets against separate passes before adopting fusion at each precision.
- D4 forward tangents cover parameter directions; S2 jets cover a scalar Taylor expansion. Keep that
  distinction explicit rather than pretending that either already supplies a complete solver Jacobian.

### S3. Structured series algebra — general kernels, QSC-driven priority

Sketch: `convolve_truncated(a,b,n,out)`, `series_inverse(a,n,out)`,
`toeplitz_apply(coeffs,x,out)` and `triangular_solve(A,B,out)` with multiple RHS.

Start with direct convolution and fixed-order triangular kernels at the small series lengths seen
in these solvers. Parallelise independent output coefficients and series; respect sequential
dependencies in inversion and triangular solves. Reuse D2/D3 accumulation contracts.

Treat Laurent valuation and the range of known coefficients as metadata. Truncation is a mathematical
contract separate from floating-point precision: never fill an unknown coefficient with zero.
Initial low-level kernels can take ordinary coefficient arrays; a Laurent wrapper comes only when
range propagation rules have independent tests. Support shared triangular matrices across RHS,
exact zero-pivot status, and deterministic reductions. FFT convolution is a later measured alternative,
requiring its own accuracy contract; it is not a prerequisite for this family.

### S4. Device-side convergence summaries — small API, broad usefulness

Sketch: `norm_inf`, `norm2`, `scaled_residual`, and `summarize_status`, with optional segmented outputs.
Return multiprecision scalars and a status summary in resident buffers; host double conversion is an
explicit presentation operation. QSC currently converts several norms to double, while BSolver uses
maximum absolute residuals and a separate scaled recurrence residual.

- Define complex infinity norm as `max_i |z_i|`, separately from a componentwise max norm.
- For the 2-norm, use exponent scaling and a fixed reduction tree to avoid preventable intermediate
  overflow/underflow. Document every rounding step; this does not promise a correctly rounded exact norm.
- Specify empty-input values, exact tie rules for argmax, and zero-scale handling. Aggregate all existing
  fatal status bits and the first failing index; failed lanes must not disappear from a norm reduction.
- Support threshold comparison on-device, followed by one small readback for the host's acceptance
  decision. Damping/line-search order remains the consumer's existing order, even if trials run in parallel.

### S5. Reusable factorizations and accurate least squares — promote QR in D6

*Status (round 23-D6b): factor/solve separation for D6 Cholesky done as host-array calls (`Linalg::cholesky` returns L
and the first failing pivot; `cholesky_solve` / `trsm` take several right-hand sides). No factor handle or QR yet.*

Sketch: `factor_qr(A, workspace, options) -> QRFactor`, `factor.solve(B)`,
`factor.apply_q(B, transpose)`, plus factor/solve separation for D5 LU and D6 Cholesky.
Both real and complex multiple-RHS operations matter: QSC explicitly factors `JCD` once and solves
each column of `JCF`; BSolver's Omega construction solves the same 4x4 matrix four times.

Introduce full-rank Householder QR first; then deterministic column pivoting with explicit rank
tolerance and per-factor rank/status. Rank-deficient least squares needs a separately specified
complete orthogonal factorization or SVD path; basic pivoted QR alone must not claim minimum-norm solutions.
[LAPACK's least-squares API](https://www.netlib.org/lapack/lug/node27.html) distinguishes these contracts
and supports multiple RHS. Provide reusable factor objects, rather than silently caching by buffer address:
updating A invalidates its factor unless the caller explicitly supplies a new generation.

For LM, offer QR of the augmented system `[J; sqrt(mu) D]`, with the consumer supplying its damping
diagonal D, alongside D6 normal equations. Compare linear residual and solver convergence near difficult
points before choosing a default. CPU factorization with resident GPU products is a valid first backend;
move QR to the GPU only if end-to-end timing justifies it. Optional iterative refinement must recompute
residuals at an explicitly higher supported precision and report stagnation, not certify the entire solver.

### S6. Explicit precision conversion and reusable execution plans

Two runtime additions make the above practical across many iterations:

1. `cast<ToBits>(input,out)`: exact widening, nearest-even narrowing, componentwise for complex values,
   with existing status propagation and exponent-overflow checks. A consumer may explicitly switch among
   supported 32-bit-multiple precisions, rebuild precision-specific tables, and retry. Widening rounded
   data cannot recover lost digits; verification must recompute from the original inputs.
2. `ExecutionPlan`: reusable operation descriptions, buffer bindings and preallocated workspace for
   same-shape repeated evaluations. Separate preparation, rebinding and submission. Validate shapes,
   precision, aliasing and RAW/WAR/WAW dependencies; retain resources until completion. The first backend
   may re-encode each submission with cached metadata. Measure that before adding a Metal indirect
   command-buffer backend; do not assume arbitrary existing kernels can be captured and replayed.

Plans complement `CommandBatch` and C7 pipeline prewarming. They accept existing library operations;
arbitrary CPU residual callbacks cannot execute on the GPU. Shared tables require explicit data-generation
keys; consumers invalidate their tables when model parameters, truncation, nodes or precision change. Benchmark preparation
cost, amortisation, replay/encoding cost and workspace bytes separately.

### Later: matrix-free sensitivities, only after the forward path is productive

Extend D4 with blocks of directional tangents (`Jv`); consider an adjoint recurrence (`J^T v`) with
checkpoint/recompute control if profiling shows dense Jacobian storage or construction dominates.
Derivatives refer to the mathematical recurrence evaluated with a specified rounded arithmetic sequence,
not the discontinuous derivative of a floating-point program. QSC's Hermitian fit and antiholomorphic
rows require real parameter directions and real inner-product adjoints; a holomorphic shortcut is unsafe
for the complete residual. End-to-end JVP/VJP requires consumer derivatives of seeds, constraints,
matching and fits, not just a differentiated recurrence kernel. Do not replace the current solver algorithm
until directional finite-difference checks and convergence comparisons pass.

### Implementation order and benchmark gates

**First establish fixtures**, then add S1 and S4; extend recurrence outputs before recurrence jets;
build S3 on D2/D3; introduce factor handles and multiple-RHS solves alongside D5/D6, bringing QR ahead
of optional GPU Cholesky if consumer timings support it. Add casts early where needed and execution
plans after the repeated operation sequence is known. Matrix-free adjoints remain a later project.

For each API round, record a CPU MPFR/MPC baseline and the current composition of existing operations
before implementation. Use real consumer input shapes and cancellation-heavy cases at 224/256 bits
for QSC and 384 bits for the existing BSolver benchmark, plus the all-precision checks in section 6.
Include small BSolver batches and large QSC Jacobian batches, shared tables, multiple RHS, memory use,
device time and full wall time. Compare serial and available multicore CPU paths under equivalent
arithmetic contracts; charge setup and conversions consistently. A new arithmetic mode must be tested
against its own independent reference, not assumed bit-identical to an older composition.

After every accepted round, rerun affected solver fixtures and the existing regression suite. Check
BSolver's independent on-shell verification with increased terms/shifts and QSC's unfitted mode diagnostics,
not only an internal recurrence identity. Working precision and small residuals alone are not certified
solution accuracy. Publish performance claims only after correctness and matched CPU/GPU timing pass.
Before stabilising each API family, also run its independent example from the reuse table above and
verify that its public interface requires no solver-specific types or assumptions.

---

## 8. Literature-driven experiments (surveyed 2026-10-07)

These are research candidates, not accepted speedups. The target machine was checked locally:
Apple M5 Max, macOS 26.6.2, installed macOS SDK 26.5. Separate what each source demonstrates
from the proposed Metal adaptation. Preserve the generic public operations from section 7.

### L1. Ozaki Scheme II / residue-number GEMM — highest-upside new direction

[Ozaki, Uchino and Imamura (2025)](https://arxiv.org/abs/2504.08009) reconstruct matrix products
using modular arithmetic and the Chinese remainder theorem, with low-precision GEMMs as building
blocks. A [2026 multiple-precision CPU follow-up](https://arxiv.org/abs/2609.27831) tests 53–2048-bit
significands; its shared-exponent conversion introduces an accuracy issue, and the reported within-one-ulp
results are not a proof of correct rounding. The [2026 FP8 adaptation](https://arxiv.org/abs/2603.10634)
extends the approach to FP8 matrix units.

**Our proposed adaptation:** an experimental backend for dense `gemm`/`syrk`, retaining the limb
backend for pointwise arithmetic and recurrences. Apple documents M5 acceleration through
[Metal TensorOps](https://developer.apple.com/videos/play/wwdc2026/330/), including quantized inputs.
That API availability alone does not establish exact integer accumulation or an applicable error bound.

First probe the installed SDK's supported operand/accumulator types and small integer products.
Then derive safe digit sizes, bounded inner-product lengths, and partial-sum bounds. If an FP16/FP32
route is used, every intermediate partial sum must fit its exactness bound; quantized input support
must not be mistaken for INT8-to-INT32 arithmetic. Check cancellation and dequantization semantics.
Only proceed if those requirements can be established, rather than relying on empirical agreement alone.

At 224/256/384 bits, include all residue planes, reconstruction, scaling and transfers in wall time.
The CPU follow-up notes that byte-sized coprime moduli alone have insufficient reconstruction capacity
for its higher precisions, so include multi-digit residue costs. Account for the full input exponent spread:
shared-exponent conversion can lose small entries. Either prove a rounding decision with conversion-error
bounds and an exact fallback, or expose a separately named accuracy mode. Dense matrices are the initial
target; four-component updates are too small to assume the same benefit.

**Result of the capability probe (round 16, `tests/tensorops_probe.mm`,
`benchmarks/results/round16_tensorops_probe.txt`).** On this M5 Max with the installed SDK,
`matmul2d` (`relaxed_precision=false`, MSL 4.0 compiled at run time, tensors built in-shader from
device pointers) gives:

- int8×int8→int32 and uint8×uint8→int32 products equal to 64-bit CPU sums for K ≤ 16,384 on random
  data and on sampled entries of 2048³/4096³ products.
- int32 overflow **wraps modulo 2³²** (deterministic, not saturating), exactly at the predicted K
  (131,072 for int8 extremes; 33,026 for uint8 extremes). The accumulator is therefore an exact residue
  mod 2³² for any K, and an exact integer for K ≤ 33,025 (uint8) or ≤ 131,071 (int8).
- 29–54 int8 TOPS for 2048³–4096³ on a heavily loaded host (raw bit-throughput ≈ 2.5× our scalar
  limb products; the advantage for GEMM would come from matrix-unit data reuse).

The exactness gate is passed, so L1 continues with the two steps below. Stop L1 if step L1b is not at
least 5× faster than the limb GEMM baseline at n ≥ 512 and 256 bits, including all conversion costs.

- **L1b — exact integer residue GEMM.** For integer matrices with ≤ 256-bit entries (one shared
  scale), compute C = A·B exactly: 16-bit coprime moduli covering 2·256 + log₂K + guard bits (≈ 34
  moduli), each residue split into two uint8 digits (4 digit GEMMs per modulus, K blocked at ≤ 33,025),
  per-modulus reduction, and GPU CRT reconstruction. Validate against GMP; measure the full pipeline
  against a straightforward sequential-`fma` limb GEMM at n = 128…2048 (that baseline doubles as D6's
  first kernel). Cost model to check: ~136 int8 GEMMs ≈ 7 ms at n = 1000 versus ~0.4 s estimated for
  limb `fma` (to be measured).
  **L1b result (round 18, `benchmarks/residue_gemm.mm`):** exact, equal to GMP on every checked output
  (all 16,384 at n = 128). 33 moduli. n = 512: 4.2 ms vs 59 ms composed `mul`+`add` limb GEMM (14×) and
  469 ms `fma` limb GEMM; n = 1024: 21 ms vs 419 ms (20×) / 3,759 ms; n = 2048: 104 ms. **Gate passed**
  (baselines untiled, so the ratio is an upper estimate; a tiled limb GEMM is the fair D6 comparison).
  At n = 1024 the int8 products take 6.2 ms (≈ 45 TOPS); Garner reconstruction (8.4 ms, 32-bit divisions)
  and residue conversion (5.3 ms) dominate next: use Barrett/Montgomery reduction and fewer, wider limb
  passes. Side finding: the generic `fma` is ~8× slower than composed `mul`+`add` in this loop, so D2
  needs a specialised two-term path.
- **L1c — floating-point layer.** Scale rows/columns to fixed point with exact handling of the input
  exponent spread (exponent bands or an exact fallback; never silent truncation), then one rounding of
  each exact dot product. This is the L3 `exact_dot` contract; reference: exact MPFR products plus
  `mpfr_sum`. Report the band count and fallback rate on QSC-shaped Jacobians.
  **L1c result (round 20-D6, `src/linalg.mm`, `benchmarks/linalg.mm`).** Each line (row of the left operand,
  column of the right) is split into exponent bands of at most G = 64 bits. Every entry is an exact
  (bits+G)-bit integer, and all bands of one side are concatenated into one extended operand, so one
  residue GEMM per modulus holds every band pair. Single-band outputs are rounded straight from the Garner
  integer; multi-band outputs are rounded from an exact accumulator. The exact host `exact_dot` takes lines
  with more than 4 bands or a spread above 512 bits. Output is bit-identical to MPFR at 64–1024 bits, including
  ties. On QSC-like data (column scales 2^±60, 25% zeros, rare 2^-150 outliers), 4–5% of columns need two bands
  and none fall back. The cost is 37 moduli at 256 bits (34 for one-exponent-range data).
  Stages at n = 1000 (moderate data):
  - int8 products: ≈ 8 ms (≈ 36 TOPS, near the TensorOps peak, so further gains need fewer int8 operations);
  - digits: 3.9 ms (Barrett, 4 moduli per input read);
  - reconstruction: 2.5 ms;
  - combine: 1.5 ms;
  - host: ≈ 4 ms.
  Possible further steps: fuse the three products and the combine through cooperative tensors to save the
  int32 round trip; try 14-bit moduli with Karatsuba digits (3 GEMMs of K instead of 4K-equivalent per modulus).

### L2. Certified approximations with compacted exact retries

The established [MPFR rounding-certification mechanism](https://mpfr.org/mpfr-current/mpfr.html#Rounding_002dRelated-Functions)
determines whether an approximation with a known error bound permits the requested rounding.
This is an older idea, not a newly published GPU algorithm.

**Our proposed GPU design:** extend B5's bounded partial-product idea to selected fused dots and
matrix outputs. Compute an approximation and an outward error enclosure; accept only when all values
in that enclosure round to the same target value, including ties and exponent boundaries. Compact
uncertain output indices into a second exact kernel instead of making a whole SIMD group follow a
long fallback branch. Replay from original inputs. Do not certify only the last stage of a rounded
recurrence and assume the whole trajectory has the same bits.

This can preserve a single-round exact-dot contract. It does not automatically preserve D3's sequential
FMA contract, which needs its own validation. Measure retry rates, enclosure cost and compaction cost;
adversarial halfway and catastrophic-cancellation cases must exercise the fallback. Approximate low
precision alone cannot generally supply hundreds of correct bits.

### L3. Delayed normalisation and exact dot accumulation

[MPFR's sum/dot interface](https://mpfr.org/mpfr-current/mpfr.html#Arithmetic-Functions) supplies useful
single-round reference semantics; its dot operation is marked experimental and does not handle
intermediate overflows/underflows, so supplement it with independently exact references for those cases.

**Our experiment:** accumulate unrounded limb products in bounded integer bins or a carry-save
representation, then propagate carries and normalise once per output. This could reduce repeated
packing in long dots, Gram products and Fourier projections. Benchmark an opt-in `exact_dot` mode
alongside sequential and pairwise modes; all three have distinct rounding contracts.

Do not allocate a dense accumulator spanning the library's entire exponent range. Start with measured
bounded exponent spans, prove accumulator capacity including segment length, and provide an exact
wider/sparse fallback. No silent discard of tiny terms: cancellation can expose them. Delayed carry
propagation also requires an explicit overflow bound before any integer addition is deferred.

### L4. Cooperative limb arithmetic with scan-based carries

[Oancea and Watt, GPU Implementations for Midsize Integer Addition and Multiplication (2024)](https://cs.uwaterloo.ca/~smwatt/pub/reprints/2024-langcompan-gpu-arith.pdf)
describes parallel carry propagation and register-oriented multiplication work partitioning.
Its largest-integer results do not predict performance at LimbForge's 64–1024 bits.

Use it to make E1 concrete: prototype 4/8/16/32 cooperating SIMD lanes per number, with an associative
generate/propagate carry scan and a bounded number of limbs per lane. Preserve the exact integer result
before existing packing. Measure recurrence step latency at small trajectory counts separately from
large-batch throughput. This is an alternative backend selected by shape, not a universal replacement
for one-thread-per-number kernels. FFT multiplication is not justified at current widths by that paper.

### L5. Multiple-component arithmetic — useful comparison, hardware-specific port

[Chen and Verschelde, Multiple Double Arithmetic on NVIDIA Tensor Cores (2026)](https://arxiv.org/abs/2607.06881)
separates matrix work from expansion renormalisation using an Ozaki-like approach on FP64 tensor cores.
Treat this as an alternate representation/backend study; the demonstrated FP64 hardware path is not
an Apple implementation. An FP32-component Metal variant would require new range, error and rounding
analysis. Do not equate component count with a guaranteed number of accurate bits or replace the public
limb representation without evidence that conversion and renormalisation costs pay off.

### L6. Mixed-precision linear solves — established method, modern accelerator opportunity

[Carson and Higham (2018)](https://epubs.siam.org/doi/10.1137/17M1140819) analyses iterative refinement
using distinct factorization, working and residual precisions, with explicit convergence conditions.
This supplies a foundation for S5's refinement proposal rather than a new core arithmetic contract.

Try a cheaper factorization/preconditioner and higher-precision residuals on the linear systems built
by consumers. Benchmark supported limb precisions first; an accelerated native-precision factorization
is a separate backend experiment. Report convergence/stagnation and fall back when needed. A small
backward error does not imply small forward error for an ill-conditioned system, and this experiment
does not establish correctness of a nonlinear solver's branch selection or truncation.

### Recent comparator: GPU multiprecision already has active implementations

[Kouya, Construction and Performance Evaluation of an Arbitrary-Precision Floating-Point Arithmetic Environment on CUDA (2026)](https://arxiv.org/abs/2608.00085)
describes `mpc_cuda`, per-thread scratch arenas and compile-time fixed-precision types. It claims
host-identical arithmetic for those types but gives a different accuracy contract for fixed-precision
elementary functions. These are reported CUDA results, not reproduced Metal benchmarks.

Use its contracts and workloads when designing comparisons. LimbForge's fixed-size core is already
allocation-free; adding an arena to basic arithmetic would not address the same bottleneck. Explicit
workspace sizing/reuse may matter later for transcendental or dynamically sized operations. Match
precision, rounding, operation order and timing boundaries before comparing implementations.

### Research order

*Status (2026-10-07): L1 probe, L1b and L1c passed (see L1; L1c is the D6 `Linalg` API, round 20-D6). Next is L4 (in progress).*

Run a small **L1 Metal arithmetic-capability probe** first; stop that branch if exactness requirements
cannot be met or established. In parallel conceptually, L4 supplies the strongest direct candidate for
small-batch recurrence latency. Next measure L3 packing/normalisation cost before investing in L2's
certified retry machinery. L6 depends on the linear-algebra APIs; L5 remains a comparison study.

For every experiment: preserve a trusted baseline, specify the arithmetic contract, test against an
independent reference on the physical GPU, then measure total wall time and memory on representative
shapes. No performance ratios from another GPU or paper are promised for this Mac.
