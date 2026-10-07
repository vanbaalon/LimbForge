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
- **D3. Segmented dot products and batched GEMV (tips #3)** — M. `segmented_dot(a,b,K,out)`,
  `segmented_dot_shared(T,x,K,out)` (shared table broadcast via C1), real and complex. **Fixed order contract:**
  `s₀=RN(a₀b₀)`, `s_k=fma(a_k,b_k,s_{k−1})` sequentially (bitwise reproducible regardless of thread configuration); optional
  `pairwise` mode with a fixed adjacent-pair tree (reuse round-10/11 tree code) for long segments. One thread per segment
  for K ≤ ~128; SIMD-group-per-segment variant for low segment counts. Fourier transforms = batched GEMM → D6 kernel.
- **D4. Vector recurrence (tips #1, highest consumer value)** — L, needs D2 (and C1 for sharing).
  `vector_recurrence(bits, v0, p, q, r, out, lanes, steps, lanes_per_weight, flags)`; per step
  `s=q_k·v` (fixed order via D2/D3), `v_i ← cfma(p_{k,i}, s, v_i) (+ r_{k,i})`. Flags: affine source `r`; general 4×4
  `M_k` instead of `1+p qᵀ`; write `v` every step (layout `out[k*lanes+lane]`). **Also evaluate a fused tangent mode**
  (base `v` and tangent `dv` advanced together: `ds=q·dv+dq·v`, `dv ← dv + p·ds + dp·s`) which removes the stored base
  chain and the separate `r` pass — confirm the `dp/dq` data shape with the qscmx owner before building it.
  Kernel design: one thread per lane; `v` (and `dv`) in registers — check spills with `pipeline_info` at N=7,8 (4 complex
  ≈ 88 words each); shared `p,q` staged per step in threadgroup memory for the `lanes_per_weight` group. Test: MPFR chains
  (same contract) at 224/256 bits, 150 steps, random data with |p qᵀ|≈1 plus cancellation-heavy cases; then qscmx end-to-end.
- **D5. Batched 4×4 complex LU / solve / inverse / det (tips #5)** — M. One thread per matrix. Specify a deterministic
  pivot rule that needs no rounding (e.g. max of `max(|re|,|im|)` by exact exponent/limb compare, lowest index on ties) so
  the MPFR reference reproduces it bitwise. Ops `lu4`, `solve4(nrhs)`, `inv4`, `det4` with per-matrix status (singular).
- **D6. Dense real GEMM / SYRK / Cholesky (tips #6)** — L. Tiled GEMM: each thread owns one output computed with the
  D3 sequential-fma order over k (reproducible); A/B tiles in threadgroup memory. `syrk` (`JᵀJ`, lower triangle) first —
  it dominates for J ≈ 2n×n. Cholesky: start on CPU (n³/3), move to a GPU blocked right-looking version if profiling shows
  it dominates. QR optional later. Sizes 200–1000, 256 bits.
- **D7. Transcendentals at buffer level (tips #8)** — L, lowest priority: complex `exp`, `log`, integer `powi` with explicit
  accuracy contracts (faithful or correctly rounded via Ziv-style retry).

### Phase E — Latency-bound workloads: cooperative intra-number arithmetic — Track T1, after B

For BSolver (≈128 lanes × 600 steps) the GPU runs ~128 threads. Two complementary steps:
- **E0 (consumer-side, S):** round 13 measured ~165 µs per 384-bit recurrence step independent of lane
  count up to 4,096 lanes (saturation near 16,384). Recommend BSolver batch all `u` points / Newton finite-difference evaluations into one
  `recurrence` call (more trajectories per submission) — cheapest win; document in the BSolver integration notes.
- **E1 (L, research):** one number per 8/16/32 SIMD lanes: limb-parallel multiply with `simd_shuffle` broadcasts, carry
  resolution with `simd_prefix_exclusive_sum`/ballot loops, normalisation via ballot + `clz`. Prototype `mul` + `add` in a
  `recurrence_coop` kernel at 384 bits; compare per-step latency against the thread-per-lane kernel; division via a
  cooperative reciprocal only if the prototype wins. Keep the thread-per-lane kernel as the default for large lane counts.

---

## 5. Order, tracks and dependencies

| Sprint | T1 kernels (owns `core.hpp`) | T2 runtime (`engine.*`) | T3 consumer prims | T4 linear algebra |
|---|---|---|---|---|
| 0 | commit this plan (`docs/optimization-plan.md`) and `tips.md` | — | — | — |
| 1 | A1 ✓, A2 ✓ (round 13), A3; **B0** | C1 operand descriptors | D1 bridge ✓ (round 13-D1) | — |
| 2 | B3/B4 (under gpu-codegen rules) | C2, C3, C4 | D2 fma (CPU+ref first) | — |
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
