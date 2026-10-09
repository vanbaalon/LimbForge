# Rejected arithmetic experiments

These patches are research records, not enabled library features. Each failed
GPU/MPFR validation and was removed from the installed implementation.

- `known_product_top_bit.patch`: based on accepted Round 3 (`2172965`); removes
  the generic highest-bit scan for multiplication. The Round 3 rejected CSV is
  incomplete because 1024-bit complex division fails.
- `comba_full.patch`: based on Round 4 (`7f25fe5`); 96-bit column accumulator.
  CPU tests pass; GPU multiplication fails at 800 bits.
- `comba_hybrid.patch`: based on Round 4; Comba at 64–384 bits and schoolbook
  above 384 bits. GPU complex arithmetic fails at 128 bits.
- `comba_32bit_carries.patch`: based on Round 4; explicit low/middle/high words,
  with bounded 33-bit carry sums. GPU real and resident checks fail.

- `square_full.patch`: based on the initial accepted unary-square core in
  `bea92b5`; removes the width fallback and uses square inside complex division.
  The full-width square fails at 992 bits and the complex-division variant fails
  at 384 bits. The public GPU symmetric square is selected only at 384 bits;
  complex division retains ordinary multiplication.

- `coop_divergent_branches.patch`: based on round 17-L4 (`3e1446d` plus
  `benchmarks/coop_recurrence.mm`; apply to that file). It is the first
  cooperative prototype: its shuffles and ballots sit in branches that are
  uniform within a trajectory's lane group but divergent across the groups of a
  SIMD. It is bit-exact on the physical GPU without validation, but under
  `MTL_SHADER_VALIDATION=1` (with `MTL_DEBUG_LAYER=1`) 1024-bit G=4/8 recurrences
  and one SIMD group of the G=4 `cmul` kernel fail nondeterministically. The kept
  version branches only on SIMD-uniform conditions, but its G=4 shapes (from 512
  bits) still fail under validation; see ../../docs/experiments.md.

- `cholesky_gpu_panel.patch`: based on round 23-D6b (apply on top of that commit).
  Not a failure but a measured rejection: the Cholesky panel rows below the
  diagonal block on the GPU (`panel_rows`, one thread per row, exact accumulator of
  `LF_ACC_WORDS` words, rows with product exponents spread over 1,072 bits finished
  on the host). Bit-identical to the MPFR sequence (all `linalg_cholesky` cases), but
  slower: about 4 ms of GPU time per panel at n = 1000, 256 bits, block 32 (one exact
  multiply-add chain per thread is latency-bound), against about 3.5 ms for the host
  panel on a host with load average 150-260; total factorization 220-250 ms against
  165-195 ms.
- `numerics_core_square.patch`: based on round 32 (S1/S4; apply on top of that commit).
  Squares in the norm kernels (`src/numerics.metal`) through core `square()`, which is
  the symmetric product on Metal at 384 bits. GPU-only failure: `norm2` of one real
  384-bit entry was wrong for 200 of 200 random inputs (wrong exponent and limbs),
  while 352 and 416 bits (schoolbook square) and the engine's unary `square` kernel
  at 384 bits were exact. The library squares with `mul(a, a)` there (same exact
  square, same rounding). See ../../docs/gpu-codegen.md section 8.
- `qr_panel_team.patch`: based on round 31-S5 (apply on top of that commit with
  `git apply`). A measured rejection: the QR panel as one team per panel (one pool
  dispatch, spin barriers between the per-column phases, serial steps on member 0)
  instead of two pool dispatches per column. Bit-identical to the MPFR sequence, but
  slower on the loaded host (n = 400, 256 bits, load about 40, interleaved: panel
  250-740 ms against 185-455 ms), because spinning members are descheduled. A pool whose
  workers poll before blocking was slower too and is not kept.
- `pool_dispatch_probe.cpp` (diagnostic, round 40-S5b; build as in its header): dispatch
  cost of the host worker pool before round 40 (waits for every worker), the round 40 pool
  (returns when every item is done) and a variant that spins 20 or 100 µs before blocking.
  On the loaded host (load 30-60) the claim-based pool was faster in most paired runs
  (e.g. 2000 dispatches of 72 × 3 µs: 167-414 ms against 191-280 ms; 72 × 10 µs: 598-772 ms
  against 741-811 ms) and is kept; the spinning variant had no consistent gain (85-884 ms)
  and is rejected, as the unbounded spin barrier of `qr_panel_team.patch` was before.
  A static split of the QR panel rows into one chunk per worker (round 40 development) was
  slower than dynamically claimed row blocks on the mixed performance/efficiency cores and
  was not kept.
- `qr_lookahead_depth2.patch`: based on round 45 (complex column pivoting and workspace release; apply on
  top of that commit with `git apply`). A measured rejection: a second block of look-ahead in the unpivoted
  real and complex QR (`LIMBFORGE_QR_LOOKAHEAD=2`). The trailing updates run in order on one side thread
  (`SideQueue`), block b's split into the panel after next and the rest, so the rest overlaps two panels and
  the calling thread waits only for its next panel's columns; the side thread gets its own scratch pool and
  report so that two products can run at once. Bit-identical to the MPFR replays (`test_limbforge_qr` and
  `test_limbforge_qr_complex --quick` with depth 2) and to depth 1 in every benchmark repeat, but not faster:
  interleaved A/B (`qr_limbforge --lookahead`, `qr_complex_limbforge --lookahead`, load 20-33,
  `../results/round45_qr_lookahead.csv`), depth 1 / depth 2 factor + solve medians 0.97-1.03 at n = 400/1000,
  224/256 bits, real and complex, except two noisy real n = 400, 224-bit runs (1.25 and 0.73, equal minima).
  Instrumented single runs (real, n = 1000, 256 bits): the waits for the side thread fell from 46-61 ms to
  0-20 ms, but the next panel's update (on the critical path with the panel) rose from 65-88 ms to 118-128 ms
  under the concurrent GPU products.
- `sum_width_probe.mm` (diagnostic, not a patch; build as in its header): the exact
  two-term sum of the update kernels at every precision and both instantiated widths
  against MPFR. With argument 1 (no rounding of the workspace width) 8 of 62 widths
  fail on the GPU; with 4 (the library) none fail. See ../../docs/gpu-codegen.md
  section 7.

- `coop_validation_probe.mm` / `.metal` (diagnostic, CMake target `coop_validation_probe`;
  round 41-L4b): a switchable copy of `src/cooperative.metal` (exchange through
  threadgroup memory, SIMD barriers, active-lane and lane-mapping counters, per-step trace,
  extra register pressure, the old short-circuit rounding vote) whose every dispatch is
  compared with the CPU. It found the divergent rounding ballot and bisected the remaining
  validation-only G = 4/8 failures (they need concurrent GPU work from another process to
  appear often). `simd_validation_repro.mm` (target `simd_validation_repro`) is a
  LimbForge-independent shuffle kernel that did not reproduce them. See
  ../../docs/experiments.md (41-L4b) and ../../docs/gpu-codegen.md section 10.

- `vr_compile_probe.mm` (diagnostic, built by hand; see its header): cold pipeline compile time of one
  kernel specialisation from given `core.hpp`/`kernels.metal` files, with a nonce that defeats the OS shader
  cache and a watchdog. Round 46 used it for the fused `vector_recurrence` compile table
  (../results/round46_fused_wide_compile.txt) and the `noinline` experiment (../../docs/gpu-codegen.md section 11).

The cause of these GPU mismatches is unresolved. Do not treat the rejected
measurements as validated performance results. The accepted library uses exact
schoolbook multiplication and a restricted, separately validated unary square. See ../../docs/optimizations.md and the `round5_*`
metadata/test logs in ../results.

To investigate a patch, create a separate checkout at its stated base revision,
apply it with `git apply`, build in Release mode, and run the full CPU and physical
GPU tests before measuring it. Never overwrite accepted result files.

- `section9_fma_only.patch` (base `10eb888`): rejected partial isolation of exact complex FMA in
  batched GEMM; the dense audit still found a composed 352-bit mismatch. Context and logs:
  [section9_gemm_dense.md](section9_gemm_dense.md).
- `section9_reference_abi.patch`, `section9_global_load.patch`, and
  `section9_real_isolation.patch` (base `10eb888`): rejected attempts to isolate
  dense complex GEMM failures. Selected widths passed, subsequent widths failed;
  see `section9_gemm_dense.md` and `../results/section9_p0_*`.
- `section9_component_parity.patch` / `section9_component_grid.patch` (base
  `10eb888`): separate real and imaginary GEMM outputs. The alternating-component
  variant fails at 960 bits; the uniform-component sweep was interrupted and
  establishes no acceptance. Padded local-storage follow-up is still experimental.

- Section 9 shared-source variants, base `e91a215`: private/packed/split/component
  Horner variants and the callback-sensitive vector adapter are retained in
  `section9_shared_sources_*_rejected.patch`; failure logs and candidate status are in
  `section9_shared_sources.md`. Standalone vector baseline does not independently
  reproduce the expanded audit's failure. No compiler root cause is asserted.
