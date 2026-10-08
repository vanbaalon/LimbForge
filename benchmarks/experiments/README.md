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
- `sum_width_probe.mm` (diagnostic, not a patch; build as in its header): the exact
  two-term sum of the update kernels at every precision and both instantiated widths
  against MPFR. With argument 1 (no rounding of the workspace width) 8 of 62 widths
  fail on the GPU; with 4 (the library) none fail. See ../../docs/gpu-codegen.md
  section 7.

The cause of these GPU mismatches is unresolved. Do not treat the rejected
measurements as validated performance results. The accepted library uses exact
schoolbook multiplication and a restricted, separately validated unary square. See ../../docs/optimizations.md and the `round5_*`
metadata/test logs in ../results.

To investigate a patch, create a separate checkout at its stated base revision,
apply it with `git apply`, build in Release mode, and run the full CPU and physical
GPU tests before measuring it. Never overwrite accepted result files.
