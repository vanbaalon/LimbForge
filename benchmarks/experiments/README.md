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

The cause of these GPU mismatches is unresolved. Do not treat the rejected
measurements as validated performance results. The accepted library uses exact
schoolbook multiplication and a restricted, separately validated unary square. See ../../docs/optimizations.md and the `round5_*`
metadata/test logs in ../results.

To investigate a patch, create a separate checkout at its stated base revision,
apply it with `git apply`, build in Release mode, and run the full CPU and physical
GPU tests before measuring it. Never overwrite accepted result files.
