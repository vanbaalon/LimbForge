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

The cause of these GPU mismatches is unresolved. Do not treat the rejected
measurements as validated performance results. The accepted library uses exact
schoolbook multiplication. See ../../docs/optimizations.md and the `round5_*`
metadata/test logs in ../results.

To investigate a patch, create a separate checkout at its stated base revision,
apply it with `git apply`, build in Release mode, and run the full CPU and physical
GPU tests before measuring it. Never overwrite accepted result files.
