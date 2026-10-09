# Grouped exact trailing updates: retained experiment

The isolated source at `25c8852`, based on main `f9be84b`, combines the retained
exact-trial/look-ahead API with a private grouped single-band SYRK hook. It packs
independent host-ready trial inputs contiguously and dispatches their digit,
residue-product and exact reconstruction passes together. The modulus count
covers the widest trial band. Reconstruction preserves `RN(C - exact dot)`.
Zero/status/multiband lines decline the entire fast path before encoding and
retain the existing resident SYRK/fallback/repair path. This is a private hook
for freshly host-packed inputs, with no preceding GPU producer.

The full candidate patch, every changed source file, unchanged arithmetic bases,
generator and supplementary reference are retained here. The source README in
the archive is the original pre-gate snapshot; this report records terminal results.
No installed API, version, numerical default or automatic policy is promoted.

All 31 widths pass independent MPFR direct-update and factor/solve replays,
normally and under Metal validation, using the default band configuration. Other
band configurations are not qualified for promotion. Full 26/26 normal and 19/19 Metal regressions
pass. The subsequent 131x65/count3 maximum-band fixture also passes all 31 widths
in both modes, including zero terms and a trial spanning 64 binades. Dense checks
require positive grouped counters; status/zero/multiband rejection, old-output
statuses, cancellation, tail shapes, two calls per batch, busy workspace release,
and mixed/late pivot failures are covered.

Only after these gates, a three-repeat 352-bit n944/count8/three-RHS host comparison
checked every CPU/GPU factor and solve against its own rounding contract:

| Path | Factor median (s) | Solve median (s) | Factor+solve median (s) | Wall range (s) |
|---|---:|---:|---:|---:|
| Production sequential scalar | 0.939305 | 7.071702 | 8.011007 | 7.572872–8.167252 |
| Eight exact Linalg factors/solves | 0.574930 | 0.095783 | 0.674707 | 0.656118–0.691271 |
| Grouped exact panels | 0.501112 | 0.126780 | 0.628796 | 0.615982–0.629637 |

Factor/solve medians are computed independently, so their sums need not equal
the combined median. The grouped factor is about 1.15x faster than the exact loop
in this profile; the full factor/solve is about 1.07x faster. Each grouped factor
reports 224 grouped trial updates across 28 grouped calls. The 3x factor target remains unmet. The separate
earlier look-ahead profile's 0.528-second factor is historical evidence, not a
paired same-run comparison against grouping.

This is a busy-host CPU-interleaved profile, with eight active MPFR workers despite
the requested pool of 18. Input-array bridge conversion is excluded; host packing,
transfers, encoding, finalization and factor/solve waits are included. The batched workspace retains 385,695,744 idle bytes; exact Linalg reports
137,895,192 bytes of host-thread scratch and 774,024,408 idle resident bytes.
Both report zero busy bytes at the end of each sample. Whole-process maximum
RSS is 3,067,052,032 bytes, including CPU references, with peak memory footprint
4,288,220,160 bytes. These are different memory scopes. No idle consumer timing,
cutoff or universal speed claim follows.

The 352/384/448 by n200/400/944 grid, further update/panel scheduling and consumer
acceptance remain open. Repeating this unchanged profile does not establish the
missing 3x target. Run `python3 benchmarks/experiments/section9_grouped_exact/verify.py`
to verify archived source/log hashes, exact scope, counters and raw timing medians.
