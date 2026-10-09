# Complete GEMM width/layout/clock measurement grid

Thirty-two new checked measurement records extend the unchanged production
1.3.1 library, using the verified benchmark subset selector at `dd1c236`.
Together with eight original records, they cover 36 workload/layout/clock
configurations and 120 numerical-contract keys:

- 352/384/448 bits, host/resident and CPU-interleaved/verified-warm-call clocks;
- Fourier 66x130 times 130x3800, all four contracts;
- 4x4 batches of 10,000 and 100,000, composed/fused/matrix-level Gauss contracts.

The large 4x4 comparison omits the per-matrix exact Linalg driver. A fair batched
exact API/comparison remains open. The native per-product Gauss experiment is
separate and remains rejected; the Gauss path here uses three real GEMMs.

All initial, warm-up and timed GPU outputs and every timed CPU output pass their
independent MPFR sequence. Three repetitions rotate selected CPU/GPU paths.
All four CPU references and the independent high-precision exact oracle are
prepared even when only three GPU paths are selected. Source/binary/archive
identities match the 31 CPU-width and 16 focused normal/Metal subset gates.
No new shader context, installed API, arithmetic sequence or numerical default
is introduced by this measurement round.

For the verified-warm-call Fourier profiles, median GPU wall times are:

| Bits/layout | Composed (s) | Fused (s) | Exact embedding (s) | Matrix Gauss (s) |
|---|---:|---:|---:|---:|
| 352 host | .106255 | .208530 | .010935 | .078557 |
| 352 resident | .101029 | .206623 | .006795 | .075069 |
| 384 host | .112018 | .207880 | .011907 | .094967 |
| 384 resident | .114377 | .209907 | .007608 | .091900 |
| 448 host | .135519 | .246762 | .014900 | .129048 |
| 448 resident | .135677 | .249820 | .009639 | .124704 |

Exact embedding is fastest in these bounded-exponent Fourier fixtures. It also
has lower global error units (about .47–.55) than the sequential contracts
(about 7.09–8.01) and matrix Gauss (about 10.16–11.77). These remain distinct
rounding contracts. Global error units are normalized infinity errors in units
of 2^-B; they are not per-entry relative errors or a general digits-correct claim.
The fixtures use random exponents in [-4,4], not saved solver Jacobians.

All CPU medians, device times, quartiles, ranges, accuracy and raw sample orders
are retained for every profile. Host wall includes reused-layout preparation,
transfers, encoding, submission, waiting/finalization and reconstruction. Bridge
conversion and allocation/compilation are excluded. Resident timing starts from
path-specific resident layouts; Gauss sums/reconstruction remain timed. One
verified untimed call before each sample does not prove continuously warm clocks.
CPU-interleaved measurements do not prove controlled cold clocks.

Recorded load remains part of every key. Whole-process maximum RSS ranges from
607,223,808 to 6,444,908,544 bytes; peak memory footprint ranges from 1,376,339,192
to 7,715,657,104 bytes across new profiles, including CPU reference/oracle storage.
These are process measurements, not API scratch or solver memory measurements.
No idle consumer timing, automatic default or extrapolated cutoff is accepted.

The first completed 384-bit profile was initially rejected by a collector that
expected `sample,` rather than `cpu_sample,`/`gpu_sample,`. Its successful
benchmark was not rerun. All 12 CPU and 12 GPU samples, per-path repetitions and
CSV medians were independently checked; original collector rejection bytes are
preserved under `section9_gemm_extended/collector_parser_rejection`.

Run `python3 benchmarks/experiments/section9_gemm_extended/verify.py` to check
the complete grid, raw medians and hashes. The expanded JSON/header in
`benchmarks/results/section9_p13_gemm_break_even_expanded.*` includes all 120
measurements; the old 14-key artifacts remain unchanged. A compiled C++ factory
checks all 120 actual keys, 120 unmeasured widths and unknown profile/shape queries.
Ten nonterminal/failure/hash/clock/identity/idle/overwrite guards pass. Include
one generated factory header at a time and query its exact stored profile.

Raw records live in `benchmarks/results/section9_gemm_extended`,
`section9_gemm_extended_continued` and `section9_gemm_352_fourier_warm`.
The full source/build identities and collection drivers are retained. Remaining
work includes accepted idle-consumer calibration, batched exact 4x4 products,
tile/performance candidates and the separate exact-trial 3x target.
