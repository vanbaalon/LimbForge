# WolfNum exact batched 4x4 baseline

Uninstalled experiment based on main `97d6b28`, WolfNum **1.3.1 / API 1**.
The public headers, numerical implementation, defaults and installed package are
unchanged. The explicit experimental CMake targets are excluded from the default
build and installation. This baseline is retained for the next optimization round.

## Decision

Independent correctness qualification passes. **Performance promotion is rejected:**
the exact prototype takes 1.054–4.582 times composed GEMM wall time in all 24
recorded configurations. The operations have different rounding contracts; the
exact result rounds the full dot once, while composed GEMM rounds intermediate
products and sums. No public exact4 API, minor release or default change follows.

## Qualification

- All 31 supported widths: small and dense real/complex cases, normally and under
  Metal validation; dense cases contain 4097 matrices with three resident repeats.
- All 26 normal library suites and 19 Metal-validation suites pass.
- All 31 widths: supplemental overflow cancellation, far-epsilon midpoint and
  old-output component statuses, normally and under Metal validation.
- All 31 widths: independently checked 1/32-call batches in both validation modes.
- All 31 comparison CPU references, six rejected selectors/shapes and eight
  focused host/resident GPU comparison checks pass.

The original full-suite collector rejected a passing normal suite because CTest
used a different success-summary spelling. The continuation verified the raw
26 unique passing test names and ran only the outstanding Metal suite. Both
original rejected metadata and the explicit correction are preserved.

## Measured grid

352/384/448 bits; 10,000/100,000 complex 4x4 matrices; host/resident layouts;
CPU-interleaved/verified-warm-call profiles. Each of 24 configurations contains
all four contracts and three rotating, checked CPU/GPU samples. All 96 contract
rows and every exact result match their independent MPFR references.

Host wall scope includes upload, encoding, snapshots, wait and download, with
reusable benchmark buffers and prebuilt pipelines. Resident wall excludes
transfers; internal prototype scratch allocation and completion work remain timed.
The exact bounded fixtures report zero host repairs. All raw sample orders,
accuracy measures, wall/device times, load samples and process memory are retained.

| Bits | Matrices | Layout | Clock | Exact wall (ms) | Composed wall (ms) | Exact / composed |
|---:|---:|---|---|---:|---:|---:|
| 352 | 10000 | host | cpu-interleaved | 7.349 | 6.137 | 1.198 |
| 352 | 10000 | host | verified-warm-call | 6.176 | 5.454 | 1.132 |
| 352 | 10000 | resident | cpu-interleaved | 5.029 | 4.495 | 1.119 |
| 352 | 10000 | resident | verified-warm-call | 3.749 | 2.919 | 1.284 |
| 352 | 100000 | host | cpu-interleaved | 88.204 | 28.464 | 3.099 |
| 352 | 100000 | host | verified-warm-call | 57.698 | 28.406 | 2.031 |
| 352 | 100000 | resident | cpu-interleaved | 43.593 | 21.584 | 2.020 |
| 352 | 100000 | resident | verified-warm-call | 33.203 | 19.730 | 1.683 |
| 384 | 10000 | host | cpu-interleaved | 6.267 | 5.816 | 1.078 |
| 384 | 10000 | host | verified-warm-call | 5.973 | 5.238 | 1.140 |
| 384 | 10000 | resident | cpu-interleaved | 6.079 | 5.338 | 1.139 |
| 384 | 10000 | resident | verified-warm-call | 3.604 | 2.785 | 1.294 |
| 384 | 100000 | host | cpu-interleaved | 44.321 | 30.605 | 1.448 |
| 384 | 100000 | host | verified-warm-call | 42.885 | 31.615 | 1.356 |
| 384 | 100000 | resident | cpu-interleaved | 34.279 | 21.724 | 1.578 |
| 384 | 100000 | resident | verified-warm-call | 98.856 | 21.573 | 4.582 |
| 448 | 10000 | host | cpu-interleaved | 6.782 | 6.433 | 1.054 |
| 448 | 10000 | host | verified-warm-call | 7.433 | 6.635 | 1.120 |
| 448 | 10000 | resident | cpu-interleaved | 5.333 | 4.545 | 1.173 |
| 448 | 10000 | resident | verified-warm-call | 5.067 | 4.413 | 1.148 |
| 448 | 100000 | host | cpu-interleaved | 67.113 | 40.017 | 1.677 |
| 448 | 100000 | host | verified-warm-call | 49.277 | 35.926 | 1.372 |
| 448 | 100000 | resident | cpu-interleaved | 41.959 | 26.026 | 1.612 |
| 448 | 100000 | resident | verified-warm-call | 49.015 | 25.452 | 1.926 |

Clock probes at the same three widths check nine repetitions each of a single
call after 100 ms delay and 32 calls in one batch after at least 200 ms of
accumulated checked device work (54 samples total). These profiles establish
neither controlled GPU frequency nor continuous warmth. Busy-host CPU ratios
(2.96–9.31 for this exact fixture) are observations, not an idle consumer policy.
Whole-process memory includes CPU references, oracle and layouts; it is not
a measure of API scratch or solver memory.

The first measurement collector incorrectly assumed interpolated quartiles.
The benchmark uses the sorted lower-rank sample at `floor(q*(n-1))`; with three
samples p25 is the minimum and p75 is the median. After independently checking
the original successful record, the continuation retained it and the three
clock records and ran only the 23 outstanding comparisons. The rejected collector,
raw records and correction remain available. No successful measurement was rerun.

## Reproduce verification

From the repository root, with Python 3 (no GPU execution):

```sh
python3 benchmarks/experiments/exact_gemm4/verify.py
python3 benchmarks/experiments/exact_gemm4/verify_measurements.py
python3 benchmarks/experiments/exact_gemm4/check_collector.py
```

Add `--live-root "$PWD"` to the two verifiers only in the original matching
build checkout to also check retained binary identities. The frozen generated
shader supports offline verification. [Machine-readable summary](baseline_summary.json)
links every configuration to its raw CSV and metadata. [Qualification details](QUALIFICATION.md)
explain the checks and remaining scope. The original experiment README is
preserved byte-for-byte because correctness metadata hashes it.

Next candidates: reusable scratch with safe in-flight lifetimes, then a compact
device fallback counter to avoid the full mask scan. New shader contexts require
fresh physical-GPU correctness gates. The separate 3x factor target, remaining
tile candidates and idle consumer timing/memory/damping/base work remain open.

## Final main integration checks

After fast-forward integration, the version and independent installed-consumer
checks pass (2/2), as does the optional Baxter smoke enabled in the local build.
Production `include`, `src` and package configuration files remain identical to
v1.3.1. The rebuilt main archive retains its recorded production SHA-256;
installation rewrites the archive index but all seven compiled object members
match byte-for-byte. The experimental targets are not installed. The published
v1.3.1 tag is unchanged. [Final receipts](../../results/section9_exact_gemm4_final_release_checks.json)
record the exact source and validation scopes. Repository renaming remains with
the maintainer; compatible 1.x integration names remain supported.
