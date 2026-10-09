# Section 9 P1.3: complex GEMM contract comparison

Original base `70ec257` / 1.1.0, now rebased on `bffa58b` / WolfNum 1.3.0. No library arithmetic, API or default change.
`section9_gemm_limbforge` is an opt-in benchmark excluded from routine builds/CTest.
Checked 352-bit Fourier-shape and small-matrix comparisons are now recorded.
No numerical library source or default changes are accepted by these measurements.

The four paths have separate MPFR references:

| Path | Rounding contract |
|---|---|
| Complex composed | Four rounded real products per complex product, rounded real difference/imaginary sum, then ascending rounded accumulation |
| Complex fused | Each real/imaginary component rounds once from the two exact signed products plus the previous rounded component, at every ascending inner index |
| Exact real embedding | Rows `[ar,-ai]` and `[ai,ar]` times stacked `[br;bi]`; one exact real dot over `2k` terms per component, then one rounding |
| Gauss three real composed | Round `ar+ai` and `br+bi`; independently accumulate `P=ar*br`, `Q=ai*bi`, `S=(ar+ai)*(br+bi)` using rounded real multiply/add; output `RN(P-Q)` and `RN(RN(S-P)-Q)` |

Gauss here is a matrix-level three-real-GEMM experiment. It is a different sequence
from a three-product complex multiply followed by complex accumulation. It can lose
accuracy through cancellation and must not silently replace an existing complex GEMM.
The exact embedding uses one Linalg call per matrix; it is not a new batched exact API.
Its outputs are final only after wait, including any host exact fallback, which is timed.

MPFR storage, pointer tables, worker scratch and output buffers are prepared once.
The exact timed CPU path uses `mpfr_dot` at B bits, checked against an independent
2B+64-bit sum of exact products. Fixture exponents lie in [-4,4] and k<=4096, so this
precision retains the exact sum. All 31 CPU widths pass that check. Gauss input sums
and reconstruction are inside both CPU/GPU timings. Every initial, optional warm-up
and timed GPU result is checked against its own MPFR sequence; timed CPU outputs
are rechecked outside timing. Accuracy includes the count of complex entries that
differ from the once-rounded result and maximum component error normalized by the
maximum exact component magnitude, in units of 2^-B.

Host staging includes split/embedding preparation, uploads, downloads and output
reconstruction using reused buffers. MPFR/MPC bridge conversion remains excluded
and belongs in the consumer measurement. Resident mode starts with path-specific
resident layouts and excludes transfers and host output reconstruction; Gauss
device sums/reconstruction remain timed. It does not promise an interleaved complex
input/output interface for either real-path experiment.

CPU and GPU path orders rotate over repetitions. CSV retains medians, quartiles and
range; stderr preserves raw sample order, wall and device times. `--warm` performs
one verified untimed call before each sample, without claiming continuously warm clocks.
The default is CPU-interleaved rather than a controlled cold-clock measurement.

Build with `cmake --build build --target section9_gemm_limbforge -j4`.
352-bit 2x(9x17x7) host-staging and 1024-bit 3x(4x4x4) resident check-only cases pass,
both normally and under shader validation. Logs: `section9_p13_gemm_*_reference.txt`
and `...validation.txt`; CPU-only all-width log: `section9_p13_gemm_cpu_reference.txt`.
These are focused checks, not all-width certification of a new primitive.

The default shape is the Fourier 66x130 times 130x3800. For large 4x4 batches, select
`--paths composed,fused,gauss` to rotate these paths within one process, or use the
legacy `--path composed`, `fused` or `gauss` for a single path. Only selected GPU
layouts are allocated. All four CPU references and the independent exact oracle
are still prepared and checked. The subset omits the per-matrix Linalg encoding
cost and extra buffers of the exact embedding. A fair large-batch exact comparison
still needs measurement and may motivate an exact batched API. The Fourier and resident 1e4/1e5 profiles below are measured. Other widths,
host 4x4 batches, register blocking/TK variants and performance acceptance remain
pending; the historical 7.9x ratio is not a current baseline.

The current harness explicitly matches Linalg host-worker count to the MPFR
worker count. Clock labels are `cpu-interleaved` and `verified-warm-call`; neither
proves controlled cold or continuously warm clocks. Four rebuilt focused physical-GPU checks pass for this updated 1.3.0-linked
harness (352-bit host and 1024-bit resident, normal and shader validation).
`section9_p1_gemm_wolfnum_fixture_metadata.json` records commands and identities;
the later 1.3.1 measurements below use separately recorded rebuilt fixtures.

## Checked 1.3.1 measurements

All outputs pass their own independent MPFR sequence, including every timed CPU/GPU
call. Clock mode is CPU-interleaved; bridge conversion is excluded. The Fourier
fixture uses bounded random exponents at the requested shape, not saved solver data.

| Fourier 66x130 times 130x3800 | Host GPU wall (s) | Resident GPU wall (s) | Global error units |
|---|---:|---:|---:|
| Composed complex | 0.103959 | 0.104439 | 7.720741 |
| Fused complex | 0.212559 | 0.216581 | 8.007584 |
| Exact real embedding | 0.015914 | 0.010170 | 0.548310 |
| Three-real-GEMM Gauss | 0.080504 | 0.078172 | 11.772728 |

Exact embedding has the lowest measured wall time and error in this fixture; the
once-rounded contract remains distinct from either sequential path. Gauss is faster
than composed here but has larger error and cannot silently replace its sequence.

| Resident 4x4 batches | Composed (s) | Fused (s) | Matrix-level Gauss (s) |
|---|---:|---:|---:|
| 10,000 | 0.004640 | 0.015358 | 0.006108 |
| 100,000 | 0.039772 | 0.046511 | 0.030021 |

Gauss is not uniformly faster. Its global error units are 2.592/3.083 for these two
batches versus composed 1.529/1.585 and fused 1.222/1.134. These are global infinity
error comparisons in units of 2^-B, not per-entry relative errors or digits correct.
Every profile has busy-host load samples; CPU ratios are library observations, not
idle consumer acceptance. All sample ranges and identities are retained in
`section9_p13_fourier_*` and `section9_p13_small4_*` raw triples. Large-batch exact
encoding remains outside this small-matrix comparison: one Linalg call per matrix
is not a batched exact implementation. Other widths, transfer profiles, native
three-product multiplication and tile experiments remain follow-up work.

## Offline calibration

The eight completed profiles export 14 exact lookup keys through
`section9_gemm_break_even()` in `section9_p13_gemm_break_even_measured.hpp`.
Its JSON retains binary/archive, contract, clock and busy-host load identity.
The generated C++ consumer checks every measured key and unknown width/profile/shape
queries; terminal, raw-hash, idle and overwrite guards pass. This installs no default
and does not establish consumer or idle-host acceptance. See [calibration](../../docs/calibration.md).

## Production benchmark subset gate

The benchmark-only change at `dd1c236` uses the unchanged published 1.3.1
archive. Eight malformed/duplicate/multiple selectors are rejected; all 31 CPU
reference/oracle widths pass. Sixteen focused GPU checks cover composed/Gauss
and exact/fused subsets, the four-path default and legacy single-path Gauss at
352-bit host and 1024-bit resident layouts, normally and under Metal validation.
This verifies selection and allocation behavior; it introduces no new numerical
kernel or all-width GPU certification. Historical timing records remain unchanged.

Raw commands, frozen source/binary/archive identities and logs are in
[`section9_production_gemm_subset_gate`](../results/section9_production_gemm_subset_gate).
Run `python3 benchmarks/verify_gemm_subset_gate.py` to check coverage and hashes;
`--live-archive build/liblimbforge.a` additionally checks the local archive identity.
The library remains at 1.3.1; this tool change is unreleased on main.
