# Section 9 P1.1: compact power storage

Base: corrected `72637b4` / 1.0.1. Accepted addition: 1.1.0, Apple M5 Max.

An unscaled power checkpoint per `(batch,step)` is seeded once with the original powi.
Eight-row panels advance that checkpoint in the original order, multiply E afterwards,
and use the existing composed/fused GEMM to write output rows directly. No final extra
multiplication, prefix replay or panel-boundary reseeding is introduced. Compact storage
is explicit; full-table storage remains the default. The first draft omitted the real
GEMM output-row offset and failed at 352 bits; its log is retained and the offset is fixed.

Auxiliary powers (inputs/output/host W/driver allocation excluded):

| Shape, complex | Full table bytes | Compact bytes | Reduction |
|---|---:|---:|---:|
| 520×60×160, 352 bits | 559,104,000 | 83,865,600 | 6.67× |
| 900×101×250, 352 bits | 2,545,200,000 | 226,800,000 | 11.22× |
| 900×101×250, 448 bits | 3,090,600,000 | 275,400,000 | 11.22× |

## Timings

Seven repeats, rotating table/panel order; 18-worker MPFR. Compilation, reference checking
and initial warm-up are excluded. Host wall includes transfers; resident wall excludes upload
and download. Every initial and timed output is checked independently. The 0.25 s GPU warm
loops are not individually checked. Raw CSV retains CPU/device timings, shape, version, clock
mode and scratch. Load averages before 7.14/10.09/10.99 and after 28.90/63.89/67.94: these are **busy-host library comparisons**,
not idle-host solver or CPU speedup claims. Warm-clock central cases add 9–24% latency;
most other cases add latency too. The 448-bit small host/table row is an outlier and is not
used to select a faster default. No new headline speedup is claimed.

| Bits | Count×rows×steps, ncols=16 | Mode | Table wall ms | Compact wall ms | Compact / table |
|---:|---|---|---:|---:|---:|
| 352 | 400×41×100 | host / cpu-interleaved | 116.5 | 116.2 | 0.997 |
| 352 | 400×41×100 | resident / cpu-interleaved | 103.6 | 106.7 | 1.030 |
| 352 | 520×60×160 | host / cpu-interleaved | 327.3 | 370.6 | 1.132 |
| 352 | 520×60×160 | host / warm | 357.1 | 442.7 | 1.240 |
| 352 | 520×60×160 | resident / cpu-interleaved | 312.3 | 351.6 | 1.126 |
| 352 | 520×60×160 | resident / warm | 340.2 | 401.0 | 1.179 |
| 352 | 900×101×250 | host / cpu-interleaved | 1662.4 | 1768.7 | 1.064 |
| 352 | 900×101×250 | resident / cpu-interleaved | 1762.6 | 1918.2 | 1.088 |
| 384 | 400×41×100 | host / cpu-interleaved | 123.2 | 124.6 | 1.011 |
| 384 | 400×41×100 | resident / cpu-interleaved | 116.7 | 119.1 | 1.020 |
| 384 | 520×60×160 | host / cpu-interleaved | 366.7 | 427.2 | 1.165 |
| 384 | 520×60×160 | host / warm | 377.4 | 461.9 | 1.224 |
| 384 | 520×60×160 | resident / cpu-interleaved | 355.7 | 403.3 | 1.134 |
| 384 | 520×60×160 | resident / warm | 406.4 | 442.9 | 1.090 |
| 384 | 900×101×250 | host / cpu-interleaved | 2167.5 | 2389.2 | 1.102 |
| 384 | 900×101×250 | resident / cpu-interleaved | 2278.8 | 2238.0 | 0.982 |
| 448 | 400×41×100 | host / cpu-interleaved | 384.5 | 161.0 | 0.419 |
| 448 | 400×41×100 | resident / cpu-interleaved | 153.8 | 143.0 | 0.929 |
| 448 | 520×60×160 | host / cpu-interleaved | 430.0 | 474.5 | 1.103 |
| 448 | 520×60×160 | host / warm | 466.5 | 553.9 | 1.187 |
| 448 | 520×60×160 | resident / cpu-interleaved | 408.9 | 446.3 | 1.091 |
| 448 | 520×60×160 | resident / warm | 429.9 | 509.5 | 1.185 |
| 448 | 900×101×250 | host / cpu-interleaved | 2299.5 | 2473.2 | 1.076 |
| 448 | 900×101×250 | resident / cpu-interleaved | 2291.7 | 2491.5 | 1.087 |

The panel benchmark source used a private environment selector; its exact source snapshot
is [retained](section9_panel_power_candidate.patch). The production API removes that selector
and chooses storage per call. Shader sources are unchanged by this host/API selection change. The final source also corrects one stale comment; reversing that comment alone reproduces all validated candidate hashes exactly. Final rebuilt package/API smoke and shader-validation smoke pass.
Reproduce with `section9_limbforge --operation power --bits 352 --count 520 --m 60 --n 16
--k 160 --workers 18 --repeats 7 --compare-power` (append `--resident` or `--gpu-warm 0.25`).
Results and platform metadata: `benchmarks/results/section9_p1_power_*`.

## Correctness gates

The numerical candidate passed 26/26 CTest suites, 19/19 shader-validation GPU suites,
all 31 complete dense widths in both modes, expanded three-batch negative powers including
fully valid batches, fused/composed accumulation, eight-row boundaries and real 4×4 tails.
The final API's all-width power sweeps pass normally and under validation; focused smoke
checks include explicit host/resident compact overloads, empty reductions and invalid storage.
Generated-source hashes identify the numerical candidate independently of metadata versions.
Raw logs: `benchmarks/results/section9_p1_{panel,table,compact_api}_*`.

The private arithmetic clone uses rolled multiplication inside cmul/cdiv workspaces to avoid
pathological compilation, with logical precision unchanged. This is local implementation
storage; packed public types and `PowerMoments` layout remain unchanged.
