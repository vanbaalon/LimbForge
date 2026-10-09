# Native per-product Gauss GEMM prototype

Base: WolfNum 1.3.1/main `78f7da6`. Isolated, unreleased P1.3 candidate; no default
or existing public request layout changes. Explicit `gemm_gauss` host/resident calls
and separate `prewarm_gauss` compile a native shader. Accepted public additions
would need a minor release; this prototype is not installed by production main.

The shader performs three real products once per inner term and complex output,
with rounded input sums and per-term real/imaginary reconstruction. The sequence is
specified in `docs/numerics.md`. This is distinct from the existing benchmark's
three complete real GEMMs. A rolled three-product loop keeps one multiplication
call site; padded private significands preserve the production compiler workaround.

`test_limbforge_section9_gauss` independently replays MPFR scalar operations, and
covers 4x4 and 9x17x5, broadcast A/B, strides/output padding, accumulated final
negation, cancellation, statuses and empty dots. Dense mode uses 4097/257 matrices
and repeats three times; wide exponents are included. Host and resident paths are
checked at small shapes. Fused requests are rejected explicitly.

Builds of the native audit and five-path benchmark pass. MPFR fixture preparation
completes at all 31 widths, and the benchmark CPU exact-dot oracle checks pass at
352/1024. Physical-GPU correctness evidence is recorded below; performance and
minor-release acceptance remain separate requirements.

The 352/1024 focused audit and five-contract benchmark checks now pass normally
and under shader validation (8 completed checks, source/binary identities retained).
The all31 dense/wide resident sweeps now pass normally and under Metal validation,
as do all31 small host/resident checks. Full builds and 26 normal/19 Metal
regression suites pass. Terminal raw logs and frozen source/archive/binary hashes
are retained in `section9_p13_native_gauss_full_gate_metadata.json` and
`section9_p13_native_gauss_host31_gate_metadata.json`; they were independently
verified after completion. Dense mode skips the host wrapper; the separate small
checks cover it. The completed performance comparison and decision follow below;
this correctness result alone establishes no numerical or performance advantage.

The benchmark accepts `--paths composed,fused,gauss,native` to rotate a practical
subset in one process. Its default still selects all five paths, and `--path`
still selects one. A 100,000-matrix 4x4 comparison can omit the per-matrix exact
embedding driver, whose separate Linalg calls do not constitute a batched exact
backend. All five immutable MPFR references and the independent exact oracle are
still prepared. Repeated selectors, duplicate/empty/unknown names are rejected.

After this benchmark-only change, 16 checks pass at 352-bit host and 1024-bit
resident shapes, normally and under Metal validation: the composed/matrix/native
subset, exact/fused subset, default all paths and legacy native selection. Parser
guards and both CPU exact-oracle checks pass. Numerical source, archive and audit
binary hashes remain identical to the completed full/all-width gates. The older
benchmark snapshot remains identified by those gates; the updated benchmark has
its own focused manifest `section9_p13_native_gauss_selector_gate_metadata.json`.


## Measured decision: retain the prototype, do not promote it

All 22 completed profiles check every initial, warm and timed CPU/GPU output
against its immutable MPFR sequence. The grid covers 352/384/448 bits, the Fourier
shape and 10,000/100,000 4x4 batches, with host and resident execution. Four extra
352-bit profiles use one verified untimed warm call before each sample. Fourier
runs have five rotating samples; small batches have four, covering each path order.
The harness reports observed quantiles: for even counts its 50th percentile is the
lower middle observation. Raw CPU/GPU samples, device time, ranges, accuracy,
whole-process RSS, load and source/binary/gate hashes are retained.

| Fourier bits / residency | Composed GPU wall (s) | Native GPU wall (s) | Native / composed |
|---|---:|---:|---:|
| 352 / host | 0.102746 | 0.214644 | 2.089 |
| 352 / resident | 0.101639 | 0.213088 | 2.097 |
| 384 / host | 0.114369 | 0.215417 | 1.884 |
| 384 / resident | 0.111972 | 0.214936 | 1.920 |
| 448 / host | 0.138670 | 0.226509 | 1.633 |
| 448 / resident | 0.134561 | 0.223887 | 1.664 |

Every profile's native wall-time quantile loses to composed: 1.25–2.44 times as
long across the full grid. Fourier ratios are 1.63–2.10, with disjoint observed
ranges; some small-batch ranges overlap. Reducing the number of real products
therefore does not establish an optimization on this GPU. Register pressure or
instruction scheduling has not been isolated as the cause. Global error units
are reported per contract in the CSV; the native sequence has no universal
accuracy advantage and cannot replace the composed default.

These are loaded-host library observations. Numeric-array transfers are included
for host calls and excluded for resident calls; bridge conversion and compilation
are excluded. CPU ratios are not idle-consumer acceptance. Extra verified warm
calls do not establish controlled cold or continuously warm clocks.

No installed API, default or new minor release is accepted from this candidate.
Further native Gauss scheduling requires a new isolated implementation and fresh
correctness gates before timings. Exact batched small-matrix products, register
blocking and the remaining P1.3/P2/P3 requirements stay open.

The measurement grid and independent check are in
`section9_p13_native_gauss_measurement_grid_metadata.json` and
`section9_p13_native_gauss_measurement_check.json`. Verify the archived raw samples
and numerical-gate provenance without executing GPU kernels:

```sh
python3 benchmarks/experiments/check_section9_gauss_measurements.py
```
