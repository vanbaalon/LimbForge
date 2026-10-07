# Performance and validation

The baseline and optimization rounds were recorded on **October 7, 2026**, on an
**Apple M5 Max** (18 CPU cores, 40 GPU cores), macOS 26.6.2, AppleClang 21, Release
builds, and MPFR 4.2.2. The current measurements are in
[final.csv](../benchmarks/results/final.csv); the original arithmetic baseline is
[baseline.csv](../benchmarks/results/baseline.csv). See
[environment.json](../benchmarks/results/environment.json) and the per-round
metadata and test logs for provenance.

## Methodology

The default matrix uses 256, 384, and 1024 bits; 256, 4,096, and 65,536 values;
four real operations; three complex operations; and a 16-step multiplication
chain. Each case has two warmups and nine measured samples. Inputs use a fixed
seed (`20261007 + bits`); ordinary arithmetic exponents span −5 through +5, and
chain exponents start at zero. Every final output is compared with independent
MPFR operations. This benchmark covers moderate exponents; the separate test
suite covers error statuses and exponent boundaries.

MPFR arrays and GPU buffers are preallocated. Decimal conversion and pipeline
compilation are excluded from warmed measurements. CPU columns report serial
MPFR and a persistent **18-worker MPFR pool**, including pool dispatch costs.
Both preserve the same real rounding steps as the GPU. Complex arithmetic does
not use a different single-round MPC contract.

The columns distinguish three execution costs:

- Host-array operations: GPU wall time includes input/output copies, encoding,
  submission, and synchronization.
- Standalone `_resident` operations: inputs are uploaded before timing; wall
  time covers batch construction, encoding, submission, and waiting. These
  columns exclude upload/download costs and describe reuse of existing data.
- `mul_chain_resident`: wall time includes one upload of each input, all sixteen
  dependent dispatches, waiting, and the final download. `mul_chain_host`
  includes sixteen host-array calls with repeated copies and waits. CPU chain
  times exclude resetting the preallocated output to its initial value.

Device timestamps measure execution separately. CSV wall statistics include the
median, minimum, and empirical lower-order 90th percentile. The serial and
parallel CPU execution order alternates for ordinary arithmetic samples. GPU
samples follow CPU samples; this is a working desktop, with no locked clocks or
isolated thermal conditions. Some unchanged operations vary substantially across
runs. Treat individual cross-round ratios as observations, and repeat a targeted
comparison before choosing a precision-specific kernel.

## Representative current measurements

65,536 values, milliseconds, medians of nine warmed samples. Host-array GPU wall
includes transfers; resident-only operation times are available in the raw CSV.

| Bits | Operation | Serial MPFR | 18-worker MPFR | GPU device | GPU wall |
|---:|---|---:|---:|---:|---:|
| 256 | add | 1.834 | 0.437 | 0.092 | 0.800 |
| 256 | mul | 2.242 | 0.472 | 0.119 | 0.815 |
| 256 | div | 6.560 | 1.168 | 0.278 | 0.681 |
| 384 | add | 1.856 | 0.561 | 0.118 | 0.806 |
| 384 | mul | 2.858 | 0.583 | 0.452 | 1.213 |
| 384 | div | 8.903 | 1.804 | 0.689 | 1.345 |
| 1024 | add | 2.469 | 0.538 | 0.216 | 1.212 |
| 1024 | mul | 18.387 | 1.902 | 0.482 | 1.622 |
| 1024 | div | 40.831 | 3.604 | 0.787 | 2.088 |

Real addition and small-precision multiplication remain faster on multicore MPFR
in these host-array measurements. GPU division benefits from exact reciprocal
quotient estimates. GPU speedups depend on both arithmetic cost and data reuse;
serial CPU ratios alone do not describe the multicore comparison.

## Speedup against CPU MPFR

These ratios compare CPU elapsed time with **GPU wall time**, including transfers,
encoding, submission, and waiting. They use the same `final.csv` measurements
above: 65,536 values and nine warmed samples per case. The chain uses resident
execution with one upload per input and one final download; individual operations
use the synchronous host-array interface. Each ratio is CPU median / GPU wall
median. A value below 1 means the CPU is faster.

| Bits | Workload | vs serial MPFR | vs 18-worker MPFR |
|---:|---|---:|---:|
| 256 | Real addition | 2.29× | 0.55× |
| 256 | Real multiplication | 2.75× | 0.58× |
| 256 | Real division | 9.63× | 1.72× |
| 256 | Complex division | 30.76× | 3.65× |
| 256 | 16-step multiplication chain (resident) | 38.36× | 4.34× |
| 384 | Real addition | 2.30× | 0.70× |
| 384 | Real multiplication | 2.36× | 0.48× |
| 384 | Real division | 6.62× | 1.34× |
| 384 | Complex division | 36.30× | 2.35× |
| 384 | 16-step multiplication chain (resident) | 24.84× | 2.09× |
| 1024 | Real addition | 2.04× | 0.44× |
| 1024 | Real multiplication | 11.33× | 1.17× |
| 1024 | Real division | 19.55× | 1.73× |
| 1024 | Complex division | 29.09× | 2.37× |
| 1024 | 16-step multiplication chain (resident) | 82.31× | 6.70× |

The README highlights the 1024-bit cases, where the recorded resident chain
achieves 82.3× the serial MPFR throughput and 6.7× the 18-worker MPFR throughput.
These are measured workload-specific comparisons on one device. The serial,
multicore, and GPU implementations preserve rounding after every real operation;
complex comparisons use the documented composition of real primitives.

## Sixteen dependent multiplications

Both GPU wall columns include transfers, submission, encoding, and waiting.
Buffers are allocated outside the timed region for both interfaces.

| Bits | 18-worker MPFR (ms) | Repeated GPU host calls (ms) | Resident GPU chain (ms) | Host / resident |
|---:|---:|---:|---:|---:|
| 256 | 8.277 | 9.583 | 1.909 | 5.02× |
| 384 | 6.661 | 17.060 | 3.187 | 5.35× |
| 1024 | 26.384 | 27.910 | 3.937 | 7.09× |

## Counterbalanced baseline comparison

After the final matrix, the frozen original executable and the final executable
were run in baseline–final–final–baseline order, each using 65,536 values and nine
samples. All four runs passed MPFR comparisons. The table uses the arithmetic
mean of each version's two run medians; ratios divide those means. Raw records
are `baseline_recheck_a/b.csv` and `final_recheck_a/b.csv`; see
[counterbalanced metadata](../benchmarks/results/counterbalanced_metadata.json)
and the [complete comparison](../benchmarks/results/counterbalanced_comparison.csv).

| Bits | Operation | Baseline wall (ms) | Final wall (ms) | Wall speedup | Device speedup |
|---:|---|---:|---:|---:|---:|
| 256 | divide | 1.224 | 1.024 | 1.20× | 2.43× |
| 384 | divide | 1.934 | 1.009 | 1.92× | 3.06× |
| 1024 | divide | 4.081 | 1.946 | 2.10× | 3.15× |

The unchanged schoolbook multiplier also varies across these runs, so changes in
its timing should not be attributed to a new multiplication algorithm. Some
addition wall times regress even while device times improve: transfers and host
scheduling remain a substantial fraction of latency. The comparison supports a
division improvement on this device, rather than a universal speedup claim.

## Reproduction

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
./build/benchmark_limbforge > results.csv
./build/benchmark_limbforge --bits 384 --count 65536 --operation div --repeats 9
python3 benchmarks/run_round.py my-round
python3 benchmarks/compare.py benchmarks/results/baseline.csv results.csv
```

Use `--workers N` to control the MPFR pool. The default is hardware concurrency.
`--quick` uses 4,096 values at all three benchmark precisions. `--bits` selects
256, 384, or 1024; `--operation mul_chain` selects the two chain interfaces.
The benchmark exits with failure on a detected arithmetic mismatch.

## Test coverage and limitations

Real CPU and GPU arithmetic is checked against MPFR in 507,904 cases across every
multiple of 32 bits from 64 to 1024. Fixtures include cancellation, limb and ulp
alignment, halfway rounding, status propagation, and products rounding across
both exponent boundaries. One million exact scalar quotient/remainder checks
exercise the reciprocal primitive.

Complex and generic recurrence tests cover 128, 384, and 1024 bits. An additional
1024-bit suite tests 65,536 dense complex inputs for all three complex operations,
with three GPU repeats. Resident tests cover in-place chains, dependent barriers,
wrong formats and counts, bounds, foreign engines, pending mapping rejection,
empty batches, repeated waits, and resource lifetimes.

GitHub Actions runs CPU/MPFR checks and package installation. Physical GPU
validation runs locally. The final GPU suites also pass with
`MTL_SHADER_VALIDATION=1`, and the installed resident API passes a separate
consumer check without MPFR; both logs are committed under `benchmarks/results`. This is substantial empirical coverage, not an exhaustive
proof over every representable input or validation of other GPU models.

The [optimization log](optimizations.md) records accepted changes and failed
experiments. Files named `round3_rejected*` contain an incomplete failed benchmark;
`round5_*` trials stop after GPU test failures. They are not accepted speedup
records. Historical `m5_max_arithmetic.csv` and the optional application comparison
predate this standalone benchmark workflow and are excluded from the tables above.
