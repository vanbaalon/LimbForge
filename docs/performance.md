# Performance and validation

The baseline and optimization rounds were recorded on **October 7, 2026**, on an
**Apple M5 Max** (18 CPU cores, 40 GPU cores), macOS 26.6.2, AppleClang 21, Release
builds, and MPFR 4.2.2. The original Round 1–5 release snapshot is
[final.csv](../benchmarks/results/final.csv); the latest pointwise and reduction
matrices are in [Round 12](../benchmarks/results/round12_reduction_policy.csv) and
[its reduction matrix](../benchmarks/results/round12_reduction_policy_reduction.csv).
The original arithmetic baseline is
[baseline.csv](../benchmarks/results/baseline.csv). See
[environment.json](../benchmarks/results/environment.json) and the per-round
metadata and test logs for provenance.

## Methodology

The default matrix uses 256, 384, and 1024 bits; 256, 4,096, and 65,536 values;
six real operations (including square and square root); three complex
operations; and a 16-step multiplication chain. The main matrix has 180 rows; the separate reduction matrix has 36.
Each case has two warmup calls and nine measured samples. Warmups initialize
pipelines and storage; they do not establish a controlled GPU clock frequency. Inputs use a fixed
seed (`20261007 + bits`); ordinary arithmetic exponents span −5 through +5, and
chain exponents start at zero. Square-root inputs are the absolute values of
the real corpus; other existing input streams remain unchanged. Every final output is compared with independent
MPFR operations. This benchmark covers moderate exponents; the separate test
suite covers error statuses and exponent boundaries.

MPFR input/output arrays and GPU buffers are preallocated. In records through
Round 12, three MPFR scratch variables are initialized and cleared inside each
serial or worker job, including that allocation cost in the CPU timing. The audit
revision reuses thread-local scratch initialized during warmups; new measurements
are recorded separately. Decimal conversion and pipeline compilation are excluded
from these measurements. CPU columns report serial
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

`gpu_s` uses whole command-buffer timestamps. For one dispatch this includes
command-buffer execution overhead; for chains and reductions it includes the
entire set of dispatches and barriers. It is not an isolated arithmetic-instruction
or per-dispatch hardware-counter measurement. CSV wall statistics include the
median, minimum, and empirical lower-order 90th percentile. The serial and
parallel CPU execution order alternates for ordinary arithmetic samples. GPU
samples follow CPU samples; this is a working desktop, with no locked clocks or
isolated thermal conditions. Some unchanged operations vary substantially across
runs. Treat individual cross-round ratios as observations, and repeat a targeted
comparison before choosing a precision-specific kernel.

## Round 1–5 snapshot

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

## Later standalone rounds

Workgroup and layout experiments, square, square root, and reductions have
separate records; historical measurements above remain unchanged. The current
pointwise matrix includes six real primitives. These Round 12 host-array results
use 65,536 values and include transfers, with the same MPFR methodology:

| Bits | Operation | Serial MPFR (ms) | 18-worker MPFR (ms) | GPU wall (ms) | vs serial | vs 18-worker |
|---:|---|---:|---:|---:|---:|---:|
| 256 | square | 2.313 | 0.650 | 0.815 | 2.84× | 0.80× |
| 256 | sqrt | 10.433 | 1.719 | 2.126 | 4.91× | 0.81× |
| 384 | square | 2.525 | 0.569 | 1.021 | 2.47× | 0.56× |
| 384 | sqrt | 11.123 | 1.919 | 4.082 | 2.73× | 0.47× |
| 1024 | square | 7.774 | 1.636 | 1.449 | 5.37× | 1.13× |
| 1024 | sqrt | 25.572 | 5.215 | 6.743 | 3.79× | 0.77× |

Square uses MPFR `mpfr_sqr`, which can be cheaper than multiplication, for the
CPU comparison. The symmetric GPU square at 384 bits wins its targeted resident
device-time comparison, but that does not guarantee a host-array wall-time win.
Square root is an exact restoring baseline; it does not yet consistently beat
multicore MPFR.

### Reduction methodology and results

The reduction benchmark uses 257, 4,096, and 65,537 inputs, with both real and
complex values at 256, 384, and 1024 bits. It interleaves four methods: serial
MPFR, pooled MPFR, resident GPU, and GPU with transfers. All preserve the same
adjacent-pair tree. CPU inputs and intermediate MPFR arrays are preallocated;
the persistent worker pool runs large levels and finishes levels smaller than
`workers*16` serially. Binary conversion and validation follow timing. GPU
input/output buffers are preallocated, but batch encoding includes allocation
of its internal scratch buffers. The resident result excludes upload/download;
the plain result includes both. Every measured output is checked.

65,537 inputs, nine-sample medians, selected policy with transfers:

| Bits | Reduction | Serial MPFR (ms) | 18-worker MPFR (ms) | GPU wall (ms) | vs serial | vs 18-worker |
|---:|---|---:|---:|---:|---:|---:|
| 256 | real | 1.647 | 0.675 | 0.603 | 2.73× | 1.12× |
| 256 | complex | 3.533 | 3.030 | 1.328 | 2.66× | 2.28× |
| 384 | real | 1.763 | 0.781 | 0.768 | 2.30× | 1.02× |
| 384 | complex | 3.785 | 1.555 | 1.752 | 2.16× | 0.89× |
| 1024 | real | 2.307 | 1.206 | 2.104 | 1.10× | 0.57× |
| 1024 | complex | 5.020 | 2.257 | 4.635 | 1.08× | 0.49× |

Cooperation reduces dispatches and scratch storage while retaining the same
rounded tree. The two [interleaved Round 11 comparisons](../benchmarks/results/round11_interleaved_reduction.csv)
and [repeat](../benchmarks/results/round11_interleaved_reduction_recheck.csv)
compare both implementations on resident data, checking every dispatch.
At 65,537 inputs, real wall gains were about 1.46–1.58× at 384 bits and
1.47–1.51× at 1024 bits; 256/384-bit complex gains were about 1.21–1.46×.
The 1024-bit complex cooperative kernel was slower in both sweeps
(0.92–0.95× wall ratio), so the default retains the global-memory tree above
384-bit complex precision. Inputs of size 0–2 also use the simple global path.
`EngineOptions{0,false}` disables cooperation for comparison.

The selected policy passes the full arithmetic/reduction matrices and a
[final interleaved comparison](../benchmarks/results/round12_interleaved_reduction.csv).
These records show an implementation improvement, not a universal advantage
over multicore MPFR. Wider device validation and further reduction tuning remain
necessary.

## Benchmark/documentation audit

The audit started from GitHub revision `f5663e4` on October 7, 2026. It verified
173 published numeric cells against their committed source CSVs, accepted-record
row counts and test logs, API descriptions, and local Markdown links. The
[record validator](../benchmarks/audit_records.py) runs in GitHub Actions to catch
future numeric or link drift.

The arithmetic core and GPU kernels are unchanged by the audit. The main harness
now retains thread-local MPFR scratch across warmed jobs, and the round runner
passes `--workers` and `--repeats` to reductions and requires CPU/GPU suites.
A fresh [180-row pointwise/chain matrix](../benchmarks/results/audit_20261007.csv)
and [36-row reduction matrix](../benchmarks/results/audit_20261007_reduction.csv)
passed independent MPFR comparisons, all four suites, and
[Metal validation](../benchmarks/results/audit_20261007_metal_tests.txt).
The original headline and Round 12 records remain separate historical snapshots.
Differences between runs cannot be attributed to scratch reuse alone: desktop
load, CPU/GPU scheduling, and clock state were not controlled.

The [option-propagation smoke run](../benchmarks/results/audit_options_20261007_metadata.json)
checks three samples and one CPU worker in both matrices. Its timings are not
used as performance evidence.

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

The round runner also runs `reduction_limbforge` and preserves its separate CSV.
`reduction_limbforge --compare` interleaves the selected and global-memory
reduction policies; `--global-tree` benchmarks the latter against CPU MPFR.
`tune_limbforge`, `layout_limbforge`, and `square_limbforge` provide targeted
comparisons. Use `--threads N` to override the pointwise workgroup size.

`--workers N` and `--repeats N` apply to both benchmarks when invoked through
the audited round runner. Earlier runners applied these options only to the main
arithmetic/chain matrix; reduction rows used 9 samples and hardware concurrency.
`--bits`, `--count`, and `--operation` filter the main matrix; the reduction matrix
retains its fixed three-precision/three-size coverage.

Use `--workers N` to control the MPFR pool. The default is hardware concurrency.
`--quick` uses 4,096 values at all three benchmark precisions. `--bits` selects
256, 384, or 1024; `--operation mul_chain` selects the two chain interfaces.
The main benchmark validates the final output of each host/resident case;
reduction and targeted A/B harnesses validate every measured dispatch. The
benchmark exits with failure on a detected arithmetic mismatch. The audited
round runner requires all four CPU/GPU suites to be registered before benchmarking;
a CPU-only or test-disabled build cannot be reported as a complete round.

## Test coverage and limitations

Real CPU and GPU arithmetic is checked against MPFR in 761,856 cases across every
multiple of 32 bits from 64 to 1024. Fixtures include cancellation, limb and ulp
alignment, halfway rounding, status propagation, and products rounding across
both exponent boundaries, square-root domains, negative odd exponents,
and near-midpoint square-root inputs. One million exact scalar quotient/remainder checks
exercise the reciprocal primitive.

Complex and generic recurrence tests cover 128, 384, and 1024 bits. An additional
1024-bit suite tests 65,536 dense complex inputs for all three complex operations,
with three GPU repeats. Resident tests cover in-place chains, dependent barriers,
wrong formats and counts, bounds, foreign engines, pending mapping rejection,
empty batches, repeated waits, and resource lifetimes. Real and complex
reductions cover all 31 precisions, zero/one/odd counts, status masks,
intermediate overflow, fixed pairing under cancellation, dependent operations,
alternate groups, and scratch storage surviving the original engine.

GitHub Actions runs CPU/MPFR checks and package installation. Physical GPU
validation runs locally. The final GPU suites also pass with
`MTL_SHADER_VALIDATION=1`, and the installed resident API passes a separate
consumer check without MPFR; the logs are committed under `benchmarks/results`. This is substantial empirical coverage, not an exhaustive
proof over every representable input or validation of other GPU models.

The [optimization log](optimizations.md) records accepted changes and failed
experiments. Files named `round3_rejected*` contain an incomplete failed benchmark;
`round5_*` trials stop after GPU test failures. They are not accepted speedup
records. Historical `m5_max_arithmetic.csv` and the optional application comparison
predate this standalone benchmark workflow and are excluded from the tables above.
