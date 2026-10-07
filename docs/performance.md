# Performance and validation

## Arithmetic benchmark

The committed [CSV](../benchmarks/m5_max_arithmetic.csv) was recorded on
October 7, 2026, on an Apple M5 Max, with Release builds and AppleClang 21.
It contains addition, multiplication, and division at 256, 384, and 1024 bits,
with 65,536 independent values per dispatch.

```sh
./build/benchmark_limbforge 65536
```

Each operation is warmed once. The benchmark then takes five samples and reports
medians. Inputs and MPFR variables are prepared outside the timed region. MPFR
uses one CPU thread and reuses its initialized variables. GPU pipelines compile
outside the timed region; buffers are reused. GPU wall time includes input/output
copies, encoding, submission, and synchronization. Device timestamps report GPU
time separately. Results are checked against MPFR after the measurements.

| Bits | Operation | MPFR CPU (ms) | GPU device (ms) | GPU wall (ms) | CPU / GPU wall |
|---:|---|---:|---:|---:|---:|
| 256 | add | 1.631 | 0.110 | 0.797 | 2.05× |
| 256 | multiply | 2.080 | 0.118 | 0.805 | 2.58× |
| 256 | divide | 6.230 | 0.683 | 1.321 | 4.72× |
| 384 | add | 1.791 | 0.160 | 0.928 | 1.93× |
| 384 | multiply | 2.930 | 0.454 | 1.242 | 2.36× |
| 384 | divide | 7.977 | 1.301 | 2.070 | 3.85× |
| 1024 | add | 2.250 | 0.459 | 1.487 | 1.51× |
| 1024 | multiply | 22.295 | 0.486 | 3.001 | 7.43× |
| 1024 | divide | 43.747 | 2.482 | 6.540 | 6.69× |

These are measurements of this batch on this device. A multicore CPU baseline,
other batch sizes, and other GPUs may give different comparisons. In particular,
single-scalar dispatches are not the intended use case.

## Baxter propagation experiment

The [recorded experiment](../benchmarks/m5_max_baxter.txt) evaluates 16 complex
points, with four Q-functions in each of two propagation directions: 128
trajectories, each taking 600 shifts. GPU arithmetic uses 384 significand bits;
the reference and CPU preparation use 110 decimal digits. Asymptotic series
construction is common setup and excluded from both measurements.

```sh
./build/baxter_batch 16 600
```

| Measurement | Seconds |
|---|---:|
| CPU seed/coefficient preparation and packing | 0.8461 |
| GPU propagation, device time | 0.0994 |
| GPU propagation, wall time | 0.0998 |
| Preparation plus GPU wall time | 0.9459 |
| Existing MPFR solver's Q evaluations | 1.3275 |

The largest error scaled by `1 + abs(Q_reference)` is **8.97 × 10⁻¹⁰⁶**.
These application timings are a single warmed measurement, not five-run medians.
Preparation plus propagation is about 1.40× faster than this serial reference
in the recorded experiment. GPU-only ratios omit a substantial preparation cost.

The weights rearrange the Baxter recurrence algebraically, changing rounding
order relative to the solver. Validation checks the resulting Q-values against
an independent MPFR computation, with an 80-digit scaled-error threshold.
It does not certify asymptotic boundary conditions for arbitrary inputs or
validate GPU quantization, Newton steps, connection matrices, or coupling/gluing.
The production BSolver4D executable remains MPFR-based.

## Implemented optimizations

- Compile-time limb counts and operation specialization through Metal function constants.
- Normalized base-2^32 arithmetic with full-width integer intermediates.
- Quadratic-time Knuth division, avoiding a bit-at-a-time quotient loop.
- No device heap allocation; local working storage has fixed capacity.
- Exact alignment before the final nearest-even rounding of addition/subtraction.
- Reused host-visible Metal buffers and cached pipelines.
- One dispatch per recurrence batch; four persistent states per trajectory.
- Shared coefficients for groups of trajectories, such as the four Baxter Q-functions.

The initial implementation has not been exhaustively tuned for every precision
or GPU. Better data layouts, CPU multicore comparisons, larger recurrence batches,
and a device-resident public API are useful next experiments.

## Test coverage

`ctest` runs CPU checks and, when enabled, GPU checks. Real arithmetic is compared
with independent MPFR operations in 81,920 cases across 64, 128, 256, 384, and
1024 bits. Inputs include zero, invalid status, cancellation, alignment changes,
halfway rounding, and exponent boundaries. Complex results are compared with
MPFR real operations in the documented order. Recurrence tests compare complete
trajectories, including shared coefficients and zero-step behavior.

GitHub Actions builds the macOS library, runs CPU/MPFR checks, and installs the
package. It does not claim GPU execution coverage. Physical-device validation
was performed locally on the M5 Max.
