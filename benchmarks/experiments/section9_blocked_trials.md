# Section 9 P1.2: blocked damping-trial candidate

Original arithmetic gates began on `e91a215` (1.1.0); the candidate is rebased
on `02bca53` (1.2.0), branch `round-section9-blocked-trials`, commit `3e30d47`.
Production default remains the original scalar-column path. Private experiment selector:
`LIMBFORGE_CHOLESKY_PATH=scalar|panel|block`.

The block path uses three dispatches per panel: 32-thread per-trial panel factorization,
outside-row triangular work, and a trailing update replaying each rounded multiply/subtract
in ascending pivot order. Panel width is 16 through 896 bits and 8 above it; packed shared
panels stay below 32 KB. Barriers are unconditional for tails and failed trials. A failed
pivot retains the original completed-column prefix and status; the usual final pass
invalidates the rest. This candidate changes scheduling, preserving the documented sequence.

## Completed correctness checks

- All 31 widths pass trials/factor/solve MPFR references, normally and with Metal shader
  validation (`section9_p12_block_all_widths_reference.txt` / `...validation.txt`).
- Cases include 3x3, 65x65 and 49x49 panels; four/ten trials, three RHS, ignored upper
  entries, invalid inputs and failures at pivots 1, 8, 9, 16, 17, 32, 33 and 49.
- All 26 library CTest suites pass (`section9_p12_block_full_suite.txt`, 995.16 s).
- All 19 unique GPU suites pass under shader validation: 15 in
  `section9_p12_block_gpu_validation_suite.txt` (1575.05 s), plus the four
  complementary suites in `section9_p12_block_gpu_validation_complement.txt`
  (196.81 s). No shader source changed between these runs.
- Solver-size benchmark fixtures pass all four GPU paths against their own MPFR factor
  and solve sequence: n=200, eight trials at 352/384/448 bits; n=65, four trials at 1024;
  n=944, two trials at 352. These are checks only, with no recorded performance claim.

The initial 352 build failed because the new kernels were inserted inside an existing
Metal function. Its error log is retained; correcting the insertion passes the focused
check. This was a source-assembly error, not an accepted numerical candidate.

## Pending acceptance and measurements

The required suite validation is complete. The first measured large-shape result below
is unfavorable; no production performance change is accepted.

`section9_trials_limbforge` compares scalar columns, deferred panels, block panels and
exact `Linalg` block-32 factors/solves. Path order rotates over repetitions; every timed
output is checked. `--warm` executes and verifies an untimed call immediately before
each sample; default records calls after CPU/reference work. Device/wall medians,
quartiles, range and raw sample order are recorded separately. Batched host measurements
include uploads/downloads; resident measurements exclude them. The exact Linalg comparison
is a host-array path and is labelled accordingly even in a resident run.

CPU work uses at most min(trials,workers) active workers (eight for eight trials).
Both timed CPU contracts reuse persistent MPFR factors, solutions and per-trial scratch.
The timed exact-block32 path uses `mpfr_dot` for each exact update including the previous
entry; it is independently checked against the Float/exact-product block32 replay.
Sequential and exact CPU sample orders rotate, and every timed factor/solve is rechecked
outside timing. CSV rows select the matching CPU contract for each GPU path.

Focused 352-bit n=65/two-trial host and 1024-bit n=17/two-trial resident checks pass
normally and under shader validation after the rebase, with 2/2 package checks passing.
Shader and trial encoder bodies are unchanged from the full arithmetic gates; see
`section9_p12_rebase1_2_metadata.json`. The private path selector has not been accepted
as a public policy.

The default clock label is `cpu-interleaved`; `--warm` is `verified-warm-call`. Neither
proves controlled cold clocks or continuously warm clocks. Measurements at n=200/400/944 and
352/384/448, repeated stability checks and load metadata remain to be collected before
choosing a production path. The requested 3x target is not established.

## First large-shape measurement (352 bits)

At n=944, eight trials and three RHS, three CPU-interleaved checked repetitions
give the following factor-plus-solve wall medians. Host-array staging is included.

| GPU path | Wall (s) | Matching optimized MPFR CPU contract | CPU wall (s) |
|---|---:|---|---:|
| scalar | 7.892603 | sequential | 12.239791 |
| panel | 8.028042 | sequential | 12.239791 |
| block | 8.553135 | sequential | 12.239791 |
| exact_linalg_block32 | 0.763315 | exact_block32 | 15.102400 |

The blocked sequential candidate fails to improve the scalar path here and loses
strongly to the existing exact-block32 Linalg loop. These contracts round differently;
all CPU and GPU outputs match their own independent references. The 3x objective is
not met by this result; retain the current production default and pursue the exact-contract
batched design (P2.2) rather than promoting this candidate. This is a combined factor/solve
measurement, not a standalone factor timing. Host loads are 12–24, so CPU ratios are
busy-host observations and not consumer timing acceptance.

Raw samples and measured source/archive identities are retained in
`section9_p12_measured_352_944_8_interleaved.*` and `section9_p12_first_352_metadata.json`.
The completed comparison grid follows below.

## Completed comparison grid

Every initial, warm-up and timed CPU/GPU factor/solve matched its own reference.
Eight trials, three RHS, host-array staging, three repeats; wall medians in seconds.

| Bits | n | Clock label | Scalar | Deferred panel | Block panel | Exact Linalg loop |
|---|---:|---|---:|---:|---:|---:|
| 352 | 944 | cpu-interleaved | 7.892603 | 8.028042 | 8.553135 | 0.763315 |
| 352 | 944 | verified-warm-call | 7.944118 | 7.983401 | 8.309244 | 0.824959 |
| 384 | 944 | cpu-interleaved | 8.554781 | 8.748549 | 9.002318 | 0.853044 |
| 448 | 944 | cpu-interleaved | 11.374544 | 11.554683 | 11.737764 | 1.049240 |
| 352 | 200 | cpu-interleaved | 0.347786 | 0.356316 | 0.380037 | 0.063961 |
| 352 | 400 | cpu-interleaved | 1.380208 | 1.384497 | 1.421159 | 0.190053 |

The sequential block candidate is slower than scalar in every measured row and
5.9–11.2x slower than the exact block32 loop at these combined factor/solve fixtures.
The historical small-matrix advantage is not reproduced by this corrected-backend
comparison. These results reject this candidate as a production optimization. They
do not establish a universal cutoff or prove that a different batched exact design
cannot win. The exact loop and sequential candidates retain distinct numerical
contracts, and no existing default is changed.

Raw commands, CPU contract-specific medians, quartiles/ranges, rotating orders, host
loads and source/binary/archive identities are in `section9_p12_large_measurements_metadata.json`
and its six named CSV/log pairs. Measured CPU baselines use at most eight active workers;
18 pool workers does not imply 18 independent trial tasks. No idle-host consumer
speed claim follows. Further repeats of this unsuccessful design are unnecessary
without a new implementation; P1.2's performance objective remains open.
