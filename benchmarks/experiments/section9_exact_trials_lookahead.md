# Exact damping trials: isolated look-ahead experiment

Source snapshot `05f22fa`, based on main `4f1bb62`. The accompanying patch applies
to that base and remains an experiment: its API is not installed in WolfNum 1.3.1.
It builds on the [serialized baseline](section9_exact_trials_baseline.md), which
missed the 3x factor target. A first checked measurement is recorded below; no stable speedup is accepted.

`WOLFNUM_EXACT_TRIAL_SCHEDULE=lookahead` recomputes and factors the next-panel
Schur strip while the GPU updates separate trailing scratch. Packed current
columns stay immutable. After wait resolves repairs, scatter preserves prefactored
columns and failed trials' invalid tails. Original buffers remain claimed for the
entire synchronous call. The exact block-dot rounding sequence is preserved.
Serialized scheduling remains the experimental default.

All 31 widths pass normally and under shader validation, including a guaranteed
failure at zero-based pivot 32 (status 33) while another trial succeeds. Independent
references check completed prefixes, invalid tails, actual device updates and
accounting. The full 26 normal and 19 shader-validation regression suites pass.
Raw logs and source/archive/binary hashes use `section9_p22_lookahead_gate_*`;
focused evidence uses `section9_p22_lookahead_focus_*`.

These gates cover the experimental snapshot, separately from the final 1.3.1
release checks. Arithmetic/shader sources and existing request layouts are
unchanged. Panel/update accounting can overlap; elapsed factor wall time is the
performance measure. The 352/384/448-bit solver-size timings, memory comparison,
3x target and release acceptance remain pending.

## First checked measurement

At 352 bits, n=944, eight trials and three RHS, three checked CPU-interleaved
repetitions give median factor walls of 0.612 s for the exact Linalg loop and
0.528 s for look-ahead panels. Factor/solve totals are 0.719 s and 0.627 s.
Sample ranges overlap; the approximately 14% factor reduction is not accepted
as a stable speedup and the 3x target remains unmet. Busy CPU ratios remain
outside idle-consumer acceptance. Retained batched scratch is 385,695,744 bytes;
whole-process maximum RSS is 3,067,379,712 bytes, including reference arrays.
The driver still issues 28 shared submissions and 224 device updates.

Raw records use `section9_p22_lookahead_measured_352_944_8_host_interleaved`.
The explicit schedule and preceding correctness gate are recorded in metadata.
The next factor optimization needs better trailing-update throughput; repeating
the unchanged full timing grid would not establish the requested target.
