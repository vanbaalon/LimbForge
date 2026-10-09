# Exact damping trials: isolated look-ahead experiment

Source snapshot `05f22fa`, based on main `4f1bb62`. The accompanying patch applies
to that base and remains an experiment: its API is not installed in WolfNum 1.3.1.
It builds on the [serialized baseline](section9_exact_trials_baseline.md), which
missed the 3x factor target. No look-ahead performance measurement is claimed.

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
