# Section 9 P1.4: normal-equation measurement harness

Base `e91a215` / release 1.1.0; branch `round-section9-normal-measurements`.
No library kernel or API change. No accepted timings yet.

`section9_normal_limbforge` compares sequential composed/fused normal equations,
augmented exact normal equations and plain exact SYRK plus exact GEMM for the RHS.
The GPU paths and their CPU references compute one triangle and mirror the normal
matrix. CPU rows/outputs are balanced over the requested worker pool. The exact CPU
timing uses MPFR's `mpfr_dot`, not a deliberately slower sequence at excess precision.

A separate high-precision oracle (2B+64, bounded input exponents, rows<=4096) retains
the exact sum before final rounding. MPFR composed/fused sequences and every GPU sample
are checked bit for bit against their corresponding reference; timed MPFR dots are
checked against the high-precision oracle outside the timer. Accuracy reports count
entries differing from the once-rounded exact result and the infinity-norm error in
units of 2^-B, normalized by the maximum exact output magnitude. This global measure
should not be confused with per-entry relative errors on cancellation-small entries.

Path order rotates across repetitions. Host staging includes input uploads and output
downloads using reusable buffers. Resident timing excludes these transfers, while
wait-time exact fallback remains timed. Optional `--warm` verifies an untimed call
immediately before each sample; default is CPU-interleaved, without claiming actual
cold clocks while other GPU jobs may be running. CSV retains quartiles and range;
raw samples preserve order and device/wall costs.

Build the `section9_normal_limbforge` target. Small check-only cases pass at
352 (9x4 host staging) and 1024 (11x7 resident); they are not the requested large-shape
measurement. Actual n=944, K=1100 and 352/384/448 timing/accuracy runs, repeated stability
checks, load metadata and guidance decision remain pending. Existing sequential calls
retain their numerical meaning and immediate device-chainability.

Logs: `section9_p14_normal_352_final_check.txt` and
`section9_p14_normal_1024_resident_final_check.txt`.
