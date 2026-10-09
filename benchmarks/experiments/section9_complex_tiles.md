# Private complex GEMM tiling and register blocking

Base: WolfNum 1.3.1, `deee518`. No public header/layout/signature or default changes.
The private selector `WOLFNUM_COMPLEX_GEMM_TILE` is read once by the owning cache;
prewarm and dispatch use the same selected kernels. Unset/`scalar` preserves the
production scalar-component kernel. No new timing or acceptance claim is made yet.

Candidates: 8x4 tiles with automatic, four or eight K values; 8x8 tiles with two
outputs per component thread; 16x8 with four outputs per component thread. Register
blocking keeps ascending K accumulation separately for every output. Shared inputs
retain logical storage; arithmetic retains the existing padded 33-word private
scalar representation. Composed/fused rounding and final negation are unchanged.
All configurations use 64 threads per group. The largest shared-memory allocation
is 26,880 bytes at 1024 bits, within the 32 KB limit. A separate 4x4 kernel caches
two matrices per group; odd batch tails remain masked.

The first compile probes failed before dispatch because thread-position/group-position
vectors had different dimensions. The saved initial patch and raw logs retain that
failure. After both coordinate declarations use uint3, all five variants pass the
352/1024 independent GEMM reference checks, including fused/composed, broadcast,
strides, output padding, accumulation/negation, statuses, empty dots and product3.

All-width, wide-exponent, dense and shader-validation checks and broad regressions
remain required before performance measurement. Library defaults remain unchanged;
these kernels are not merged into production.

## First all-width sweep: register-blocked candidate rejected

The frozen `e84473f` sweep passes all 31 precisions for `8x4auto`, `8x4k4` and
`8x4k8`, each with small cases and dense/wide-exponent cases, normally and under
shader validation. The 8x8/two-output variant passes its two normal sweeps and
small shader cases, but fails the 64-bit dense/wide composed shader case at output
106 (wrong real exponent/limbs; imaginary component agrees). The full gate stops
at that failure; the 16x8 variant has not run. No compact-power or broad follow-up
checks are claimed: the queued runner stops before any dispatch when this gate fails.

Raw failure and completed-step source/binary identities are in
`section9_p13_tiles_allwidth_gate_metadata.json` and its named logs. This rejects
current multi-output acceptance; it does not identify a compiler root cause.
The three single-output tiles remain candidates requiring remaining gates/timing.

The separate named-accumulator correction also fails the 64-bit dense/wide shader
probe (index184, imaginary high word lost). Replacing the indexed sum array is
insufficient. No timing or wider sweep is performed on this failed candidate.

Both retained patches are based on production1.3.1 `deee518`; source snapshots
`e84473f` and `0650238` identify the first and named-accumulator attempts. They
are not applied to main. Full-library gates and timing are not claimed for either.
