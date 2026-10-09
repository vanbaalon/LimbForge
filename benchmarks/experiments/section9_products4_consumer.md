# P3: batched tangent 4x4 consumer prototype

This isolated qscmx adapter builds on the
[Fourier staging candidate](section9_fourier_consumer.md). The accompanying patch
applies to the recorded Fourier consumer snapshot, not directly to the original
consumer. Base/prepared hashes are in
`benchmarks/results/section9_p3_products4_prototype/preparation_manifest.json`.
Original consumer files, binaries and running jobs are untouched.

`QSC_GPU_PRODUCTS4=1` explicitly opts into batching the inverse derivative
`-Qi * dQ * Qi` for every tangent column and collocation point. Workers prepare
their tangent descent matrices first. One resident `product3` submission computes
both matrix products for all eligible columns; the original CPU cut-row updates
and residual construction follow. Non-tangent columns retain their existing
path. This avoids a GPU submission for each Jacobian column. The owning batched
prewarm request includes the 4x4 specialization when this switch is selected.

The switch is off by default, respects global GPU opt-out, startup failure and
the supported width guard, and publishes outputs only after wait, status checks
and successful conversion. Conversion/arithmetic errors warn once and replay the
original sequential MPC products on the CPU. Pack/upload, library/device and
download/unpack times have separate markers. No cutoff or performance claim is
accepted yet.

The explicit contract is the existing library composed GEMM sequence: scalar
real products are rounded before the complex sum/difference, each ascending
inner-term addition is rounded, the intermediate matrix is rounded, and the
second product is negated. Operations use the rounded-up GPU width before copying
back to the consumer precision. This differs from MPC complex multiplication
at consumer precision and requires consumer agreement checks.

The matching 1.3.1 consumer and helper build. CPU helper fixtures at 352/374/448
bits exercise the global GPU guard. Default, forced staged-CPU, and combined
Fourier/products staged-CPU J=3 g=.202 replays exactly match the recorded Delta,
three-entry history and residual `5.57752179966997436e-25`. The staged run's adjoint
check passes. These establish CPU staging correctness; GPU evidence is separate.

The GPU helper independently replays both products with MPFR scalar operations
and covers an odd point count, skipped-column mapping, high-precision decimal
inputs, cancellation, large exponent spread, malformed/nonfinite inputs,
transactional output, the sub-switch and >1024-bit fallback. Normal and Metal
validation runs pass at 352/374/448 bits. The saved J3 product-only and combined
Fourier/product GPU replays pass independently parsed acceptance checks. Maximum
component Delta differences are `7.10245197342712e-125` and
`1.539673154646756417e-92`, respectively. Both retain the three serialized CPU
history entries and residual `5.57752179966997436e-25`, below input tolerance
`1e-22`. Requested GPU paths and adjoint checks run without fallback. Terminal
status, frozen source/binary identities and raw log/output hashes are verified.

[Source patch](section9_products4_consumer.patch) and
[helper](section9_products4_consumer_check.cpp) remain archived consumer prototypes.
The subsequent [LU4 prototype](section9_lu4_consumer.md) also passes its saved J3
combined replay. Original consumer integration, the requested converged g=.1/.2/.5 cases and idle-host timings
remain open. This preparation does not change the production library or its
version and does not complete the full P3 checklist.
