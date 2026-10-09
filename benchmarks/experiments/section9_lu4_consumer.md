# P3: batched LU4 consumer prototype

This isolated qscmx source builds on the prepared
[Fourier](section9_fourier_consumer.md) and
[tangent product3](section9_products4_consumer.md) consumer adapters. The patch is
incremental on their frozen source. The original consumer and running jobs are
untouched; production WolfNum remains 1.3.1.

The subsequent [local integration](section9_consumer_integrated.md) applies this
checked prototype with a separate matching build and its switch off by default.

`QSC_GPU_LU4=1` explicitly selects the existing library's composed, pivoted 4x4
inverse. Descent matrices are collected across all collocation points before one
LU4 call; cut-row construction follows the completed inverses. Mode Jacobian
workers collect all fitted gluing matrices for both directions before a second
batched inversion, then prepare their samples and project them. This batches the
per-column gluing inverses rather than issuing a GPU call for each fit. Stored
mode rows are released before Fourier projection. Large-shape peak memory still
requires measurement.

The option is off by default, respects the global/sub-switch, startup failure and
supported-width guard, and preserves the CPU Gauss-Jordan/MPC fallback. Input,
shape, pivot, arithmetic and conversion failures leave the caller's output
unchanged and warn once. Engine's synchronous host LU4 call reuses staging buffers,
so the helper serializes that call if multiple finite-difference workers reach it.
Its pack, library/device and unpack phases are profiled separately. Startup owns
and drains an additional Engine LU4 prewarm future. Deferred CPU sampling and
projection are included in the residual work counter.

The selected numerical sequence differs from the original Gauss-Jordan inverse:
LU uses the exact max-component pivot predicate with lowest-row ties, composed
rounded complex products/subtractions/divisions and ordered forward/back
substitution. It runs at the supported GPU width and then rounds to consumer
precision. Existing library defaults, requests, layouts and arithmetic are
unchanged. See the library's [LU4 contract](../../docs/numerics.md).

Compilation passes after correcting a helper indexing error; the initial failed
build log is retained. Final default, forced LU4 staged-CPU and combined
LU4/products/Fourier staged-CPU J3 g=.202 runs exactly reproduce the recorded
Delta, three-entry history and residual `5.57752179966997436e-25`. Adjoint checks
pass. The CPU helper's fixtures/global guard run at all 31 supported widths plus
374-bit consumer precision. These are CPU staging/guard checks, not GPU arithmetic
or performance acceptance. `final_cpu_metadata.json` identifies the final source
and binaries; preliminary records retain their earlier identities.

The prepared GPU helper independently replays the scalar MPFR LU sequence. It
checks inverses, row swaps, exact pivot ties, scaled rows, rounded-up width,
malformed/NaN input, a singular companion matrix, transactional publication,
empty batches, sub-switches and >1024-bit fallback. All 31 supported widths and
374-bit consumer precision pass normally and under Metal shader validation.
The J3 LU4-only and combined LU4/products/Fourier GPU replays pass: maximum Delta
component differences are `1.787604346388709787e-92` and
`2.307675812432015830e-92`. Both retain the three serialized CPU history entries
and residual `5.57752179966997436e-25`, below the input tolerance `1e-22`.
All required GPU and adjoint markers occur without fallback. A separate verifier
checks terminal status, frozen source/binary identities, raw hashes and output
acceptance independently of the runner.

[Incremental patch](section9_lu4_consumer.patch),
[helper](section9_lu4_consumer_check.cpp) and raw/hash records under
`benchmarks/results/section9_p3_lu4_prototype` archive this consumer prototype. Apply only to the
recorded base after checking drift. Requested converged g=.1/.2/.5 comparisons,
idle timings and production consumer integration remain open. No speed claim,
cutoff or new production library release is accepted by these correctness checks.

Verify all three archived GPU record sets without rerunning a solver or GPU test:

```sh
python3 benchmarks/experiments/check_section9_consumer_records.py
```
