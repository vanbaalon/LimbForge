# P3: batched Fourier Jacobian consumer prototype

This isolated qscmx adapter leaves the user's original consumer sources, binaries
and running jobs untouched. `QSC_GPU_FOURIER=1` is explicit and off by default.
The existing residual implementation remains the baseline.

Each Jacobian worker constructs the fitted gluing matrix and its point samples for
both finite-difference directions. One real-embedding exact GEMM projects all saved
samples using the shared Fourier phases; postprocessing then assembles the Jacobian.
This matches the large Fourier workload instead of issuing one tiny GPU call per
column. The helper packs once, waits for exact-dot finalization, reconstructs all
outputs transactionally and falls back to the original sequential MPC projection
on any failure. Packing/library/device/unpacking markers are separate. Existing
algebra prewarm also prepares the exact helper when this switch alone is enabled.

Numerical contract: imported samples/phases round to the supported GPU width;
real/imaginary exact dots round once there, then copy to consumer precision. This
is distinct from the CPU's sequential rounded MPC products/additions and is not a
new default. Unsupported widths/global opt-out preserve CPU behavior.

Compilation passes. A default CPU J=3 g=.202 replay and a private forced split-CPU
replay both exactly match the recorded Delta, three-entry history and residual.
The split replay exercises sample extraction, CPU projection and postprocessing;
it is numerical evidence for the staging refactor, not GPU acceptance. CPU helper
fixtures at352/374/448 and disabled-GPU/output-preservation guards pass. The physical-
GPU helper independently compares each result against MPFR exact dots and checks
malformed-input preservation, but its GPU runs remain pending.

[Prepared patch](section9_fourier_consumer.patch) and
[helper source](section9_fourier_consumer_check.cpp) are unmerged. Raw CPU evidence
and hashes are in `benchmarks/results/section9_p3_fourier_prototype`. Reapply only to
the recorded consumer baseline after checking source drift. GPU/reference/shader,
converged-history and idle-timing acceptance are still required. This does not
complete the requested g=.1/.2/.5 checks or any remaining 4x4/damping/base switches.
