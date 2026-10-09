# P3: batched Fourier Jacobian consumer prototype

This isolated qscmx adapter leaves the user's original consumer sources, binaries
and running jobs untouched. `QSC_GPU_FOURIER=1` is explicit and off by default.
The existing residual implementation remains the baseline.

The subsequent [local integration](section9_consumer_integrated.md) applies this
checked prototype with a separate matching build and its switch off by default.

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
malformed-input preservation. It passes at 352/374/448 bits normally and with
Metal shader validation.

The saved J3 GPU replay also passes: Delta differs from the recorded CPU result
by at most `1.539673154646756417e-92`, the three history entries match exactly at
their serialized precision, and both residuals are `5.57752179966997436e-25`,
below the input tolerance `1e-22`. Existing GPU paths, the Fourier path and the
adjoint check run without fallback. Terminal status, frozen source/binary hashes,
raw logs and output values have been independently verified.

[Prepared patch](section9_fourier_consumer.patch) and
[helper source](section9_fourier_consumer_check.cpp) are archived consumer prototypes. Raw CPU/GPU evidence
and hashes are in `benchmarks/results/section9_p3_fourier_prototype`. Reapply only to
the recorded consumer baseline after checking source drift. Original consumer
integration, the requested converged g sweep and idle-timing acceptance remain open. This does not
complete the requested g=.1/.2/.5 checks or any remaining 4x4/damping/base switches.
