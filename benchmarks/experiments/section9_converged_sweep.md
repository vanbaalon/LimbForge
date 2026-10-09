# Requested J3 g=.1/.2/.5 convergence sweep

All three requested couplings now pass the matching 1.3.1 consumer comparison.
Each uses `QSC_GPU=0` for the CPU baseline and `QSC_GPU=1` with exact normals,
Fourier projection, product3 and LU4 explicitly enabled. Existing stage-2 power
moments, normal matrix and first Cholesky also execute on the GPU. All requested
markers and adjoint checks pass, with no CPU fallback. The switches remain off
by default; the production library stays at 1.3.1.

| g | Nc/NQ/Nsh/nPts | Consumer bits | CPU and GPU residual | Maximum component Delta error | History |
|---|---|---:|---:|---:|---|
| .1 | 21/25/300/27 | 374 | 9.75172865e-28 | 3.55747e-88 | Two entries, equal exactly |
| .2 | 21/25/300/27 | 374 | 1.48318644e-25 | 3.04846e-89 | Two entries, equal exactly |
| .5 | 33/33/2235/39 | 440 | 3.08195359e-23 | 3.18795e-101 | Two entries, equal exactly |

Every final modes solve preserves the original gtol=1e-22 and satisfies Delta
agreement <=1e-25. Each history includes the initial residual and one Newton update,
so all Jacobian/GPU paths are exercised. Histories are compared at the solver's
17-digit serialization precision. Headers/archive/binary and source hashes match
the previously checked integrated consumer; no new numerical library source is used.

The .2 fixture starts from the saved same-coupling J3 point, with Delta perturbed
by 1e-24. For .1 and .5, scaled nearby seeds needed the solver's CPU collocation
preparation before the final Fourier-modes solve. The .1 collocation residual is
2.90e-23. The .5 collocation step stalls at 6.37e-20, below the native
collocation-to-modes handoff threshold of 1e-8. That preparatory result is not
counted as final convergence; the subsequent modes CPU/GPU pair independently
converges below 1e-22. Raw v is preserved when transferring C++ seeds, with a
tiny perturbation to ensure the final comparison exercises Newton.

The initial invalid exports and corrected direct-seed stalls remain locally
recorded. The archived .1 preparation manifest retains the mixed driver's
`FAILED_ACCEPTANCE` status caused by the .5 collocation stall. The independent
checker accepts only its completed .1 modes step; the later .5 modes check has
its own successful terminal manifest. No failed result is relabeled as convergence.

Raw inputs, CPU/GPU logs and outputs, preparation records and frozen identities:
[g=.1](../results/section9_p3_g01_verified),
[g=.2](../results/section9_p3_g02_verified),
[g=.5](../results/section9_p3_g05_verified).
Run `python3 benchmarks/experiments/check_converged_consumer.py`; `--live` also
checks the original frozen local source/input/archive/binary files. All three
cases pass both modes of that independent verification.

This completes the requested combined-switch convergence sweep at these fixtures.
Idle Nc23/31/59 Jacobian timing, large-shape memory measurement, remaining damping
trials and optional base recurrence remain open. These loaded-host correctness
runs provide no accepted consumer timing, default switch or cutoff policy.
