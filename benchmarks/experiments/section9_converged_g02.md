# Converged J3 g=0.2 consumer check

The matching 1.3.1 `build-wolfnum-section9` consumer now passes the requested
g=0.2 comparison with exact normals, Fourier projection, product3 and LU4
explicitly enabled. Existing stage-2 power moments, normal matrix and first
Cholesky also execute on the GPU. All these switches retain their existing defaults.

The J3 X2Y seed is the saved same-coupling point at Nc21/NQ25/Nsh300/nPts27,
374-bit consumer precision. Delta is perturbed by 1e-24 to exercise Newton and
the Jacobian; a starting point already below tolerance would not exercise all
required paths. The original J3 v-parameterization, gluing mask, constraint cutoff,
zero initial damping and gtol=1e-22 are preserved. No convergence criterion is relaxed.

| Check | Result |
|---|---|
| CPU/GPU residual | Both 1.48318644267521618e-25 |
| Serialized history | Two entries, equal exactly; one Newton update |
| Maximum component Delta difference | 3.04846e-89, below 1e-25 |
| Adjoint and selected GPU markers | Pass, with no CPU fallback |
| Consumer/library identities | Frozen sources, matching headers/archive and binary |

The initial exports omitted the J3 cutoff/mask and incorrectly enabled constraint
residuals; those local failed records are preserved. Corrected nearby-coupling
single-point scaled seeds still stalled, despite CPU/GPU agreement. They are not
converged acceptance. A CPU evaluation of an exact saved g=.202 point independently
reproduces its recorded residual, with and without v-parameterization, isolating
the issue to seed preparation. The accepted g=.2 fixture uses its saved point.

[Input, raw CPU/GPU logs and outputs, exporter and provenance](../results/section9_p3_g02_verified)
are retained. Run `python3 benchmarks/experiments/check_converged_consumer.py`
to recompute the comparison from raw outputs; `--live` also checks frozen local
source/input/archive/binary hashes. The full g=.1/.2/.5 sweep, idle-host timing,
large-shape memory and remaining damping/base work are not established by this case.
Recorded host load exceeds the idle gate; no timing acceptance follows.
