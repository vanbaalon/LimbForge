# P3: integrated opt-in consumer adapters

The checked Fourier, tangent product3 and batched LU4 prototypes are now applied
to the original qscmx source, with their switches off by default. A separate
`build-wolfnum-section9` build uses matching WolfNum 1.3.1 headers/archive at
`0a08644` and ten inline limbs. Existing binaries are retained. Source drift checks,
backups and exact staged reconstruction preceded application. No library numerical
source, installed API, version or default changed.

`QSC_GPU_FOURIER`, `QSC_GPU_PRODUCTS4` and `QSC_GPU_LU4` explicitly select the
sequences described in their prototype reports. Transactional CPU fallback,
global/sub-switches, supported-width guards, owning prewarm and separate conversion
profiles remain in place. Engine host LU4 staging is serialized. The comparison
runner accepts `--fourier`, `--products4` and `--lu4`, clears private staging audit
variables and explicitly controls every candidate switch. It requires all selected
GPU markers. Ordinary comparisons disable these adapters even when inherited
environment variables enable them.

The existing startup/robustness helper and all three new MPFR helpers pass in
normal and Metal validation modes. New helpers cover 352/374/448 bits here;
the isolated LU4 gate separately covers all 31 supported widths plus 374 bits.
Two integrated comparison pairs pass on the unchanged saved J3 g=.202 fixture:

| GPU selection | Maximum component Delta difference | Residual | History |
|---|---:|---:|---|
| Existing GPU paths, new adapters disabled | 7.10245197342712e-125 | 5.57752179966997436e-25 | Three serialized values match |
| Existing GPU paths plus all three adapters | 2.307675812432015830e-92 | 5.57752179966997436e-25 | Three serialized values match |

Both CPU results exactly match the previously recorded CPU baseline. Every solve
meets its unchanged input tolerance `1e-22`; all requested GPU and adjoint markers
pass without fallback. The runner was deliberately given inherited candidate and
private staging switches, verifying its explicit selection behavior. Frozen source,
archive and binary hashes, terminal status and raw artifacts were checked independently.

[Complete source patch](section9_consumer_integrated.patch) reconstructs all six
changed/added files exactly from the backed-up baseline. Records are archived in
`benchmarks/results/section9_p3_integrated_consumer`; the consumer also has local
backups and its updated `LIMBFORGE_STATUS.md`. Verify the archived records without
rerunning a solver or GPU test:

```sh
python3 benchmarks/experiments/check_section9_consumer_records.py
```

The requested converged g=.1/.2/.5 cases, idle Nc=23/31/59 timing table, large-shape
memory and damping/base switches remain open. This integration establishes no
speedup, cutoff, new default or completion of P3.
