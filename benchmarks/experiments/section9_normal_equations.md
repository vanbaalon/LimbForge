# Section 9 P1.4: checked normal-equation comparisons

WolfNum 1.3.0 numerical backend, opt-in `section9_normal_limbforge`, n=944 and K=1100.
All six 352/384/448-bit host/resident comparisons complete three repetitions with
rotating CPU/GPU order. Every initial and timed CPU/GPU output matches its own
independent composed/fused/exact contract. The persistent 18-worker CPU references
compute one triangle and mirror it; their exact timed baseline uses `mpfr_dot` at B
bits, independently checked against a 2B+64 exact sum. Linalg uses 18 host workers too.
The CPU-only all 31 and rebuilt focused physical-GPU/shader fixtures pass beforehand.
No numerical library source or API change is introduced by the harness.

## GPU wall measurements

All values below are median seconds for the normal matrix AND its one RHS.
Host staging includes uploads/readback using reused buffers; resident excludes them.
Exact fallback resolution stays inside timing. Numeric bridge conversion is excluded.

| Bits | Storage | Composed | Fused | Exact augmented | Exact SYRK + GEMM | Composed / augmented |
|---|---|---:|---:|---:|---:|---:|
| 352 | host | 0.219159 | 1.487367 | 0.049917 | 0.031334 | 4.39x |
| 352 | resident | 0.206483 | 0.905758 | 0.024285 | 0.019263 | 8.50x |
| 384 | host | 0.253707 | 1.149127 | 0.027911 | 0.019899 | 9.09x |
| 384 | resident | 0.238819 | 1.227243 | 0.024466 | 0.019214 | 9.76x |
| 448 | host | 0.310288 | 2.058552 | 0.037930 | 0.031535 | 8.18x |
| 448 | resident | 0.298880 | 1.973909 | 0.038354 | 0.027808 | 7.79x |

The exact augmented path has 4.39–9.76x lower median wall time
than composed sequential normals at these profiles. Plain exact SYRK plus GEMM has
the lowest median in all six; its sample ranges can overlap the augmented path.
The raw records retain each sample, quartiles and range. Clock mode is
`cpu-interleaved`, with no controlled-cold or continuously-warm guarantee.

## Accuracy and usage decision

| Bits | Composed global error units | Fused global error units | Either exact path |
|---|---:|---:|---:|
| 352 | 21.137290 | 21.047510 | 0.554940 |
| 384 | 21.773523 | 21.773523 | 0.555361 |
| 448 | 25.687258 | 26.795326 | 0.553506 |

Error units are 2^B times maximum absolute output error divided by maximum exact
output magnitude. They are a global infinity-norm comparison, not per-entry relative
errors or a claimed number of correct bits. Sequential results differ from the exact
result at 846,593–847,055 of 892,080 matrix/RHS entries; both exact paths match the
once-rounded exact oracle at every entry. This fixture uses bounded random exponents
in [-4,4]; wider exponent distributions can alter exact-backend placement and costs.

**Recommend `normal_equations_exact` for independent normal-matrix/RHS calls at
these measured profiles.** Plain `Linalg::syrk` plus `gemm` is another exact choice
when the caller already has the needed resident layouts. Preserve existing
`normal_equations` meaning/defaults: the sequential form remains useful for immediate
within-batch chaining. Exact products require wait-time finalization and repair before
a dependent read; the recommendation does not remove those ownership rules or
select a qscmx switch by default.

Loads exceed 4 throughout parts of every run. CPU ratios therefore remain busy-host
library evidence, not idle-consumer acceptance or a new CPU speedup headline.
Whole-process maximum RSS spans 788.7–916.9 MB, including MPFR truth/reference storage,
inputs, outputs and caches. It is not a consumer or isolated API memory guarantee.
Consumer Jacobian timing, converged histories and per-problem calibration remain P3.

## Reproduce and calibrate

Build `section9_normal_limbforge`, then run, for each width and both storage modes:

```sh
./build/section9_normal_limbforge --bits 352 --rows 1100 --cols 944 --workers 18 --repeats 3
./build/section9_normal_limbforge --bits 352 --rows 1100 --cols 944 --workers 18 --repeats 3 --resident
```

Raw CSV/log/metadata triples use the `section9_p14_measured_*_interleaved` prefix.
Manifests retain exact commands, source/binary/archive hashes, sampled loads and RSS.
The offline operation exporter yields 24 shape/contract/residency/profile-specific
normal keys; every generated C++ key loads and unmeasured widths/profiles stay
unknown. No table is installed as a policy or used to change production defaults.
Source recurrence retains its separate exporter; trial keys cover factor AND solve,
never a factor-only CPU comparison. See [offline calibration](../../docs/calibration.md).
