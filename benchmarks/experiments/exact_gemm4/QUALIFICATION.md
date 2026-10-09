# Qualification and measurement status

All exact4 qualification and baseline measurements are terminal PASS. The
uninstalled prototype is based on production main `97d6b28`, WolfNum 1.3.1.
Existing installed APIs and numerical defaults are unchanged.

The four 31-width small/dense normal/Metal sweeps pass, followed by all 26 normal
and 19 Metal library regressions. All 31 supplemental overflow-cancellation,
far-epsilon ties and accumulated-component-status checks pass in both modes.
Eight 352/1024-bit host/resident normal/Metal comparison checks pass. Every width
also passes independently checked 1/32-live-call batches normally and under
Metal validation. Comparison CPU references pass at all 31 widths; six invalid
shape/selector checks pass.

The normal regression collector rejected an alternative CTest success-summary
spelling after the suite had passed. Original metadata/source are retained;
`resume_gates.py` independently checked all 26 unique passing test names, zero
exit and raw hashes, then ran only the outstanding 19 Metal suites. No numerical
failure occurred. Original post/clock dependency failures are retained separately
from the successful continued records.

The comparison preserves four distinct MPFR contracts, an independent
high-precision oracle, rotating CPU/GPU sample orders, accuracy accounting and
separate wall/device scopes. Every bounded exact call reports zero host repair.
24 configurations cover 352/384/448 bits, 10,000/100,000 4x4 matrices, host/resident
and CPU-interleaved/verified-warm-call profiles, with three checked samples per
contract. Three nine-repetition clock probes retain idle-delayed single calls
and 32-call batches after accumulated checked device work.

The first measurement collector assumed interpolated quartiles, whereas the
benchmark uses lower-rank sorted quantiles. Original rejected records and source
are retained. The corrected continuation independently validated and retained
three clock records and the first comparison, then ran only 23 missing profiles.
Raw orders, medians, lower-rank quartiles, ranges, accuracy, hashes, load and
process memory independently verify. Twenty synthetic rejection guards pass.

Run `verify.py` and `verify_measurements.py` from the repository root for offline
verification. In the original matching build add `--live-root "$PWD"` to check
binary identities. These checks establish numerical/measurement provenance,
not performance or consumer acceptance. [Full baseline and decision](BASELINE.md).

The exact candidate is slower than composed in all 24 recorded profiles and is
not promoted. Next optimization rounds must preserve this baseline before
changing workspace lifetimes or shader fallback accounting. A future additive
public API requires a minor release and final package/numerical checks. The
separate 3x exact-trial target and consumer idle/memory/damping/base tasks remain
open. No owned qualification or measurement jobs remain running.
