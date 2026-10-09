# Section 9 P2.2: exact batched trial baseline

This is a correctness-checked **unreleased candidate**, not an API shipped on main.
The source snapshot is `70d87d4` on `round-section9-exact-trials`, rebased on WolfNum
main `bffa58b` / 1.3.0. The retained patch applies to that explicit base. It includes
the numerical/ownership contract, independent test and opt-in benchmark. Do not
interpret its unchanged development version macros as a published API release.

The driver computes each trial with the production Linalg block32 sequence, using
parallel host panels and shared resident exact SYRK submissions. It passes all31
widths normally and under shader validation, the full26 normal suites, and all19
distinct shader-validation suites. The last gate spans five completed suites from
an interrupted run plus fourteen complementary suites on the same frozen source.
Four linked fixtures pass both before and after the WolfNum rebase. The rebase does
not change the gated numerical driver, independent test, public addition or shaders.
Raw manifests distinguish archive identities and check scopes; no new GPU compiler
failure is inferred from the retained initial C++ namespace build error.

## First solver-size measurement

Eight trials, n=944, three RHS, 352 bits, 18 requested Linalg/MPFR workers (at most
eight CPU reference trials active), three repetitions, host-array staging included.
Every initial and timed CPU/GPU factor and solve matches its independent reference.
The MPFR pool and scratch are persistent. Numeric bridge conversion is excluded.

| Path | Factor wall (s) | Solve wall (s) | Factor + solve wall (s) |
|---|---:|---:|---:|
| `sequential_scalar` | 1.461449 | 7.291717 | 8.753166 |
| `exact_linalg_loop` | 0.619183 | 0.102270 | 0.721453 |
| `exact_batched_panels` | 0.684590 | 0.119002 | 0.803593 |

The batched factor is 1.106x the exact-loop wall time
(10.6% slower), so the required 3x improvement is **not met**.
Component medians are separate statistics; they need not sum to the median total.
Batched panel accounting is 0.165–0.206 s and trailing-update accounting 0.440–0.491 s.
The existing Linalg loop overlaps panel and update work, so its summed component
accounting is not its critical-path wall time. The new driver waits between these
phases. This is evidence to investigate scheduling and update throughput before
repeating the unchanged candidate over a complete grid, not proof of a new speedup.

All timed batch updates report 33,341,952 actual GPU outputs and zero exact fallback
outputs. It retains 385,695,744 bytes of batched scratch; the shared Linalg instance
reports 137,895,192 scratch and 602,959,232 idle resident-workspace bytes after both
exact paths have run. These shared caches are not an isolated per-call allocation.
Whole-process maximum RSS is 3,065,823,232 bytes; the macOS time log also reports a
4,145,875,872-byte peak memory footprint. Both include MPFR oracles, inputs, outputs,
staging and caches, and do not describe consumer peak memory.

Recorded loads exceed4. These are CPU-interleaved busy-host library measurements,
not idle qscmx timing, controlled cold clocks or a continuously warm GPU. Sequential
and exact contracts have separate matched CPU baselines; no cross-contract CPU
speedup or updated production cutoff is claimed. Statuses/finality/defaults in
production stay unchanged.

## Remaining work

Improve the dependency schedule and/or true batched exact update throughput, rerun
independent correctness gates, then measure n=944 at352/384/448 plus n=200/400 and
memory before accepting the 3x target or preparing a compatible minor release.
Normal-equation, Fourier/4x4, idle calibration and consumer histories remain separate
Section9 requirements. The source candidate and all records remain available for
reproducible comparisons; this baseline does not complete P1.2/P2.2.
