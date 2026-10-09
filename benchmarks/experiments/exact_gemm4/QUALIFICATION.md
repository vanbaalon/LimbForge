# Qualification and measurement order

Prototype/source gate commit: `1e748c6`, based on main `97d6b28`.
The four 31-width small/dense normal/Metal sweeps are terminal PASS.
The required 26 normal and 19 Metal library regressions are running in the
identified `gates.py` process. This is incomplete qualification, not acceptance.

`post_gates.py` waits on that exact PID and process birth/command identity.
Only a terminal PASS starts the extra 31-width overflow-cancellation, midpoint
plus/minus far epsilon and accumulated-component-status checks, then eight
352/1024-bit host/resident normal/Metal comparison checks. `clock_gates.py`
waits on that identified post-gate process and checks 1/32 independent live
calls in one submission at every width in both modes. A missing handle fails;
no driver restarts a process or overwrites an existing result folder.

The copied comparison replaces only the old per-matrix exact embedding driver
with the single-dispatch prototype. It keeps all four MPFR references, the
independent high-precision oracle, fixture generator, rotating CPU/GPU orders,
wall/device scopes and accuracy/error accounting. Its31 CPU widths and six
shape/selector rejection checks pass. `exact_batched4` is an experimental
backend label, with the same once-rounded complex exact-dot contract as the
embedding, excluding accumulation in this fixture. Every exact GPU call must
also report zero host repairs for the bounded-exponent fixture.

Use `sh benchmarks/experiments/exact_gemm4/build_tools.sh` from the repository
root after building `wolfnum_exact_gemm4`. The tools are intentionally not
installed and do not change the default build or public API1.3.1.
Run `verify.py --live-root "$PWD"` after all three correctness drivers terminate;
it checks raw source/log/binary identities and actual coverage, not just PASS
fields. An unfinished gate makes verification fail.

Only then measure352/384/448-bit4x4 batches of1e4/1e5 matrices, host/resident,
CPU-interleaved/verified-warm-call clocks, with three rotating checked samples
per configuration and all four contracts. Preserve raw sample orders, medians,
quartiles/ranges, normalized accuracy, process memory and host load. These are
library measurements; busy-host CPU ratios do not establish idle consumer
policy. No new measurement has run while qualification is pending.

The separate clock probe reports100ms idle-delayed single calls and32 calls
in one batch after at least200ms accumulated checked device work. Snapshots and
dispatches are in device time; encoding, scratch allocation, wait and report
are in wall time. Verification/transfers are outside resident timing. These
conditions do not prove a controlled GPU frequency or continuous warmth.

After the checked baseline, candidate follow-up rounds include: a compact
fallback counter to avoid scanning every mask on the bounded fast path;
workspace reuse with the existing busy-release/lifetime rules; and a fully
on-device clustered exact sum for large exponent spreads, with a rigorous
rounding proof and fresh physical-GPU gates. Do not change the current frozen
kernel before its baseline is recorded. Promotion requires measured benefit,
final source/package checks and an additive minor release. The separate3x
exact-trial target and consumer idle/memory/damping/base tasks remain open.
