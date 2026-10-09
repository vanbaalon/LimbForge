# Section 9 P1.5: shared polynomial-source experiments

Base: `e91a215` / 1.1.0, branch `round-section9-shared-sources`.
No production acceptance, default change or speed claim.

A proposed additive `polynomial_sources` API writes p/q per step, shared group and
component, for reuse by arbitrary consumers. The current adapter handles incomplete sharing groups directly. This avoids duplicate Horner work and preserves
Horner/multiply/recurrence order. Production polynomial behavior remains the old inline
path while these experiments are gated.

Rejected/context-sensitive variants are retained as patches and raw logs:

- Padded private Complex Horner + combined p/q selection: dense 192-bit source/recurrence
  failures. Initially separately checked sources passed; later probes reproduced wrong q.
- Packed original Horner + combined selection: dense 192-bit q sources fail.
- Packed Horner with separate p/q dispatches: focused 192/288 probes pass, but the normal
  all-width sweep fails recurrence at 288. A completion-time dump of actual adapter
  sources then reproduces wrong p at 256. Correct source values in a separate dispatch
  do not prove that the adapter's dispatch produced correct ones.
- A component-oriented private-number candidate uses separate p/q dispatches and a
  single rolled exact-dot call site for real/imaginary Horner updates. Its source-only
  normal sweep passes all 31 widths, but shader validation fails a q source at 384 bits
  (`section9_p15_components_sources_all_widths_validation.txt`); rejected.

A further independent failure appeared at 128 bits: `CommandBatch::vector_recurrence`
with CPU/MPFR-generated p/q weights disagrees with MPFR, before the new source pass is
called for that case. Raw log: `section9_p15_components_all_widths_reference.txt`,
`baseline vector from CPU Horner sources index 3975`. This branch does not edit
`core.hpp` or `kernels.metal`; the underlying vector shader is the 1.1.0 baseline.
A separately compiled 128-bit reproducer linked to the unchanged production 1.1.0
archive passes its initial run, two repeats and shader validation. The failure has not
yet been reproduced independently of the expanded audit context. Do not infer a
compiler root cause or call the fused recurrence optimization verified.

Diagnostic cases use seed bits+31, 513 lanes sharing pairs (257 groups), three terms
and three steps; variants include shared/per-group coefficients, composed/fused,
reverse and all_steps, empty terms/steps and one invalid coefficient. Private diagnostic
flags check source buffers independently and can dump adapter p/q/seed/result after GPU
completion. Dumps are temporary and not a user API. Repeated focused 288 runs passing
alongside a failed all-width sweep are explicitly not acceptance.

The next candidate stages each Horner degree in separate device passes with ping-pong
rounded states and component threads, then scales by E. It computes sources once per
shared group but adds degree-sized dispatch counts and two small scratch tables. The
component arithmetic has one exact-dot call site without live complex recurrence states.
Its source-only all-width normal and shader-validation checks pass at all 31 widths
(`section9_p15_staged_sources_fixed_all_widths_reference.txt` and `...validation.txt`).
The earlier vector adapter passed both sweeps with a temporary completion-time dump,
but its final API form without that callback fails at 64 bits (fused, reverse, all_steps,
index 3975). Raw record: `section9_p15_api_all_widths_reference.txt`; saved patch:
`section9_shared_sources_vector_adapter_rejected.patch`. The callback-sensitive pass
is insufficient evidence; no root cause is established.

The current adapter stages a four-term dot and a separate update in device passes per
step, with one scalar component per thread and padded private significands. It preserves
the original sequence without keeping a live four-complex private state or padding the
tail. The 64-bit reproducing fixture and the initial 31-width dense normal sweep pass.
The expanded normal and shader-validation sweeps both pass all 31 widths, covering
sharing factors 1, 3 and 7 (the last exceeds the five-lane input), up to 17 Horner terms,
and the dense 513-lane fixture, composed/fused, reverse/all_steps, empty inputs and statuses.
Records: `section9_p15_staged_recurrence_expanded_reference.txt` / `...validation.txt`.

The standalone source API's final expanded all-width normal and shader-validation
rechecks also pass, `section9_p15_staged_sources_final_reference.txt` / `...validation.txt`.
The complete 26/26 normal and 19/19 shader-validation suite gates pass (1238.38 s
and 1975.83 s respectively). Raw logs and source/archive identities are retained in
`section9_p15_staged_gate_metadata.json`. The final benchmark-linked archive also
passes three check-only fixtures: 352-bit dense/reverse/all-steps, 1024-bit resident
17-term per-group fused sources, and 64-bit zero-step/zero-term inputs. No timings
or performance acceptance follow from these correctness checks.

## Benchmark and first measurements

The candidate is rebased on 1.2.0/current main. Metal source and the source/recurrence
encoder bodies are unchanged from the all-width/full-suite-gated implementation.
Focused smoke, three linked benchmark fixtures and version/package checks pass normally
and under shader validation, recorded in `section9_p15_rebase1_2_*`. Historical gate
identities remain separate from the rebuilt 1.2.0-linked archive.

At 352 bits, 1041 lanes, 100 steps, 24 terms and sharing 16 (including a tail), three
interleaved repetitions give the following wall medians in milliseconds:

| Mode | Per-lane GPU | Shared-source GPU | Optimized 18-worker MPFR CPU |
|---|---:|---:|---:|
| Host staging, CPU-interleaved | 700.688 | 31.773 | 53.170 |
| Resident, CPU-interleaved | 710.588 | 31.686 | 52.228 |
| Host staging, verified warm call | 699.637 | 31.638 | 46.399 |
| Resident, verified warm call | 700.287 | 31.687 | 54.516 |

Every initial/warm/timed output matches its independent MPFR sequence. Shared-source
wall time is 22.1–22.4x lower than the legacy per-lane GPU on this fixture. The CPU
already shares Horner sources; it is not a deliberately redundant baseline. Host load
averages are 8.7–11.9, so CPU ratios are busy-host library observations and not accepted
idle-host consumer speedups. Raw samples/ranges and hashes are retained in
`section9_p15_352_1041_100_24_share16_*` and `...measurements_metadata.json`.
Additional widths, larger term/step counts and sharing factors are being measured;
no global backend/default selection follows from one fixture.

`section9_sources_limbforge` compares the legacy per-lane path with the explicit shared
candidate. Its CPU baseline also evaluates each group's sources only once, then applies
sequential MPFR recurrence updates on an 18-worker pool. MPFR objects and packed inputs
are prepared outside timing; source evaluation and every recurrence update are timed on
both CPU and GPU. The CPU uses four rounded products in composed mode and `mpfr_dot`
for each exact fused component. A separate CPU-only check compares that helper with
exact-product/`mpfr_sum` references at all 31 widths, including aliasing, zeros, exact
cancellation and wide exponent gaps (all pass). No GPU or benchmark timings are involved
in that helper check.

GPU path order alternates per repetition, with every initial, warm-up and timed output
checked against MPFR. `--resident` excludes input upload and output download from wall
time; the default includes them using already allocated buffers. MPFR/MPC bridge
conversion is outside this numeric-array benchmark and must be timed separately by the
consumer. CPU timed results are checked again outside timing. Raw sample order/device/wall
values go to stderr; CSV gives medians, quartiles and range. `--warm` performs one verified
untimed call immediately before each sample, labelled `verified-warm-call`; it is not a
claim of continuously warm clocks. The default is labelled `cpu-interleaved`, not a
controlled cold-clock experiment. Scratch bytes are logical p/q + two Horner tables +
one lane-dot table; minimum Metal allocation granularity is excluded.

Build first with `cmake -S . -B build`, then
`cmake --build build --target section9_sources_limbforge -j4`. The target is excluded
from default builds and CTest. Run shape/contract checks before recording measurements:

```sh
./build/section9_sources_limbforge --bits 352 --lanes 513 --steps 3 --terms 3 --share 2 --fused --reverse --all-steps --check-only
./build/section9_sources_limbforge --bits 1024 --lanes 17 --steps 7 --terms 17 --share 7 --resident --per-group --fused --check-only
```

Measure 352/384/448-bit point/column workloads with terms 24/60/100, steps 100/160/250,
and sharing factors 1/16/60, including sharing tails. Start at practical shapes before
attempting the largest workload. Repeated interleaved host/resident runs, load metadata
and solver-size fixture checks remain pending. Keep source tables reusable by other
consumers; the recurrence helper remains rank-one, while Engine provides the other modes.


The initial staged source build had an address-space mismatch on a copied coefficient;
the compile-error log is retained and the device-load expression is corrected.
