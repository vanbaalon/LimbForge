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

The standalone source API's final expanded all-width recheck and the complete 26/19
suite gates remain pending. No benchmark acceptance yet.

The initial staged source build had an address-space mismatch on a copied coefficient;
the compile-error log is retained and the device-load expression is corrected.
