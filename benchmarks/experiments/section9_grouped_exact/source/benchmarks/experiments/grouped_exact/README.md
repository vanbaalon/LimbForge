# Grouped exact trailing-update experiment

Based on main `f9be84b` plus the retained exact-trial/look-ahead prototype,
rebased without changing its rounding sequence in `31598e7`. This is an isolated
candidate, outside the released WolfNum 1.3.1 API.

`WOLFNUM_EXACT_TRIAL_SCHEDULE=grouped` uses the look-ahead panel schedule but
packs independent trial inputs/outputs contiguously. A private host-ready hook
classifies all lines. If every line has exactly one exponent band, common digit,
residue-product and reconstruction dispatches cover all trials through their z
coordinate. The modulus count covers the widest band of every trial. Different
exponents and mantissas remain independent. The generated reconstruction keeps
the existing single-band rounding expression and omits the unreachable multi-band
accumulator branch. Each result is `RN(C - exact sum_k A[k,i]*A[k,j])`.

This hook is restricted to buffers freshly packed on the host, with no preceding
GPU producer. It is not a general resident API. Any zero/status/fallback/multiple-
band line declines the entire fast path before encoding; the factor driver then
uses its existing resident exact SYRK calls and wait-time repairs. The driver
preserves completed look-ahead columns and failed trials' invalid tails.

`generate_kernels.py` reproduces `src/linalg_grouped.metal` from the unchanged
`src/linalg.metal`. It records the base source hash and changes only trial
address/grid offsets and the proven single-band reconstruction selection.
Original pipelines and source remain unchanged. The unreleased `BlockedTrialsInfo`
prototype adds a grouped-update count; benchmark profiles report it explicitly. Per-call grouped workspace stays
alive until the batch completes, including after workspace release or abandonment.
The combined residue buffer grows with trial count; memory must be measured as
well as runtime before any acceptance.

Build the excluded `test_wolfnum_grouped_syrk` and `test_limbforge_blocked_trials`
targets. The first uses independent MPFR exact references at each width for three
trial shapes (including 131x65 dense updates), cancellation, varying exponent
ranges, old-output statuses, tails, two calls per batch, busy workspace release,
and all-or-nothing fast-path rejection. The factor harness independently replays
blocked panels, mixed/late pivot failures and exact fallbacks. Focused 352-bit
normal/Metal checks passed for the initial grouped prototype; the final frozen
snapshot still needs complete all-width and broad gates before timing.

No performance claim, cutoff, production default or installed API is accepted.
The original 3x factor target and full 352/384/448 x 200/400/944 grid remain open.
