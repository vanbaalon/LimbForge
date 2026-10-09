# Offline operation calibration

WolfNum's `BreakEvenTable` stores measured keys and returns unknown for unmeasured
contracts, shapes, widths, worker counts, storage or profiles. It does not install a
backend default or extrapolate. Preserve the documented arithmetic and finalization
contract when applying a recommendation.

`benchmarks/export_section9_operation_break_even.py` accepts completed, checked
GEMM, normal-equation and factor-plus-solve measurement manifests. A manifest must
contain terminal exit 0, source/binary/archive identities, checked CSV row count,
matching raw CSV/log SHA256 hashes, clock mode and sampled load evidence. Failed,
interrupted, modified or incomplete comparisons are rejected. Existing output files
are preserved. `--require-idle` additionally rejects any sampled load average ≥ 4;
current recorded busy-host profiles do not satisfy that consumer acceptance gate.

```sh
python3 benchmarks/export_section9_operation_break_even.py \
  --kind normal \
  --metadata benchmarks/results/section9_p14_measured_352_1100_944_host_interleaved_metadata.json \
  --profile 'WolfNum / Apple M5 Max / Release / measured busy-host library' \
  --out /tmp/my-normal-calibration
```

Repeat `--metadata` to combine recorded profiles. The generated header defines
`section9_normal_break_even()`, `section9_gemm_break_even()` or
`section9_trials_break_even()`, returning the existing `limbforge::BreakEvenTable`.
The accompanying JSON retains keys, timing samples, provenance and recommendations.
Use its exact profile string when querying; machine/build/binary/archive, clock,
CPU contract/active workers and the measured load-series identity are part of it.
A successful laboratory recommendation is not a universal CPU/GPU speed claim.

| Kind | Operation includes | `count,m,n,k` key layout |
|---|---|---|
| GEMM | Selected complex contract, including Gauss/embedding work | batches, rows, columns, inner dimension |
| Normal | Normal matrix AND one RHS | 1, input rows, columns, 1 RHS |
| Trials | Factors AND multiple-RHS solves | trials, matrix order, RHS, 32 exact block width or 0 scalar updates |

A trial CPU time covering factor plus solve must not be paired with `factor_wall`;
there is no factor-only calibration in these files. The current exact-batched trial
record is an unreleased experiment; its key does not make that API available in
production. The source-recurrence format and factory remain handled by
`benchmarks/export_section9_break_even.py`.

The six checked normal profiles yield 24 keys. Three additional trial-baseline keys
retain the explicit experimental scope. Actual generated factories load all keys;
changed width/profile and factor-only operations stay unknown. Idle-host export of
the measured busy data and overwrites are rejected. These artifacts remain offline;
The checked 352-bit Fourier host/resident and 10,000/100,000 resident 4x4
measurements additionally yield 14 GEMM keys in
`benchmarks/results/section9_p13_gemm_break_even_measured.hpp/.json`. The
generated C++ factory checks all 14 keys; changed width, profile and shape stay
unknown. Busy-host idle export, nonterminal/failed records, changed raw hashes
and existing-output overwrites are rejected. Other widths, clock modes, small
host batches and accepted idle-consumer calibration remain pending.
