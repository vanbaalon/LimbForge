# Section 9 dense GEMM probes

Base: `10eb888` (API 1.0.0). Apple M5 Max, macOS 26.6.2, Release,
AppleClang 21.0.0.21000101. References replay independent MPFR primitives.

The baseline's 26 CTest suites and all 31 small-shape audit widths passed.
Expanded dense cases exposed complex GEMM errors. Rejected/incomplete candidates were kept separate from main. The private-storage
correction passed the full recorded gates and is included in 1.0.1; selected-width
passes alone were insufficient.
No benchmark has been taken for these candidates.

| Candidate | Evidence | Decision |
|---|---|---|
| Original tiled backend | `section9_p0_all_widths_dense.txt`: fused complex tiled GEMM fails at 192 bits, index 4704; identical-input rerun first fails at index 284 | Corrected by the accepted private-storage backend |
| Private noinline complex fused helper (`section9_fma_only.patch`) | 192-bit dense passes; `section9_p0_fixed_dense.txt`: composed complex 4x4 fails at 352 bits, index 2052 | Rejected |
| Also isolate complex multiply | 352-bit dense and 192-bit validated dense pass; `section9_p0_isolated_all_widths_dense.txt`: fused tiled fails at 448 bits, index 11176 | Rejected |
| Classical `cfma` in the helper | 448-bit dense passes; `section9_p0_classic_spread.txt`: wide-exponent fused case fails at 544 bits; dense also fails in powers at 544 bits | Rejected |
| Complex helpers with reference/output parameters (`section9_reference_abi.patch`) | `section9_p0_544_reference_abi_dense.txt`: composed tiled fails at 544 bits, index 5904 | Rejected |
| Original arithmetic with direct device loads (`section9_global_load.patch`) | 544-bit dense and wide-exponent cases pass; `section9_p0_global_load_all_widths_dense.txt`: composed tiled fails at 608 bits, index 2108; repeat first fails at index 2260 | Rejected |
| Private noinline real primitives, padded multiplication (`section9_real_isolation.patch`) | 608-bit dense passes; `section9_p0_real_isolation_all_dense.txt`: fused accumulated tiled fails at 160 bits, index 6336; focused rerun passes | Rejected |
| One component per thread, alternating components within a SIMD group (`section9_component_parity.patch`) | `section9_p0_component_all_dense.txt`: 30 widths pass; composed 4x4 fails at 960 bits, index 59907 | Rejected |
| Uniform component per SIMD group (`section9_component_grid.patch`) | Interrupted during the dense sweep to run serialized baseline probes; no conclusion | Incomplete experiment |
| Padded private significands, uniform components and explicit packed stores | All 31 dense/wide-exponent and complete expanded references pass normally and under validation; 26/26 CTest + 19/19 GPU validation pass | Accepted correctness fix in 1.0.1 |

The first mismatching entry can change with identical inputs. Some errors affect
only one limb of one component. This suggests a scheduling or code-generation
problem. The coupled private-storage workaround passes the recorded gates, but
the root cause and necessity of each individual change remain unisolated.
Early runs overlapped baseline-suite work. The serialized unchanged baseline
also fails (`section9_p0_baseline_serialized_dense.txt`), so concurrency is not
sufficient to explain the problem. A focused 192-bit serialized trace fails the
fused tiled dot, while all 17 separately dispatched engine primitives and all
17 CPU primitives match MPFR (`section9_p0_192_baseline_serialized_trace.txt`). The CPU core wide-exponent fused replay at
544 bits agrees with MPFR (`section9_p0_544_spread_corecheck.txt`).

The runner now includes three repetitions of dense GEMM (257 tiled or 4,097 small
matrices), dense recurrence/import batches, a 65x65 dense Cholesky factor with
three RHS and failed-trial payloads, and large exponent gaps. Diagnostic options:

```sh
cmake --build build --target test_limbforge_section9_audit -j4
./build/test_limbforge_section9_audit --all-widths --dense --keep-going
./build/test_limbforge_section9_audit --all-widths --spread-only --keep-going
LIMBFORGE_AUDIT_TRACE=1 ./build/test_limbforge_section9_audit --bits 608 --dense --gemm-only
```

`--keep-going` records each selected width and exits nonzero if any failed.
`LIMBFORGE_AUDIT_TRACE` diagnoses the first failed dot by replaying its primitives
in separate engine dispatches and on the CPU. It never suppresses the mismatch;
MPFR remains the acceptance oracle. Patches apply to the base revision above.
