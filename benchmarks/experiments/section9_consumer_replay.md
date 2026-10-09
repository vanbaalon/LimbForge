# Section 9 P3: verified current consumer replay

The external qscmx consumer now builds matching WolfNum 1.3.1 (`78f7da6`) in
`build-wolfnum-hp`, ten inline limbs. This record archives existing evidence and
independently checks its source/archive/binary identities, raw serialized histories,
Delta components, residuals and GPU/adjoint log markers. It performs no new GPU
run, changes no consumer source, and accepts no busy-host timing.

| Fixture | Delta component difference | History relative difference | Both residuals | Input gtol | Acceptance |
|---|---:|---:|---:|---:|---|
| J=3 X²Y, g=.202, Nc21/NQ25/Nsh300/nPts27, 374 bits | 7.10245197342712e-125 | 0, three entries | 5.57752179966997436e-25 | 1e-22 | Pass for this fixture |
| Historical J=2, g=.2, Nc15/NQ19/Nsh100/nPts21, 352 bits | 3.0523092452e-118 | 0, five entries | 5.34350971445300872e-14 | 1e-25 | Not converged |

All three existing GPU paths execute without fallback and adjoint checks pass.
The combined exact-normal candidate is disabled. The two-fixture command's exit 2
is correct: one row remains unconverged. The J=3 row does not establish g=.1/.2/.5
coverage, new-switch convergence or idle timing, and its tolerance is stated above.

The archived [independent verification](../results/section9_p3_wolfnum_1_3_1_consumer/independent_record_check.json)
links the raw inputs, outputs, CPU/GPU logs, original metadata and comparison reader
through SHA256 identities. [Original comparison](../results/section9_p3_wolfnum_1_3_1_consumer/comparison.json)
and [consumer identity](../results/section9_p3_wolfnum_1_3_1_consumer/original_metadata.json)
retain their original bytes and paths. Original binaries and user jobs are untouched.
