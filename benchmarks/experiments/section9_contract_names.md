# Section 9 P2.1: explicit contract names

Base `9ea6f3c` / 1.1.0, local branch `round-section9-contract-api`.
Additive inline forwarding names expose exact products, sequential products/trials/solves,
and blocked exact-update factorizations. No old names are deprecated or removed; no
signature/default, shader or arithmetic implementation changes. Names are universal.

The Numerics comparison table includes rounding/accuracy, measured-status limits and
residency/finality. Factor/solve names use `blocked`, avoiding a promise of one rounding
for an entire factorization or solve. A small independent cancellation reference
contrasts exact whole-dot rounding with composed and fused sequential accumulation.

Build and both version/package checks pass (8.91 s). `test_limbforge_contract_names`
compiles all additive host/resident signatures and checks the 64-bit dot
[2^100,1,-2^100] · [1,1,1]: exact result1, composed/fused sequential0. Physical normal
and shader-validation checks both pass (`section9_p2_contract_reference.txt` and
`...validation.txt`). The opt-in target is excluded from default builds/CTest.
Final combined API checks and a compatible minor release remain pending.
