# Experiments

Research prototypes that are not (yet) library features. Each records its design, correctness evidence,
measurements, and a recommendation. Raw data lives in `benchmarks/results/`.

## 17-L4: cooperative limb arithmetic for latency-bound recurrences

Plan items E1 / L4 (`optimization-plan.md`). Tool: `benchmarks/coop_recurrence.mm` (target `coop_recurrence`).
Data: `benchmarks/results/round17_coop_recurrence.csv` and `_metadata.txt`. Device: Apple M5 Max, macOS 26.6.2.

### Problem

The `recurrence` kernel runs one thread per trajectory. Each step does 4 complex multiplications and 4 complex
additions, which is 16 real `mul` and 16 real `add`. Below about 4,096 trajectories, per-step time does not depend on
the lane count (165 µs at 384 bits), so BSolver (about 128 lanes × 600 steps) uses less than 1% of the GPU.

### Design

`recurrence_coopG` gives each trajectory G ∈ {4, 8, 16, 32} lanes of one 32-wide SIMD group. Each lane holds
L = ⌈N/G⌉ limbs. A number is top-aligned in M = G·L limb slots; the M−N low slots are zero. The exponent, sign and
status are replicated on every lane of the group.

- **Exact product (2M slots).** Lane k owns blocks k ("lo") and k+G ("hi") of the workspace. Iteration t multiplies
  the broadcast block B_t by A_{(k−t) mod G}, both fetched with `simd_shuffle`. The L×L block product lands at block k
  when t ≤ k, otherwise at block k+G. Every lane therefore does the same G−T0 block products (T0 counts the all-zero
  low blocks), accumulated read-modify-write as in gpu-codegen rule 2. Overlapping accumulator tops are folded in
  from the neighbouring lane with two shuffles. A final 0/1 carry chain is resolved by one generate/propagate scan.
- **Carry scan.** Each block reports generate g and propagate p (all ones) through `simd_ballot`. The group's
  2G-bit masks give every block's carry-in as `((g|p) + g + c0) ^ (g|p) ^ g`. This is a prefix computation done
  with one integer addition, so there is no ripple across lanes.
- **Addition.** Order the operands by magnitude (ballot plus a shuffle from the highest differing lane). Shift A
  right by one bit to leave carry headroom. Align B by gap+1 bits with a two-stage shuffle shifter (whole blocks,
  then limbs and bits from the neighbour block). B's lost bits are jammed into bit 0, at least 32N−1 bits below A's
  lsb, so the sticky collapse cannot move a rounding boundary. Then add or subtract with the scan. A gap of more
  than 32N+2 returns A, as in the core.
- **Rounding (`pack`).** Find the top nonzero block by ballot, and its top bit with `clz` broadcast from that lane.
  Normalise with the shifter; for products the shift is known to be 0 or 1 bit. Gather the round, sticky and lsb
  bits with two ballots and round to nearest, ties to even. The increment carry uses the scan. Exponent-range checks
  match `pack`/`checked` exactly.
- **Composition.** `cmul = {sub(mul,mul), add(mul,mul)}` and `cadd` use the same step order and weight indexing as
  `recurrence`. The result is therefore the same composed-rounding value, bit for bit.

Every branch that contains a shuffle or ballot is uniform across the whole SIMD group (`simd_any`/`simd_all`).
Group-specific special cases (zeros, status, far gaps, exponent overflow) are computed alongside and selected at the
end.

### Correctness

The test data covers real add, sub and mul. It has 4,096 random pairs plus cancellation pairs (exact cancellation,
1 to N−1 equal top limbs, ±1 ulp) and alignment gaps of 1–65 and 32N−33 to 32N+40 in both signs. It also has ties,
all-ones and power-of-two operands, carry-out and rounding overflow, 2,048 structured-limb pairs (0, 1, ~0, …) for
long carry chains, zeros, status values, and exponent overflow in both directions.

Complex add and mul are tested on random and structured inputs. Four recurrence configurations are tested:

- the `tests/test_arithmetic.cpp` data (seed exponent span 5, weight span 3 shifted by −5; 17×31, per-lane weights)
- replicated seeds with shared weights (68×31)
- 256×64
- 96×200 with shared weights

Each result is compared bit for bit with the core (CPU) and with the existing GPU `recurrence` kernel.

- **0 mismatches** at all 31 precisions for every G.
- The final output of every timed configuration (up to 32,768 trajectories × 64 steps) is also `memcmp`-equal to the
  existing kernel's output. The harness now compares every timed dispatch.
- Ten injected arithmetic faults are each detected, including dropped carries, round-half-up, no sticky jam, a wrong
  far-gap cutoff and a wrong compare lane. Two of them (product carry scan, h-carry) were caught only after the
  structured-limb cases were added; random limbs almost never make long carry chains.

**Shader-validation finding (unresolved).** With `MTL_SHADER_VALIDATION=1`, G=4 recurrences from 512 bits upward
(L ≥ 4 limbs per lane) give wrong results for a few trajectories, often one whole SIMD group:

- **Nondeterministic.** The failing widths change between runs: one full run failed at 512–832 bits and passed at
  896–1024 bits; earlier runs failed at 896 bits and failed 1024 bits in 5 of 5 runs.
- **G=8.** It failed once (768 bits, L=3).
- **G=16 and G=32.** They never failed under validation.
- **Unit kernels.** The real and complex unit kernels pass.

Without validation nothing fails:

- every run passes, and so does the API debug layer alone;
- a dedicated stress run on the failing shapes found 0 of 2,216 dispatches different from the existing kernel
  (512–1024 bits, 128 and 2,048 trajectories, G=4/8, every dispatch compared).

The first prototype placed shuffles in branches that were uniform per group but divergent across the groups of a
SIMD (`benchmarks/experiments/coop_divergent_branches.patch`). It failed the same way under validation, and with
`MTL_DEBUG_LAYER=1` added it also failed one SIMD group of the 1024-bit G=4 `cmul` unit kernel. Making every
SIMD-op branch SIMD-uniform did not remove the G=4 failures.

The failures follow per-lane state size: none at L ≤ 2, rare at L=3, frequent at L ≥ 4. This points to
validation-instrumented code under heavy register pressure (spills) around SIMD operations. A latent compiler
defect that is merely rarer without instrumentation has not been ruled out. Treat G=4/8 shapes with L ≥ 3 as
unvalidated until the cause is found (for example with a reduced reproducer filed with Apple).

### Latency (µs per step, warm, median of 9 interleaved samples, repeat 1; repeat 2 agrees within ~5%)

| bits | trajectories | existing | G=4 | G=8 | G=16 | G=32 | best speed-up |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 32 | 102.8 | 47.8 | 37.9 | 38.6 | **29.7** | 3.5× |
| 256 | 128 | 103.2 | 49.2 | 38.4 | 41.4 | **33.0** | 3.1× |
| 256 | 512 | 106.4 | 49.2 | **41.9** | 42.3 | 45.2 | 2.5× |
| 256 | 2048 | 115.2 | **52.3** | 57.4 | 88.4 | 151.4 | 2.2× |
| 256 | 8192 | 171.9 | 173.0 | 275.2 | 516.5 | 872.1 | 1.0× (tie) |
| 256 | 16384 | 204.3 | 319.5 | 492.4 | 990.3 | 1592.5 | 0.64× |
| 384 | 32 | 162.9 | 58.9 | 39.1 | 29.7 | **27.2** | 6.0× |
| 384 | 128 | 164.1 | 62.0 | 39.2 | 30.4 | **27.6** | 6.0× |
| 384 | 512 | 169.2 | 62.1 | 44.2 | **32.4** | 53.6 | 5.2× |
| 384 | 2048 | 176.3 | **64.0** | 70.4 | 98.9 | 174.8 | 2.8× |
| 384 | 8192 | 239.9 | 228.5 | 308.6 | 497.5 | 902.8 | 1.05× (tie) |
| 384 | 16384 | 265.8 | 393.3 | 579.4 | 928.5 | 1725.9 | 0.68× |
| 1024 | 32 | 554.8 | 287.2 | 91.7 | 50.5 | **41.0** | 13.5× |
| 1024 | 128 | 559.8 | 290.5 | 94.6 | 54.3 | **41.3** | 13.5× |
| 1024 | 512 | 560.7 | 291.7 | 94.6 | **66.2** | 93.8 | 8.5× |
| 1024 | 2048 | 603.3 | 443.9 | 191.1 | **190.5** | 319.1 | 3.2× |
| 1024 | 8192 | 952.5 | 1571.7 | **789.6** | 912.5 | 1680.4 | 1.2× |
| 1024 | 16384 | 1381.8 | 3111.8 | 1601.7 | 1846.4 | 3204.5 | 0.86× |

The CSV also has 32,768 trajectories, where every cooperative variant is slower (best 0.68×, 0.72× and 0.85×).

### Findings

1. **The cooperative kernel wins wherever the existing kernel is latency-bound.** For 32–512 trajectories it is
   3.1–3.5× faster at 256 bits, 5.2–6.0× at 384 bits and 8.5–13.5× at 1024 bits. For BSolver's ≈128 lanes at
   384 bits, a step drops from 164 µs to 28 µs.
2. **Crossover: about 8,192 trajectories.** At 2,048 the best G still wins by 2.2–3.2×. At 8,192 the result is a tie
   (256 and 384 bits) or a 1.2× win (1024 bits, G=8). From 16,384 the thread-per-trajectory kernel is faster
   (0.64–0.86×), because the cooperative form spends extra instructions on shuffles, ballots and padded slots once
   the GPU is full.
3. **Best G depends on the number of busy threads (trajectories × G).** The optimum lies at roughly 1–16 K threads:
   G=32 up to 128 trajectories, G=16 at about 512, G=4 (≤ 384 bits) or G=8/16 (1024 bits) at about 2,048. G=4 is
   poor at 1024 bits, where 8 limbs per lane cause heavy register pressure and give the slowest cooperative form.
4. Threadgroup size 32 is as fast as or faster than 128.

### Recommendation

Adopt as a shape-selected backend for `Engine::recurrence`, in two stages:

1. **Now: G=16/32 only.** These shapes have L ≤ 2 and are clean under every check, including shader validation. Use
   G=32 for ≤ 128 trajectories and G=16 up to about 1,024 trajectories, and keep the existing kernel above that.
   This covers the BSolver shape (5–13× at 128 trajectories) and gives up little: at 2,048 trajectories G=16 still
   gives 1.3× (256 bits), 1.8× (384) and 3.2× (1024).
2. **Later: G=4/8 for 1,024–8,192 trajectories** (2.2–2.8× at 2,048 trajectories, ≤ 384 bits), only after the
   shader-validation discrepancy is root-caused or shown to be a validation-layer artefact.

When moving the code into `kernels.metal` behind the existing API:

- run the same bit-for-bit tests through `Engine`;
- add the structured-limb and gap cases used here to CTest;
- run the GPU suites under validation as usual.

The same building blocks (product, scan, shifter, pack) are the natural basis for cooperative `div` and fused
operations.
