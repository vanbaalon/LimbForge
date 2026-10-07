# Optimization rounds

The target is a standalone multiprecision library on Apple Metal. Acceptance
requires independent MPFR agreement on both CPU and GPU, followed by the same
benchmark matrix. A faster-looking result with an arithmetic mismatch is rejected.

The original arithmetic baseline is revision `773329f`; the benchmark harness and
its frozen measurements are committed in `8fbcc79`. The moderate-exponent
benchmark inputs do not exercise the baseline's known exponent-boundary bug.

| Round | Change | Decision and evidence |
|---|---|---|
| 1 | Fix rounding into the minimum supported exponent; constrain template precision | Accepted, `c706d1a`. All 31 precisions pass, including adversarial products immediately below a binade boundary. No speedup claimed. |
| 2 | Typed resident buffers, asynchronous tickets, one command buffer for a dependent chain | Accepted, `07c934d`. Ownership/lifetime tests and full MPFR benchmark matrix pass. Copies and repeated waits are amortized across 16 operations. |
| 3 | One exact addition workspace; use `N+2` limbs for exponent gaps up to 32 | Accepted, `2172965`. Preserve every aligned bit and cancellation behavior. Dense complex GPU regression added. |
| 3 trial | Derive the product's highest bit from normalization instead of scanning | Rejected. The large 1024-bit complex-division benchmark fails on the GPU, despite CPU agreement and passing smaller tests. |
| 4 | Reuse an integer reciprocal for exact two-limb quotient estimates | Accepted, `7f25fe5`. One million quotient/remainder checks, all-precision MPFR checks, dense GPU tests, and full matrix pass. |
| 5 trials | 96-bit Comba multiplication; restrict it to smaller widths; express carries using bounded sums | All rejected, recorded in `fccf54d`. Each passes CPU checks but fails physical GPU checks. The runner stops before benchmarking. Schoolbook multiplication is retained. |
| 6 | Interleaved workgroup sweep and configurable SIMD-aligned group sizes | Accepted, `0d3cbf7`. Five sizes, three precisions, three counts, seven operations: 315 MPFR-checked rows, with a repeat sweep. Heavy complex kernels use one SIMD group by default. Gains vary by case; no universal policy speedup claimed. |
| 7 | Word-plane (SoA) layout probe | Experimental harness retained, `db959a2`; public AoS layout retained. Mixed modest device gains and wall regressions; conversions excluded from the probe. |
| 8 | Unary square with exact symmetric integer products | Accepted, `bea92b5` / `50ae317`, selected only at 384 bits on the GPU. Other widths reuse schoolbook multiplication. All-precision tests and 162-case matrix pass. |
| 8 trials | Full-width symmetric square and its use in complex division | Rejected: GPU square fails at 992 bits; complex division fails at 384 bits. CPU agreement is insufficient. |
| 9 | Correctly rounded integer restoring square root | Accepted, `7127a70`. Exact residual controls rounding; negative-domain/status and near-midpoint fixtures pass at all 31 precisions. Full 180-case matrix passes. This is a correctness baseline; multicore MPFR can be faster. |
| 10 | Fixed-order resident real/complex tree sums | Accepted, `ba14968`. Every precision, empty/odd inputs, cancellation, overflow, barriers, and scratch lifetimes validated. 180 ordinary plus 36 reduction benchmark rows. Many small dispatches limit performance. |
| 11 | Cooperative shared-memory reduction with the same tree | Numerics validated, `46e1a82`: all-precision checks, 180 ordinary and 36 reduction rows, Metal validation, and two 36-row interleaved comparisons. Full power-of-two groups retain adjacent-pair order. Wider complex performance needs a fallback. |
| 12 | Select the measured reduction policy | Accepted, `76e997b`. Cooperative real reductions and complex reductions through 384 bits; wider complex and inputs of size 0–2 use the global path. The 1024-bit complex cooperative trial was slower in both interleaved rechecks and is excluded from the default. |
| 13 | Warm-clock sweep (`kernel_sweep`) and GPU codegen probe (`gpu_codegen_probe`) | Measurement and diagnosis, no library change; see [gpu-codegen.md](gpu-codegen.md). Cold single dispatches are 2–4× slower than warm. Runtime-indexed arrays of ≤ 32 words cause a 3–5× multiply/divide cliff at 288–480 bits. Comba failures are lost stores when a column-built product is consumed in registers; 32-bit `mulhi` products are not faster. |
| 13-D1 bridge | Direct limb-copy MPFR/MPC bridge, array and multithreaded versions (CPU only) | Accepted (round 13-D1). Fast paths match the retained `mpz` path bit for bit, including the inexact flag, with 20.2M checks: 31 precisions, source and target precisions below, at, and above `bits`, ties, carries into the next binade, specials, ±1e9 exponents, four MPFR exponent ranges, custom-allocated numbers, and arrays. All 11 injected fast-path mutants fail the suite. With 65,536 values, scalar conversion is 20–28× faster from MPFR and 8–15× faster to MPFR (`benchmarks/results/round13_bridge.csv`). Array conversion takes 2–4 ns/value on 3–16 threads. The machine was loaded (load average ≈ 41), so read the ratios rather than absolute times. |
| 14 | Pad runtime-indexed scratch arrays of 13–32 words on Metal (`scratch()` in `core.hpp`) | Accepted. Real `mul`, `square`, and `div` keep the product/remainder out of register promotion at 192–480 bits; the addition workspace is padded from 512 bits; complex compositions keep the previous layout. Interleaved A/B against `703751d` (two repeats, warm `kernel_sweep`, heavily loaded host): `mul` 1.6–5.4×, `square` 1.8–7.6×, `div` 1.7–4.0× at 192–480 bits; `add`/`sub` 1.2–1.8× and recurrence 1.35× at 512–960 bits; other cases within noise. Earlier policies that also padded complex kernels or small additions regressed some complex/add cases by 10–30% and were not kept. All suites pass, also under Metal validation. |
| 15 | Fused `fma`/`fms`/`complex_fma`/`complex_fms` (plan D2) | Accepted. One rounding per real component (contract in `numerics.md`). CPU, GPU, and resident in-place chains match `mpfr_fma`/`mpfr_fms` and exact-product `mpfr_sum` references at all 31 precisions, including ties, near-ties, deep cancellation, products cancelled by the addend, and intermediate exponent overflow. Two injected faults (eager sticky collapse; unordered three-term sum) are caught. Performance tuning pending. |
| 16 | TensorOps integer-matmul capability probe (plan L1) | Diagnostic, no library change. `matmul2d` int8/uint8→int32 is exact against 64-bit CPU sums; int32 overflow wraps mod 2³²; 29–54 int8 TOPS at 2048³–4096³ on a loaded host. Gate for a residue-number GEMM passed; next steps L1b/L1c in the plan. |
| 18 | Exact integer GEMM through int8 TensorOps residues (plan L1b) | Experiment, not yet a library API. 256-bit signed integer matrices, 33 moduli, Garner reconstruction on the GPU; equal to GMP on all checked outputs. 14–20× faster than an untiled composed `mul`+`add` limb GEMM at n = 512–1024 (computing a different, rounded quantity); reconstruction and residue conversion are now the largest stages. |

## Exact arithmetic retained

Addition uses a single exact aligned integer, obtaining the second operand's
limbs as needed. For nearby exponents, `N+2` words hold both operands and their
possible carry; large gaps retain the wider exact alignment. There is no
approximate sticky-bit subtraction under cancellation.

Division computes an integer reciprocal once per divisor, corrects each
quotient estimate, and retains the existing Knuth correction, subtraction,
add-back, and exact remainder comparison. The final rounding decision still
compares twice the remainder with the full divisor. No floating-point reciprocal
or approximate convergence criterion enters the result.

Resident execution changes scheduling and storage ownership. Each arithmetic
operation still rounds separately. Buffer barriers preserve dependencies, and
submissions retain their buffers until completion.

## New measurement harnesses

- `tune_limbforge`: interleaves 32, 64, 128, 256, and 512 threads per group; every
  dispatch is checked against MPFR.
- `layout_limbforge`: compares AoS and word-plane storage with the same arithmetic
  core. Uploads and layout conversion are outside the timing; results alone do
  not justify changing the public representation.
- `square_limbforge`: interleaves unary square and `mul(x,x)` using resident data.
  At 384 bits / 65,536 values, the selected square has about a 1.29× device
  speedup; wall time is effectively unchanged in that comparison. At other widths
  the selected GPU square uses the same multiplication specialization.
- `reduction_limbforge`: interleaves serial MPFR, pooled MPFR, resident GPU, and
  GPU with transfers. All methods follow the same tree and every sample is
  checked. `--global-tree` selects the original reduction implementation;
  `--compare` interleaves it with the selected policy.

Raw sweeps, repeat measurements, and test logs are in
[benchmarks/results](../benchmarks/results). Main pointwise/chain matrices are
144 rows before square, 162 with square, and 180 with square root. From Round 10
the runner adds a separate 36-row real/complex reduction matrix.

## Rejected experiments

Patches under [benchmarks/experiments](../benchmarks/experiments) preserve the trial
implementations. Raw failure logs are under [benchmarks/results](../benchmarks/results).
The Comba arithmetic passes the independent CPU reference but fails the Metal
execution checks, including a simple real multiplication at 800 bits and complex
arithmetic at 128 bits in the restricted variant. The precise GPU failure cause
is unresolved. These implementations are excluded from the public library;
passing host tests alone is insufficient evidence to accept them.

## Repeat the workflow

```sh
python3 benchmarks/run_round.py my-round
python3 benchmarks/compare.py benchmarks/results/baseline.csv \
  benchmarks/results/my-round.csv --count 65536
```

Use a fresh label: existing records are never overwritten. The runner builds,
runs CTest, and only then benchmarks. It records revision, working-tree state,
platform, command, logs, and pass/fail status. The compare tool matches precision,
operation, batch size, and step count; a ratio above one means the new run is faster.

## Further experiments

Cooperative limb arithmetic within a number, a single-round fused multiply-add,
faster square root with exact certification, and transcendental functions remain
unfinished. The cooperative reduction parallelizes the summation tree; it does
not distribute the limbs of one multiplication or division across SIMD lanes.
Alternate layouts remain experimental. Broader GPU validation is also pending.

Evaluate one change per round, retain a reliable fallback, and distinguish
small-batch latency from large-batch throughput. New primitives need explicit
rounding contracts and independent references before performance tuning.
