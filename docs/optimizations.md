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

Cooperative SIMD arithmetic, alternate layouts, workgroup tuning, specialized
squaring, and fused primitives remain candidates. Evaluate one change per round,
retain a reliable fallback, and distinguish small-batch latency from large-batch
throughput. New primitives need their own explicit rounding contracts and MPFR
references before performance tuning.
