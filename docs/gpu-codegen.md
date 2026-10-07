# GPU code generation: findings and kernel rules

Round 13 (plan items A1/A2) measured warm-clock kernel costs at all 31 precisions and probed the
GPU-only failures behind the rejected Comba, top-bit, and full-width square experiments. Tools:
`benchmarks/kernel_sweep.cpp` and `tests/gpu_codegen_probe.mm` with `tests/probe_variants.hpp`. Raw
data: `benchmarks/results/round13_*`. Device: Apple M5 Max, macOS 26.6.2.

## 1. Cold clocks inflate single-dispatch device times

A single dispatch after 100 ms of GPU idleness takes 2.3× (add), 3.0× (mul), and 4.0× (complex
multiply/divide) longer than the same dispatch inside a busy command buffer (medians over all
precisions). Earlier benchmark device columns were measured after CPU/MPFR phases and are cold.
Use `kernel_sweep` (warm, 32 dispatches per command buffer) for kernel decisions; report cold and
warm numbers separately. Warm `add` reaches 370–550 GB/s and is bandwidth-limited.

## 2. Small register arrays with runtime indices are slow

Real multiplication and division showed a cliff: at 288–480 bits (N = 9–15 limbs) they were 3–5×
slower than at 512 bits, e.g. warm 384-bit `mul` 101 µs vs 512-bit 42 µs for 65,536 values. The raw
product kernel without rounding shows the same cliff, and `#pragma clang loop unroll(disable)` or
outer-loop-only rolling does not remove it.

Cause (by experiment): private arrays of roughly ≤ 32 words are promoted to registers; a
runtime index such as `product[i+j]` then costs a select chain over the whole array. From N = 16 the
2N+1-word product no longer fits and stays in fast private memory. Two remedies restore speed at
every affected width, with identical results:

| bits | schoolbook `mul` (µs) | scratch padded to ≥ 33 words | both loops fully unrolled |
|---:|---:|---:|---:|
| 256 | 21.5 | 18.1 | 18.5 |
| 288 | 42.6 | 25.1 | 23.6 |
| 384 | 104.2 | 35.3 | 35.4 |
| 448 | 172.6 | 41.5 | 42.9 |
| 480 | 206.9 | 44.4 | 47.1 |
| 512 | 42.1 | 43.0 | 40.8 |

Division's `u[2N+1]` window has the same cliff (warm 480-bit `div` 250 µs vs 512-bit 74 µs).

## 3. GPU-only wrong results: lost stores in column-accumulated products

All product formulations write bit-exact unrounded products when the product is stored straight to
device memory. Wrong results appear only when the product array is consumed in registers:

- Comba (64-bit accumulator, 32-bit-word accumulator, or `mulhi` accumulator): rounded `mul` fails
  for every input at N ≥ 25; complex multiplication fails for every input at N ≥ 3.
- Writing each word once without prior zeroing fixes `mul` through 800 bits but not 1024 bits or
  complex multiplication.
- Fully unrolled Comba fails in complex multiplication at N ≤ 15 and passes at N ≥ 16.
- Comba with `unroll(disable)` on both loops, and both schoolbook forms, pass in every context.

The failing 1 × 1 product rounds to zero: the loads in `pack` observe the words that existed before
the column loop's stores (zero or stale data from another inlined product). The fault is in the
Metal compiler's handling of these stores (the same source is exact on the CPU; ASan/UBSan and
pattern-initialised locals find nothing in the core). It is independent of the
`optimizationLevel`. A standalone reproducer is `gpu_codegen_probe --bits 800 --variant comba64 --dump`.

## 4. 32-bit products are not faster

Schoolbook built from `x*y` and `mulhi(x,y)` with explicit 32-bit carries is 0–15% slower than the
existing 64-bit form. The compiler already maps 32×32→64-bit products well; plan item B1 is dropped.
Rolled Comba is slower than schoolbook except for 384-bit `mul`, so B2 is not pursued.

## 5. One recurrence thread is latency-bound

`recurrence` costs ~165 µs per step at 384 bits regardless of lane count up to 4,096 lanes;
throughput saturates near 16,384 lanes (12 ns per lane-step). Callers with ≈ 128 trajectories use
under 1% of the GPU: batching more trajectories per call is nearly free.

## Rules for kernel code

1. Do not index arrays of ≤ 32 words with runtime indices in hot loops. Either make every index a
   compile-time constant (full unrolling) or keep the scratch array ≥ 33 words.
2. Do not build a product or remainder column by column (write-only stores after zeroing) and
   consume it in registers. Use read-modify-write accumulation (schoolbook), or disable unrolling
   and validate in complex kernels as well as real ones.
3. Validate every arithmetic change in real, complex, and resident chain kernels on the physical
   GPU at all 31 precisions; a passing raw-product or CPU check proves nothing about fused kernels.
4. Make performance decisions from warm `kernel_sweep` numbers.
