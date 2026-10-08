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

## 6. A second register-band failure (round 19)

Real `fma` with an unpadded 16–32-word addend array gave wrong GPU results from 512 bits (CPU exact,
cancellation case first), while the same code with the addend padded to 33 words is exact at all
31 precisions. Treat runtime-indexed arrays of 13–32 words as a correctness risk, not only a speed
issue, and keep the all-precision GPU fused tests as the gate.

## 7. A third failure: some odd widths of the exact two-term sum (round 23)

The dense-product updates `RN(c - x)` add a wide exact integer `x` (the Garner integer, `LF_TERM_WORDS`,
or the multi-band accumulator, `LF_ACC_WORDS`) and the old entry `c` with `exact_add<N, W>` of `core.hpp`
and round once. The CPU result was exact at every width (20,000 MPFR-checked sums per width), but on the
GPU about 45% of the sums were wrong (typically the top word lost) for some workspace widths `W` and
correct for others, with no simple pattern: with N = 7, `W` = 53, 59, 61, 67, 71, 81 words failed while
51, 52, 54–58, 60, 62, 63, 65, 75, 91, 109 passed. Among the widths the library instantiates, 8 of 62
failed (`acc` at 160, 224 and 352 bits; `term` at 480, 512, 608, 672 and 768 bits). Rounding `W` up to
a multiple of 4 (`sum_words` in `linalg.metal`) gave 0 failures in 8,192 sums per width at all 31
precisions and both instantiations; even widths passed in every probe. Reproducer:
`benchmarks/experiments/sum_width_probe.mm` (argument: rounding multiple, 1 shows the failures).
The fused `fma` of round 19 uses other widths and remains covered by its all-precision tests.

## 8. The 384-bit symmetric square inside new kernels (round 32)

On Metal, core `square()` uses the symmetric product only at 384 bits (round 8). Inside the S4 norm
kernels (`src/numerics.metal`: exponent-scaled squares followed by a threadgroup-memory tree) it
returned a wrong result for every input at 384 bits (`norm2` of one real entry: 200 of 200 random
inputs, wrong exponent and limbs), while 352 and 416 bits, the CPU, and the engine's unary `square`
kernel at 384 bits were exact. Squaring with `mul(a, a)` (the same exact square and rounding) passes
everywhere. Record: `benchmarks/experiments/numerics_core_square.patch`.
## 9. Width-dependent failures of the transcendental kernels (round 35)

The transcendental kernels (`src/transcendental.metal`) evaluate at a working width of `W` words. With
`W = N+2` at every precision, the all-width GPU test (3,000 random points plus hard cases per function and
width, against MPFR/MPC) found wrong GPU results while the same code on the CPU was exact (MPFR-validated,
and its error bounds checked white-box): for `W` = 11, 12, 13, 15, 16 the functions using a `W+2`-word
`fma` (exp, expm1, complex exp and log) were wrong; for `W` = 17, 19–21, 23–25, 27, 28 every function was
wrong, typically from the second limb on (e.g. `sin(0.1)` at 480 bits, `W` = 17; the bound check passed on
the wrong value, so certification cannot catch a miscompiled kernel). `W` = 4–10, 14, 18, 22, 26 and 29–34
passed. The cause was not isolated (making the long-division quotient store read-modify-write changed
nothing). The kernels therefore use the smallest validated width `≥ N+2` (`gpu_words` in
`src/transcendental.mm`; e.g. 288–384 bits use `W` = 14), and the all-width run is the gate
(`benchmarks/results/round35_transcendental_tests_all_widths.txt`). `LIMBFORGE_TRANSCENDENTAL_GUARD_WORDS=g`
forces `W = N+g` for probing; `benchmarks/results/round35_transcendental_bad_widths.txt` is the failing run.

## 10. Cooperative kernels under shader validation (round 41)

The cooperative recurrence (`src/cooperative.metal`, G lanes per trajectory, L = ⌈N/G⌉ limbs per lane) gave
nondeterministic wrong trajectories under `MTL_SHADER_VALIDATION=1` for G = 4/8 (round 17). Tools:
`benchmarks/experiments/coop_validation_probe.mm` (a switchable copy of the kernel, every dispatch compared with the
CPU) and `simd_validation_repro.mm`. Log: `benchmarks/results/round41_coop_validation_probe.txt`.

**A real defect, now fixed.** The rounding decision was `bool up = vote(rb)!=0 && vote(sticky)!=0;`. The `&&`
short-circuits per lane, and `vote(rb)` differs between the lane groups of one SIMD group, so the second
`simd_ballot` ran in divergent control flow. An active-lane counter (`simd_ballot(true)` before every SIMD
operation) counted 226,728 such partial ballots in two 832-bit G=4 dispatches (also at G=8 and G=16, the shipped
kernel; none at G=32). Both ballots now run unconditionally; the counter reads 0 at every G, with and without
validation. Results are unchanged, because the hardware returned the right bits for the active groups.

**Validation-only failures remain after the fix.** Without other GPU work the failures are rare (1 of 4 full
`coop_recurrence` runs). When another process keeps the GPU busy with an unvalidated 1024-bit recurrence, they are
frequent: at 832 bits G=4, 1–15 of 20–40 dispatches.
With validation and load, every exchange formulation of the same arithmetic fails: `simd_shuffle`, threadgroup
memory with `simdgroup_barrier`, and threadgroup memory with `threadgroup_barrier` and no SIMD intrinsic at all.
They do not depend on divergence, lane mapping or early exits:

- every SIMD operation runs with 32 active lanes, also under validation (counter 0);
- `thread_index_in_simdgroup == gid % 32` holds for every thread (counter 0), and indexing from the SIMD-group identity fails as well;
- all shapes have whole SIMD groups and no early return.

Further evidence that the kernel source is not at fault:

- **Data independent.** The inputs are the same in every repeat, but other SIMD groups and steps fail each time.
  The first wrong step follows an exact one, and often the same limb slots of every lane of a SIMD group change at once.
- **Needs device-memory instrumentation.** It passes with all checks off. It fails when only global-memory checking
  or only resource-usage checking is on, and passes when only stack, threadgroup or texture checking is on. It also
  fails with `FAIL_MODE=allow`, and validation reports no fault (`REPORT_TO_STDERR`). Weight limbs re-read with
  atomic loads match the values used, so the loads are right and the in-kernel state is corrupted.
- **Follows lane state, not dispatch length.** Before the fix, 96 extra live words per lane raised the failure rate
  to 90% and made G=8 fail. A 1024-bit G=4 dispatch of 16 steps (11 ms) fails, while G=16/32 dispatches of 85 ms do
  not. The one-thread kernel never failed, even with 256 extra words.
- **Never without validation.** 0 of 26,100 unvalidated dispatches of 16,384 G=4 1024-bit trajectories
  (the worst shape) differ, run concurrently with the validated jobs. 0 of 480 unvalidated 832/1024-bit G=4/8
  dispatches under load, with 0, 32 or 96 extra words, differ (before the fix).

| shapes, fixed source, validation + load | bad dispatches |
|---|---:|
| L ≤ 2 (G=4 to 256 bits, G=8 to 512, G=16/32 everywhere) | 0 of 4,780 |
| G=4, L ≥ 3 (288–1024 bits) | 33 of 960 |
| G=8, L ≥ 3 (544–1024 bits) | 1 of 640 (and 10 of 20 with extra pressure) |

The cause sits inside the validation-instrumented pipeline or its scheduling with concurrent work. The source
cannot be inspected below AIR, and a synthetic shuffle kernel with up to 512 live words did not reproduce it, so
the mechanism is not explained. The library therefore uses only cooperative shapes with L ≤ 2. This is the class
of the G=16/32 kernels shipped in round 22, which never failed in any validated run.

## 11. Retry-rung kernels: width validity and a ceiling at 35 words (round 44)

The GPU retry rungs (`src/transcendental.metal` with `LF_RUNG`) are a second variant of the transcendental kernels:
the element comes from a compacted list and only decided elements are written. Each rung width was probed by forcing
every element through the rung (`test_limbforge_transcendental --force-level r`, 1,000 random points plus the hard
cases per function, all 31 precisions) with the later rungs sent to the host
(`LIMBFORGE_TRANSCENDENTAL_RUNG_WORDS="N:w1/0/0"`), so that a failure is attributed to one (N, W) pair. Without that
isolation, hard cases that the forced rung leaves undecided are evaluated by the next rung, and a bad next width shows
up as a failure of the rung under test (round-1 probes blamed rung 1 at W = 8, 10 for errors of rung 2 at W = 12, 16).
Logs: `benchmarks/results/round44_rung_probes.txt`.

- **W ≤ 35: the section-9 widths fail here too; all others passed.** Rung 3 at W = 16 (N = 2: exp, expm1, complex
  exp/log) and at W = 20, 24, 28 (N = 3–5: every function) was wrong, as in the first pass; the other section-9 widths
  were not instantiated again. Every other width tried passed at every N where it was used: 6–10, 14, 18, 22, 26 and
  29–35 (rung 1 for N = 2–32, rung 2 for N = 2–30, rung 3 for N = 2–15).
- **W ≥ 36: every function is wrong.** W = 36, 37, 38, 40, 44, 48, 52, 56, 60, 64, 68, 100, 136 (rung 2 at N = 16–18,
  rung 3 at N = 7–13, and W = 36–136 at N = 8) failed for every function, with grossly wrong values (`exp(0.0279)` returned
  `1.00000000016`, garbage statuses), while the same code on the CPU is exact (the host ladder uses 48, 68 and 136 words)
  and the constant tables are read correctly. `benchmarks/experiments/transcendental_wide_probe.mm` reduces it to core
  `add`/`sub` with an operand computed by `mul` or `div_word` in the same kernel: `add(1, mul(s, s))`,
  `sub(mul(x, x), s)`, `add(1, div_word(x, 7))` are wrong for every input from W = 36, exact at W = 34 and 35, while
  `mul`, `div_word`, `add` of loaded operands and `div_word(mul(x, x), 7)` are exact. (A plain `mul` at W = 68 is wrong
  in that probe as well.) The engine's arithmetic stops at N = 32 and the first pass at 34 words, so no shipped kernel
  reaches these widths; the cause is not isolated further.
- **Stack limit.** The complex log kernel at 136 words does not compile ("Compute function exceeds available stack
  space"); the other functions compile at 136 words in 1.5–5 s.

The rung widths are therefore the smallest validated width ≥ N+4, 2N+4, 4N+8 (and above the previous level), capped at
35 words; a rung whose capped width is not above the previous level runs on the host (`rung_table` in
`src/transcendental.mm`). A capped rung still decides most retries (35 words exceed the cubic-term need of 3N words up
to 320 bits); correctness never depends on the width, only on `certify`.
## 12. Compile time grows with inlined copies of the exact fma (round 46)

Metal inlines every call, so each call site of `cfma` (two `dot2_add`, each with two products, three window
accumulations and eight inlined `exact_add_ordered` on 132–200-word workspaces at 1024 bits) adds its full body
to the kernel, and the backend compile time grew faster than the number of copies. Cold pipeline compilation
(`benchmarks/experiments/vr_compile_probe.mm`, which injects a nonce so the OS shader cache cannot answer; a
repeated identical source compiles in milliseconds): `complex_fused` (one `cfma`) 5–8 s; fused
`vector_recurrence` matrix form (one rolled `cfma` site) 7–11 s, rank-one (two sites) 16–31 s, tangent (six
sites) 117–207 s at 512 bits and more than 300 s at 768/1024 bits; one 1024-bit specialisation crashed the
compiler service (round 36). With the fused step as one rolled loop around a single `cfma_rolled` (operands
selected by role; the terms, components and both pairwise exact additions also in rolled loops, one
`exact_add_ordered` instance) every one of the 24 fused flag combinations compiles in 0.3–4 s at 256–1024 bits.
Runtime was unchanged or better at 256/384 bits (0.78–1.00×) and 6–8% slower at 1024 bits.

`__attribute__((noinline))` is honoured: on the unchanged `dot2_add` it cut the 512-bit rank-one compile from
16 s to 1.4 s and the tangent compile from 117 s to 2.2 s, with runtime within ±15% of the rolled form
(`benchmarks/results/round46_fused_wide_*`). It was not adopted because it changes the code of every caller of the
shared `dot2_add` (complex fused, segmented dot, LU, numerics, transcendentals), which would all need revalidation.

## Rules for kernel code

1. Do not index arrays of ≤ 32 words with runtime indices in hot loops. Either make every index a
   compile-time constant (full unrolling) or keep the scratch array ≥ 33 words.
2. Do not build a product or remainder column by column (write-only stores after zeroing) and
   consume it in registers. Use read-modify-write accumulation (schoolbook), or disable unrolling
   and validate in complex kernels as well as real ones.
3. Validate every arithmetic change in real, complex, and resident chain kernels on the physical
   GPU at all 31 precisions; a passing raw-product or CPU check proves nothing about fused kernels.
4. Make performance decisions from warm `kernel_sweep` numbers.
5. Round exact-sum workspaces (`exact_add<N, W>`) up to a multiple of 4 words, and probe every width a
   kernel instantiates on the GPU (section 7): a correct width says nothing about its neighbours.
6. In new kernels square with `mul(a, a)`; the 384-bit symmetric `square()` is validated only in the
   engine's unary kernel (section 8).
7. A rigorous error bound computed by the same kernel does not detect a miscompilation (section 9);
   only the comparison with an independent reference at every instantiated width does. Validate each kernel variant
   separately, and isolate the variant under test from later stages that see its leftovers (section 11).
8. Never put a SIMD-group function (shuffle, ballot, `simd_any`/`simd_all`) or a barrier in an operand of `&&`,
   `||` or `?:` that some lanes skip. Evaluate it unconditionally into a variable first (section 10). Check
   cooperative kernels with an active-lane counter (`PROBE_CHECK_ACTIVE` in `coop_validation_probe`).
9. Validate cooperative kernels under `MTL_SHADER_VALIDATION=1` while another process keeps the GPU busy; on an idle
   GPU the section-10 failures are rare. Keep cooperative shapes at ≤ 2 limbs per lane until wider lane state
   passes that test.
10. Do not instantiate the core arithmetic at more than 35 words in a Metal kernel (section 11); probe any wider use with
    `benchmarks/experiments/transcendental_wide_probe.mm` first.
11. When a kernel needs an exact fused routine (`fma`, `dot2_add`, `cfma`) at several places, call it from one
    place: a rolled loop that selects the operands (or a non-inlined function). Measure cold compile time of
    every specialisation at 512–1024 bits with `vr_compile_probe` (section 12).

## 13. Dense section 9 complex GEMM and private significands

The expanded audit of `10eb888` passed all 31 widths at small shapes, but the
serialized dense sweep failed at 11 widths (192, 224, 256, 288, 352, 416, 608,
896, 928, 960 and 1024 bits). First failing positions change with identical
inputs. Errors can affect a few limbs or just a component's sign. The ordinary
26-suite checks and 19 GPU suites under shader validation all passed, so they
were insufficient to detect this problem.

At a failing 192-bit fused tiled dot, every one of its 17 primitives agrees with
MPFR when dispatched separately through Engine; the CPU replay also agrees.
Isolating complex or real helpers, changing their argument/return forms, direct
device loads, and alternating one-component threads were insufficient. Logs and
rejected patches: `benchmarks/experiments/section9_gemm_dense.md`.

The 1.0.1 correction uses one component per thread, uniform component selection
within a SIMD group, and private significands with 33 storage words. CMake derives
a private arithmetic namespace from the shared core, changing only the local
limb array and its zero initialization. **The logical precision remains N words**;
all arithmetic loops and rounding still use N. Public buffers retain their original
packed `Number<N>`/`Complex<N>` layout; explicit field loads/stores bridge the two.
This also makes every private Number occupy 144 bytes. It does not instantiate
arithmetic at 33-word precision.

This correction passed all 31 dense GEMM and wide-exponent widths normally and
under shader validation, and the complete expanded audit in both modes. The rebuilt
library passes 26/26 CTest suites and 19/19 GPU suites under shader validation.
Additional valid/status normal-equation and negative-power sweeps pass at all widths.
The exact compiler defect and the necessity of each part of the workaround have not
been isolated. Do not infer correctness of another kernel from this result.
