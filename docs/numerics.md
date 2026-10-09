# Numerical representation and recurrences

## Numeric contract

Every nonzero value has a normalized unsigned `bits`-bit significand, sign ±1,
and a separate binary exponent:

```
value = sign * integer(limb) * 2^(exponent - (bits - 1))
```

Limbs are little-endian 32-bit words. Valid exponents range from -1,000,000,000
to +1,000,000,000. Real operations use round-to-nearest, ties-to-even. Zero is
canonical, with all fields zero. There are no NaNs, infinities, signed zeros,
or subnormal encodings. Invalid conversion or square-root domain, division by zero, and exponent
overflow produce a zero payload with explicit status bits; further real
operations propagate those bits. Host dispatch failures throw exceptions.

| Status | Value | Meaning |
|---|---:|---|
| `ok` | 0 | Valid result |
| `division_by_zero` | 1 | Zero divisor |
| `exponent_overflow` | 2 | Result outside the supported exponent range |
| `invalid` | 4 | Conversion of a non-finite MPFR value or square root of a negative input |

Statuses are a bitmask. Each real component has its own status; check both
components of complex results. `to_mpfr` throws for an error status. For a
nonzero raw input, set the high bit of the top limb, sign to ±1, and the exponent
within the documented range. Raw payloads that violate this contract are not
validated inside every arithmetic kernel.

Complex multiplication/division compose the rounded real operations in
`core.hpp`, with a separately rounded result after each real primitive. They do **not**
promise a single correctly rounded MPC complex operation. The recurrence kernel
preserves the documented sequential accumulation order; it does not fuse away
real rounding steps. Construct raw values only according to this contract.

The same core header works on the CPU and inside a Metal shader. Include
`mpfr_bridge.hpp` only on the CPU. The raw real struct occupies `bits/8 + 12`
bytes; a complex struct occupies twice that. Input and output buffers must match
the requested precision and operation.

## MPFR and MPC bridge

`from_mpfr<bits>(x)` rounds to nearest, ties to even, to `bits`; NaN and
infinity give `invalid`, either zero gives canonical zero, and a rounded
LimbForge exponent (MPFR exponent − 1) outside ±1,000,000,000 gives
`exponent_overflow`. `to_mpfr<bits>(out, x)` is exact when
`mpfr_get_prec(out) >= bits` and otherwise rounds to nearest, ties to even, to
the output precision; zero becomes +0 and an error status throws. Both set the
MPFR inexact flag exactly when they round.

On little-endian 64-bit-limb GMP builds (arm64 macOS), a 64-bit MPFR limb is two
LimbForge words, low half first. The significand is therefore copied by
`memcpy` between the top words of MPFR's left-aligned significand and
`limb[]`; lower MPFR words are zero-filled. A wider MPFR input is rounded from
its discarded words directly; a narrower output is rounded at bit position
`prec`. The value is then installed with `mpfr_custom_init_set`, which also
works for numbers with `mpfr_custom_init` storage. These fast paths agree bit
for bit, including flags, with the former `mpz` conversion, which is retained
as `detail::from_mpfr_slow` / `detail::to_mpfr_slow`. Cases that touch the
current MPFR exponent range use that path: an input or result outside
`[emin, emax]`, a rounding carry at `emax`, or a range that excludes `bits`.

`from_mpfr_array`, `to_mpfr_array`, `from_mpc_array`, and `to_mpc_array`
convert contiguous `mpfr_t[]` / `mpc_t[]` arrays. Elements match the scalar
calls. With `threads=0`, each thread receives at least `2^22/bits` values
(`2^21/bits` for complex), with a single thread below twice that. Worker
threads adopt the caller's MPFR exponent range, which is thread-local in MPFR.
If an element has an error status, the array call rethrows the lowest failing
chunk's exception after all chunks finish; other chunks may already be written.
`from_mpc` / `to_mpc` convert `Complex<bits/32>`; `to_mpc` checks both statuses
before writing. The MPC functions appear when `<mpc.h>` is on the include path,
need no libmpc symbols, and are disabled by `LIMBFORGE_NO_MPC`.

Runtime-width scalar and array overloads accept `bits` in [64,1024], divisible by 32.
`bridge_element_bytes(bits,complex)` gives the packed element size; provide sufficient
capacity. Unaligned byte storage is supported by memcpy. Rounding, statuses, worker selection
and error behavior match the typed bridge. External inline MPC import requires source precision
already equal to `bits`, uses the same zero/non-finite/exponent conversion rules, and writes
ordinary `Complex` values; see [its storage/lifetime rules](execution.md#batched-numerical-operations).

## Fused multiply-add

`Operation::fma` and `fms` return `RN(a·b + c)` and `RN(a·b − c)` with a single rounding, as
`mpfr_fma` / `mpfr_fms`. `complex_fma` and `complex_fms` round each component once from its exact
value:

```
re = RN(a.re·b.re − a.im·b.im ± c.re)
im = RN(a.re·b.im + a.im·b.re ± c.im)
```

This is more accurate than the composed `complex_mul` followed by `complex_add` (up to four
roundings per component) and is not the MPC contract for `mpc_fma`. Products are formed exactly;
terms are combined exactly in a bounded workspace. A term more than one bit below the extended
window of a larger term (at least `bits + 34` bits below that term's leading bit) only contributes
the sign of a sticky unit, which cannot change a `bits`-bit round-to-nearest decision. For three
terms the two largest are summed exactly before the smallest is added, so cancellation between them
never exposes a collapsed term. Statuses propagate as for other real operations; intermediate
products may exceed the exponent range when the final result does not. Independent references:
`mpfr_fma`/`mpfr_fms`, and exact products at twice the precision summed by `mpfr_sum`.

Use the ternary overloads `CommandBatch::run(op, a, b, c, out)` and
`Engine::run_ternary(bits, op, a, b, c, out, count)`; any operand may alias `out`.

## Square and square root

`square(a)` computes the exact integer product and rounds once, as `mul(a,a)`
does. A symmetric squaring kernel is selected on the GPU at 384 bits; other
precisions use the validated schoolbook multiplication kernel. The CPU uses the
symmetric implementation at every supported precision. Squaring does not change
the complex-division composition.

`sqrt(a)` rounds the mathematical square root once. Zero returns canonical zero;
a negative input returns `invalid`; an existing error status propagates. It uses
an integer restoring square root, including correct floor division of negative
odd exponents. If `Z` is the scaled integer radicand and `Q=floor(sqrt(Z))`, the
exact residual is `R=Z-Q*Q`. The result rounds upward precisely when `R>Q`,
because `(Q+1/2)^2=Q^2+Q+1/4` and the integer radicand cannot lie exactly at that
midpoint. No floating-point approximation decides the final rounding.

## Pairwise reductions

`CommandBatch::tree_sum(input,out)` accepts real or complex resident buffers and
requires a one-element output. At each level it adds adjacent pairs `(0,1)`,
`(2,3)`, and so on using the ordinary rounded add primitive. An unpaired final
value is copied unchanged. Levels repeat until one value remains. Empty inputs
return canonical zero; a one-element input is copied and may alias its output.
Complex sums follow this order independently for their real and imaginary parts.

This is a reproducible tree of separately rounded additions, rather than a
single correctly rounded exact sum. Its result can differ from a sequential
left-to-right sum. Intermediate exponent overflow and error statuses follow the
ordinary add contract; a later cancellation does not undo an earlier error.
Dependent dispatches and the internal scratch buffers remain in one submission.
Cooperative threadgroups evaluate consecutive power-of-two blocks of this same
tree in shared memory, then reduce their roots. Padding a final block with
canonical zero preserves the valid-input result and error behavior. Group size
changes scheduling and storage without changing the mathematical tree.

## Precision casts

`Engine::cast(from_bits, to_bits, complex, in, out, count)` and `CommandBatch::cast(in, out)` convert
between any two supported widths: widening is exact, narrowing rounds to nearest even (carrying into the
next binade when needed), componentwise for complex values. Statuses are kept; a result outside the
exponent range gets `exponent_overflow`. Widening rounded data cannot restore lost digits: when a
solver raises its working precision, recompute precision-sensitive inputs from their source.

## Broadcast operands

Binary and fused element-wise operations can read `b` (and `c`) through `Broadcast{stride, period}`:
element `i` uses index `(i / stride) % period` (`period = 0`: no wrap). `stride = k` shares each value
across `k` adjacent outputs; `period = m` cycles a table of `m` values. The operand then holds
`broadcast_elements(count, broadcast)` values. Available as `Engine::run(…, b_index)`,
`Engine::run_ternary(…, b_index, c_index)` and the matching `CommandBatch::run` overloads; rounding is
unchanged. The `a` operand and the output keep one element per index.

## Batched 4×4 systems

`Engine::lu4(bits, shape, A, B, X, det, status)` factors many independent 4×4 complex matrices, one GPU
thread each (`A` is `[count][4][4]` row-major). Column `k` pivots on the row `r ≥ k` with the largest
`max(|re|, |im|)` (exact comparison, ties to the lowest row; no rounding involved). A zero pivot sets
`status = k + 1`, the determinant to zero and every solution entry to zero with `division_by_zero`.
Elimination: `l = cdiv(a_rk, a_kk)`, `a_rc ← a_rc − l·a_kc` for `c > k`. Solutions (`rhs` columns of
`B`/`X`, `[count][4][rhs]`, or the inverse with `shape.inverse`) apply the row permutation, forward
substitution with `y_i ← y_i − l_it·y_t` (t ascending) and back substitution
`y_i ← cdiv(y_i − Σ_{t>i} u_it·y_t, u_ii)` (t ascending). The determinant is the ordered product of
the pivots, negated for an odd number of row swaps. Each multiply-subtract is composed
(`csub(c, cmul(a, b))`) or, with `shape.fused`, one rounding per component via `cfms`. The reference in
`tests/test_batched4.cpp` replays this sequence with MPFR. `CommandBatch::lu4(shape, A, B, X, det, status)` encodes
the same operation on resident buffers (`Buffer<std::uint32_t>` holds statuses; unused buffers may be
empty), so inverses can feed later operations in the same batch.

## Acceptance checks

`Numerics::check_threshold(bits, count, values, threshold, info)` and the resident
`check_threshold(batch, values, count, threshold, info)` compare real values (typically norms or scaled residuals)
with one real threshold on the GPU. Entry `i` passes when it carries no status and `values[i] ≤ threshold` by
exact comparison (no rounding). `ThresholdInfo` reports the number of failing entries, the lowest failing index
(`no_index` if none), the OR of the entry statuses and the threshold's (a threshold with a status fails every entry),
and the number compared. The resident form leaves only these 16 bytes for the host's acceptance decision, after norms
or residuals computed in the same batch.

## Transcendental functions

`Transcendentals::run(bits, f, a, out, count, b)` (`transcendental.hpp`, plan D7) evaluates element-wise
real `exp`, `expm1`, `log`, `log1p`, `sin`, `cos`, `atan2(a = y, b = x)` on `Float<bits>` arrays and
complex `exp`, `log` on `Complex<bits/32>` arrays; `Transcendentals::powi(bits, z, k, k_count, out, count)`
computes `z^k`. `transcendental_cpu` gives the same results on the host. `out` may alias an input.
Resident passes (round 39, GPU retries round 44): `Transcendentals tr(engine); auto t = tr.run(batch, f, a,
out)` (`atan2`: `tr.run(batch, f, y, x, out)`; `tr.powi(batch, z, k, out)` with `Buffer<std::int32_t> k`)
encode the same first pass and GPU retry rungs into the batch, so operations encoded later in the batch
read final values whenever the GPU rungs decide every element (`t.report().host_retried() == 0`; random
inputs essentially always, the hard-case sets of the tests up to 352 bits). Inputs whose decision needs more
than the 35-word cap of the GPU rungs (short dyadics near a rounding midpoint need about `2N` words) are left
to the host: one hard case per width at 384–544 bits, most of them from 576 bits. Elements left for the host final step are patched by `Submission::wait()` and are
provisional until then (`provisional_reads`); `out` cannot be written again in the batch, and equals the
host-array call bit for bit once waited (`docs/execution.md`, "Resident units").

**Contract.** Every real function and every component of complex `exp` and `log` is **correctly
rounded**: the result is `RN(f(x))`, round to nearest, ties to even, of the exact mathematical value,
bit-identical to `mpfr_exp`, `mpfr_expm1`, `mpfr_log`, `mpfr_log1p`, `mpfr_sin`, `mpfr_cos`, `mpfr_atan2` and to
`mpc_exp` / `mpc_log` with `MPC_RNDNN` (each component rounded once from its exact value). Exactly
representable results occur only at the trivial points and are returned exactly: `exp(0) = 1`,
`expm1(0) = log(1) = log1p(0) = sin(0) = 0`, `cos(0) = 1`, `atan2(0, x > 0) = 0`, `exp(a + 0i) = (exp a, 0)`,
`log|z| = 0` for `|z| = 1` (dyadic `z` on the unit circle are `±1, ±i`). `powi` is a documented sequence of
fused products and is **not** correctly rounded (below).

**Conventions** (no signed zeros, no infinities):

| Function | Domain / special values | Status |
|---|---|---|
| `exp`, `expm1` | `\|x\| ≥ 2^31`: result beyond `2^±3·10^9` | `exponent_overflow`; `expm1` of large negative `x` rounds to `−1` (ok) |
| `log` | `x ≤ 0` | `invalid` |
| `log1p` | `x ≤ −1` | `invalid` |
| `sin`, `cos` | `\|x\| ≥ 2^8192` (implementation limit of the 2/π table) | `invalid` |
| `atan2(y, x)` | `(0, 0)`; result in `(−π, π]`, `atan2(0, x < 0) = RN(π)` | `invalid` for `(0, 0)` |
| complex `exp` | `\|Im z\| ≥ 2^8192`: both parts `invalid`; `\|Re z\| ≥ 2^31` and `Im z ≠ 0`: both `exponent_overflow`; `Im z = 0`: `(exp(Re z), 0)` | as listed |
| complex `log` | principal branch, `Im ∈ (−π, π]`; negative real axis gives `Im = RN(π)` (the upper side, as MPC with `+0i`); `log 0` | both parts `invalid` for `z = 0` |

Results outside the exponent range (including underflow below `2^−10^9`) give `exponent_overflow`
with a zero payload, decided after rounding (as `pack`). An input status propagates: every output
component becomes zero with the OR of the input statuses (both input components for complex
functions, both arguments for `atan2`).

**Algorithm (Ziv).** Each element is evaluated on the GPU at `W ≥ N+2` words (at least 64 guard bits; the
smallest GPU-validated width, `docs/gpu-codegen.md` section 9: `W = N+2` up to 256 bits, 14 words at
288–384 bits, 18 at 416–512, 22 at 544–640, 26 at 672–768, 29 at 800–832, `N+2` from 864 bits) with the correctly
rounded `core.hpp` primitives, and the code carries a rigorous bound `|y − f(x)| ≤ err · ulp_W(y)`
(`src/transcendental_core.hpp`). `certify` accepts `y` only if the low `W−N` words differ from the
rounding midpoint by more than `err + 1` units (and `err < 2^62`), so every value inside the bound has
the same RN result, including at binade boundaries. Undecided elements are appended to a compacted list
(device atomic counter). Since round 44 three **GPU retry rungs** follow in the same command buffer (or
batch): rung `r` evaluates the elements listed by the previous level with the same code at a validated
width ≥ `N+4`, `2N+4`, `4N+8` words capped at 35 words (`rung_table` in `src/transcendental.mm`; rung kernels
at ≥ 36 words miscompile, `docs/gpu-codegen.md` section 11): 14/18/35 words at 224 bits, 14/22/35 at 256, 18/29/35
at 384, 22/35/host at 512, 35/host/host at 992–1024 bits. It writes the elements it decides and compacts
the rest into the next list (two ping-pong lists of `count` entries; a one-thread kernel turns the device
counter into the threadgroup count of an indirect dispatch, so a rung without work launches nothing and the
host never waits between levels). Results stay
unwritten until a level certifies them. Elements still undecided after the last GPU rung keep the RN value
of that rung's approximation and their saved operands, and the **host final step** re-evaluates them with
the host ladder (the smallest ladder widths ≥ `N+4`, `2N+4`, `4N+8` words, ladder 4…136, the same code as
`transcendental_cpu`), which makes the results bit-identical to the round-35 host retries in every case. No
input has ever needed more than the third rung; if no host rung decides either, the RN value of the
136-word approximation is written (error < ulp/2 + 2^−3000 relative) and `report().unresolved` counts it.
The report gives the GPU per-rung counts `resolved[0..2]`, the host final step `resolved[3]` and the widths
`first_words`, `rung_words[3]` (0: the rung runs in the host final step). A GPU rung is latency-bound: each undecided
element is one GPU thread at 14–35 words, which costs 1–5 ms whatever the number of elements, while the host
ladder needs ~10–200 µs per element and thread. Host-array calls therefore run the first pass alone and send
up to `TranscendentalOptions::gpu_retry_threshold` (default 512) undecided elements straight to the host
ladder (`resolved[3]`, `rung_words` 0), and more than that to the GPU rungs in a second command buffer;
threshold 0 encodes all levels in one command buffer. Resident passes always encode the GPU rungs. Results
are identical on every route.
Algorithms, with constants from exact integer series built on the host (Machin π, power-of-two
series for `ln 2`, `log(1 ± 2^−i)`, `atan(2^−i)`, each within 1 ulp_W, `ln 2` also at `W+2` words, and
`2/π` to 12,864 bits by long division):

- `exp`: `k = nearest(x/ln 2)` from 64-bit truncations, `r = RN(x − k·ln 2)` with a `W+2`-word fused
  product (|k| < 2^32 keeps the constant's error below `2^(−32W−32)`; the relative error of a small `r`
  is accounted as `2^(−31−e_r)` units), `expm1(r/2^s)` by a Horner Taylor polynomial (truncation < u/16),
  then `s` doublings `E ← E(E+2)` — each adds two roundings and scales the relative error by
  `1 + |E|/(2+E)`, so the doublings lose no bits (unlike squaring `exp`). `exp = 2^k (1+E)`, `expm1 =
  2^k(1+E) − 1` without cancellation for `k ≠ 0`. `(s, K)` minimise `s+K` per width.
- `log x = e·ln 2 + log1p(m−1)`, `m ∈ [0.7071, 1.4142]` (`m−1` exact, and `|result| ≥ 0.346|e|` for
  `e ≠ 0`). `log1p(f)` uses restoring steps `1+f ← (1+f)(1 ∓ 2^−i)` with table sums, then
  `2 atanh(f/(2+f))`. The steps run on `f` itself, so errors stay relative to `f` and `x → 1` loses
  nothing. The factors `1+2^−i` converge in one pass; the factors `1−2^−i` do not (a test found `f ∈
  [0.30, 1/3)` ending near `2^−6`), so for `f > 0` an applied factor is tried once more. `log1p(x)` for
  `|x| ≥ 1/4` is `log(RN(1+x))` (bound `+2^−32W`).
- `sin`, `cos`: Payne–Hanek reduction `x = qπ/2 + r` (exact product of the significand with a
  `2W+3`-word window of `2/π`, ≥ `64W+62` fraction bits), `r/2^h`, Horner series of `sin` and of
  `vers = 1 − cos`, then `h` doublings `s ← 2s(1−v)`, `v ← 2s²`. Arguments near multiples of π/2 get a
  large, honest bound (`2^(64W−F−e_f)`) and are retried at more words.
- `atan2`: octant reduction, restoring rotations `(A, B) ← (A + B2^−i, B − A2^−i)` with `atan(2^−i)` table
  sums (each rotation moves the argument by exactly `atan(2^−i)`, and the convergence condition
  `atan 2^−i ≤ Σ_{j>i} atan 2^−j` holds), a Horner `atan` tail, then `π/2 − θ`, `π − φ`. Exponent spreads
  over `32W+8` bits use `b/a` directly (or `π/2`, `π`).
- complex `exp`: `e^a cos b`, `e^a sin b` with `2^k` kept apart (no spurious overflow).
- complex `log`: `log|z| = ½ log1p(RN(a²+b²−1))` with exact squares (`dot2_add`) when the larger
  exponent is −1 or 0 (|z| near 1), otherwise `e ln 2 + ½ log(RN((a²+b²)4^−e))`; `Im = atan2(b, a)`.

The bounds are verified white-box: `tests/test_transcendental.cpp` evaluates the `W`-word approximations
for W = 4, 9, 10, 14, 34 and measures `|y − f|` with MPFR at `32W+256` bits.

**Integer powers.** `powi(z, k)`: `k = 0` gives `(1, 0)` (also for `z = 0`). Otherwise binary powering of
`n = |k|` from the least significant bit: `acc ← acc·p` for each set bit (the first set bit copies
`p`), `p ← p·p` while higher bits remain, where every product is fused per component
(`re = RN(a.re b.re − a.im b.im)`, `im = RN(a.re b.im + a.im b.re)`, `core.hpp` `cfma` with zero addend).
For `k < 0` the result is `1/acc = (RN(acc.re/d), RN(−acc.im/d))`, `d = RN(acc.re² + acc.im²)`, evaluated
as with an unbounded exponent range (power-of-two prescaling) and range-checked at the end. Each
product is range-checked (`exponent_overflow` propagates, so an intermediate overflow fails the
result); `z = 0, k < 0` gives `division_by_zero`. The reference is an MPFR replay of this sequence
(`mpfr_fmma` / `mpfr_fmms`, `mpfr_div`), bit for bit; against `mpc_pow_si` the normwise error grows
roughly linearly with `|k|` (measured: at most 39 ulp of the larger component for `|k| ≤ 64`).

**Hard cases and retries.** Random full-precision arguments essentially never need a retry (an element is
undecided only within about `2^−64·err` ulp of a midpoint, `err` the 10–200 ulp_W bound). Retries come from exact dyadic structure:
short-significand arguments where the leading Taylor terms are exactly representable and land on a
rounding midpoint, so that a cubic term decides — `exp(±2^−32N)`, `expm1(2^(1−32N))`, `log1p(2^−32N)`,
`cos(2^−16N)`, `log|z|` of `(1, 2^−16N)`, and nearby short significands. Their distance to the midpoint is
about `|x|³`, which the `2N+4` (cubic terms of `exp`, `log1p`, quartic of `cos`) or `4N+8`-word rungs
resolve; arguments near multiples of `ln 2` or `π/2` raise the bound of the reduced argument and are
resolved at the first or second rung. Measured (`benchmarks/results/round35_transcendental_tests_*.txt`):
18 M random GPU points at 64/224/256/384/1024 bits (1/8 with short significands) gave 0 mismatches and a
retry rate of 0 (`exp`, `expm1`, `log`, `atan2`, complex `log`) to 9·10⁻⁵ (`log1p`), all resolved by the
second rung; with the hard-case sets (~600 per function and width) up to 0.5% retry, at most 27 of them
reaching the third rung, never unresolved. All 31 widths pass on the GPU, also under
`MTL_SHADER_VALIDATION=1`; 0.49 M white-box bound checks gave a largest actual error of 9.8 ulp_W and a
largest actual/bound ratio of 0.98 (the half-ulp final rounding).

**GPU retry rungs (round 44).** Every rung width of `rung_table` was validated with every element forced
through that rung (`test_limbforge_transcendental --force-level r`, later rungs on the host) at all 31
widths, also under `MTL_SHADER_VALIDATION=1` at 224/256/384/1024 bits (`docs/gpu-codegen.md` section 11,
`benchmarks/results/round44_*`). The all-width gate (3,000 random points plus the hard cases per function
and width) gave 0 mismatches; its 3,279 retries were resolved by GPU rungs 1/2/3 (340/1,423/7) and by the host
final step (1,509, all hard cases above 352 bits whose decision needs more than 35 words), none unresolved.
The per-rung counts equal a host replica of the GPU decisions (same code and widths) at 256, 384 and 1024
bits, and resident reports equal the host-array reports rung by rung.

**Performance** (`benchmarks/results/round35_transcendental*.csv`, 10⁶ elements, loaded host): GPU wall
17–86 ms at 224–384 bits, 36–91× serial MPFR/MPC and 3–12× an 18-worker pool; at 10⁴ elements the call
is latency-bound (1.3–3.7 ms). First use of a width compiles about 0.4 s of library plus 0.4–2 s per real
function, 1.3–4 s for complex `exp` and 5–14 s for complex `log` and `powi` (OS-cached afterwards; use
`Transcendentals::prewarm`). Since round 44 a function's first use also compiles its GPU rungs (up to three
more libraries and pipelines, compiled concurrently): measured cold under load 25–50, exp 3.2–4.9 s, sin
3.8–6.0 s, complex `exp` 7.3–9.2 s and complex `log` 24–28 s at 224–1024 bits, against 1.4–2.8, 1.2–3.7, 2.4–7.2
and 6.9–20 s for the first pass alone (`benchmarks/results/round44_transcendental_compile.txt`).
Retries on the GPU (`benchmarks/results/round44_transcendental_retries_*.csv`, interleaved medians against
round 39): without retries the cost is unchanged (host-array and resident wall ratios 0.81–1.28, median
1.00, at 10⁴–10⁶ elements and 224–384 bits). A GPU rung adds 1.2–5 ms of device time whenever it has work
(one GPU thread per element at up to 35 words), whereas the host ladder costs ~10–200 µs per element and
thread; with the default threshold host-array calls keep the host ladder for ≤ 512 retries (unchanged within
noise) and gain 0.98–1.32× wall (median 1.24) at 10⁶ elements with 1% hard inputs (3,333–5,000 retries,
GPU route).
Resident passes always use the GPU rungs: 0.99–1.38× faster at 10⁶ elements with 1% hard inputs, but up to
3× slower at 10⁴–10⁵ elements with 3–500 retries (the rung latency), in exchange for final values inside the
batch.

## Segmented dot products

`Engine::segmented_dot(bits, complex, shape, a, b, out)` and `CommandBatch::segmented_dot` compute
`out[i] = Σ_{k<K} a[i·K+k]·b[i·K+k]`, or with `shape.shared_right` one table `b[k]` for every
segment, real or complex. The order is fixed and sequential: `s = fma(a₀, b₀, 0)`, then
`s = fma(a_k, b_k, s)` for k = 1 … K−1 (`cfma` for complex, one rounding per component per step), so
results are bitwise reproducible independently of the launch configuration. `K = 0` gives zero;
statuses propagate. One GPU thread per segment (suited to K ≈ 10–120). Reference: the same chain with
`mpfr_fma` / exact products plus `mpfr_sum` (`tests/test_segmented_dot.cpp`). For one rounding of the
whole dot product, use the dense products in `linalg.hpp`.

## Vector recurrences

`Engine::vector_recurrence(bits, shape, start, p, q, r, out, base, dp, dq)` advances many
independent four-component complex vectors (one GPU thread each; plan D4). Each multiply-add `mac(a, b, c)` is either composed, `cadd(c, cmul(a, b))` (default; the rounding of
the existing complex operations), or fused with `shape.fused = true`, `cfma(a, b, c)` with one rounding
per component (see "Fused multiply-add"; about 3× the composed time at 256 bits and 1.8× at 1024 bits, round 46). With
`dot(a, v) = mac(a3, v3, mac(a2, v2, mac(a1, v1, mac(a0, v0, 0))))`, one step with coefficient index
`k` is (writing `cfma` for `mac`):

```
rank-one (default):  s = dot(q[k], v);  v_i = cfma(p[k]_i, s, v_i)
matrix:              v_i = dot(M[k]_i, v)           (all components from the previous v)
tangent:             s = dot(q[k], v);  sb = dot(q[k], b);  tq = dot(dq[k], b)
                     v_i = cadd(cfma(p_i, s, v_i), cfma(p_i, tq, cfma(dp_i, sb, 0)))
affine (any form):   v_i = cadd(v_i, r[k]_i)       (applied last)
```

The tangent form evaluates `dv + p (q·dv) + dp (q·b) + p (dq·b)` for a stored base trajectory `b`
(an `all_steps` run of the base recurrence in the same direction; `b[t]` is the state before step
`t`). `reverse` applies coefficient steps `k = steps−1, …, 0`; outputs and base states are indexed by
sequence position. Layouts are component-major (`[j][lane]`); coefficients shared by
`lanes_per_weight` adjacent lanes are `[steps][4][G]` (`[steps][4][4][G]` for `M`), with
`G = lanes / lanes_per_weight`; `r` is `[steps][4][lanes]`; tangent inputs use `lanes_per_base` and
`lanes_per_tangent` in the same way. The independent reference (`tests/test_vector_recurrence.cpp`)
implements this sequence with MPFR (`mpfr_sum` per fused component).

`CommandBatch::vector_recurrence(shape, start, p, q, r, out, base, dp, dq)` encodes the same operation
on resident buffers (unused inputs may be empty buffers; `out` may alias `start`). A base pass with
`all_steps` and the tangent pass reading its output can share one batch, so the base chain never
leaves the GPU; buffer sizes are checked against the shape before encoding.

Current limits: shapes are validated before submission. Both modes run at every width through 1024 bits.
Composed mode is tested against MPFR at 64/224/256/384/512/768/1024 bits; fused mode at the same widths in all
forms and, with hard cases (exact and near cancellation, a tiny product exposed, far-apart terms, ties), at all
31 widths. In fused mode every multiply-add of a step is one iteration of a rolled loop around a single exact
complex fma (`cfma_rolled` in `core.hpp`, the same contract and results as `cfma`), which keeps the kernel small:
a cold pipeline compiles in 0.3–4 s for every flag combination at 256–1024 bits (round 46; before, 7–31 s for
the rank-one and matrix forms, 117–207 s for the tangent form at 512 bits and over 300 s above, and once a
compiler crash at 1024 bits).
Pipelines are still compiled on first use, so warm them before timing.

## Recurrences

`Engine::recurrence` computes independent trajectories:

```
next = weights[0]*q0 + weights[1]*q1 + weights[2]*q2 + weights[3]*q3
q0 = q1; q1 = q2; q2 = q3; q3 = next
```

Each trajectory has four complex seeds, ordered oldest to newest in propagation
order. Seeds are arranged as `seeds[j*count+i]`. With `states_per_weight=1`,
weights are `weights[(step*4+j)*count+i]`. When adjacent trajectories share
coefficients, pass `states_per_weight`; the weight layout becomes
`weights[(step*4+j)*(count/states_per_weight)+i/states_per_weight]`. The output is
the final `q3`; zero steps return the fourth seed.

## Dense products: SYRK and GEMM

`include/limbforge/linalg.hpp` (`class Linalg`, with its own Metal device, queue and MSL 4.0 library):

```cpp
Linalg la;                                     // LinalgOptions: band_bits, max_bands, max_spread, host_threads
la.syrk(bits, A, rows, cols, C, lower_only);   // C (cols x cols) = A^T A
la.gemm(bits, transpose_a, A, B, m, n, k, C);  // C (m x n) = op(A) B; B is k x n; op(A) = A (m x k) or A^T (A is k x m)
la.syrk(bits, A, rows, cols, C, true, true);   // update: C = RN(C - A^T A), lower triangle
la.gemm(bits, transpose_a, A, B, m, n, k, C, true); // update: C = RN(C - op(A) B)
la.report();                                   // bands, fallbacks, moduli and stage times of the last call
```

Matrices are host arrays, row-major, with elements `Float<bits>` (`Number<bits/32>`, `bits/8 + 12`
bytes), and `bits` a multiple of 32 in [64, 1024]. C must not overlap A or B. With `lower_only`,
SYRK writes `C[i][j]` for `i >= j` and leaves the rest of C untouched; otherwise it writes the full
symmetric matrix. The GPU uses page-aligned inputs and outputs in place (no copy). The call returns
after the GPU has finished.

**Contract.** Every output is one rounding of the exact dot product,
`C[i][j] = RN(sum_k L[i][k] * R[k][j])`, round to nearest, ties to even. This holds for any K,
term order, exponent spread, or cancellation. Products and partial sums are never rounded, and
tiny entries are never dropped. Statuses propagate as for the other real operations: if any entry
of row i of the left operand or column j of the right operand has a status, the output is zero
with the OR of those statuses. An empty K, an all-zero line, or exact cancellation gives canonical
zero. A rounded result outside ±1,000,000,000 gives `exponent_overflow`; intermediate products
may lie outside that range. SYRK results are exactly symmetric. The independent reference
(`reference::dot` in `tests/reference.hpp`) forms every product exactly with MPFR at `2*bits` and
sums them with one `mpfr_sum`.

**Updates.** With `subtract`, each output becomes `C[i][j] = RN(C[i][j] - sum_k L[i][k] * R[k][j])`,
again one rounding of the exact value (the old entry is one more exact term). A status on the old
entry joins the OR above; without nonzero products (empty K or a zero line) the entry is unchanged.
SYRK updates read and write the lower triangle only. On the GPU the reconstructed exact sum (Garner
integer or multi-band accumulator) and the old entry are added by the two-term exact sum of `fma`
(a term more than one bit below the other's extended window contributes only its sign) and rounded
once; the `reconstruct` and `finish` pipelines are specialised by a function constant. The host
fallback is `exact_dot_add<N>(c, subtract, a, stride_a, b, stride_b, K)`, which treats `c` as one
more exact term of `exact_dot`. Reference: `reference::dot_sub` (exact products and `c`, one
`mpfr_sum`).

**Algorithm.** A *line* is a row of the left operand or a column of the right operand (for SYRK,
a column of A on both sides). Each line is split into exponent *bands*. Starting from the largest
remaining exponent `hi`, the band `[hi - G, hi]` (`G = band_bits`, default 64) takes every entry
in that range, and `lo` is the smallest exponent in it. An entry with mantissa `m` and exponent `e`
becomes the exact integer `X = m * 2^(e - lo)` of at most `bits + G` bits, scaled by
`2^(lo - bits + 1)`, so nothing is truncated. All bands of one side are concatenated into an
*extended* operand, so one integer GEMM per modulus computes every band pair `(b, c)` as a
sub-block. The integer GEMM is the exact residue GEMM of plan item L1b:

- primes `m <= 65279`;
- balanced residues split into two int8 digits;
- three TensorOps int8 matrix products per modulus, exact in int32 for `K <= 65472`;
- the products recombined modulo `m`.

Since round 40 one dispatch per batch of moduli (every modulus for small products; batches limited
by a digit-plane budget of 64 MB for large ones) forms the three products of each 64 × 32 output
tile in registers (MPP cooperative tensors) and writes only the residue, instead of three int32
planes per modulus and a separate combine pass: per output and modulus, 4 instead of 28 bytes of
memory traffic (int32 planes written and read back, residue written and read by `reconstruct`),
which dominates a QR trailing update `RN(X - V Y)` (K = 32); a small product needs a handful of
dispatches instead of four per modulus. The residues are the same exact values.

The modulus count makes `prod m > 4 * K * 2^(Pb + Pc)`, where `Pb` and `Pc` are the widest integer
widths (`bits` plus the widest band). This is strictly more than the `2 * |sum|` that the signed
Garner reconstruction needs. When both lines of an output have one band, its reconstructed integer
is rounded directly with `pack`, as in every other operation. Otherwise the band-pair integers of
that output are added exactly into a two's-complement accumulator, at their offsets from the
lowest bands (bounded by `max_spread`), and the total is rounded once. SYRK computes only the lower
triangle and skips tiles above the diagonal. An off-diagonal band pair `(b, c)` with `b > c` also
supplies the transposed `(c, b)` terms. It is reconstructed in two dispatches so that no two
threads of one dispatch update the same accumulator.

**Exact host fallback.** `exact_dot` on the CPU (threads: `host_threads`, default all) computes:

- every line whose nonzero exponents need more than `max_bands` bands (default 4);
- every line whose nonzero exponents span more than `max_spread` bits (default 512);
- every line when `K > 65472`.

`exact_dot` sorts the exact products by exponent. It starts a new cluster whenever every remaining
product lies more than `bits + ceil(log2 K) + 4` bits below the current cluster's lowest bit, and
sums each cluster exactly. Let `S` be the first nonzero cluster sum (a multiple of `2^lsb`) and `F`
the rest. Then `|F| < 2^(lsb - bits - 2)`, so no midpoint of the target precision lies strictly
between `S + F` and `S + sign(F)·eps`. Each nonzero cluster dominates everything below it, so
`sign(F)` is the sign of the next nonzero cluster. Hence `RN(S + F) = RN(S + sign(F)·eps)` exactly,
including ties. `exact_dot<N>(a, stride_a, b, stride_b, K)` is public and host-only. When all
term scales of one call span at most `8128 - 64N - ceil(log2(terms)) - 1` bits (254 words of
accumulator), it skips the clusters and adds every term exactly into one positive and one negative accumulator,
then rounds their difference once (the same value). The host resolves statuses and zero lines
without arithmetic.

`LinalgReport` lists:

- lines per band count (index 0: zero or status lines);
- fallback lines and outputs, and multi-band outputs;
- band pairs and extended GEMM sizes;
- the number of int8 GEMMs (three per modulus);
- host stage times.

Bands help only when a few lines carry entries far below their neighbours. With one narrow band
per line the modulus count is at its minimum: 33 at 256 bits for K = 2000 when each line has a
single exponent. A spread up to `G` within a band costs about `2*spread/16` extra moduli, and the
widest line sets this for the whole product.

**QR.** The normal equations are solved by the blocked Cholesky below; Householder QR (least
squares, also of the augmented Levenberg–Marquardt matrix `[J; sqrt(mu) D]`) is in "QR factorization
and least squares".

### Resident products

`syrk` and `gemm` (with `subtract`) also encode into an Engine's `CommandBatch` on `Buffer<Float<bits>>`
operands of that engine (round 42; `docs/execution.md`, "Resident units"):

```cpp
Engine gpu; Linalg la(gpu); Numerics nm(gpu);   // Linalg(Engine&): the engine's Metal device
using F = Float<256>;                           // A: rows x cols, C: cols x cols, row-major as the host calls
auto batch = gpu.batch();
batch.run(Operation::mul, J, S, A);             // any producer of A in the same batch
LinalgTicket t = la.syrk(batch, A, rows, cols, C);         // lower_only, subtract as the host call
// la.gemm(batch, transpose_a, A, B, m, n, k, C, subtract);
nm.norm2(batch, Segments{cols, cols, false}, C, norms);    // a reader of C (flagged, see below)
batch.submit().wait();                          // also runs t's host-fallback outputs; then C is final
t.report();                                     // the host call's analysis and placement (no timings)
```

**Contract.** After `Submission::wait()` C equals the host-array call with the same `LinalgOptions` bit
for bit (every output is one rounding of the exact value, and the placement is the same). The
`Linalg`'s own `report()` is not changed; `LinalgTicket::report()` (after wait) gives the band
histograms, fallback lines and outputs, multi-band and trivial outputs, band pairs, extended sizes and
`int8_gemms` exactly as the host call reports them for the same data (tested).

**GPU analysis.** The host path reads the exponents on the host before it sizes anything. The resident
call computes the same decisions in the batch and drives every data-dependent size by indirect
dispatch (a dispatch with nothing to do has zero threadgroups):

1. `analyze_lines`, one SIMD group per line: status OR, largest and smallest nonzero exponent, the class
   (GPU bands, zero/status line, host fallback) and the greedy bands `[hi - G, hi]` of `analyze`
   (one pass over the line per band);
2. `plan_side`, one threadgroup per side: chunked SIMD prefix sums give the band member lists in line
   order, the line table (band count, lowest band exponent, accumulator slot) and a summary (members
   per band, multi- and single-band lines, widest band, fallback lines);
3. `plan_final`, one thread: the extended sizes, the modulus count, `Wc` and the Garner bounds of the
   widest band (from a table that the host precomputes for this `ceil(log2 K)` and every width
   `0..G`: the host path's formula), and the `Params` record and threadgroup counts of every later
   dispatch;
4. the product pipeline of the host path (digits, `product_residue` per modulus batch, one
   `reconstruct` per band pair, `finish`), then the zero/status outputs (`trivial_outputs`) and, for
   a full SYRK, the upper triangle (`mirror_upper`).

All dispatches of the call are in the batch's encoder with a buffer barrier before each: about
`12 + 3·batches + R²`. `analyze_lines` reads every entry once, and again once per band of a multi-band
line; analysis and plan take 0.02–0.16 ms of GPU time for n = 200–800 (one thread per line and serial
scans took up to 0.9 ms). Measured against the host-array calls (256 and 224 bits, n = 200–1000,
loaded host): 1.05–1.38× less wall time per call, with 0.1–0.6 ms more GPU time
(`benchmarks/results/round42_resident_linalg_*.txt`).

**Data-independent bounds.** A GPU line has at most `R = min(max_bands, max_spread/(band_bits+1) + 1)`
bands (each band starts more than G below the previous one; at most 64). The workspace holds extended
operands of `min(R, resident_bands) * lines` band rows per side (`LinalgOptions::resident_bands`,
default 2: on average two bands per line), the modulus count of the widest possible band (`band_bits`)
at this K, digit planes in batches of the 64 MB plane budget, and `M * N * acc_words` accumulator
words when `R > 1`. If the band rows of a call exceed it (`resident_overflow` in the ticket report),
the plan disables the GPU products and every output without a zero or status line is computed by the
host fallback at wait: correct, but slow. Raise `resident_bands` (up to `max_bands`) when most lines
have several bands. Example, QSC-like 800 × 400 SYRK at 256 bits (`resident_bands = 2`): residues
57 MB, digit planes 66 MB, accumulators 32 MB, snapshot 14 MB.

**Host fallback (at wait).** Outputs of lines that the host path sends to `exact_dot` (more than
`max_bands` bands, spread above `max_spread`, `K > 65472`, `force_fallback`), and all non-trivial
outputs after an overflow, are computed at `Submission::wait()` by a completion step, as the
transcendental retries are. A GPU exact-dot kernel was not chosen: the spread of a fallback line is
unbounded (exponents span ±10⁹), so it would need its own host fallback, and an exact chain per thread
was latency-bound in round 23's Cholesky panel experiment. The plan copies A and B inside the batch
when (and only when) there are fallback lines or an overflow, so later operations of the batch may
overwrite them; the step computes `exact_dot_add` from that snapshot and the old C entry (updates) on
`host_threads` threads, mirrors fallback outputs of a full SYRK, and writes C before the submission
releases its buffers. GPU outputs and zero/status outputs are final when the GPU completes.

**C is provisional within its batch.** Writing C again in the same batch (an engine or unit operation,
or a second product into the same C) throws `logic_error`: split the batch. Reading C is allowed and
sets `LinalgTicket::provisional_reads()`. When the ticket also reports `fallback_outputs > 0` (or
`resident_overflow`), the reader saw provisional values for those outputs and must be recomputed (or
the batch split at the product); without fallback outputs a reader sees final values. QSC-like data
(column scales, a few far entries) has no fallback lines.

**Ownership and lifetime.** As for the other resident units: foreign buffers throw `invalid_argument`,
buffers of an unwaited submission `logic_error`; C must not be A or B; sizes are checked (buffers may be
larger); SYRK `subtract` needs `lower_only`. Workspaces are kept by the `Linalg` and reused: per call
(line tables, plan, snapshots: read at wait) and per batch (digit planes, residues, accumulators: the
calls of one batch share them because their dispatches run in order). A workspace is busy from
encoding until the batch's completion step has run (or the unsubmitted batch is destroyed). First use
of new workspace memory costs its page wiring (about 28 ms/GB measured), later calls none. The batch,
its submission and the ticket may outlive the `Linalg`; a discarded batch leaves its ticket
unresolved. A `Linalg` may encode into batches of several threads (pipeline and workspace caches are
locked); its host-array calls stay on one thread.

**Factorizations on buffers.** `cholesky`, `trsm`, `cholesky_solve`, `factor_qr`, `QRFactor::solve` and
`QRFactor::apply_q` also take `Buffer` operands (round 42). Their host panels run between dependent
GPU updates, so they cannot be one submission: these forms are synchronous and outside any batch.
The buffers must be idle; the call owns them as a submission does until it returns (a concurrent
`upload`, `mapped()` or submission using them throws `logic_error`), the GPU uses them in place (the
Cholesky work matrix L, the solution X and the operand of `apply_q` need no scratch copy, which the
host-array forms make for arrays that are not page aligned), and the results equal the host-array
calls bit for bit. L may be A and X may be B (in place); X must not be L, and a QR solution must not
be B. A `QRFactor` made from a buffer owns copies as before (`factor_qr` only reads A).

### Workspace release

```cpp
LinalgWorkspaces w = la.workspaces();          // scratch_bytes, idle_bytes, busy_bytes, busy_workspaces
LinalgWorkspaces r = la.release_workspaces();  // what was freed now (scratch, idle) and detached (busy)
```

A `Linalg` keeps memory between calls so that repeated calls do not allocate and wire pages again (about
28 ms/GB):

- **scratch** of the host-array calls, by role: staging of the products (copies of operands that are not
  page aligned, band member lists, digit planes, residues, accumulators, copied outputs) and the
  factorization and solve scratch (QR and complex QR work matrices, W and Y blocks, right-hand sides, the
  Cholesky work matrix). It is used only during a call on the `Linalg`'s host thread, including the solves
  and Q applications of the factors it made;
- the **resident-product workspaces** above, per call and per batch.

`release_workspaces()` frees the scratch and every idle resident workspace at once. A busy workspace (held by
an unsubmitted batch, or by a submission whose completion step has not run: the host fallback at wait reads
the snapshots in its call workspace) is **detached**: the `Linalg` forgets it, it stays valid for the batch
that holds it, and it is freed when that completion step has run (`Submission::wait()` or the submission's
destructor) or the unsubmitted batch is destroyed. No memory still used by an unwaited batch is freed. Later
calls allocate afresh; a later call into a batch that held a released workspace gets a new one (the batch
then uses both, which changes no result). `workspaces()` reports the retained bytes; detached workspaces
count as busy until they are freed. Compiled pipelines, modulus tables and worker threads are kept. Factor
objects own their arrays and are not affected: a factor made before a release still solves (its solve
allocates scratch again) with the same bits.

Call both on the `Linalg`'s host thread between its host-array calls (not during a factorization, solve or
product of that thread). Other threads may encode resident products concurrently: the resident pools are
locked. Tested in `tests/test_resident_linalg.cpp` (`release`): scratch after QR calls, idle workspaces
after wait, a pending submission with host-fallback outputs released before its wait, a later call into the
same batch, an unsubmitted batch released and destroyed, and identical results throughout.

## Cholesky factorization and triangular solves

```cpp
FactorOptions o;                                 // block = 32, host_macs, solve_host_macs, gpu
CholeskyInfo info = la.cholesky(bits, A, n, L, o);   // A = L L^T; info.pivot == n on success
la.trsm(bits, transpose, L, n, B, nrhs, X, o);   // L X = B, or L^T X = B with transpose
la.cholesky_solve(bits, L, n, B, nrhs, X, o);    // (L L^T) X = B: trsm, then trsm with transpose
```

Host arrays as for the products: row-major `Float<bits>`, B and X are `n x nrhs`. `cholesky` reads
the lower triangle of A and writes all of L (strict upper triangle zero); L may equal A. The solves
read the lower triangle of L; X may equal B. The calls return after the GPU has finished.

**Rounding sequence.** Write `D(c; x, y; S) = RN(c - sum_{k in S} x_k y_k)` for one rounding of the
exact value (ties to even), computed by the update GEMM of "Dense products" or by `exact_dot_add`;
both give the same bits. Statuses follow that contract: zero with the OR of the statuses of `c` and
of every `x_k`, `y_k` with `k in S`; without nonzero products the result is `c`. Columns are split
into blocks `B_b = [b nb, min(n, (b+1) nb))` with `nb = block` (0: one block); `blk(j) = floor(j/nb)`,
`r_j = blk(j) nb`. Rows `l_i` of L are indexed by column.

```
factorization, every column j in order, rows i >= j:
  a_ij^(0) = a_ij                                    (lower triangle of A)
  a_ij^(b+1) = D(a_ij^(b); l_i, l_j; B_b)            b = 0 .. blk(j)-1   (trailing updates)
  s_ij = D(a_ij^(blk(j)); l_i, l_j; [r_j, j))                            (in-block dot)
  l_jj = sqrt(s_jj),  l_ij = div(s_ij, l_jj)  (i > j)                    (correctly rounded)
forward,  L x = b (each right-hand side independently), i = 0 .. n-1:
  b_i^(r+1) = D(b_i^(r); l_i, x; B_r)  for r < blk(i);    x_i = div(D(b_i^(blk(i)); l_i, x; [r_i, i)), l_ii)
backward, L^T x = b, i = n-1 .. 0, with e_i = min(n, r_i + nb) and column i of L as the vector:
  b_i <- D(b_i; L[., i], x; B_r)  for r = last block .. blk(i)+1;   x_i = div(D(b_i; L[., i], x; (i, e_i)), l_ii)
```

An entry `l_ij` thus takes `blk(j) + 2` roundings: one per earlier column block (each the single
rounding of a block of `nb` exact products and the running value), one for the in-block dot and
one for `sqrt` or the division. With a single block every entry is `RN(RN(a_ij - exact dot) / l_jj)`.
`cholesky_solve` is the forward then the backward sequence with the same `nb`.

**Pivots and statuses.** `s_jj` must be positive and without status. The first column `p` (in
order) where it is zero, negative or carries a status stops the factorization: `info.pivot = p`,
`info.pivot_sign` is the sign of `s_pp` (0 for zero or a status), `info.pivot_status` its status.
Columns `j < p` of L are then final, and every lower entry with `j >= p` is zero with status
`invalid`, so nothing downstream uses unfinished values. A status in A reaches a pivot through the
dot products (an entry `a_ij` with a status gives `l_ij` that status and fails pivot `i`). In the
solves, a status in B or L propagates through D, and a zero diagonal entry of a caller's L gives
`division_by_zero` (the division contract).

**Determinism and accuracy.** The result depends only on the inputs, `bits` and `block`: not on
which updates run on the GPU or on the host (`host_macs`, `gpu`), the number of host threads,
page alignment, or the overlap below. Each step is correctly rounded, but the factorization and the
solves as a whole are **not** correctly rounded: it is a fixed, reproducible sequence of
`blk(j) + 2` roundings per entry, more accurate than an `fma` chain (one rounding per product) or
the consumer's `mpfr_mul` + `mpfr_sub` loop (two per product). Measured errors against a `2*bits`
MPFR solve are in "Cholesky accuracy" below. `block` is part of the result: changing it changes
the low bits.

**Execution.** The work matrix is L itself when page aligned, otherwise a shared scratch copy; it
stays on the GPU for the whole call. Blocks are processed right-looking with one block of
look-ahead. For block `b`:

1. Panel on the host: the diagonal block row by row (`sqrt` and `div` of `core.hpp`), then the rows
   below it in parallel; every in-block dot is `exact_dot_add`.
2. The next panel's columns `[k1, k2)` are updated by block `b` first: one update GEMM (rows `>= k1`
   times `nb` columns; the strict upper part of that diagonal block is written and reset to zero).
3. A second host thread updates the rest of the trailing matrix (`j >= k2`, lower triangle, one
   update SYRK) while this thread factors the next panel. The regions are disjoint.

Updates with fewer than `host_macs` multiply-adds (outputs × nb; default 5·10^4) run on the host;
the solves use `solve_host_macs` (default 4·10^5), so a few right-hand sides stay on the host. The
independent reference (`tests/test_cholesky.cpp`) is a left-looking, column-by-column MPFR program
of the sequence above (`reference::dot_sub`, `mpfr_sqrt`, `mpfr_div`).

A GPU panel kernel (one thread per panel row, exact accumulator, host fallback for rows with
exponent ranges over 1,072 bits) was bit-identical but slower: about 4 ms per panel at n = 1000,
256 bits, against about 3.5 ms for the host panel on the loaded host, because one exact multiply-add chain per thread is
latency-bound. It is kept as `benchmarks/experiments/cholesky_gpu_panel.patch`.

### Cholesky accuracy

QSC-like normal equations (`A = J^T J` with Marquardt damping, J of 2n × n with column scales
2^±60, four right-hand sides; `benchmarks/results/round23_cholesky_accuracy.csv`). Errors are log2,
maximum over the right-hand sides, against a `2*bits` MPFR Cholesky solve of the same A and B:

| bits | n | forward, normwise | forward, componentwise | backward, componentwise | MPFR sequential (fwd norm / comp / bwd comp) |
|---:|---:|---:|---:|---:|---|
| 224 | 200 | −222.1 | −211 | −224 | −222.1 / −212 / −221 |
| 224 | 400 | −224.5 | −212 | −223 | −223.2 / −210 / −221 |
| 224 | 800 | −220.8 | −210 | −222 | −220.2 / −209 / −221 |
| 224 | 1000 | −221.9 | −206 | −223 | −219.3 / −206 / −221 |
| 256 | 200 | −255.6 | −245 | −256 | −255.2 / −243 / −254 |
| 256 | 400 | −254.4 | −246 | −255 | −253.6 / −243 / −253 |
| 256 | 800 | −254.2 | −243 | −255 | −251.5 / −241 / −253 |
| 256 | 1000 | −255.4 | −242 | −255 | −251.1 / −240 / −252 |

The componentwise (Oettli–Prager) backward error stays at about one unit roundoff (`2^-bits`) for all
sizes, 2–3 bits below the consumer-style MPFR loop at the same precision; the forward errors are
equal or up to 4 bits smaller. The blocked order is therefore numerically sound for these problems.
The normwise backward error (−342 at 224 bits, −377 at 256) is not informative here because the
column scaling makes `||A|| ||x||` far larger than the residual scale.

## Polynomial values and jets

```cpp
#include "limbforge/numerics.hpp"
Numerics nm;                                             // own Metal device, queue and library (as Linalg)
Polynomial p{points, terms, points_per_set, complex, fused};
nm.poly_eval(bits, p, coeffs, x, values);                // values[i] = p_g(x_i)
nm.poly_eval_jet(bits, p, order, coeffs, x, jets);       // jets[i*(order+1) + j], order <= 2
```

Host arrays of `Float<bits>` or, with `complex`, `Complex<bits/32>` (coefficients, points and results
share the type). Coefficients are in ascending powers, one contiguous set per polynomial:
`coeffs[g*terms + k]` multiplies `x^k`. Point `i` uses set `g = floor(i / points_per_set)`, so adjacent
points share a set; there are `ceil(points / points_per_set)` sets. `terms = degree + 1`; `terms = 0` is
the zero polynomial. Jets are Taylor coefficients with respect to the argument: `(p, p', p''/2!)`. For
inverse-power evaluation pass `x = 1/u`; the chain rule and any prefactor in `u` belong to the caller.
`values` may equal the points array for `poly_eval`; otherwise outputs must not overlap inputs.
Resident form (round 39): `Numerics nm(engine); nm.poly_eval(batch, p, C, X, V)` and
`nm.poly_eval_jet(batch, p, order, C, X, J)` on `Buffer<T>` operands of the batch's engine, bit-identical
to the host calls (`docs/execution.md`, "Resident units"); `p.complex` must match `T`.

**Rounding sequence.** `mac(a, x, b)` is one of

```
composed real:     add(mul(a, x), b)      = RN(RN(a x) + b)
fused real:        fma(a, x, b)           = RN(a x + b)
composed complex:  cadd(cmul(a, x), b)    re = RN(RN(RN(a.re x.re) - RN(a.im x.im)) + b.re)
                                          im = RN(RN(RN(a.re x.im) + RN(a.im x.re)) + b.im)
fused complex:     cfma(a, x, b)          re = RN(a.re x.re - a.im x.im + b.re)
                                          im = RN(a.re x.im + a.im x.re + b.im)
```

The composed forms are the roundings of `mul`/`add` and `complex_mul`/`complex_add`, i.e. the
consumer loop `r = r*u + c`; the fused forms are those of "Fused multiply-add". Every point runs

```
r0 = r1 = r2 = 0
for k = terms-1 down to 0:
    r2 = mac(r2, x, r1)        (order >= 2)
    r1 = mac(r1, x, r0)        (order >= 1)
    r0 = mac(r0, x, c_k)
jets = (r0, r1, r2)
```

This is simultaneous Horner evaluation, highest order first, so that each update reads the previous
step's lower-order value. In exact arithmetic `r0 = p(x)`, `r1 = p'(x)`, `r2 = p''(x)/2`. Lower orders
never read higher ones, so `poly_eval` equals jet 0 of `poly_eval_jet` bit for bit, and order 1 equals
the first two jets of order 2. The first step from zero is exact (`r0 = c_{terms-1}`, `r1 = r2 = 0`),
but it multiplies by `x`, so a status in `x` reaches every jet. Statuses and exponent overflow follow
the primitives, per component: a status in `x` or in a coefficient of the set gives a zero payload
with the OR of those statuses. `terms = 0` gives canonical zero jets without inspecting statuses. The
result is not correctly rounded: composed steps round twice per real step (complex: up to four times
per component), fused steps once per component, with the usual Horner error growth.

**Execution.** One GPU thread per point, one SIMD group per threadgroup; points sharing a set read the
same coefficient (a broadcast within the SIMD group). The jet loop has a single `mac` call site in a
rolled loop, so the exact `cfma` is instantiated once and the kernel compiles at 1024 bits.
Real/complex, composed/fused and the order are function-constant specialisations of one library per
precision. The reference (`tests/test_numerics.cpp`) replays the sequence with `mpfr_mul`/`mpfr_add`,
`mpfr_fma`, and exact products plus `mpfr_sum` for complex fused steps. It is compared bitwise at
64/224/256/384/1024 bits for degrees 0, 1, 23, 31, 61 and 100 and the empty polynomial, with zero and
alternating coefficients, coefficient and point statuses, a zero point, and `(x-1)^n` near `x = 1`
(cancellation). As a sanity check the jets of `(x-3)^k`, `k = 1, 2, 5, 12`, agree with
`((x-3)^k, k (x-3)^(k-1), C(k,2) (x-3)^(k-2))` to `2^(24-bits)` relatively.

**Performance** (`benchmarks/numerics.cpp`, `benchmarks/results/round32_numerics*.csv`; complex, one
shared coefficient set, composed; medians of 5; GPU wall includes transfers; CPU columns run the same
composed sequence with `mpfr_t`, measured on 4,096 points and scaled; host load average 30–40):

| bits | points | degree | order | GPU wall | MPFR serial | MPFR 18 workers | workers / GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 10^3 | 4 | 0 | 0.9 ms | 1.0 ms | 0.17 ms | 0.19 |
| 256 | 10^3 | 64 | 2 | 7.6 ms | 43 ms | 10.4 ms | 1.4 |
| 256 | 10^5 | 16 | 0 | 5.0 ms | 0.36 s | 48 ms | 9.6 |
| 256 | 10^5 | 64 | 2 | 64 ms | 7.0 s | 0.67 s | 10.5 |
| 256 | 10^6 | 4 | 0 | 13 ms | 1.1 s | 0.30 s | 23 |
| 256 | 10^6 | 64 | 0 | 0.28 s | 19 s | 2.9 s | 10.3 |
| 224 | 10^6 | 16 | 2 | 0.17 s | 27 s | 2.6 s | 15.7 |
| 384 | 10^6 | 64 | 2 | 0.86 s | 70 s | 6.5 s | 7.6 |

The consumer-style MPC loop (`mpc_mul` + `mpc_add`, a different rounding) on 18 workers takes about
the same time as the MPFR pool. Up to about 10^3 points the call is latency-bound and the CPU pool is
faster. Order 2 costs about 3× order 0 (three multiply-adds per coefficient). Fused steps are
1.8–2.2× slower than composed at 256 bits (interleaved, 10^5 points, degree 32); threadgroups of 32,
64, 128 and 256 threads measured within noise.

## Norms and status summaries

```cpp
Segments s{count, length, complex};                     // count segments of length entries, segment-major
nm.norm_inf(bits, s, x, values, info);                  // max |x_i|            (complex: modulus)
nm.norm_max(bits, s, x, values, info);                  // max max(|re|, |im|)  (componentwise, exact)
nm.norm2(bits, s, x, values, info);                     // sqrt(sum |x_i|^2)
nm.scaled_residual(bits, s, r, scale, values, info);    // max |r_i| / |scale_i|, scale real
nm.summarize_status(bits, s, x, info);                  // statuses only
```

Entry `i` of segment `g` is `x[g*length + i]` (`Float<bits>`, or `Complex<bits/32>` with `complex`);
`scale` is always real with the same layout. Each segment gives one real `Float<bits>` value and a
`NormInfo {status, failing, first_failing, index}` (`info` may be null except for `summarize_status`).

**Statuses.** An entry fails when a component (for `scaled_residual` also its scale) has a status, or
when its scaled residual fails (below). `failing` counts failing entries, `first_failing` is the
lowest failing index (`no_index` if none), and `status` is the OR of all entry statuses. A segment
with a failing entry returns `zero(status)` and `index = first_failing`: a failed lane never
disappears from a norm. `status` also includes the value's own status, which can be
`exponent_overflow` when the result lies outside the exponent range although no entry failed.
An empty segment returns canonical zero and `info = {0, 0, no_index, no_index}`; an all-zero segment
returns zero with index 0.

**Maxima and ties.** `norm_inf`, `norm_max` and `scaled_residual` compare per-entry keys exactly (no
rounding in a comparison) and report the lowest index attaining the maximum. The choice "larger key,
lower index on equality" is associative, so the reduction order cannot change it.

- Real `norm_inf` and `norm_max`: key `|x_i|`, exact. Complex `norm_max`: key `max(|re_i|, |im_i|)`, exact.
- Complex modulus: with `e` the larger exponent of the nonzero components (`|c| in [2^e, 2^(e+1))`),
  `a' = |re| 2^-e` and `b' = |im| 2^-e` exactly; a component with exponent below `e - (bits+32)` is
  replaced by zero. Then `t = RN(a'^2)`, `u = RN(b'^2)`, `s = RN(t + u)` (in `[1, 8)`), and the key is
  `q = s 4^e`, an exact scaling held with an extended exponent. `|z| = RN(sqrt(q)) = RN(sqrt(s)) 2^e`.
  The scaling removes the intermediate overflow/underflow of `re^2 + im^2` beyond exponents of about
  ±5·10^8. A dropped component has a square below `2^(-2 bits - 64)`, less than half an ulp of
  `t >= 1`, so it could not have changed `s`. Complex `norm_inf` returns `RN(sqrt(max q_i))`, which
  equals `max_i |z_i|` because the rounded square root is monotone; its index is the lowest among the
  maximal `q_i` (two entries with different `q` that round to the same `|z|` are told apart by `q`).
- `scaled_residual`: real `rho_i = RN(|r_i| / |s_i|)`; complex `rho_i = RN(RN(sqrt(q_i)) / |s_i|)`,
  where the rounded modulus is not range-limited (only `rho_i` must lie in range). `0/0 = 0`; a nonzero
  residual over a zero scale fails with `division_by_zero`; `rho_i` outside the exponent range fails
  with `exponent_overflow`. The value is `max rho_i`.

**2-norm.** `E` is the largest exponent of the nonzero components in the segment. Every component is
scaled by `2^-E` exactly, with the same flush below `-(bits+32)`; the terms are `t_i = RN(y_i^2)`
(real) or `RN(RN(a'^2) + RN(b'^2))` (complex). They are summed by the adjacent-pair tree of `tree_sum`
(level by level `(0,1), (2,3), ...`, each addition rounded, an odd tail copied), and the value is
`RN(sqrt(s)) 2^E`. The largest term is at least 1 and every term is below 4, so no intermediate can
overflow or underflow; only a norm outside the exponent range gives `exponent_overflow`. The flushed
terms total less than `2^(-2 bits - 32)` of the sum (segments shorter than `2^31`). The result is not
correctly rounded: about `ceil(log2 length) + 3` roundings separate it from the exact norm (relative
error of that order in units of `2^-bits`). It is reproducible: it depends only on the inputs, `bits`
and the documented tree.

**Execution.** A segment is split into blocks of `T` entries, `T` a power of two in [32, 128] (one entry
per thread). The first pass computes keys and statuses; SIMD-group reductions and one device atomic
per SIMD group accumulate the OR, the count and the lowest failing index (and, for `norm2`, `E` in an
extra first scan); these are integer operations independent of order. Each threadgroup reduces its
block with the adjacent-pair tree in threadgroup memory; further passes reduce the block roots in the
same way until one root per segment remains, and a one-thread-per-segment pass forms the value (one
`sqrt` per segment for complex `norm_inf` and `norm2`) and `info`. Aligned power-of-two blocks of the
adjacent-pair tree are complete subtrees, so the result does not depend on `T`. All passes share one
command buffer. Squares use `mul(a, a)` (docs/gpu-codegen.md section 8). The reference
(`tests/test_numerics.cpp`) replays every definition with MPFR (`mpfr_sqr`, `mpfr_add`, `mpfr_sqrt`,
`mpfr_div`, exact `mpfr_mul_2si` scalings in an extended exponent range) and is compared bitwise at
64/224/256/384/1024 bits, real and complex, for lengths 1–3, 31–33, 127–129, 1000 and 70,000 (three
tree passes) and 3,000 segments of 7: random data, zeros, ties (`(a,b)`, `(b,a)`, `(-a,-b)`),
exponents near ±10^9, statuses, zero scales and overflowing results.

**Performance** (complex, 10^6 entries in one segment, 256 bits, same benchmark): GPU wall 2–5 ms for
`summarize_status`, `norm_max`, `norm_inf` and `norm2` (device 0.2–2.3 ms; the rest is the host
call and transfers), 10 ms for `scaled_residual` (one square root and one division per entry). The
consumer-style CPU maximum of `mpfr_hypot` takes 1.1 s serial and 0.10 s on 18 workers. 10^4
segments of 100 entries take the same time.

**Resident versions (round 39).** `Numerics(Engine&)` encodes both families into an engine's
`CommandBatch` on `Buffer<T>` operands: `poly_eval(batch, shape, coeffs, points, values)`,
`poly_eval_jet(batch, shape, order, ...)`, `norm_inf` / `norm_max` / `norm2(batch, segments, x, values[, info])`,
`scaled_residual(batch, segments, r, scale, values[, info])` and `summarize_status(batch, segments, x, info)`,
with `values` a `Buffer<Float<bits>>` (one per segment) and `info` a `Buffer<NormInfo>`. They run the
same pipelines and dispatch sequence as the host calls (one code path, `encode_norm_passes` in
`src/numerics.mm`), so results are bit-identical; the summary words are initialised on the host in
fresh per-call scratch (no extra dispatch), block roots use per-call scratch retained until the
submission is waited, and empty segments are written by the finish pass alone. Tested bitwise against
the host calls at 64/224/256/384/1024 bits, real and complex, including three tree passes inside one
batch after an engine operation and chained unit/engine operations (`tests/test_resident_units.cpp`).
Open: the on-device threshold comparison (it fits in the finish pass, so the host would read back only
`info`).

## QR factorization and least squares

```cpp
QROptions o;                                       // FactorOptions (block = 32, host_macs, solve_host_macs, gpu) + rank_bits, pivot
QRFactor qr = la.factor_qr(bits, A, m, n, o);      // A (m x n, m >= n) = Q R; qr.info().rank == n when full rank
                                                   // o.pivot: A P = Q R, P = qr.permutation() (column j of R is column perm[j] of A)
qr.solve(B, nrhs, X);                              // X (n x nrhs) minimises ||A x - b|| for each column b of B (m x nrhs)
qr.apply_q(B, nrhs, transpose);                    // B (m x nrhs) <- Q^T B (transpose) or Q B, in place
qr.r(); qr.v(); qr.tau(); qr.t();                  // R (n x n), V (m x n), tau (n), T (nb x nb per block)
QRFactor lm = la.factor_qr_augmented(bits, J, m, n, d, o); // [J; diag(d)], d = sqrt(mu) D supplied by the caller
```

Host arrays as for the products (row-major `Float<bits>`). `QRFactor` is an owning, move-only object:
`factor_qr` copies A into its own work space and keeps only V, R, tau and T, so A may change or be
freed afterwards. There is no caching by address: when A changes, the caller factors it again (a
new generation is a new `QRFactor`). The factor refers to the `Linalg` that made it (scratch
buffers and the GPU): that `Linalg` must outlive it, and its calls follow the one-thread-per-`Linalg`
rule. `factor_qr_augmented` forms the `(m+n) x n` matrix `[J; diag(d)]` and factors it with the same
code (the solve then takes `B` with `m+n` rows, e.g. `[r; 0]`); it does not exploit the diagonal.

**Rounding sequence.** `P(x, y; S) = RN(sum_{k in S} x_k y_k)` and `D(c; x, y; S) = RN(c - sum_{k in S} x_k y_k)`
are single roundings of exact values (the products of "Dense products" or `exact_dot` /
`exact_dot_add` on the host; identical bits). Blocks are `B_b = [b nb, min(n, (b+1) nb))` with
`nb = block` (0: one block), `k0 = b nb`, `c = j - k0`; `T_b` is the `nb x nb` upper triangular
compact-WY matrix of block b, `V` holds the reflectors `v_j` (zero above row j). `p` is the rank
(n until a column fails).

```
columns j = 0 .. n-1 in order; the reflectors [0, min(j, p)) act on column a_j in groups
G = B_b ∩ [0, min(j, p)), b = 0, 1, ... (the last group is partial inside j's own block); for each group:
  w_i  = P(v_i, a_j; rows [k0, m))                         i in G
  y_i  = P(T_b[k0..i][i], w; [k0, i])
  a_rj = D(a_rj; V[r][G], y; G)                            every row r >= k0 (one rounding per entry)
if j < p, reflector j from x = (a_jj, ..., a_{m-1,j}):
  s = P(x, x)
  column j fails (p = j) if s has a status, x = 0, or rank_bits > 0 and exponent(r_jj) + rank_bits < max_{k<j} exponent(r_kk)
  x_{j+1..m-1} all zero: tau_j = 0, v_j = e_j, r_jj = a_jj  (H_j = I)
  otherwise: beta = -sgn(a_jj) sqrt(s) (sgn(0) = +1); v0 = RN(a_jj - beta); e = exponent(v0);
             v_j = 2^-e (v0, x_{j+1}, ..., x_{m-1}) (exact); tau_j = div(2, P(v_j, v_j)); r_jj = beta
  T_b: g_l = P(v_{k0+l}, v_j) (l < c); T_cc = tau_j; T_lc = -RN(tau_j * P(T_b[l][l..c), g[l..c)))   l = 0 .. c-1
Q^T C (apply_q with transpose, and solve): blocks b = 0, 1, ... over [0, p); per column of C:
  w = P(V_b^T c), y_i = P(T_b[k0..i][i], w), c_r = D(c_r; V[r][B_b], y) for r >= k0
Q C (apply_q): blocks in reverse order with y_i = P(T_b[i][i..], w[i..])
solve: c = Q^T b, then R x = c[0, n) by the backward substitution of "Cholesky" with row i of R in place of
column i of L (blocks nb): x_i = div(D(c_i; R[i], x; ...), r_ii)
```

`H_j = I - tau_j v_j v_j^T` and `Q = H_0 ... H_{n-1}`; block b is `I - V_b T_b V_b^T`. The power of two
`2^-e` makes `v0` lie in `[1, 2)` and every `|v_ij| < 2` without a rounding (LAPACK divides by `v0`
instead); `tau_j` is then in `(1/4, 2]`, and every row of `V_b` and column of `Y` keep the exponent
range of the data, which keeps the residue GEMM to few bands. Since `tau_j` is computed from the
stored `v_j`, `H_j` is orthogonal up to the two roundings of `P(v, v)` and the division.

Each step is correctly rounded; the factorization as a whole is **not**. An entry of column j is
rounded once per group (block) that reaches its row; the correction it receives carries the
roundings of its `w` (one) and `y` (one, over the exact combination of the rounded `w`). The
trailing update of a block applies all of its reflectors at once; inside the panel, column j uses
the partial group `[k0, j)`. A trailing column receiving the whole block gives the same bits as the
group formula because `T_b` is upper triangular and `V_b` is zero above its diagonal, and zero
terms do not change an exact sum. After a failing column the reflectors `[0, p)` still reach every
later column, so rows `< p` of R are final; rows `>= p` of R hold the unreduced remainder in
columns `>= p` (zero elsewhere), and V, tau and T are zero from column p. When p is the first column
of its block, that block has no reflectors and leaves the later columns unchanged (before round 40 an
all-zero block update was still applied, which turned a later column with a status entry into
zero-with-status entries; the sequence above never did).

**Status and rank.** `info().rank = p` and `info().reason`: `zero_column` (the reduced column is
exactly zero), `status_column` (a status in the column; `info().status` is the OR), or
`small_column` (the optional exponent test above, off by default). Exactly dependent columns are
not exactly zero after rounding: they pass with `rank_bits = 0` and fail with a test such as
`rank_bits = bits/2`. A rank-deficient factor still applies Q (its first p reflectors), but `solve`
of an unpivoted factor writes zero with status `invalid` to every entry of X: without pivoting the
failing column says nothing about the numerical rank. Pivoted factors give the basic solution
("Column pivoting" below); no minimum-norm solution is claimed (that needs a complete orthogonal
factorization or SVD, later work). Statuses in B reach X through the dot products, as in the
Cholesky solves.

**Determinism.** The result depends only on the inputs, `bits`, `block` and `pivot`: not on GPU/host
placement (`host_macs`, `solve_host_macs`, `gpu`), host threads, page alignment or the look-ahead
overlap. `block` is part of the result.

**Execution.** The work matrix is a resident GPU buffer; V and T live in page-aligned factor
storage that the GPU uses in place. For block b:

1. Panel on the host (columns contiguous in a scratch copy), two parallel phases per column, each
   over row blocks of about `mr / (4 threads)` rows claimed dynamically (the cores differ in speed):
   - the update of column j by the earlier reflectors of the block (`exact_dot_add`, at most nb
     terms per row); in the same pass every row block sums its part of `x^2` exactly into a window
     anchored at its lowest term. Merging the windows gives the exact `S = sum x_r^2`, so `s = RN(S)`
     and `P(v_j, v_j) = RN(2^-2e (v0^2 + S - x_j^2))` need no serial pass over the column (the exact
     value is the same, hence the same bits; `exact_dot` is used when a window would exceed 128
     words or `v` would leave the exponent range);
   - the Gram column `g` and row c of the panel's W (`P(v_j, a_c')` for the later panel columns,
     unreduced by this block): every thread adds the rows it claims into its own exact accumulator
     per dot, anchored at the product of the two vectors' exponent ranges (which bounds every term),
     and the accumulators are added exactly and rounded once (`exact_dot` beyond 128 words).
   Then column c of `T_b` and the next column's `y`, serially.
2. The next panel's columns are updated by block b: `W = P(V_b^T A)`, `Y = P(T_b^T W)`,
   `A = D(A; V_b, Y)`: three products of "Dense products" (`gemm`-style views of the resident
   buffers, the last with `subtract`), each placed by its own size (exact host dots below
   `host_macs` multiply-adds, so the small `Y` of the next panel avoids a GPU round trip).
3. A second host thread updates the rest of the trailing matrix the same way while this thread
   factors the next panel (one block of look-ahead, as in the Cholesky). A second block of look-ahead
   (round 45: block b's rest overlapping two panels, the calling thread waiting only for its next panel's
   columns) gave identical bits but no gain for the real or complex QR at n = 400/1000: it removed the
   waits for the side thread, but the next panel's update, which is on the critical path with the panel,
   slowed by as much under the concurrent GPU work (`benchmarks/experiments/qr_lookahead_depth2.patch`).

Host phases use a worker pool whose dispatch returns when every item is done, not when every
worker has been scheduled (a worker the loaded host runs late finds no item left). The exact host
dots form products with 64-bit limbs. The solves apply the blocks to a copy of B the same way (GPU
above `solve_host_macs`), then call the backward substitution. The independent reference
(`tests/test_qr.cpp`) is a left-looking, column-by-column MPFR program of the sequence above
(`reference::dot`, `reference::dot_sub`, `mpfr_sqrt`, `mpfr_sub`, `mpfr_div`, `mpfr_mul`, `mpfr_mul_2si`).

### Column pivoting

With `QROptions::pivot` the factorization is column pivoted as LAPACK `xGEQP3`: `A P = Q R`, and R,
V, tau and T are exactly what the sequence above gives for the matrix `A P` (`permutation()[j]` is
the column of A in position j). The permutation is chosen during the factorization from squared
column norms `nu_k` that are downdated after every step, with exact comparisons:

```
nu_k = nuref_k = P(a_k, a_k) over all rows                        every column k, before the first step
step j (block b, c = j - k0), while no column has failed:
  pivot: p = the column in positions [j, n) with the largest nu (a nu with a status ranks above every value;
         exact comparison, no rounding); ties go to the lowest original column index. Swap positions j and p
         (the column, its W and Y rows so far, nu, nuref, perm).
  column j: reduced, tested and turned into reflector j exactly as above (rank_bits is the rank tolerance)
  every trailing column k > j (positions):
    w_k(c) = P(v_j, a_k; rows [j, m))       a_k as at the start of block b
    y_k(c) = P(T_b[k0..j][j], w_k(0..c))
    r_jk   = D(a_jk; V[j][k0..j], y_k(0..c))                       = R[j][k], the final row-j entry
    nu_k   = D(nu_k; r_jk, r_jk)
    if nuref_k != 0, neither has a status, and (nu_k <= 0 or exponent(nu_k) + bits/2 < exponent(nuref_k)):
      nu_k = nuref_k = P(x, x) over rows (j, m) of x_r = D(a_rk; V[r][k0..j], y_k(0..c))
  after the block: X = D(X; V_b, Y) for the trailing columns with the W, Y rows formed above
after a failing column p: no further pivoting; block b's reflectors [k0, p) are applied to columns [p, n) as above
```

`r_jk` is the entry the unpivoted sequence of `A P` puts in R (`V[j][i] = 0` for `i > j`, so the
block update of row j only involves the reflectors up to j), so the downdates use the final row of
R. The recomputation rule is LAPACK's (`tol3z = sqrt(eps)` on the norm ratio) in squared form, with
an exponent comparison. With exact comparisons and the documented tie rule, the permutation is as
deterministic as the rest of the sequence. A column with a status is chosen first, so it fails at
once (`status_column`, rank = its position). Zero columns are moved last. `info().norm_recomputations`
counts the recomputed norms.

**Rank and solve.** `|r_jj|` decreases (up to rounding) along the diagonal, so the first failing
column gives a numerical rank: `rank_bits > 0` stops at the first `|r_pp|` more than `rank_bits`
binades below the largest earlier `|r_kk|` (e.g. `bits/2`, or the problem's own tolerance). For
rank p the solve returns the **basic solution** `x[perm[i]] = z_i` for `i < p`, `x[perm[i]] = 0`
otherwise, where `R[0:p,0:p] z = (Q^T b)[0:p)` is solved by the backward substitution with blocks
`min(nb, p)`, as `xGEQP3` followed by a truncated triangular solve. It is a least-squares solution
for the column space spanned by the selected columns, **not** the minimum-norm solution of the
rank-deficient problem (that needs a complete orthogonal factorization or SVD). A `status_column`
failure gives zero with that status and `invalid` in every entry of X.

**Cost.** The pivot choice needs the row of R in every trailing column after every step, i.e. one
exact dot of length `m - j` per trailing column per step (`~ m n^2 / 2` multiply-adds, done on the
host in column tiles that read the work matrix by rows), as the BLAS-2 half of `xGEQP3`. These are
the W rows of the trailing update, so the GPU only applies `X = D(X; V_b, Y)`; there is no
look-ahead. The independent reference (`ref_qrp` in `tests/test_qr.cpp`) is a right-looking MPFR
program of the pivoted sequence, and full-rank status-free cases are also compared with the
unpivoted reference applied to `A P`.

### QR accuracy

Least squares `min ||J x - b||` with J of 2n × n (`benchmarks/qr.mm --accuracy`,
`benchmarks/results/round31_qr_accuracy.csv`), four right-hand sides, against `x*` from an MPFR
Householder solve at `3*bits + 2 kappa` bits of the same exact J and b. "QSC" is the QSC-like
Jacobian of the Cholesky benchmark (column scales 2^±60, 25% zeros, outliers); `kappa = 40, 100`
is `J = G1 diag(2^(-kappa j/(n-1))) G2` (random G1, G2, one rounding per entry) with the same column
scales, so its singular values spread over about `kappa` more binades. "random" b has a residual of
the size of b; "consistent" b = RN(J x0) has a residual at the rounding level of b. Methods at the
same precision: LimbForge QR (block 32), the normal equations (LimbForge SYRK, `J^T b` by `gemm`,
blocked Cholesky and `cholesky_solve`, round 23), and the consumer-style MPFR Householder
(qsccpp `mx.hpp` `QR`: unblocked, every operation rounded). Errors are log2, maximum over the
right-hand sides: forward normwise `max|x - x*| / max|x*|`, and the residual excess
`||J (x - x*)|| / ||b - J x*||` (how far the residual is above the optimum). n = 400:

| bits | J | b | fwd QR | fwd normal eq. | fwd MPFR Householder | excess QR | excess normal eq. |
|---:|---|---|---:|---:|---:|---:|---:|
| 224 | QSC | random | −222.4 | −220.8 | −218.1 | −222.2 | −222.1 |
| 224 | QSC | consistent | −106.5 | −105.7 | −103.5 | 3.3 | 3.5 |
| 224 | kappa 40 | random | −181.8 | −138.4 | −176.5 | −184.5 | −143.4 |
| 224 | kappa 40 | consistent | −69.0 | −24.9 | −65.9 | 2.9 | 43.5 |
| 224 | kappa 100 | random | −119.2 | −15.9 | −115.8 | −123.9 | −20.4 |
| 224 | kappa 100 | consistent | −9.4 | +98.4 | −5.0 | 2.5 | 104.4 |
| 256 | QSC | random | −254.3 | −253.5 | −250.6 | −254.3 | −254.1 |
| 256 | QSC | consistent | −138.7 | −138.4 | −135.4 | 3.6 | 4.3 |
| 256 | kappa 40 | random | −210.2 | −166.1 | −207.8 | −213.0 | −169.0 |
| 256 | kappa 40 | consistent | −101.2 | −50.3 | −95.9 | 2.8 | 46.5 |
| 256 | kappa 100 | random | −150.3 | −43.3 | −145.8 | −152.1 | −46.8 |
| 256 | kappa 100 | consistent | −41.4 | +71.6 | −36.2 | 2.4 | 107.0 |

n = 200 gives the same picture within 4 bits. On the QSC-like J (badly scaled columns but otherwise
well conditioned) QR and the normal equations agree within 2 bits, and both are 3–4 bits more
accurate than the consumer MPFR Householder. The large errors for consistent b there come from
the 2^120 column-scale ratio, equally for every method. With conditioning beyond the column scaling
the normal equations lose about `kappa` more bits than QR, as `kappa(J)^2` against `kappa(J)`
predicts: 43–44 bits at `kappa = 40` and 101–107 bits at `kappa = 100`. At `kappa = 100` the
normal-equation solution has no correct bits for consistent b (relative error 2^72–2^98), while QR
keeps 9–41 bits, 4–5 bits more than the MPFR Householder. QR's residual stays within a factor of 2^2–2^4 of
the optimum in every case; the normal equations' residual exceeds it by up to 2^107. The componentwise
normal-equation backward error `max_j |J^T r|_j / (|J|^T (|b| + |J| |x|))_j` (in the CSV) is about
`2^-bits` for all three methods, so it does not separate them.

Column-pivoted QR (`round40_qr_accuracy.csv`, n = 200, same J and b; the unpivoted rows there are
identical to round 31's) is within 1.5 bits of unpivoted QR in every case, forward error and residual
excess alike (e.g. 256 bits, `kappa = 100`, consistent b: −43.5 against −43.4; QSC-like random b:
−252.5 against −253.5). These J have full numerical rank, so pivoting buys rank decisions, not accuracy.

## Complex QR factorization and least squares

```cpp
QROptions o;                                          // as for factor_qr: block, host_macs, solve_host_macs, gpu, rank_bits, pivot
ComplexQRFactor qr = la.factor_qr_complex(bits, A, m, n, o); // A: m x n Complex<bits/32>, row-major, m >= n; A = Q R
                                                      // o.pivot: A P = Q R, P = qr.permutation() (column j of R is column perm[j] of A)
qr.solve(B, nrhs, X);                                 // X (n x nrhs) minimises ||A x - b|| for each complex column b of B (m x nrhs)
qr.apply_q(B, nrhs, adjoint);                         // B (m x nrhs) <- Q^H B (adjoint) or Q B, in place
qr.r(); qr.v(); qr.tau(); qr.t();                     // complex R (real diagonal), V, tau, T (nb x nb per block)
qr.info();                                            // QRInfo: rank, reason, status, timings
```

Ownership, threading, options, rank and status rules, and the refusal to solve a rank-deficient unpivoted
factor are those of `QRFactor` ("QR factorization and least squares"). Column pivoting (`QROptions::pivot`,
round 45) follows the real pivoted QR; see "Complex column pivoting" below. Implementation:
`src/linalg_complex.mm`; it reaches the real QR's block update and triangular solve through the private
hooks of `src/linalg_internal.hpp`.

**Convention (LAPACK `zgeqrf` / `zlarfg`).** `H_j = I - tau_j v_j v_j^H` with complex `tau_j`, and
`Q = H_0 H_1 ... H_{n-1}`, so `Q^H A = R`. The reflector of the reduced column `(alpha; x)` satisfies
`H_j^H (alpha; x) = (beta; 0)` with **real** `beta = -sgn(Re alpha) ||(alpha; x)||` (`sgn(0) = +1`).
R therefore has a real diagonal, `r_jj = beta_j`. `H_j` is unitary but not Hermitian. With
`v = (alpha - beta, x)`, `tau = 1 / (beta (beta - conj(alpha)))`, so `1/tau = -beta conj(v0)`. Its real
part is `v^H v / 2` (the unitarity condition); its imaginary part `beta Im v0` fixes the phase.
This convention is preferred over the Hermitian reflector (`beta = -phase(alpha) ||x||`, real
`tau = 2 / v^H v`, as in qsccpp's `mx.hpp` `QR`) for these reasons:

- `beta` needs only the real column norm, exactly as in the real QR, and `v0 = (RN(Re alpha - beta), Im alpha)`
  needs one rounding (its imaginary part is exact). The Hermitian form needs `alpha / |alpha|` (a square root
  and two divisions) and gives a complex diagonal.
- A real diagonal makes the solve divide each component by a real number and lets the `rank_bits` test
  compare real exponents.
- For a real A (zero imaginary parts) the sequence below *is* the real QR's sequence. R, V, tau, T, solutions
  and Q applications equal `Linalg::factor_qr`'s bit for bit in the real parts, with zero imaginary parts
  (tested).
- It is the convention of LAPACK's complex QR (V is scaled differently, see below).

**Complex dot products.** For complex vectors u, w over an index set S:

- `P(u^H w) = (RN(sum u_re w_re + u_im w_im), RN(sum u_re w_im - u_im w_re))`;
- `P(u w) = (RN(sum u_re w_re - u_im w_im), RN(sum u_re w_im + u_im w_re))` (no conjugate);
- `D(c; u, w) = (RN(c_re - sum (u_re w_re - u_im w_im)), RN(c_im - sum (u_re w_im + u_im w_re)))`.

Each component is **one rounding** of its exact value: 2|S| exact products, plus the old component for D.
The operations are `exact_dot` / `exact_dot_add` or the residue products of "Dense products". Statuses:
a component's result is zero with the OR of the statuses of both components of every operand pair (and,
for D, of its own old component).

**Rounding sequence.** Notation as in the real QR (blocks `B_b`, `k0 = b nb`, `c = j - k0`, rank p):

```
columns j = 0 .. n-1; the reflectors [0, min(j, p)) act on column a_j in groups G = B_b ∩ [0, min(j, p)):
  w_i  = P(v_i^H a_j; rows [k0, m))                            i in G
  y_i  = P(T_b[k0..i][i]^H w; [k0, i])                         (conjugated column of T_b)
  a_rj = D(a_rj; V[r][G], y)                                   every row r >= k0, one rounding per component
if j < p, reflector j from (alpha; x) = (a_jj; a_{j+1,j}, ..., a_{m-1,j}):
  s = P((alpha; x)^H (alpha; x))                               (real: the squares of both components)
  column j fails (p = j) if s has a status, alpha = 0 and x = 0, or
    rank_bits > 0 and exponent(beta) + rank_bits < max_{k<j} exponent(r_kk)
  x = 0 and Im alpha = 0: tau_j = 0, v_j = e_j, r_jj = alpha  (H_j = I)
  otherwise: beta = -sgn(Re alpha) sqrt(s); v0 = (RN(Re alpha - beta), Im alpha); e = exponent(Re v0);
             v_j = 2^-e (v0, x) (exact); sigma = P(v_j^H v_j); a = sigma / 2 (exact);
             Im v_j0 = 0 (no status): tau_j = (RN(2 / sigma), 0)                       (the real QR's tau)
             otherwise: b = RN(2^-e beta * Im v_j0); d = RN(a^2 + b^2); tau_j = (RN(a / d), -RN(b / d))
             r_jj = (beta, 0)
  T_b: g_l = P(v_{k0+l}^H v_j) (l < c); T_cc = tau_j; T_lc = -RN(tau_j * P(T_b[l][l..c) g[l..c)))   l = 0 .. c-1
       (the complex product rounded once per component of its exact value, as complex_fma with a zero addend)
Q^H C (apply_q with adjoint, and solve): blocks b = 0, 1, ... over [0, p); per column of C:
  w = P(V_b^H c), y_i = P(T_b[k0..i][i]^H w), c_r = D(c_r; V[r][B_b], y) for r >= k0
Q C (apply_q): blocks in reverse order with y_i = P(T_b[i][i..] w[i..])
solve: c = Q^H b, then R x = c[0, n) backward in blocks nb: u_i = D(...; R[i], x) for one later block at a
  time, then for the rest of i's block; Im x_i = RN(u_im / r_ii) first; the real component's in-block D
  also carries the zero term (-Im r_ii) Im x_i (it can only add a status); Re x_i = RN(u_re / r_ii)
```

`e` makes `Re v0` lie in `[1, 2)`. Since `|Re v0| = |Re alpha| + |beta| >= |beta| >= |alpha|, ||x||`, every
component of `v_j` is below 2 in magnitude without a rounding. `H_j` is unitary up to the roundings of
`sigma`, `d` and the divisions (the imaginary part of `1/tau` does not enter the unitarity condition). Each
step is correctly rounded; the factorization as a whole is not. The real QR's remarks also apply
unchanged: groups, trailing columns receiving whole blocks, rank-deficient remainders, `block` being part of
the result, and placement independence.

**Execution: real embeddings.** The trailing updates and Q applications reuse the real QR's three block
products on real embeddings. In these, every complex output component is one real exact dot of the same
nonzero terms (only zero terms differ, and zero terms do not change an exact sum):

- V and each `T_b` become real matrices of 2×2 blocks `[[re, -im], [im, re]]` (V: 2m × 2n, `T_b`: 2nb × 2nb),
  so the real `V^T` is the embedding of `V^H`;
- the work matrix and right-hand sides are "row split": row 2r holds the real parts of row r and row 2r+1
  the imaginary parts (a 2m × n real matrix), so a contiguous run of rows is a run of [re; im] pairs.

`W = P(V_b^T X)`, `Y = P(T_b^T W)` (or `P(T_b W)` for Q) and `X = D(X; V_b, Y)` on the embeddings are then
`P(V_b^H X)`, `P(T_b^H W)` and `D(X; V_b, Y)` in row-split form, each component rounded once. They run on the
GPU residue products or the host `exact_dot` (identical bits).

The solve is the real blocked backward substitution of `QRFactor::solve` on the 2n × 2n embedding of R, with
blocks of 2nb. Its diagonal blocks are `diag(r_ii, r_ii)` because `Im r_ii = 0`, which gives the
componentwise division above; its order (row 2i+1 before 2i) gives the coupling-term rule.

The host panel stores complex columns contiguously. They are interleaved real sequences
`(re_0, im_0, re_1, ...)`: `Re P(u^H w)` is the real exact dot of the sequences of u and w, `Im P(u^H w)` that
of `i u` and w, and the column update uses `conj(y)` and `i conj(y)`. As in the real panel:

- each row block sums its part of `s` exactly into a window while updating the column;
- `sigma = RN(2^-2e (Re(v0)^2 + S - Re(alpha)^2))` comes from the same exact `S`
  (`|v0|^2 - |alpha|^2 = Re(v0)^2 - Re(alpha)^2`);
- the Gram/W dots go to per-thread exact windows anchored at the product of the operands' exponent ranges;
- `exact_dot` is used when a window would exceed 128 words or v would leave the exponent range.

There is one block of look-ahead, as in the real QR.

**Memory.** Besides the complex V (m × n), R (n × n), T and tau, the factor keeps the embeddings of V
(4mn reals), T and R (4n² reals). For example, at 256 bits (44-byte reals) with m = 2000 and n = 1000: V
takes 176 MB, its embedding 352 MB, and R's embedding 176 MB. During the factorization the row-split work
matrix takes 2mn reals.

**Reference.** `tests/test_qr_complex.cpp` replays the sequence left-looking, column by column, on complex
entries, and never forms an embedding:

- every P and D is two `mpfr_sum` calls over the exact component products (`reference::dot` / `dot_sub`
  on explicit term lists);
- the reflector uses `mpfr_sqrt`, `mpfr_sub`, `mpfr_mul_2si`, `mpfr_div` and `reference::dot2_add`;
- `T` uses `reference::complex_fused`.

**Open.** An augmented `[J; diag(d)]` entry point (for now, stack the rows and call `factor_qr_complex`), and
storing V only once (the complex V is kept for `v()`; the embedding alone would suffice for the products).

### Complex column pivoting

With `QROptions::pivot` the complex factorization is column pivoted as LAPACK `zgeqp3`, in the same way as
the real one ("Column pivoting"): `A P = Q R`, and R, V, tau and T are exactly what the sequence above gives
for the matrix `A P` (`permutation()[j]` is the column of A in position j; `pivoted()`). The squared column
norms are real; they are downdated by `|r_jk|^2` and compared exactly:

```
nu_k = nuref_k = RN(sum_r |a_rk|^2) over all rows (one rounding of 2m exact squares)        every column k
step j (block b, c = j - k0), while no column has failed:
  pivot: p = the column in positions [j, n) with the largest nu (a nu with a status ranks above every value;
         exact comparison, no rounding); ties go to the lowest original column index. Swap positions j and p
         (the column, its W and Y rows so far, nu, nuref, perm).
  column j: reduced, tested and turned into reflector j exactly as above (rank_bits is the rank tolerance)
  every trailing column k > j (positions):
    w_k(c) = P(v_j^H a_k; rows [j, m))                       a_k as at the start of block b
    y_k(c) = P(T_b[k0..j][j]^H w_k(0..c))                    (conjugated column of T_b)
    r_jk   = D(a_jk; V[j][k0..j], y_k(0..c))                 per component; = R[j][k], the final row-j entry
    nu_k   = RN(nu_k - (Re r_jk)^2 - (Im r_jk)^2)            one rounding
    if nuref_k != 0, neither has a status, and (nu_k <= 0 or exponent(nu_k) + bits/2 < exponent(nuref_k)):
      nu_k = nuref_k = RN(sum_r |x_rk|^2) over rows (j, m) of x_rk = D(a_rk; V[r][k0..j], y_k(0..c))
  after the block: X = D(X; V_b, Y) for the trailing columns with the W, Y rows formed above
after a failing column p: no further pivoting; block b's reflectors [k0, p) are applied to columns [p, n)
```

As in the real case, `r_jk` is the entry that the unpivoted sequence of `A P` puts in R (`V[j][i] = 0` for
`i > j`), the recomputation rule is LAPACK's (`tol3z = sqrt(eps)` on the norm ratio) in squared form with an
exponent comparison, and the permutation is as deterministic as the rest of the sequence. A column with a
status in either component is chosen first, so it fails at once (`status_column`, rank = its position).
Zero columns move last. An exactly dependent column (a copy, or `i` times another column) keeps a norm of
rounding size and fails only with a rank test such as `rank_bits = bits/2`.

**Rank and solve.** `|r_jj| = |beta_j|` is real and decreases (up to rounding) along the diagonal, so
`rank_bits > 0` gives a numerical rank. For rank p the solve returns the **basic solution** `x[perm[i]] = z_i`
for `i < p` and `x[perm[i]] = 0` otherwise, where `R[0:p,0:p] z = (Q^H b)[0:p)` is solved by the backward
substitution above on the leading `2p x 2p` block of R's embedding with blocks `2 min(nb, p)` (the same
coupling-term rule). It is not the minimum-norm solution of the rank-deficient problem. A `status_column`
failure gives zero with that status and `invalid` in both components of every entry of X.

**Execution and cost.** As the real pivoted QR, with no look-ahead (the next panel's columns are chosen
during it). The panel loads each column after its pivot step. W row c of every trailing column (two exact
real dots per column over rows `[j, m)` of the block-start work matrix, read by rows in tiles of 16 columns,
each component in an exact window anchored at the product of the exponent ranges; `exact_dot` on the
interleaved sequences beyond 128 words) runs in the same pool dispatch as the Gram windows, after the rows of
`v`, `i v` and the reflector rows are set. Y row c, `r_jk` and the downdate follow per tile. The trailing
update is the embedded `X = D(X; V_b, Y)` with the host-formed Y (the private block-update hook takes
`have_y`). The W rows are about `2 m n^2` real multiply-adds (four per complex product), done on the host
(the BLAS-2 half of `zgeqp3`). On QSC-like J (m = 2n, four right-hand sides, interleaved with the unpivoted
factorization, load 33–48; `benchmarks/results/round45_qr_complex_pivot.csv`) factor + solve takes 1.6–1.7×
the unpivoted time at n = 200, 2.0–2.1× at n = 400 and 2.9–4.2× at n = 1000 (6.2 s at 224 bits, 7.6 s at 256
bits, of which the panel with the W rows is 5.5–6.5 s); the real pivoted QR costs 1.2× at n = 400 and 2.6×
at n = 1000 (round 40). Pivoting moved nearly every column of these J and recomputed no norm.

**Reference.** `ref_qrp` in `tests/test_qr_complex.cpp` is a right-looking MPFR program of the pivoted
sequence on complex entries (the replay functions above; norms as `mpfr_sum` of the exact squares; pivot
comparisons with `mpfr_cmp`). Rank, reason, status, `norm_recomputations`, the permutation, R, V, tau, T,
the basic solutions and the Q / Q^H applications are compared bit for bit, and full-rank status-free inputs
are also compared with the unpivoted replay `ref_qr` of `A P`.

### Complex QR accuracy and speed

Least squares `min ||J x - b||` with complex J of 2n × n (`benchmarks/qr_complex.mm --accuracy`,
`benchmarks/results/round43_qr_complex_accuracy.csv`), four right-hand sides. The reference `x*` is an MPC
Householder solve at `3*bits + 2 kappa` bits (at least twice the working precision) of the same exact J
and b. "QSC" is the QSC-like J of the real benchmark with independent real and imaginary parts.
`kappa = 40, 100` is `J = G1 diag(2^(-kappa j/(n-1))) G2` with complex random G1, G2 and the same column scales.
The methods, all at the same precision, are:

- LimbForge complex QR (block 32);
- the LimbForge real QR of the 2m × 2n real embedding `[[Re J, -Im J], [Im J, Re J]]` with `[Re b; Im b]`;
- the normal equations of that embedding (real SYRK, `gemm`, Cholesky: the complex normal equations in real form);
- the consumer-style MPC Householder (qsccpp `mx.hpp` `QR`: Hermitian reflectors, every operation rounded).

The table gives log2 errors at n = 400, as the maximum over the right-hand sides of the forward normwise
error `max|x - x*| / max|x*|` (complex moduli) and of the residual excess `||J (x - x*)|| / ||b - J x*||`.
n = 200 gives the same picture within 2 bits.

| bits | J | b | fwd complex QR | fwd real QR (embedding) | fwd normal eq. | fwd MPC Householder | excess complex QR | excess normal eq. | excess MPC |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 224 | QSC | random | −221.0 | −221.5 | −220.7 | −218.8 | −222.3 | −221.9 | −219.2 |
| 224 | QSC | consistent | −107.8 | −108.0 | −106.8 | −104.4 | 3.2 | 4.7 | 5.9 |
| 224 | kappa 40 | consistent | −70.6 | −69.5 | −26.0 | −65.2 | 2.5 | 43.2 | 5.8 |
| 224 | kappa 100 | random | −123.2 | −122.8 | −22.7 | −118.9 | −126.0 | −26.8 | −122.6 |
| 224 | kappa 100 | consistent | −11.0 | −11.0 | +93.7 | −7.1 | 2.8 | 102.2 | 5.8 |
| 256 | QSC | random | −253.4 | −253.6 | −252.6 | −250.3 | −254.3 | −253.8 | −251.1 |
| 256 | QSC | consistent | −139.0 | −138.5 | −138.5 | −135.8 | 3.6 | 3.8 | 6.1 |
| 256 | kappa 40 | consistent | −100.4 | −101.2 | −53.4 | −96.8 | 2.6 | 45.1 | 5.9 |
| 256 | kappa 100 | random | −152.1 | −153.9 | −52.4 | −148.0 | −157.6 | −57.7 | −153.7 |
| 256 | kappa 100 | consistent | −42.4 | −43.4 | +62.7 | −39.4 | 2.5 | 103.0 | 5.6 |

The complex QR and the real QR of the embedding agree within 2 bits in every case (the embedding is not
more accurate for doing twice the work). Both are 3–5 bits more accurate than the consumer MPC
Householder, whose residual is 2^5–2^6 above the optimum against 2^2.5–2^3.6 for the complex QR. On the
QSC-like J the normal equations of the embedding are within 1–2 bits of QR. With conditioning beyond the
column scaling they lose about `kappa` bits more (40–45 at `kappa = 40`, about 100 at `kappa = 100`, no
correct bits for consistent b), as for real J.

**Speed** (`benchmarks/results/round43_qr_complex.csv`; metadata `round43_qr_complex_metadata.txt`). The
setting is QSC-like complex J with m = 2n and four right-hand sides; times are factor + solve, as medians
of interleaved repeats (complex QR and the real QR of the embedding alternate). The host load was 20–43, so
differences under ~15% are noise:

| bits | n | complex QR | real QR of the embedding | embedding / complex | MPC 18 threads (factor + solve) | MPC serial |
|---:|---:|---:|---:|---:|---:|---:|
| 224 | 200 | 0.17 s | 0.27 s | 1.58× | 0.64 s (3.7×) | 3.2 s (19×) |
| 256 | 200 | 0.08 s | 0.11 s | 1.46× | 0.71 s (9.4×) | 3.4 s (45×) |
| 224 | 400 | 0.25 s | 0.35 s | 1.38× | 4.7 s (19×) | 27 s* (107×) |
| 256 | 400 | 0.26 s | 0.34 s | 1.32× | 4.7 s (18×) | 26 s* (100×) |
| 224 | 1000 | 1.27 s | 3.12 s | 2.47× | — | 424 s* (335×) |
| 256 | 1000 | 1.67 s (5 repeats; 2.41 s in a 3-repeat run with load up to 45) | 4.00 s | 2.39× | 75 s (45×) | 674 s* (400×) |

`*` marks serial MPC extrapolated from n = 200 by n^3. Complex arithmetic does about half the real
operations of the 2m × 2n embedding's QR, but only in the trailing updates. The panel's exact dots (four
real products per complex product, over half as many columns) cost the same, and the panel is the critical
path: at n = 1000 and 224 bits it takes 0.96 s of the 1.12 s factorization, while the trailing updates
(0.68 s wall, 0.38 s GPU) run behind it on the look-ahead thread. The ratio to the embedding therefore
grows with n, from 1.3–1.6× at n ≤ 400 to 2.4× at n = 1000. The trailing updates use the same residue
products as the real QR. A host-only factorization gives the same bits (checked in the n ≤ 400 timing rows and in the tests).

## Batched products and polynomial-source recurrences

`BatchedLinalg` provides sequential composed/fused operations distinct from `Linalg`'s
single-round exact products. All widths 64–1024 in steps of 32 are supported. The following
sequences are part of the 1.x compatibility contract; performance changes preserve them.
Shapes and buffer/completion rules are in [Execution](execution.md#batched-numerical-operations).

| Operation | Rounding contract | Finality |
|---|---|---|
| `Linalg::gemm` / `syrk` | One rounding per entry from its exact dot | Resident ticket final at wait |
| `BatchedLinalg::gemm` / `power_moments` | Ascending sequential composed or fused dot updates; powers composed | Final within batch |
| `normal_equations` | Ascending-row composed or fused dots | Final within batch |
| `normal_equations_exact` | Augmented exact SYRK, one rounding per dot | Final at wait, after any fallback repair |
| `cholesky_trials` / `BatchedLinalg::cholesky_solve` | Scalar-column updates / sequential multiply-subtract-divide | Final within batch |
| `Linalg::cholesky` / `Linalg::cholesky_solve` | Existing blocked exact-update / triangular-solve contracts above | Synchronous |

### Sequential products and powers

Products visit k in ascending order. By default, complex multiplication uses the existing
four separately rounded real products, followed by rounded real add/subtract, and each
dot update uses a separate rounded addition. `fused=true` uses fma/cfma for each update.
Accumulation starts from the supplied output; `negative=true` negates the completed
GEMM result, including its initial accumulator. Power generation uses composed integer
exponentiation by squaring at n0, then repeated composed multiplication by y; E is
multiplied afterwards. Thus the result has a deterministic **sequence of roundings**,
not the single-round exact-dot contract of `Linalg::gemm`.
`PowerStorage::compact` preserves this sequence exactly: the unscaled power is carried
from one eight-row panel to the next, without reseeding, prefix replay or an extra multiply
at the final row. Panel GEMMs write their own output rows with the same ascending k updates.
Storage mode changes scratch and dispatch count, not precision or rounding.

Integer exponentiation starts from one, visits the absolute exponent's bits low to high,
multiplies the accumulator by the current base for each set bit, and squares the base only
when another bit remains. Negative n0 takes the composed reciprocal of the result. Each row
then multiplies the previous unscaled power by y; multiplying by E does not feed back into
that sequence. In particular, recomputing `powi(y,n0+row)` can give different bits.
`product3` executes the two selected GEMM sequences with a rounded resident intermediate;
the optional final negation applies after the second product.

### Normal equations and damping trials

`normal_equations` computes both JᵀJ and Jᵀg by ascending row-index updates, composed by
default or explicitly fused. `normal_equations_exact` forms the exact Gram matrix of `[J g]`
and extracts JᵀJ and Jᵀg, with the same bits as the corresponding `Linalg` exact products.

`cholesky_trials` factors `A + mu[t]*diag(D)` for all trials. A is a shared row-major
`[n][n]` matrix whose lower triangle is used, D is `[n]`, mu is `[count]`, L is
`[count][n][n]`, and status is `[count]` of `uint32_t`. Columns run in ascending order:
round sqrt of the pivot, round each division, then subtract separately rounded products
from the trailing lower triangle. The diagonal damping update is a separately rounded
multiply and add. This differs from the blocked exact-update `Linalg::cholesky` sequence.

Status is zero on success, otherwise the first unusable pivot plus one (one-based).
Remaining lower factor columns carry `invalid`; the upper triangle is canonical zero.
The solve returns canonical zero with `invalid` for a failed trial. It supports a shared
B `[n][nrhs]` and writes X `[count][n][nrhs]`, with ascending forward and backward dot
indices and separately rounded multiply/subtract/divide. Keep L/status resident for
several solves or use them in subsequent batches after waiting. Inputs and X must differ.

A failed pivot has an arithmetic status or is nonpositive. Factor columns before that pivot
remain as computed; in the remaining lower columns the payload is canonical zero with `invalid`.
For a successful solve, forward rows ascend and backward rows descend. Within each row,
subtraction terms visit column indices in ascending order before the rounded division.

### Polynomial-source rank-one recurrence

`polynomial_recurrence` computes `v <- v + p*(q^T v)` for four-component complex trajectories.

* cp/cq: `[coefficient_sets][4][terms]` in **ascending degree**.
* coefficient_sets: one shared set or `ceil(lanes/lanes_per_weight)` sets.
* y: `[steps][weight_groups]`; Ep/Eq: `[steps][4][weight_groups]`.
* start: component-major `[4][lanes]`; out: the same, or `[steps+1][4][lanes]`.

The sources are `p_a = Ep_a*Horner(cp_a,y)` and `q_a = Eq_a*Horner(cq_a,y)`.
Horner runs from the highest degree down, source scaling uses composed multiplication,
and the rank-one dot visits components 0,1,2,3 before updating any component.
`fused` selects cfma for Horner and recurrence updates. `reverse` applies steps in reverse
coefficient order; `all_steps` writes states in **application order**, starting with the
input state. Adjacent lanes may share weights; the final group may be incomplete.
Zero terms means the zero polynomial. Zero steps copies the initial state.

The transpose is not conjugated. With zero terms, Horner returns canonical zero; the subsequent
source scaling and recurrence operations still execute and propagate arithmetic statuses.
