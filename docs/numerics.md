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

## Transcendental functions

`Transcendentals::run(bits, f, a, out, count, b)` (`transcendental.hpp`, plan D7) evaluates element-wise
real `exp`, `expm1`, `log`, `log1p`, `sin`, `cos`, `atan2(a = y, b = x)` on `Float<bits>` arrays and
complex `exp`, `log` on `Complex<bits/32>` arrays; `Transcendentals::powi(bits, z, k, k_count, out, count)`
computes `z^k`. `transcendental_cpu` gives the same results on the host. Host arrays only; `out` may
alias an input.

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
smallest GPU-validated width, `docs/gpu-codegen.md` section 8: `W = N+2` up to 256 bits, 14 words at
288–384 bits, 18 at 416–512, 22 at 544–640, 26 at 672–768, 29 at 800–832, `N+2` from 864 bits) with the correctly
rounded `core.hpp` primitives, and the code carries a rigorous bound `|y − f(x)| ≤ err · ulp_W(y)`
(`src/transcendental_core.hpp`). `certify` accepts `y` only if the low `W−N` words differ from the
rounding midpoint by more than `err + 1` units (and `err < 2^62`), so every value inside the bound has
the same RN result, including at binade boundaries. Undecided elements are appended to a compacted list
(device atomic counter) and re-evaluated on the host by the same code at the smallest ladder widths
≥ `N+4`, `2N+4` and `4N+8` words (ladder 4…136 words); results outside a rung stay unwritten until a
rung certifies. No input has ever needed more than the third rung; if none decides, the RN value of the
136-word approximation is written (error < ulp/2 + 2^−3000 relative) and `report().unresolved` counts it.
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

**Performance** (`benchmarks/results/round35_transcendental*.csv`, 10⁶ elements, loaded host): GPU wall
17–86 ms at 224–384 bits, 36–91× serial MPFR/MPC and 3–12× an 18-worker pool; at 10⁴ elements the call
is latency-bound (1.3–3.7 ms). First use of a width compiles about 0.4 s of library plus 0.4–2 s per real
function, 1.3–4 s for complex `exp` and 5–14 s for complex `log` and `powi` (OS-cached afterwards; use
`Transcendentals::prewarm`).

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
per component (see "Fused multiply-add"; ~3–4× slower than composed after round 25). With
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

Current limits: shapes are validated before submission. Composed mode is tested at every width
through 1024 bits. Fused mode is limited to 512 bits and rejected above (a fused 1024-bit
specialisation crashed the Metal compiler; 768/1024-bit rank-one variants compiled in 23–34 s). Pipeline compilation for a new
width/flag combination can take minutes on first use, so warm the pipelines before timing.

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

**Resident buffers (later).** The host-array API owns its own device and queue. A resident version
would take `Buffer<Float<bits>>` operands and encode the same pipeline into a `CommandBatch`. Band
analysis needs the exponents on the host, or a small GPU pass over exponent and status words
followed by a readback. The natural form is therefore to run `analyze` once per matrix generation.
Digits, products, combine, reconstruct and finish are then encoded as ordinary dispatches into the
caller's batch, with scratch planes from a workspace query. Fallback lines would then need a host
step after the submission, or a GPU `exact_dot` kernel.

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

**Resident versions (later).** Both families are short dispatch sequences with small scratch, so
`CommandBatch` versions would encode the same pipelines on `Buffer<T>` operands: one dispatch for
`poly_eval_jet`; for the norms the passes above plus a small initialisation dispatch for the summary
words (written by the host today), with block-root scratch of `2 * count * ceil(length/32)` keys and
indices from a workspace query, and `values`/`info` written to resident buffers. A threshold
comparison (value against a caller's tolerance) fits in the final pass, so the host reads back only
`info`. This needs the private `Buffer`/`CommandBatch` internals of `engine.hpp`, which this round
did not modify.
## QR factorization and least squares

```cpp
QROptions o;                                       // FactorOptions (block = 32, host_macs, solve_host_macs, gpu) + rank_bits
QRFactor qr = la.factor_qr(bits, A, m, n, o);      // A (m x n, m >= n) = Q R; qr.info().rank == n when full rank
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
columns `>= p` (zero elsewhere), and V, tau and T are zero from column p.

**Status and rank.** `info().rank = p` and `info().reason`: `zero_column` (the reduced column is
exactly zero), `status_column` (a status in the column; `info().status` is the OR), or
`small_column` (the optional exponent test above, off by default). Exactly dependent columns are
not exactly zero after rounding: they pass with `rank_bits = 0` and fail with a test such as
`rank_bits = bits/2`. A rank-deficient factor still applies Q (its first p reflectors), but `solve`
writes zero with status `invalid` to every entry of X: least squares needs full rank, and no
minimum-norm solution is claimed (column pivoting and a complete orthogonal factorization are later
work, plan S5). Statuses in B reach X through the dot products, as in the Cholesky solves.

**Determinism.** The result depends only on the inputs, `bits` and `block`: not on GPU/host
placement (`host_macs`, `solve_host_macs`, `gpu`), host threads, page alignment or the look-ahead
overlap. `block` is part of the result.

**Execution.** The work matrix is a resident GPU buffer; V and T live in page-aligned factor
storage that the GPU uses in place. For block b:

1. Panel on the host (columns contiguous in a scratch copy): for each column, its `y` and the update
   by the earlier reflectors of the block (exact dots over at most nb terms, rows in parallel), the
   reflector, then the Gram column `g` and row c of the panel's W (`P(v_j, a_c')` for the later
   panel columns, unreduced by this block, in parallel), then column c of `T_b`.
2. The next panel's columns are updated by block b: `W = P(V_b^T A)`, `Y = P(T_b^T W)`,
   `A = D(A; V_b, Y)`: three products of "Dense products" (`gemm`-style views of the resident
   buffers, the last with `subtract`), or exact host dots below `host_macs` multiply-adds.
3. A second host thread updates the rest of the trailing matrix the same way while this thread
   factors the next panel (one block of look-ahead, as in the Cholesky).

The solves apply the blocks to a copy of B the same way (GPU above `solve_host_macs`), then call
the backward substitution. The independent reference (`tests/test_qr.cpp`) is a left-looking,
column-by-column MPFR program of the sequence above (`reference::dot`, `reference::dot_sub`,
`mpfr_sqrt`, `mpfr_sub`, `mpfr_div`, `mpfr_mul`, `mpfr_mul_2si`).

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
