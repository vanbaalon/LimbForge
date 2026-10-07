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

## Vector recurrences

`Engine::vector_recurrence(bits, shape, start, p, q, r, out, base, dp, dq)` advances many
independent four-component complex vectors (one GPU thread each; plan D4). Each multiply-add `mac(a, b, c)` is either composed, `cadd(c, cmul(a, b))` (default; the rounding of
the existing complex operations), or fused with `shape.fused = true`, `cfma(a, b, c)` with one rounding
per component (see "Fused multiply-add"; currently ~30× slower). With
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

Current limits: shapes are validated before submission; widths through 512 bits are tested. At
1024 bits the Metal compiler failed while specialising this kernel (the exact complex `fma`
workspaces are large); wider widths are not supported yet. Pipeline compilation for a new
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
