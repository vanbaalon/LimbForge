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

**QR (later).** Plan item S5 prefers QR of `[J; sqrt(mu) D]` for difficult Levenberg–Marquardt
points; the normal equations are solved by the blocked Cholesky below.

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
