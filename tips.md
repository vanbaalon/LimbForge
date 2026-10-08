# API requests from the twisted-QSC workload

This note lists the features LimbForge would need in order to serve as the GPU backend of the
twisted Quantum Spectral Curve solver `qscmx`. Requests are ordered by expected payoff.
Its source lives in `2026 Near N=4/qsccpp/include/mx.hpp`. Each request states the operation, the sizes involved and
how to check it, written against the API at the time of writing (`ba14968`): elementwise `add/sub/mul/div`,
`complex_add/mul/div`, `square`, `sqrt`, resident `tree_sum`, the scalar four-value `recurrence` with
`states_per_weight`, and `mpfr_bridge.hpp`.

## The workload in one paragraph

One Newton step of the solver evaluates a residual and its Jacobian. For every Chebyshev node j (40 to 70 nodes)
and every column i of a 4×4 complex matrix Q, a four-component complex vector v is carried down Nsh steps
(Nsh = 40 to 200):

    v  <-  v + p_k (q_k · v),        k = Nsh-1, ..., 0

Here p_k and q_k are four-component complex vectors that depend on the node and the step but not on the column.
The Jacobian needs one such chain per unknown (8·Nc ≈ 100 to 500 unknowns), either a full chain or its tangent:

    dv  <-  dv + p_k (q_k · dv) + dp_k (q_k · v) + p_k (dq_k · v)

That gives about nodes × 4 × unknowns ≈ 2·10⁴ to 1.4·10⁵ independent lanes, each Nsh steps long. The working
precision is 200 to 256 bits; 224 and 256 matter most. Everything else, P-series sums, Fourier transforms and the
dense Newton algebra, consists of dot products and one dense linear solve.

## 1. Small-vector matrix recurrence (highest value)

This generalizes `recurrence` from a scalar four-value ring to a four-component vector that is updated by a matrix
at each step. Each GPU thread keeps v, and for the tangent form also dv, in registers through all steps.

Suggested form:

    Timing vector_recurrence(int bits, const void* v0, const void* p, const void* q, const void* r,
                             void* out, std::size_t lanes, unsigned steps,
                             unsigned lanes_per_weight, unsigned flags);
    // per step k:  v <- v + p[k] * (q[k] . v) + r[k]        (complex, 4 components)

- p and q are shared by groups of `lanes_per_weight` lanes. The four Q columns at a node share p and q, and so do
  all Jacobian columns at that node in the full-recompute form.
- r is an optional per-lane affine source. This is the tangent form, with r_k = dp_k (q_k · v_k) + p_k (dq_k · v_k)
  built from a stored base chain.
- An option for a general 4×4 matrix M_k in place of 1 + p_k q_kᵀ.
- An option to write v after every step, not just the final value. The tangent pass needs the base chain v_k.
- Check: compare with MPFR/MPC chains at 224 and 256 bits, 150 steps, on random data with |p_k q_k| ≈ 1. The
  descent is well-conditioned, but do include cancellation-heavy cases.

## 2. Fused complex multiply-add

`out = a*b + c` and `out = a*b - c`, complex and real, with one rounding of the real and imaginary parts if that is
cheap; otherwise state the rounding contract. Nearly every inner loop of the solver is a chain of these. Today each
one costs two dispatches, two roundings and an extra resident buffer.

## 3. Segmented reductions and batched dot products

`tree_sum` reduces a whole buffer to one value. We need thousands of short sums per dispatch:

    out[i] = Σ_{k<K} a[i*K + k] * b[i*K + k]      (segmented dot product, K = 10 to 120)
    out[i] = Σ_{k<K} T[k] * x[i*K + k]            (shared table T: one resident vector for all segments)
    out[m] = Σ_j  F[m, j] * f[j]                  (small dense GEMV/GEMM with a resident matrix: Fourier modes)

Uses:
- the P-series values Σ_n c_n x^{-n} at all shifted points, about 4 × nodes × Nsh segments of length Nc + 1;
- the discrete Fourier transform to modes, with 2·nodes = 80 to 140 points and 100 to 130 modes per function.

Please keep the reduction order fixed, so that results are bitwise reproducible. Finite-difference Jacobians
subtract nearby residuals and are hurt by order-dependent noise.

## 4. Broadcast and gather operands for elementwise operations

The same mechanism as `states_per_weight`, available to every `run`: operand b read as `b[i / stride]`, or as
`b[i % period]`. Without it, shared coefficients have to be copied into lane-sized buffers before each operation.

## 5. Batched 4×4 complex linear algebra

Per-lane LU factorization, solve, inverse and determinant of independent 4×4 complex matrices, e.g.
`Q⁻¹` at every node for every Jacobian column. That is 10⁴ to 10⁵ matrices per Newton step.

## 6. Dense GEMM, SYRK and Cholesky or QR

These are the Newton or Levenberg–Marquardt normal equations: A = JᵀJ with J of size about 2·(8 Nc) × 2·(8 Nc),
i.e. 200 to 1000 square, then a Cholesky or QR solve. On the CPU this is O(N³) at 200 to 256 bits and becomes the
next bottleneck once the Jacobian is cheap. Real-valued routines are enough: the system is solved in real and
imaginary parts.

## 7. A faster binary bridge, and MPC

`from_mpfr` and `to_mpfr` go through `mpz` import and export. Please add:
- a direct limb copy for numbers already rounded to the target precision;
- array versions for whole `mpfr_t` and `mpc_t` arrays (`Complex<N>` from and to `mpc_t`);
- an optional fixed-limb layout matching MPFR custom-allocation numbers, for zero-copy host staging.

Our host type is MPC with inline limbs (`mpfr_custom_init`), at 4 to 8 limbs of 64 bits.

## 8. Lower priority

- Complex `exp`, `log` and integer powers at buffer level. We precompute these on the CPU once per run.
- A documented break-even batch size per operation and precision against 18-thread MPFR. The solver decides per
  kernel whether to offload.
- Asynchronous submissions with a CPU-side callback or polling, so the CPU can build the next Jacobian block
  meanwhile.

## How we will check it

For every kernel we integrate:
1. Lane-by-lane comparison with the existing MPFR/MPC CPU path at the solver's precision.
2. End-to-end: the solver's converged Δ at three reference points (couplings g = 0.1, 0.2 and 0.5) must agree with
   the CPU result to 10⁻²⁵.
3. Speed: wall time per Newton step against the 18-thread CPU engine, at Nc = 23, 31 and 61.

## 9. Second round: what we would like next (8 Oct 2026)

### 9.0 Where we stand

`qscmx` uses LimbForge for the normal equations (build option `-DQSC_LIMBFORGE`, run-time switch `QSC_GPU=1`):
`Linalg::syrk` forms JᵀJ and `Linalg::cholesky` factors the damped matrix, at the working precision rounded up to a
multiple of 32 bits (352 bits for 100 digits). Newton histories and Δ are identical to the MPFR path. Measured at Nc = 59
(m2 ≈ 944 real unknowns, K ≈ 1100 rows): normal matrix 11.8 s → 0.7 s, LM phase 6.9 s → 1.4 s per 3 iterations. Standalone,
SYRK n = 1200, K = 1600 runs in 0.04 s with zero error against MPFR, and Cholesky in 0.14 s. Thank you.

Cost of one Jacobian now (Nc = 59, Nsh = 160, nPts = 65, 352 bits, 16 CPU threads, ≈ 7 s wall, thread-summed below):

| part | thread-s | structure |
|---|---|---|
| adjoint descent tables | ≈ 32 | 520 independent complex products [60 × 160]·[160 × 16]; the left factor is generated (powers) |
| per-column post-processing | ≈ 8 | 30 000 small 4×4 complex products and one 4×4 "inverse derivative" −Q⁻¹ dQ Q⁻¹ each |
| per-column residual | ≈ 32 | Fourier modes of the updated P: a fixed phase matrix times per-column vectors, plus a 6-unknown fit |
| base pass, large-u series | ≈ 3 | one descent per point (4×4 rank-one chain), triangular solves of size ≈ 34 |

The sizes scale like Nc ≈ 40–100, Nsh ≈ 100–250, nPts ≈ Nc + 6, at 352–448 bits (up to 704 for difficult states).

### 9.1 Batched "power-moment" GEMM (highest value)

    // out[b][n][c] (+)= sum_k E[b][k] * y[b][k]^(n + n0) * W[b][k][c],   n = 0..nmax, c = 0..ncols-1, b = 0..count-1
    Timing power_moments(int bits, bool complex, std::size_t count, unsigned steps /*k*/, unsigned nmax, unsigned ncols,
                         int n0, const void* E, const void* y, const void* W, void* out, bool accumulate);

The left factor is a Vandermonde-type matrix that we now build on the CPU (tables of 2.5e6 complex values at Nc = 59);
generating the powers on the device by repeated multiplication along n avoids that upload entirely. Our shape: count = nPts × 8
≈ 400–900, steps = Nsh ≈ 100–250, nmax = Nc ≈ 40–100, ncols = 16, complex. ≈ 8e7–5e8 complex fma per Jacobian.
One rounding per output is not needed; your documented deterministic sequence is enough. A plain **strided-batched complex
GEMM** (count × [m×k]·[k×n] in one dispatch, strides for A, B, C) would also do, if we upload the powers ourselves.

### 9.2 Batched small complex matrix algebra

- Batched 4 × 4 complex products C = A·B and C = A·B·D for 1e4–1e5 independent matrices (`lu4` already gives the inverses).
  We need dQ⁻¹ = −Q⁻¹ dQ Q⁻¹ and Φ·dQ_top per (column, point).
- Batched complex dot/GEMV with a **shared** matrix: out[i] = F · v_i, F of size nm × 2 nPts ≈ 66 × 130 (Fourier phases), for
  8 functions × 470 columns: equivalently one complex GEMM [66 × 130]·[130 × 3800]. The real `gemm` works through the real
  embedding; a native complex GEMM (3 real products, Karatsuba) would halve it.

### 9.3 Resident normal-equation path

- `cholesky` and `cholesky_solve` on device buffers inside a batch (or a resident factor object), so that the chord steps
  (several solves per factor) and the 8 damping trials (8 factorizations of A + μ diag A) stay on the GPU. A batched
  "factor A + μ_t D for t = 1..8" call would be ideal: the trials currently run sequentially on the host.
- `syrk` that also returns Jᵀ g (one extra column) in the same pass.

### 9.4 Descent recurrence with on-device sources (for the base pass and Q functions)

The base pass is the chain Q ← Q + p (qᵀ Q) per point, with p, q the P-functions at u + ik. These are themselves the
sums p_a(u + ik) = E_a(k) Σ_n c_{a,n} y(k)^n. A recurrence variant that evaluates p and q on the fly from (c, y, E) — a
`poly_eval` per step feeding `vector_recurrence(matrix=false)` — would remove another 2.5e6-entry table per Jacobian. Same
for writing all intermediate Q (we need them for the adjoint form): `all_steps` already does this.

### 9.5 Infrastructure

- Run-time width in the host bridge (`from_mpfr(bits, …)` without a template parameter): we dispatch over 25 widths
  with a switch today.
- Zero-copy staging for MPC numbers with inline limbs (`mpfr_custom_init` layout, 6 or 10 limbs of 64 bits).
- Completion callbacks or an event to wait on several submissions, so that the CPU can evaluate the constraints and the
  gluing fit while the GPU forms the Jacobian.
- Fused complex fma above 512 bits (we go to 448–704 bits near degenerate twists).
- A break-even table (size × width) per kernel, so that the solver can choose CPU or GPU without timing each call.

### 9.6 What we will measure

For each item we compare against the MPFR path column by column (the adjoint form already agrees exactly with the
recurrence at working precision) and end to end (Δ to 1e-25), and we report the wall time per Newton iteration at
Nc = 41, 59, 79 for the Z̄X state (twists φ = 5/16, γ = 3/10, g = 2.5).

### 9.7 Implementation status (light audit, 8 Oct 2026)

The standalone `BatchedLinalg` API now provides strided real/complex GEMM, device-generated power moments,
two-stage triple products, normal equations plus rhs (sequential resident and exact augmented-SYRK variants),
resident damping-trial Cholesky and reusable solves, and polynomial-source rank-one recurrences with sharing,
reverse order and all_steps. Runtime-width MPFR/MPC bridges, inline-significand GPU staging and asynchronous
multi-submission waiting are available. A measured dispatch table and small GEMM/power calibration tool are included.

See [docs/section9.md](docs/section9.md) for layout, rounding, ownership and fallback contracts. The new complex
GEMM uses four composed real products; three-product/TensorOps complex GEMM remains an optimization opportunity.
Large-shape calibration and §9.6 solver integration measurements are deferred to the full audit. This work
builds on `3d0eb0c`, which independently lifts the fused Engine recurrence limit to 1024 bits; the new complex
kernels reuse its rolled exact-FMA primitive.
