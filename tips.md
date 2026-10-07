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
