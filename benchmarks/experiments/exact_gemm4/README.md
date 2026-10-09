# Exact batched 4x4 product prototype

Isolated, uninstalled experiment based on production main `97d6b28` (WolfNum
1.3.1). This is the fair exact-batch candidate for section9 P1.3, not a published
API or a performance claim. Existing library sources and defaults are unchanged.

## Contract and dispatch

Real: each output is RN(C_old + sum of four exact real products), with C_old
omitted without accumulation. Complex: RN of the eight signed real products per
component, plus the optional old component. There is no rounded complex product
or intermediate dot accumulation. Optional negation changes the final sign.
Each component ORs the statuses of all operands used by that component and its
old value, returning canonical zero on failure. Empty/cancelling sums return
canonical zero; exponent overflow follows the existing Number contract.

Matrices are fixed row-major 4x4, with element strides, independent input
broadcasts, nonoverlapping output strides and untouched padding. Host pointers
may overlap because they are staged before encoding; Buffer outputs must differ
from inputs. Supported logical precisions are 64–1024 bits in steps of 32.

One GPU dispatch handles every matrix, with a separate real-component grid and
padded private significands. The accumulator has max(36, round4(2N+8)) words,
with a conservative bound reserving four carry bits and a sign bit for at most
nine terms. Every included product/addend fits exactly; no sticky approximation
or premature rounding is used. Wider exponent spreads trigger per-component
exact host repair at submission wait, using the existing cluster-sum routine.
Status and zero components require no repair.

Inputs are snapshotted by ordered blits before the dispatch, including the old
output when accumulating. This permits earlier on-device producers and later
input writes while preserving the fallback values. Scratch belongs to the batch;
pipelines and completion captures survive unit destruction. Outputs are
provisional until wait: later writes are rejected, reads are reported by the
ticket and should be avoided. Final input operands are required. Reports count
GPU/host components and become available only after successful completion.
Host timing includes staging, snapshots, submission/wait, host repair and output
copy; resident timing includes encoding, snapshots, wait and repair. No buffer
reuse optimization is claimed by this first candidate.

## Independent verification

The test computes products exactly in 2B-bit MPFR and sums them once at B bits.
It does not use the arithmetic core or host fallback as an oracle. Cases include
real/complex, independent broadcasts, strides/padding, accumulation/negation,
statuses, deep cancellation, wide exponent clusters, tie-to-even rounding,
exponent limits, delayed input mutation, unit destruction, destructor wait,
foreign/busy/aliased/undersized buffers, empty work and provisional-output rules.
Dense mode checks 4097 matrices with three resident repeats plus the host call.

Build the explicit targets and start with a focused gate:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_wolfnum_exact_gemm4 -j4
./build/test_wolfnum_exact_gemm4 --bits 352
```

 Before any measured comparison, require all 31
small/dense/wide widths normally and under Metal validation, the normal and Metal
library regressions, and a final frozen-source gate. Benchmark composed/fused,
three-real-GEMM Gauss and exact-batch against their own contracts, reporting host
and resident wall/device scopes separately. Performance/release/consumer
acceptance remains open. A public additive API would require a minor release.
