# LimbForge

**Multiprecision arithmetic on the Apple Silicon GPU.**

[![CI](https://github.com/vanbaalon/LimbForge/actions/workflows/ci.yml/badge.svg)](https://github.com/vanbaalon/LimbForge/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://en.cppreference.com/w/cpp/17)
[![Backend: Metal](https://img.shields.io/badge/backend-Metal-555555.svg)](https://developer.apple.com/metal/)

LimbForge is a standalone library for batched, high-precision real and complex
arithmetic. It stores significands in 32-bit integer limbs and executes
precision-specialized Metal kernels, supporting **64–1024 significand bits** in
steps of 32. Real arithmetic uses round-to-nearest, ties-to-even.

**Author: Nikolay Gromov.**

> **Development preview:** CPU and physical GPU validation use independent MPFR
> operations. The API is evolving, and device validation currently covers an
> Apple M5 Max. See the [numerical contract](docs/numerics.md).

## GPU versus CPU

**82.3× faster than serial MPFR and 6.7× faster than 18-worker MPFR** in the
recorded 1024-bit multiplication-chain benchmark on an Apple M5 Max.
The GPU processes 65,536 independent values through 16 rounded multiplications
each in **3.94 ms**, including input/output transfers, encoding, submission,
and waiting.

The same machine, precision, and batch size also show gains for individual
operations:

| 1024-bit workload | Serial MPFR (ms) | 18-worker MPFR (ms) | GPU wall (ms) | vs serial CPU | vs 18-worker CPU |
|---|---:|---:|---:|---:|---:|
| Real multiplication | 18.39 | 1.90 | 1.62 | **11.3×** | **1.17×** |
| Real division | 40.83 | 3.60 | 2.09 | **19.6×** | **1.73×** |
| Complex division | 231.63 | 18.84 | 7.96 | **29.1×** | **2.37×** |
| 16-step multiplication chain (resident) | 324.10 | 26.38 | 3.94 | **82.3×** | **6.70×** |

Measured October 7, 2026; medians of nine warmed samples. GPU wall times include
transfers; compilation, decimal conversion, and buffer allocation are excluded.
CPU MPFR variables are preallocated, and results are checked against MPFR with
the same real rounding steps. Large batches and data reuse offer the strongest
gains; CPUs can be faster for small batches and simpler operations. See
[full methodology and measurements](docs/performance.md) and the
[raw benchmark data](benchmarks/results/final.csv). Those headline numbers preserve
the Round 1–5 snapshot; [later rounds](docs/optimizations.md) include workgroup
tuning, square, square root, and cooperative reductions, with their own raw
measurements.

## Features

- Correctly rounded real addition, subtraction, multiplication, division, square, and square root.
- Complex addition, multiplication, and division composed from rounded real primitives.
- Fused real `fma`/`fms` and complex `complex_fma`/`complex_fms`, each component rounded once.
- Exact widening and round-to-nearest narrowing casts between any two supported precisions.
- Broadcast (stride/period) operands for element-wise operations, so shared coefficients need no copies.
- Batched 4×4 complex LU with deterministic pivoting: solve, inverse and determinant per matrix.
- Segmented real/complex dot products with a fixed, reproducible fused order and optional shared table.
- Strided real/complex batched GEMM, device-generated power moments, matrix triple products,
  polynomial-source recurrences, and resident damping trials/solves (`batched_linalg.hpp`).
  See [contracts, usage and light-audit scope](docs/section9.md); large-workload timings are pending.
- Dense real `syrk` (AᵀA) and `gemm` with one rounding of each exact dot product (`linalg.hpp`): exact
  integer GEMMs over exponent bands on TensorOps int8 residues, with an exact host fallback; updates
  `C - op(A) B` rounded once, and a blocked Cholesky factorization with multiple-right-hand-side
  triangular solves built on them (a fixed, reproducible rounding sequence).
- Batched real/complex polynomial values and Taylor jets `(p, p', p''/2)` by simultaneous Horner, with
  coefficient sets shared by adjacent points and composed or fused steps (`numerics.hpp`).
- Segmented norms (`norm_inf`, componentwise `norm_max`, exponent-scaled `norm2` on a fixed tree),
  `scaled_residual` and `summarize_status`: exact tie rule (lowest index), failing entries counted and
  never dropped from a norm.
- Correctly rounded element-wise `exp`, `expm1`, `log`, `log1p`, `sin`, `cos`, `atan2` and complex `exp`/`log`
  (`transcendental.hpp`, bit-identical to MPFR/MPC): certified GPU evaluation with GPU retry rungs at more limbs (host only for the rare cases beyond 35 words),
  plus complex integer powers by a documented fused sequence.
- Typed resident buffers, asynchronous submissions, and dependent operations in one command batch, including
  polynomials, norms, transcendental functions and dense `syrk`/`gemm` (`docs/execution.md`, "Resident units").
- In-place pointwise arithmetic, cached pipelines, and explicit buffer ownership checks.
- Fixed-order real and complex tree reductions with GPU-resident intermediate levels.
- Configurable threadgroup sizes and per-pipeline limits for measured tuning.
- A generic four-state complex recurrence primitive (with cooperative multi-lane execution for small batches), and a batched four-component vector recurrence (rank-one, matrix, affine, tangent and all-steps forms).
- Optional MPFR/MPC conversion through decimal strings or direct limb copies, with multithreaded `mpfr_t[]` / `mpc_t[]` array versions.
- Runtime-width bridges, GPU import from external inline MPC significands, asynchronous multi-submission
  waiting, and an offline measured CPU/GPU dispatch table.
- Reproducible benchmarks against both serial and multicore MPFR.

The GPU library depends on Metal and Foundation. MPFR/GMP are needed for the
optional conversion header, tests, benchmarks, and examples.

## Build and test

Use macOS 15 or newer, a Metal GPU, and Apple Command Line Tools:

```sh
git clone https://github.com/vanbaalon/LimbForge.git
cd LimbForge
brew install cmake mpfr gmp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
./build/limbforge_division
./build/limbforge_resident
./build/limbforge_reduction
```

Shaders are embedded and compiled at runtime. The command-line Metal compiler
and a full Xcode installation are unnecessary. Link `LimbForge::limbforge` in a
CMake consumer; [building and installation](docs/building.md) covers integration,
CPU-only validation, and building the GPU library without MPFR.

## Keep a computation on the GPU

```cpp
#include <limbforge/engine.hpp>
#include <limbforge/mpfr_bridge.hpp>
#include <vector>

using namespace limbforge;
using F = Float<384>;
Engine gpu;
std::vector<F> values(65536, from_decimal<384>("1"));
std::vector<F> factors(values.size(), from_decimal<384>("1.01"));
auto x = gpu.make_buffer<F>(values.size());
auto y = gpu.make_buffer<F>(values.size());
x.upload(values.data(), values.size());
y.upload(factors.data(), factors.size());

auto batch = gpu.batch();
for (int step = 0; step < 16; ++step)
    batch.run(Operation::mul, x, y, x);
auto submission = batch.submit();
// Do independent CPU work while the GPU executes.
auto timing = submission.wait();
x.download(values.data(), values.size());
```

Use `Buffer<Complex<12>>` for 384-bit complex values. Buffers stay resident across
batches. Mapping and transfers are rejected until the submission is waited;
previously obtained pointers also require caller coordination. See
[execution, ownership, and timing](docs/execution.md) and the complete
[resident example](examples/resident.cpp). A [norm example](examples/reduction.cpp)
combines square, tree sum, and square root in one GPU submission.

Unary operations use `batch.run(Operation::sqrt, x, x)` or
`batch.run(Operation::square, x, x)`. To sum a resident array, allocate a
one-element output and call `batch.tree_sum(x, total)`; each adjacent pair rounds
separately. See the [rounding contract](docs/numerics.md).

For a single operation, `Engine::run(bits, op, a, b, out, count)` provides a
synchronous host-array interface; `run_unary(bits, op, a, out, count)` handles
square and square root. Supply precise inputs through decimal strings
or MPFR; converting an existing `double` cannot restore lost digits.

## Precision and validation

| Significand bits | Approximate decimal capacity |
|---:|---:|
| 128 | 39 digits |
| 256 | 77 digits |
| 384 | 116 digits |
| 1024 | 308 digits |

Capacity does not guarantee final numerical accuracy: cancellation and
conditioning may require guard precision. The separate binary exponent ranges
from −1,000,000,000 through +1,000,000,000. Arithmetic errors produce explicit
status bits. There are no NaNs, infinities, signed zeros, or subnormal encodings.

Complex arithmetic preserves rounding after each real primitive; it does not
promise a single correctly rounded MPC complex result.

Tests cover **761,856 real-operation cases across all 31 supported precisions**,
complex operations and recurrences, dense 65,536-value complex batches with
repeated GPU execution, and resident-buffer ownership and lifetime checks.
Reduction tests cover every precision, odd tails, cancellation, statuses,
barriers, and scratch-buffer lifetimes. One million scalar quotient/remainder
cases validate the reciprocal used by
division. MPFR provides independent arithmetic references.

## Benchmark and optimization rounds

```sh
./build/benchmark_limbforge > results.csv
./build/benchmark_limbforge --bits 384 --count 65536 --operation div --repeats 9
python3 benchmarks/run_round.py my-round
python3 benchmarks/compare.py benchmarks/results/baseline.csv results.csv
```

The default matrix covers three precisions, three batch sizes, real and complex
operations, and 16-step multiplication chains. It reports device and wall times,
serial MPFR, and a persistent multicore MPFR worker pool. Every result is checked
against MPFR. The round runner builds, tests, and benchmarks in order, preserving
logs and stopping on failure.

The original baseline and every optimization round are committed under
[benchmarks/results](benchmarks/results). Resident batches substantially reduce
repeated transfers and submission costs; exact reciprocal division and reduced
addition storage improve arithmetic throughput. Improvements depend on precision,
batch size, and workload. [Performance measurements](docs/performance.md) include
raw records and limitations; the [optimization log](docs/optimizations.md) also
records rejected experiments.

## Scope

The current backend is Apple Metal, with fixed precision per dispatch. Further
work includes wider device validation, cooperative SIMD arithmetic, fused
multiply-add, and transcendental functions with explicit accuracy contracts.
These features are not implemented. Alternate layouts have been measured as an
experiment; the public representation remains an array of structs.
GPU multiprecision has prior art; [related work](docs/prior-art.md) explains the
context and the Metal target.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the validation and measurement workflow.
Authored by **Nikolay Gromov** and maintained by
[vanbaalon](https://github.com/vanbaalon); see [AUTHORS.md](AUTHORS.md).
LimbForge is distributed under the [MIT license](LICENSE).
