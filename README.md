# LimbForge

**GPU-powered multiprecision arithmetic for Apple Silicon.**

[![CI](https://github.com/vanbaalon/LimbForge/actions/workflows/ci.yml/badge.svg)](https://github.com/vanbaalon/LimbForge/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://en.cppreference.com/w/cpp/17)
[![Backend: Metal](https://img.shields.io/badge/backend-Metal-555555.svg)](https://developer.apple.com/metal/)

LimbForge brings high-precision real and complex arithmetic to the Mac GPU.
It represents numbers with integer limbs and a separate exponent, then executes
batches through precision-specialized Metal kernels. The initial implementation
supports **64–1024 significand bits**, nearest-even real rounding, and fused
four-term complex recurrences.

**Author: Nikolay Gromov.**

> **Development preview:** the implemented operations are tested against MPFR on
> an Apple M5 Max. The API is evolving; this is a focused arithmetic library,
> with transcendental functions and broader linear algebra still to come.

## What it does

- **Real arithmetic:** addition, subtraction, multiplication, and division.
- **Complex arithmetic:** addition, multiplication, and division built from the rounded real operations.
- **Fused recurrences:** propagate independent four-state trajectories in one GPU dispatch, with coefficients shared between trajectories when useful.
- **Selectable precision:** any multiple of 32 bits from 64 through 1024, fixed within a dispatch.
- **MPFR interoperability:** optional host-side conversion without passing through ordinary floating-point values.
- **Reusable execution:** cached pipelines and buffers, with device and wall timings reported separately.

The GPU library depends on Metal and Foundation. MPFR/GMP support the optional
conversion header, tests, and examples; Boost is needed only for the Baxter
comparison.

## Quick start

Use a Mac running **macOS 15 or newer**, with a Metal GPU and Apple Command Line
Tools installed. GPU validation and timings have been performed on an M5 Max;
other Metal devices have not yet been validated.

```sh
git clone https://github.com/vanbaalon/LimbForge.git
cd LimbForge
brew install cmake mpfr gmp

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
./build/limbforge_division
```

The example computes a batch of `1 / 3` values at 384 bits and prints 100 decimal
digits. Shaders are embedded and compiled at runtime; the command-line Metal
compiler and a full Xcode installation are unnecessary.

## A small example

```cpp
#include <limbforge/engine.hpp>
#include <limbforge/mpfr_bridge.hpp>
#include <vector>

limbforge::Engine gpu;  // Reuse this object across dispatches.
using F = limbforge::Float<384>;

std::vector<F> a(65536, limbforge::from_decimal<384>("1"));
std::vector<F> b(65536, limbforge::from_decimal<384>("3"));
std::vector<F> result(a.size());

auto timing = gpu.run(384, limbforge::Operation::div,
                     a.data(), b.data(), result.data(), result.size());
// Inspect result[i].status, then convert with to_mpfr<384>().
```

Link the `LimbForge::limbforge` CMake target. The optional MPFR bridge also needs
MPFR/GMP includes and libraries. A complete executable is in
[examples/division.cpp](examples/division.cpp); integration and installation
instructions are in [docs/building.md](docs/building.md).

Use decimal strings or MPFR values to supply precise inputs. Converting a value
that already passed through `double` cannot restore the missing digits.

## Precision and correctness

| Significand bits | Approximate decimal capacity |
|---:|---:|
| 128 | 39 digits |
| 256 | 77 digits |
| 288 | 87 digits |
| 384 | 116 digits |
| 1024 | 308 digits |

These are representation capacities, not guarantees of final numerical accuracy.
Choose guard precision for cancellation and conditioning. Precision does not
change the exponent range: the representation uses a separate binary exponent
from −1,000,000,000 through +1,000,000,000.

Real operations round to nearest, ties to even. Complex operations compose
those rounded real operations; they do not promise a single correctly rounded
MPC complex result. Arithmetic failures carry explicit status bits. See the
[numerical contract and recurrence layout](docs/numerics.md).

Validation includes **81,920 real-operation cases** at five precisions, comparing
both the shared CPU implementation and actual GPU results with independent MPFR
operations. Additional tests cover complex arithmetic, recurrence trajectories,
shared coefficients, cancellation, halfway rounding, and error propagation.
The [Baxter example](benchmarks/baxter_batch.cpp) compares 600-shift Q-propagation
in both directions with the existing MPFR-based
[BSolver4D](https://github.com/vanbaalon/BSolver4D) solver.

## Measured performance

Representative multiplication measurements on an **Apple M5 Max**, for 65,536
independent values. Times are medians of five warmed runs; the CPU baseline is
**single-threaded MPFR** with preallocated variables.

| Precision | MPFR CPU | GPU wall time | CPU / GPU wall |
|---:|---:|---:|---:|
| 256 bits | 2.08 ms | 0.81 ms | 2.58× |
| 384 bits | 2.93 ms | 1.24 ms | 2.36× |
| 1024 bits | 22.29 ms | 3.00 ms | 7.43× |

GPU wall time includes input/output copies, submission, and synchronization;
shader compilation and decimal conversion are excluded. These measurements do
not compare against MPFR using all CPU cores, and do not establish a full-solver
speedup. Batch size and the cost of preparing data matter.

See [benchmark methodology and results](docs/performance.md) for all operations,
the Baxter experiment, raw records, and reproduction commands.

## Scope and next steps

The current backend is **Apple Metal**. Precision is fixed per dispatch, and
execution is synchronous. The library currently has no transcendental functions,
reductions, matrix solvers, public device-resident asynchronous API, or
multi-GPU backend. GPU multiprecision already exists in other libraries;
[related work](docs/prior-art.md) explains the context and the local Metal use case.

Useful next contributions include broader device validation, improved batch and
memory layouts, device-resident execution, and new operations with independent
MPFR checks. See [CONTRIBUTING.md](CONTRIBUTING.md).

## People and license

Authored by **Nikolay Gromov** and maintained by
[vanbaalon](https://github.com/vanbaalon). See [AUTHORS.md](AUTHORS.md).

LimbForge is distributed under the [MIT license](LICENSE). MPFR, GMP, Boost,
and Apple's frameworks retain their own licenses and terms.
