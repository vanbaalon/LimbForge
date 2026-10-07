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

## Features

- Real addition, subtraction, multiplication, and division.
- Complex addition, multiplication, and division composed from rounded real primitives.
- Typed resident buffers, asynchronous submissions, and dependent operations in one command batch.
- In-place pointwise arithmetic, cached pipelines, and explicit buffer ownership checks.
- A generic four-state complex recurrence primitive.
- Optional MPFR conversion through decimal strings or exact binary values.
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
[resident example](examples/resident.cpp).

For a single operation, `Engine::run(bits, op, a, b, out, count)` provides a
synchronous host-array interface. Supply precise inputs through decimal strings
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

Tests cover **507,904 real-operation cases across all 31 supported precisions**,
complex operations and recurrences, dense 65,536-value complex batches with
repeated GPU execution, and resident-buffer ownership and lifetime checks.
One million scalar quotient/remainder cases validate the reciprocal used by
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
work includes wider device validation, cooperative arithmetic and memory-layout
experiments, reductions, fused multiply-add, square root, and transcendental
functions with explicit accuracy contracts. These features are not implemented.
GPU multiprecision has prior art; [related work](docs/prior-art.md) explains the
context and the Metal target.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the validation and measurement workflow.
Authored by **Nikolay Gromov** and maintained by
[vanbaalon](https://github.com/vanbaalon); see [AUTHORS.md](AUTHORS.md).
LimbForge is distributed under the [MIT license](LICENSE).
