# Building and integration

## Local build

LimbForge's GPU backend requires macOS 15 or newer, a Metal GPU, CMake 3.20 or
newer, and Apple Command Line Tools. The shared arithmetic header also compiles
as ordinary C++17; the CMake project builds the macOS GPU library.

```sh
brew install cmake mpfr gmp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

Tests run both CPU/MPFR and GPU/MPFR checks. A machine without an exposed Metal
device can build the project and run CPU checks:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLIMBFORGE_ENABLE_GPU_TESTS=OFF
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

This is the GitHub Actions configuration. It validates host arithmetic and
compilation, not GPU shader execution. Run GPU checks locally after kernel changes.

## Library without MPFR

The GPU library has no MPFR/GMP dependency. Disable targets that use the optional
host adapter to build only the library:

```sh
cmake -S . -B build-library -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DLIMBFORGE_BUILD_BENCHMARK=OFF \
  -DLIMBFORGE_BUILD_EXAMPLES=OFF
cmake --build build-library -j4
```

## Embed with CMake

```cmake
add_subdirectory(path/to/LimbForge)
target_link_libraries(my_application PRIVATE LimbForge::limbforge)
```

The target supplies public headers, C++17 requirements, and Metal/Foundation
frameworks. If using `limbforge/mpfr_bridge.hpp`, also supply MPFR/GMP include
paths and link both libraries in your application.

## Install and find the package

```sh
cmake --install build --prefix /path/to/limbforge-install
```

A consumer can then use:

```cmake
find_package(LimbForge 0.1 CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE LimbForge::limbforge)
```

Set `CMAKE_PREFIX_PATH=/path/to/limbforge-install` when configuring the consumer.
The installed library embeds its shader source and does not depend on source
files being present at runtime.

## CMake options

| Option | Default | Purpose |
|---|---|---|
| `BUILD_TESTING` | `ON` | Build MPFR reference tests |
| `LIMBFORGE_ENABLE_GPU_TESTS` | `ON` | Register tests that execute Metal kernels |
| `LIMBFORGE_BUILD_EXAMPLES` | `ON` | Build the division and resident examples |
| `LIMBFORGE_BUILD_BENCHMARK` | `ON` | Build arithmetic benchmarks |
| `LIMBFORGE_BUILD_BAXTER_EXAMPLE` | `OFF` | Build the external BSolver4D comparison |
| `LIMBFORGE_BAXTER_SOURCE_DIR` | Adjacent `BSolver4D/cpp` | Locate the physics reference |

An `Engine` owns cached pipelines, a command queue, and reusable shared buffers.
Use one engine per calling host thread. `Engine::run` accepts host arrays and
blocks until their results are copied back. Typed resident buffers and command
batches support asynchronous execution and dependent arithmetic without repeated
host copies; see [execution and ownership](execution.md).

The external Baxter comparison remains an optional, disabled CMake target for
historical experiments. It is not required by the library, its benchmark matrix,
or its default tests.
