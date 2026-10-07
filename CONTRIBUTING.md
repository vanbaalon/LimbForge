# Contributing to LimbForge

Open an issue with a concrete use case, a reproducible bug, or a proposed
optimization. Small, focused pull requests are easiest to review.

## Development

```sh
brew install cmake mpfr gmp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

GPU checks need a physical Mac with Metal. To run the CPU/MPFR checks without
creating a Metal device, configure with `-DLIMBFORGE_ENABLE_GPU_TESTS=OFF`.
The GitHub Actions workflow uses this mode; local GPU validation remains a
separate requirement for changes to the kernels.

## Arithmetic changes

Keep the shared CPU/Metal implementation compatible with both compilers. Preserve
normalization, nearest-even real rounding, status propagation, and the documented
complex accumulation order. Add independent MPFR cases for a bug or new operation;
tests that merely compare the shared CPU and GPU implementations are insufficient.

For an optimization, report hardware, compiler, precision, batch size, warm-up,
sample count, and both device and wall timings. Include the cost of copies and
preparation when claiming an application speedup. Identify whether the CPU
baseline is single-threaded. Run the Baxter example when changing recurrences.

New features should include a minimal example and update the numeric contract
where relevant. Contributions are distributed under the project's MIT license.
