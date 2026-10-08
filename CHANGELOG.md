# Changelog

## Unreleased

No changes since 1.0.0.

## 1.0.0 — 2026-10-08

First versioned public API baseline, incorporating the development work through section 9.

- Public compile-time version/API macros, constexpr version comparisons, and linked-library
  version queries. An exact startup check detects mismatched headers and archive.
- One canonical version header drives CMake and installed package metadata. Compatible-major
  and exact version requests are supported, with an independent installed-consumer check.
- Documented source/numerical compatibility and deprecation policy, immutable release tags,
  production dependency pinning and contribution rules. Existing entry points are preserved.
- The baseline includes 64–1024-bit arithmetic, fused recurrences through 1024 bits, resident
  buffers, exact dense products, Cholesky/QR, polynomial and transcendental operations,
  batched products/power moments, polynomial-source recurrences, damping trials and bridges.
- Validation for this release metadata: focused CPU version and installed-package checks.
  Section 9 retains its recorded light GPU/MPFR smoke coverage; its full numerical/solver
  audit and large-workload calibration are pending. No new performance claims are made.

## 0.1.0 — Initial development version

- Apple Metal backend for fixed-precision floating-point arithmetic, 64–1024 bits.
- Real addition, subtraction, multiplication, and division; complex arithmetic.
- Fused four-term complex recurrences with shared coefficients.
- Optional MPFR conversion, independent arithmetic tests, and benchmarks.
- Baxter Q-propagation comparison against BSolver4D.
- CMake integration, installed package support, examples, and documentation.
