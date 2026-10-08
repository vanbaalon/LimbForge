# Changelog

## Unreleased

- Added the section 9 follow-up review and optimization priorities. Aligned planning documents
  with the 1.x compatibility policy, clarified recorded test coverage, and documented power-tile
  rounding and recurrence-sharing constraints. Documentation only; no numerical behavior changes.
- Added an opt-in section 9 MPFR reference runner selecting all 31 widths, expanded arithmetic,
  status and shape cases, and repeated dense GEMM stress cases. The default smoke test remains
  light. Smoke and expanded 352/1024-bit checks passed; full-width, dense and shader-validation
  runs are pending. Public API and kernels unchanged.
- Consolidated section 9 numerical contracts and completion/storage rules into the canonical
  numerics and execution documents, with links from the usage guide and follow-up records.

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
