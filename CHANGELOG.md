# Changelog

## Unreleased

## 1.3.1 — 2026-10-09

- Renamed the project to **WolfNum**, the numerical library in the Wolfbook family.
  Updated documentation, comments, diagnostics and benchmark method labels. The
  CMake project is now `WolfNum`; former project variables remain available.
  Existing `limbforge` headers, C++ symbols, version macros, package, build targets,
  options and environment variables retain their production-compatible names.
  Repository URLs, published tags and recorded benchmark evidence retain their
  original identities. Numerical behavior and API major remain unchanged.
  Validation: rebuilt the affected library, example and benchmark targets; all
  three focused CPU version/package/bridge checks pass. An embedded consumer
  also confirms legacy targets and directory variables remain available. Core
  arithmetic and shader sources are byte-identical to the pre-rename baseline.

- Added opt-in normal-equation comparisons and tightened the GEMM benchmark worker
  and timing scopes. Each timed CPU/GPU output is checked against its documented
  contract. Recorded six normal-equation profiles and scoped exact-path guidance;
  an offline exporter adds checked, profile-specific calibration without changing defaults.
  Exact-trial prototypes and their measurements remain experimental;
  they are not installed public APIs in this release.

## 1.3.0 — 2026-10-08

- Added `polynomial_sources` for reusable device p/q tables and explicit
  `PolynomialEvaluation::shared_sources` recurrence overloads. Staged rounded Horner
  states and dot/update passes preserve order, tails, statuses and ownership. Existing
  calls retain per-lane evaluation.
- Added separate `PolynomialPrewarm` requests and synchronous/asynchronous preparation
  methods; the existing `BatchedPrewarm` layout and calls stay compatible.
- All 31 source/recurrence widths pass normally and under shader validation, with
  26/26 normal and 19/19 GPU-validation suites passing. Final API, concurrency, lifetime,
  linked fixtures and version/package checks pass in their focused scopes.
- Correctness-checked 352/384/448-bit measurements show 22–63x lower wall time than
  the old GPU recurrence on the recorded shapes. CPU ratios remain provisional under
  busy-host load; no solver or universal CPU speedup is claimed. Added an offline
  exporter and 50 profile-specific break-even keys; other-operation calibration remains
  pending. See `benchmarks/experiments/section9_shared_sources.md`.

## 1.2.0 — 2026-10-08

- Added reusable intermediates for `BatchedLinalg::product3` and
  `normal_equations_exact`, with `workspaces()` and `release_workspaces()` queries.
  Busy storage survives release until its owning batch is destroyed or waited.
  The arithmetic sequence, provisional-output rules and existing signatures are unchanged.
- Added operation-aware `BatchedPrewarm` and shape-aware `LinalgPrewarm`
  requests, with synchronous and asynchronous preparation on the owning algebra cache.
  Background tasks retain pipeline/table state and the Metal device, while Linalg host
  scratch/worker lifetime stays on its owner thread. Cache access is synchronized;
  numerical shader sources, public layouts, defaults and rounding sequences are unchanged.
- Added explicit contract names: Linalg exact GEMM/SYRK and blocked
  Cholesky/solve; BatchedLinalg sequential GEMM, normal equations and damping
  factor/solve. Existing entry points remain supported with identical defaults,
  validation, status and completion behavior. No arithmetic/backend changes.
- Validation: all three additions pass their combined focused physical-GPU checks
  normally and under shader validation, including busy workspace release, distinct
  exact fallback repairs, prewarm/encode overlap, detached cache lifetime and an
  independent cancellation example distinguishing exact from sequential dots.
  Matching 1.2.0 header/archive and installed-package checks pass 2/2. No numerical
  shader or core arithmetic changed; these are focused host/API checks, not a new
  full arithmetic audit. Raw logs and build hashes are retained. No speedup claim.

## 1.1.0 — 2026-10-08

- Added `PowerStorage::compact` overloads of host-array and resident `power_moments`.
  An eight-row power panel and an unscaled checkpoint replace the full auxiliary table
  when requested. Existing entry points retain full-table storage; public shape/number
  layouts and all rounding/ownership contracts are unchanged. Compact scratch is 6.7–11.2×
  smaller at the measured large shapes, with additional launch costs. Most measured cases
  are slower, so this release makes no power-kernel speedup claim.
- Recorded interleaved full-table/compact benchmarks at 352/384/448 bits, host and resident,
  with seven repeats and independent MPFR checks of initial and every timed output.
  These are busy-host library measurements, not accepted end-to-end solver timings.
- Validation: the numerical candidate passed 26/26 library suites, 19/19 shader-validation
  GPU suites and all-width dense/reference sweeps. The exposed compact API passes its
  31-width power reference sweep normally and under shader validation; focused tests check
  host/resident selection, negative powers, empty reductions and invalid enum rejection.
  Version/package checks are rebuilt for 1.1.0. Raw records distinguish candidate gates
  from final API checks.

## 1.0.1 — 2026-10-08

- Corrected dense complex `BatchedLinalg::gemm` outputs that could have incorrect limbs
  or signs in the original Metal kernels. The implementation uses a private arithmetic
  namespace with padded local storage, uniform component threads and explicit packed
  loads/stores. Logical precision, public layouts, defaults and rounding sequences are
  unchanged. This also corrects the GEMM stages of `product3` and `power_moments`.
- Expanded the opt-in MPFR/MPC audit with repeated dense GEMM, wide exponent gaps,
  damping factors and multiple RHS, polynomial sharing tails and dense inline imports
  at all 31 widths. Retained baseline failures and rejected workaround patches. The
  benchmark now checks every measured output, rather than only its warm-up output.

- Validation: 26/26 CTest suites and 19/19 GPU suites under shader validation pass.
  All 31 widths pass the complete expanded dense audit and broad-exponent GEMM checks,
  normally and under shader validation. Additional valid/status normal-equation and
  negative-power sweeps pass at all widths. Solver-size integration and performance
  calibration remain separate follow-up work.

- Added the section 9 follow-up review and optimization priorities. Aligned planning documents
  with the 1.x compatibility policy, clarified recorded test coverage, and documented power-tile
  rounding and recurrence-sharing constraints. Documentation only; no numerical behavior changes.
- Added an opt-in section 9 MPFR reference runner selecting all 31 widths, expanded arithmetic,
  status and shape cases, and repeated dense GEMM stress cases. The default smoke test remains
  light. The runner preparation itself changed no public API or kernels; the subsequent
  expanded audit exposed and led to the complex GEMM correction above.
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
