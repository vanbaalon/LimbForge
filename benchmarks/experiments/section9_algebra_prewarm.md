# Section 9 P3: algebra prewarm candidate

Base `e2d7c15` / compatible numerical baseline 1.1.0, local branch
`round-section9-algebra-prewarm`. No production acceptance or timing claim.

The API prepares the caches actually used by BatchedLinalg power moments and Linalg
SYRK/Cholesky updates, rather than a separate Engine arithmetic cache. Requests select
widths and operation specializations; resident exact requests additionally select K tables.
Public classes retain their existing PImpl layout. Cache state lives independently of
unit host scratch and worker pools, so async compilation can finish after unit/Engine
destruction without moving thread-local host-pool cleanup onto a background thread.
Batched cache access gains a mutex; Linalg preserves its recursive cache locking.

No Metal source or core arithmetic changes. Build and both version/package checks pass
(4.84 s). The focused harness is compiled and prepared for physical GPU checks, after
existing GPU gate jobs finish. It overlaps two prewarm tasks with ordinary owner-thread
encoding, repeats requests, checks host/resident exact SYRK and GPU-updated Cholesky,
real compact 4-row tails, composed/fused polynomial/normal paths and malformed requests,
and waits after both units and the Engine are destroyed. Widths 64/352 exercise numerics;
224 exercises detached-cache lifetime. These checks remain pending, normally and under
shader validation. No full numerical sweep is warranted by an unchanged shader body;
run the focused lifetime/concurrency checks before accepting this host/cache addition.

Build explicitly:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_limbforge_algebra_prewarm -j4
./build/test_limbforge_algebra_prewarm
MTL_SHADER_VALIDATION=1 ./build/test_limbforge_algebra_prewarm
```

The opt-in target is excluded from routine builds/CTest while this is a candidate.
Consumer startup wiring and a compatible minor release remain pending; do not relink the
production solver to this unreleased branch or claim first-Jacobian timing improvements.
