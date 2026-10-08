# API versions and production integration

**1.0.0 establishes the current public API baseline.** Releases follow
[Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html):

| Release change | Version increment |
|---|---|
| Compatible fixes or optimizations preserving the documented contract | Patch: 1.0.0 → 1.0.1 |
| Compatible API additions or deprecations | Minor: 1.0.0 → 1.1.0 |
| Incompatible signatures, layouts, defaults or numerical/ownership contracts | Major: 1.x → 2.0.0 |

The contract includes the documented functions and types in installed headers, status values,
`Number`/`Complex` storage representation, precision support, rounding/reduction order, layouts,
and buffer/submission ownership rules. An optimization must preserve its API's documented
rounding sequence; introduce a new operation or explicit option to select a different sequence.
A bug fix restoring that contract can change previously incorrect answers; record it in the
changelog and validate the corrected behavior.

Existing entry points stay source compatible within 1.x. A deprecated API stays available
through that major; document its replacement and removal version. Private implementation
helpers (including undocumented arithmetic helpers), `detail` namespaces, generated shaders,
benchmarks and experimental probes are outside this public compatibility guarantee.
Raising required C++/macOS versions or removing a supported precision is a breaking change.

Compatibility is a **source and documented-behavior guarantee**. Rebuild consumers when
upgrading the static library and use headers and archive from the same release. Stable C++
binary ABI across releases is not promised. The version labels describe API compatibility;
numerical validation coverage and measured performance remain documented separately.

## Pin a production dependency

Use the immutable `v1.0.1` tag, or its commit, rather than a moving main branch:

```sh
git clone --branch v1.0.1 --depth 1 https://github.com/vanbaalon/LimbForge.git
```

With an installed package, require exactly the validated release:

```cmake
find_package(LimbForge 1.0.1 EXACT CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE LimbForge::limbforge)
```

If a project deliberately allows compatible upgrades, use
`find_package(LimbForge 1.0.1...<2.0.0 CONFIG REQUIRED)` or a minimum of `1.0.1`.
The generated package version file only accepts the same major and a sufficient version.
The config exposes `LimbForge_VERSION` and `LimbForge_API_VERSION` (the major).
Consumers previously requesting the unversioned development baseline with `0.1` should
update that requirement to `1.0.1`; source names and signatures are unchanged by this release.

For `add_subdirectory`/FetchContent, pin the same Git tag or commit. The checked-in
`version.hpp` is the single version source: CMake reads its three numeric macros, so installed,
source-tree and header-only consumers see the same release number.

## Compile-time and runtime checks

```cpp
#include <limbforge/version.hpp>

#if LIMBFORGE_API_VERSION != 1 || !LIMBFORGE_VERSION_AT_LEAST(1,0,1)
#error This application requires LimbForge API 1, release 1.0.1 or newer
#endif

int main() {
    // At startup, before constructing Engine:
    limbforge::check_library_version(); // throws on any header/archive version mismatch
    // Useful for logs: limbforge::library_version_string()
}
```

`LIMBFORGE_VERSION_MAJOR`, `_MINOR`, `_PATCH` and `_STRING` identify the headers.
`header_version` is a constexpr `Version`; `library_version()` and
`library_version_string()` report the version compiled into the linked archive.
`version_at_least` orders versions by components, and `version_compatible` additionally
requires equal majors. `require_library_version({1,0,1})` checks a compatible minimum;
pass `true` as the second argument to require an exact match. It throws `std::runtime_error`
with actual and requested numbers when the check fails. These calls create no Metal device.
The original `LIMBFORGE_RESIDENT_API` capability macro retains its meaning.

## Release discipline

Keep unreleased changes in `CHANGELOG.md`. Before a release, choose the appropriate bump,
edit the three literal numeric definitions in `include/limbforge/version.hpp`, move the
changelog entries into a dated release section, and record relevant validation and limits.
API-breaking edits need a migration note and a major bump; do not silently change behavior
in an existing API while keeping the version compatible.

Verify the version and installed-consumer checks (no GPU kernels):

```sh
cmake --build build --target test_limbforge_version -j4
ctest --test-dir build -R '^(version_cpu|package_version_cpu)$' --output-on-failure
```

The package check installs into a temporary build-tree prefix, builds an independent
consumer, rejects mixed headers/archive and unavailable versions, and checks a compatible
major range. Run the numerical tests appropriate to the actual code changes separately.

Commit the release, create an annotated `vMAJOR.MINOR.PATCH` tag on that commit, and push
the commit and tag. Published tags are immutable: corrections get a new release.
The tag pins the complete implementation, numerical contracts, build files and changelog.
Main can contain unreleased changes; its version number alone is not a production pin.
