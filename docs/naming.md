# WolfNum naming and compatibility

**WolfNum** is the numerical library in the **Wolfbook family**, previously named
LimbForge. It remains a standalone C++/Metal multiprecision library; using it does
not require Wolfbook. Nikolay Gromov is the project author.

Documentation, diagnostics, current benchmark method labels and the CMake project
name use WolfNum. The rename changes no arithmetic, precision, rounding, storage,
ownership or default behavior. The branding is released as compatible patch
**1.3.1**; previously published tags stay unchanged.

## Production integration names

Existing source and build integrations retain their established names within 1.x:

| Interface | Supported spelling |
|---|---|
| Installed headers | `limbforge/*.hpp` |
| C++ namespace and symbols | `limbforge` |
| Version and capability macros | `LIMBFORGE_*` |
| CMake package and imported target | `LimbForge`, `LimbForge::limbforge` |
| Static archive and source-tree target | `liblimbforge.a`, `limbforge` |
| Build options and environment variables | Existing `LIMBFORGE_*` names |
| Test and benchmark executables | Existing `test_limbforge*` and `*_limbforge` names |

The former CMake project variables, including `LimbForge_VERSION`,
`LimbForge_SOURCE_DIR` and `LimbForge_BINARY_DIR`, remain available to embedded
consumers. Continue using the documented package and target names; a `WolfNum`
package or namespace is not introduced by this branding change. Rebuild consumers
with matching headers and archive as usual.

## Repository and recorded evidence

The GitHub repository URL and local checkout directory retain their current names
until the maintainer renames them. Clone commands, CI badge URLs and working paths
therefore still reference the original repository. Update those links after the
repository rename; API compatibility names remain independent of that change.

Published releases, raw benchmark results, metadata, calibration profile strings
and historical experiment patches retain their original names and bytes. They
identify the implementation actually tested. Experiment summaries use WolfNum in
prose while preserving literal historical commands and profile keys. New QR and
Cholesky accuracy output uses `wolfnum_*` method labels; CSV columns are unchanged.
