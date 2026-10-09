# WolfNum contribution rules

Use WolfNum for project prose and diagnostics. Preserve existing production API,
package and build identifiers and historical measurement records as described in
`docs/naming.md`; do not mechanically rename `limbforge` symbols or profile keys.

The current public API baseline is 1.0.0. Read `docs/versioning.md` before changing an
installed header, documented numerical behavior, storage format, or ownership contract.

- Preserve source compatibility and documented behavior within a major release. Changing
  rounding/reduction order or defaults needs an explicit new option/API or a major release.
- Record user-visible changes, corrections and validation in `CHANGELOG.md` under Unreleased.
- The three numeric macros in `include/limbforge/version.hpp` are the canonical release
  version. CMake derives its version from them. Bump them when preparing a release; document
  migrations for breaking releases. Do not repoint published version tags.
- Keep old APIs available through their major when deprecating them, with a documented replacement.
- Rebuild consumers with matching headers/archive; do not claim binary ABI compatibility.
- Run the focused version/package tests after versioning or packaging changes, and tests
  appropriate to other changes. Honor user instructions to limit testing.
