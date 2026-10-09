#pragma once
// Canonical release version. CMake reads these three definitions; keep them literal integers.
#define LIMBFORGE_VERSION_MAJOR 1
#define LIMBFORGE_VERSION_MINOR 1
#define LIMBFORGE_VERSION_PATCH 0
#define LIMBFORGE_API_VERSION LIMBFORGE_VERSION_MAJOR
#define LIMBFORGE_DETAIL_STRINGIFY_IMPL(x) #x
#define LIMBFORGE_DETAIL_STRINGIFY(x) LIMBFORGE_DETAIL_STRINGIFY_IMPL(x)
#define LIMBFORGE_VERSION_STRING \
    LIMBFORGE_DETAIL_STRINGIFY(LIMBFORGE_VERSION_MAJOR) "." \
    LIMBFORGE_DETAIL_STRINGIFY(LIMBFORGE_VERSION_MINOR) "." \
    LIMBFORGE_DETAIL_STRINGIFY(LIMBFORGE_VERSION_PATCH)
#define LIMBFORGE_VERSION_AT_LEAST(major, minor, patch) \
    (LIMBFORGE_VERSION_MAJOR > (major) || \
     (LIMBFORGE_VERSION_MAJOR == (major) && \
      (LIMBFORGE_VERSION_MINOR > (minor) || \
       (LIMBFORGE_VERSION_MINOR == (minor) && LIMBFORGE_VERSION_PATCH >= (patch)))))

namespace limbforge {
struct Version { unsigned major,minor,patch; };
inline constexpr Version header_version{LIMBFORGE_VERSION_MAJOR,LIMBFORGE_VERSION_MINOR,LIMBFORGE_VERSION_PATCH};
inline constexpr unsigned api_version=LIMBFORGE_API_VERSION;
constexpr bool version_at_least(Version available,Version required) noexcept {
    return available.major>required.major || (available.major==required.major &&
        (available.minor>required.minor || (available.minor==required.minor && available.patch>=required.patch)));
}
// Source compatibility within a major; headers and binary should still come from the same release.
constexpr bool version_compatible(Version available,Version required) noexcept {
    return available.major==required.major && version_at_least(available,required);
}
Version library_version() noexcept;
const char* library_version_string() noexcept;
// Throws std::runtime_error on a major mismatch, insufficient version, or (exact=true) any mismatch.
void require_library_version(Version required,bool exact=false);
// Pass the caller's header version to the binary: never compare two constants from the binary itself.
inline void check_library_version(){require_library_version({LIMBFORGE_VERSION_MAJOR,LIMBFORGE_VERSION_MINOR,LIMBFORGE_VERSION_PATCH},true);}
}
