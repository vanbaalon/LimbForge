#include "limbforge/version.hpp"
#include <stdexcept>
#include <string>
namespace limbforge {
Version library_version() noexcept {return {LIMBFORGE_VERSION_MAJOR,LIMBFORGE_VERSION_MINOR,LIMBFORGE_VERSION_PATCH};}
const char* library_version_string() noexcept {return LIMBFORGE_VERSION_STRING;}
void require_library_version(Version required,bool exact){
    const auto available=library_version();
    if(version_compatible(available,required) && (!exact ||
       (available.minor==required.minor && available.patch==required.patch)))return;
    throw std::runtime_error(std::string("WolfNum library version ")+library_version_string()+
        (exact?" does not match headers ":" does not satisfy required API version ")+
        std::to_string(required.major)+"."+std::to_string(required.minor)+"."+std::to_string(required.patch));
}
}
