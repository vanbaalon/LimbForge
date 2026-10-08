#include "limbforge/engine.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
using namespace limbforge;
#if !LIMBFORGE_VERSION_AT_LEAST(1,0,0)
#error Expected the versioned production API
#endif
static_assert(LIMBFORGE_API_VERSION==1,"API baseline");
static_assert(version_at_least({1,10,0},{1,9,99}),"component-wise ordering");
static_assert(!version_at_least({1,9,99},{1,10,0}),"minor ordering");
static_assert(!version_compatible({2,0,0},{1,0,0}),"major isolation");
static_assert(version_compatible({1,1,0},{1,0,99}),"additive compatibility");
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main(){try{
    check_library_version();const auto linked=library_version();
    require(linked.major==header_version.major&&linked.minor==header_version.minor&&linked.patch==header_version.patch,"header/binary mismatch");
    require(std::string(library_version_string())==LIMBFORGE_VERSION_STRING,"version string mismatch");
    require_library_version({LIMBFORGE_API_VERSION,0,0});
    const Version bad[]={{linked.major+1,0,0},{linked.major-1,0,0},{linked.major,linked.minor+1,0},{linked.major,linked.minor,linked.patch+1}};
    for(auto requested:bad){bool rejected=false;try{require_library_version(requested);}catch(const std::runtime_error&){rejected=true;}require(rejected,"incompatible version accepted");}
    require(sizeof(Number<11>)==56&&sizeof(Complex<11>)==112,"352-bit public representation changed");
    require(ok==0&&division_by_zero==1&&exponent_overflow==2&&invalid==4,"public status values changed");
    std::cout<<"LimbForge "<<library_version_string()<<": API/version checks passed (no GPU execution).\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
