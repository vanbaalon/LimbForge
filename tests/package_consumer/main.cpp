#include <limbforge/version.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
int main(){
    if(std::string(limbforge::library_version_string())!=LF_EXPECTED_VERSION)return 1;
    bool mismatch=false;try{limbforge::check_library_version();}catch(const std::runtime_error&){mismatch=true;}
#ifdef LF_EXPECT_MISMATCH
    if(!mismatch)return 2;
#else
    if(mismatch)return 3;
#endif
    std::cout<<"Installed library "<<limbforge::library_version_string()<<": header compatibility checked\n";
}
