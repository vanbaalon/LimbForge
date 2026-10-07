#include <limbforge/engine.hpp>
#include <limbforge/mpfr_bridge.hpp>
#include <iostream>
#include <vector>

int main() {
    try {
        constexpr int bits=384;
        using F=limbforge::Float<bits>;
        limbforge::Engine gpu;
        std::vector<F> a(4096,limbforge::from_decimal<bits>("1"));
        std::vector<F> b(4096,limbforge::from_decimal<bits>("3")),out(a.size());
        auto time=gpu.run(bits,limbforge::Operation::div,a.data(),b.data(),out.data(),out.size());
        mpfr_t value;mpfr_init2(value,bits);
        limbforge::to_mpfr<bits>(value,out[0]);
        std::cout<<"Device: "<<gpu.device_name()<<"\n1 / 3 = ";
        mpfr_printf("%.100Rg\n",value);mpfr_clear(value);
        std::cout<<"Batch: "<<out.size()<<" values, GPU wall time: "<<time.wall_seconds<<" s\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"LimbForge: "<<error.what()<<"\n";return 1;
    }
}
