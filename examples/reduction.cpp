#include <limbforge/engine.hpp>
#include <limbforge/mpfr_bridge.hpp>
#include <iostream>
#include <vector>
int main(){try{
    using namespace limbforge;using F=Float<384>;Engine gpu;
    constexpr std::size_t count=65537;
    std::vector<F> values(count,from_decimal<384>("1"));
    auto x=gpu.make_buffer<F>(count),total=gpu.make_buffer<F>(1);x.upload(values.data(),count);
    auto batch=gpu.batch();batch.run(Operation::square,x,x);batch.tree_sum(x,total);batch.run(Operation::sqrt,total,total);
    auto timing=batch.submit().wait();F norm;total.download(&norm,1);
    mpfr_t expected,actual;mpfr_inits2(384,expected,actual,nullptr);
    mpfr_set_ui(expected,count,MPFR_RNDN);mpfr_sqrt(expected,expected,MPFR_RNDN);to_mpfr<384>(actual,norm);
    bool correct=mpfr_equal_p(actual,expected)!=0;
    std::cout<<gpu.device_name()<<": norm of "<<count<<" unit values, one resident batch\n";
    mpfr_printf("Norm: %.40Rg\n",actual);
    std::cout<<"GPU: "<<timing.gpu_seconds*1000<<" ms; "<<(correct?"MPFR validation passed":"MPFR mismatch")<<'\n';
    mpfr_clears(expected,actual,nullptr);return correct?0:1;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
