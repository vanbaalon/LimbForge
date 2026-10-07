#include <limbforge/engine.hpp>
#include <limbforge/mpfr_bridge.hpp>
#include <iostream>
#include <vector>
int main(){try{
    using namespace limbforge;using F=Float<384>;Engine gpu;
    constexpr std::size_t count=4096;constexpr int steps=16;
    std::vector<F> values(count,from_decimal<384>("1")),factors(count,from_decimal<384>("1.01"));
    auto x=gpu.make_buffer<F>(count),y=gpu.make_buffer<F>(count);
    x.upload(values.data(),count);y.upload(factors.data(),count);
    auto batch=gpu.batch();for(int i=0;i<steps;++i)batch.run(Operation::mul,x,y,x);
    auto submission=batch.submit();auto timing=submission.wait();x.download(values.data(),count);
    // Independent MPFR validation, preserving rounding after every multiplication.
    mpfr_t expected,factor,actual;mpfr_inits2(384,expected,factor,actual,nullptr);
    mpfr_set_ui(expected,1,MPFR_RNDN);mpfr_set_str(factor,"1.01",10,MPFR_RNDN);
    for(int i=0;i<steps;++i)mpfr_mul(expected,expected,factor,MPFR_RNDN);
    bool correct=true;for(auto value:values){to_mpfr<384>(actual,value);correct&=mpfr_equal_p(actual,expected)!=0;}
    std::cout<<gpu.device_name()<<": "<<count<<" values, "<<steps<<" resident multiplications\n";
    mpfr_printf("Result: %.40Rg\n",actual);
    std::cout<<"GPU: "<<timing.gpu_seconds*1000<<" ms; "<<(correct?"MPFR validation passed":"MPFR mismatch")<<'\n';
    mpfr_clears(expected,factor,actual,nullptr);return correct?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
