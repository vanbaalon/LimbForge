#pragma once
#include "limbforge/engine.hpp"
#include "limbforge/mpfr_bridge.hpp"
#include <random>
#include <vector>
#include <iostream>
#include <cstring>
namespace reference {
// Keep the oracle finite for products/quotients beyond LimbForge's exponent
// limits, so conversion can report exponent_overflow rather than MPFR infinity.
struct ExponentRange {
    mpfr_exp_t emin=mpfr_get_emin(),emax=mpfr_get_emax();
    ExponentRange(){if(mpfr_set_emin(-3000000000L)||mpfr_set_emax(3000000000L))throw std::runtime_error("MPFR reference exponent range unsupported");}
    ~ExponentRange(){mpfr_set_emax(emax);mpfr_set_emin(emin);}
};
struct MP {
    mpfr_t x;
    explicit MP(int bits){mpfr_init2(x,bits);mpfr_set_zero(x,1);}
    ~MP(){mpfr_clear(x);} MP(const MP&)=delete;
};
template<int Bits> limbforge::Float<Bits> random_number(std::mt19937_64& rng,int exponent_span=1000) {
    auto x=limbforge::zero<Bits/32>();
    for(auto& limb:x.limb)limb=std::uint32_t(rng());
    x.limb[Bits/32-1]|=0x80000000u;x.sign=(rng()&1)?1:-1;
    x.exponent=int(rng()%(2*exponent_span+1))-exponent_span;return x;
}
template<int Bits> bool equal(const limbforge::Float<Bits>& a,const limbforge::Float<Bits>& b) {
    return a.sign==b.sign&&a.exponent==b.exponent&&a.status==b.status&&std::memcmp(a.limb,b.limb,sizeof(a.limb))==0;
}
template<int Bits> bool equal_complex(const limbforge::Complex<Bits/32>& a,const limbforge::Complex<Bits/32>& b) {
    return equal<Bits>(a.re,b.re)&&equal<Bits>(a.im,b.im);
}
template<int Bits> limbforge::Float<Bits> real(limbforge::Operation op,const limbforge::Float<Bits>& a,const limbforge::Float<Bits>& b) {
    if(a.status||(!limbforge::operation_is_unary(op)&&b.status))return limbforge::zero<Bits/32>(a.status|(limbforge::operation_is_unary(op)?0:b.status));
    if(op==limbforge::Operation::div&&!b.sign)return limbforge::zero<Bits/32>(limbforge::division_by_zero);
    ExponentRange range;MP x(Bits),y(Bits),z(Bits);limbforge::to_mpfr<Bits>(x.x,a);if(!limbforge::operation_is_unary(op))limbforge::to_mpfr<Bits>(y.x,b);
    switch(op){case limbforge::Operation::add:mpfr_add(z.x,x.x,y.x,MPFR_RNDN);break;
    case limbforge::Operation::sub:mpfr_sub(z.x,x.x,y.x,MPFR_RNDN);break;
    case limbforge::Operation::mul:mpfr_mul(z.x,x.x,y.x,MPFR_RNDN);break;
    case limbforge::Operation::square:mpfr_sqr(z.x,x.x,MPFR_RNDN);break;
    default:mpfr_div(z.x,x.x,y.x,MPFR_RNDN);}
    return limbforge::from_mpfr<Bits>(z.x);
}
template<int Bits> limbforge::Complex<Bits/32> complex(limbforge::Operation op,limbforge::Complex<Bits/32> a,limbforge::Complex<Bits/32> b) {
    using O=limbforge::Operation;
    auto plus=[](auto x,auto y){return real<Bits>(O::add,x,y);};
    auto minus=[](auto x,auto y){return real<Bits>(O::sub,x,y);};
    auto times=[](auto x,auto y){return real<Bits>(O::mul,x,y);};
    if(op==O::complex_add)return {plus(a.re,b.re),plus(a.im,b.im)};
    if(op==O::complex_mul)return {minus(times(a.re,b.re),times(a.im,b.im)),plus(times(a.re,b.im),times(a.im,b.re))};
    auto d=plus(times(b.re,b.re),times(b.im,b.im));
    return {real<Bits>(O::div,plus(times(a.re,b.re),times(a.im,b.im)),d),
            real<Bits>(O::div,minus(times(a.im,b.re),times(a.re,b.im)),d)};
}
}
