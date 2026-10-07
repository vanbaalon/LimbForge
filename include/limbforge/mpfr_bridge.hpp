#pragma once
// Optional CPU conversion layer. Link MPFR/GMP only when using this header.
#include "core.hpp"
#include <mpfr.h>
#include <stdexcept>
#include <string>
namespace limbforge {
template<int Bits> inline Float<Bits> from_mpfr(mpfr_srcptr input) {
    static_assert(Bits>=64&&Bits<=1024&&Bits%32==0,"unsupported precision");
    if(!mpfr_number_p(input))return zero<Bits/32>(invalid);
    if(mpfr_zero_p(input))return zero<Bits/32>();
    mpfr_t x;mpfr_init2(x,Bits);mpfr_set(x,input,MPFR_RNDN);
    mpfr_exp_t exp=mpfr_get_exp(x)-1;
    if(exp>1000000000||exp< -1000000000){mpfr_clear(x);return zero<Bits/32>(exponent_overflow);}
    Float<Bits> r=zero<Bits/32>();r.sign=mpfr_sgn(x);r.exponent=int(exp);
    mpfr_abs(x,x,MPFR_RNDN);mpfr_mul_2si(x,x,Bits-1-exp,MPFR_RNDN);
    mpz_t z;mpz_init(z);mpfr_get_z(z,x,MPFR_RNDN);
    std::size_t count=0;mpz_export(r.limb,&count,-1,sizeof(word),0,0,z);
    mpz_clear(z);mpfr_clear(x);return r;
}
template<int Bits> inline void to_mpfr(mpfr_ptr out,const Float<Bits>& x) {
    if(x.status)throw std::runtime_error("LimbForge arithmetic status "+std::to_string(x.status));
    if(!x.sign){mpfr_set_zero(out,1);return;}
    mpz_t z;mpz_init(z);mpz_import(z,Bits/32,-1,sizeof(word),0,0,x.limb);
    mpfr_set_z(out,z,MPFR_RNDN);mpfr_mul_2si(out,out,x.exponent-(Bits-1),MPFR_RNDN);
    if(x.sign<0)mpfr_neg(out,out,MPFR_RNDN);mpz_clear(z);
}
template<int Bits> inline Float<Bits> from_decimal(const std::string& text) {
    mpfr_t x;mpfr_init2(x,Bits);
    if(mpfr_set_str(x,text.c_str(),10,MPFR_RNDN)){mpfr_clear(x);throw std::invalid_argument("invalid decimal number");}
    auto r=from_mpfr<Bits>(x);mpfr_clear(x);return r;
}
}
