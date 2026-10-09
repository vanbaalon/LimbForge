#pragma once
// Optional CPU conversion layer. Link MPFR/GMP only when using this header. The mpc_t functions need only <mpc.h>
// on the include path (they use no libmpc symbols); define LIMBFORGE_NO_MPC to leave them out.
#include "core.hpp"
#include <mpfr.h>
#if !defined(LIMBFORGE_NO_MPC)&&__has_include(<mpc.h>)
#include <mpc.h>
#define LIMBFORGE_HAS_MPC 1
#endif
#include <algorithm>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>
namespace limbforge {
namespace detail {
// Reference implementations through mpz; the fast paths below must agree with them bit for bit (tests/test_bridge.cpp).
template<int Bits> inline Float<Bits> from_mpfr_slow(mpfr_srcptr input) {
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
template<int Bits> inline void to_mpfr_slow(mpfr_ptr out,const Float<Bits>& x) {
    if(x.status)throw std::runtime_error("WolfNum arithmetic status "+std::to_string(x.status));
    if(!x.sign){mpfr_set_zero(out,1);return;}
    mpz_t z;mpz_init(z);mpz_import(z,Bits/32,-1,sizeof(word),0,0,x.limb);
    mpfr_set_z(out,z,MPFR_RNDN);mpfr_mul_2si(out,out,x.exponent-(Bits-1),MPFR_RNDN);
    if(x.sign<0)mpfr_neg(out,out,MPFR_RNDN);mpz_clear(z);
}
// A 64-bit little-endian MPFR limb is two WolfNum words, low half first, so significands map by memcpy.
#if defined(__BYTE_ORDER__)&&__BYTE_ORDER__==__ORDER_LITTLE_ENDIAN__&&GMP_NUMB_BITS==64&&GMP_NAIL_BITS==0
constexpr bool direct_limbs=true;
#else
constexpr bool direct_limbs=false;
#endif
struct Range { mpfr_exp_t emin,emax; }; // MPFR's exponent range is per thread
inline Range current_range(){return {mpfr_get_emin(),mpfr_get_emax()};}
inline word load_word(const void* p,std::size_t i){word w;std::memcpy(&w,static_cast<const char*>(p)+4*i,4);return w;}
inline std::size_t significand_words(mpfr_prec_t p){return 2*((std::size_t(p)+63)/64);}
// RN to Bits by truncating whole words (the MPFR significand is left aligned and Bits%32==0); identical to
// from_mpfr_slow including the inexact flag. Inputs outside the current MPFR range, a carry past emax, or a range
// that cannot hold the slow path's integer scaling take the slow path.
template<int Bits> inline Float<Bits> from_mpfr_fast(mpfr_srcptr input,Range range) {
    static_assert(Bits>=64&&Bits<=1024&&Bits%32==0,"unsupported precision");
    constexpr std::size_t N=Bits/32;
    if(!direct_limbs||!mpfr_regular_p(input))return from_mpfr_slow<Bits>(input);
    mpfr_exp_t e=mpfr_get_exp(input);
    if(e<range.emin||e>range.emax||range.emin>Bits||range.emax<Bits)return from_mpfr_slow<Bits>(input);
    const void* s=mpfr_custom_get_significand(input);std::size_t M=significand_words(mpfr_get_prec(input));
    Float<Bits> r;r.status=ok;r.sign=mpfr_signbit(input)?-1:1;
    if(M<=N){std::memset(r.limb,0,4*(N-M));std::memcpy(r.limb+(N-M),s,4*M);}
    else{std::size_t low=M-N;std::memcpy(r.limb,static_cast<const char*>(s)+4*low,4*N);
        word g=load_word(s,low-1);bool sticky=g&0x7fffffffu;for(std::size_t i=0;!sticky&&i+1<low;++i)sticky=load_word(s,i);
        if((g>>31)||sticky)mpfr_set_inexflag();
        if((g>>31)&&(sticky||(r.limb[0]&1))){std::size_t i=0;while(i<N&&!++r.limb[i])++i;
            if(i==N){r.limb[N-1]=word(1)<<31;if(e==range.emax)return from_mpfr_slow<Bits>(input);++e;}}}
    if(e-1>1000000000||e-1< -1000000000)return zero<N>(exponent_overflow);
    r.exponent=int(e-1);return r;
}
// Writes the words into the top of the significand, zeroes the rest; a narrower output is rounded RN at bit
// position prec exactly as mpfr_set_z does. Results outside the current MPFR range take the slow path.
template<int Bits> inline void to_mpfr_fast(mpfr_ptr out,const Float<Bits>& x,Range range) {
    constexpr int N=Bits/32;
    if(x.status)throw std::runtime_error("WolfNum arithmetic status "+std::to_string(x.status));
    if(!x.sign){mpfr_set_zero(out,1);return;}
    if(!direct_limbs)return to_mpfr_slow<Bits>(out,x);
    mpfr_prec_t p=mpfr_get_prec(out);std::size_t D=significand_words(p);mpfr_exp_t e=mpfr_exp_t(x.exponent)+1;
    char* s=static_cast<char*>(mpfr_custom_get_significand(out));
    if(p>=Bits){
        if(e<range.emin||e>range.emax||range.emin>Bits||range.emax<Bits)return to_mpfr_slow<Bits>(out,x);
        std::memset(s,0,4*(D-N));std::memcpy(s+4*(D-N),x.limb,4*N);
    }else{
        word q[N];std::memcpy(q,x.limb,sizeof q);int cut=Bits-int(p),rw=(cut-1)/32,rb=(cut-1)%32,lw=cut/32,lb=cut%32;
        bool round=(q[rw]>>rb)&1,sticky=q[rw]&((word(1)<<rb)-1),odd=(q[lw]>>lb)&1;
        for(int i=0;!sticky&&i<rw;++i)sticky=q[i];
        for(int i=0;i<lw;++i)q[i]=0;q[lw]&=~((word(1)<<lb)-1);
        if(round&&(sticky||odd)){dword c=dword(q[lw])+(word(1)<<lb);q[lw]=word(c);bool carry=c>>32;
            for(int i=lw+1;carry&&i<N;++i)carry=!++q[i];if(carry){q[N-1]=word(1)<<31;++e;}}
        if(e<range.emin||e>range.emax||range.emin>Bits||range.emax<=Bits)return to_mpfr_slow<Bits>(out,x);
        if(round||sticky)mpfr_set_inexflag();
        if(D>=std::size_t(N)){std::memset(s,0,4*(D-N));std::memcpy(s+4*(D-N),q,4*N);}else std::memcpy(s,q+(N-D),4*D);
    }
    mpfr_custom_init_set(out,x.sign<0?-MPFR_REGULAR_KIND:MPFR_REGULAR_KIND,e,p,s);
}
// Runs f(begin,end,range) over contiguous chunks. threads=0 gives each thread at least `grain` values (one thread
// below 2*grain), capped at the core count; conversions are memory bound, so on M5 Max ~2^22 bits of values per
// thread is the measured sweet spot. Workers adopt the caller's MPFR exponent range; the lowest failing chunk's
// exception is rethrown after all chunks finish.
template<class F> inline void parallel_chunks(std::size_t n,unsigned threads,std::size_t grain,F&& f) {
    Range range=current_range();
    if(!threads)threads=n<2*grain?1:unsigned(std::min<std::size_t>(std::max(1u,std::thread::hardware_concurrency()),n/grain));
    threads=unsigned(std::min<std::size_t>(threads,n));
    if(threads<=1){if(n)f(std::size_t(0),n,range);return;}
    std::size_t chunk=(n+threads-1)/threads;std::vector<std::exception_ptr> errors(threads);std::vector<std::thread> pool;
    for(unsigned t=1;t<threads;++t){std::size_t b=t*chunk,end=std::min(n,b+chunk);if(b>=end)break;
        try{pool.emplace_back([&f,&errors,range,t,b,end]{
            try{mpfr_set_emin(range.emin);mpfr_set_emax(range.emax);f(b,end,range);}catch(...){errors[t]=std::current_exception();}
            mpfr_free_cache2(MPFR_FREE_LOCAL_CACHE);});}
        catch(const std::system_error&){try{f(b,end,range);}catch(...){errors[t]=std::current_exception();}}} // no thread: run inline
    try{f(std::size_t(0),std::min(n,chunk),range);}catch(...){errors[0]=std::current_exception();}
    for(auto& worker:pool)worker.join();
    for(auto& error:errors)if(error)std::rethrow_exception(error);
}
}
// RN to Bits (ties to even); NaN/Inf -> invalid, zero -> canonical zero, |exponent| > 1e9 -> exponent_overflow.
template<int Bits> inline Float<Bits> from_mpfr(mpfr_srcptr input){return detail::from_mpfr_fast<Bits>(input,detail::current_range());}
// Exact when mpfr_get_prec(out) >= Bits, otherwise RN to the output precision. Throws on a nonzero status.
template<int Bits> inline void to_mpfr(mpfr_ptr out,const Float<Bits>& x){detail::to_mpfr_fast<Bits>(out,x,detail::current_range());}
template<int Bits> inline void from_mpfr_array(const mpfr_t* in,Float<Bits>* out,std::size_t n,unsigned threads=0) {
    detail::parallel_chunks(n,threads,(std::size_t(1)<<22)/Bits,[&](std::size_t b,std::size_t e,detail::Range r){for(std::size_t i=b;i<e;++i)out[i]=detail::from_mpfr_fast<Bits>(in[i],r);});
}
// Same element results as to_mpfr; on a nonzero status the elements of other chunks may already be written.
template<int Bits> inline void to_mpfr_array(mpfr_t* out,const Float<Bits>* in,std::size_t n,unsigned threads=0) {
    detail::parallel_chunks(n,threads,(std::size_t(1)<<22)/Bits,[&](std::size_t b,std::size_t e,detail::Range r){for(std::size_t i=b;i<e;++i)detail::to_mpfr_fast<Bits>(out[i],in[i],r);});
}
#ifdef LIMBFORGE_HAS_MPC
template<int Bits> inline Complex<Bits/32> from_mpc(mpc_srcptr z){auto r=detail::current_range();return {detail::from_mpfr_fast<Bits>(mpc_realref(z),r),detail::from_mpfr_fast<Bits>(mpc_imagref(z),r)};}
namespace detail {
template<int Bits> inline void to_mpc_fast(mpc_ptr out,const Complex<Bits/32>& x,Range r) {
    if(x.re.status|x.im.status)throw std::runtime_error("WolfNum arithmetic status "+std::to_string(x.re.status|x.im.status));
    to_mpfr_fast<Bits>(mpc_realref(out),x.re,r);to_mpfr_fast<Bits>(mpc_imagref(out),x.im,r);
}
}
// Each part converted as to_mpfr; throws before writing anything if either part has a nonzero status.
template<int Bits> inline void to_mpc(mpc_ptr out,const Complex<Bits/32>& x){detail::to_mpc_fast<Bits>(out,x,detail::current_range());}
template<int Bits> inline void from_mpc_array(const mpc_t* in,Complex<Bits/32>* out,std::size_t n,unsigned threads=0) {
    detail::parallel_chunks(n,threads,(std::size_t(1)<<21)/Bits,[&](std::size_t b,std::size_t e,detail::Range r){
        for(std::size_t i=b;i<e;++i)out[i]={detail::from_mpfr_fast<Bits>(mpc_realref(in[i]),r),detail::from_mpfr_fast<Bits>(mpc_imagref(in[i]),r)};});
}
template<int Bits> inline void to_mpc_array(mpc_t* out,const Complex<Bits/32>* in,std::size_t n,unsigned threads=0) {
    detail::parallel_chunks(n,threads,(std::size_t(1)<<21)/Bits,[&](std::size_t b,std::size_t e,detail::Range r){for(std::size_t i=b;i<e;++i)detail::to_mpc_fast<Bits>(out[i],in[i],r);});
}
#endif
// Runtime-width bridge. Buffers contain the same Float<bits>/Complex<bits/32> structs as the typed API;
// caller-provided byte storage needs bridge_element_bytes(bits, complex)*count bytes. No alignment required.
inline std::size_t bridge_element_bytes(int bits,bool complex=false){
    if(bits<64||bits>1024||bits%32)throw std::invalid_argument("bridge bits must be a multiple of 32 in [64,1024]");
    return std::size_t(bits/8+12)*(complex?2:1);
}
namespace detail {
template<int Bits=64,class F> inline void bridge_width(int bits,F&& f){
    if(bits==Bits){f(std::integral_constant<int,Bits>{});return;}
    if constexpr(Bits<1024)bridge_width<Bits+32>(bits,f);
}
}
namespace detail {
template<class T> inline void bridge_store(void* out,const T& value){std::memcpy(out,&value,sizeof(T));}
template<class T> inline T bridge_load(const void* input){T value;std::memcpy(&value,input,sizeof(T));return value;}
}
inline void from_mpfr(int bits,mpfr_srcptr input,void* out){bridge_element_bytes(bits);if(!input||!out)throw std::invalid_argument("null bridge operand");
    detail::bridge_width(bits,[&](auto b){detail::bridge_store(out,from_mpfr<b.value>(input));});}
inline void to_mpfr(int bits,mpfr_ptr out,const void* input){bridge_element_bytes(bits);if(!input||!out)throw std::invalid_argument("null bridge operand");
    detail::bridge_width(bits,[&](auto b){to_mpfr<b.value>(out,detail::bridge_load<Float<b.value>>(input));});}
inline void from_mpfr_array(int bits,const mpfr_t* input,void* out,std::size_t count,unsigned threads=0){auto bytes=bridge_element_bytes(bits);if(count&&(!input||!out))throw std::invalid_argument("null bridge array");if(count>std::size_t(-1)/bytes)throw std::invalid_argument("bridge array size overflow");
    detail::bridge_width(bits,[&](auto b){detail::parallel_chunks(count,threads,(std::size_t(1)<<22)/b.value,[&](std::size_t begin,std::size_t end,detail::Range range){for(auto i=begin;i<end;++i)detail::bridge_store(static_cast<char*>(out)+i*bytes,detail::from_mpfr_fast<b.value>(input[i],range));});});}
inline void to_mpfr_array(int bits,mpfr_t* out,const void* input,std::size_t count,unsigned threads=0){auto bytes=bridge_element_bytes(bits);if(count&&(!input||!out))throw std::invalid_argument("null bridge array");if(count>std::size_t(-1)/bytes)throw std::invalid_argument("bridge array size overflow");
    detail::bridge_width(bits,[&](auto b){detail::parallel_chunks(count,threads,(std::size_t(1)<<22)/b.value,[&](std::size_t begin,std::size_t end,detail::Range range){for(auto i=begin;i<end;++i)detail::to_mpfr_fast<b.value>(out[i],detail::bridge_load<Float<b.value>>(static_cast<const char*>(input)+i*bytes),range);});});}
#ifdef LIMBFORGE_HAS_MPC
inline void from_mpc(int bits,mpc_srcptr input,void* out){bridge_element_bytes(bits,true);if(!input||!out)throw std::invalid_argument("null bridge operand");
    detail::bridge_width(bits,[&](auto b){detail::bridge_store(out,from_mpc<b.value>(input));});}
inline void to_mpc(int bits,mpc_ptr out,const void* input){bridge_element_bytes(bits,true);if(!input||!out)throw std::invalid_argument("null bridge operand");
    detail::bridge_width(bits,[&](auto b){to_mpc<b.value>(out,detail::bridge_load<Complex<b.value/32>>(input));});}
inline void from_mpc_array(int bits,const mpc_t* input,void* out,std::size_t count,unsigned threads=0){auto bytes=bridge_element_bytes(bits,true);if(count&&(!input||!out))throw std::invalid_argument("null bridge array");if(count>std::size_t(-1)/bytes)throw std::invalid_argument("bridge array size overflow");
    detail::bridge_width(bits,[&](auto b){detail::parallel_chunks(count,threads,(std::size_t(1)<<21)/b.value,[&](std::size_t begin,std::size_t end,detail::Range range){for(auto i=begin;i<end;++i){Complex<b.value/32> value{detail::from_mpfr_fast<b.value>(mpc_realref(input[i]),range),detail::from_mpfr_fast<b.value>(mpc_imagref(input[i]),range)};detail::bridge_store(static_cast<char*>(out)+i*bytes,value);}});});}
inline void to_mpc_array(int bits,mpc_t* out,const void* input,std::size_t count,unsigned threads=0){auto bytes=bridge_element_bytes(bits,true);if(count&&(!input||!out))throw std::invalid_argument("null bridge array");if(count>std::size_t(-1)/bytes)throw std::invalid_argument("bridge array size overflow");
    detail::bridge_width(bits,[&](auto b){detail::parallel_chunks(count,threads,(std::size_t(1)<<21)/b.value,[&](std::size_t begin,std::size_t end,detail::Range range){for(auto i=begin;i<end;++i)detail::to_mpc_fast<b.value>(out[i],detail::bridge_load<Complex<b.value/32>>(static_cast<const char*>(input)+i*bytes),range);});});}
#endif
template<int Bits> inline Float<Bits> from_decimal(const std::string& text) {
    mpfr_t x;mpfr_init2(x,Bits);
    if(mpfr_set_str(x,text.c_str(),10,MPFR_RNDN)){mpfr_clear(x);throw std::invalid_argument("invalid decimal number");}
    auto r=from_mpfr<Bits>(x);mpfr_clear(x);return r;
}
}
