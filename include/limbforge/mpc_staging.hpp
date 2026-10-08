#pragma once
#include "mpfr_bridge.hpp"
#include "batched_linalg.hpp"
#ifdef LIMBFORGE_HAS_MPC
namespace limbforge {
// Describe MPC numbers whose significands live inside allocation. No LimbForge limb arrays are copied
// on the host. The GPU import reads the original allocation and writes ordinary resident Complex values.
// Source precisions must equal bits. Use the ordinary bridge if rounding or widening is required.
inline std::vector<InlineComplexRecord> describe_inline_mpc(int bits,const void* allocation,std::size_t bytes,const mpc_srcptr* values,std::size_t count){
    bridge_element_bytes(bits,true);if(count&&(!allocation||!values))throw std::invalid_argument("null inline MPC allocation");
    if(!detail::direct_limbs)throw std::runtime_error("inline MPC staging requires little-endian 64-bit GMP limbs");
    auto begin=reinterpret_cast<std::uintptr_t>(allocation);if(bytes>std::uintptr_t(-1)-begin)throw std::invalid_argument("inline MPC allocation overflow");
    auto part=[&](mpfr_srcptr x,std::uint64_t& offset,std::int32_t& exponent,std::int32_t& sign,std::uint32_t& status){
        if(mpfr_get_prec(x)!=bits)throw std::invalid_argument("inline MPC precision must equal target bits");
        offset=0;exponent=0;sign=0;status=0;
        if(!mpfr_number_p(x)){status=invalid;return;}if(mpfr_zero_p(x))return;
        auto e=mpfr_get_exp(x)-1;if(e>1000000000||e< -1000000000){status=exponent_overflow;return;}
        auto p=reinterpret_cast<std::uintptr_t>(mpfr_custom_get_significand(x));std::size_t n=8*((std::size_t(bits)+63)/64);
        if(p<begin||p>begin+bytes||n>begin+bytes-p)throw std::invalid_argument("inline MPC significand outside allocation");
        offset=p-begin;exponent=std::int32_t(e);sign=mpfr_signbit(x)?-1:1;
    };
    std::vector<InlineComplexRecord> records(count);for(std::size_t i=0;i<count;++i){if(!values[i])throw std::invalid_argument("null inline MPC record");auto& r=records[i];
        part(mpc_realref(values[i]),r.real_offset,r.real_exponent,r.real_sign,r.real_status);
        part(mpc_imagref(values[i]),r.imag_offset,r.imag_exponent,r.imag_sign,r.imag_status);}
    return records;
}
}
#endif
