#pragma once
// Shared, allocation-free CPU / Metal implementation. Finite binary numbers;
// each real operation rounds to nearest, ties to even.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <cstdint>
#endif
namespace limbforge {
#ifdef __METAL_VERSION__
#define LIMBFORGE_THREAD thread
using word = uint;
using dword = ulong;
using exponent_type = long;
#else
#define LIMBFORGE_THREAD
using word = std::uint32_t;
using dword = std::uint64_t;
using exponent_type = std::int64_t;
#endif
enum Status : word { ok=0, division_by_zero=1, exponent_overflow=2, invalid=4 };
template<int N> struct Number {
    word limb[N]; // little endian; top bit set for every nonzero value
    int exponent; // value = sign * mantissa * 2^(exponent - (32*N-1))
    int sign;     // -1, 0, +1; canonical unsigned zero
    word status;
};
template<int Bits> struct FloatType {
    static_assert(Bits>=64&&Bits<=1024&&Bits%32==0,"precision must be a multiple of 32 in [64,1024]");
    using type=Number<Bits/32>;
};
template<int Bits> using Float = typename FloatType<Bits>::type;
template<int N> struct Complex { Number<N> re, im; };
template<int N> inline Number<N> zero(word status=ok) {
    Number<N> r; for(int i=0;i<N;++i)r.limb[i]=0;
    r.exponent=0;r.sign=0;r.status=status;return r;
}
template<int N> inline Number<N> negate(Number<N> a){a.sign=-a.sign;return a;}
template<int W> inline int highest(LIMBFORGE_THREAD const word (&a)[W]) {
    for(int i=W-1;i>=0;--i)if(a[i]) {
#ifdef __METAL_VERSION__
        return 32*i+31-int(metal::clz(a[i]));
#else
        return 32*i+31-__builtin_clz(a[i]);
#endif
    }return -1;
}
template<int W> inline word extract(LIMBFORGE_THREAD const word (&a)[W],int bit) {
    if(bit<0)return 0;
    int i=bit/32,s=bit%32;
    word x=i<W?a[i]>>s:0;
    if(s&&i+1<W)x|=a[i+1]<<(32-s);
    return x;
}
template<int W> inline bool any_below(LIMBFORGE_THREAD const word (&a)[W],int bit) {
    if(bit<=0)return false;
    int whole=bit/32;
    for(int i=0;i<W&&i<whole;++i)if(a[i])return true;
    return whole<W&&(bit%32)&&(a[whole]&((word(1)<<(bit%32))-1));
}
template<int N> inline void increment(LIMBFORGE_THREAD Number<N>& r) {
    for(int i=0;i<N;++i)if(++r.limb[i])return;
    r.limb[N-1]=word(1)<<31;++r.exponent;
}
template<int N> inline Number<N> checked(Number<N> r) {
    if(r.exponent>1000000000||r.exponent< -1000000000)return zero<N>(r.status|exponent_overflow);
    return r;
}
template<int N,int W> inline Number<N> pack(LIMBFORGE_THREAD const word (&a)[W],exponent_type scale,int sign,word status) {
    int h=highest(a);if(h<0)return zero<N>(status);
    exponent_type e=scale+h;
    // Rounding can carry a result at emin-1 back into the supported range.
    if(e>1000000000||e< -1000000001)return zero<N>(status|exponent_overflow);
    Number<N> r=zero<N>(status);r.sign=sign;r.exponent=int(e);
    int shift=h-(32*N-1);
    if(shift>=0) {
        for(int i=0;i<N;++i)r.limb[i]=extract(a,shift+32*i);
        if(shift>0&&((extract(a,shift-1)&1)!=0)&&
           (any_below(a,shift-1)||(r.limb[0]&1)))increment(r);
    } else {
        int left=-shift,w=left/32,s=left%32;
        for(int i=0;i<W;++i)if(i+w<N) {
            r.limb[i+w]|=a[i]<<s;
            if(s&&i+w+1<N)r.limb[i+w+1]|=a[i]>>(32-s);
        }
    }return checked(r);
}
template<int N> inline int magnitude_compare(LIMBFORGE_THREAD const Number<N>& a,LIMBFORGE_THREAD const Number<N>& b) {
    if(!a.sign)return b.sign?-1:0;if(!b.sign)return 1;
    if(a.exponent!=b.exponent)return a.exponent>b.exponent?1:-1;
    for(int i=N-1;i>=0;--i)if(a.limb[i]!=b.limb[i])return a.limb[i]>b.limb[i]?1:-1;
    return 0;
}
template<int N,int W> inline void insert(LIMBFORGE_THREAD word (&out)[W],LIMBFORGE_THREAD const Number<N>& a,int shift) {
    int w=shift/32,s=shift%32;
    for(int i=0;i<N;++i){out[i+w]|=a.limb[i]<<s;if(s)out[i+w+1]|=a.limb[i]>>(32-s);}
}
template<int N> inline word shifted_limb(LIMBFORGE_THREAD const Number<N>& a,int shift,int position){
    int source=position-shift/32,s=shift%32;
    word x=source>=0&&source<N?a.limb[source]<<s:0;
    if(s&&source>0&&source<=N)x|=a.limb[source-1]>>(32-s);
    return x;
}
template<int N,int W> inline Number<N> aligned_add(LIMBFORGE_THREAD const Number<N>& a,LIMBFORGE_THREAD const Number<N>& b,int a_shift,int b_shift,exponent_type scale){
    word x[W]={};insert(x,a,a_shift);dword carry=0;
    if(a.sign==b.sign)for(int i=0;i<W;++i){dword v=dword(x[i])+shifted_limb(b,b_shift,i)+carry;x[i]=word(v);carry=v>>32;}
    else for(int i=0;i<W;++i){dword v=dword(shifted_limb(b,b_shift,i))+carry;word old=x[i];x[i]=word(dword(old)-v);carry=dword(old)<v;}
    return pack<N>(x,scale,a.sign,ok);
}
template<int N> inline Number<N> add(Number<N> a,Number<N> b) {
    word status=a.status|b.status;if(status)return zero<N>(status);
    if(!a.sign)return b;if(!b.sign)return a;
    if(magnitude_compare(a,b)<0){Number<N> t=a;a=b;b=t;}
    exponent_type gap=exponent_type(a.exponent)-b.exponent;
    // Below one quarter ulp; a is exactly representable, so it rounds to a.
    if(gap>32*N+2)return a;
    // Close exponents need only N+2 exact limbs, even under deep cancellation.
    // Far exponents retain the complete exact alignment; no sticky-bit shortcut.
    if(gap<=32)return aligned_add<N,N+2>(a,b,int(gap),0,exponent_type(b.exponent)-(32*N-1));
    return aligned_add<N,2*N+2>(a,b,32*N+2,32*N+2-int(gap),exponent_type(a.exponent)-(32*N-1)-(32*N+2));
}
template<int N> inline Number<N> sub(Number<N> a,Number<N> b){return add(a,negate(b));}
template<int N> inline Number<N> mul(LIMBFORGE_THREAD const Number<N>& a,LIMBFORGE_THREAD const Number<N>& b) {
    word status=a.status|b.status;if(status||!a.sign||!b.sign)return zero<N>(status);
    word product[2*N+1]={};
    for(int i=0;i<N;++i){dword carry=0;
        for(int j=0;j<N;++j){dword v=dword(a.limb[i])*b.limb[j]+product[i+j]+carry;product[i+j]=word(v);carry=v>>32;}
        product[i+N]=word(carry);
    }
    return pack<N>(product,exponent_type(a.exponent)+b.exponent-2*(32*N-1),a.sign*b.sign,status);
}
template<int N> inline Number<N> square(LIMBFORGE_THREAD const Number<N>& a){
#ifdef __METAL_VERSION__
    // The symmetric implementation is selected on Metal at 384 bits.
    // Other widths retain the validated schoolbook product.
    if(N!=12)return mul(a,a);
#endif
    if(a.status||!a.sign)return zero<N>(a.status);
    word product[2*N+1]={};
    // Accumulate each off-diagonal product once, then double the exact integer.
    for(int i=0;i<N-1;++i){dword carry=0;
        for(int j=i+1;j<N;++j){dword v=dword(a.limb[i])*a.limb[j]+product[i+j]+carry;product[i+j]=word(v);carry=v>>32;}
        product[i+N]=word(carry);
    }
    word carry=0;for(int i=0;i<2*N+1;++i){word old=product[i];product[i]=(old<<1)|carry;carry=old>>31;}
    // Add the diagonal with bounded sums; never double a full 64-bit product.
    for(int i=0;i<N;++i){dword p=dword(a.limb[i])*a.limb[i];
        dword v=dword(product[2*i])+word(p);product[2*i]=word(v);
        v=dword(product[2*i+1])+(p>>32)+(v>>32);product[2*i+1]=word(v);
        dword c=v>>32;for(int j=2*i+2;c&&j<2*N+1;++j){v=dword(product[j])+c;product[j]=word(v);c=v>>32;}
    }
    return pack<N>(product,2*exponent_type(a.exponent)-2*(32*N-1),1,ok);
}
template<int N> inline int limb_compare(LIMBFORGE_THREAD const word (&a)[N+1],LIMBFORGE_THREAD const Number<N>& b) {
    if(a[N])return 1;
    for(int i=N-1;i>=0;--i)if(a[i]!=b.limb[i])return a[i]>b.limb[i]?1:-1;
    return 0;
}
// Exact two-limb / normalized one-limb quotient using a reused reciprocal.
// Preconditions: d >= 2^31, high < d; inv = floor((2^64-1)/d) - 2^32.
struct WordDivision { word quotient,remainder; };
inline WordDivision divide_word(word high,word low,word d,word inv){
    dword product=dword(high)*inv;
    dword sum=dword(word(product))+low;
    word q=word((product>>32)+high+1+(sum>>32));
    word r=low-q*d;
    if(r>word(sum)){--q;r+=d;}
    if(r>=d){++q;r-=d;}
    return {q,r};
}
template<int N> inline Number<N> div(LIMBFORGE_THREAD const Number<N>& a,LIMBFORGE_THREAD const Number<N>& b) {
    word status=a.status|b.status;if(status)return zero<N>(status);
    if(!b.sign)return zero<N>(division_by_zero);if(!a.sign)return zero<N>();
    // Normalized base-2^32 Knuth division. O(N^2), no bit-at-a-time quotient.
    int cmp=0;for(int i=N-1;i>=0;--i)if(a.limb[i]!=b.limb[i]){cmp=a.limb[i]>b.limb[i]?1:-1;break;}
    int shift=32*N-1+(cmp<0);
    word u[2*N+1]={};insert(u,a,shift);
    Number<N> r=zero<N>();r.sign=a.sign*b.sign;
    exponent_type e=exponent_type(a.exponent)-b.exponent-(cmp<0);
    if(e>1000000000||e< -1000000001)return zero<N>(exponent_overflow);
    r.exponent=int(e);
    constexpr dword B=dword(1)<<32;
    word d=b.limb[N-1],inv=word(~dword(0)/d-B);
    for(int j=N;j>=0;--j) {
        dword top=(dword(u[j+N])<<32)|u[j+N-1];
        dword q,rem;
        if(u[j+N]>=d){q=B-1;rem=top-q*d;}
        else {auto estimate=divide_word(u[j+N],u[j+N-1],d,inv);q=estimate.quotient;rem=estimate.remainder;}
        if(N>1)while(rem<B&&q*b.limb[N-2]>(rem<<32)+u[j+N-2]){--q;rem+=b.limb[N-1];}
        dword borrow=0;
        for(int i=0;i<N;++i){dword p=q*b.limb[i]+borrow;word low=word(p),old=u[j+i];u[j+i]=old-low;borrow=(p>>32)+(old<low);}
        bool under=dword(u[j+N])<borrow;u[j+N]=word(dword(u[j+N])-borrow);
        if(under){--q;dword carry=0;for(int i=0;i<N;++i){dword v=dword(u[j+i])+b.limb[i]+carry;u[j+i]=word(v);carry=v>>32;}u[j+N]+=word(carry);}
        if(j<N)r.limb[j]=word(q);
    }
    word twice[N+1]={};word carry=0;
    for(int i=0;i<N;++i){twice[i]=(u[i]<<1)|carry;carry=u[i]>>31;}twice[N]=carry;
    int halfway=limb_compare<N>(twice,b);
    if(halfway>0||(halfway==0&&(r.limb[0]&1)))increment(r);
    return checked(r);
}
template<int N> inline Complex<N> cadd(Complex<N> a,Complex<N> b){return {add(a.re,b.re),add(a.im,b.im)};}
template<int N> inline Complex<N> cmul(Complex<N> a,Complex<N> b){return {sub(mul(a.re,b.re),mul(a.im,b.im)),add(mul(a.re,b.im),mul(a.im,b.re))};}
template<int N> inline Complex<N> cdiv(Complex<N> a,Complex<N> b){
    Number<N> d=add(mul(b.re,b.re),mul(b.im,b.im));
    return {div(add(mul(a.re,b.re),mul(a.im,b.im)),d),div(sub(mul(a.im,b.re),mul(a.re,b.im)),d)};
}
} // namespace limbforge

#undef LIMBFORGE_THREAD
