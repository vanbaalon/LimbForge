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
// Metal keeps private arrays of up to ~32 words in registers, where each runtime index costs a
// select chain over the array (docs/gpu-codegen.md). Runtime-indexed scratch of 13-32 words is
// padded past that size on the GPU (measured 1.3-8x at 192-480 bits); smaller arrays showed no
// reliable gain. The extra words stay zero, so results are unchanged.
constexpr int scratch(int words){
#ifdef __METAL_VERSION__
    return words>=13&&words<33?33:words;
#else
    return words;
#endif
}
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
    // Padding the addition workspace helps only from 512 bits (interleaved A/B, round 14).
    word x[N>=16?scratch(W):W]={};insert(x,a,a_shift);dword carry=0;
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
// Padded=false keeps the register-resident layout: complex kernels measured no gain from padding.
template<int N,bool Padded=true> inline Number<N> mul(LIMBFORGE_THREAD const Number<N>& a,LIMBFORGE_THREAD const Number<N>& b) {
    word status=a.status|b.status;if(status||!a.sign||!b.sign)return zero<N>(status);
    word product[Padded?scratch(2*N+1):2*N+1]={};
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
    word product[scratch(2*N+1)]={};
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
template<int N,bool Padded=true> inline Number<N> div(LIMBFORGE_THREAD const Number<N>& a,LIMBFORGE_THREAD const Number<N>& b) {
    word status=a.status|b.status;if(status)return zero<N>(status);
    if(!b.sign)return zero<N>(division_by_zero);if(!a.sign)return zero<N>();
    // Normalized base-2^32 Knuth division. O(N^2), no bit-at-a-time quotient.
    int cmp=0;for(int i=N-1;i>=0;--i)if(a.limb[i]!=b.limb[i]){cmp=a.limb[i]>b.limb[i]?1:-1;break;}
    int shift=32*N-1+(cmp<0);
    word u[Padded?scratch(2*N+1):2*N+1]={};insert(u,a,shift);
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
template<int N> inline Number<N> sqrt(LIMBFORGE_THREAD const Number<N>& a){
    if(a.status||!a.sign)return zero<N>(a.status);
    if(a.sign<0)return zero<N>(invalid);
    int e=a.exponent/2;if(a.exponent<0&&a.exponent%2)--e;
    int odd=a.exponent-2*e,shift=32*N-1+odd;
    Number<N> root=zero<N>();root.sign=1;root.exponent=e;
    word remainder[N+1]={},trial[N+1]={};
    for(int bit=32*N-1;bit>=0;--bit){
        int position=2*bit;word pair=(shifted_limb(a,shift,position/32)>>(position%32))&3;
        word carry=pair;for(int i=0;i<N+1;++i){word old=remainder[i];remainder[i]=(old<<2)|carry;carry=old>>30;}
        carry=0;for(int i=0;i<N;++i){word old=root.limb[i];root.limb[i]=(old<<1)|carry;carry=old>>31;}
        carry=1;for(int i=0;i<N;++i){word old=root.limb[i];trial[i]=(old<<1)|carry;carry=old>>31;}trial[N]=carry;
        int cmp=0;for(int i=N;i>=0;--i)if(remainder[i]!=trial[i]){cmp=remainder[i]>trial[i]?1:-1;break;}
        if(cmp>=0){dword borrow=0;for(int i=0;i<N+1;++i){dword v=dword(trial[i])+borrow;word old=remainder[i];remainder[i]=word(dword(old)-v);borrow=dword(old)<v;}root.limb[0]|=1;}
    }
    // Integer radicand: sqrt(Z) > q+1/2 exactly when Z-q*q > q.
    // Equality to a half-integer is impossible, so no tie-breaking is needed.
    if(limb_compare<N>(remainder,root)>0)increment(root);
    return checked(root);
}
// ---- Fused operations: one rounding of an exact sum of products and addends ----
// value = sign * integer(w) * 2^scale; sign 0 is zero.
template<int W> struct Exact { word w[W]; exponent_type scale; int sign; };
template<int W> inline exponent_type exact_msb(LIMBFORGE_THREAD const Exact<W>& x){return x.scale+highest(x.w);}
// Words for an exact sum of A- and B-word terms, including the sticky collapse below.
constexpr int exact_words(int n,int a,int b){
    int x=(a>n+2?a:n+2)+b,y=(b>n+2?b:n+2)+a;return (x>y?x:y)+2;
}
template<int W,int A> inline void place(LIMBFORGE_THREAD word (&out)[W],LIMBFORGE_THREAD const word (&in)[A],exponent_type shift){
    int w=int(shift/32),s=int(shift%32);
    for(int i=0;i<A;++i){if(i+w<W)out[i+w]|=in[i]<<s;if(s&&i+w+1<W)out[i+w+1]|=in[i]>>(32-s);}
}
template<int W,int A> inline Exact<W> exact_copy(LIMBFORGE_THREAD const Exact<A>& x){
    Exact<W> r;for(int i=0;i<W;++i)r.w[i]=i<A?x.w[i]:0;r.scale=x.scale;r.sign=x.sign;return r;
}
// x has the larger msb. x's window is extended to at least 32N+34 bits below its msb. If y lies
// more than one bit below that window, |y| < 2^(low-1) changes no rounding decision of a 32N-bit
// result: only its sign matters, kept as one unit below the window. Otherwise the sum is exact.
template<int N,int W,int A,int B> inline Exact<W> exact_add_ordered(LIMBFORGE_THREAD const Exact<A>& x,LIMBFORGE_THREAD const Exact<B>& y){
    Exact<W> r;for(int i=0;i<W;++i)r.w[i]=0;
    exponent_type mx=exact_msb(x),my=exact_msb(y),low=mx-(32*N+33);if(x.scale<low)low=x.scale;
    if(my<low-1){
        r.scale=low-32;r.sign=x.sign;place(r.w,x.w,x.scale-r.scale);
        if(x.sign==y.sign)r.w[0]=1;else for(int i=0;i<W;++i){word old=r.w[i];r.w[i]=old-1;if(old)break;}
        return r;
    }
    r.scale=x.scale<y.scale?x.scale:y.scale;word t[W];for(int i=0;i<W;++i)t[i]=0;
    place(r.w,x.w,x.scale-r.scale);place(t,y.w,y.scale-r.scale);
    if(x.sign==y.sign){dword c=0;for(int i=0;i<W;++i){dword v=dword(r.w[i])+t[i]+c;r.w[i]=word(v);c=v>>32;}r.sign=x.sign;return r;}
    int cmp=0;for(int i=W-1;i>=0;--i)if(r.w[i]!=t[i]){cmp=r.w[i]>t[i]?1:-1;break;}
    if(!cmp){r.sign=0;r.scale=0;for(int i=0;i<W;++i)r.w[i]=0;return r;}
    dword borrow=0;
    for(int i=0;i<W;++i){word p=cmp>0?r.w[i]:t[i],q=cmp>0?t[i]:r.w[i];dword v=dword(q)+borrow;r.w[i]=word(dword(p)-v);borrow=dword(p)<v;}
    r.sign=cmp>0?x.sign:y.sign;return r;
}
template<int N,int W,int A,int B> inline Exact<W> exact_add(LIMBFORGE_THREAD const Exact<A>& x,LIMBFORGE_THREAD const Exact<B>& y){
    if(!y.sign)return exact_copy<W>(x);if(!x.sign)return exact_copy<W>(y);
    if(exact_msb(x)>=exact_msb(y))return exact_add_ordered<N,W,A,B>(x,y);
    return exact_add_ordered<N,W,B,A>(y,x);
}
template<int N> constexpr int product_words(){return scratch(2*N+1);}
template<int N> inline Exact<product_words<N>()> exact_product(LIMBFORGE_THREAD const Number<N>& a,LIMBFORGE_THREAD const Number<N>& b){
    Exact<product_words<N>()> r;for(int i=0;i<product_words<N>();++i)r.w[i]=0;
    r.scale=exponent_type(a.exponent)+b.exponent-2*(32*N-1);r.sign=a.sign*b.sign;if(!r.sign){r.scale=0;return r;}
    for(int i=0;i<N;++i){dword carry=0;
        for(int j=0;j<N;++j){dword v=dword(a.limb[i])*b.limb[j]+r.w[i+j]+carry;r.w[i+j]=word(v);carry=v>>32;}
        r.w[i+N]=word(carry);}
    return r;
}
template<int N> inline Exact<product_words<N>()> exact_number(LIMBFORGE_THREAD const Number<N>& c){
    Exact<product_words<N>()> r;for(int i=0;i<product_words<N>();++i)r.w[i]=i<N?c.limb[i]:0;
    r.scale=exponent_type(c.exponent)-(32*N-1);r.sign=c.sign;return r;
}
template<int N,int W> inline Number<N> round_exact(LIMBFORGE_THREAD const Exact<W>& x){
    if(!x.sign)return zero<N>();return pack<N>(x.w,x.scale,x.sign,ok);
}
// RN(a*b+c) with a single rounding (MPFR mpfr_fma contract).
template<int N> inline Number<N> fma(LIMBFORGE_THREAD const Number<N>& a,LIMBFORGE_THREAD const Number<N>& b,LIMBFORGE_THREAD const Number<N>& c){
    word status=a.status|b.status|c.status;if(status)return zero<N>(status);
    constexpr int P=product_words<N>();
    auto p=exact_product(a,b);auto q=exact_number(c);
    return round_exact<N>(exact_add<N,exact_words(N,P,P)>(p,q));
}
template<int N> inline Number<N> fms(LIMBFORGE_THREAD const Number<N>& a,LIMBFORGE_THREAD const Number<N>& b,LIMBFORGE_THREAD const Number<N>& c){
    Number<N> d=negate(c);return fma(a,b,d);
}
// RN(a*b + c*d + e) with a single rounding. With terms ordered by msb, the two smaller terms are
// either both far below the largest (only the sign of their exact sum matters) or the largest two
// overlap and are summed exactly before the third is added; each pairwise step is then exact or a
// valid sticky collapse, so no cancellation can expose a collapsed term.
template<int N> inline Number<N> dot2_add(LIMBFORGE_THREAD const Number<N>& a,LIMBFORGE_THREAD const Number<N>& b,LIMBFORGE_THREAD const Number<N>& c,
                                          LIMBFORGE_THREAD const Number<N>& d,LIMBFORGE_THREAD const Number<N>& e){
    word status=a.status|b.status|c.status|d.status|e.status;if(status)return zero<N>(status);
    constexpr int P=product_words<N>(),W1=exact_words(N,P,P),W2=exact_words(N,P,W1);
    Exact<P> t[3]={exact_product(a,b),exact_product(c,d),exact_number(e)};
    exponent_type key[3];for(int i=0;i<3;++i)key[i]=t[i].sign?exact_msb(t[i]):-(exponent_type(1)<<62);
    // Three named terms ordered by msb, largest first.
    int i1=key[0]>=key[1]?(key[0]>=key[2]?0:2):(key[1]>=key[2]?1:2);
    int i3=key[0]<key[1]?(key[0]<key[2]?0:2):(key[1]<key[2]?1:2);if(i3==i1)i3=(i1+1)%3;
    int i2=3-i1-i3;
    Exact<P> t1=t[i1],t2=t[i2],t3=t[i3];
    if(!t1.sign)return zero<N>();
    exponent_type low=exact_msb(t1)-(32*N+33);if(t1.scale<low)low=t1.scale;
    if(!t2.sign||exact_msb(t2)<low-1){auto s=exact_add<N,W1>(t2,t3);return round_exact<N>(exact_add<N,W2>(t1,s));}
    auto r=exact_add<N,W1>(t1,t2);return round_exact<N>(exact_add<N,W2>(r,t3));
}
// Complex a*b+c and a*b-c: each component is one rounding of its exact value.
template<int N> inline Complex<N> cfma(Complex<N> a,Complex<N> b,Complex<N> c){
    Number<N> m=negate(a.im);return {dot2_add(a.re,b.re,m,b.im,c.re),dot2_add(a.re,b.im,a.im,b.re,c.im)};
}
template<int N> inline Complex<N> cfms(Complex<N> a,Complex<N> b,Complex<N> c){
    Complex<N> d={negate(c.re),negate(c.im)};return cfma(a,b,d);
}
template<int N> inline Complex<N> cadd(Complex<N> a,Complex<N> b){return {add(a.re,b.re),add(a.im,b.im)};}
template<int N> inline Complex<N> cmul(Complex<N> a,Complex<N> b){return {sub(mul<N,false>(a.re,b.re),mul<N,false>(a.im,b.im)),add(mul<N,false>(a.re,b.im),mul<N,false>(a.im,b.re))};}
template<int N> inline Complex<N> cdiv(Complex<N> a,Complex<N> b){
    Number<N> d=add(mul<N,false>(b.re,b.re),mul<N,false>(b.im,b.im));
    return {div<N,false>(add(mul<N,false>(a.re,b.re),mul<N,false>(a.im,b.im)),d),div<N,false>(sub(mul<N,false>(a.im,b.re),mul<N,false>(a.re,b.im)),d)};
}
} // namespace limbforge

#undef LIMBFORGE_THREAD
