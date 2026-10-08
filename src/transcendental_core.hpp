#pragma once
// Elementary functions shared by the host (src/transcendental.mm) and the Metal kernels (src/transcendental.metal);
// contract in docs/numerics.md, "Transcendental functions". Every *_approx<W> evaluates at W words with the correctly
// rounded primitives of core.hpp and returns y with a rigorous bound |y*2^shift - f| <= err * ulp_W(y) * 2^shift.
// certify<N> rounds to N < W words only when every value within the bound has the same round-to-nearest-even result;
// otherwise the caller retries at more words (Ziv). Bounds are floats rounded generously upward (the factors >= 1.01
// absorb float rounding); p2() clamps tiny contributions up to 2^-100 ulp and huge ones to 2^100 (always rejected).
// Errors are tracked in units of u = 2^(-32W) relative ("relative units") or in ulps of a named value.
#ifndef __METAL_VERSION__
#include "limbforge/core.hpp"
#endif
namespace limbforge {
namespace transcendental {
#ifdef __METAL_VERSION__
#define LF_T thread
#define LF_D device
inline int lf_clz(word x){return int(metal::clz(x));}
inline float lf_float(word b){return as_type<float>(b);}
#else
#define LF_T
#define LF_D
inline int lf_clz(word x){return __builtin_clz(x);}
inline float lf_float(word b){float f;__builtin_memcpy(&f,&b,4);return f;}
#endif
enum : int {
    TRIG_EMAX=8192, // sin, cos and the imaginary part of complex exp: |x| < 2^TRIG_EMAX, else invalid
    TMAX=64         // entries of the restoring-step tables
};
// Per working width W (built on the host, src/transcendental.mm): loop counts and constants, each constant within
// 1 ulp_W (ln2x within 1 ulp at W+2 words). up[i] = log(1+2^-i), down[i] = log(1-2^-i), atn[i] = atan(2^-i).
template<int W> struct Tables {
    int s_exp,k_exp,s_trig,k_trig,steps,k_series,reserved0,reserved1;
    Number<W+2> ln2x;
    Number<W> ln2,pi,half_pi;
    Number<W> up[TMAX+1],down[TMAX+1],atn[TMAX+1];
};
template<int W> struct Approx { Number<W> y; exponent_type shift; float err; }; // err < 0: y*2^shift is exact
template<int W> inline Approx<W> exact(Number<W> y,exponent_type shift=0){Approx<W> a;a.y=y;a.shift=shift;a.err=-1.f;return a;}
template<int W> inline Approx<W> approx(Number<W> y,float err,exponent_type shift=0){Approx<W> a;a.y=y;a.shift=shift;a.err=err;return a;}
// Upper bound of 2^e as a float, clamped to [2^-100, 2^100].
inline float p2(exponent_type e){if(e< -100)e=-100;if(e>100)e=100;return lf_float(word(e+127)<<23);}
template<int W> inline Number<W> unit(int e=0,int sign=1){Number<W> r=zero<W>();r.limb[W-1]=0x80000000u;r.exponent=e;r.sign=sign;return r;}
template<int W> inline Number<W> scaled(Number<W> a,int k){if(a.sign)a.exponent+=k;return a;}
template<int W> inline Number<W> absolute(Number<W> a){if(a.sign<0)a.sign=1;return a;}
template<int W,int N> inline Number<W> widen(LF_T const Number<N>& a){
    Number<W> r=zero<W>(a.status);if(a.status)return r;r.sign=a.sign;r.exponent=a.exponent;
    for(int i=0;i<N;++i)r.limb[W-N+i]=a.limb[i];return r;
}
template<int W,int M> inline Number<W> narrow(LF_T const Number<M>& a){
    if(a.status||!a.sign)return zero<W>(a.status);return pack<W>(a.limb,exponent_type(a.exponent)-(32*M-1),a.sign,ok);
}
template<int W> inline Number<W> integer(exponent_type v){ // |v| < 2^63, exact
    Number<W> r=zero<W>();if(!v)return r;dword m=dword(v<0?-v:v);int h=63;while(!((m>>h)&1))--h;
    m<<=(63-h);r.limb[W-1]=word(m>>32);r.limb[W-2]=word(m);r.exponent=h;r.sign=v<0?-1:1;return r;
}
// RN(a/d), 1 <= d < 2^32: base-2^32 long division of the significand times 2^64; the remainder is a sticky bit far below
// the rounding position (the quotient has >= 32W+32 bits).
template<int W> inline Number<W> div_word(Number<W> a,word d){
    if(a.status||!a.sign||d==1)return a;
    int s=lf_clz(d);word dn=d<<s,inv=word(~dword(0)/dn-(dword(1)<<32));
    word u[scratch(W+3)]={},q[scratch(W+3)]={};
    for(int i=0;i<W;++i){u[i+2]|=a.limb[i]<<s;if(s)u[i+3]|=a.limb[i]>>(32-s);}
    word rem=0;for(int i=W+2;i>=0;--i){WordDivision r=divide_word(rem,u[i],dn,inv);q[i]=r.quotient;rem=r.remainder;}
    if(rem)q[0]|=1;
    return pack<W>(q,exponent_type(a.exponent)-(32*W-1)-64,a.sign,ok);
}
// Rounds y*2^shift to N < W words when |y*2^shift - v| <= err * ulp_W(y)*2^shift decides RN(v): the low D = W-N words
// L of y must differ from the midpoint H = 2^(32D-1) by more than err (+1 ulp margin), and err < 2^(32D-3) keeps the
// interval clear of the halved spacing below a power of two. Exact y (err < 0) is rounded to nearest even.
template<int N,int W> inline bool certify(LF_T const Number<W>& y,exponent_type shift,float err,LF_T Number<N>& out){
    if(y.status){out=zero<N>(y.status);return true;}
    if(!y.sign){out=zero<N>();return err<0;}
    constexpr int D=W-N;
    word top=y.limb[D-1]^0x80000000u;bool neg=(top>>31)!=0,high=false;dword low=0; // v = L-H in two's complement
    for(int i=0;i<D;++i){word w=i==D-1?top:y.limb[i];if(neg)w=~w;if(i>=2){if(w)high=true;}else low|=dword(w)<<(32*i);}
    bool up;
    if(err>=0){
        float bound=err*1.01f+1.f,cap=D>=2?p2(62):p2(32*D-3);if(!(bound<cap))return false;
        dword e=dword(bound)+1;if(!(high||(neg?low>=e:low>e)))return false;
        up=!neg;
    }else up=!neg&&(high||low||(y.limb[D]&1));
    Number<N> r=zero<N>();r.sign=y.sign;exponent_type ex=exponent_type(y.exponent)+shift;
    for(int i=0;i<N;++i)r.limb[i]=y.limb[D+i];
    if(up){int i=0;for(;i<N;++i)if(++r.limb[i])break;if(i==N){r.limb[N-1]=0x80000000u;++ex;}}
    if(ex>1000000000||ex< -1000000000){out=zero<N>(exponent_overflow);return true;}
    r.exponent=int(ex);out=r;return true;
}
// ---- exp and expm1 ----
inline Number<2> inv_ln2(){Number<2> r;r.limb[0]=0x5C17F0BBu;r.limb[1]=0xB8AA3B29u;r.exponent=0;r.sign=1;r.status=0;return r;}
// Nearest integer to q, |q| < 2^32.
inline exponent_type nearest(LF_T const Number<2>& q){
    if(!q.sign||q.exponent< -1)return 0;dword m=(dword(q.limb[1])<<32)|q.limb[0];
    exponent_type k=exponent_type(((m>>(62-q.exponent))+1)>>1);return q.sign<0?-k:k;
}
// E = expm1(r) for |r| <= 0.35 with relative error dr; returns E with relative error d (units u).
// r' = r/2^s (|r'| < 2^-S), Horner Taylor of degree K (truncation < u/16), then s doublings E <- E(E+2): each adds
// two roundings and scales the error by 1 + |E|/(2+E) (2+E >= 1.7), so the relative error stays small (no 2^s loss).
template<int W> inline Number<W> expm1_small(Number<W> r,float dr,LF_D const Tables<W>& t,LF_T float& d){
    if(!r.sign){d=dr;return r;}
    int s=r.exponent+t.s_exp+1;if(s<0)s=0;
    Number<W> x=scaled(r,-s),one=unit<W>(),two=unit<W>(1),h=one;
    for(int j=t.k_exp;j>=2;--j)h=add(one,div_word(mul(x,h),word(j)));
    Number<W> e=mul(x,h);d=1.01f*dr+2.4f;
    for(int i=0;i<s;++i){float g=p2(exponent_type(e.exponent)+1)*0.59f;Number<W> f=add(e,two);e=mul(e,f);d=d*(1.f+g)+2.01f;}
    return e;
}
// x = k ln2 + r with |r| <= ln2/2 + 2^-28: k from 64-bit truncations, r = RN(RN_{W+2}(x - k*ln2x)); |k| < 2^32 gives
// |k*(ln2x - ln2)| < 2^(-32W-32), so the relative error of r is <= 1.1 + 2^(-31-e_r) (units u).
template<int W> inline Number<W> reduce_ln2(LF_T const Number<W>& x,LF_D const Tables<W>& t,LF_T exponent_type& k,LF_T float& dr){
    Number<2> x2;x2.limb[0]=x.limb[W-2];x2.limb[1]=x.limb[W-1];x2.exponent=x.exponent;x2.sign=x.sign;x2.status=0;
    Number<2> c=inv_ln2();k=nearest(mul(x2,c));dr=0;if(!k)return x;
    Number<W+2> l=t.ln2x,xl=widen<W+2>(x),kl=integer<W+2>(-k);Number<W+2> rl=fma(kl,l,xl);
    if(!rl.sign){dr=p2(100);return x;} // impossible for the exact ln 2; forces a retry
    Number<W> r=narrow<W>(rl);dr=1.1f+p2(-31-exponent_type(r.exponent));return r;
}
// exp(x) = 2^k * p, p = 1 + expm1(r) in [0.70, 1.42]; dp = relative error of p (units u). x nonzero, |x| < 2^31.
template<int W> inline Number<W> exp_parts(LF_T const Number<W>& x,LF_D const Tables<W>& t,LF_T exponent_type& k,LF_T float& dp,LF_T Number<W>& e,LF_T float& de){
    float dr;Number<W> r=reduce_ln2(x,t,k,dr);e=expm1_small(r,dr,t,de);
    dp=de*p2(exponent_type(e.exponent)+1)*1.43f+1.01f; // |E|/(1+E) <= |E|/0.7
    return add(unit<W>(),e);
}
template<int W> inline float approx_value(LF_T const Number<W>& x){return float(x.sign)*float(x.limb[W-1])*p2(exponent_type(x.exponent)-31);} // |exponent| < 60
// exp(x), or expm1(x) with minus_one.
template<int W> inline Approx<W> exp_approx(Number<W> x,LF_D const Tables<W>& t,bool minus_one){
    if(x.status)return exact(zero<W>(x.status));
    if(!x.sign)return exact(minus_one?zero<W>():unit<W>());
    // expm1(x) for x < -(23W+3): e^x < 2^-(32W+2), the value is -1 + less than ulp_W(1)/8.
    if(minus_one&&x.sign<0&&(x.exponent>=31||(x.exponent>=4&&approx_value(x)< -float(23*W+3))))return approx(unit<W>(0,-1),0.25f);
    if(x.exponent>=31)return exact(zero<W>(exponent_overflow)); // e^x outside 2^(+-3e9)
    if(x.exponent< -(32*W+2))return minus_one?approx(x,1.f):approx(unit<W>(),1.f); // |e^x-1-x| < x^2
    exponent_type k;float dp,de;Number<W> e,p=exp_parts(x,t,k,dp,e,de);
    if(!minus_one)return approx(p,dp*1.02f,k);
    if(!k)return approx(e,de*1.02f);
    if(k>=32*W+4)return approx(p,dp*1.02f+0.2f,k); // the -1 is below 2^(-32W-4) relative
    // |k| <= 33W+5 here; e^x - 1 has no cancellation (|result| >= 0.29 when k != 0).
    Number<W> y=sub(scaled(p,int(k)),unit<W>());
    return approx(y,(dp*p2(exponent_type(p.exponent)+k-y.exponent)+0.5f)*1.02f);
}
// ---- log and log1p ----
// log(1+f) for exact f in [-0.293, 0.415], f != 0, as an absolute-error computation in units of ulp_W(f0):
// restoring steps 1+f <- (1+f)(1-+2^-i) (each applied step: two roundings <= ulp(f), an error carried by later steps
// with factor < 1 and into log1p with factor 1/(1+f) <= 1.43), table sums (entry 1 ulp + rounding), then the tail
// 2 atanh(f/(2+f)) by Horner (relative 5.2 units + truncation bounded from the actual |z|).
template<int W> inline Approx<W> log1p_reduced(Number<W> f,LF_D const Tables<W>& t){
    if(f.exponent< -(32*W+2))return approx(f,1.f); // |log1p(f) - f| < f^2/2
    exponent_type ref=f.exponent;bool positive=f.sign>0;
    Number<W> acc=zero<W>();float chain=0,accerr=0;
    // The factors 1+2^-i satisfy log(1+2^-i) <= sum_{j>i} log(1+2^-j), so one pass converges for f < 0. The factors
    // 1-2^-i do not (-log(1-2^-i) exceeds the sum of the later ones by ~4^-i/3) and one pass can lag (f in [0.30, 1/3)
    // ends near 2^-6); a factor that applied is therefore tried once more for f > 0.
    for(int i=2;i<=t.steps;++i)for(int rep=0;rep<(positive?2:1);++rep){
        Number<W> fs=scaled(f,-i);if(positive)fs=negate(fs);
        Number<W> g=add(add(f,unit<W>(-i,positive?-1:1)),fs);
        if(!(positive?g.sign>=0:g.sign<=0))break;
        chain+=p2(exponent_type(f.exponent)-ref);
        Number<W> c=negate(positive?t.down[i]:t.up[i]);acc=add(acc,c);
        accerr+=p2(exponent_type(c.exponent)-ref)+0.5f*p2(exponent_type(acc.exponent)-ref);f=g;
    }
    Number<W> tail=zero<W>();float tailerr=0;
    if(f.sign){
        Number<W> den=add(unit<W>(1),f),z=div(f,den),w=mul(z,z),h=div_word(unit<W>(),word(2*t.k_series+1));
        for(int j=t.k_series-1;j>=0;--j){Number<W> c=div_word(unit<W>(),word(2*j+1));h=add(c,mul(w,h));}
        tail=scaled(mul(z,h),1);
        float tau=p2((2*exponent_type(z.exponent)+2)*(t.k_series+1)+32*W)*1.05f; // w^(K+1)/(1-w), relative units
        tailerr=(5.2f+tau)*p2(exponent_type(tail.exponent)-ref);
    }
    Number<W> y=add(acc,tail);if(!y.sign)return approx(unit<W>(),p2(100));
    float total=1.43f*chain+accerr+tailerr+0.5f*p2(exponent_type(y.exponent)-ref);
    return approx(y,total*p2(ref-exponent_type(y.exponent))*1.02f);
}
// log x (plus_one: log(1+x)) = e ln2 + log1p(m-1) with m in [0.7071, 1.4142]: m-1 is exact and |result| >= 0.346|e|
// for e != 0. log1p(x) with |x| < 1/4 is log1p_reduced(x) directly; otherwise log(RN(1+x)), the rounding moving the
// logarithm by at most 2^-32W. One log1p_reduced per call (each inlined copy costs Metal compile time).
template<int W> inline Approx<W> log_core(Number<W> x,bool plus_one,LF_D const Tables<W>& t){
    if(x.status)return exact(zero<W>(x.status));
    if(plus_one){
        if(!x.sign)return exact(zero<W>());
        if(x.sign<0){Number<W> m=unit<W>(0,-1);if(magnitude_compare(x,m)>=0)return exact(zero<W>(invalid));}
    }else if(x.sign<=0)return exact(zero<W>(invalid));
    exponent_type e=0;Number<W> f=x;bool rounded=false;
    if(!plus_one||x.exponent>=-2){
        Number<W> X=x;if(plus_one){X=add(unit<W>(),x);rounded=true;}
        e=X.exponent;Number<W> m=X;m.exponent=0;if(m.limb[W-1]>=0xB504F334u){m.exponent=-1;++e;}
        f=sub(m,unit<W>());
        if(!f.sign&&!e)return exact(zero<W>()); // x = 1
    }
    Approx<W> l=f.sign?log1p_reduced(f,t):exact(zero<W>());
    if(e){
        Number<W> l2=t.ln2,E=integer<W>(e),y=fma(E,l2,l.y);float ae=float(e<0?-e:e),le=l.err<0?0.f:l.err;
        l=approx(y,(ae*p2(-exponent_type(y.exponent)-1)+le*p2(exponent_type(l.y.exponent)-y.exponent)+0.5f)*1.02f);
    }
    if(rounded)l.err+=p2(-exponent_type(l.y.exponent)-1)*1.02f;
    return l;
}
template<int W> inline Approx<W> log_approx(Number<W> x,LF_D const Tables<W>& t){return log_core(x,false,t);}
template<int W> inline Approx<W> log1p_approx(Number<W> x,LF_D const Tables<W>& t){return log_core(x,true,t);}
// ---- sin and cos ----
// sin r, cos r for |r| <= pi/4 (1+2^-60) with relative error dr: r' = r/2^h (|r'| < 2^-S), Horner series of sin and of
// vers = 1 - cos (first omitted terms < u/16), then h doublings s <- 2 s (1-v), v <- 2 s^2 (v <= 0.293, 1-v >= 0.7).
// Relative errors (units u) in ds, dc.
template<int W> inline void sincos_small(Number<W> r,float dr,LF_D const Tables<W>& t,LF_T Number<W>& s,LF_T Number<W>& c,LF_T float& ds,LF_T float& dc){
    Number<W> one=unit<W>();
    if(!r.sign){s=r;c=one;ds=dr;dc=dr;return;}
    if(r.exponent< -(16*W+2)){s=r;c=one;ds=dr*1.01f+0.1f;dc=0.2f;return;} // r^2/2 < 2^(-32W-3)
    int h=r.exponent+t.s_trig+1;if(h<0)h=0;
    Number<W> x=scaled(r,-h),w=mul(x,x),a=one,b=one;
    for(int j=t.k_trig;j>=1;--j){
        a=sub(one,div_word(mul(w,a),word((2*j)*(2*j+1))));
        b=sub(one,div_word(mul(w,b),word((2*j+1)*(2*j+2))));
    }
    s=mul(x,a);Number<W> v=scaled(mul(w,b),-1);
    float d_s=1.02f*dr+2.2f,d_v=2.02f*dr+3.2f;
    for(int i=0;i<h;++i){
        float q=p2(exponent_type(v.exponent)+1)*1.43f;
        Number<W> cc=sub(one,v),sn=scaled(mul(s,cc),1);v=scaled(mul(s,s),1);s=sn;
        float d_c=q*d_v+1.01f;d_v=2.f*d_s+1.01f;d_s=d_s+d_c+1.01f;
    }
    c=sub(one,v);ds=d_s;dc=p2(exponent_type(v.exponent)+1)*1.43f*d_v+1.01f;
}
// x = q pi/2 + r, |r| <= pi/4 (1+2^-60), by Payne-Hanek: the exact product of the significand with the bits of 2/pi that
// matter mod 4 (a window of TW = 2W+3 words from `bits`, MSB first: bit j has weight 2^-j). Dropped bits change
// x*2/pi by < 2^(32W-F) with F >= 64W+62 fraction bits, so r's relative error is 3.4 + 2^(64W-F-e_f) units.
// |x| < 2^TRIG_EMAX. Small |x| (< pi/4) is returned unchanged with q = 0.
template<int W> inline Number<W> reduce_half_pi(LF_T const Number<W>& x,LF_D const word* bits,LF_D const Tables<W>& t,LF_T int& q,LF_T float& dr){
    q=0;dr=0;
    if(x.exponent< -1||(x.exponent==-1&&x.limb[W-1]<0xC90FDAA2u))return x;
    constexpr int TW=2*W+3,P=W+TW;
    int sigma=x.exponent-(32*W-1),jlo=sigma-1>1?sigma-1:1;
    word tw[scratch(TW)]={},p[scratch(P)]={},fr[scratch(P)]={};
    for(int k=0;k<TW;++k){int p0=jlo-1+32*k,wi=p0>>5,sh=p0&31;word v=bits[wi]<<sh;if(sh)v|=bits[wi+1]>>(32-sh);tw[TW-1-k]=v;}
    for(int i=0;i<W;++i){if(!x.limb[i])continue;dword c=0;
        for(int j=0;j<TW;++j){dword v=dword(x.limb[i])*tw[j]+p[i+j]+c;p[i+j]=word(v);c=v>>32;}
        p[i+TW]=word(c);}
    int F=jlo+32*TW-1-sigma; // |x| * window = p * 2^-F
    word k2=extract(p,F)&3u;bool neg=(extract(p,F-1)&1u)!=0; // fraction >= 1/2: use fraction - 1
    q=int((k2+(neg?1u:0u))&3u);
    int fw=F>>5,fb=F&31;
    for(int i=0;i<P;++i)fr[i]=i<fw?p[i]:(i==fw&&fb?(p[i]&((word(1)<<fb)-1)):0u);
    if(neg){dword c=1;for(int i=0;i<P;++i){dword v=dword(~fr[i])+c;fr[i]=word(v);c=v>>32;}
        for(int i=0;i<P;++i)if(i>fw||(i==fw&&!fb))fr[i]=0;else if(i==fw)fr[i]&=(word(1)<<fb)-1;}
    Number<W> f=pack<W>(fr,-exponent_type(F),1,ok);
    if(!f.sign){dr=p2(100);return f;} // numerically a multiple of pi/2: retry with more words
    Number<W> hp=t.half_pi,r=mul(f,hp);if(neg)r=negate(r);
    if(x.sign<0){r=negate(r);q=(4-q)&3;}
    dr=p2(64*exponent_type(W)-F-f.exponent)*1.01f+3.4f;
    return r;
}
template<int W> inline void sincos(LF_T const Number<W>& x,LF_D const word* bits,LF_D const Tables<W>& t,LF_T Number<W>& s,LF_T Number<W>& c,LF_T float& ds,LF_T float& dc){
    int q;float dr,a,b;Number<W> r=reduce_half_pi(x,bits,t,q,dr),sr,cr;sincos_small(r,dr,t,sr,cr,a,b);
    if(q==0){s=sr;c=cr;ds=a;dc=b;}
    else if(q==1){s=cr;c=negate(sr);ds=b;dc=a;}
    else if(q==2){s=negate(sr);c=negate(cr);ds=a;dc=b;}
    else{s=negate(cr);c=sr;ds=b;dc=a;}
}
template<int W> inline Approx<W> sin_approx(Number<W> x,LF_D const word* bits,LF_D const Tables<W>& t,bool cosine){
    if(x.status)return exact(zero<W>(x.status));
    if(!x.sign)return exact(cosine?unit<W>():zero<W>());
    if(x.exponent>=TRIG_EMAX)return exact(zero<W>(invalid));
    Number<W> s,c;float ds,dc;sincos(x,bits,t,s,c,ds,dc);
    return cosine?approx(c,dc*1.02f):approx(s,ds*1.02f);
}
// ---- atan2 ----
// theta = atan(b/a) for 0 < b <= a by restoring rotations (A,B) <- (A + B 2^-i, B - A 2^-i) when B >= A 2^-i, which
// lower the argument by exactly atan(2^-i); each rounding moves the argument by <= 3*2^(e_B-e_A-33W) after the step.
// The tail atan(B/A) is a Horner series (relative 4.3 units + alternating truncation). Then pi/2 - theta, pi - phi.
template<int W> inline Approx<W> atan2_approx(Number<W> y,Number<W> x,LF_D const Tables<W>& t){
    word st=x.status|y.status;if(st)return exact(zero<W>(st));
    Number<W> pi=t.pi,hp=t.half_pi;
    if(!y.sign){if(!x.sign)return exact(zero<W>(invalid));return x.sign>0?exact(zero<W>()):approx(pi,1.f);}
    if(!x.sign){hp.sign=y.sign;return approx(hp,1.f);}
    Number<W> a=absolute(x),b=absolute(y);bool swap=magnitude_compare(b,a)>0;
    if(swap){Number<W> tmp=a;a=b;b=tmp;}
    exponent_type d=exponent_type(b.exponent)-a.exponent;bool direct=!swap&&x.sign>0;
    Number<W> th;float th_err;exponent_type th_shift=0;
    if(d< -(32*W+8)){ // theta < 2^(d+1): theta = b/a (1 - q^2/3 ...) directly, or negligible next to pi/2, pi
        if(direct){Number<W> bs=b;bs.exponent=a.exponent;th=div(bs,a);th.sign=y.sign;return approx(th,0.7f,d);}
        Number<W> r=swap?hp:pi;r.sign=y.sign;return approx(r,1.1f);
    }else{
        Number<W> A=a,B=b,acc=zero<W>();A.exponent=0;B.exponent=int(d);
        float ang=0,accerr=0; // units 2^(d+1-32W)
        for(int i=1;i<=t.steps;++i){
            Number<W> As=scaled(A,-i);
            if(magnitude_compare(B,As)>=0){
                Number<W> A2=add(A,scaled(B,-i));B=sub(B,As);A=A2;
                if(B.sign)ang+=3.f*p2(exponent_type(B.exponent)-A.exponent-d-1);
                Number<W> c=t.atn[i];acc=add(acc,c);
                accerr+=p2(exponent_type(c.exponent)-d)+0.5f*p2(exponent_type(acc.exponent)-d);
            }
        }
        Number<W> tail=zero<W>();float tailerr=0;
        if(B.sign){
            Number<W> q=div(B,A),w=mul(q,q),h=div_word(unit<W>(),word(2*t.k_series+1));
            for(int j=t.k_series-1;j>=0;--j){Number<W> c=div_word(unit<W>(),word(2*j+1));h=sub(c,mul(w,h));}
            tail=mul(q,h);
            float tau=p2((2*exponent_type(q.exponent)+2)*(t.k_series+1)+32*W)*1.3f;
            tailerr=(4.3f+tau)*p2(exponent_type(tail.exponent)-d);
        }
        th=add(acc,tail);
        th_err=(ang*1.01f+accerr+tailerr+0.5f*p2(exponent_type(th.exponent)-d))*p2(d-exponent_type(th.exponent))*1.02f;
    }
    if(direct){th.sign=y.sign;return approx(th,th_err,th_shift);}
    Number<W> phi=th;float err=th_err;
    if(swap){phi=sub(hp,th);err=(p2(-exponent_type(phi.exponent))+th_err*p2(exponent_type(th.exponent)-phi.exponent)+0.5f)*1.02f;}
    if(x.sign<0){Number<W> r=sub(pi,phi);err=(p2(1-exponent_type(r.exponent))+err*p2(exponent_type(phi.exponent)-r.exponent)+0.5f)*1.02f;phi=r;}
    phi.sign=y.sign;return approx(phi,err);
}
// ---- complex exp and log ----
// exp(a+ib) = e^a (cos b + i sin b): each component RN of its exact value. b = 0 gives an exact zero imaginary part.
template<int W> inline void cexp_approx(Number<W> a,Number<W> b,LF_D const word* bits,LF_D const Tables<W>& t,LF_T Approx<W>& re,LF_T Approx<W>& im){
    word st=a.status|b.status;if(st){re=exact(zero<W>(st));im=re;return;}
    if(b.sign&&b.exponent>=TRIG_EMAX){re=exact(zero<W>(invalid));im=re;return;}
    if(a.sign&&a.exponent>=31){re=exact(zero<W>(exponent_overflow));im=b.sign?re:exact(zero<W>());return;}
    Number<W> s=zero<W>(),c=unit<W>(),E=unit<W>();float ds=0,dc=0,de=0;exponent_type k=0;
    if(b.sign)sincos(b,bits,t,s,c,ds,dc); // b = 0: cos 1, sin 0 exactly
    if(a.sign){
        if(a.exponent< -(32*W+2))de=0.51f;
        else{Number<W> e;float d;E=exp_parts(a,t,k,de,e,d);}
    }
    // Products at exponent 0, the scales in the shifts: no intermediate range check (e^a sin b can be in range when
    // e^a or sin b alone is not).
    exponent_type es=s.exponent,ec=c.exponent;s.exponent=0;c.exponent=0;
    re=approx(mul(E,c),(de+dc+1.01f)*1.02f,k+ec);
    im=b.sign?approx(mul(E,s),(de+ds+1.01f)*1.02f,k+es):exact(zero<W>());
}
// log(a+ib) = log|z| + i atan2(b,a), principal branch. log|z|: for max exponent e in {-1,0} as log1p(RN(a^2+b^2-1))/2
// with exact squares (no cancellation loss near |z| = 1); otherwise e ln2 + log(RN((a^2+b^2) 4^-e))/2.
template<int W> inline void clog_approx(Number<W> a,Number<W> b,LF_D const Tables<W>& t,LF_T Approx<W>& re,LF_T Approx<W>& im){
    word st=a.status|b.status;if(st){re=exact(zero<W>(st));im=re;return;}
    if(!a.sign&&!b.sign){re=exact(zero<W>(invalid));im=re;return;}
    im=atan2_approx(b,a,t);
    // One logarithm: v (plus_one: log1p), then halving and e ln2. extra: absolute error c 2^(x-32W) of the logarithm
    // from rounding its argument (1 + t >= 1/4: RN(t) moves log1p by <= 2 ulp(t); RN(s) moves log s by <= 2^-32W).
    Number<W> v;bool p1=false,halve=false;int e=0;float extra=0;exponent_type ex=0;
    if(!b.sign)v=absolute(a);else if(!a.sign)v=absolute(b);
    else{
        e=a.exponent>b.exponent?a.exponent:b.exponent;halve=true;
        if(e==0||e==-1){
            Number<W> m1=unit<W>(0,-1);v=dot2_add(a,a,b,b,m1);e=0;
            if(v.status||!v.sign){re=exact(zero<W>(v.status));return;} // |t| < 2^-1e9: |log|z|| out of range too
            p1=true;extra=2.f;ex=exponent_type(v.exponent)+1;
        }else{Number<W> as=scaled(a,-e),bs=scaled(b,-e);v=dot2_add(as,as,bs,bs,zero<W>());extra=1.f;} // v in [1, 8)
    }
    Approx<W> l=log_core(v,p1,t);
    if(extra>0){if(l.err<0)l.err=0;l.err+=extra*p2(ex-exponent_type(l.y.exponent)-1)*1.02f;}
    if(halve)l.y=scaled(l.y,-1);
    if(e){
        Number<W> l2=t.ln2,E=integer<W>(e),y=fma(E,l2,l.y);float ae=float(e<0?-e:e);
        l=approx(y,(ae*p2(-exponent_type(y.exponent)-1)+l.err*p2(exponent_type(l.y.exponent)-y.exponent)+0.5f)*1.02f);
    }
    re=l;
}
// ---- integer powers (N words, documented composition; not correctly rounded) ----
// Fused complex product: each component one rounding of its exact value (core.hpp cfma with c = 0).
template<int N> inline Complex<N> cmul_fused(Complex<N> a,Complex<N> b){Complex<N> c={zero<N>(),zero<N>()};return cfma(a,b,c);}
// 1/w = conj(w)/d with d = RN(re^2 + im^2) (one rounding) and RN quotients, evaluated as if the exponent range were
// unbounded (power-of-two prescaling commutes with RN); then the range check.
template<int N> inline Complex<N> creciprocal(Complex<N> w){
    word st=w.re.status|w.im.status;if(st)return {zero<N>(st),zero<N>(st)};
    if(!w.re.sign&&!w.im.sign)return {zero<N>(division_by_zero),zero<N>(division_by_zero)};
    int e=!w.re.sign?w.im.exponent:!w.im.sign?w.re.exponent:(w.re.exponent>w.im.exponent?w.re.exponent:w.im.exponent);
    Number<N> a=scaled(w.re,-e),b=scaled(w.im,-e),d=dot2_add(a,a,b,b,zero<N>());
    Number<N> parts[2]={a,negate(b)},out[2];
    for(int i=0;i<2;++i){
        Number<N> p=parts[i];if(!p.sign){out[i]=zero<N>();continue;}
        exponent_type ep=p.exponent;p.exponent=0;Number<N> q=div(p,d);
        exponent_type ex=exponent_type(q.exponent)+ep-e;
        if(ex>1000000000||ex< -1000000000)q=zero<N>(exponent_overflow);else q.exponent=int(ex);
        out[i]=q;
    }
    Complex<N> r={out[0],out[1]};
    return r;
}
// z^k by right-to-left binary powering of |k| with fused products: acc <- acc*p for each set bit (the first set bit
// copies p), p <- p*p while bits remain; then creciprocal for k < 0. k = 0 gives 1 (also for z = 0).
template<int N> inline Complex<N> cpowi(Complex<N> z,int k){
    word st=z.re.status|z.im.status;if(st)return {zero<N>(st),zero<N>(st)};
    if(!k)return {unit<N>(),zero<N>()};
    word n=k<0?word(-(k+1))+1u:word(k);
    // One product call site (each inlined fused product costs Metal compile time): acc*p for a set low bit, else p*p.
    Complex<N> p=z,acc=z;bool have=false;
    for(;;){
        bool m=(n&1u)!=0;
        if(m&&!have){acc=p;have=true;n^=1u;continue;}
        if(!m&&n<=1u)break;
        Complex<N> x=m?acc:p,r=cmul_fused(x,p);
        if(m){acc=r;n^=1u;}else{p=r;n>>=1;}
    }
    return k<0?creciprocal(acc):acc;
}
} // namespace transcendental
} // namespace limbforge
