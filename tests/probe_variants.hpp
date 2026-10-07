// Exact 2N-limb product formulations for the GPU codegen probe. Valid C++ and Metal;
// included after core.hpp. Every variant must produce the schoolbook integer exactly.
#ifdef __METAL_VERSION__
#define PROBE_THREAD thread
#define PROBE_UNROLL _Pragma("clang loop unroll(full)")
#define PROBE_NOUNROLL _Pragma("clang loop unroll(disable)")
inline limbforge::word probe_mulhi(limbforge::word a,limbforge::word b){return metal::mulhi(a,b);}
#else
#define PROBE_THREAD
#define PROBE_UNROLL
#define PROBE_NOUNROLL
inline limbforge::word probe_mulhi(limbforge::word a,limbforge::word b){return limbforge::word((limbforge::dword(a)*b)>>32);}
#endif
namespace limbforge { namespace probe {
enum Variant { schoolbook, comba64, comba32, comba_mulhi, comba64_unrolled, comba64_rolled, schoolbook_mulhi, comba64_once, comba_mulhi_once, schoolbook_rolled, schoolbook_outer_rolled, schoolbook_padded, schoolbook_unrolled, variant_count };
template<int N> inline void clear(PROBE_THREAD word (&p)[2*N+1]){for(int k=0;k<2*N+1;++k)p[k]=0;}
// The accepted library product.
template<int N> inline void school(PROBE_THREAD const Number<N>& a,PROBE_THREAD const Number<N>& b,PROBE_THREAD word (&p)[2*N+1]){
    clear<N>(p);
    for(int i=0;i<N;++i){dword carry=0;
        for(int j=0;j<N;++j){dword v=dword(a.limb[i])*b.limb[j]+p[i+j]+carry;p[i+j]=word(v);carry=v>>32;}
        p[i+N]=word(carry);}
}
// Schoolbook with only 32-bit operations: low/high halves via mulhi, explicit carries.
template<int N> inline void school_hi(PROBE_THREAD const Number<N>& a,PROBE_THREAD const Number<N>& b,PROBE_THREAD word (&p)[2*N+1]){
    clear<N>(p);
    for(int i=0;i<N;++i){word carry=0;
        for(int j=0;j<N;++j){word x=a.limb[i],y=b.limb[j],lo=x*y,hi=probe_mulhi(x,y);
            word s=p[i+j]+lo;hi+=s<lo;word t=s+carry;hi+=t<s;p[i+j]=t;carry=hi;}
        p[i+N]=carry;}
}
// Rejected round-5 formulation (comba_full.patch): 64-bit low accumulator plus a carry word.
template<int N> inline void comba(PROBE_THREAD const Number<N>& a,PROBE_THREAD const Number<N>& b,PROBE_THREAD word (&p)[2*N+1]){
    clear<N>(p);dword low=0;word high=0;
    for(int column=0;column<2*N-1;++column){
        int first=column<N?0:column-(N-1),last=column<N?column:N-1;
        for(int i=first;i<=last;++i){dword term=dword(a.limb[i])*b.limb[column-i];dword old=low;low+=term;high+=word(low<old);}
        p[column]=word(low);low=(low>>32)|(dword(high)<<32);high=0;}
    p[2*N-1]=word(low);
}
// Rejected round-5 formulation (comba_32bit_carries.patch): three words, bounded 33-bit sums.
template<int N> inline void comba_words(PROBE_THREAD const Number<N>& a,PROBE_THREAD const Number<N>& b,PROBE_THREAD word (&p)[2*N+1]){
    clear<N>(p);word low=0,middle=0,high=0;
    for(int column=0;column<2*N-1;++column){
        int first=column<N?0:column-(N-1),last=column<N?column:N-1;
        for(int i=first;i<=last;++i){dword term=dword(a.limb[i])*b.limb[column-i];
            dword sum=dword(low)+word(term);low=word(sum);
            sum=dword(middle)+(term>>32)+(sum>>32);middle=word(sum);high+=word(sum>>32);}
        p[column]=low;low=middle;middle=high;high=0;}
    p[2*N-1]=low;
}
// Column accumulation using 32-bit operations only.
template<int N> inline void comba_hi(PROBE_THREAD const Number<N>& a,PROBE_THREAD const Number<N>& b,PROBE_THREAD word (&p)[2*N+1]){
    clear<N>(p);word c0=0,c1=0,c2=0;
    for(int column=0;column<2*N-1;++column){
        int first=column<N?0:column-(N-1),last=column<N?column:N-1;
        for(int i=first;i<=last;++i){word x=a.limb[i],y=b.limb[column-i],lo=x*y,hi=probe_mulhi(x,y);
            c0+=lo;word k=c0<lo;c1+=hi;word k2=c1<hi;c1+=k;k2+=c1<k;c2+=k2;}
        p[column]=c0;c0=c1;c1=c2;c2=0;}
    p[2*N-1]=c0;
}
// Same arithmetic as comba, with fixed inner bounds so full unrolling makes every index constant.
template<int N> inline void comba_unrolled(PROBE_THREAD const Number<N>& a,PROBE_THREAD const Number<N>& b,PROBE_THREAD word (&p)[2*N+1]){
    clear<N>(p);dword low=0;word high=0;
    PROBE_UNROLL for(int column=0;column<2*N-1;++column){
        PROBE_UNROLL for(int i=0;i<N;++i){int j=column-i;if(j<0||j>=N)continue;
            dword term=dword(a.limb[i])*b.limb[j];dword old=low;low+=term;high+=word(low<old);}
        p[column]=word(low);low=(low>>32)|(dword(high)<<32);high=0;}
    p[2*N-1]=word(low);
}
// Same arithmetic as comba, with unrolling forbidden: every index stays dynamic.
template<int N> inline void comba_rolled(PROBE_THREAD const Number<N>& a,PROBE_THREAD const Number<N>& b,PROBE_THREAD word (&p)[2*N+1]){
    clear<N>(p);dword low=0;word high=0;
    PROBE_NOUNROLL for(int column=0;column<2*N-1;++column){
        int first=column<N?0:column-(N-1),last=column<N?column:N-1;
        PROBE_NOUNROLL for(int i=first;i<=last;++i){dword term=dword(a.limb[i])*b.limb[column-i];dword old=low;low+=term;high+=word(low<old);}
        p[column]=word(low);low=(low>>32)|(dword(high)<<32);high=0;}
    p[2*N-1]=word(low);
}
// As comba / comba_hi, but every word is stored exactly once and nothing is zeroed first.
template<int N> inline void comba_once(PROBE_THREAD const Number<N>& a,PROBE_THREAD const Number<N>& b,PROBE_THREAD word (&p)[2*N+1]){
    dword low=0;word high=0;
    for(int column=0;column<2*N-1;++column){
        int first=column<N?0:column-(N-1),last=column<N?column:N-1;
        for(int i=first;i<=last;++i){dword term=dword(a.limb[i])*b.limb[column-i];dword old=low;low+=term;high+=word(low<old);}
        p[column]=word(low);low=(low>>32)|(dword(high)<<32);high=0;}
    p[2*N-1]=word(low);p[2*N]=0;
}
template<int N> inline void comba_hi_once(PROBE_THREAD const Number<N>& a,PROBE_THREAD const Number<N>& b,PROBE_THREAD word (&p)[2*N+1]){
    word c0=0,c1=0,c2=0;
    for(int column=0;column<2*N-1;++column){
        int first=column<N?0:column-(N-1),last=column<N?column:N-1;
        for(int i=first;i<=last;++i){word x=a.limb[i],y=b.limb[column-i],lo=x*y,hi=probe_mulhi(x,y);
            c0+=lo;word k=c0<lo;c1+=hi;word k2=c1<hi;c1+=k;k2+=c1<k;c2+=k2;}
        p[column]=c0;c0=c1;c1=c2;c2=0;}
    p[2*N-1]=c0;p[2*N]=0;
}
// The accepted schoolbook with unrolling forbidden in both loops, or only in the outer loop.
template<int N> inline void school_rolled(PROBE_THREAD const Number<N>& a,PROBE_THREAD const Number<N>& b,PROBE_THREAD word (&p)[2*N+1]){
    clear<N>(p);
    PROBE_NOUNROLL for(int i=0;i<N;++i){dword carry=0;
        PROBE_NOUNROLL for(int j=0;j<N;++j){dword v=dword(a.limb[i])*b.limb[j]+p[i+j]+carry;p[i+j]=word(v);carry=v>>32;}
        p[i+N]=word(carry);}
}
template<int N> inline void school_outer_rolled(PROBE_THREAD const Number<N>& a,PROBE_THREAD const Number<N>& b,PROBE_THREAD word (&p)[2*N+1]){
    clear<N>(p);
    PROBE_NOUNROLL for(int i=0;i<N;++i){dword carry=0;
        for(int j=0;j<N;++j){dword v=dword(a.limb[i])*b.limb[j]+p[i+j]+carry;p[i+j]=word(v);carry=v>>32;}
        p[i+N]=word(carry);}
}
// Schoolbook into a scratch array of at least 33 words (kept out of registers), then copied.
template<int N> inline void school_padded(PROBE_THREAD const Number<N>& a,PROBE_THREAD const Number<N>& b,PROBE_THREAD word (&p)[2*N+1]){
    constexpr int W=2*N+1<33?33:2*N+1;word q[W];for(int k=0;k<W;++k)q[k]=0;
    for(int i=0;i<N;++i){dword carry=0;
        for(int j=0;j<N;++j){dword v=dword(a.limb[i])*b.limb[j]+q[i+j]+carry;q[i+j]=word(v);carry=v>>32;}
        q[i+N]=word(carry);}
    for(int k=0;k<2*N+1;++k)p[k]=q[k];
}
// Schoolbook with both loops fully unrolled: every array index is a compile-time constant.
template<int N> inline void school_unrolled(PROBE_THREAD const Number<N>& a,PROBE_THREAD const Number<N>& b,PROBE_THREAD word (&p)[2*N+1]){
    clear<N>(p);
    PROBE_UNROLL for(int i=0;i<N;++i){dword carry=0;
        PROBE_UNROLL for(int j=0;j<N;++j){dword v=dword(a.limb[i])*b.limb[j]+p[i+j]+carry;p[i+j]=word(v);carry=v>>32;}
        p[i+N]=word(carry);}
}
template<int N> inline void product(int variant,PROBE_THREAD const Number<N>& a,PROBE_THREAD const Number<N>& b,PROBE_THREAD word (&p)[2*N+1]){
    switch(variant){case comba64:comba<N>(a,b,p);break;case comba32:comba_words<N>(a,b,p);break;case comba_mulhi:comba_hi<N>(a,b,p);break;
        case comba64_unrolled:comba_unrolled<N>(a,b,p);break;case comba64_rolled:comba_rolled<N>(a,b,p);break;
        case schoolbook_mulhi:school_hi<N>(a,b,p);break;case comba64_once:comba_once<N>(a,b,p);break;case comba_mulhi_once:comba_hi_once<N>(a,b,p);break;case schoolbook_rolled:school_rolled<N>(a,b,p);break;case schoolbook_outer_rolled:school_outer_rolled<N>(a,b,p);break;case schoolbook_padded:school_padded<N>(a,b,p);break;case schoolbook_unrolled:school_unrolled<N>(a,b,p);break;default:school<N>(a,b,p);}
}
template<int N> inline Number<N> mul(int variant,PROBE_THREAD const Number<N>& a,PROBE_THREAD const Number<N>& b){
    word status=a.status|b.status;if(status||!a.sign||!b.sign)return zero<N>(status);
    word p[2*N+1];product<N>(variant,a,b,p);
    return pack<N>(p,exponent_type(a.exponent)+b.exponent-2*(32*N-1),a.sign*b.sign,status);
}
template<int N> inline Complex<N> cmul(int variant,Complex<N> a,Complex<N> b){
    return {sub(mul<N>(variant,a.re,b.re),mul<N>(variant,a.im,b.im)),add(mul<N>(variant,a.re,b.im),mul<N>(variant,a.im,b.re))};
}
}}
