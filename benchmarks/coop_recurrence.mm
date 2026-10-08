// Experiment L4 (docs/optimization-plan.md §8, E1): G SIMD lanes cooperate on one recurrence trajectory.
// Each number is top-aligned in M=G*L limb slots (L=ceil(N/G) limbs per lane; the M-N low slots are zero).
// Exact products and sums live in a 2M-slot workspace: lane k holds block k (lo) and block k+G (hi).
// Carries use an associative generate/propagate scan over ballot masks; rounding is pack()'s RN-even.
// Results must equal the core (CPU) and the existing `recurrence` kernel bit for bit; then per-step latency
// is compared with interleaved A/B samples. Not a library feature: nothing outside this file changes.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "reference.hpp"
#include "shader_source.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <map>
#include <sstream>
using namespace limbforge;
using Clock=std::chrono::steady_clock;
const char* coop_source=R"METAL(
namespace coop {
constexpr int limbs(int g){return (N+g-1)/g;}
template<int L> struct CN { uint l[L]; int exponent,sign; uint status; };
template<int L> struct CW { uint lo[L],hi[L]; };
template<int L> struct CC { CN<L> re,im; };
struct Ctx { uint k,base; };
inline uint shfl(uint x,uint lane){return simd_shuffle(x,ushort(lane));}
inline int shfl(int x,uint lane){return simd_shuffle(x,ushort(lane));}
// This group's G ballot bits, lane k at bit k.
template<int G> inline ulong vote(bool b,uint base){ulong v=ulong(static_cast<simd_vote::vote_t>(simd_ballot(b)));return G==32?v&0xffffffffUL:(v>>base)&((ulong(1)<<G)-1);}
// Block carry-ins from generate/propagate masks (exclusive per block) and an initial carry: bit t = carry into block t.
inline ulong carries(ulong g,ulong p,ulong c0){ulong x=g|p;return (x+g+c0)^x^g;}
inline uint funnel_left(uint hi,uint lo,uint s){return uint(((ulong(hi)<<32)|lo)>>(32-s));}
inline uint funnel_right(uint hi,uint lo,uint s){return uint(((ulong(hi)<<32)|lo)>>s);}
inline uint lost_bits(uint x,uint pos,uint sh){return pos+32<=sh?x:pos<sh?x&((1u<<(sh-pos))-1):0u;}
template<int L> inline CN<L> czero(uint status=0){CN<L> r;for(int j=0;j<L;++j)r.l[j]=0;r.exponent=0;r.sign=0;r.status=status;return r;}
template<int L> inline CN<L> cnegate(CN<L> a){a.sign=-a.sign;return a;}
template<int L> inline bool any_set(thread const uint (&x)[L]){uint r=0;for(int j=0;j<L;++j)r|=x[j];return r!=0;}
template<int L> inline bool all_ones(thread const uint (&x)[L]){uint r=~0u;for(int j=0;j<L;++j)r&=x[j];return r==~0u;}
template<int L> inline uint add_small(thread uint (&x)[L],uint c){for(int j=0;j<L;++j){ulong v=ulong(x[j])+c;x[j]=uint(v);c=uint(v>>32);}return c;}
template<int L> inline int top_bit(thread const uint (&x)[L]){int h=-1;for(int j=0;j<L;++j)h=x[j]?32*j+31-int(clz(x[j])):h;return h;}
template<int G> inline CN<limbs(G)> load(device const Number<N>& x,uint k){
    constexpr int L=limbs(G),P=G*L-N;CN<L> r;
    for(int j=0;j<L;++j){int q=int(k)*L+j-P;r.l[j]=q>=0?x.limb[max(q,0)]:0u;}
    r.exponent=x.exponent;r.sign=x.sign;r.status=x.status;return r;
}
template<int G> inline void store(device Number<N>& x,thread const CN<limbs(G)>& r,uint k){
    constexpr int L=limbs(G),P=G*L-N;
    for(int j=0;j<L;++j){int q=int(k)*L+j-P;if(q>=0)x.limb[q]=r.l[j];}
    if(!k){x.exponent=r.exponent;x.sign=r.sign;x.status=r.status;}
}
// Exact 2M-slot product. Iteration t multiplies broadcast block B_t by A_{(k-t) mod G}: the L x L block product
// lands at block k (t<=k) or k+G (t>k), so every lane does the same G-T0 block products (T0 = all-zero low blocks).
template<int G> inline CW<limbs(G)> product(thread const CN<limbs(G)>& a,thread const CN<limbs(G)>& b,Ctx c){
    constexpr int L=limbs(G),T0=(G*L-N)/L,A=2*L+1;
    uint acc[A],save[A];for(int i=0;i<A;++i)acc[i]=save[i]=0;
    for(int t=T0;t<G;++t){
        uint src=c.base+((c.k-uint(t))&(G-1)),x[L],y[L],pr[2*L];
        for(int j=0;j<L;++j){x[j]=shfl(a.l[j],src);y[j]=shfl(b.l[j],c.base+uint(t));pr[j]=pr[j+L]=0;}
        if(t>T0){bool sw=c.k+1==uint(t);for(int i=0;i<A;++i){save[i]=sw?acc[i]:save[i];acc[i]=sw?0u:acc[i];}}
        for(int i=0;i<L;++i){ulong carry=0;for(int j=0;j<L;++j){ulong v=ulong(x[i])*y[j]+pr[i+j]+carry;pr[i+j]=uint(v);carry=v>>32;}pr[i+L]=uint(carry);}
        ulong carry=0;for(int i=0;i<2*L;++i){ulong v=ulong(acc[i])+pr[i]+carry;acc[i]=uint(v);carry=v>>32;}acc[2*L]+=uint(carry);
    }
    bool first=c.k==0,last=c.k==G-1;uint src=c.base+((c.k-1)&(G-1)),la[A],ha[A];
    for(int i=0;i<A;++i){la[i]=last?acc[i]:save[i];ha[i]=last?0u:acc[i];}
    // Block s = low L limbs of acc_s + limbs L..2L of acc_{s-1} (+ small h_s), then + h_{s-1}, then a 0/1 carry scan.
    CW<L> w;uint hl=0,hh=0;ulong cl=0,ch=0;
    for(int j=0;j<=L;++j){uint pl=shfl(la[L+j],src),ph=shfl(ha[L+j],src),il=first?0u:pl,ih=first?pl:ph;
        if(j<L){ulong v=ulong(la[j])+il+cl;w.lo[j]=uint(v);cl=v>>32;v=ulong(ha[j])+ih+ch;w.hi[j]=uint(v);ch=v>>32;}
        else{hl=il+uint(cl);hh=ih+uint(ch);}}
    uint pl=shfl(hl,src),ph=shfl(hh,src),el=add_small(w.lo,first?0u:pl),eh=add_small(w.hi,first?pl:ph);
    ulong C=carries(vote<G>(el!=0,c.base)|(vote<G>(eh!=0,c.base)<<G),vote<G>(all_ones(w.lo),c.base)|(vote<G>(all_ones(w.hi),c.base)<<G),0);
    add_small(w.lo,uint(C>>c.k)&1);add_small(w.hi,uint(C>>(c.k+G))&1);
    return w;
}
// w << s, s in {0,1}: only the previous block's top limb crosses lanes.
template<int G> inline void shift_left_small(thread CW<limbs(G)>& w,uint s,Ctx c){
    constexpr int L=limbs(G);uint src=c.base+((c.k-1)&(G-1)),xl=shfl(w.lo[L-1],src),xh=shfl(w.hi[L-1],src),pl=c.k?xl:0u,ph=c.k?xh:xl;
    for(int j=L-1;j>=0;--j){w.lo[j]=funnel_left(w.lo[j],j?w.lo[j-1]:pl,s);w.hi[j]=funnel_left(w.hi[j],j?w.hi[j-1]:ph,s);}
}
// General left shift of the 2M-slot workspace: whole blocks by one shuffle source, then limbs+bits from the previous block.
template<int G> inline void shift_left(thread CW<limbs(G)>& w,uint sh,Ctx c){
    constexpr int L=limbs(G);uint wl=sh/32,s=sh%32,wb=wl/L,wr=wl%L;
    if(simd_any(wb!=0)){uint src=c.base+((c.k-wb)&(G-1));bool in=c.k>=wb,in2=c.k+G>=wb;
        for(int j=0;j<L;++j){uint xl=shfl(w.lo[j],src),xh=shfl(w.hi[j],src);w.lo[j]=in?xl:0u;w.hi[j]=in?xh:in2?xl:0u;}}
    if(simd_any(wr!=0||s!=0)){uint src=c.base+((c.k-1)&(G-1)),cl[2*L],ch[2*L];bool first=c.k==0;
        for(int j=0;j<L;++j){uint xl=shfl(w.lo[j],src),xh=shfl(w.hi[j],src);cl[j]=first?0u:xl;ch[j]=first?xl:xh;cl[L+j]=w.lo[j];ch[L+j]=w.hi[j];}
        for(int j=0;j<L;++j){uint l0=0,l1=0,h0=0,h1=0;
            for(int r=0;r<L;++r)if(uint(r)==wr){l0=cl[L+j-r];l1=cl[L+j-r-1];h0=ch[L+j-r];h1=ch[L+j-r-1];}
            w.lo[j]=funnel_left(l0,l1,s);w.hi[j]=funnel_left(h0,h1,s);}}
}
// General right shift; returns whether any set bit fell below slot 0 (group-uniform).
template<int G> inline bool shift_right(thread CW<limbs(G)>& w,uint sh,Ctx c){
    constexpr int L=limbs(G),M=G*L;uint lost=0;
    for(int j=0;j<L;++j){uint q=32*(c.k*L+j);lost|=lost_bits(w.lo[j],q,sh)|lost_bits(w.hi[j],q+32*M,sh);}
    bool sticky=vote<G>(lost!=0,c.base)!=0;uint wl=sh/32,s=sh%32,wb=wl/L,wr=wl%L;
    if(simd_any(wb!=0)){uint src=c.base+((c.k+wb)&(G-1)),t=c.k+wb;
        for(int j=0;j<L;++j){uint xl=shfl(w.lo[j],src),xh=shfl(w.hi[j],src);w.lo[j]=t<G?xl:t<2*G?xh:0u;w.hi[j]=t<G?xh:0u;}}
    if(simd_any(wr!=0||s!=0)){uint src=c.base+((c.k+1)&(G-1)),cl[2*L],ch[2*L];bool last=c.k==G-1;
        for(int j=0;j<L;++j){uint xl=shfl(w.lo[j],src),xh=shfl(w.hi[j],src);cl[j]=w.lo[j];ch[j]=w.hi[j];cl[L+j]=last?xh:xl;ch[L+j]=last?0u:xh;}
        for(int j=0;j<L;++j){uint l0=0,l1=0,h0=0,h1=0;
            for(int r=0;r<L;++r)if(uint(r)==wr){l0=cl[j+r];l1=cl[j+r+1];h0=ch[j+r];h1=ch[j+r+1];}
            w.lo[j]=funnel_right(l1,l0,s);w.hi[j]=funnel_right(h1,h0,s);}}
    return sticky;
}
// pack() for a workspace whose msb is bit 64M-1 and whose value is w * 2^(e-(64M-1)): keep the top N limbs,
// round to nearest even from the round bit and sticky bits, as core pack()/increment()/checked().
template<int G> inline CN<limbs(G)> round(thread const CW<limbs(G)>& w,long e,int sign,uint status,Ctx c){
    constexpr int L=limbs(G),M=G*L,P=M-N,R=2*M-N-1,BR=R/L,PR=R%L,KQ=P/L,PQ=P%L;
    bool range=e>1000000000||e< -1000000001;uint below=0;bool rb;
    for(int j=0;j<L;++j){int q=int(c.k)*L+j,q2=q+M;below|=q<R?w.lo[j]:q==R?w.lo[j]&0x7fffffffu:0u;below|=q2<R?w.hi[j]:q2==R?w.hi[j]&0x7fffffffu:0u;}
    if(BR<G)rb=c.k==uint(BR)&&(w.lo[PR]>>31);else rb=c.k==uint(BR-G)&&(w.hi[PR]>>31);
    bool lsb=c.k==uint(KQ)&&(w.hi[PQ]&1);
    bool up=vote<G>(rb,c.base)!=0&&vote<G>(below!=0||lsb,c.base)!=0;
    CN<L> r;r.sign=sign;r.status=status;
    for(int j=0;j<L;++j)r.l[j]=int(c.k)*L+j>=P?w.hi[j]:0u;
    if(simd_any(up)){uint cc=up&&c.k==uint(KQ)?1u:0u;bool prop=c.k>uint(KQ)&&all_ones(r.l);
        for(int j=PQ;j<L;++j){ulong v=ulong(r.l[j])+cc;r.l[j]=uint(v);cc=uint(v>>32);}
        ulong C=carries(vote<G>(cc!=0,c.base),vote<G>(prop,c.base),0);add_small(r.l,uint(C>>c.k)&1);
        if((C>>G)&1){for(int j=0;j<L;++j)r.l[j]=0;if(c.k==G-1)r.l[L-1]=0x80000000u;++e;}}
    r.exponent=int(e);
    return range||e>1000000000||e< -1000000000?czero<L>(status|exponent_overflow):r;
}
// General pack(): value = w * 2^scale.
template<int G> inline CN<limbs(G)> pack(thread CW<limbs(G)>& w,long scale,int sign,uint status,Ctx c){
    constexpr int L=limbs(G),M=G*L;
    ulong z=vote<G>(any_set(w.lo),c.base)|(vote<G>(any_set(w.hi),c.base)<<G);
    uint tb=z?63-uint(clz(z)):0;int hb=shfl(tb>=uint(G)?top_bit(w.hi):top_bit(w.lo),c.base+(tb&(G-1)));
    long h=max(long(tb)*32*L+hb,0L);shift_left<G>(w,uint(64*M-1-h),c);
    CN<L> r=round<G>(w,scale+h,sign,status,c);return z?r:czero<L>(status);
}
template<int L> inline CN<L> pick(bool p,thread const CN<L>& x,thread const CN<L>& y){
    CN<L> r;for(int j=0;j<L;++j)r.l[j]=p?x.l[j]:y.l[j];r.exponent=p?x.exponent:y.exponent;r.sign=p?x.sign:y.sign;r.status=p?x.status:y.status;return r;}
// Every branch that contains a SIMD shuffle or ballot is uniform across the whole SIMD group (simd_any/simd_all);
// group-specific special cases are selected afterwards (docs/experiments.md: SIMD ops in divergent control flow).
template<int G> inline CN<limbs(G)> mul(thread const CN<limbs(G)>& a,thread const CN<limbs(G)>& b,Ctx c){
    constexpr int L=limbs(G);
    uint status=a.status|b.status;bool special=status||!a.sign||!b.sign;CN<L> r=czero<L>(status);
    if(!simd_all(special)){CW<L> w=product<G>(a,b,c);
        // Normalized inputs: the product's msb is bit 64M-1 or 64M-2.
        uint d=(shfl(w.hi[L-1],c.base+G-1)>>31)^1;shift_left_small<G>(w,d,c);
        CN<L> p=round<G>(w,long(a.exponent)+b.exponent+1-d,a.sign*b.sign,0,c);r=pick(special,r,p);}
    return r;
}
// add(): order by magnitude; A>>1 leaves a carry bit; B aligned by gap+1 with its lost bits jammed into bit 0
// (>= 32N-1 bits below A's lsb, so the jam cannot move a rounding boundary). Exact otherwise; gap > 32N+2 returns A.
template<int G> inline CN<limbs(G)> add(CN<limbs(G)> a,CN<limbs(G)> b,Ctx c){
    constexpr int L=limbs(G),M=G*L;
    uint status=a.status|b.status;CN<L> early=status?czero<L>(status):!a.sign?b:a;bool special=status||!a.sign||!b.sign;
    int cmp=a.exponent>b.exponent?1:a.exponent<b.exponent?-1:0;
    if(simd_any(!cmp)){int lc=0;for(int j=L-1;j>=0;--j)lc=lc?lc:a.l[j]>b.l[j]?1:a.l[j]<b.l[j]?-1:0;
        ulong m=vote<G>(lc!=0,c.base);int t=shfl(lc,c.base+(m?63-uint(clz(m)):0));cmp=cmp?cmp:m?t:0;}
    if(cmp<0){CN<L> t=a;a=b;b=t;}
    long gap=long(a.exponent)-b.exponent;if(!special&&gap>32*N+2){special=true;early=a;}
    CN<L> r=early;
    if(!simd_all(special)){
        CW<L> x,y;bool last=c.k==G-1;uint nx=shfl(a.l[0],c.base+((c.k+1)&(G-1)));
        for(int j=0;j<L;++j){x.hi[j]=funnel_right(j<L-1?a.l[j<L-1?j+1:0]:last?0u:nx,a.l[j],1);x.lo[j]=0;y.lo[j]=0;y.hi[j]=b.l[j];}
        x.lo[L-1]=last?nx<<31:0u;
        if(shift_right<G>(y,uint(clamp(gap,0L,long(32*N+2)))+1,c)&&!c.k)y.lo[0]|=1;
        bool subtract=a.sign!=b.sign;uint m=subtract?~0u:0u;ulong cl=0,ch=0;
        for(int j=0;j<L;++j){ulong v=ulong(x.lo[j])+(y.lo[j]^m)+cl;x.lo[j]=uint(v);cl=v>>32;v=ulong(x.hi[j])+(y.hi[j]^m)+ch;x.hi[j]=uint(v);ch=v>>32;}
        ulong C=carries(vote<G>(cl!=0,c.base)|(vote<G>(ch!=0,c.base)<<G),vote<G>(all_ones(x.lo),c.base)|(vote<G>(all_ones(x.hi),c.base)<<G),subtract?1:0);
        add_small(x.lo,uint(C>>c.k)&1);add_small(x.hi,uint(C>>(c.k+G))&1);
        CN<L> p=pack<G>(x,long(a.exponent)+2-64*M,a.sign,0,c);r=pick(special,r,p);}
    return r;
}
template<int G> inline CN<limbs(G)> sub(CN<limbs(G)> a,CN<limbs(G)> b,Ctx c){return add<G>(a,cnegate(b),c);}
template<int G> inline CC<limbs(G)> cadd(CC<limbs(G)> a,CC<limbs(G)> b,Ctx c){return {add<G>(a.re,b.re,c),add<G>(a.im,b.im,c)};}
template<int G> inline CC<limbs(G)> cmul(CC<limbs(G)> a,CC<limbs(G)> b,Ctx c){
    return {sub<G>(mul<G>(a.re,b.re,c),mul<G>(a.im,b.im,c),c),add<G>(mul<G>(a.re,b.im,c),mul<G>(a.im,b.re,c),c)};}
template<int G> inline CC<limbs(G)> cload(device const Complex<N>& x,uint k){return {load<G>(x.re,k),load<G>(x.im,k)};}
template<int G> inline void cstore(device Complex<N>& x,thread const CC<limbs(G)>& r,uint k){store<G>(x.re,r.re,k);store<G>(x.im,r.im,k);}
template<int G> inline void real_body(device const Number<N>* a,device const Number<N>* b,device Number<N>* out,constant Params& p,uint gid,uint lane){
    uint i=gid/G;if(i>=p.count)return;Ctx c={lane&(G-1),lane&~uint(G-1)};
    CN<limbs(G)> x=load<G>(a[i],c.k),y=load<G>(b[i],c.k),z;
    switch(p.operation){case 0:z=add<G>(x,y,c);break;case 1:z=sub<G>(x,y,c);break;default:z=mul<G>(x,y,c);}
    store<G>(out[i],z,c.k);
}
template<int G> inline void complex_body(device const Complex<N>* a,device const Complex<N>* b,device Complex<N>* out,constant Params& p,uint gid,uint lane){
    uint i=gid/G;if(i>=p.count)return;Ctx c={lane&(G-1),lane&~uint(G-1)};
    CC<limbs(G)> x=cload<G>(a[i],c.k),y=cload<G>(b[i],c.k),z=p.operation==4?cadd<G>(x,y,c):cmul<G>(x,y,c);
    cstore<G>(out[i],z,c.k);
}
// Same step order, weights indexing and composed roundings as kernels.metal `recurrence`.
template<int G> inline void recurrence_body(device const Complex<N>* seeds,device const Complex<N>* weights,device Complex<N>* out,constant Params& p,uint gid,uint lane){
    uint i=gid/G;if(i>=p.count)return;Ctx c={lane&(G-1),lane&~uint(G-1)};
    CC<limbs(G)> q0=cload<G>(seeds[i],c.k),q1=cload<G>(seeds[p.count+i],c.k),q2=cload<G>(seeds[2*p.count+i],c.k),q3=cload<G>(seeds[3*p.count+i],c.k);
    uint wi=i/p.states_per_weight;
    for(uint step=0;step<p.steps;++step){
        CC<limbs(G)> sum={czero<limbs(G)>(),czero<limbs(G)>()};uint base=step*4*p.weight_count+wi;
        sum=cadd<G>(sum,cmul<G>(cload<G>(weights[base],c.k),q0,c),c);
        sum=cadd<G>(sum,cmul<G>(cload<G>(weights[base+p.weight_count],c.k),q1,c),c);
        sum=cadd<G>(sum,cmul<G>(cload<G>(weights[base+2*p.weight_count],c.k),q2,c),c);
        sum=cadd<G>(sum,cmul<G>(cload<G>(weights[base+3*p.weight_count],c.k),q3,c),c);
        q0=q1;q1=q2;q2=q3;q3=sum;
    }
    cstore<G>(out[i],q3,c.k);
}
} // namespace coop
#define COOP_KERNELS(G) \
kernel void coop_real##G(device const Number<N>* a [[buffer(0)]],device const Number<N>* b [[buffer(1)]],device Number<N>* out [[buffer(2)]],constant Params& p [[buffer(3)]],\
    uint gid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){coop::real_body<G>(a,b,out,p,gid,lane);}\
kernel void coop_complex##G(device const Complex<N>* a [[buffer(0)]],device const Complex<N>* b [[buffer(1)]],device Complex<N>* out [[buffer(2)]],constant Params& p [[buffer(3)]],\
    uint gid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){coop::complex_body<G>(a,b,out,p,gid,lane);}\
kernel void recurrence_coop##G(device const Complex<N>* s [[buffer(0)]],device const Complex<N>* w [[buffer(1)]],device Complex<N>* out [[buffer(2)]],constant Params& p [[buffer(3)]],\
    uint gid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){coop::recurrence_body<G>(s,w,out,p,gid,lane);}
COOP_KERNELS(4)
COOP_KERNELS(8)
COOP_KERNELS(16)
COOP_KERNELS(32)
)METAL";
struct Params {std::uint32_t count,operation,steps,weight_count,states_per_weight,b_stride=0,b_period=0,c_stride=0,c_period=0;}; // matches kernels.metal
struct Options {std::vector<int> bits={256,384,1024};std::vector<unsigned> lanes={32,128,512,2048,8192},groups={4,8,16,32};
    unsigned steps=64,tg=32;int samples=9;double warm=0.2;bool time=true,check=true,keep_going=false;std::size_t failed=0,count=4096;};
Options opt;id<MTLDevice> device;id<MTLCommandQueue> queue;
void check(bool good,const std::string& message){if(!good)throw std::runtime_error(message);}
double median(std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];}
id<MTLBuffer> buffer(const void* data,std::size_t bytes){id<MTLBuffer> b=[device newBufferWithLength:std::max<std::size_t>(bytes,1) options:MTLResourceStorageModeShared];
    check(b!=nil,"buffer allocation failed");if(data)std::memcpy(b.contents,data,bytes);return b;}
struct Library {
    id<MTLLibrary> library;std::map<std::string,id<MTLComputePipelineState>> states;
    id<MTLComputePipelineState> get(const std::string& name){auto& s=states[name];if(s)return s;NSError* error=nil;
        id<MTLFunction> f=[library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];check(f!=nil,"missing kernel "+name);
        s=[device newComputePipelineStateWithFunction:f error:&error];check(s!=nil,"pipeline "+name+": "+(error?[[error localizedDescription] UTF8String]:""));
        check(s.threadExecutionWidth==32,"cooperative kernels assume 32-wide SIMD groups");return s;}
};
// One dispatch of `threads` threads; returns device seconds.
double run(id<MTLComputePipelineState> state,id<MTLBuffer> a,id<MTLBuffer> b,id<MTLBuffer> out,const Params& p,std::size_t threads,unsigned group){
    @autoreleasepool{id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:state];[encoder setBuffer:a offset:0 atIndex:0];[encoder setBuffer:b offset:0 atIndex:1];[encoder setBuffer:out offset:0 atIndex:2];
    [encoder setBytes:&p length:sizeof(p) atIndex:3];NSUInteger g=std::min<NSUInteger>(group,state.maxTotalThreadsPerThreadgroup);g-=g%32;
    [encoder dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(g,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];
    if(command.status==MTLCommandBufferStatusError)throw std::runtime_error(std::string("execution failed: ")+[[command.error localizedDescription] UTF8String]);
    return command.GPUEndTime-command.GPUStartTime;}
}
template<int N> struct Width {
    static constexpr int Bits=32*N;using F=Number<N>;using C=Complex<N>;Library lib;std::mt19937_64 rng{20261007u+Bits};
    std::size_t real_bad=0,complex_bad=0,recurrence_bad=0,checks=0;
    Width(){NSString* source=[NSString stringWithFormat:@"#define MP_BITS %d\n%s\n%s",Bits,limbforge_shader_source,coop_source];
        MTLCompileOptions* options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;options.languageVersion=MTLLanguageVersion3_1;
        NSError* error=nil;lib.library=[device newLibraryWithSource:source options:options error:&error];
        check(lib.library!=nil,std::string("Metal compilation: ")+(error?[[error localizedDescription] UTF8String]:""));}
    F rnd(int span){return reference::random_number<Bits>(rng,span);}
    static F ones(int e,int s){F x=zero<N>();for(auto& l:x.limb)l=~0u;x.exponent=e;x.sign=s;return x;}
    static F power(int e,int s){F x=zero<N>();x.limb[N-1]=0x80000000u;x.exponent=e;x.sign=s;return x;}
    // Random pairs plus cancellation, alignment, tie, carry, overflow, zero and status edge cases.
    void real_cases(std::vector<F>& a,std::vector<F>& b){
        for(std::size_t i=0;i<opt.count;++i){a.push_back(rnd(5));b.push_back(rnd(5));}
        auto pair=[&](F x,F y){a.push_back(x);b.push_back(y);};
        for(int r=0;r<16;++r){F x=rnd(5),y=x;pair(x,y);y.sign=-x.sign;pair(x,y);
            for(int keep=1;keep<N;++keep){y=x;y.sign=-x.sign;for(int j=0;j<N-keep;++j)y.limb[j]=std::uint32_t(rng());pair(x,y);}
            y=x;y.sign=-x.sign;y.limb[0]^=1;pair(x,y);y=x;y.sign=-x.sign;y.limb[0]+=1;pair(x,y);
            pair(power(x.exponent,1),ones(x.exponent-1,-1));pair(power(x.exponent,-1),ones(x.exponent-1,1));pair(power(x.exponent,1),rnd(0));
            for(int gap:{1,2,3,4,31,32,33,34,63,64,65,32*N-33,32*N-32,32*N-31,32*N-3,32*N-2,32*N-1,32*N,32*N+1,32*N+2,32*N+3,32*N+4,32*N+40})
                for(int s:{1,-1}){y=rnd(0);y.exponent=x.exponent-gap;y.sign=s*x.sign;pair(x,y);pair(y,x);
                    pair(x,power(x.exponent-gap,s));F z=power(x.exponent-gap,s);z.limb[0]=1;pair(x,z);pair(x,ones(x.exponent-gap,s));
                    pair(ones(x.exponent,1),power(x.exponent-gap,s));pair(ones(x.exponent,-1),ones(x.exponent-gap,s));}
            F o=ones(3,1);pair(o,o);pair(o,power(-29,1));pair(power(0,1),power(0,1));pair(o,rnd(5));
        }
        // Structured limbs (0, 1, all ones, ...) give long carry chains in block products and sums.
        const std::uint32_t pattern[]={0u,1u,~0u,~0u-1,0x80000000u,0x7fffffffu,~0u,~0u};
        for(int r=0;r<2048;++r){F x=rnd(5),y=rnd(5);for(int j=0;j<N;++j){x.limb[j]=pattern[rng()%8];y.limb[j]=pattern[rng()%8];}
            x.limb[N-1]|=0x80000000u;y.limb[N-1]|=0x80000000u;if(r%4==0)y.exponent=x.exponent-int(rng()%(32*N+4));pair(x,y);}
        pair(zero<N>(),rnd(5));pair(rnd(5),zero<N>());pair(zero<N>(),zero<N>());
        F big=rnd(0);big.exponent=1000000000;pair(big,big);F small=rnd(0);small.exponent=-1000000000;pair(small,small);
        F half=rnd(0);half.exponent=600000000;pair(half,half);half.exponent=-600000000;pair(half,half);pair(ones(1000000000,1),ones(1000000000,1));
        F bad=rnd(5);bad.status=invalid;pair(bad,rnd(5));pair(rnd(5),bad);
    }
    template<class T> std::size_t compare(const std::vector<T>& got,const std::vector<T>& want,const std::string& what,int G){
        std::size_t bad=0;long first=-1;for(std::size_t i=0;i<got.size();++i){bool same;
            if constexpr(std::is_same_v<T,F>)same=reference::equal<Bits>(got[i],want[i]);else same=reference::equal_complex<Bits>(got[i],want[i]);
            if(!same){if(first<0)first=long(i);++bad;}}
        checks+=got.size();if(bad)std::cerr<<Bits<<" bits G="<<G<<' '<<what<<": "<<bad<<'/'<<got.size()<<" mismatches, first case "<<first<<'\n';
        return bad;
    }
    void validate(int G){
        std::vector<F> a,b;real_cases(a,b);std::size_t n=a.size();std::vector<F> out(n),want(n);
        auto ba=buffer(a.data(),n*sizeof(F)),bb=buffer(b.data(),n*sizeof(F)),bo=buffer(nullptr,n*sizeof(F));
        const char* names[]={"add","sub","mul"};
        for(std::uint32_t op=0;op<3;++op){for(std::size_t i=0;i<n;++i)want[i]=op==0?add(a[i],b[i]):op==1?sub(a[i],b[i]):mul(a[i],b[i]);
            std::memset(bo.contents,0xa5,bo.length);run(lib.get("coop_real"+std::to_string(G)),ba,bb,bo,Params{std::uint32_t(n),op,0,0,1},n*G,opt.tg);
            std::memcpy(out.data(),bo.contents,n*sizeof(F));real_bad+=compare(out,want,names[op],G);}
        std::size_t m=opt.count;std::vector<C> x(m),y(m),z(m),zw(m);
        for(std::size_t i=0;i<m;++i){x[i]={rnd(5),rnd(5)};y[i]={rnd(5),rnd(5)};}
        for(std::size_t i=0;i<16;++i){y[i]={negate(x[i].re),x[i].im};}
        for(std::size_t i=16;i<m/2;++i)for(F* v:{&x[i].re,&x[i].im,&y[i].re,&y[i].im}){for(int j=0;j<N;++j)v->limb[j]=(rng()&1)?~0u:std::uint32_t(rng()%3);v->limb[N-1]|=0x80000000u;}
        auto bx=buffer(x.data(),m*sizeof(C)),by=buffer(y.data(),m*sizeof(C)),bz=buffer(nullptr,m*sizeof(C));
        for(std::uint32_t op:{4u,5u}){for(std::size_t i=0;i<m;++i)zw[i]=op==4?cadd(x[i],y[i]):cmul(x[i],y[i]);
            std::memset(bz.contents,0xa5,bz.length);run(lib.get("coop_complex"+std::to_string(G)),bx,by,bz,Params{std::uint32_t(m),op,0,0,1},m*G,opt.tg);
            std::memcpy(z.data(),bz.contents,m*sizeof(C));complex_bad+=compare(z,zw,op==4?"cadd":"cmul",G);}
        // Recurrences: tests/test_arithmetic.cpp data (per-lane weights), replicated shared weights, and a longer batch.
        for(auto [batch,steps,spw]:{std::tuple<unsigned,unsigned,unsigned>{17,31,1},{68,31,4},{256,64,1},{96,200,96}}){
            std::vector<C> seed(4*batch),weight(std::size_t(4)*(batch/spw)*steps),cpu(batch),gpu(batch),co(batch);
            for(auto& v:seed)v={rnd(5),rnd(5)};
            for(auto& v:weight){v={rnd(3),rnd(3)};v.re.exponent-=5;v.im.exponent-=5;}
            if(spw==4)for(unsigned j=0;j<4;++j)for(unsigned i=0;i<batch;++i)seed[j*batch+i]=seed[j*batch+i/4*4];
            unsigned wc=batch/spw,cpu_lanes=batch<=68?batch:12;
            for(unsigned i=0;i<cpu_lanes;++i){C ring[4];for(int j=0;j<4;++j)ring[j]=seed[j*batch+i];int head=0;
                for(unsigned t=0;t<steps;++t){C sum={zero<N>(),zero<N>()};for(int j=0;j<4;++j)sum=cadd(sum,cmul(weight[(t*4+j)*wc+i/spw],ring[(head+j)%4]));ring[head]=sum;head=(head+1)%4;}
                cpu[i]=ring[(head+3)%4];}
            auto bs=buffer(seed.data(),seed.size()*sizeof(C)),bw=buffer(weight.data(),weight.size()*sizeof(C)),bo1=buffer(nullptr,batch*sizeof(C)),bo2=buffer(nullptr,batch*sizeof(C));
            Params p{batch,0,steps,wc,spw};run(lib.get("recurrence"),bs,bw,bo1,p,batch,32);std::memcpy(gpu.data(),bo1.contents,batch*sizeof(C));
            std::memset(bo2.contents,0xa5,bo2.length);run(lib.get("recurrence_coop"+std::to_string(G)),bs,bw,bo2,p,std::size_t(batch)*G,opt.tg);std::memcpy(co.data(),bo2.contents,batch*sizeof(C));
            std::vector<C> gpu_head(gpu.begin(),gpu.begin()+cpu_lanes),co_head(co.begin(),co.begin()+cpu_lanes),cpu_head(cpu.begin(),cpu.begin()+cpu_lanes);
            std::string tag="recurrence "+std::to_string(batch)+"x"+std::to_string(steps)+" spw="+std::to_string(spw);
            check(compare(gpu_head,cpu_head,tag+" existing-vs-cpu",1)==0,"existing recurrence kernel disagrees with the core");
            recurrence_bad+=compare(co_head,cpu_head,tag+" coop-vs-cpu",G)+compare(co,gpu,tag+" coop-vs-existing",G);
        }
        std::cout<<"# "<<Bits<<" bits G="<<G<<" L="<<(N+G-1)/G<<": real "<<real_bad<<", complex "<<complex_bad<<", recurrence "<<recurrence_bad<<" mismatches (cumulative)"<<std::endl;
    }
    // Interleaved A/B: each sample round runs every variant once, in rotated order; median device time per step.
    void time(){
        std::vector<std::pair<std::string,unsigned>> variants={{"recurrence",1}};for(auto G:opt.groups)variants.push_back({"recurrence_coop"+std::to_string(G),G});
        for(unsigned n:opt.lanes){
            std::vector<C> seed(4*n),weight(4*opt.steps);for(auto& v:seed)v={rnd(5),rnd(5)};
            for(auto& v:weight){v={rnd(3),rnd(3)};v.re.exponent-=5;v.im.exponent-=5;}
            auto bs=buffer(seed.data(),seed.size()*sizeof(C)),bw=buffer(weight.data(),weight.size()*sizeof(C));
            std::vector<id<MTLBuffer>> outs;for(std::size_t v=0;v<variants.size();++v)outs.push_back(buffer(nullptr,n*sizeof(C)));
            Params p{n,0,opt.steps,1,n};std::vector<std::vector<double>> t(variants.size());
            // Every cooperative dispatch, warm-up included, is compared with the existing kernel's output.
            std::vector<std::size_t> bad(variants.size()),dispatches(variants.size());
            auto once=[&](std::size_t v){double t=run(lib.get(variants[v].first),bs,bw,outs[v],p,std::size_t(n)*variants[v].second,variants[v].second==1?32:opt.tg);
                if(v){++dispatches[v];bad[v]+=std::memcmp(outs[v].contents,outs[0].contents,n*sizeof(C))!=0;}return t;};
            once(0);
            for(auto start=Clock::now();std::chrono::duration<double>(Clock::now()-start).count()<opt.warm;)for(std::size_t v=0;v<variants.size();++v)once(v);
            for(int s=0;s<opt.samples;++s)for(std::size_t k=0;k<variants.size();++k){std::size_t v=(k+s)%variants.size();t[v].push_back(once(v)/opt.steps);}
            for(std::size_t v=1;v<variants.size();++v){std::cerr<<"# "<<Bits<<" bits "<<n<<" lanes G="<<variants[v].second<<": "<<bad[v]<<'/'<<dispatches[v]<<" dispatches differ\n";
                check(!bad[v]||opt.keep_going,"timed coop output differs from existing kernel");}
            double base=median(t[0]);
            for(std::size_t v=0;v<variants.size();++v){double m=median(t[v]);
                std::cout<<Bits<<','<<n<<','<<(v?"coop":"thread")<<','<<variants[v].second<<','<<(v?(N+variants[v].second-1)/variants[v].second:N)<<','
                    <<std::size_t(n)*variants[v].second<<','<<opt.steps<<','<<opt.samples<<','<<m*1e6<<','<<*std::min_element(t[v].begin(),t[v].end())*1e6<<','<<base/m<<std::endl;}
        }
    }
};
template<int N> void width(bool full){@autoreleasepool{
    bool listed=std::find(opt.bits.begin(),opt.bits.end(),32*N)!=opt.bits.end();
    if(listed||full){Width<N> w;if(opt.check){for(auto G:opt.groups)w.validate(int(G));
            bool good=!w.real_bad&&!w.complex_bad&&!w.recurrence_bad;opt.failed+=!good;check(good||opt.keep_going,"cooperative results differ at "+std::to_string(32*N)+" bits");}
        if(opt.time&&listed)w.time();}
    if constexpr(N<32)width<N+1>(full);
}}
std::vector<unsigned> list(const std::string& s){std::vector<unsigned> r;std::stringstream in(s);std::string x;while(std::getline(in,x,','))r.push_back(unsigned(std::stoul(x)));return r;}
int main(int argc,char** argv){try{
    bool full=false;
    for(int i=1;i<argc;++i){std::string arg=argv[i];auto next=[&]{check(i+1<argc,"missing value for "+arg);return std::string(argv[++i]);};
        if(arg=="--bits"){opt.bits.clear();for(auto b:list(next()))opt.bits.push_back(int(b));}else if(arg=="--lanes")opt.lanes=list(next());else if(arg=="--groups")opt.groups=list(next());
        else if(arg=="--steps")opt.steps=unsigned(std::stoul(next()));else if(arg=="--samples")opt.samples=std::stoi(next());else if(arg=="--warm")opt.warm=std::stod(next());
        else if(arg=="--threadgroup")opt.tg=unsigned(std::stoul(next()));else if(arg=="--count")opt.count=std::stoull(next());
        else if(arg=="--all-bits")full=true;else if(arg=="--no-time")opt.time=false;else if(arg=="--no-check")opt.check=false;else if(arg=="--keep-going")opt.keep_going=true;
        else throw std::invalid_argument("usage: coop_recurrence [--bits 256,384,1024] [--all-bits] [--groups 4,8,16,32] [--lanes 32,128,...] [--steps S] [--samples K] [--warm SECONDS] [--threadgroup T] [--count N] [--no-time] [--no-check]");}
    for(auto G:opt.groups)check(G==4||G==8||G==16||G==32,"groups must be 4, 8, 16 or 32");
    check(opt.tg>=32&&opt.tg%32==0&&opt.samples>0&&opt.steps>0,"invalid options");
    device=MTLCreateSystemDefaultDevice();check(device!=nil,"no Metal GPU");queue=[device newCommandQueue];
    std::cerr<<[device.name UTF8String]<<"; cooperative recurrence experiment (L4); every cooperative result compared bit for bit\n";
    std::cout<<std::setprecision(6)<<"bits,lanes,kernel,G,limbs_per_lane,threads,steps,samples,median_us_per_step,min_us_per_step,speedup_vs_thread\n";
    width<2>(full);if(opt.failed)std::cerr<<opt.failed<<" precision(s) with cooperative mismatches\n";return opt.failed?1:0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
