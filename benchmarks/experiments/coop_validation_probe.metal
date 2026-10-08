// Round 41-L4b probe (benchmarks/experiments/coop_validation_probe.mm): a copy of the round-17/22 cooperative
// arithmetic (src/cooperative.metal) with switches that replace or fence its SIMD operations, so that the
// MTL_SHADER_VALIDATION discrepancy of G = 4/8 can be bisected. Namespace coopx avoids the library's coop.
// Switches (preprocessor macros, default 0):
//   PROBE_TG       shuffles, ballots and simd_any/simd_all through threadgroup memory and threadgroup barriers
//                  (no SIMD intrinsic left; requires threadgroup size 32 and implies PROBE_NO_RETURN)
//   PROBE_TG=2     the same with simdgroup_barrier(mem_threadgroup); PROBE_TG=3: only the shuffles go through
//                  threadgroup memory (simdgroup barriers), ballots and simd_any/simd_all stay SIMD intrinsics
//   PROBE_BARRIER  simdgroup_barrier(mem_none) before every SIMD operation
//   PROBE_NO_RETURN no early return: lanes past the last trajectory recompute the last one and skip the store
//   PROBE_ANY_BALLOT simd_any/simd_all computed from a full-SIMD ballot
//   PROBE_TRACE    store q3 after every step (buffer 4) to locate the first wrong step
//   PROBE_SHORT_CIRCUIT the round 17-22 rounding vote `vote(rb)!=0&&vote(sticky)!=0`: its second ballot runs in
//                  divergent control flow (only lane groups with a round bit), the defect found in round 41-L4b
//   PROBE_CHECK_ACTIVE count SIMD operations executed with fewer than 32 active lanes, per source line (buffer 5)
//   PROBE_PRESSURE=K carry K extra live words through the step loop (more registers / spilling, same results)
//   PROBE_CHECK_LANE count threads with lane != gid%32 etc.; PROBE_SIMD_INDEX index items by SIMD-group identity
//   PROBE_CHECK_LOAD re-read every weight limb with an atomic load and count differences (recurrence only)
#ifndef PROBE_TG
#define PROBE_TG 0
#endif
#ifndef PROBE_BARRIER
#define PROBE_BARRIER 0
#endif
#ifndef PROBE_NO_RETURN
#define PROBE_NO_RETURN PROBE_TG
#endif
#ifndef PROBE_ANY_BALLOT
#define PROBE_ANY_BALLOT 0
#endif
#ifndef PROBE_TRACE
#define PROBE_TRACE 0
#endif
#ifndef PROBE_SHORT_CIRCUIT
#define PROBE_SHORT_CIRCUIT 0
#endif
#ifndef PROBE_CHECK_ACTIVE
#define PROBE_CHECK_ACTIVE 0
#endif
#ifndef PROBE_PRESSURE
#define PROBE_PRESSURE 0
#endif
#ifndef PROBE_CHECK_LANE
#define PROBE_CHECK_LANE 0
#endif
#ifndef PROBE_SIMD_INDEX
#define PROBE_SIMD_INDEX 0
#endif
namespace coopx {
constexpr int limbs(int g){return (N+g-1)/g;}
template<int L> struct CN { uint l[L]; int exponent,sign; uint status; };
template<int L> struct CW { uint lo[L],hi[L]; };
template<int L> struct CC { CN<L> re,im; };
struct Ctx { uint k,base,lane;
#if PROBE_TG
    threadgroup uint* s;
#endif
    device atomic_uint* diag;
};
inline void fence(Ctx c,uint site){
#if PROBE_BARRIER
    simdgroup_barrier(mem_flags::mem_none);
#endif
#if PROBE_CHECK_ACTIVE
    // Every SIMD operation must run with all 32 lanes active (full SIMD groups, uniform control flow).
    if(uint(ulong(static_cast<simd_vote::vote_t>(simd_ballot(true))))!=0xffffffffu)atomic_fetch_add_explicit(c.diag+site,1u,memory_order_relaxed);
#endif
}
#if PROBE_TG
#if PROBE_TG==1
#define PROBE_XBAR threadgroup_barrier(mem_flags::mem_threadgroup)
#else
#define PROBE_XBAR simdgroup_barrier(mem_flags::mem_threadgroup)
#endif
inline uint xchg(uint x,uint lane,Ctx c){PROBE_XBAR;c.s[c.lane]=x;PROBE_XBAR;return c.s[lane];}
inline uint shfl_(uint x,uint lane,Ctx c,uint){return xchg(x,lane,c);}
inline int shfl_(int x,uint lane,Ctx c,uint){return int(xchg(uint(x),lane,c));}
#endif
#if PROBE_TG==1 || PROBE_TG==2
template<int G> inline ulong vote_(bool b,Ctx c,uint){xchg(b?1u:0u,0,c);ulong v=0;for(int j=0;j<G;++j)v|=ulong(c.s[c.base+j]&1)<<j;return v;}
inline bool any_(bool b,Ctx c,uint){xchg(b?1u:0u,0,c);uint r=0;for(int j=0;j<32;++j)r|=c.s[j];return r!=0;}
inline bool all_(bool b,Ctx c,uint site){return !any_(!b,c,site);}
#else
#if !PROBE_TG
inline uint shfl_(uint x,uint lane,Ctx c,uint site){fence(c,site);return simd_shuffle(x,ushort(lane));}
inline int shfl_(int x,uint lane,Ctx c,uint site){fence(c,site);return simd_shuffle(x,ushort(lane));}
#endif
template<int G> inline ulong vote_(bool b,Ctx c,uint site){fence(c,site);ulong v=ulong(static_cast<simd_vote::vote_t>(simd_ballot(b)));return G==32?v&0xffffffffUL:(v>>c.base)&((ulong(1)<<G)-1);}
#if PROBE_ANY_BALLOT
inline bool any_(bool b,Ctx c,uint site){fence(c,site);return ulong(static_cast<simd_vote::vote_t>(simd_ballot(b)))!=0;}
inline bool all_(bool b,Ctx c,uint site){fence(c,site);return ulong(static_cast<simd_vote::vote_t>(simd_ballot(!b)))==0;}
#else
inline bool any_(bool b,Ctx c,uint site){fence(c,site);return simd_any(b);}
inline bool all_(bool b,Ctx c,uint site){fence(c,site);return simd_all(b);}
#endif
#endif
#define shfl(x,lane) shfl_(x,lane,c,__LINE__)
#define vote(b) vote_<G>(b,c,__LINE__)
#define ANY(b) any_(b,c,__LINE__)
#define ALL(b) all_(b,c,__LINE__)
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
    CW<L> w;uint hl=0,hh=0;ulong cl=0,ch=0;
    for(int j=0;j<=L;++j){uint pl=shfl(la[L+j],src),ph=shfl(ha[L+j],src),il=first?0u:pl,ih=first?pl:ph;
        if(j<L){ulong v=ulong(la[j])+il+cl;w.lo[j]=uint(v);cl=v>>32;v=ulong(ha[j])+ih+ch;w.hi[j]=uint(v);ch=v>>32;}
        else{hl=il+uint(cl);hh=ih+uint(ch);}}
    uint pl=shfl(hl,src),ph=shfl(hh,src),el=add_small(w.lo,first?0u:pl),eh=add_small(w.hi,first?pl:ph);
    ulong C=carries(vote(el!=0)|(vote(eh!=0)<<G),vote(all_ones(w.lo))|(vote(all_ones(w.hi))<<G),0);
    add_small(w.lo,uint(C>>c.k)&1);add_small(w.hi,uint(C>>(c.k+G))&1);
    return w;
}
template<int G> inline void shift_left_small(thread CW<limbs(G)>& w,uint s,Ctx c){
    constexpr int L=limbs(G);uint src=c.base+((c.k-1)&(G-1)),xl=shfl(w.lo[L-1],src),xh=shfl(w.hi[L-1],src),pl=c.k?xl:0u,ph=c.k?xh:xl;
    for(int j=L-1;j>=0;--j){w.lo[j]=funnel_left(w.lo[j],j?w.lo[j-1]:pl,s);w.hi[j]=funnel_left(w.hi[j],j?w.hi[j-1]:ph,s);}
}
template<int G> inline void shift_left(thread CW<limbs(G)>& w,uint sh,Ctx c){
    constexpr int L=limbs(G);uint wl=sh/32,s=sh%32,wb=wl/L,wr=wl%L;
    if(ANY(wb!=0)){uint src=c.base+((c.k-wb)&(G-1));bool in=c.k>=wb,in2=c.k+G>=wb;
        for(int j=0;j<L;++j){uint xl=shfl(w.lo[j],src),xh=shfl(w.hi[j],src);w.lo[j]=in?xl:0u;w.hi[j]=in?xh:in2?xl:0u;}}
    if(ANY(wr!=0||s!=0)){uint src=c.base+((c.k-1)&(G-1)),cl[2*L],ch[2*L];bool first=c.k==0;
        for(int j=0;j<L;++j){uint xl=shfl(w.lo[j],src),xh=shfl(w.hi[j],src);cl[j]=first?0u:xl;ch[j]=first?xl:xh;cl[L+j]=w.lo[j];ch[L+j]=w.hi[j];}
        for(int j=0;j<L;++j){uint l0=0,l1=0,h0=0,h1=0;
            for(int r=0;r<L;++r)if(uint(r)==wr){l0=cl[L+j-r];l1=cl[L+j-r-1];h0=ch[L+j-r];h1=ch[L+j-r-1];}
            w.lo[j]=funnel_left(l0,l1,s);w.hi[j]=funnel_left(h0,h1,s);}}
}
template<int G> inline bool shift_right(thread CW<limbs(G)>& w,uint sh,Ctx c){
    constexpr int L=limbs(G),M=G*L;uint lost=0;
    for(int j=0;j<L;++j){uint q=32*(c.k*L+j);lost|=lost_bits(w.lo[j],q,sh)|lost_bits(w.hi[j],q+32*M,sh);}
    bool sticky=vote(lost!=0)!=0;uint wl=sh/32,s=sh%32,wb=wl/L,wr=wl%L;
    if(ANY(wb!=0)){uint src=c.base+((c.k+wb)&(G-1)),t=c.k+wb;
        for(int j=0;j<L;++j){uint xl=shfl(w.lo[j],src),xh=shfl(w.hi[j],src);w.lo[j]=t<G?xl:t<2*G?xh:0u;w.hi[j]=t<G?xh:0u;}}
    if(ANY(wr!=0||s!=0)){uint src=c.base+((c.k+1)&(G-1)),cl[2*L],ch[2*L];bool last=c.k==G-1;
        for(int j=0;j<L;++j){uint xl=shfl(w.lo[j],src),xh=shfl(w.hi[j],src);cl[j]=w.lo[j];ch[j]=w.hi[j];cl[L+j]=last?xh:xl;ch[L+j]=last?0u:xh;}
        for(int j=0;j<L;++j){uint l0=0,l1=0,h0=0,h1=0;
            for(int r=0;r<L;++r)if(uint(r)==wr){l0=cl[j+r];l1=cl[j+r+1];h0=ch[j+r];h1=ch[j+r+1];}
            w.lo[j]=funnel_right(l1,l0,s);w.hi[j]=funnel_right(h1,h0,s);}}
    return sticky;
}
template<int G> inline CN<limbs(G)> round(thread const CW<limbs(G)>& w,long e,int sign,uint status,Ctx c){
    constexpr int L=limbs(G),M=G*L,P=M-N,R=2*M-N-1,BR=R/L,PR=R%L,KQ=P/L,PQ=P%L;
    bool range=e>1000000000||e< -1000000001;uint below=0;bool rb;
    for(int j=0;j<L;++j){int q=int(c.k)*L+j,q2=q+M;below|=q<R?w.lo[j]:q==R?w.lo[j]&0x7fffffffu:0u;below|=q2<R?w.hi[j]:q2==R?w.hi[j]&0x7fffffffu:0u;}
    if(BR<G)rb=c.k==uint(BR)&&(w.lo[PR]>>31);else rb=c.k==uint(BR-G)&&(w.hi[PR]>>31);
    bool lsb=c.k==uint(KQ)&&(w.hi[PQ]&1);
#if PROBE_SHORT_CIRCUIT
    bool up=vote(rb)!=0&&vote(below!=0||lsb)!=0;   // round 17-22 form: the second ballot runs only in groups with a round bit
#else
    ulong round_vote=vote(rb),sticky_vote=vote(below!=0||lsb);bool up=round_vote!=0&&sticky_vote!=0;
#endif
    CN<L> r;r.sign=sign;r.status=status;
    for(int j=0;j<L;++j)r.l[j]=int(c.k)*L+j>=P?w.hi[j]:0u;
    if(ANY(up)){uint cc=up&&c.k==uint(KQ)?1u:0u;bool prop=c.k>uint(KQ)&&all_ones(r.l);
        for(int j=PQ;j<L;++j){ulong v=ulong(r.l[j])+cc;r.l[j]=uint(v);cc=uint(v>>32);}
        ulong C=carries(vote(cc!=0),vote(prop),0);add_small(r.l,uint(C>>c.k)&1);
        if((C>>G)&1){for(int j=0;j<L;++j)r.l[j]=0;if(c.k==G-1)r.l[L-1]=0x80000000u;++e;}}
    r.exponent=int(e);
    return range||e>1000000000||e< -1000000000?czero<L>(status|exponent_overflow):r;
}
template<int G> inline CN<limbs(G)> pack(thread CW<limbs(G)>& w,long scale,int sign,uint status,Ctx c){
    constexpr int L=limbs(G),M=G*L;
    ulong z=vote(any_set(w.lo))|(vote(any_set(w.hi))<<G);
    uint tb=z?63-uint(clz(z)):0;int hb=shfl(tb>=uint(G)?top_bit(w.hi):top_bit(w.lo),c.base+(tb&(G-1)));
    long h=max(long(tb)*32*L+hb,0L);shift_left<G>(w,uint(64*M-1-h),c);
    CN<L> r=round<G>(w,scale+h,sign,status,c);return z?r:czero<L>(status);
}
template<int L> inline CN<L> pick(bool p,thread const CN<L>& x,thread const CN<L>& y){
    CN<L> r;for(int j=0;j<L;++j)r.l[j]=p?x.l[j]:y.l[j];r.exponent=p?x.exponent:y.exponent;r.sign=p?x.sign:y.sign;r.status=p?x.status:y.status;return r;}
template<int G> inline CN<limbs(G)> mul(thread const CN<limbs(G)>& a,thread const CN<limbs(G)>& b,Ctx c){
    constexpr int L=limbs(G);
    uint status=a.status|b.status;bool special=status||!a.sign||!b.sign;CN<L> r=czero<L>(status);
    if(!ALL(special)){CW<L> w=product<G>(a,b,c);
        uint d=(shfl(w.hi[L-1],c.base+G-1)>>31)^1;shift_left_small<G>(w,d,c);
        CN<L> p=round<G>(w,long(a.exponent)+b.exponent+1-d,a.sign*b.sign,0,c);r=pick(special,r,p);}
    return r;
}
template<int G> inline CN<limbs(G)> add(CN<limbs(G)> a,CN<limbs(G)> b,Ctx c){
    constexpr int L=limbs(G),M=G*L;
    uint status=a.status|b.status;CN<L> early=status?czero<L>(status):!a.sign?b:a;bool special=status||!a.sign||!b.sign;
    int cmp=a.exponent>b.exponent?1:a.exponent<b.exponent?-1:0;
    if(ANY(!cmp)){int lc=0;for(int j=L-1;j>=0;--j)lc=lc?lc:a.l[j]>b.l[j]?1:a.l[j]<b.l[j]?-1:0;
        ulong m=vote(lc!=0);int t=shfl(lc,c.base+(m?63-uint(clz(m)):0));cmp=cmp?cmp:m?t:0;}
    if(cmp<0){CN<L> t=a;a=b;b=t;}
    long gap=long(a.exponent)-b.exponent;if(!special&&gap>32*N+2){special=true;early=a;}
    CN<L> r=early;
    if(!ALL(special)){
        CW<L> x,y;bool last=c.k==G-1;uint nx=shfl(a.l[0],c.base+((c.k+1)&(G-1)));
        for(int j=0;j<L;++j){x.hi[j]=funnel_right(j<L-1?a.l[j<L-1?j+1:0]:last?0u:nx,a.l[j],1);x.lo[j]=0;y.lo[j]=0;y.hi[j]=b.l[j];}
        x.lo[L-1]=last?nx<<31:0u;
        if(shift_right<G>(y,uint(clamp(gap,0L,long(32*N+2)))+1,c)&&!c.k)y.lo[0]|=1;
        bool subtract=a.sign!=b.sign;uint m=subtract?~0u:0u;ulong cl=0,ch=0;
        for(int j=0;j<L;++j){ulong v=ulong(x.lo[j])+(y.lo[j]^m)+cl;x.lo[j]=uint(v);cl=v>>32;v=ulong(x.hi[j])+(y.hi[j]^m)+ch;x.hi[j]=uint(v);ch=v>>32;}
        ulong C=carries(vote(cl!=0)|(vote(ch!=0)<<G),vote(all_ones(x.lo))|(vote(all_ones(x.hi))<<G),subtract?1:0);
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
// OP 0: the library recurrence; 1: q = cmul(w_t, q); 2: q = cadd(q, w_t); 3: real mul chains; 4: real add chains.
template<int G,int OP> inline void body(device const Complex<N>* seeds,device const Complex<N>* weights,device Complex<N>* out,
    device Complex<N>* trace,constant Params& p,uint gid,uint lane,threadgroup uint* tg,device atomic_uint* diag){
    uint i=gid/G;bool active=i<p.count;
#if PROBE_NO_RETURN
    i=active?i:p.count-1;
#else
    if(!active)return;
#endif
    Ctx c={lane&(G-1),lane&~uint(G-1),lane
#if PROBE_TG
        ,tg
#endif
        ,diag};
#if PROBE_TG
    if(!lane)for(int j=0;j<32;++j)tg[j]=0;   // slots of threads past the grid end read as 0
#endif
    CC<limbs(G)> q0=cload<G>(seeds[i],c.k),q1=cload<G>(seeds[p.count+i],c.k),q2=cload<G>(seeds[2*p.count+i],c.k),q3=cload<G>(seeds[3*p.count+i],c.k);
    uint wi=i/p.states_per_weight;
#if PROBE_PRESSURE
    // Extra live state (PROBE_PRESSURE words) carried through the loop: raises register pressure / spilling
    // without changing results (it only decides an impossible extra store).
    uint ballast[PROBE_PRESSURE];for(int j=0;j<PROBE_PRESSURE;++j)ballast[j]=q0.re.l[j%limbs(G)]^uint(j*0x9e3779b9u);
#endif
    for(uint step=0;step<p.steps;++step){
#if PROBE_PRESSURE
        for(int j=0;j<PROBE_PRESSURE;++j)ballast[j]=ballast[j]*1664525u+ballast[(j+1)%PROBE_PRESSURE]+step;
#endif
        if(OP==0){
            CC<limbs(G)> sum={czero<limbs(G)>(),czero<limbs(G)>()};uint base=step*4*p.weight_count+wi;
#if PROBE_CHECK_LOAD
            // Reload every weight limb with a relaxed atomic load and count differences from the value used (diag[1003]).
            CC<limbs(G)> w0=cload<G>(weights[base],c.k),w1=cload<G>(weights[base+p.weight_count],c.k),
                w2=cload<G>(weights[base+2*p.weight_count],c.k),w3=cload<G>(weights[base+3*p.weight_count],c.k);
            {constexpr int L=limbs(G),P=G*L-N;uint bad=0;
             for(int t=0;t<4;++t){thread const CC<L>& w=t==0?w0:t==1?w1:t==2?w2:w3;
                device atomic_uint* a=reinterpret_cast<device atomic_uint*>(const_cast<device Complex<N>*>(weights+base+t*p.weight_count));
                for(int j=0;j<L;++j){int q=int(c.k)*L+j-P;if(q<0)continue;
                    bad+=atomic_load_explicit(a+q,memory_order_relaxed)!=w.re.l[j];
                    bad+=atomic_load_explicit(a+(N+3)+q,memory_order_relaxed)!=w.im.l[j];}}
             if(bad)atomic_fetch_add_explicit(diag+1003,bad,memory_order_relaxed);}
            sum=cadd<G>(sum,cmul<G>(w0,q0,c),c);sum=cadd<G>(sum,cmul<G>(w1,q1,c),c);
            sum=cadd<G>(sum,cmul<G>(w2,q2,c),c);sum=cadd<G>(sum,cmul<G>(w3,q3,c),c);
#else
            sum=cadd<G>(sum,cmul<G>(cload<G>(weights[base],c.k),q0,c),c);
            sum=cadd<G>(sum,cmul<G>(cload<G>(weights[base+p.weight_count],c.k),q1,c),c);
            sum=cadd<G>(sum,cmul<G>(cload<G>(weights[base+2*p.weight_count],c.k),q2,c),c);
            sum=cadd<G>(sum,cmul<G>(cload<G>(weights[base+3*p.weight_count],c.k),q3,c),c);
#endif
            q0=q1;q1=q2;q2=q3;q3=sum;
        }else{
            CC<limbs(G)> w=cload<G>(weights[step*p.weight_count+wi],c.k);
            if(OP==1)q3=cmul<G>(w,q3,c);
            else if(OP==2)q3=cadd<G>(q3,w,c);
            else if(OP==3){q3.re=mul<G>(w.re,q3.re,c);q3.im=mul<G>(w.im,q3.im,c);}
            else{q3.re=add<G>(q3.re,w.re,c);q3.im=add<G>(q3.im,w.im,c);}
        }
#if PROBE_TRACE
        if(active)cstore<G>(trace[step*p.count+i],q3,c.k);
#endif
    }
#if PROBE_PRESSURE
    uint x=0;for(int j=0;j<PROBE_PRESSURE;++j)x^=ballast[j];if(x==0x7a5c3e1fu&&p.operation==0xdeadu)q3.re.status|=x;
#endif
    if(active)cstore<G>(out[i],q3,c.k);
}
} // namespace coopx
// The one-thread recurrence of kernels.metal plus the optional ballast (--groups 1 in the probe).
kernel void probe_thread(device const Complex<N>* seeds [[buffer(0)]],device const Complex<N>* weights [[buffer(1)]],device Complex<N>* out [[buffer(2)]],
    constant Params& p [[buffer(3)]],uint i [[thread_position_in_grid]]){
    if(i>=p.count)return;
    Complex<N> q0=seeds[i],q1=seeds[p.count+i],q2=seeds[2*p.count+i],q3=seeds[3*p.count+i];uint wi=i/p.states_per_weight;
#if PROBE_PRESSURE
    uint ballast[PROBE_PRESSURE];for(int j=0;j<PROBE_PRESSURE;++j)ballast[j]=q0.re.limb[j%N]^uint(j*0x9e3779b9u);
#endif
    for(uint step=0;step<p.steps;++step){
#if PROBE_PRESSURE
        for(int j=0;j<PROBE_PRESSURE;++j)ballast[j]=ballast[j]*1664525u+ballast[(j+1)%PROBE_PRESSURE]+step;
#endif
        Complex<N> sum={zero<N>(),zero<N>()};uint base=step*4*p.weight_count+wi;
        sum=cadd(sum,cmul(weights[base],q0));sum=cadd(sum,cmul(weights[base+p.weight_count],q1));
        sum=cadd(sum,cmul(weights[base+2*p.weight_count],q2));sum=cadd(sum,cmul(weights[base+3*p.weight_count],q3));
        q0=q1;q1=q2;q2=q3;q3=sum;
    }
#if PROBE_PRESSURE
    uint x=0;for(int j=0;j<PROBE_PRESSURE;++j)x^=ballast[j];if(x==0x7a5c3e1fu&&p.operation==0xdeadu)q3.re.status|=x;
#endif
    out[i]=q3;
}
// The cooperative kernels assume lane == gid % 32 (trajectory from gid, limb block from the SIMD lane).
// PROBE_CHECK_LANE counts threads that break it (diag[1000..1002]); PROBE_SIMD_INDEX derives the item index from
// the SIMD-group identity instead, so that lane and trajectory are consistent whatever the mapping.
inline uint probe_index(uint gid,uint lane,uint tid,uint sg,uint tgp,uint sgs,uint tpt,device atomic_uint* diag){
#if PROBE_CHECK_LANE
    if(lane!=(gid&31u))atomic_fetch_add_explicit(diag+1000,1u,memory_order_relaxed);
    if(lane!=(tid&31u))atomic_fetch_add_explicit(diag+1001,1u,memory_order_relaxed);
    if(sg!=tid/32u)atomic_fetch_add_explicit(diag+1002,1u,memory_order_relaxed);
#endif
#if PROBE_SIMD_INDEX
    return tgp*tpt+sg*32u+lane;
#else
    return gid;
#endif
}
#define PROBE_KERNEL(G,OP) \
kernel void probe_g##G##_op##OP(device const Complex<N>* s [[buffer(0)]],device const Complex<N>* w [[buffer(1)]],device Complex<N>* out [[buffer(2)]],\
    constant Params& p [[buffer(3)]],device Complex<N>* trace [[buffer(4)]],device atomic_uint* diag [[buffer(5)]],uint gid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]],\
    uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint tgp [[threadgroup_position_in_grid]],uint sgs [[simdgroups_per_threadgroup]],\
    uint tpt [[threads_per_threadgroup]]){\
    PROBE_TG_DECL coopx::body<G,OP>(s,w,out,trace,p,probe_index(gid,lane,tid,sg,tgp,sgs,tpt,diag),lane,PROBE_TG_PTR,diag);}
#if PROBE_TG
#define PROBE_TG_DECL threadgroup uint scratch[32];
#define PROBE_TG_PTR scratch
#else
#define PROBE_TG_DECL
#define PROBE_TG_PTR nullptr
#endif
#define PROBE_KERNELS(G) PROBE_KERNEL(G,0) PROBE_KERNEL(G,1) PROBE_KERNEL(G,2) PROBE_KERNEL(G,3) PROBE_KERNEL(G,4)
PROBE_KERNELS(4)
PROBE_KERNELS(8)
PROBE_KERNELS(16)
PROBE_KERNELS(32)
#undef shfl
#undef vote
#undef ANY
#undef ALL
