// Cooperative recurrence (round 17-L4, docs/experiments.md): each trajectory's numbers are spread over
// G SIMD lanes (limbs(G) limbs per lane). Bit-identical to the one-thread recurrence kernel; selected by
// Engine::recurrence for small trajectory counts. Only G = 16 and 32 are compiled: G = 4/8 gave
// nondeterministic mismatches under MTL_SHADER_VALIDATION (never without it) and stay experimental.
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
#define COOP_RECURRENCE(G) \
kernel void recurrence_coop##G(device const Complex<N>* s [[buffer(0)]],device const Complex<N>* w [[buffer(1)]],device Complex<N>* out [[buffer(2)]],constant Params& p [[buffer(3)]],\
    uint gid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){coop::recurrence_body<G>(s,w,out,p,gid,lane);}
COOP_RECURRENCE(16)
COOP_RECURRENCE(32)
