// Dense products with one rounding per output (src/linalg.mm, docs/numerics.md "Dense products").
// Compiled with MSL 4.0 per (LF_BITS, LF_TERM_WORDS, LF_ACC_WORDS, LF_MAX_MODULI, LF_GROUP).
using namespace metal;
using namespace mpp::tensor_ops;
using namespace limbforge;
constant int N=LF_BITS/32;
// RN(x + c) for an exact x and a number c without status, with one rounding: the two-term exact sum of core.hpp (fma),
// where a term more than one bit below the other's extended window only contributes its sign. Runtime-indexed widths
// stay >= 33 words from 13 words (docs/gpu-codegen.md rule 1). The sum workspace is a multiple of 4 words: some odd
// widths (53, 59, 61, 67, 71, 81 words) gave wrong GPU results for this sum while the CPU was exact (gpu-codegen.md 7).
constant int ADDEND_WORDS=scratch(N);
constexpr int sum_words(int w){return (scratch(w)+3)/4*4;}
template<int W> inline Number<N> round_sum(thread const Exact<W>& x,Number<N> c){
    Exact<ADDEND_WORDS> y=exact_number<ADDEND_WORDS>(c);
    return round_exact<N>(exact_add<N,sum_words(exact_words(N,W,ADDEND_WORDS))>(x,y));
}
struct Params { uint rows,cols,K,Kp,Mp,Np,j,L,Wc,tri,lower,fold,ls_line,ls_k,rs_line,rs_k,out_cols,multi_rows,multi_cols,both,j0,group,upper,row0,col0,out_stride; };
// Per line (row of the left / column of the right operand): bands (0 = handled on the host), lowest band
// exponent, accumulator slot (>= 0: index among multi-band lines; < 0: -1 - index among single-band lines).
struct Line { int bands,low,slot; };
// Member r of one band: line index and the band's exponent range [lo, hi].
struct Member { int line,lo,hi; };
// x mod m for x < 2^32, m < 2^16 not a power of two, c = floor(2^32/m): the quotient estimate is short by at most one.
inline uint reduce32(uint x,uint m,uint c){uint y=x-mulhi(x,c)*m;return y>=m?y-m:y;}
// Entry x in band [lo,hi] is the exact integer X = mantissa * 2^(e-lo) < 2^(bits+hi-lo); value = X * 2^(lo-bits+1).
// For LF_GROUP consecutive moduli its balanced residue v in (-m/2, m/2] is split into int8 digits v = 256*d1 + d0
// (m <= 65279 keeps |d1| <= 127). The entry is read once per group.
struct Digits { char d1[LF_GROUP],d0[LF_GROUP]; };
inline Digits digits(device const Number<N>& x,int lo,int hi,device const uint* moduli,device const uint* recips,uint j0){
    Digits d;for(int q=0;q<LF_GROUP;++q){d.d1[q]=0;d.d0[q]=0;}
    if(x.status||x.sign==0||x.exponent<lo||x.exponent>hi)return d;
    uint m[LF_GROUP],c[LF_GROUP],r[LF_GROUP];for(int q=0;q<LF_GROUP;++q){m[q]=moduli[j0+q];c[q]=recips[j0+q];r[q]=0;}
    for(int w=N-1;w>=0;--w){uint v=x.limb[w];
        for(int q=0;q<LF_GROUP;++q){r[q]=reduce32((r[q]<<16)|(v>>16),m[q],c[q]);r[q]=reduce32((r[q]<<16)|(v&0xffffu),m[q],c[q]);}}
    for(int s=x.exponent-lo;s>0;s-=16){int t=min(s,16);for(int q=0;q<LF_GROUP;++q)r[q]=reduce32(r[q]<<t,m[q],c[q]);}
    for(int q=0;q<LF_GROUP;++q){int v=int(r[q]),mq=int(m[q]);if(x.sign<0)v=v?mq-v:0;if(v>mq/2)v-=mq;
        int d0=((v+128)&255)-128;d.d1[q]=char((v-d0)/256);d.d0[q]=char(d0);}
    return d;
}
// Left band, one plane per modulus of the batch (p.j0 .. p.j0+p.group-1; grid z = group of LF_GROUP moduli, each input entry
// read once per group): cat = [d1 | d0] (Mp x 2Kp), d1, d0 (Mp x Kp); padding is written as zero.
// With p.both (SYRK) the right band of the same lines is written too: rcat = [d0 ; d1] (2Kp x Np), Np = Mp.
kernel void digits_left(device const Number<N>* src [[buffer(0)]],device const Member* band [[buffer(1)]],device int8_t* cat [[buffer(2)]],
                        device int8_t* d1 [[buffer(3)]],device int8_t* d0 [[buffer(4)]],constant Params& p [[buffer(5)]],
                        device int8_t* rcat [[buffer(6)]],device const uint* moduli [[buffer(11)]],device const uint* recips [[buffer(12)]],
                        uint3 g [[thread_position_in_grid]]){
    uint k=g.x,r=g.y,q0=g.z*LF_GROUP;if(k>=p.Kp||r>=p.Mp||q0>=p.group)return;
    Digits d;for(int q=0;q<LF_GROUP;++q){d.d1[q]=0;d.d0[q]=0;}
    if(k<p.K&&r<p.rows){Member b=band[r];d=digits(src[ulong(b.line)*p.ls_line+ulong(k)*p.ls_k],b.lo,b.hi,moduli,recips,p.j0+q0);}
    ulong row=r,cp=ulong(p.Mp)*2*p.Kp,hp=ulong(p.Mp)*p.Kp,rp=ulong(2)*p.Kp*p.Np;uint count=min(uint(LF_GROUP),p.group-q0);
    for(uint q=0;q<count;++q){ulong z=q0+q;
        cat[z*cp+row*2*p.Kp+k]=d.d1[q];cat[z*cp+row*2*p.Kp+p.Kp+k]=d.d0[q];d1[z*hp+row*p.Kp+k]=d.d1[q];d0[z*hp+row*p.Kp+k]=d.d0[q];
        if(p.both){rcat[z*rp+ulong(k)*p.Np+r]=d.d0[q];rcat[z*rp+ulong(p.Kp+k)*p.Np+r]=d.d1[q];}
    }
}
// Right band: rcat = [d0 ; d1] (2Kp x Np) per modulus of the batch.
kernel void digits_right(device const Number<N>* src [[buffer(0)]],device const Member* band [[buffer(1)]],device int8_t* cat [[buffer(2)]],
                         constant Params& p [[buffer(5)]],device const uint* moduli [[buffer(11)]],device const uint* recips [[buffer(12)]],
                         uint3 g [[thread_position_in_grid]]){
    uint c=g.x,k=g.y,q0=g.z*LF_GROUP;if(c>=p.Np||k>=p.Kp||q0>=p.group)return;
    Digits d;for(int q=0;q<LF_GROUP;++q){d.d1[q]=0;d.d0[q]=0;}
    if(k<p.K&&c<p.cols){Member b=band[c];d=digits(src[ulong(b.line)*p.rs_line+ulong(k)*p.rs_k],b.lo,b.hi,moduli,recips,p.j0+q0);}
    ulong rp=ulong(2)*p.Kp*p.Np;uint count=min(uint(LF_GROUP),p.group-q0);
    for(uint q=0;q<count;++q){ulong z=q0+q;cat[z*rp+ulong(k)*p.Np+c]=d.d0[q];cat[z*rp+ulong(p.Kp+k)*p.Np+c]=d.d1[q];}
}
inline uint reduce(int g,uint m){int r=g%int(m);return uint(r<0?r+int(m):r);}
// The three int8 products of one 64 x 32 tile for modulus j = p.j + z (grid z; digit planes of batch member z), exact in int32
// (|sum| <= 2 Kp 2^14 < 2^31), kept in registers (cooperative tensors of one op, hence one layout), and residue_j =
// (65536*G11 + 256*Gmid + G00) mod m written directly: no int32 planes in memory (round 40; before, three products and a
// combine pass per modulus). st: element strides of the lcat, a1/a0 and rcat planes. Skips tiles above the diagonal (p.tri).
kernel void product_residue(device int8_t* lcat [[buffer(0)]],device int8_t* a1 [[buffer(1)]],device int8_t* a0 [[buffer(2)]],device int8_t* rcat [[buffer(3)]],
                            device ushort* res [[buffer(4)]],constant Params& p [[buffer(5)]],constant uint4& st [[buffer(6)]],
                            device const uint* moduli [[buffer(11)]],device const uint* recips [[buffer(12)]],uint3 tgid [[threadgroup_position_in_grid]]){
    if(p.tri&&int(tgid.y)*64+63<int(tgid.x)*32)return;
    const int M=int(p.Mp),Nn=int(p.Np),K=int(p.Kp);const ulong z=tgid.z;
    lcat+=z*st.x;a1+=z*st.y;a0+=z*st.y;rcat+=z*st.z;
    auto tL=tensor<device int8_t,dextents<int32_t,2>,tensor_inline>(lcat,dextents<int32_t,2>(2*K,M));
    auto t1=tensor<device int8_t,dextents<int32_t,2>,tensor_inline>(a1,dextents<int32_t,2>(K,M));
    auto t0=tensor<device int8_t,dextents<int32_t,2>,tensor_inline>(a0,dextents<int32_t,2>(K,M));
    auto tR=tensor<device int8_t,dextents<int32_t,2>,tensor_inline>(rcat,dextents<int32_t,2>(Nn,2*K));
    auto tRlo=tensor<device int8_t,dextents<int32_t,2>,tensor_inline>(rcat,dextents<int32_t,2>(Nn,K));
    auto tRhi=tensor<device int8_t,dextents<int32_t,2>,tensor_inline>(rcat+ulong(K)*Nn,dextents<int32_t,2>(Nn,K));
    constexpr auto d=matmul2d_descriptor(64,32,static_cast<int>(dynamic_extent),false,false,false);
    matmul2d<d,execution_simdgroups<4>> op;
    const int r0=int(tgid.y)*64,c0=int(tgid.x)*32;
    auto mL=tL.slice(0,r0);auto m1=t1.slice(0,r0);auto m0=t0.slice(0,r0);auto mR=tR.slice(c0,0);auto mRlo=tRlo.slice(c0,0);auto mRhi=tRhi.slice(c0,0);
    auto c11=op.get_destination_cooperative_tensor<decltype(m1),decltype(mRhi),int32_t>();op.run(m1,mRhi,c11);
    auto c00=op.get_destination_cooperative_tensor<decltype(m0),decltype(mRlo),int32_t>();op.run(m0,mRlo,c00);
    auto cmid=op.get_destination_cooperative_tensor<decltype(mL),decltype(mR),int32_t>();op.run(mL,mR,cmid);
    const uint j=p.j+tgid.z,m=moduli[j],cm=recips[j];device ushort* out=res+ulong(j)*ulong(M)*ulong(Nn);
    for(uint16_t i=0;i<c11.get_capacity();++i){if(!c11.is_valid_element(i))continue;
        auto idx=c11.get_multidimensional_index(i);const uint c=uint(c0+int(idx[0])),r=uint(r0+int(idx[1]));
        if(r>=p.rows||c>=p.cols||(p.tri&&r<c))continue;
        uint v=reduce32(reduce(c11[i],m)<<16,m,cm)+reduce32(reduce(cmid[i],m)<<8,m,cm);v=(v>=m?v-m:v)+reduce(c00[i],m);
        out[ulong(r)*ulong(Nn)+c]=ushort(v>=m?v-m:v);}
}
// SUB (pipeline specialisation): outputs are updates RN(c - x) of the old output c, for the exact dot product
// x = sign * X * 2^scale, with one rounding (round_sum). A status on c gives zero with that status.
constant bool SUB [[function_constant(0)]];
template<int W> inline Number<N> rounded(thread const word (&X)[W],exponent_type scale,int sign,device const Number<N>& old){
    if(!SUB)return pack<N>(X,scale,sign,ok);
    Number<N> c=old;if(c.status)return zero<N>(c.status);
    Exact<W> x;for(int k=0;k<W;++k)x.w[k]=X[k];x.scale=scale;x.sign=highest(X)<0?0:-sign;
    return round_sum(x,c);
}
inline long slot_of(Line li,Line lj,uint j,constant Params& p){
    return li.slot>=0?long(li.slot)*p.out_cols+j:long(p.multi_rows)*p.out_cols+long(-1-li.slot)*p.multi_cols+lj.slot;
}
// Garner reconstruction of the exact integer sums of one band pair, the sub-block at (row0, col0) of the extended product
// (symmetric range), then either one rounding (single-band row and column) or exact two's-complement accumulation at its
// offset from the lowest bands. One dispatch never maps two entries to the same output.
// p.fold (SYRK pair b > c): entry (i, j) with i < j is the (c, b) term of output (j, i); diagonal entries count twice.
kernel void reconstruct(device const ushort* res [[buffer(0)]],device const uint* moduli [[buffer(11)]],device const uint* inv [[buffer(2)]],
                        device const uint* bounds [[buffer(3)]],device const Member* lband [[buffer(4)]],constant Params& p [[buffer(5)]],
                        device const Member* rband [[buffer(6)]],device const Line* lline [[buffer(7)]],device const Line* rline [[buffer(8)]],
                        device Number<N>* out [[buffer(9)]],device uint* acc [[buffer(10)]],device const uint* recips [[buffer(12)]],
                        uint2 g [[thread_position_in_grid]]){
    uint c=g.x,r=g.y;if(r>=p.rows||c>=p.cols||(p.tri&&r<c))return;
    // A folded pair is reconstructed in two dispatches (p.upper = 0: entries i >= j, 1: i < j) so that no two threads of one
    // dispatch accumulate into the same output.
    Member a=lband[p.row0+r],b=rband[p.col0+c];if(p.lower&&(p.fold?(a.line<b.line)!=bool(p.upper):a.line<b.line))return;
    ulong MN=ulong(p.Mp)*p.Np,idx=ulong(p.row0+r)*p.Np+p.col0+c;uint L=p.L,Wc=p.Wc;uint v[LF_MAX_MODULI];
    for(uint j=0;j<L;++j){uint m=moduli[j],cm=recips[j],t=res[ulong(j)*MN+idx];
        for(uint i=0;i<j;++i){uint u=v[i]>=m?v[i]-m:v[i];t=t>=u?t-u:t+m-u;t=reduce32(t*inv[i*LF_MAX_MODULI+j],m,cm);}v[j]=t;}
    word X[LF_TERM_WORDS];for(int k=0;k<LF_TERM_WORDS;++k)X[k]=0;X[0]=v[L-1];
    for(int j=int(L)-2;j>=0;--j){ulong carry=v[j],m=moduli[j];for(uint k=0;k<Wc;++k){ulong q=ulong(X[k])*m+carry;X[k]=uint(q);carry=q>>32;}}
    device const uint* mid=bounds;device const uint* full=bounds+Wc;
    int cmp=0;for(int k=int(Wc)-1;k>=0;--k)if(X[k]!=mid[k]){cmp=X[k]>mid[k]?1:-1;break;}
    int sign=1;if(cmp>0){long borrow=0;for(uint k=0;k<Wc;++k){long d=long(full[k])-long(X[k])-borrow;X[k]=uint(d);borrow=d<0;}sign=-1;}
    int row=a.line,col=b.line,times=1;if(p.fold&&row<=col){times=row==col?2:1;row=b.line;col=a.line;}
    Line li=lline[row],lj=rline[col];
    if(li.bands==1&&lj.bands==1){device Number<N>& o=out[ulong(row)*p.out_stride+col];o=rounded(X,exponent_type(a.lo)+b.lo-2*(32*N-1),sign,o);return;}
    device uint* s=acc+slot_of(li,lj,uint(col),p)*LF_ACC_WORDS;
    uint shift=uint(a.lo-lline[a.line].low)+uint(b.lo-rline[b.line].low),w=shift/32,sh=shift%32;
    for(int t=0;t<times;++t){ulong carry=0;
        for(uint k=0;k+w<LF_ACC_WORDS;++k){
            if(k>Wc&&!carry)break;
            uint x=k<Wc?X[k]<<sh:0;if(sh&&k>0&&k-1<Wc)x|=X[k-1]>>(32-sh);
            if(sign>0){ulong u=ulong(s[k+w])+x+carry;s[k+w]=uint(u);carry=u>>32;}
            else{ulong u=ulong(x)+carry;uint old=s[k+w];s[k+w]=uint(ulong(old)-u);carry=ulong(old)<u;}
        }}
}
// Multi-band outputs: one rounding of the accumulated exact sum (or of the update).
kernel void finish(device const Line* lline [[buffer(7)]],device const Line* rline [[buffer(8)]],device Number<N>* out [[buffer(9)]],
                   device const uint* acc [[buffer(10)]],constant Params& p [[buffer(5)]],uint2 g [[thread_position_in_grid]]){
    uint j=g.x,i=g.y;if(i>=p.rows||j>=p.cols||(p.lower&&i<j))return;
    Line li=lline[i],lj=rline[j];if(li.bands<1||lj.bands<1||(li.bands==1&&lj.bands==1))return;
    device const uint* s=acc+slot_of(li,lj,j,p)*LF_ACC_WORDS;word m[LF_ACC_WORDS];
    for(int k=0;k<LF_ACC_WORDS;++k)m[k]=s[k];
    int sign=1;if(m[LF_ACC_WORDS-1]>>31){sign=-1;ulong c=1;for(int k=0;k<LF_ACC_WORDS;++k){ulong t=ulong(~m[k])+c;m[k]=uint(t);c=t>>32;}}
    device Number<N>& o=out[ulong(i)*p.out_stride+j];o=rounded(m,exponent_type(li.low)+lj.low-2*(32*N-1),sign,o);
}
// ---- Resident products (Linalg::syrk/gemm into an Engine's CommandBatch; docs/numerics.md "Resident products") ----
// The host path's band analysis, member lists, modulus count and dispatch sizes, computed on the GPU so that one batch holds
// the whole product: the later dispatches of a call read their Params and threadgroup counts from the plan (indirect dispatch).
struct LineInfo { uint status; int cls,nb,pad; }; // cls: 0 GPU bands, 1 zero or status line, 2 exact host fallback (at wait)
struct ScanParams { uint lines,K,line_stride,k_stride,R,force; int G,max_bands,max_spread; };
// One SIMD group per line (lane l reads entries k = l, l+32, ...): the status OR, the largest and smallest nonzero exponent,
// then the host's greedy bands [hi-G, hi] from the largest remaining exponent down (analyze in linalg.mm, the same classes),
// one pass over the line per band; bands[i*R + b] = (lo, hi). Every SIMD reduction runs unconditionally in all 32 lanes (the
// grid is whole SIMD groups, a group exits as a whole, the band loop is uniform: docs/gpu-codegen.md rule 8).
constant int no_exponent_hi=INT_MIN,no_exponent_lo=INT_MAX; // outside the exponent range (|e| <= 1000000001)
kernel void analyze_lines(device const Number<N>* src [[buffer(0)]],device LineInfo* info [[buffer(1)]],device int2* bands [[buffer(2)]],
                          constant ScanParams& p [[buffer(3)]],uint g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    const uint i=g/32;if(i>=p.lines)return;
    device const Number<N>* line=src+ulong(i)*p.line_stride;
    uint st=0;int top=no_exponent_hi,bottom=no_exponent_lo;
    for(uint k=lane;k<p.K;k+=32){device const Number<N>& x=line[ulong(k)*p.k_stride];st|=x.status;
        if(!x.status&&x.sign){top=max(top,x.exponent);bottom=min(bottom,x.exponent);}}
    st=simd_or(st);top=simd_max(top);bottom=simd_min(bottom);
    LineInfo r;r.status=st;r.cls=1;r.nb=0;r.pad=0;
    if(!st&&top!=no_exponent_hi){
        if(p.force)r.cls=2;
        else if(p.max_bands>=1&&long(top)-bottom<=long(p.G)){r.nb=1;r.cls=0;if(lane==0)bands[ulong(i)*p.R]=int2(bottom,top);}
        else{long hi=top;bool fallback=long(top)-bottom>long(p.max_spread);int nb=0;
            while(!fallback){int lo=int(hi),next=no_exponent_hi;
                for(uint k=lane;k<p.K;k+=32){device const Number<N>& x=line[ulong(k)*p.k_stride];
                    if(x.status||!x.sign||long(x.exponent)>hi)continue;
                    if(long(x.exponent)>=hi-long(p.G))lo=min(lo,x.exponent);else next=max(next,x.exponent);}
                lo=simd_min(lo);next=simd_max(next);
                if(uint(nb)<p.R&&lane==0)bands[ulong(i)*p.R+nb]=int2(lo,int(hi));++nb;if(nb>p.max_bands)fallback=true;
                if(next==no_exponent_hi)break;hi=next;}
            r.nb=fallback?0:nb;r.cls=fallback?2:0;}
    }
    if(lane==0)info[i]=r;
}
// One threadgroup per side (a whole number of SIMD groups, at most 1024 threads): band b members in line order (from offset
// sum_{b'<b} count[b'], at most cap entries), the line table (bands, lowest band exponent, accumulator slot) and summary =
// [count[0..R), multi lines, single lines, widest band, fallback lines]. Each counter is one chunked scan (thread t owns lines
// [t*chunk, (t+1)*chunk)): SIMD prefix sums, then one SIMD group over the group totals. SIMD operations run in all lanes of a
// group (the branch on the group index is uniform per group).
struct SideParams { uint lines,R,cap; };
// Exclusive prefix of c over the threadgroup; total of all threads in *all.
inline uint group_prefix(uint c,uint t,uint T,threadgroup uint* part,threadgroup uint* all){
    const uint sg=t/32,lane=t%32;const uint pre=simd_prefix_exclusive_sum(c),sum=simd_sum(c);
    if(lane==0)part[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0){const uint v=lane<T/32?part[lane]:0,e=simd_prefix_exclusive_sum(v),s=simd_sum(v);if(lane<T/32)part[lane]=e;if(lane==0)*all=s;}
    threadgroup_barrier(mem_flags::mem_threadgroup);const uint r=part[sg]+pre;threadgroup_barrier(mem_flags::mem_threadgroup);return r;
}
kernel void plan_side(device const LineInfo* info [[buffer(0)]],device const int2* bands [[buffer(1)]],device Member* members [[buffer(2)]],
                      device Line* line [[buffer(3)]],device uint* summary [[buffer(4)]],constant SideParams& p [[buffer(5)]],
                      uint t [[thread_position_in_threadgroup]],uint T [[threads_per_threadgroup]]){
    threadgroup uint part[32];threadgroup uint total;
    const uint chunk=(p.lines+T-1)/T,first=min(p.lines,t*chunk),last=min(p.lines,first+chunk);
    uint widest=0,fallback=0;
    for(uint i=first;i<last;++i){LineInfo li=info[i];fallback+=li.cls==2;Line l;l.bands=0;l.low=0;l.slot=0;
        if(li.cls==0){l.bands=li.nb;l.low=bands[ulong(i)*p.R+li.nb-1].x;for(int b=0;b<li.nb;++b){int2 x=bands[ulong(i)*p.R+b];widest=max(widest,uint(x.y-x.x));}}
        line[i]=l;}
    uint offset=0;
    for(uint q=0;q<p.R+2;++q){
        uint c=0;for(uint i=first;i<last;++i){LineInfo li=info[i];c+=li.cls==0&&(q<p.R?li.nb>int(q):q==p.R?li.nb>1:li.nb==1);}
        uint rank=group_prefix(c,t,T,part,&total);const uint all=total;
        for(uint i=first;i<last;++i){LineInfo li=info[i];if(li.cls!=0)continue;
            if(q<p.R){if(li.nb>int(q)){uint r=offset+rank++;if(r<p.cap){int2 x=bands[ulong(i)*p.R+q];Member m;m.line=int(i);m.lo=x.x;m.hi=x.y;members[r]=m;}}}
            else if(q==p.R){if(li.nb>1)line[i].slot=int(rank++);}
            else if(li.nb==1)line[i].slot=-1-int(rank++);}
        if(t==0)summary[q]=all;
        if(q<p.R)offset+=all;
    }
    widest=simd_max(widest);if(t%32==0)part[t/32]=widest;threadgroup_barrier(mem_flags::mem_threadgroup);
    if(t==0){uint w=0;for(uint u=0;u<T/32;++u)w=max(w,part[u]);summary[p.R+2]=w;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    group_prefix(fallback,t,T,part,&total);if(t==0)summary[p.R+3]=total;
}
// Host constants of one call: base Params (data-independent fields), shapes, caps and threadgroup heights.
struct PlanConst { Params base; uint M,Nc,R,syrk,me_cap,ne_cap,B,batches,acc_words,copy_a,copy_b,tg_left,tg_right,tg_rec,tg_fin; };
// Plan layout (64-word records: constant-buffer offsets stay 256-byte aligned): 0 header {Me, Ne, L, Wc, overflow, gpu, any_fallback, slots, Mp, Np}, 1 finish Params, 2.. batch
// Params, then R*R band-pair Params, then the product strides; then 4-word indirect arguments: clear, copy a, copy b, finish,
// (digits left, digits right, product) per batch, one per band pair. table[3*pmax] = {moduli, Wc, offset in all_bounds}.
inline uint ceil_div(uint a,uint b){return (a+b-1)/b;}
inline void args(device uint* a,uint x,uint y,uint z){bool on=x&&y&&z;a[0]=on?x:0;a[1]=on?y:0;a[2]=on?z:0;a[3]=0;}
kernel void plan_final(device const uint* ls [[buffer(0)]],device const uint* rs [[buffer(1)]],device uint* plan [[buffer(2)]],
                       device const uint* table [[buffer(3)]],device const uint* all_bounds [[buffer(4)]],device uint* bnd [[buffer(5)]],
                       constant PlanConst& c [[buffer(6)]],uint g [[thread_position_in_grid]]){
    if(g)return;
    const uint R=c.R;uint Me=0,Ne=0,loff[64],roff[64];
    for(uint b=0;b<R;++b){loff[b]=Me;Me+=ls[b];roff[b]=Ne;Ne+=rs[b];}
    const bool overflow=Me>c.me_cap||Ne>c.ne_cap,gpu=Me&&Ne&&!overflow;
    const uint pmax=max(ls[R+2],rs[R+2]),L=table[3*pmax],Wc=table[3*pmax+1],boff=table[3*pmax+2];
    for(uint w=0;w<2*Wc;++w)bnd[w]=all_bounds[boff+w];
    const uint Mp=(Me+63)/64*64,Np=c.syrk?Mp:(Ne+63)/64*64,slots=gpu?ls[R]*c.Nc+ls[R+1]*rs[R]:0;
    const uint pairs=R*R,arg0=64*(3+c.batches+pairs);
    device uint* h=plan;h[0]=Me;h[1]=Ne;h[2]=L;h[3]=Wc;h[4]=overflow;h[5]=gpu;h[6]=overflow||ls[R+3]+rs[R+3]>0;h[7]=slots;h[8]=Mp;h[9]=Np;
    Params p=c.base;p.rows=Me;p.cols=Ne;p.Mp=Mp;p.Np=Np;p.L=L;p.Wc=Wc;p.multi_rows=ls[R];p.multi_cols=rs[R];
    Params f=p;f.rows=c.M;f.cols=c.Nc;*(device Params*)(plan+64)=f;
    args(plan+arg0,gpu?ceil_div(slots*c.acc_words,256):0,1,1);
    args(plan+arg0+4,h[6]?ceil_div(c.copy_a,256):0,1,1);args(plan+arg0+8,h[6]?ceil_div(c.copy_b,256):0,1,1);
    args(plan+arg0+12,slots?ceil_div(c.Nc,32):0,ceil_div(c.M,c.tg_fin),1);
    for(uint t=0;t<c.batches;++t){const uint j0=t*c.B,count=gpu&&L>j0?min(c.B,L-j0):0,groups=ceil_div(count,uint(LF_GROUP));
        Params q=p;q.j0=j0;q.group=count;q.j=j0;*(device Params*)(plan+64*(2+t))=q;device uint* a=plan+arg0+16+12*t;
        args(a,ceil_div(c.base.Kp,32),ceil_div(Mp,c.tg_left),groups);args(a+4,c.syrk?0:ceil_div(Np,32),ceil_div(c.base.Kp,c.tg_right),groups);
        args(a+8,Np/32,Mp/64,count);}
    uint q=0;
    for(uint b=0;b<R;++b)for(uint d=0;d<(c.syrk?b+1:R);++d)for(uint u=0;u<(c.syrk&&b>d?2u:1u);++u,++q){
        Params x=p;x.rows=ls[b];x.cols=rs[d];x.row0=loff[b];x.col0=roff[d];x.tri=c.syrk&&b==d;x.fold=c.syrk&&b>d;x.upper=u;
        *(device Params*)(plan+64*(2+c.batches+q))=x;args(plan+arg0+16+12*c.batches+4*q,gpu?ceil_div(x.cols,32):0,ceil_div(x.rows,c.tg_rec),1);}
    device uint* st=plan+64*(2+c.batches+pairs);st[0]=2*Mp*c.base.Kp;st[1]=Mp*c.base.Kp;st[2]=2*c.base.Kp*Np;st[3]=0;
}
// Zero / status lines (as the host assembly): an output with a zero or status line is zero with the OR of the line statuses
// (and of the old entry for an update); an update by a zero line leaves the entry unchanged. SYRK: lower triangle.
struct TrivialParams { uint M,Nc,syrk,sub; };
kernel void trivial_outputs(device const LineInfo* li [[buffer(0)]],device const LineInfo* ri [[buffer(1)]],device Number<N>* out [[buffer(2)]],
                            constant TrivialParams& p [[buffer(3)]],uint2 g [[thread_position_in_grid]]){
    const uint j=g.x,i=g.y;if(i>=p.M||j>=p.Nc||(p.syrk&&j>i))return;
    LineInfo a=li[i],b=ri[j];if(a.cls!=1&&b.cls!=1)return;
    device Number<N>& o=out[ulong(i)*p.Nc+j];uint st=a.status|b.status;if(p.sub)st|=o.status;
    if(!p.sub||st)o=zero<N>(st);
}
// SYRK with the full matrix: the strict upper triangle copies the lower.
kernel void mirror_upper(device Number<N>* out [[buffer(0)]],constant uint& n [[buffer(1)]],uint2 g [[thread_position_in_grid]]){
    const uint j=g.x,i=g.y;if(i>=n||j>=n||j<=i)return;out[ulong(i)*n+j]=out[ulong(j)*n+i];
}
kernel void copy_words(device const uint* src [[buffer(0)]],device uint* dst [[buffer(1)]],constant uint& count [[buffer(2)]],uint i [[thread_position_in_grid]]){
    if(i<count)dst[i]=src[i];
}
kernel void clear_words(device uint* dst [[buffer(0)]],device const uint* plan [[buffer(1)]],constant uint& acc_words [[buffer(2)]],uint i [[thread_position_in_grid]]){
    if(i<plan[7]*acc_words)dst[i]=0;
}
