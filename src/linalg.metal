// Dense products with one rounding per output (src/linalg.mm, docs/numerics.md "Dense products").
// Compiled with MSL 4.0 per (LF_BITS, LF_TERM_WORDS, LF_ACC_WORDS, LF_MAX_MODULI, LF_GROUP).
using namespace metal;
using namespace mpp::tensor_ops;
using namespace limbforge;
constant int N=LF_BITS/32;
struct Params { uint rows,cols,K,Kp,Mp,Np,j,L,Wc,tri,lower,fold,ls_line,ls_k,rs_line,rs_k,out_cols,multi_rows,multi_cols,both,j0,group,upper,row0,col0,pad; };
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
// Left band, one plane per modulus of the group: cat = [d1 | d0] (Mp x 2Kp), d1, d0 (Mp x Kp); padding is written as zero.
// With p.both (SYRK) the right band of the same lines is written too: rcat = [d0 ; d1] (2Kp x Np), Np = Mp.
kernel void digits_left(device const Number<N>* src [[buffer(0)]],device const Member* band [[buffer(1)]],device int8_t* cat [[buffer(2)]],
                        device int8_t* d1 [[buffer(3)]],device int8_t* d0 [[buffer(4)]],constant Params& p [[buffer(5)]],
                        device int8_t* rcat [[buffer(6)]],device const uint* moduli [[buffer(11)]],device const uint* recips [[buffer(12)]],
                        uint2 g [[thread_position_in_grid]]){
    uint k=g.x,r=g.y;if(k>=p.Kp||r>=p.Mp)return;
    Digits d;for(int q=0;q<LF_GROUP;++q){d.d1[q]=0;d.d0[q]=0;}
    if(k<p.K&&r<p.rows){Member b=band[r];d=digits(src[ulong(b.line)*p.ls_line+ulong(k)*p.ls_k],b.lo,b.hi,moduli,recips,p.j0);}
    ulong row=r,cp=ulong(p.Mp)*2*p.Kp,hp=ulong(p.Mp)*p.Kp,rp=ulong(2)*p.Kp*p.Np;
    for(uint q=0;q<p.group;++q){
        cat[q*cp+row*2*p.Kp+k]=d.d1[q];cat[q*cp+row*2*p.Kp+p.Kp+k]=d.d0[q];d1[q*hp+row*p.Kp+k]=d.d1[q];d0[q*hp+row*p.Kp+k]=d.d0[q];
        if(p.both){rcat[q*rp+ulong(k)*p.Np+r]=d.d0[q];rcat[q*rp+ulong(p.Kp+k)*p.Np+r]=d.d1[q];}
    }
}
// Right band: rcat = [d0 ; d1] (2Kp x Np) per modulus of the group.
kernel void digits_right(device const Number<N>* src [[buffer(0)]],device const Member* band [[buffer(1)]],device int8_t* cat [[buffer(2)]],
                         constant Params& p [[buffer(5)]],device const uint* moduli [[buffer(11)]],device const uint* recips [[buffer(12)]],
                         uint2 g [[thread_position_in_grid]]){
    uint c=g.x,k=g.y;if(c>=p.Np||k>=p.Kp)return;
    Digits d;for(int q=0;q<LF_GROUP;++q){d.d1[q]=0;d.d0[q]=0;}
    if(k<p.K&&c<p.cols){Member b=band[c];d=digits(src[ulong(b.line)*p.rs_line+ulong(k)*p.rs_k],b.lo,b.hi,moduli,recips,p.j0);}
    ulong rp=ulong(2)*p.Kp*p.Np;
    for(uint q=0;q<p.group;++q){cat[q*rp+ulong(k)*p.Np+c]=d.d0[q];cat[q*rp+ulong(p.Kp+k)*p.Np+c]=d.d1[q];}
}
// C (M x N) = A (M x K) B (K x N), int8 -> exact int32 (|sum| <= K*2^14 < 2^31). mnk.w: skip tiles strictly above the diagonal.
kernel void product(device int8_t* A [[buffer(0)]],device int8_t* B [[buffer(1)]],device int32_t* C [[buffer(2)]],
                    constant uint4& mnk [[buffer(3)]],uint2 tgid [[threadgroup_position_in_grid]]){
    if(mnk.w&&int(tgid.y)*64+63<int(tgid.x)*32)return;
    int M=int(mnk.x),Nn=int(mnk.y),K=int(mnk.z);
    auto tA=tensor<device int8_t,dextents<int32_t,2>,tensor_inline>(A,dextents<int32_t,2>(K,M));
    auto tB=tensor<device int8_t,dextents<int32_t,2>,tensor_inline>(B,dextents<int32_t,2>(Nn,K));
    auto tC=tensor<device int32_t,dextents<int32_t,2>,tensor_inline>(C,dextents<int32_t,2>(Nn,M));
    constexpr auto d=matmul2d_descriptor(64,32,static_cast<int>(dynamic_extent),false,false,false);
    matmul2d<d,execution_simdgroups<4>> op;
    auto mA=tA.slice(0,int(tgid.y)*64);auto mB=tB.slice(int(tgid.x)*32,0);auto mC=tC.slice(int(tgid.x)*32,int(tgid.y)*64);
    op.run(mA,mB,mC);
}
inline uint reduce(int g,uint m){int r=g%int(m);return uint(r<0?r+int(m):r);}
// residue_j = (65536*G11 + 256*Gmid + G00) mod m.
kernel void combine(device const int* g11 [[buffer(0)]],device const int* gmid [[buffer(1)]],device const int* g00 [[buffer(2)]],
                    device ushort* res [[buffer(3)]],constant Params& p [[buffer(5)]],device const uint* moduli [[buffer(11)]],
                    device const uint* recips [[buffer(12)]],uint2 g [[thread_position_in_grid]]){
    uint c=g.x,r=g.y;if(r>=p.rows||c>=p.cols||(p.tri&&r<c))return;
    ulong idx=ulong(r)*p.Np+c;uint m=moduli[p.j],cm=recips[p.j];
    uint v=reduce32(reduce(g11[idx],m)<<16,m,cm)+reduce32(reduce(gmid[idx],m)<<8,m,cm);v=(v>=m?v-m:v)+reduce(g00[idx],m);
    res[ulong(p.j)*p.Mp*p.Np+idx]=ushort(v>=m?v-m:v);
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
    if(li.bands==1&&lj.bands==1){out[ulong(row)*p.out_cols+col]=pack<N>(X,exponent_type(a.lo)+b.lo-2*(32*N-1),sign,ok);return;}
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
// Multi-band outputs: one rounding of the accumulated exact sum.
kernel void finish(device const Line* lline [[buffer(7)]],device const Line* rline [[buffer(8)]],device Number<N>* out [[buffer(9)]],
                   device const uint* acc [[buffer(10)]],constant Params& p [[buffer(5)]],uint2 g [[thread_position_in_grid]]){
    uint j=g.x,i=g.y;if(i>=p.rows||j>=p.cols||(p.lower&&i<j))return;
    Line li=lline[i],lj=rline[j];if(li.bands<1||lj.bands<1||(li.bands==1&&lj.bands==1))return;
    device const uint* s=acc+slot_of(li,lj,j,p)*LF_ACC_WORDS;word m[LF_ACC_WORDS];
    for(int k=0;k<LF_ACC_WORDS;++k)m[k]=s[k];
    int sign=1;if(m[LF_ACC_WORDS-1]>>31){sign=-1;ulong c=1;for(int k=0;k<LF_ACC_WORDS;++k){ulong t=ulong(~m[k])+c;m[k]=uint(t);c=t>>32;}}
    out[ulong(i)*p.out_cols+j]=pack<N>(m,exponent_type(li.low)+lj.low-2*(32*N-1),sign,ok);
}
