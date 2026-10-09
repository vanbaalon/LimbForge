using namespace metal;
using namespace limbforge;
constant int N=LF_BITS/32;
constant bool batch_fused [[function_constant(0)]];
struct BatchParams {ulong count,m,n,k,sa,sb,sc; int n0; uint flags; ulong row_offset;};
inline Number<N> bz(Number<N>){return zero<N>();}
inline Complex<N> bz(Complex<N>){return {zero<N>(),zero<N>()};}
inline Number<N> bone(Number<N>){Number<N> z=zero<N>();z.sign=1;z.limb[N-1]=0x80000000u;return z;}
inline Complex<N> bone(Complex<N> z){return {bone(z.re),zero<N>()};}
inline Number<N> bm(Number<N> a,Number<N> b){return mul(a,b);}
inline Complex<N> bm(Complex<N> a,Complex<N> b){return cmul(a,b);}
inline Number<N> bn(Number<N> x){return negate(x);}
inline Complex<N> bn(Complex<N> x){return {negate(x.re),negate(x.im)};}
inline Number<N> bi(Number<N> x){return div(bone(x),x);}
inline Complex<N> bi(Complex<N> x){return cdiv(bone(x),x);}
inline Number<N> ba(Number<N> a,Number<N> b,Number<N> c){return batch_fused?limbforge::fma(a,b,c):add(mul(a,b),c);}
inline Complex<N> ba(Complex<N> a,Complex<N> b,Complex<N> c){return batch_fused?cfma_rolled(a,b,c):cadd(cmul(a,b),c);}
template<class T> T bp(T x,int exponent){
    T r=bone(x);long e=exponent;bool inv=e<0;ulong k=inv?ulong(-e):ulong(e);
    while(k){if(k&1)r=bm(r,x);k>>=1;if(k)x=bm(x,x);}return inv?bi(r):r;
}
// Tiles 8x4, a SIMD group per tile. Narrower K tiles at wide complex precisions keep shared storage below 32 KB.
constant uint TK=N<=14?16:8;
template<class T> void bgemm(device const T* A,device const T* B,device T* C,constant BatchParams& p,
                            uint tid,uint3 tile,threadgroup T* a,threadgroup T* b){
    ulong row=ulong(tile.y)*8+tid/4,col=ulong(tile.x)*4+tid%4,base=ulong(tile.z);
    bool valid=row<p.m&&col<p.n;T z=bz(T{}),sum=(valid&&(p.flags&1))?C[base*p.sc+(row+p.row_offset)*p.n+col]:z;
    _Pragma("clang loop unroll(disable)") for(ulong k0=0;k0<p.k;k0+=TK){
        for(uint t=tid;t<8*TK;t+=32){ulong r=ulong(tile.y)*8+t/TK,k=k0+t%TK;a[t]=r<p.m&&k<p.k?A[base*p.sa+r*p.k+k]:z;}
        for(uint t=tid;t<TK*4;t+=32){ulong k=k0+t/4,c=ulong(tile.x)*4+t%4;b[t]=k<p.k&&c<p.n?B[base*p.sb+k*p.n+c]:z;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // No padded zero operations: avoids contaminating statuses after the actual end of the dot.
        _Pragma("clang loop unroll(disable)") for(uint k=0;k<TK&&k0+k<p.k;++k)if(valid)sum=ba(a[(tid/4)*TK+k],b[k*4+tid%4],sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(valid)C[base*p.sc+(row+p.row_offset)*p.n+col]=(p.flags&2)?bn(sum):sum;
}
kernel void batch_gemm_real(device const Number<N>* A [[buffer(0)]],device const Number<N>* B [[buffer(1)]],device Number<N>* C [[buffer(2)]],
                            constant BatchParams& p [[buffer(3)]],uint tid [[thread_index_in_threadgroup]],uint3 tile [[threadgroup_position_in_grid]]){
    threadgroup Number<N> a[8*TK],b[TK*4];bgemm(A,B,C,p,tid,tile,a,b);}
kernel void batch_gemm_complex(device const Complex<N>* A [[buffer(0)]],device const Complex<N>* B [[buffer(1)]],device Complex<N>* C [[buffer(2)]],
                               constant BatchParams& p [[buffer(3)]],uint tid [[thread_index_in_threadgroup]],uint3 tile [[threadgroup_position_in_grid]]){
    threadgroup Complex<N> a[8*TK],b[TK*4];bgemm(A,B,C,p,tid,tile,a,b);}
// Two independent 4x4 products per SIMD group, with all 32 lanes producing an output.
template<class T> void bgemm4(device const T* A,device const T* B,device T* C,constant BatchParams& p,uint tid,uint group,threadgroup T* a,threadgroup T* b){
    ulong batch=ulong(group)*2+tid/16;uint local=tid%16,row=local/4,col=local%4,offset=(tid/16)*16;
    bool valid=batch<p.count;T z=bz(T{});a[tid]=valid?A[batch*p.sa+local]:z;b[tid]=valid?B[batch*p.sb+local]:z;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(valid){T sum=(p.flags&1)?C[batch*p.sc+p.row_offset*p.n+local]:z;
        _Pragma("clang loop unroll(disable)") for(uint k=0;k<4;++k)sum=ba(a[offset+row*4+k],b[offset+k*4+col],sum);
        C[batch*p.sc+p.row_offset*p.n+local]=(p.flags&2)?bn(sum):sum;}
}
kernel void batch_gemm4_real(device const Number<N>* A [[buffer(0)]],device const Number<N>* B [[buffer(1)]],device Number<N>* C [[buffer(2)]],constant BatchParams& p [[buffer(3)]],uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]){
    threadgroup Number<N> a[32],b[32];bgemm4(A,B,C,p,tid,group.x,a,b);}
kernel void batch_gemm4_complex(device const Complex<N>* A [[buffer(0)]],device const Complex<N>* B [[buffer(1)]],device Complex<N>* C [[buffer(2)]],constant BatchParams& p [[buffer(3)]],uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]){
    threadgroup Complex<N> a[32],b[32];bgemm4(A,B,C,p,tid,group.x,a,b);}
template<class T> void powers(device const T* E,device const T* Y,device T* P,constant BatchParams& p,uint i){
    if(ulong(i)>=p.count*p.k)return;ulong b=i/p.k,k=i%p.k;T y=Y[i],pw=bp(y,p.n0),e=E[i];
    _Pragma("clang loop unroll(disable)") for(ulong n=0;n<p.m;++n){P[(b*p.m+n)*p.k+k]=bm(e,pw);if(n+1<p.m)pw=bm(pw,y);}
}
kernel void batch_powers_real(device const Number<N>* E [[buffer(0)]],device const Number<N>* Y [[buffer(1)]],device Number<N>* P [[buffer(2)]],constant BatchParams& p [[buffer(3)]],uint i [[thread_position_in_grid]]){powers(E,Y,P,p,i);}
kernel void batch_powers_complex(device const Complex<N>* E [[buffer(0)]],device const Complex<N>* Y [[buffer(1)]],device Complex<N>* P [[buffer(2)]],constant BatchParams& p [[buffer(3)]],uint i [[thread_position_in_grid]]){powers(E,Y,P,p,i);}
kernel void batch_normal(device const Number<N>* J [[buffer(0)]],device const Number<N>* g [[buffer(1)]],device Number<N>* A [[buffer(2)]],
                         constant BatchParams& p [[buffer(3)]],device Number<N>* rhs [[buffer(4)]],uint x [[thread_position_in_grid]]){
    if(ulong(x)>=p.n*(p.n+1))return;ulong i=x/(p.n+1),j=x%(p.n+1);if(j<p.n&&i<j)return;
    Number<N> s=zero<N>();_Pragma("clang loop unroll(disable)") for(ulong k=0;k<p.k;++k){Number<N> a=J[k*p.n+i],b=j==p.n?g[k]:J[k*p.n+j];s=ba(a,b,s);}
    if(j==p.n)rhs[i]=s;else{A[i*p.n+j]=s;if(i!=j)A[j*p.n+i]=s;}
}
kernel void batch_augment(device const Number<N>* J [[buffer(0)]],device const Number<N>* g [[buffer(1)]],device Number<N>* out [[buffer(2)]],constant BatchParams& p [[buffer(3)]],uint x [[thread_position_in_grid]]){
    if(ulong(x)>=p.k*(p.n+1))return;ulong row=x/(p.n+1),col=x%(p.n+1);out[x]=col==p.n?g[row]:J[row*p.n+col];}
kernel void batch_extract(device const Number<N>* gram [[buffer(0)]],device Number<N>* A [[buffer(2)]],constant BatchParams& p [[buffer(3)]],device Number<N>* rhs [[buffer(4)]],uint x [[thread_position_in_grid]]){
    if(ulong(x)>=p.n*p.n)return;ulong i=x/p.n,j=x%p.n;A[x]=gram[i*(p.n+1)+j];if(!j)rhs[i]=gram[i*(p.n+1)+p.n];}
// Fully resident, right-looking scalar-column Cholesky; each dispatch is a global dependency boundary.
kernel void batch_chol_init(device const Number<N>* A [[buffer(0)]],device const Number<N>* D [[buffer(1)]],device Number<N>* L [[buffer(2)]],
                            constant BatchParams& p [[buffer(3)]],device const Number<N>* mu [[buffer(4)]],device uint* status [[buffer(5)]],uint x [[thread_position_in_grid]]){
    if(ulong(x)>=p.count*p.n*p.n)return;ulong trial=x/(p.n*p.n),t=x%(p.n*p.n),i=t/p.n,j=t%p.n;
    Number<N> z=zero<N>();if(i>=j){z=A[t];if(i==j){Number<N> u=mu[trial],d=D[i];z=add(z,mul(u,d));}}
    L[x]=z;if(!t)status[trial]=0;
}
kernel void batch_chol_pivot(device Number<N>* L [[buffer(2)]],constant BatchParams& p [[buffer(3)]],device uint* status [[buffer(5)]],uint t [[thread_position_in_grid]]){
    if(t>=p.count||status[t])return;ulong o=ulong(t)*p.n*p.n+p.k*p.n+p.k;Number<N> z=L[o];
    if(z.status||z.sign<=0){status[t]=uint(p.k+1);return;}L[o]=limbforge::sqrt(z);
    if(L[o].status)status[t]=uint(p.k+1);
}
kernel void batch_chol_column(device Number<N>* L [[buffer(2)]],constant BatchParams& p [[buffer(3)]],device const uint* status [[buffer(5)]],uint x [[thread_position_in_grid]]){
    ulong remaining=p.n-p.k-1;if(ulong(x)>=p.count*remaining)return;ulong t=x/remaining,i=p.k+1+x%remaining;if(status[t])return;ulong o=t*p.n*p.n;
    Number<N> a=L[o+i*p.n+p.k],d=L[o+p.k*p.n+p.k];L[o+i*p.n+p.k]=div(a,d);
}
kernel void batch_chol_update(device Number<N>* L [[buffer(2)]],constant BatchParams& p [[buffer(3)]],device const uint* status [[buffer(5)]],uint x [[thread_position_in_grid]]){
    ulong remaining=p.n-p.k-1;if(ulong(x)>=p.count*remaining*remaining)return;ulong t=x/(remaining*remaining),ij=x%(remaining*remaining),i=p.k+1+ij/remaining,j=p.k+1+ij%remaining;
    if(status[t]||i<j)return;ulong o=t*p.n*p.n;Number<N> a=L[o+i*p.n+p.k],b=L[o+j*p.n+p.k],c=L[o+i*p.n+j];L[o+i*p.n+j]=sub(c,mul(a,b));
}
kernel void batch_chol_finish(device Number<N>* L [[buffer(2)]],constant BatchParams& p [[buffer(3)]],device const uint* status [[buffer(5)]],uint x [[thread_position_in_grid]]){
    if(ulong(x)>=p.count*p.n*p.n)return;ulong t=x/(p.n*p.n),ij=x%(p.n*p.n),j=ij%p.n,i=ij/p.n;
    if(status[t]&&i>=j&&j>=status[t]-1)L[x]=zero<N>(invalid);
}
// One thread owns a (trial,RHS) column. No global row barriers or per-row dispatches.
kernel void batch_chol_solve(device const Number<N>* L [[buffer(0)]],device const uint* status [[buffer(1)]],device Number<N>* X [[buffer(2)]],constant BatchParams& p [[buffer(3)]],device const Number<N>* B [[buffer(4)]],uint x [[thread_position_in_grid]]){
    if(ulong(x)>=p.count*p.m)return;ulong trial=x/p.m,c=x%p.m,a=trial*p.n*p.n,b=trial*p.n*p.m;
    if(status[trial]){for(ulong i=0;i<p.n;++i)X[b+i*p.m+c]=zero<N>(invalid);return;}
    _Pragma("clang loop unroll(disable)") for(ulong i=0;i<p.n;++i){Number<N> z=B[i*p.m+c];
        _Pragma("clang loop unroll(disable)") for(ulong j=0;j<i;++j){Number<N> u=L[a+i*p.n+j],v=X[b+j*p.m+c];z=sub(z,mul(u,v));}
        Number<N> d=L[a+i*p.n+i];X[b+i*p.m+c]=div(z,d);}
    _Pragma("clang loop unroll(disable)") for(ulong i=p.n;i-->0;){Number<N> z=X[b+i*p.m+c];
        _Pragma("clang loop unroll(disable)") for(ulong j=i+1;j<p.n;++j){Number<N> u=L[a+j*p.n+i],v=X[b+j*p.m+c];z=sub(z,mul(u,v));}
        Number<N> d=L[a+i*p.n+i];X[b+i*p.m+c]=div(z,d);}
}
struct InlineRecord {ulong ro,io;int re,ie,rs,is;uint rst,ist;};
inline Number<N> import_part(device const uchar* bytes,ulong offset,int exponent,int sign,uint status){
    Number<N> z=zero<N>(status);if(status||!sign)return z;z.sign=sign;z.exponent=exponent;
    // Odd word widths are left aligned in ceil(N/2) 64-bit MPFR limbs.
    device const uint* s=reinterpret_cast<device const uint*>(bytes+offset)+(N%2);
    for(int j=0;j<N;++j)z.limb[j]=s[j];return checked(z);
}
kernel void batch_inline_complex(device const uchar* bytes [[buffer(0)]],device const InlineRecord* r [[buffer(1)]],device Complex<N>* out [[buffer(2)]],constant BatchParams& p [[buffer(3)]],uint i [[thread_position_in_grid]]){
    if(i>=p.count)return;InlineRecord d=r[i];out[i]={import_part(bytes,d.ro,d.re,d.rs,d.rst),import_part(bytes,d.io,d.ie,d.is,d.ist)};
}

struct PolyParams {ulong lanes,groups,sets;uint steps,terms,share,flags;};
inline Complex<N> phorner(device const Complex<N>* coeff,ulong offset,uint terms,Complex<N> y){
    Complex<N> z={zero<N>(),zero<N>()};if(!terms)return z;z=coeff[offset+terms-1];
    _Pragma("clang loop unroll(disable)") for(uint n=terms-1;n-->0;)z=ba(z,y,coeff[offset+n]);return z;
}
kernel void batch_polynomial_recurrence(device const Complex<N>* start [[buffer(0)]],device const Complex<N>* cp [[buffer(1)]],device Complex<N>* out [[buffer(2)]],constant PolyParams& p [[buffer(3)]],device const Complex<N>* cq [[buffer(4)]],device const Complex<N>* Y [[buffer(5)]],device const Complex<N>* Ep [[buffer(6)]],device const Complex<N>* Eq [[buffer(7)]],uint lane [[thread_position_in_grid]]){
    if(ulong(lane)>=p.lanes)return;ulong g=lane/p.share,set=p.sets==1?0:g;
    Complex<N> v[4];for(uint a=0;a<4;++a){v[a]=start[ulong(a)*p.lanes+lane];if(p.flags&1)out[ulong(a)*p.lanes+lane]=v[a];}
    _Pragma("clang loop unroll(disable)") for(uint step=0;step<p.steps;++step){
        uint k=(p.flags&2)?p.steps-1-step:step;Complex<N> y=Y[ulong(k)*p.groups+g],dot={zero<N>(),zero<N>()};
        _Pragma("clang loop unroll(disable)") for(uint a=0;a<4;++a){Complex<N> q=bm(Eq[(ulong(k)*4+a)*p.groups+g],phorner(cq,(set*4+a)*p.terms,p.terms,y));dot=ba(q,v[a],dot);}
        _Pragma("clang loop unroll(disable)") for(uint a=0;a<4;++a){Complex<N> z=bm(Ep[(ulong(k)*4+a)*p.groups+g],phorner(cp,(set*4+a)*p.terms,p.terms,y));v[a]=ba(z,dot,v[a]);
            if(p.flags&1)out[(ulong(step+1)*4+a)*p.lanes+lane]=v[a];}
    }
    if(!(p.flags&1))for(uint a=0;a<4;++a)out[ulong(a)*p.lanes+lane]=v[a];
}

// Padded local significands avoid runtime-indexed register arrays. Public buffers stay packed.
using PrivateNumber=limbforge_batched_private::Number<N>;
inline PrivateNumber batch_private_load(device const Number<N>& x){
    PrivateNumber r=limbforge_batched_private::zero<N>();
    for(int j=0;j<N;++j)r.limb[j]=x.limb[j];r.exponent=x.exponent;r.sign=x.sign;r.status=x.status;return r;
}
inline void batch_private_store(device Number<N>& out,thread const PrivateNumber& x){
    for(int j=0;j<N;++j)out.limb[j]=x.limb[j];out.exponent=x.exponent;out.sign=x.sign;out.status=x.status;
}
kernel void batch_private_complex(device const Complex<N>* A [[buffer(0)]],device const Complex<N>* B [[buffer(1)]],device Number<N>* C [[buffer(2)]],constant BatchParams& p [[buffer(3)]],uint2 index [[thread_position_in_grid]]){
    ulong entry=index.x,part=index.y,t=entry/(p.m*p.n),cell=entry%(p.m*p.n),row=cell/p.n,col=cell%p.n;
    if(t>=p.count)return;ulong o=2*(t*p.sc+p.row_offset*p.n+cell)+part;
    PrivateNumber sum=(p.flags&1)?batch_private_load(C[o]):limbforge_batched_private::zero<N>();
    _Pragma("clang loop unroll(disable)") for(ulong k=0;k<p.k;++k){
        PrivateNumber ar=batch_private_load(A[t*p.sa+row*p.k+k].re),ai=batch_private_load(A[t*p.sa+row*p.k+k].im);
        PrivateNumber br=batch_private_load(part?B[t*p.sb+k*p.n+col].im:B[t*p.sb+k*p.n+col].re);
        PrivateNumber bi=batch_private_load(part?B[t*p.sb+k*p.n+col].re:B[t*p.sb+k*p.n+col].im);
        if(batch_fused)sum=limbforge_batched_private::dot2_add_rolled(ar,br,part?ai:limbforge_batched_private::negate(ai),bi,sum);
        else {PrivateNumber x=limbforge_batched_private::mul<N,true>(ar,br),y=limbforge_batched_private::mul<N,true>(ai,bi);
            sum=limbforge_batched_private::add(part?limbforge_batched_private::add(x,y):limbforge_batched_private::sub(x,y),sum);}
    }
    if(p.flags&2)sum=limbforge_batched_private::negate(sum);batch_private_store(C[o],sum);
}

// Compact power panels preserve the unscaled repeated-multiplication state.
// Requires BatchParams::row_offset and private cmul using padded product workspaces.
using PrivateComplex=limbforge_batched_private::Complex<N>;
inline PrivateComplex batch_private_load(device const Complex<N>& x){return {batch_private_load(x.re),batch_private_load(x.im)};}
inline void batch_private_store(device Complex<N>& out,thread const PrivateComplex& x){batch_private_store(out.re,x.re);batch_private_store(out.im,x.im);}
inline PrivateNumber panel_one(PrivateNumber){auto r=limbforge_batched_private::zero<N>();r.sign=1;r.limb[N-1]=0x80000000u;return r;}
inline PrivateComplex panel_one(PrivateComplex){PrivateNumber x=limbforge_batched_private::zero<N>();return {panel_one(x),x};}
inline PrivateNumber panel_mul(PrivateNumber a,PrivateNumber b){return limbforge_batched_private::mul(a,b);}
inline PrivateComplex panel_mul(PrivateComplex a,PrivateComplex b){return limbforge_batched_private::cmul(a,b);}
inline PrivateNumber panel_inv(PrivateNumber x){return limbforge_batched_private::div(panel_one(x),x);}
inline PrivateComplex panel_inv(PrivateComplex x){return limbforge_batched_private::cdiv(panel_one(x),x);}
template<class T> T panel_powi(T x,int exponent){
    T r=panel_one(x);long e=exponent;bool inv=e<0;ulong k=inv?ulong(-e):ulong(e);
    while(k){if(k&1)r=panel_mul(r,x);k>>=1;if(k)x=panel_mul(x,x);}return inv?panel_inv(r):r;
}
template<class Packed> void panel_seed(device const Packed* Y,device Packed* state,constant BatchParams& p,uint i){
    if(ulong(i)>=p.count*p.k)return;auto value=panel_powi(batch_private_load(Y[i]),p.n0);batch_private_store(state[i],value);
}
kernel void batch_panel_seed_real(device const Number<N>* Y [[buffer(0)]],device Number<N>* state [[buffer(2)]],constant BatchParams& p [[buffer(3)]],uint i [[thread_position_in_grid]]){panel_seed(Y,state,p,i);}
kernel void batch_panel_seed_complex(device const Complex<N>* Y [[buffer(0)]],device Complex<N>* state [[buffer(2)]],constant BatchParams& p [[buffer(3)]],uint i [[thread_position_in_grid]]){panel_seed(Y,state,p,i);}
template<class Packed> void panel_powers(device const Packed* E,device const Packed* Y,device Packed* P,device Packed* state,constant BatchParams& p,uint i){
    if(ulong(i)>=p.count*p.k)return;ulong b=i/p.k,k=i%p.k;
    auto y=batch_private_load(Y[i]),pw=batch_private_load(state[i]),e=batch_private_load(E[i]);
    _Pragma("clang loop unroll(disable)") for(ulong n=0;n<p.m;++n){
        auto value=panel_mul(e,pw);batch_private_store(P[(b*p.m+n)*p.k+k],value);
        if(p.row_offset+n+1<p.sa)pw=panel_mul(pw,y);
    }
    batch_private_store(state[i],pw);
}
kernel void batch_panel_powers_real(device const Number<N>* E [[buffer(0)]],device const Number<N>* Y [[buffer(1)]],device Number<N>* P [[buffer(2)]],constant BatchParams& p [[buffer(3)]],device Number<N>* state [[buffer(4)]],uint i [[thread_position_in_grid]]){panel_powers(E,Y,P,state,p,i);}
kernel void batch_panel_powers_complex(device const Complex<N>* E [[buffer(0)]],device const Complex<N>* Y [[buffer(1)]],device Complex<N>* P [[buffer(2)]],constant BatchParams& p [[buffer(3)]],device Complex<N>* state [[buffer(4)]],uint i [[thread_position_in_grid]]){panel_powers(E,Y,P,state,p,i);}

// Evaluate p/q once per (step, shared group, component).
// Degree-by-degree Horner keeps rounded states in device memory, avoiding live complex temporaries.
kernel void batch_source_seed(device const Complex<N>* coeff [[buffer(1)]],device Number<N>* out [[buffer(2)]],constant PolyParams& p [[buffer(3)]],uint2 index [[thread_position_in_grid]]){
    ulong x=index.x;if(x>=ulong(p.steps)*p.groups)return;ulong k=x/p.groups,g=x%p.groups,set=p.sets==1?0:g,a=index.y/2,part=index.y%2;
    ulong dst=2*((k*4+a)*p.groups+g)+part;auto value=zero<N>();
    if(p.terms){auto z=coeff[(set*4+a)*p.terms+p.terms-1];value=part?z.im:z.re;}out[dst]=value;
}
kernel void batch_source_step(device const Complex<N>* state [[buffer(0)]],device const Complex<N>* coeff [[buffer(1)]],device Number<N>* out [[buffer(2)]],constant PolyParams& p [[buffer(3)]],device const Complex<N>* Y [[buffer(5)]],uint2 index [[thread_position_in_grid]]){
    ulong x=index.x;if(x>=ulong(p.steps)*p.groups)return;ulong k=x/p.groups,g=x%p.groups,set=p.sets==1?0:g,a=index.y/2,part=index.y%2,dst=(k*4+a)*p.groups+g;
    auto zr=batch_private_load(state[dst].re),zi=batch_private_load(state[dst].im),yr=batch_private_load(part?Y[x].im:Y[x].re),yi=batch_private_load(part?Y[x].re:Y[x].im);
    ulong offset=(set*4+a)*p.terms+p.flags;auto addend=batch_private_load(part?coeff[offset].im:coeff[offset].re);PrivateNumber value;
    if(batch_fused)value=limbforge_batched_private::dot2_add_rolled(zr,yr,part?zi:limbforge_batched_private::negate(zi),yi,addend);
    else{auto r=limbforge_batched_private::mul<N,true>(zr,yr),i=limbforge_batched_private::mul<N,true>(zi,yi);
        value=limbforge_batched_private::add(part?limbforge_batched_private::add(r,i):limbforge_batched_private::sub(r,i),addend);}
    batch_private_store(out[2*dst+part],value);
}
kernel void batch_source_scale(device const Complex<N>* state [[buffer(0)]],device Number<N>* out [[buffer(2)]],constant PolyParams& p [[buffer(3)]],device const Complex<N>* E [[buffer(6)]],uint2 index [[thread_position_in_grid]]){
    ulong x=index.x;if(x>=ulong(p.steps)*p.groups)return;ulong k=x/p.groups,g=x%p.groups,a=index.y/2,part=index.y%2,dst=(k*4+a)*p.groups+g;
    auto er=batch_private_load(E[dst].re),ei=batch_private_load(E[dst].im),zr=batch_private_load(part?state[dst].im:state[dst].re),zi=batch_private_load(part?state[dst].re:state[dst].im);
    auto r=limbforge_batched_private::mul<N,true>(er,zr),i=limbforge_batched_private::mul<N,true>(ei,zi);
    auto value=part?limbforge_batched_private::add(r,i):limbforge_batched_private::sub(r,i);batch_private_store(out[2*dst+part],value);
}
// One component per SIMD group; the four-term dot retains ascending component order.
// A separate update pass ensures no lane observes partially updated state.
kernel void batch_source_dot(device const Complex<N>* state [[buffer(0)]],device const Complex<N>* Q [[buffer(1)]],device Number<N>* dot [[buffer(2)]],constant BatchParams& p [[buffer(3)]],uint2 index [[thread_position_in_grid]]){
    ulong lane=index.x;if(lane>=p.count)return;uint part=index.y;ulong group=lane/p.n;
    auto sum=limbforge_batched_private::zero<N>();
    _Pragma("clang loop unroll(disable)") for(uint a=0;a<4;++a){
        ulong q=(p.k*4+a)*p.m+group,v=p.sa+ulong(a)*p.count+lane;
        auto qr=batch_private_load(Q[q].re),qi=batch_private_load(Q[q].im);
        auto vr=batch_private_load(part?state[v].im:state[v].re),vi=batch_private_load(part?state[v].re:state[v].im);
        if(batch_fused)sum=limbforge_batched_private::dot2_add_rolled(qr,vr,part?qi:limbforge_batched_private::negate(qi),vi,sum);
        else{auto r=limbforge_batched_private::mul<N,true>(qr,vr),i=limbforge_batched_private::mul<N,true>(qi,vi);
            sum=limbforge_batched_private::add(part?limbforge_batched_private::add(r,i):limbforge_batched_private::sub(r,i),sum);}
    }
    batch_private_store(dot[2*lane+part],sum);
}
kernel void batch_source_update(device const Complex<N>* state [[buffer(0)]],device const Complex<N>* P [[buffer(1)]],device Number<N>* out [[buffer(2)]],constant BatchParams& p [[buffer(3)]],device const Complex<N>* dot [[buffer(4)]],uint2 index [[thread_position_in_grid]]){
    ulong lane=index.x;if(lane>=p.count)return;uint part=index.y;ulong group=lane/p.n;
    auto dr=batch_private_load(part?dot[lane].im:dot[lane].re),di=batch_private_load(part?dot[lane].re:dot[lane].im);
    _Pragma("clang loop unroll(disable)") for(uint a=0;a<4;++a){
        ulong weight=(p.k*4+a)*p.m+group,v=ulong(a)*p.count+lane;
        auto pr=batch_private_load(P[weight].re),pi=batch_private_load(P[weight].im);
        auto old=batch_private_load(part?state[p.sa+v].im:state[p.sa+v].re);PrivateNumber value;
        if(batch_fused)value=limbforge_batched_private::dot2_add_rolled(pr,dr,part?pi:limbforge_batched_private::negate(pi),di,old);
        else{auto r=limbforge_batched_private::mul<N,true>(pr,dr),i=limbforge_batched_private::mul<N,true>(pi,di);
            value=limbforge_batched_private::add(part?limbforge_batched_private::add(r,i):limbforge_batched_private::sub(r,i),old);}
        batch_private_store(out[2*(p.sc+v)+part],value);
    }
}
