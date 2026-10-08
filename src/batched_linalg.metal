using namespace metal;
using namespace limbforge;
constant int N=LF_BITS/32;
constant bool batch_fused [[function_constant(0)]];
struct BatchParams {ulong count,m,n,k,sa,sb,sc; int n0; uint flags;};
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
    bool valid=row<p.m&&col<p.n;T z=bz(T{}),sum=(valid&&(p.flags&1))?C[base*p.sc+row*p.n+col]:z;
    _Pragma("clang loop unroll(disable)") for(ulong k0=0;k0<p.k;k0+=TK){
        for(uint t=tid;t<8*TK;t+=32){ulong r=ulong(tile.y)*8+t/TK,k=k0+t%TK;a[t]=r<p.m&&k<p.k?A[base*p.sa+r*p.k+k]:z;}
        for(uint t=tid;t<TK*4;t+=32){ulong k=k0+t/4,c=ulong(tile.x)*4+t%4;b[t]=k<p.k&&c<p.n?B[base*p.sb+k*p.n+c]:z;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // No padded zero operations: avoids contaminating statuses after the actual end of the dot.
        _Pragma("clang loop unroll(disable)") for(uint k=0;k<TK&&k0+k<p.k;++k)if(valid)sum=ba(a[(tid/4)*TK+k],b[k*4+tid%4],sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(valid)C[base*p.sc+row*p.n+col]=(p.flags&2)?bn(sum):sum;
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
    if(valid){T sum=(p.flags&1)?C[batch*p.sc+local]:z;
        _Pragma("clang loop unroll(disable)") for(uint k=0;k<4;++k)sum=ba(a[offset+row*4+k],b[offset+k*4+col],sum);
        C[batch*p.sc+local]=(p.flags&2)?bn(sum):sum;}
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
