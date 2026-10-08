// Polynomial jets (plan S1) and norms / status summaries (plan S4): src/numerics.mm, docs/numerics.md.
// Compiled per precision (LF_BITS); kernels are specialised by function constants. Loops stay rolled so that each
// exact fma/cfma is instantiated once (docs/gpu-codegen.md); no runtime-indexed private arrays.
using namespace metal;
using namespace limbforge;
constant int N=LF_BITS/32;
constant uint NO_INDEX=0xffffffffu;
constant int NO_EXPONENT=-2147483647-1;

// ---- Polynomials: r_j = 0; for k = terms-1 .. 0: r_2 = mac(r_2,x,r_1), r_1 = mac(r_1,x,r_0), r_0 = mac(r_0,x,c_k) ----
constant uint poly_flags [[function_constant(0)]];
constant uint poly_order=poly_flags&3;
constant bool poly_fused=(poly_flags&4)!=0;
struct PolyParams { uint count,terms,points_per_set; };
// a*x + b: fused (one rounding per component) or composed (mul then add; cmul then cadd).
inline Number<N> mac(Number<N> a,Number<N> x,Number<N> b){return poly_fused?limbforge::fma(a,x,b):add(mul(a,x),b);}
inline Complex<N> mac(Complex<N> a,Complex<N> x,Complex<N> b){return poly_fused?cfma(a,x,b):cadd(cmul(a,x),b);}
inline Number<N> zero_value(Number<N>){return zero<N>();}
inline Complex<N> zero_value(Complex<N>){return {zero<N>(),zero<N>()};}
template<class T> inline void horner(device const T* coeffs,device const T* points,device T* out,constant PolyParams& p,uint i){
    if(i>=p.count)return;
    T x=points[i],r0=zero_value(x),r1=r0,r2=r0;const ulong base=ulong(i/p.points_per_set)*p.terms;
    _Pragma("clang loop unroll(disable)") for(uint t=p.terms;t>0;--t){
        T c=coeffs[base+t-1];
        // Highest order first; one mac call site keeps the fused code instantiated once.
        _Pragma("clang loop unroll(disable)") for(int j=int(poly_order);j>=0;--j){
            T a=j==2?r2:(j==1?r1:r0),b=j==2?r1:(j==1?r0:c),v=mac(a,x,b);
            if(j==2)r2=v;else if(j==1)r1=v;else r0=v;
        }
    }
    const ulong o=ulong(i)*(poly_order+1);out[o]=r0;if(poly_order>=1)out[o+1]=r1;if(poly_order>=2)out[o+2]=r2;
}
kernel void poly_real(device const Number<N>* coeffs [[buffer(0)]],device const Number<N>* points [[buffer(1)]],device Number<N>* out [[buffer(2)]],
                      constant PolyParams& p [[buffer(3)]],uint i [[thread_position_in_grid]]){horner(coeffs,points,out,p,i);}
kernel void poly_complex(device const Complex<N>* coeffs [[buffer(0)]],device const Complex<N>* points [[buffer(1)]],device Complex<N>* out [[buffer(2)]],
                         constant PolyParams& p [[buffer(3)]],uint i [[thread_position_in_grid]]){horner(coeffs,points,out,p,i);}

// ---- Norms and status summaries ----
// nm_flags = complex | kind << 1. Kinds: 0 status, 1 inf, 2 componentwise max, 3 scan (statuses and the largest
// exponent, for norm2), 4 norm2 terms (add tree), 5 scaled residual.
constant uint nm_flags [[function_constant(1)]];
constant bool nm_complex=(nm_flags&1)!=0;
constant uint nm_kind=nm_flags>>1;
constant bool nm_tree=nm_kind==1||nm_kind==2||nm_kind==4||nm_kind==5;
constant bool nm_sum=nm_kind==4;
constant uint TG=128; // largest threadgroup of the tree passes (power of two)
struct NormParams { uint segments,length,blocks; };
inline Number<N> absolute(Number<N> a){if(a.sign)a.sign=1;return a;}
// RN(a^2) through the schoolbook product: core square() (symmetric on Metal at 384 bits) returned wrong values for
// every input inside these kernels at 384 bits (GPU only; docs/gpu-codegen.md). Both round the same exact square.
inline Number<N> sq(Number<N> a){return mul(a,a);}
// |a| * 2^-e (exact); flushed to zero more than LF_BITS+32 bits below 2^0.
inline Number<N> scaled(Number<N> a,int e){
    if(!a.sign)return a;long d=long(a.exponent)-e;if(d< -(LF_BITS+32))return zero<N>();
    a.exponent=int(d);a.sign=1;return a;
}
// Modulus key of a complex entry without status: q = s * 4^e with e the larger component exponent and
// s = RN(RN(a'^2) + RN(b'^2)) in [1, 8) for a' = a 2^-e, b' = b 2^-e (scaled). Returned as s with exponent
// exp(s) + 2e, which may lie outside the number range (|.| <= 2e9 + 2; keys are only compared and rooted).
inline Number<N> modulus_key(Number<N> a,Number<N> b){
    if(!a.sign&&!b.sign)return zero<N>();
    int e=!b.sign||(a.sign&&a.exponent>=b.exponent)?a.exponent:b.exponent;
    Number<N> s=add(sq(scaled(a,e)),sq(scaled(b,e)));s.exponent+=2*e;return s;
}
// RN(sqrt(k)) = r * 2^e: k = k' 4^e with k' in [1, 4), r = RN(sqrt(k')) in [1, 2) (sqrt commutes with the power of 4).
inline Number<N> key_root(Number<N> k,thread long& e){
    long K=k.exponent;e=K>=0?K/2:-((1-K)/2);k.exponent=int(K-2*e);return limbforge::sqrt(k);
}
inline Number<N> with_exponent(Number<N> r,long e){
    if(!r.sign||r.status)return r;long x=long(r.exponent)+e;
    if(x>1000000000||x< -1000000000)return zero<N>(r.status|exponent_overflow);r.exponent=int(x);return r;
}
// |r| / |s| rounded (real r: one division; complex: RN(RN(|r|) / |s|), |r| not range limited). 0/0 = 0.
inline Number<N> ratio(Number<N> a,Number<N> b,Number<N> s,thread word& st){
    bool rzero=!a.sign&&!b.sign;
    if(!s.sign){if(!rzero)st|=division_by_zero;return zero<N>();}
    if(rzero)return zero<N>();
    Number<N> q;
    if(!nm_complex)q=div(absolute(a),absolute(s));
    else{long e;Number<N> m=key_root(modulus_key(a,b),e),d=absolute(s);long f=d.exponent;d.exponent=0;q=with_exponent(div(m,d),e-f);}
    st|=q.status;return q;
}
// Adjacent-pair tree over the m valid entries of a block (the tree_sum order): add, or the larger key with the lower
// index on ties (an associative choice, so any tree gives the same result).
inline void reduce_block(threadgroup Number<N>* tk,threadgroup uint* ti,uint lid,uint T,uint m){
    for(uint w=1;w<T;w<<=1){
        if(lid%(2*w)==0&&lid+w<m){Number<N> a=tk[lid],b=tk[lid+w];
            if(nm_sum)tk[lid]=add(a,b);else if(magnitude_compare(b,a)>0){tk[lid]=b;ti[lid]=ti[lid+w];}}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
// First pass: per-entry key and status, summary atomics (one per SIMD group), block roots of the tree.
kernel void norm_first(device const Number<N>* x [[buffer(0)]],device const Number<N>* scale [[buffer(1)]],device Number<N>* keys [[buffer(2)]],
                       device uint* indices [[buffer(3)]],device atomic_uint* summary [[buffer(4)]],device atomic_int* emax [[buffer(5)]],
                       constant NormParams& p [[buffer(6)]],uint2 group [[threadgroup_position_in_grid]],uint lid [[thread_index_in_threadgroup]],
                       uint2 size [[threads_per_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    threadgroup Number<N> tk[TG];threadgroup uint ti[TG];
    const uint T=size.x,seg=group.y,idx=group.x*T+lid;const bool valid=idx<p.length;const ulong g=ulong(seg)*p.length+idx;
    Number<N> key=zero<N>();word st=0;int e=NO_EXPONENT;
    if(valid){
        Number<N> a=nm_complex?x[2*g]:x[g],b=nm_complex?x[2*g+1]:zero<N>();st=a.status|b.status;
        if(nm_kind==5){Number<N> s=scale[g];st|=s.status;if(!st)key=ratio(a,b,s,st);}
        else if(!st){
            if(nm_kind==1)key=nm_complex?modulus_key(a,b):absolute(a);
            else if(nm_kind==2)key=absolute(magnitude_compare(a,b)>=0?a:b);
            else if(nm_kind==3){if(a.sign)e=a.exponent;if(b.sign&&b.exponent>e)e=b.exponent;}
            else if(nm_kind==4){int E=atomic_load_explicit(&emax[seg],memory_order_relaxed);key=nm_complex?add(sq(scaled(a,E)),sq(scaled(b,E))):sq(scaled(a,E));}
        }
        if(st)key=zero<N>();
    }
    if(nm_kind!=4){
        uint fail=st?1u:0u,cnt=simd_sum(fail),any=simd_or(uint(st)),first=simd_min(fail?idx:NO_INDEX);
        if(lane==0&&cnt){atomic_fetch_or_explicit(&summary[4*seg],any,memory_order_relaxed);atomic_fetch_add_explicit(&summary[4*seg+1],cnt,memory_order_relaxed);
            atomic_fetch_min_explicit(&summary[4*seg+2],first,memory_order_relaxed);}
        if(nm_kind==3){int m=simd_max(e);if(lane==0&&m!=NO_EXPONENT)atomic_fetch_max_explicit(&emax[seg],m,memory_order_relaxed);}
    }
    if(!nm_tree)return;
    tk[lid]=key;ti[lid]=idx;threadgroup_barrier(mem_flags::mem_threadgroup);
    reduce_block(tk,ti,lid,T,min(T,p.length-group.x*T));
    if(lid==0){ulong o=ulong(seg)*p.blocks+group.x;keys[o]=tk[0];indices[o]=ti[0];}
}
// Further tree levels over the block roots of the previous pass (length = previous blocks per segment).
kernel void norm_combine(device const Number<N>* in_keys [[buffer(0)]],device const uint* in_indices [[buffer(1)]],device Number<N>* keys [[buffer(2)]],
                         device uint* indices [[buffer(3)]],constant NormParams& p [[buffer(6)]],uint2 group [[threadgroup_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]],uint2 size [[threads_per_threadgroup]]){
    threadgroup Number<N> tk[TG];threadgroup uint ti[TG];
    const uint T=size.x,seg=group.y,idx=group.x*T+lid;
    if(idx<p.length){ulong g=ulong(seg)*p.length+idx;tk[lid]=in_keys[g];ti[lid]=in_indices[g];}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    reduce_block(tk,ti,lid,T,min(T,p.length-group.x*T));
    if(lid==0){ulong o=ulong(seg)*p.blocks+group.x;keys[o]=tk[0];indices[o]=ti[0];}
}
// One thread per segment: the value from the tree root and the summary (info = status, failing, first_failing, index).
kernel void norm_finish(device const Number<N>* keys [[buffer(0)]],device const uint* indices [[buffer(1)]],device const uint* summary [[buffer(2)]],
                        device const int* emax [[buffer(3)]],device Number<N>* values [[buffer(4)]],device uint* info [[buffer(5)]],
                        constant NormParams& p [[buffer(6)]],uint s [[thread_position_in_grid]]){
    if(s>=p.segments)return;
    uint st=summary[4*s],fail=summary[4*s+1],first=summary[4*s+2],index=NO_INDEX;Number<N> v=zero<N>();
    if(nm_kind!=0&&p.length){
        if(st){v=zero<N>(st);index=first;}
        else{Number<N> k=keys[s];
            if(nm_kind==4){int E=emax[s];v=k.sign?with_exponent(limbforge::sqrt(k),E):k;}
            else{index=indices[s];if(nm_kind==1&&nm_complex){long e;Number<N> r=key_root(k,e);v=with_exponent(r,e);}else v=k;}
            st|=v.status;}
    }
    if(nm_kind!=0)values[s]=v;
    info[4*s]=st;info[4*s+1]=fail;info[4*s+2]=first;info[4*s+3]=index;
}
