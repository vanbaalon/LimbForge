// Element-wise transcendental kernels (src/transcendental.mm; docs/numerics.md, "Transcendental functions"). Compiled
// per LF_BITS with LF_W >= N+2 working words (a GPU-validated width, gpu_words in src/transcendental.mm); the function
// constant LF_OP (= limbforge::Function) selects the function.
// Undecided elements are appended to the retry list (buffer 6, counter in buffer 5) and resolved on the host.
using namespace metal;
using namespace limbforge;
using namespace limbforge::transcendental;
constant int N=LF_BITS/32;
constant int W=LF_W;
constant int LF_OP [[function_constant(0)]];
inline void push(device atomic_uint* retry,device uint* list,uint i){uint s=atomic_fetch_add_explicit(retry,1u,memory_order_relaxed);list[s]=i;}
// exp, expm1, log, log1p, sin, cos (LF_OP 0-5).
kernel void lf_unary(device const Number<N>* x[[buffer(0)]],device Number<N>* out[[buffer(2)]],device const Tables<W>& t[[buffer(3)]],
                     device const word* bits[[buffer(4)]],device atomic_uint* retry[[buffer(5)]],device uint* list[[buffer(6)]],
                     constant uint& count[[buffer(7)]],uint i[[thread_position_in_grid]]){
    if(i>=count)return;
    Number<N> in=x[i];Number<W> v=widen<W>(in);Approx<W> a;
    if(LF_OP==0)a=exp_approx(v,t,false);
    else if(LF_OP==1)a=exp_approx(v,t,true);
    else if(LF_OP==2)a=log_approx(v,t);
    else if(LF_OP==3)a=log1p_approx(v,t);
    else a=sin_approx(v,bits,t,LF_OP==5);
    Number<N> r;if(!certify<N,W>(a.y,a.shift,a.err,r))push(retry,list,i);
    out[i]=r;
}
// atan2(y = a[i], x = b[i]).
kernel void lf_atan2(device const Number<N>* y[[buffer(0)]],device const Number<N>* x[[buffer(1)]],device Number<N>* out[[buffer(2)]],
                     device const Tables<W>& t[[buffer(3)]],device atomic_uint* retry[[buffer(5)]],device uint* list[[buffer(6)]],
                     constant uint& count[[buffer(7)]],uint i[[thread_position_in_grid]]){
    if(i>=count)return;
    Number<N> yi=y[i],xi=x[i];Approx<W> a=atan2_approx(widen<W>(yi),widen<W>(xi),t);
    Number<N> r;if(!certify<N,W>(a.y,a.shift,a.err,r))push(retry,list,i);
    out[i]=r;
}
// Complex exp, log (LF_OP 7, 8).
kernel void lf_complex(device const Complex<N>* z[[buffer(0)]],device Complex<N>* out[[buffer(2)]],device const Tables<W>& t[[buffer(3)]],
                       device const word* bits[[buffer(4)]],device atomic_uint* retry[[buffer(5)]],device uint* list[[buffer(6)]],
                       constant uint& count[[buffer(7)]],uint i[[thread_position_in_grid]]){
    if(i>=count)return;
    Complex<N> in=z[i];Approx<W> re,im;
    if(LF_OP==7)cexp_approx(widen<W>(in.re),widen<W>(in.im),bits,t,re,im);else clog_approx(widen<W>(in.re),widen<W>(in.im),t,re,im);
    Complex<N> r;bool a=certify<N,W>(re.y,re.shift,re.err,r.re),b=certify<N,W>(im.y,im.shift,im.err,r.im);
    if(!(a&&b))push(retry,list,i);
    out[i]=r;
}
// z^k, k[i * k_stride].
kernel void lf_powi(device const Complex<N>* z[[buffer(0)]],device Complex<N>* out[[buffer(2)]],constant uint& count[[buffer(7)]],
                    device const int* k[[buffer(8)]],constant uint& k_stride[[buffer(9)]],uint i[[thread_position_in_grid]]){
    if(i>=count)return;
    out[i]=cpowi<N>(z[i],k[i*k_stride]);
}
