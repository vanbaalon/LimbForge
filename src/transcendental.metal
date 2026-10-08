// Element-wise transcendental kernels (src/transcendental.mm; docs/numerics.md, "Transcendental functions"). A library is
// compiled per (LF_BITS, LF_W): the first pass at a GPU-validated LF_W >= N+2 (gpu_words) and each GPU retry rung at its
// validated width (rung_words in src/transcendental.mm). Function constants: LF_OP (= limbforge::Function) selects the
// function; LF_RUNG makes the kernel a retry rung over the list of the previous level.
// Levels: 0 = first pass (element = thread), 1..3 = retry rungs (element = entry of the previous level's list, whose length
// is counters[level-1]). Each level writes the certified elements and appends the undecided ones to its own list (counter
// counters[level]). The last GPU level also writes the RN value of its approximation for them (provisional: the host final
// step replaces it) and saves their operands by list slot in `saved`, so the host needs neither the inputs nor a snapshot.
// Rungs are dispatched indirectly: lf_prepare turns the previous level's counter into the threadgroup count, so a rung
// with nothing to do launches no threads.
using namespace metal;
using namespace limbforge;
using namespace limbforge::transcendental;
constant int N=LF_BITS/32;
constant int W=LF_W;
constant int LF_OP [[function_constant(0)]];
constant bool LF_RUNG [[function_constant(1)]];
// count: elements of the call; force: levels below it treat every element as undecided (test hook); last: last GPU level.
struct Level { uint count,level,force,last; };
inline bool element(uint tid,device atomic_uint* counters,device const uint* list,constant Level& p,thread uint& i){
    if(LF_RUNG){uint n=atomic_load_explicit(counters+(p.level-1),memory_order_relaxed);if(tid>=n)return false;i=list[tid];}
    else{if(tid>=p.count)return false;i=tid;}
    return true;
}
inline uint push(device atomic_uint* counters,uint level,device uint* next,uint i){uint s=atomic_fetch_add_explicit(counters+level,1u,memory_order_relaxed);next[s]=i;return s;}
// Indirect dispatch arguments (threadgroups x, y, z) of rung p.x+1 from counters[p.x], p.y threads per threadgroup.
kernel void lf_prepare(device atomic_uint* counters[[buffer(0)]],device uint* args[[buffer(1)]],constant uint2& p[[buffer(2)]],uint tid[[thread_position_in_grid]]){
    if(tid)return;
    uint n=atomic_load_explicit(counters+p.x,memory_order_relaxed);args[0]=(n+p.y-1)/p.y;args[1]=1;args[2]=1;
}
// exp, expm1, log, log1p, sin, cos (LF_OP 0-5).
kernel void lf_unary(device const Number<N>* x[[buffer(0)]],device Number<N>* out[[buffer(2)]],device const Tables<W>& t[[buffer(3)]],
                     device const word* bits[[buffer(4)]],device atomic_uint* counters[[buffer(5)]],device uint* next[[buffer(6)]],
                     constant Level& p[[buffer(7)]],device Number<N>* saved[[buffer(8)]],device const uint* list[[buffer(9)]],
                     uint tid[[thread_position_in_grid]]){
    uint i;if(!element(tid,counters,list,p,i))return;
    Number<N> in=x[i];Number<W> v=widen<W>(in);Approx<W> a;
    if(LF_OP==0)a=exp_approx(v,t,false);
    else if(LF_OP==1)a=exp_approx(v,t,true);
    else if(LF_OP==2)a=log_approx(v,t);
    else if(LF_OP==3)a=log1p_approx(v,t);
    else a=sin_approx(v,bits,t,LF_OP==5);
    Number<N> r;if(p.level>=p.force&&certify<N,W>(a.y,a.shift,a.err,r)){out[i]=r;return;}
    uint s=push(counters,p.level,next,i);
    if(p.last){saved[s]=in;certify<N,W>(a.y,a.shift,-1.f,r);out[i]=r;}
}
// atan2(y = a[i], x = b[i]); saved operands (y, x) at 2s, 2s+1.
kernel void lf_atan2(device const Number<N>* y[[buffer(0)]],device const Number<N>* x[[buffer(1)]],device Number<N>* out[[buffer(2)]],
                     device const Tables<W>& t[[buffer(3)]],device atomic_uint* counters[[buffer(5)]],device uint* next[[buffer(6)]],
                     constant Level& p[[buffer(7)]],device Number<N>* saved[[buffer(8)]],device const uint* list[[buffer(9)]],
                     uint tid[[thread_position_in_grid]]){
    uint i;if(!element(tid,counters,list,p,i))return;
    Number<N> yi=y[i],xi=x[i];Approx<W> a=atan2_approx(widen<W>(yi),widen<W>(xi),t);
    Number<N> r;if(p.level>=p.force&&certify<N,W>(a.y,a.shift,a.err,r)){out[i]=r;return;}
    uint s=push(counters,p.level,next,i);
    if(p.last){saved[2*s]=yi;saved[2*s+1]=xi;certify<N,W>(a.y,a.shift,-1.f,r);out[i]=r;}
}
// Complex exp, log (LF_OP 7, 8).
kernel void lf_complex(device const Complex<N>* z[[buffer(0)]],device Complex<N>* out[[buffer(2)]],device const Tables<W>& t[[buffer(3)]],
                       device const word* bits[[buffer(4)]],device atomic_uint* counters[[buffer(5)]],device uint* next[[buffer(6)]],
                       constant Level& p[[buffer(7)]],device Complex<N>* saved[[buffer(8)]],device const uint* list[[buffer(9)]],
                       uint tid[[thread_position_in_grid]]){
    uint i;if(!element(tid,counters,list,p,i))return;
    Complex<N> in=z[i];Approx<W> re,im;
    if(LF_OP==7)cexp_approx(widen<W>(in.re),widen<W>(in.im),bits,t,re,im);else clog_approx(widen<W>(in.re),widen<W>(in.im),t,re,im);
    Complex<N> r;
    if(p.level>=p.force){bool a=certify<N,W>(re.y,re.shift,re.err,r.re),b=certify<N,W>(im.y,im.shift,im.err,r.im);if(a&&b){out[i]=r;return;}}
    uint s=push(counters,p.level,next,i);
    if(p.last){saved[s]=in;certify<N,W>(re.y,re.shift,-1.f,r.re);certify<N,W>(im.y,im.shift,-1.f,r.im);out[i]=r;}
}
// z^k, k[i * k_stride].
kernel void lf_powi(device const Complex<N>* z[[buffer(0)]],device Complex<N>* out[[buffer(2)]],constant uint& count[[buffer(7)]],
                    device const int* k[[buffer(8)]],constant uint& k_stride[[buffer(9)]],uint i[[thread_position_in_grid]]){
    if(i>=count)return;
    out[i]=cpowi<N>(z[i],k[i*k_stride]);
}
