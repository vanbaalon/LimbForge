using namespace metal;
namespace xp=limbforge_batched_private;
constant int N=LF_BITS/32;
constant int P=xp::exact_sum_words(xp::scratch(2*N+1));
constant int F=xp::exact_sum_words(xp::scratch(2*N+8));
constant bool is_complex [[function_constant(0)]];
struct Params {ulong count,sa,sb,sc;uint accumulate,negative;};
using Local=xp::Number<N>;
inline Local load(device const limbforge::Number<N>& x){Local r=xp::zero<N>();
    for(int j=0;j<N;++j)r.limb[j]=x.limb[j];r.exponent=x.exponent;r.sign=x.sign;r.status=x.status;return r;}
inline void store(device limbforge::Number<N>& out,thread const Local& x){
    for(int j=0;j<N;++j)out.limb[j]=x.limb[j];out.exponent=x.exponent;out.sign=x.sign;out.status=x.status;}
// One component per thread, uniform component selection across a SIMD group.
// Two passes avoid retaining eight wide products and their live significands.
kernel void exact_gemm4(device const limbforge::Number<N>* A [[buffer(0)]],device const limbforge::Number<N>* B [[buffer(1)]],
    device limbforge::Number<N>* C [[buffer(2)]],constant Params& p [[buffer(3)]],device uint* fallback [[buffer(4)]],uint2 gid [[thread_position_in_grid]]){
    ulong entry=gid.x,part=gid.y,t=entry/16,cell=entry%16,row=cell/4,col=cell%4;
    if(t>=p.count)return;uint components=is_complex?2:1,terms=is_complex?8:4;
    ulong o=components*(t*p.sc+cell)+part,flag=components*entry+part;
    Local old=p.accumulate?load(C[o]):xp::zero<N>();uint status=old.status;bool any=old.sign!=0;
    long lo=any?long(old.exponent)-(32*N-1):0,hi=any?old.exponent:0;
    _Pragma("clang loop unroll(disable)") for(uint q=0;q<terms;++q){uint k=is_complex?q/2:q,sub=is_complex?q%2:0;
        Local a=load(A[components*(t*p.sa+row*4+k)+sub]);
        Local b=load(B[components*(t*p.sb+k*4+col)+(is_complex?(sub?1-part:part):0)]);
        status|=a.status|b.status;if(!a.sign||!b.sign)continue;
        long l=long(a.exponent)+b.exponent-2*(32*N-1),h=long(a.exponent)+b.exponent+1;
        if(!any||l<lo)lo=l;if(!any||h>hi)hi=h;any=true;
    }
    fallback[flag]=0;
    if(status){store(C[o],xp::zero<N>(status));return;}
    if(!any){store(C[o],xp::zero<N>());return;}
    // At most nine terms: reserve four carry bits plus the sign bit. No sticky
    // approximation is used; an out-of-window component is recomputed exactly.
    if(hi+5-lo>=long(32*F)){fallback[flag]=1;return;}
    xp::word acc[F];for(int j=0;j<F;++j)acc[j]=0;
    // Include C through the same single exact-product/accumulate site (C * 1).
    _Pragma("clang loop unroll(disable)") for(uint q=0;q<terms+uint(p.accumulate);++q){
        Local a=xp::zero<N>(),b=xp::zero<N>();
        if(q==terms){a=old;b.sign=1;b.limb[N-1]=0x80000000u;}
        else {uint k=is_complex?q/2:q,sub=is_complex?q%2:0;
            a=load(A[components*(t*p.sa+row*4+k)+sub]);b=load(B[components*(t*p.sb+k*4+col)+(is_complex?(sub?1-part:part):0)]);
            if(is_complex&&!part&&sub)a=xp::negate(a);}
        xp::Exact<P> product;
        if(q==terms)product=xp::exact_number<P>(old);else product=xp::exact_product<N,P>(a,b);
        xp::window_accumulate(acc,product,lo);
    }
    int sign=1;if(acc[F-1]>>31){sign=-1;xp::dword carry=1;
        for(int j=0;j<F;++j){xp::dword v=xp::dword(xp::word(~acc[j]))+carry;acc[j]=xp::word(v);carry=v>>32;}}
    bool nonzero=false;for(int j=0;j<F;++j)nonzero|=acc[j]!=0;
    Local result=nonzero?xp::pack<N>(acc,lo,sign,xp::ok):xp::zero<N>();if(p.negative)result=xp::negate(result);store(C[o],result);
}
