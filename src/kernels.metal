using namespace metal;
using namespace limbforge;
constant int N=MP_BITS/32;
constant uint selected_operation [[function_constant(0)]];
struct Params { uint count,operation,steps,weight_count,states_per_weight; };
kernel void arithmetic(device const Number<N>* a [[buffer(0)]],
                       device const Number<N>* b [[buffer(1)]],
                       device Number<N>* out [[buffer(2)]],
                       constant Params& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.count)return;
    Number<N> x=a[i],z;
    if(selected_operation==7)z=square(x);
    else if(selected_operation==8)z=limbforge::sqrt(x);
    else {Number<N> y=b[i];switch(selected_operation){case 0:z=add(x,y);break;case 1:z=sub(x,y);break;case 2:z=mul(x,y);break;default:z=div(x,y);}}
    out[i]=z;
}
kernel void complex_arithmetic(device const Complex<N>* a [[buffer(0)]],
                               device const Complex<N>* b [[buffer(1)]],
                               device Complex<N>* out [[buffer(2)]],
                               constant Params& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.count)return;
    Complex<N> x=a[i],y=b[i],z;
    switch(selected_operation){case 4:z=cadd(x,y);break;case 5:z=cmul(x,y);break;default:z=cdiv(x,y);}
    out[i]=z;
}
// Fused a*b +/- c with one rounding per real component; operands may alias out.
kernel void fused_arithmetic(device const Number<N>* a [[buffer(0)]],device const Number<N>* b [[buffer(1)]],device Number<N>* out [[buffer(2)]],
                             constant Params& p [[buffer(3)]],device const Number<N>* c [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.count)return;
    Number<N> x=a[i],y=b[i],z=c[i];out[i]=selected_operation==10?limbforge::fms(x,y,z):limbforge::fma(x,y,z);
}
kernel void complex_fused(device const Complex<N>* a [[buffer(0)]],device const Complex<N>* b [[buffer(1)]],device Complex<N>* out [[buffer(2)]],
                          constant Params& p [[buffer(3)]],device const Complex<N>* c [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.count)return;
    Complex<N> x=a[i],y=b[i],z=c[i];out[i]=selected_operation==12?cfms(x,y,z):cfma(x,y,z);
}
kernel void recurrence(device const Complex<N>* seeds [[buffer(0)]],
                       device const Complex<N>* weights [[buffer(1)]],
                       device Complex<N>* out [[buffer(2)]],
                       constant Params& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.count)return;
    // Named states avoid dynamic indexing of a large private array (scratch
    // traffic on Apple GPUs). The compiler can allocate each state separately.
    Complex<N> q0=seeds[i],q1=seeds[p.count+i],q2=seeds[2*p.count+i],q3=seeds[3*p.count+i];
    uint wi=i/p.states_per_weight;
    for(uint step=0;step<p.steps;++step){
        Complex<N> sum={zero<N>(),zero<N>()};
        uint base=step*4*p.weight_count+wi;
        sum=cadd(sum,cmul(weights[base],q0));
        sum=cadd(sum,cmul(weights[base+p.weight_count],q1));
        sum=cadd(sum,cmul(weights[base+2*p.weight_count],q2));
        sum=cadd(sum,cmul(weights[base+3*p.weight_count],q3));
        q0=q1;q1=q2;q2=q3;q3=sum;
    }
    out[i]=q3;
}

// A fixed adjacent-pair tree. Carrying an odd tail does not round it again.
kernel void global_tree_sum_real(device const Number<N>* input [[buffer(0)]],
                          device Number<N>* out [[buffer(2)]],
                          constant Params& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(!p.count){if(!i)out[0]=zero<N>();return;}
    uint next=p.count/2+p.count%2;if(i>=next)return;
    uint j=2*i;Number<N> x=input[j];
    if(j+1<p.count){Number<N> y=input[j+1];x=add(x,y);}out[i]=x;
}
kernel void global_tree_sum_complex(device const Complex<N>* input [[buffer(0)]],
                             device Complex<N>* out [[buffer(2)]],
                             constant Params& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(!p.count){if(!i)out[0]={zero<N>(),zero<N>()};return;}
    uint next=p.count/2+p.count%2;if(i>=next)return;
    uint j=2*i;Complex<N> x=input[j];
    if(j+1<p.count){Complex<N> y=input[j+1];x=cadd(x,y);}out[i]=x;
}

// Complete power-of-two groups preserve the global adjacent-pair tree.
kernel void tree_sum_real(device const Number<N>* input [[buffer(0)]],
                          device Number<N>* out [[buffer(2)]],
                          constant Params& p [[buffer(3)]],uint tid [[thread_index_in_threadgroup]],
                          uint3 gid [[threadgroup_position_in_grid]],uint3 size [[threads_per_threadgroup]]) {
    threadgroup Number<N> partial[128];
    ulong j=ulong(gid.x)*2*size.x+2*tid;
    Number<N> x=zero<N>();
    if(j<p.count){x=input[j];if(j+1<p.count){Number<N> y=input[j+1];x=add(x,y);}}
    partial[tid]=x;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=1;stride<size.x;stride*=2){
        // Each active thread owns both slots; its partner is inactive at this level.
        if(!(tid&(2*stride-1))){Number<N> a=partial[tid],b=partial[tid+stride];partial[tid]=add(a,b);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(!tid){Number<N> result=partial[0];out[gid.x]=result;}
}

// Complete power-of-two groups preserve the global adjacent-pair tree.
kernel void tree_sum_complex(device const Complex<N>* input [[buffer(0)]],
                          device Complex<N>* out [[buffer(2)]],
                          constant Params& p [[buffer(3)]],uint tid [[thread_index_in_threadgroup]],
                          uint3 gid [[threadgroup_position_in_grid]],uint3 size [[threads_per_threadgroup]]) {
    threadgroup Complex<N> partial[64];
    ulong j=ulong(gid.x)*2*size.x+2*tid;
    Complex<N> x={zero<N>(),zero<N>()};
    if(j<p.count){x=input[j];if(j+1<p.count){Complex<N> y=input[j+1];x=cadd(x,y);}}
    partial[tid]=x;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=1;stride<size.x;stride*=2){
        // Each active thread owns both slots; its partner is inactive at this level.
        if(!(tid&(2*stride-1))){Complex<N> a=partial[tid],b=partial[tid+stride];partial[tid]=cadd(a,b);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(!tid){Complex<N> result=partial[0];out[gid.x]=result;}
}

// Batched four-component vector recurrence (plan D4). One thread per lane. Contract
// (docs/numerics.md): dot(a,v) = cfma chain over components 0..3 starting from cfma(a0,v0,0);
// rank-1 step v_i <- cfma(p_i, dot(q,v), v_i); matrix step v_i <- dot(M_i,v); the affine source is
// added last with cadd. Loops stay rolled so each exact cfma is instantiated once: fully inlined
// copies made shader compilation take tens of minutes.
constant uint vr_flags [[function_constant(2)]];
constant bool vr_affine=(vr_flags&1)!=0,vr_matrix=(vr_flags&2)!=0,vr_all=(vr_flags&4)!=0,vr_reverse=(vr_flags&8)!=0,vr_tangent=(vr_flags&16)!=0,vr_fused=(vr_flags&32)!=0;
struct VectorParams { uint lanes,steps,lanes_per_weight,lanes_per_base,lanes_per_tangent; };
inline Complex<N> czero(){return {zero<N>(),zero<N>()};}
// c + a*b: fused (one rounding per component) or composed (cadd(c, cmul(a,b))).
inline Complex<N> mac(Complex<N> a,Complex<N> b,Complex<N> c){return vr_fused?cfma(a,b,c):cadd(c,cmul(a,b));}
// sum_j a[j*stride] * v[j] as a cfma chain; v is a 4-element private array.
inline Complex<N> dot4(device const Complex<N>* a,ulong stride,thread const Complex<N> (&v)[4]){
    Complex<N> s=czero();
    _Pragma("clang loop unroll(disable)") for(uint j=0;j<4;++j){Complex<N> x=a[j*stride],y=v[j];s=mac(x,y,s);}
    return s;
}
kernel void vector_recurrence(device const Complex<N>* start [[buffer(0)]],device const Complex<N>* pw [[buffer(1)]],device Complex<N>* out [[buffer(2)]],
                              constant VectorParams& p [[buffer(3)]],device const Complex<N>* qw [[buffer(4)]],device const Complex<N>* r [[buffer(5)]],
                              device const Complex<N>* base [[buffer(6)]],device const Complex<N>* dp [[buffer(7)]],device const Complex<N>* dq [[buffer(8)]],
                              uint lane [[thread_position_in_grid]]) {
    if(lane>=p.lanes)return;
    const ulong L=p.lanes,G=p.lanes/p.lanes_per_weight,g=lane/p.lanes_per_weight;
    const ulong B=p.lanes/p.lanes_per_base,b=lane/p.lanes_per_base,T=p.lanes/p.lanes_per_tangent,h=lane/p.lanes_per_tangent;
    Complex<N> v[4],n[4],bs[4];
    for(uint j=0;j<4;++j){v[j]=start[j*L+lane];if(vr_all)out[j*L+lane]=v[j];}
    _Pragma("clang loop unroll(disable)") for(uint t=0;t<p.steps;++t){
        ulong k=vr_reverse?p.steps-1-t:t;
        if(vr_matrix){
            _Pragma("clang loop unroll(disable)") for(uint i=0;i<4;++i)n[i]=dot4(pw+(k*16+i*4)*G+g,G,v);
            for(uint i=0;i<4;++i)v[i]=n[i];
        }else{
            Complex<N> s=dot4(qw+k*4*G+g,G,v),sb,tq;
            if(vr_tangent){
                // v holds the tangent; base[t] is the base state before this step.
                for(uint j=0;j<4;++j)bs[j]=base[(ulong(t)*4+j)*B+b];
                sb=dot4(qw+k*4*G+g,G,bs);tq=dot4(dq+k*4*T+h,T,bs);
            }
            _Pragma("clang loop unroll(disable)") for(uint i=0;i<4;++i){
                Complex<N> pi=pw[(k*4+i)*G+g],x=mac(pi,s,v[i]);
                if(vr_tangent){Complex<N> d=dp[(k*4+i)*T+h],e=mac(d,sb,czero());x=cadd(x,mac(pi,tq,e));}
                v[i]=x;
            }
        }
        if(vr_affine)for(uint i=0;i<4;++i)v[i]=cadd(v[i],r[(k*4+i)*L+lane]);
        if(vr_all)for(uint j=0;j<4;++j)out[(ulong(t+1)*4+j)*L+lane]=v[j];
    }
    if(!vr_all)for(uint j=0;j<4;++j)out[j*L+lane]=v[j];
}

// Segmented dot products (plan D3): out[i] = sum_k a[i*K+k] * b[...] as a sequential fused chain,
// s = fma(a_0,b_0,0), s = fma(a_k,b_k,s) for k = 1..K-1 (cfma for complex), so results are bitwise
// reproducible. b is per segment ([i*K+k]) or one shared table ([k]).
constant uint sd_flags [[function_constant(3)]];
constant bool sd_shared=(sd_flags&1)!=0;
struct DotParams { uint segments,length; };
kernel void segmented_dot_real(device const Number<N>* a [[buffer(0)]],device const Number<N>* b [[buffer(1)]],device Number<N>* out [[buffer(2)]],
                               constant DotParams& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.segments)return;Number<N> s=zero<N>();ulong base=ulong(i)*p.length;
    _Pragma("clang loop unroll(disable)") for(uint k=0;k<p.length;++k){Number<N> x=a[base+k],y=b[sd_shared?k:base+k];s=limbforge::fma(x,y,s);}
    out[i]=s;
}
kernel void segmented_dot_complex(device const Complex<N>* a [[buffer(0)]],device const Complex<N>* b [[buffer(1)]],device Complex<N>* out [[buffer(2)]],
                                  constant DotParams& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.segments)return;Complex<N> s={zero<N>(),zero<N>()};ulong base=ulong(i)*p.length;
    _Pragma("clang loop unroll(disable)") for(uint k=0;k<p.length;++k){Complex<N> x=a[base+k],y=b[sd_shared?k:base+k];s=cfma(x,y,s);}
    out[i]=s;
}
