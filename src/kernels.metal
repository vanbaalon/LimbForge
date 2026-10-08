using namespace metal;
using namespace limbforge;
constant int N=MP_BITS/32;
constant uint selected_operation [[function_constant(0)]];
struct Params { uint count,operation,steps,weight_count,states_per_weight,b_stride,b_period,c_stride,c_period; };
// Broadcast operand index (i / stride) % period; stride 0/1 and period 0 leave i unchanged.
inline uint operand(uint i,uint stride,uint period){uint j=stride>1?i/stride:i;return period?j%period:j;}
kernel void arithmetic(device const Number<N>* a [[buffer(0)]],
                       device const Number<N>* b [[buffer(1)]],
                       device Number<N>* out [[buffer(2)]],
                       constant Params& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.count)return;
    Number<N> x=a[i],z;
    if(selected_operation==7)z=square(x);
    else if(selected_operation==8)z=limbforge::sqrt(x);
    else {Number<N> y=b[operand(i,p.b_stride,p.b_period)];switch(selected_operation){case 0:z=add(x,y);break;case 1:z=sub(x,y);break;case 2:z=mul(x,y);break;default:z=div(x,y);}}
    out[i]=z;
}
kernel void complex_arithmetic(device const Complex<N>* a [[buffer(0)]],
                               device const Complex<N>* b [[buffer(1)]],
                               device Complex<N>* out [[buffer(2)]],
                               constant Params& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.count)return;
    Complex<N> x=a[i],y=b[operand(i,p.b_stride,p.b_period)],z;
    switch(selected_operation){case 4:z=cadd(x,y);break;case 5:z=cmul(x,y);break;default:z=cdiv(x,y);}
    out[i]=z;
}
// Fused a*b +/- c with one rounding per real component; operands may alias out.
kernel void fused_arithmetic(device const Number<N>* a [[buffer(0)]],device const Number<N>* b [[buffer(1)]],device Number<N>* out [[buffer(2)]],
                             constant Params& p [[buffer(3)]],device const Number<N>* c [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.count)return;
    Number<N> x=a[i],y=b[operand(i,p.b_stride,p.b_period)],z=c[operand(i,p.c_stride,p.c_period)];out[i]=selected_operation==10?limbforge::fms(x,y,z):limbforge::fma(x,y,z);
}
kernel void complex_fused(device const Complex<N>* a [[buffer(0)]],device const Complex<N>* b [[buffer(1)]],device Complex<N>* out [[buffer(2)]],
                          constant Params& p [[buffer(3)]],device const Complex<N>* c [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.count)return;
    Complex<N> x=a[i],y=b[operand(i,p.b_stride,p.b_period)],z=c[operand(i,p.c_stride,p.c_period)];out[i]=selected_operation==12?cfms(x,y,z):cfma(x,y,z);
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

// Batched 4x4 complex LU with partial pivoting (plan D5). One thread per matrix. Pivot in column k:
// the row r >= k with the largest max(|re|,|im|) (exact comparison, ties to the lowest row); a zero
// pivot sets status k+1 and stops. Elimination: l = cdiv(a_rk, a_kk), a_rc -= l*a_kc; forward and
// back substitution with the same multiply-subtract and a final cdiv. Determinant: product of the
// pivots in order, negated for an odd number of swaps. Fused mode uses cfms for multiply-subtract.
constant uint b4_flags [[function_constant(4)]];
constant bool b4_fused=(b4_flags&1)!=0,b4_det=(b4_flags&2)!=0,b4_inverse=(b4_flags&4)!=0;
struct Batch4Params { uint count,rhs; };
inline Complex<N> b4_minus(Complex<N> c,Complex<N> a,Complex<N> b){ // c - a*b
    if(b4_fused){Complex<N> r=cfms(a,b,c);return {negate(r.re),negate(r.im)};}
    Complex<N> q=cmul(a,b);return {sub(c.re,q.re),sub(c.im,q.im)};
}
inline Number<N> b4_mag(Complex<N> z){return magnitude_compare(z.re,z.im)>=0?z.re:z.im;}
kernel void batched4(device const Complex<N>* A [[buffer(0)]],device const Complex<N>* B [[buffer(1)]],device Complex<N>* X [[buffer(2)]],
                     constant Batch4Params& p [[buffer(3)]],device Complex<N>* det [[buffer(4)]],device uint* status [[buffer(5)]],uint m [[thread_position_in_grid]]){
    if(m>=p.count)return;
    Complex<N> a[16];uint perm[4]={0,1,2,3};uint st=0,swaps=0;
    for(uint i=0;i<16;++i)a[i]=A[ulong(m)*16+i];
    _Pragma("clang loop unroll(disable)") for(uint k=0;k<4&&!st;++k){
        uint piv=k;for(uint r=k+1;r<4;++r){Number<N> x=b4_mag(a[r*4+k]),y=b4_mag(a[piv*4+k]);if(magnitude_compare(x,y)>0)piv=r;}
        if(!a[piv*4+k].re.sign&&!a[piv*4+k].im.sign){st=k+1;break;}
        if(piv!=k){for(uint c=0;c<4;++c){Complex<N> t=a[k*4+c];a[k*4+c]=a[piv*4+c];a[piv*4+c]=t;}uint t=perm[k];perm[k]=perm[piv];perm[piv]=t;++swaps;}
        _Pragma("clang loop unroll(disable)") for(uint r=k+1;r<4;++r){
            Complex<N> l=cdiv(a[r*4+k],a[k*4+k]);a[r*4+k]=l;
            _Pragma("clang loop unroll(disable)") for(uint c=k+1;c<4;++c)a[r*4+c]=b4_minus(a[r*4+c],l,a[k*4+c]);
        }
    }
    status[m]=st;
    Complex<N> z={zero<N>(),zero<N>()};
    if(b4_det){Complex<N> d=z;if(!st){d=a[0];for(uint k=1;k<4;++k)d=cmul(d,a[k*5]);if(swaps&1)d={negate(d.re),negate(d.im)};}det[m]=d;}
    uint R=b4_inverse?4:p.rhs;
    _Pragma("clang loop unroll(disable)") for(uint j=0;j<R;++j){
        Complex<N> y[4];
        for(uint i=0;i<4;++i){if(b4_inverse){y[i]=z;if(perm[i]==j){y[i].re.limb[N-1]=0x80000000u;y[i].re.sign=1;}}else y[i]=B[(ulong(m)*4+perm[i])*R+j];}
        if(st){Complex<N> e={zero<N>(division_by_zero),zero<N>(division_by_zero)};for(uint i=0;i<4;++i)X[(ulong(m)*4+i)*R+j]=e;continue;}
        _Pragma("clang loop unroll(disable)") for(uint i=1;i<4;++i)for(uint t=0;t<i;++t)y[i]=b4_minus(y[i],a[i*4+t],y[t]);
        _Pragma("clang loop unroll(disable)") for(int i=3;i>=0;--i){for(uint t=uint(i)+1;t<4;++t)y[i]=b4_minus(y[i],a[i*4+t],y[t]);y[i]=cdiv(y[i],a[i*5]);}
        for(uint i=0;i<4;++i)X[(ulong(m)*4+i)*R+j]=y[i];
    }
}
