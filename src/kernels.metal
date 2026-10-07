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
kernel void tree_sum_real(device const Number<N>* input [[buffer(0)]],
                          device Number<N>* out [[buffer(2)]],
                          constant Params& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(!p.count){if(!i)out[0]=zero<N>();return;}
    uint next=p.count/2+p.count%2;if(i>=next)return;
    uint j=2*i;Number<N> x=input[j];
    if(j+1<p.count){Number<N> y=input[j+1];x=add(x,y);}out[i]=x;
}
kernel void tree_sum_complex(device const Complex<N>* input [[buffer(0)]],
                             device Complex<N>* out [[buffer(2)]],
                             constant Params& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(!p.count){if(!i)out[0]={zero<N>(),zero<N>()};return;}
    uint next=p.count/2+p.count%2;if(i>=next)return;
    uint j=2*i;Complex<N> x=input[j];
    if(j+1<p.count){Complex<N> y=input[j+1];x=cadd(x,y);}out[i]=x;
}
