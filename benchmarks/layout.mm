#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "reference.hpp"
#include "shader_source.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
using namespace limbforge;
const char* soa_source=R"METAL(
kernel void soa_arithmetic(device const uint* a [[buffer(0)]],device const uint* b [[buffer(1)]],
                           device uint* out [[buffer(2)]],constant Params& p [[buffer(3)]],uint i [[thread_position_in_grid]]){
    if(i>=p.count)return;Number<N> x,y,z;
    for(int j=0;j<N;++j){x.limb[j]=a[j*p.count+i];y.limb[j]=b[j*p.count+i];}
    x.exponent=as_type<int>(a[N*p.count+i]);x.sign=as_type<int>(a[(N+1)*p.count+i]);x.status=a[(N+2)*p.count+i];
    y.exponent=as_type<int>(b[N*p.count+i]);y.sign=as_type<int>(b[(N+1)*p.count+i]);y.status=b[(N+2)*p.count+i];
    switch(selected_operation){case 0:z=add(x,y);break;case 1:z=sub(x,y);break;case 2:z=mul(x,y);break;default:z=div(x,y);}
    for(int j=0;j<N;++j)out[j*p.count+i]=z.limb[j];
    out[N*p.count+i]=as_type<uint>(z.exponent);out[(N+1)*p.count+i]=as_type<uint>(z.sign);out[(N+2)*p.count+i]=z.status;
}
)METAL";
double median(std::vector<double> a){std::sort(a.begin(),a.end());return a[a.size()/2];}
void check(bool good,const char* message){if(!good)throw std::runtime_error(message);}
template<int Bits> void cases(id<MTLDevice> device,id<MTLCommandQueue> queue,std::size_t count){@autoreleasepool{
    using F=Float<Bits>;constexpr int fields=Bits/32+3;std::mt19937_64 rng(20261007+Bits);
    std::vector<F>a(count),b(count),expected(count),out(count);std::vector<word>soa[2];for(auto& v:soa)v.resize(fields*count);
    for(std::size_t i=0;i<count;++i){a[i]=reference::random_number<Bits>(rng,5);b[i]=reference::random_number<Bits>(rng,5);
        for(int j=0;j<fields;++j){word x,y;std::memcpy(&x,reinterpret_cast<const char*>(&a[i])+4*j,4);std::memcpy(&y,reinterpret_cast<const char*>(&b[i])+4*j,4);soa[0][j*count+i]=x;soa[1][j*count+i]=y;}}
    NSString* source=[NSString stringWithFormat:@"#define MP_BITS %d\n%s\n%s",Bits,limbforge_shader_source,soa_source];
    MTLCompileOptions* options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;options.languageVersion=MTLLanguageVersion3_1;
    NSError* error=nil;id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
    if(!library)throw std::runtime_error([[error localizedDescription] UTF8String]);
    id<MTLBuffer> buffers[2][3];for(int v=0;v<2;++v){for(int j=0;j<3;++j)buffers[v][j]=[device newBufferWithLength:count*sizeof(F) options:MTLResourceStorageModeShared];
        std::memcpy(buffers[v][0].contents,v?static_cast<const void*>(soa[0].data()):a.data(),count*sizeof(F));
        std::memcpy(buffers[v][1].contents,v?static_cast<const void*>(soa[1].data()):b.data(),count*sizeof(F));}
    const char* names[]={"add","sub","mul","div"};
    for(int op=0;op<4;++op){for(std::size_t i=0;i<count;++i)expected[i]=reference::real<Bits>(Operation(op),a[i],b[i]);
        id<MTLComputePipelineState> pipelines[2];for(int v=0;v<2;++v){MTLFunctionConstantValues* constants=[MTLFunctionConstantValues new];word selected=op;[constants setConstantValue:&selected type:MTLDataTypeUInt atIndex:0];
            id<MTLFunction> function=[library newFunctionWithName:v?@"soa_arithmetic":@"arithmetic" constantValues:constants error:&error];check(function!=nil,"layout function failed");
            pipelines[v]=[device newComputePipelineStateWithFunction:function error:&error];check(pipelines[v]!=nil,"layout pipeline failed");}
        std::vector<double>gpu[2],wall[2];
        for(int repeat=-2;repeat<9;++repeat)for(int k=0;k<2;++k){@autoreleasepool{
            int v=(repeat+2+k)%2;auto start=std::chrono::steady_clock::now();auto state=pipelines[v];
            id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];[encoder setComputePipelineState:state];
            for(int j=0;j<3;++j)[encoder setBuffer:buffers[v][j] offset:0 atIndex:j];
            word params[]={word(count),word(op),0,word(count),1,0,0,0,0};[encoder setBytes:params length:sizeof(params) atIndex:3];
            NSUInteger width=state.threadExecutionWidth,group=std::min(NSUInteger(128),state.maxTotalThreadsPerThreadgroup);group-=group%width;
            [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(group,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];
            check(command.status!=MTLCommandBufferStatusError,"layout execution failed");double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            if(repeat>=0){gpu[v].push_back(command.GPUEndTime-command.GPUStartTime);wall[v].push_back(elapsed);}
            if(!v)std::memcpy(out.data(),buffers[v][2].contents,count*sizeof(F));else{auto words=static_cast<const word*>(buffers[v][2].contents);for(std::size_t i=0;i<count;++i){word tmp[fields];for(int j=0;j<fields;++j)tmp[j]=words[j*count+i];std::memcpy(&out[i],tmp,sizeof(F));}}
            for(std::size_t i=0;i<count;++i)if(!reference::equal<Bits>(out[i],expected[i]))throw std::runtime_error("layout MPFR mismatch bits="+std::to_string(Bits)+" op="+names[op]+" soa="+std::to_string(v)+" case="+std::to_string(i));
        }}
        for(int v=0;v<2;++v)std::cout<<Bits<<','<<names[op]<<','<<count<<','<<(v?"soa":"aos")<<",9,"<<median(gpu[v])<<','<<median(wall[v])<<std::endl;
    }
}}
int main(){@autoreleasepool{try{auto device=MTLCreateSystemDefaultDevice();check(device!=nil,"no Metal device");auto queue=[device newCommandQueue];
    std::cerr<<[device.name UTF8String]<<"; interleaved AoS/SoA; uploads and layout conversion excluded; every dispatch checked against MPFR\n";
    std::cout<<std::setprecision(10)<<"bits,operation,count,layout,samples,gpu_median_s,wall_median_s\n";
    for(auto n:{257u,4096u,65536u}){cases<256>(device,queue,n);cases<384>(device,queue,n);cases<1024>(device,queue,n);}return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}}
