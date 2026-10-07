// L1 capability probe (plan section 8): can Metal TensorOps matmul2d supply exact integer products
// for a residue-number (Ozaki-II style) GEMM? Checks int8/uint8 -> int32 exactness against 64-bit
// CPU sums, classifies int32 overflow (wrap or saturate), and measures large-GEMM throughput.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
const char* source=R"METAL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
template<class T> void product(device T* A,device T* B,device int32_t* C,constant uint3& mnk,uint2 tgid){
    int M=int(mnk.x),N=int(mnk.y),K=int(mnk.z);
    auto tA=tensor<device T,dextents<int32_t,2>,tensor_inline>(A,dextents<int32_t,2>(K,M));
    auto tB=tensor<device T,dextents<int32_t,2>,tensor_inline>(B,dextents<int32_t,2>(N,K));
    auto tC=tensor<device int32_t,dextents<int32_t,2>,tensor_inline>(C,dextents<int32_t,2>(N,M));
    constexpr auto d=matmul2d_descriptor(64,32,static_cast<int>(dynamic_extent),false,false,false);
    matmul2d<d,execution_simdgroups<4>> op;
    auto mA=tA.slice(0,int(tgid.y)*64);auto mB=tB.slice(int(tgid.x)*32,0);auto mC=tC.slice(int(tgid.x)*32,int(tgid.y)*64);
    op.run(mA,mB,mC);
}
kernel void product_s8(device int8_t* A [[buffer(0)]],device int8_t* B [[buffer(1)]],device int32_t* C [[buffer(2)]],
                       constant uint3& mnk [[buffer(3)]],uint2 tgid [[threadgroup_position_in_grid]]){product(A,B,C,mnk,tgid);}
kernel void product_u8(device uint8_t* A [[buffer(0)]],device uint8_t* B [[buffer(1)]],device int32_t* C [[buffer(2)]],
                       constant uint3& mnk [[buffer(3)]],uint2 tgid [[threadgroup_position_in_grid]]){product(A,B,C,mnk,tgid);}
)METAL";
id<MTLDevice> device;id<MTLCommandQueue> queue;id<MTLComputePipelineState> pipelines[2];
std::string text(NSError* e){return e?std::string([[e localizedDescription] UTF8String]):"unknown";}
// C = A(MxK) * B(KxN), row-major; returns device seconds per repetition.
double gemm(bool is_unsigned,const void* a,const void* b,std::vector<int32_t>& c,int M,int N,int K,int repeats=1){@autoreleasepool{
    id<MTLBuffer> A=[device newBufferWithBytes:a length:std::size_t(M)*K options:MTLResourceStorageModeShared];
    id<MTLBuffer> B=[device newBufferWithBytes:b length:std::size_t(K)*N options:MTLResourceStorageModeShared];
    id<MTLBuffer> C=[device newBufferWithLength:std::size_t(M)*N*4 options:MTLResourceStorageModeShared];
    if(!A||!B||!C)throw std::runtime_error("buffer allocation failed");
    auto state=pipelines[is_unsigned];std::uint32_t mnk[3]={std::uint32_t(M),std::uint32_t(N),std::uint32_t(K)};
    id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:state];[encoder setBuffer:A offset:0 atIndex:0];[encoder setBuffer:B offset:0 atIndex:1];[encoder setBuffer:C offset:0 atIndex:2];
    [encoder setBytes:mnk length:sizeof(mnk) atIndex:3];
    for(int r=0;r<repeats;++r)[encoder dispatchThreadgroups:MTLSizeMake((N+31)/32,(M+63)/64,1) threadsPerThreadgroup:MTLSizeMake(state.threadExecutionWidth*4,1,1)];
    [encoder endEncoding];[command commit];[command waitUntilCompleted];
    if(command.status==MTLCommandBufferStatusError)throw std::runtime_error("execution failed: "+text(command.error));
    c.resize(std::size_t(M)*N);std::memcpy(c.data(),C.contents,c.size()*4);return (command.GPUEndTime-command.GPUStartTime)/repeats;
}}
template<class T> std::int64_t dot(const std::vector<T>& a,const std::vector<T>& b,int i,int j,int N,int K){
    std::int64_t s=0;for(int k=0;k<K;++k)s+=std::int64_t(a[std::size_t(i)*K+k])*b[std::size_t(k)*N+j];return s;}
// Exactness on random full-range data; also reports whether any reference sum leaves int32.
template<class T> void exact(bool is_unsigned,int M,int N,int K,std::mt19937& rng,const char* label){
    std::vector<T> a(std::size_t(M)*K),b(std::size_t(K)*N);for(auto& x:a)x=T(rng());for(auto& x:b)x=T(rng());
    std::vector<int32_t> c;gemm(is_unsigned,a.data(),b.data(),c,M,N,K);std::size_t bad=0;
    for(int i=0;i<M;++i)for(int j=0;j<N;++j)if(std::int64_t(c[std::size_t(i)*N+j])!=dot(a,b,i,j,N,K))++bad;
    std::cout<<label<<" random M="<<M<<" N="<<N<<" K="<<K<<": "<<(bad?"MISMATCH "+std::to_string(bad):std::string("exact"))<<"\n";
}
// Constant operands: every output equals K*x*y. Classifies behaviour beyond int32.
template<class T> void extreme(bool is_unsigned,T x,T y,int K,const char* label){
    const int M=64,N=32;std::vector<T> a(std::size_t(M)*K,x),b(std::size_t(K)*N,y);std::vector<int32_t> c;gemm(is_unsigned,a.data(),b.data(),c,M,N,K);
    std::int64_t want=std::int64_t(K)*x*y;std::int32_t wrapped=std::int32_t(std::uint32_t(want)),saturated=std::int32_t(std::clamp<std::int64_t>(want,INT32_MIN,INT32_MAX));
    bool uniform=std::all_of(c.begin(),c.end(),[&](int32_t v){return v==c[0];});
    std::cout<<label<<" K="<<K<<" exact="<<want<<" got="<<c[0]<<(uniform?"":" (non-uniform)")<<" -> "
        <<(c[0]==want?"exact":c[0]==wrapped?"wraps mod 2^32":c[0]==saturated?"saturates":"other")<<"\n";
}
int main(){try{
    device=MTLCreateSystemDefaultDevice();if(!device)throw std::runtime_error("no Metal GPU");queue=[device newCommandQueue];
    MTLCompileOptions* options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion((4u<<16)|0u);
    NSError* error=nil;id<MTLLibrary> library=[device newLibraryWithSource:@(source) options:options error:&error];
    if(!library)throw std::runtime_error("Metal compilation: "+text(error));
    int k=0;for(NSString* name:@[@"product_s8",@"product_u8"]){id<MTLFunction> f=[library newFunctionWithName:name];
        pipelines[k]=[device newComputePipelineStateWithFunction:f error:&error];if(!pipelines[k++])throw std::runtime_error("pipeline: "+text(error));}
    std::cout<<[device.name UTF8String]<<"; TensorOps matmul2d, relaxed_precision=false\n";
    std::mt19937 rng(20261007);
    for(int K:{32,1000,16384})exact<int8_t>(false,128,96,K,rng,"int8");
    for(int K:{32,1000,16384})exact<uint8_t>(true,128,96,K,rng,"uint8");
    extreme<int8_t>(false,-128,-128,131071,"int8 (-128)*(-128)");extreme<int8_t>(false,-128,-128,131072,"int8 (-128)*(-128)");
    extreme<int8_t>(false,-128,-128,262144,"int8 (-128)*(-128)");extreme<int8_t>(false,-128,127,262144,"int8 (-128)*127");
    extreme<uint8_t>(true,255,255,33025,"uint8 255*255");extreme<uint8_t>(true,255,255,33026,"uint8 255*255");
    for(int n:{2048,4096}){std::vector<int8_t> a(std::size_t(n)*n),b(std::size_t(n)*n);for(auto& x:a)x=int8_t(rng());for(auto& x:b)x=int8_t(rng());
        std::vector<int32_t> c;gemm(false,a.data(),b.data(),c,n,n,n,1);double t=gemm(false,a.data(),b.data(),c,n,n,n,8);
        std::size_t bad=0;for(int s=0;s<256;++s){int i=int(rng()%n),j=int(rng()%n);if(std::int64_t(c[std::size_t(i)*n+j])!=dot(a,b,i,j,n,n))++bad;}
        std::cout<<"int8 GEMM "<<n<<"^3: "<<t*1e3<<" ms, "<<2.0*n*n*n/t/1e12<<" TOPS; 256 sampled outputs "<<(bad?"MISMATCH":"exact")<<"\n";}
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
