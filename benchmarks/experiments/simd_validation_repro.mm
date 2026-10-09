// Round 41-L4b: a WolfNum-independent reproducer for wrong cross-lane values under MTL_SHADER_VALIDATION.
// Each thread keeps K words; every round, word j becomes v_j*1664525 + w_{r,j} + shuffle(v_j, lane ^ s_j) + v_{j+1},
// with the round weights w read from device memory (as the recurrence reads its weights). The CPU replays the
// same lockstep semantics; every dispatch is compared word for word. Exchange modes (--mode):
//   shuffle  simd_shuffle                         tgsimd  threadgroup memory + simdgroup_barrier(mem_threadgroup)
//   tgfull   threadgroup memory + threadgroup_barrier  none    no cross-lane exchange (x = v_j)
// Build: target simd_validation_repro. Run with and without MTL_SHADER_VALIDATION=1, ideally while other GPU work
// runs in another process (the failures in round 41 needed both).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
static const char* source=R"(
#include <metal_stdlib>
using namespace metal;
inline uint xchg(uint x,uint src,uint lane,threadgroup uint* s){
#if MODE==1
    return simd_shuffle(x,ushort(src));
#elif MODE==2
    simdgroup_barrier(mem_flags::mem_threadgroup);s[lane]=x;simdgroup_barrier(mem_flags::mem_threadgroup);return s[src];
#elif MODE==3
    threadgroup_barrier(mem_flags::mem_threadgroup);s[lane]=x;threadgroup_barrier(mem_flags::mem_threadgroup);return s[src];
#else
    return x;
#endif
}
kernel void repro(device const uint* in [[buffer(0)]],device const uint* w [[buffer(1)]],device uint* out [[buffer(2)]],constant uint& rounds [[buffer(3)]],
                  uint gid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    threadgroup uint s[32];
    uint v[K];for(int j=0;j<K;++j)v[j]=in[gid*K+j];
    for(uint r=0;r<rounds;++r){
        for(int j=0;j<K;++j){uint x=xchg(v[j],lane^uint(1+j%7),lane,s);v[j]=v[j]*1664525u+w[r*K+j]+x+v[(j+1)%K];}
    }
    for(int j=0;j<K;++j)out[gid*K+j]=v[j];
}
)";
int main(int argc,char** argv){try{
    int K=96,mode=1;unsigned threads=8192,rounds=64,repeats=20;
    for(int i=1;i+1<argc;i+=2){std::string a=argv[i],v=argv[i+1];
        if(a=="--words")K=std::stoi(v);else if(a=="--threads")threads=unsigned(std::stoul(v));else if(a=="--rounds")rounds=unsigned(std::stoul(v));
        else if(a=="--repeats")repeats=unsigned(std::stoul(v));
        else if(a=="--mode")mode=v=="shuffle"?1:v=="tgsimd"?2:v=="tgfull"?3:v=="none"?0:throw std::invalid_argument("mode: shuffle|tgsimd|tgfull|none");
        else throw std::invalid_argument("usage: simd_validation_repro [--words K] [--threads T] [--rounds R] [--repeats N] [--mode shuffle|tgsimd|tgfull|none]");}
    if(threads%32)throw std::invalid_argument("threads must be a multiple of 32");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> queue=[device newCommandQueue];
    MTLCompileOptions* options=[MTLCompileOptions new];options.preprocessorMacros=@{@"K":@(K),@"MODE":@(mode)};
    NSError* error=nil;id<MTLLibrary> lib=[device newLibraryWithSource:[NSString stringWithUTF8String:source] options:options error:&error];
    if(!lib)throw std::runtime_error([[error localizedDescription] UTF8String]);
    id<MTLComputePipelineState> state=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"repro"] error:&error];
    if(!state)throw std::runtime_error([[error localizedDescription] UTF8String]);
    std::mt19937 rng(41);std::vector<std::uint32_t> in(std::size_t(threads)*K),w(std::size_t(rounds)*K),want(in.size());
    for(auto& x:in)x=rng();for(auto& x:w)x=rng();
    // CPU replay: per SIMD group of 32 lanes, all lanes read the pre-update value of word j before any lane writes it.
    want=in;
    for(unsigned g=0;g<threads;g+=32)for(unsigned r=0;r<rounds;++r)for(int j=0;j<K;++j){std::uint32_t x[32];
        for(unsigned l=0;l<32;++l)x[l]=mode?want[(g+(l^unsigned(1+j%7)))*K+j]:want[(g+l)*K+j];
        for(unsigned l=0;l<32;++l){auto& v=want[(g+l)*K+j];v=v*1664525u+w[r*K+j]+x[l]+want[(g+l)*K+(j+1)%K];}}
    id<MTLBuffer> bi=[device newBufferWithBytes:in.data() length:in.size()*4 options:MTLResourceStorageModeShared],
        bw=[device newBufferWithBytes:w.data() length:w.size()*4 options:MTLResourceStorageModeShared],
        bo=[device newBufferWithLength:in.size()*4 options:MTLResourceStorageModeShared];
    unsigned bad_dispatches=0;std::size_t bad_groups=0;double seconds=0;
    for(unsigned rep=0;rep<repeats;++rep){std::memset(bo.contents,0,bo.length);
        @autoreleasepool{id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:state];[e setBuffer:bi offset:0 atIndex:0];[e setBuffer:bw offset:0 atIndex:1];[e setBuffer:bo offset:0 atIndex:2];
            [e setBytes:&rounds length:4 atIndex:3];[e dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];
            [cb commit];[cb waitUntilCompleted];if(cb.status==MTLCommandBufferStatusError)throw std::runtime_error("command buffer failed");
            seconds+=cb.GPUEndTime-cb.GPUStartTime;}
        const std::uint32_t* got=static_cast<const std::uint32_t*>(bo.contents);std::size_t groups=0;
        for(unsigned g=0;g<threads;g+=32)groups+=std::memcmp(got+std::size_t(g)*K,want.data()+std::size_t(g)*K,std::size_t(32)*K*4)!=0;
        bad_groups+=groups;bad_dispatches+=groups!=0;}
    const char* v=getenv("MTL_SHADER_VALIDATION");
    std::printf("validation=%s,mode=%d,words=%d,threads=%u,rounds=%u,repeats=%u,bad_dispatches=%u,bad_simd_groups=%zu,ms_per_dispatch=%.3f\n",
        v?v:"unset",mode,K,threads,rounds,repeats,bad_dispatches,bad_groups,seconds/repeats*1e3);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
