// Round 44 (docs/gpu-codegen.md section 11): core add after a product or a division at W >= 36 words, inside one Metal
// kernel, against the same code on the CPU. Each probe kernel writes six expressions of Number<W> operands (x, y from a
// buffer, one = 1, s = x/32); the CPU result is the reference. At W = 34, 35 every expression is exact; from W = 36 the
// sums of a computed product or quotient are wrong for every input, while the products and quotients alone are exact.
// This is why the GPU retry rungs of the transcendental kernels stop at 35 words.
// Run from the repository root (reads include/limbforge/core.hpp and src/transcendental_core.hpp). Build:
//   c++ -std=c++17 -O2 -fobjc-arc -Iinclude -Isrc benchmarks/experiments/transcendental_wide_probe.mm -o wide_probe \
//       -framework Metal -framework Foundation
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "limbforge/core.hpp"
#include "transcendental_core.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
using namespace limbforge;
namespace tr=limbforge::transcendental;
static std::string slurp(const char* path){std::ifstream f(path);std::stringstream s;s<<f.rdbuf();std::string t=s.str();
    auto p=t.find("#pragma once");if(p!=std::string::npos)t.erase(p,12);return t;}
static const char* names[6]={"add(1,s)","add(1,mul(s,s))","add(x,mul(s,s))","sub(mul(x,x),s)","div_word(mul(x,x),7)","add(1,div_word(x,7))"};
static const char* kernel=R"K(
using namespace limbforge;using namespace limbforge::transcendental;
kernel void probe(device const Number<PW>* x[[buffer(0)]],device Number<PW>* out[[buffer(1)]],uint i[[thread_position_in_grid]]){
    Number<PW> v=x[i],one=unit<PW>(),s=scaled(v,-5);
    out[6*i]=add(one,s);out[6*i+1]=add(one,mul(s,s));out[6*i+2]=add(v,mul(s,s));out[6*i+3]=sub(mul(v,v),s);
    out[6*i+4]=div_word(mul(v,v),word(7));out[6*i+5]=add(one,div_word(v,word(7)));
}
)K";
template<int W> void run(id<MTLDevice> d,const std::string& core,const std::string& tcore){
    const int n=256;std::mt19937_64 g(W);std::vector<Number<W>> a(n),o(6*n),ref(6*n);
    for(auto& p:a){for(int k=0;k<W;++k)p.limb[k]=word(g());p.limb[W-1]|=0x80000000u;p.exponent=int(g()%8)-6;p.sign=g()&1?1:-1;p.status=0;}
    for(int i=0;i<n;++i){Number<W> v=a[i],one=tr::unit<W>(),s=tr::scaled(v,-5);
        ref[6*i]=add(one,s);ref[6*i+1]=add(one,mul(s,s));ref[6*i+2]=add(v,mul(s,s));ref[6*i+3]=sub(mul(v,v),s);
        ref[6*i+4]=tr::div_word(mul(v,v),word(7));ref[6*i+5]=add(one,tr::div_word(v,word(7)));}
    std::string src="#include <metal_stdlib>\n#define PW "+std::to_string(W)+"\n"+core+tcore+kernel;
    MTLCompileOptions* co=[MTLCompileOptions new];co.languageVersion=MTLLanguageVersion((4u<<16)|0u);co.mathMode=MTLMathModeSafe;
    NSError* e=nil;id<MTLLibrary> lib=[d newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:co error:&e];
    if(!lib){std::printf("W=%d: %s\n",W,[[e localizedDescription] UTF8String]);return;}
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"probe"] error:&e];
    id<MTLBuffer> A=[d newBufferWithBytes:a.data() length:sizeof(Number<W>)*n options:0],O=[d newBufferWithLength:sizeof(Number<W>)*6*n options:0];
    id<MTLCommandQueue> q=[d newCommandQueue];id<MTLCommandBuffer> cb=[q commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:p];[enc setBuffer:A offset:0 atIndex:0];[enc setBuffer:O offset:0 atIndex:1];
    [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];
    std::memcpy(o.data(),O.contents,sizeof(Number<W>)*6*n);int bad[6]={};
    for(int i=0;i<6*n;++i)if(std::memcmp(&o[i],&ref[i],sizeof(Number<W>)))++bad[i%6];
    std::printf("W=%d wrong of %d:",W,n);for(int k=0;k<6;++k)std::printf(" %s %d",names[k],bad[k]);std::printf("\n");
}
int main(){@autoreleasepool{
    std::string core=slurp("include/limbforge/core.hpp"),tcore=slurp("src/transcendental_core.hpp");
    if(core.empty()||tcore.empty()){std::fprintf(stderr,"run from the repository root\n");return 2;}
    id<MTLDevice> d=MTLCreateSystemDefaultDevice();run<34>(d,core,tcore);run<35>(d,core,tcore);run<36>(d,core,tcore);run<40>(d,core,tcore);run<48>(d,core,tcore);
}}
