// Polynomial values and jets (plan S1), norms and status summaries (plan S4): host-array API over src/numerics.metal.
// Own device, queue and per-precision runtime-compiled library (as linalg.mm); pipelines specialised by function constants.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "limbforge/numerics.hpp"
#include "numerics_source.hpp"
#include <chrono>
#include <climits>
#include <cstring>
#include <map>
#include <stdexcept>
#include <unistd.h>
#include <vector>

namespace limbforge {
namespace {
using Clock=std::chrono::steady_clock;
double since(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
std::string message(NSError* e){return e?std::string([[e localizedDescription] UTF8String]):"unknown Metal error";}
struct PolyParams {std::uint32_t count,terms,points_per_set;};
struct NormParams {std::uint32_t segments,length,blocks;};
enum Kind : std::uint32_t {status_kind=0,inf_kind=1,max_kind=2,scan_kind=3,sum_kind=4,ratio_kind=5};
constexpr std::size_t tree_group=128; // TG in numerics.metal
std::size_t up(std::size_t x,std::size_t m){return (x+m-1)/m*m;}
void check_bits(int bits){if(bits<64||bits>1024||bits%32)throw std::invalid_argument("bits must be a multiple of 32 in [64,1024]");}
}
struct Numerics::Impl {
    id<MTLDevice> device;id<MTLCommandQueue> queue;std::map<int,id<MTLLibrary>> libraries;
    std::map<std::tuple<int,std::string,std::uint32_t>,id<MTLComputePipelineState>> pipelines;std::map<std::string,id<MTLBuffer>> pool;
    id<MTLLibrary> library(int bits){
        auto found=libraries.find(bits);if(found!=libraries.end())return found->second;
        NSString* source=[NSString stringWithFormat:@"#define LF_BITS %d\n%s",bits,limbforge_numerics_source];
        MTLCompileOptions* o=[MTLCompileOptions new];o.languageVersion=MTLLanguageVersion3_1;o.mathMode=MTLMathModeSafe;
        NSError* e=nil;id<MTLLibrary> lib=[device newLibraryWithSource:source options:o error:&e];
        if(!lib)throw std::runtime_error("Metal compilation (numerics): "+message(e));
        return libraries[bits]=lib;
    }
    id<MTLComputePipelineState> pipeline(int bits,const char* name,int constant,std::uint32_t value){
        auto key=std::make_tuple(bits,std::string(name),value|(std::uint32_t(constant)<<16));auto found=pipelines.find(key);if(found!=pipelines.end())return found->second;
        MTLFunctionConstantValues* v=[MTLFunctionConstantValues new];std::uint32_t zero=0;
        [v setConstantValue:constant==0?&value:&zero type:MTLDataTypeUInt atIndex:0];[v setConstantValue:constant==1?&value:&zero type:MTLDataTypeUInt atIndex:1];
        NSError* e=nil;id<MTLFunction> f=[library(bits) newFunctionWithName:[NSString stringWithUTF8String:name] constantValues:v error:&e];
        if(!f)throw std::runtime_error("Metal function "+std::string(name)+": "+message(e));
        id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:f error:&e];
        if(!p)throw std::runtime_error("Metal pipeline "+std::string(name)+": "+message(e));
        return pipelines[key]=p;
    }
    id<MTLBuffer> buffer(const std::string& role,std::size_t bytes){
        auto& b=pool[role];if(!b||b.length<bytes){b=nil;b=[device newBufferWithLength:std::max<std::size_t>(bytes,16) options:MTLResourceStorageModeShared];}
        if(!b)throw std::runtime_error("Metal allocation failed ("+role+")");return b;}
    // Page-aligned caller memory is wrapped without a copy (every call waits for the GPU before returning).
    id<MTLBuffer> wrap(const void* p,std::size_t bytes){std::size_t page=std::size_t(getpagesize());if(!p||!bytes||reinterpret_cast<std::uintptr_t>(p)%page)return nil;
        return [device newBufferWithBytesNoCopy:const_cast<void*>(p) length:up(bytes,page) options:MTLResourceStorageModeShared deallocator:nil];}
    id<MTLBuffer> input(const std::string& role,const void* p,std::size_t bytes){
        if(id<MTLBuffer> b=wrap(p,bytes))return b;id<MTLBuffer> b=buffer(role,bytes);if(bytes)std::memcpy(b.contents,p,bytes);return b;}
    Timing poly(int bits,const Polynomial& s,unsigned order,const void* coeffs,const void* points,void* out);
    Timing norm(int bits,Kind kind,const Segments& s,const void* x,const void* scale,void* values,NormInfo* info);
};
Numerics::Numerics():impl(std::make_unique<Impl>()){
    impl->device=MTLCreateSystemDefaultDevice();if(!impl->device)throw std::runtime_error("no Metal GPU available");
    impl->queue=[impl->device newCommandQueue];if(!impl->queue)throw std::runtime_error("cannot create Metal command queue");
}
Numerics::~Numerics()=default;
std::string Numerics::device_name()const{return [[impl->device name] UTF8String];}
Timing Numerics::poly_eval(int bits,const Polynomial& s,const void* coeffs,const void* points,void* values){return impl->poly(bits,s,0,coeffs,points,values);}
Timing Numerics::poly_eval_jet(int bits,const Polynomial& s,unsigned order,const void* coeffs,const void* points,void* jets){return impl->poly(bits,s,order,coeffs,points,jets);}
Timing Numerics::Impl::poly(int bits,const Polynomial& s,unsigned order,const void* coeffs,const void* points,void* out){@autoreleasepool{
    auto start=Clock::now();check_bits(bits);
    if(order>2)throw std::invalid_argument("poly_eval_jet: order must be at most 2");
    if(!s.points_per_set)throw std::invalid_argument("poly_eval: points_per_set must be positive");
    const std::size_t limit=std::size_t(1)<<31,sets=(s.points+s.points_per_set-1)/s.points_per_set;
    if(s.points>=limit||s.terms>=limit||s.points_per_set>=limit||sets*s.terms>=std::size_t(1)<<40)throw std::invalid_argument("poly_eval: shape exceeds 32-bit indexing");
    if(!s.points)return {0,since(start)};
    if(!coeffs&&s.terms)throw std::invalid_argument("poly_eval: missing coefficients");
    if(!points||!out)throw std::invalid_argument("poly_eval: missing points or output");
    const std::size_t element=std::size_t(bits/8+12)*(s.complex?2:1),out_bytes=s.points*(order+1)*element;
    auto p=pipeline(bits,s.complex?"poly_complex":"poly_real",0,order|(s.fused?4u:0u));
    id<MTLBuffer> C=input("coeffs",coeffs,sets*s.terms*element),X=input("points",points,s.points*element),O=wrap(out,out_bytes);bool copy=!O;if(copy)O=buffer("out",out_bytes);
    PolyParams params{std::uint32_t(s.points),std::uint32_t(s.terms),std::uint32_t(s.points_per_set)};
    id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:p];[enc setBuffer:C offset:0 atIndex:0];[enc setBuffer:X offset:0 atIndex:1];[enc setBuffer:O offset:0 atIndex:2];
    [enc setBytes:&params length:sizeof params atIndex:3];
    // One SIMD group per threadgroup: heavy one-thread-per-value kernels (round 6 policy).
    NSUInteger tg=std::min<NSUInteger>(p.threadExecutionWidth,p.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(s.points,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
    [enc endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("poly_eval: command buffer failed: "+message(cb.error));
    if(copy)std::memcpy(out,O.contents,out_bytes);
    return {cb.GPUEndTime-cb.GPUStartTime,since(start)};
}}
Timing Numerics::norm_inf(int bits,const Segments& s,const void* x,void* values,NormInfo* info){return impl->norm(bits,inf_kind,s,x,nullptr,values,info);}
Timing Numerics::norm_max(int bits,const Segments& s,const void* x,void* values,NormInfo* info){return impl->norm(bits,max_kind,s,x,nullptr,values,info);}
Timing Numerics::norm2(int bits,const Segments& s,const void* x,void* values,NormInfo* info){return impl->norm(bits,sum_kind,s,x,nullptr,values,info);}
Timing Numerics::scaled_residual(int bits,const Segments& s,const void* r,const void* scale,void* values,NormInfo* info){
    if(!scale&&s.count&&s.length)throw std::invalid_argument("scaled_residual: missing scale");return impl->norm(bits,ratio_kind,s,r,scale,values,info);}
Timing Numerics::summarize_status(int bits,const Segments& s,const void* x,NormInfo* info){return impl->norm(bits,status_kind,s,x,nullptr,nullptr,info);}
Timing Numerics::Impl::norm(int bits,Kind kind,const Segments& s,const void* x,const void* scale,void* values,NormInfo* info){@autoreleasepool{
    auto start=Clock::now();check_bits(bits);
    const std::size_t limit=std::size_t(1)<<31;
    if(s.length>=limit||s.count>=limit||s.count*s.length>=limit)throw std::invalid_argument("norm: shape exceeds 32-bit indexing");
    if(kind!=status_kind&&!values&&s.count)throw std::invalid_argument("norm: missing values");
    if(kind==status_kind&&!info&&s.count)throw std::invalid_argument("summarize_status: missing info");
    if(!s.count)return {0,since(start)};
    const std::size_t real=bits/8+12,element=real*(s.complex?2:1),S=s.count,L=s.length;
    if(!L){ // empty segments: zero, no index (no GPU work)
        for(std::size_t i=0;i<S;++i){if(values)std::memset(static_cast<unsigned char*>(values)+i*real,0,real);if(info)info[i]=NormInfo{};}
        return {0,since(start)};}
    if(!x)throw std::invalid_argument("norm: missing input");
    const std::uint32_t flags=(s.complex?1u:0u);
    auto first=pipeline(bits,"norm_first",1,flags|(std::uint32_t(kind==sum_kind?scan_kind:kind)<<1));
    auto finish=pipeline(bits,"norm_finish",1,flags|(std::uint32_t(kind)<<1));
    // Tree threadgroups: a power of two in [32, 128], within the pipeline limit (fixed tree; see docs/numerics.md).
    auto group_size=[&](id<MTLComputePipelineState> p,std::size_t len){std::size_t cap=std::min<std::size_t>(tree_group,p.maxTotalThreadsPerThreadgroup),t=32;
        while(t<len&&2*t<=cap)t*=2;if(t>cap)throw std::runtime_error("norm: pipeline threadgroup limit below one SIMD group");return t;};
    id<MTLBuffer> X=input("x",x,S*L*element),Sc=scale?input("scale",scale,S*L*real):X;
    id<MTLBuffer> summary=buffer("summary",S*16),emax=buffer("emax",S*4),vals=buffer("values",S*real),inf=buffer("info",S*16);
    {auto u=static_cast<std::uint32_t*>(summary.contents);auto e=static_cast<std::int32_t*>(emax.contents);
     for(std::size_t i=0;i<S;++i){u[4*i]=0;u[4*i+1]=0;u[4*i+2]=no_index;u[4*i+3]=0;e[i]=INT_MIN;}}
    id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    const bool tree=kind!=status_kind;id<MTLBuffer> keys[2]={nil,nil},idx[2]={nil,nil};int cur=0;
    auto pass_first=[&](id<MTLComputePipelineState> p,bool output){
        std::size_t t=group_size(p,L),blocks=(L+t-1)/t;NormParams params{std::uint32_t(S),std::uint32_t(L),std::uint32_t(blocks)};
        if(output){keys[0]=buffer("keys0",S*blocks*real);idx[0]=buffer("index0",S*blocks*4);}
        [enc setComputePipelineState:p];[enc setBuffer:X offset:0 atIndex:0];[enc setBuffer:Sc offset:0 atIndex:1];
        [enc setBuffer:output?keys[0]:vals offset:0 atIndex:2];[enc setBuffer:output?idx[0]:inf offset:0 atIndex:3];
        [enc setBuffer:summary offset:0 atIndex:4];[enc setBuffer:emax offset:0 atIndex:5];[enc setBytes:&params length:sizeof params atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(blocks,S,1) threadsPerThreadgroup:MTLSizeMake(t,1,1)];return blocks;};
    std::size_t blocks=1;
    if(kind==sum_kind)pass_first(first,false); // statuses and the largest exponent, then the terms
    if(tree){
        blocks=pass_first(kind==sum_kind?pipeline(bits,"norm_first",1,flags|(std::uint32_t(sum_kind)<<1)):first,true);
        auto combine=blocks>1?pipeline(bits,"norm_combine",1,flags|(std::uint32_t(kind)<<1)):nil;
        for(std::size_t len=blocks;len>1;len=blocks){
            std::size_t t=group_size(combine,len);blocks=(len+t-1)/t;NormParams params{std::uint32_t(S),std::uint32_t(len),std::uint32_t(blocks)};
            int nxt=1-cur;keys[nxt]=buffer(nxt?"keys1":"keys0",S*blocks*real);idx[nxt]=buffer(nxt?"index1":"index0",S*blocks*4);
            [enc setComputePipelineState:combine];[enc setBuffer:keys[cur] offset:0 atIndex:0];[enc setBuffer:idx[cur] offset:0 atIndex:1];
            [enc setBuffer:keys[nxt] offset:0 atIndex:2];[enc setBuffer:idx[nxt] offset:0 atIndex:3];[enc setBytes:&params length:sizeof params atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(blocks,S,1) threadsPerThreadgroup:MTLSizeMake(t,1,1)];cur=nxt;}
    }else pass_first(first,false);
    NormParams params{std::uint32_t(S),std::uint32_t(L),1};
    [enc setComputePipelineState:finish];[enc setBuffer:tree?keys[cur]:vals offset:0 atIndex:0];[enc setBuffer:tree?idx[cur]:inf offset:0 atIndex:1];
    [enc setBuffer:summary offset:0 atIndex:2];[enc setBuffer:emax offset:0 atIndex:3];[enc setBuffer:vals offset:0 atIndex:4];[enc setBuffer:inf offset:0 atIndex:5];
    [enc setBytes:&params length:sizeof params atIndex:6];
    [enc dispatchThreads:MTLSizeMake(S,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64,finish.maxTotalThreadsPerThreadgroup),1,1)];
    [enc endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("norm: command buffer failed: "+message(cb.error));
    if(values)std::memcpy(values,vals.contents,S*real);
    if(info){auto u=static_cast<const std::uint32_t*>(inf.contents);for(std::size_t i=0;i<S;++i)info[i]={u[4*i],u[4*i+1],u[4*i+2],u[4*i+3]};}
    return {cb.GPUEndTime-cb.GPUStartTime,since(start)};
}}
}
