// Cold compile time of one kernel specialisation (round 46-fused-wide). Reads core.hpp and kernels.metal
// from the given paths (so variants can be compared), injects a nonce into every early-return guard so the OS
// shader cache cannot serve the pipeline, and times library compilation and pipeline creation separately.
// One configuration per process: a Metal compiler crash or hang (watchdog) only ends this run.
// Build:
//   c++ -std=c++17 -O2 -fobjc-arc benchmarks/experiments/vr_compile_probe.mm -o vr_compile_probe -framework Metal -framework Foundation
// Usage: vr_compile_probe CORE KERNELS BITS KERNEL [CONSTANT_INDEX VALUE] [--warm] [--timeout S]
//   e.g. vr_compile_probe include/limbforge/core.hpp src/kernels.metal 1024 vector_recurrence 2 48
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>
static std::string slurp(const char* path){std::ifstream f(path);if(!f)throw std::runtime_error(std::string("cannot read ")+path);std::stringstream s;s<<f.rdbuf();return s.str();}
static void replace_all(std::string& s,const std::string& from,const std::string& to){for(size_t p=0;(p=s.find(from,p))!=std::string::npos;p+=to.size())s.replace(p,from.size(),to);}
static void watchdog(int){static const char m[]="TIMEOUT\n";write(1,m,sizeof m-1);_exit(3);}
int main(int argc,char** argv){@autoreleasepool{
    if(argc<5){std::cerr<<"usage: vr_compile_probe CORE KERNELS BITS KERNEL [INDEX VALUE] [--warm] [--timeout S]\n";return 2;}
    std::string core=slurp(argv[1]),kernels=slurp(argv[2]);int bits=std::stoi(argv[3]);NSString* name=[NSString stringWithUTF8String:argv[4]];
    int index=-1;unsigned value=0;bool warm=false;unsigned timeout=600;
    for(int i=5;i<argc;++i){std::string a=argv[i];if(a=="--warm")warm=true;else if(a=="--timeout"&&i+1<argc)timeout=unsigned(std::stoul(argv[++i]));
        else if(index<0&&i+1<argc){index=std::stoi(a);value=unsigned(std::stoul(argv[++i]));}else{std::cerr<<"bad argument "<<a<<"\n";return 2;}}
    replace_all(core,"#pragma once","");
    std::random_device rd;unsigned nonce=warm?0xfffffff0u:(0x80000000u|(unsigned(rd())&0x7ffffff0u));
    // Same semantics (no thread index reaches the nonce), different function bodies.
    replace_all(kernels,"if(lane>=p.lanes)return;","if(lane>=p.lanes||lane=="+std::to_string(nonce)+"u)return;");
    replace_all(kernels,"if(i>=p.count)return;","if(i>=p.count||i=="+std::to_string(nonce)+"u)return;");
    replace_all(kernels,"if(i>=p.segments)return;","if(i>=p.segments||i=="+std::to_string(nonce)+"u)return;");
    std::string source="#define MP_BITS "+std::to_string(bits)+"\n#include <metal_stdlib>\n"+core+"\n"+kernels;
    signal(SIGALRM,watchdog);alarm(timeout);
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    MTLCompileOptions* options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;options.languageVersion=MTLLanguageVersion3_1;
    NSError* error=nil;auto t0=std::chrono::steady_clock::now();
    id<MTLLibrary> lib=[device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];
    auto t1=std::chrono::steady_clock::now();
    if(!lib){std::cout<<"LIBRARY_ERROR "<<[[error localizedDescription] UTF8String]<<"\n";return 1;}
    id<MTLFunction> f;
    if(index>=0){MTLFunctionConstantValues* c=[MTLFunctionConstantValues new];[c setConstantValue:&value type:MTLDataTypeUInt atIndex:index];f=[lib newFunctionWithName:name constantValues:c error:&error];}
    else f=[lib newFunctionWithName:name];
    if(!f){std::cout<<"FUNCTION_ERROR "<<(error?[[error localizedDescription] UTF8String]:"missing")<<"\n";return 1;}
    auto t2=std::chrono::steady_clock::now();
    id<MTLComputePipelineState> state=[device newComputePipelineStateWithFunction:f error:&error];
    auto t3=std::chrono::steady_clock::now();
    if(!state){std::cout<<"PIPELINE_ERROR "<<[[error localizedDescription] UTF8String]<<"\n";return 1;}
    auto s=[](auto a,auto b){return std::chrono::duration<double>(b-a).count();};
    std::cout<<"bits="<<bits<<" kernel="<<argv[4]<<" const="<<index<<":"<<value<<" library_s="<<s(t0,t1)<<" specialise_s="<<s(t1,t2)<<" pipeline_s="<<s(t2,t3)
             <<" regs_max_threads="<<state.maxTotalThreadsPerThreadgroup<<"\n";
    return 0;
}}
