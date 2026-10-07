#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "limbforge/engine.hpp"
#include "shader_source.hpp"
#include <chrono>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <atomic>
#include <mutex>
#include <vector>
#include <algorithm>

namespace limbforge {
namespace {
std::string error_message(NSError* error){return error?std::string([[error localizedDescription] UTF8String]):"unknown Metal error";}
std::size_t checked_size(std::size_t count,std::size_t stride) {
    if(stride&&count>std::numeric_limits<std::size_t>::max()/stride)throw std::invalid_argument("buffer size overflow");
    return count*stride;
}
void validate(int bits,std::size_t count) {
    if(bits<64||bits>1024||bits%32)throw std::invalid_argument("bits must be a multiple of 32 in [64,1024]");
    if(count>std::numeric_limits<std::uint32_t>::max())throw std::invalid_argument("batch exceeds 32-bit indexing");
}
struct Params {std::uint32_t count,operation,steps,weight_count,states_per_weight;};
}
namespace detail {
struct BufferStorage {
    id<MTLBuffer> buffer;std::shared_ptr<int> owner;std::atomic<bool> busy{false};std::size_t bytes;
};
void* mapped(const std::shared_ptr<BufferStorage>& storage){
    if(!storage)throw std::invalid_argument("empty buffer handle");
    if(storage->busy.load(std::memory_order_acquire))throw std::logic_error("buffer belongs to an unwaited submission");
    return storage->buffer.contents;
}
void transfer(const std::shared_ptr<BufferStorage>& storage,void* host,std::size_t bytes,bool upload){
    auto data=mapped(storage);if(bytes>storage->bytes||(!host&&bytes))throw std::invalid_argument("invalid transfer");
    if(bytes){if(upload)std::memcpy(data,host,bytes);else std::memcpy(host,data,bytes);}
}
}
struct Engine::Impl {
    EngineOptions options;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    std::map<int,id<MTLLibrary>> libraries;
    std::map<std::pair<int,int>,id<MTLComputePipelineState>> pipelines;
    id<MTLBuffer> buffers[3];
    std::shared_ptr<int> identity=std::make_shared<int>(0);
    explicit Impl(EngineOptions requested):options(requested) {
        unsigned n=options.threads_per_threadgroup;
        if(n&&(n<32||n>1024||(n&(n-1))))throw std::invalid_argument("threadgroup size must be zero or a power of two in [32,1024]");
        device=MTLCreateSystemDefaultDevice();
        if(!device)throw std::runtime_error("no Metal GPU available");
        queue=[device newCommandQueue];if(!queue)throw std::runtime_error("cannot create Metal command queue");
    }
    id<MTLComputePipelineState> pipeline(int bits,int operation) {
        auto key=std::make_pair(bits,operation);
        auto existing=pipelines.find(key);if(existing!=pipelines.end())return existing->second;
        if(!libraries.count(bits)) {
            NSString* source=[NSString stringWithFormat:@"#define MP_BITS %d\n%s",bits,limbforge_shader_source];
            MTLCompileOptions* options=[MTLCompileOptions new];
            options.mathMode=MTLMathModeSafe;options.languageVersion=MTLLanguageVersion3_1;
            NSError* error=nil;
            id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
            if(!library)throw std::runtime_error("Metal compilation: "+error_message(error));
            libraries[bits]=library;
        }
        NSError* error=nil;id<MTLFunction> function;
        if(operation>=100){
            NSString* names[]={@"recurrence",@"tree_sum_real",@"tree_sum_complex",@"global_tree_sum_real",@"global_tree_sum_complex"};
            function=[libraries[bits] newFunctionWithName:names[operation-100]];
        }
        else {
            MTLFunctionConstantValues* constants=[MTLFunctionConstantValues new];
            std::uint32_t op=operation==7&&bits!=384?2:operation;[constants setConstantValue:&op type:MTLDataTypeUInt atIndex:0];
            function=[libraries[bits] newFunctionWithName:operation>=4&&operation<=6?@"complex_arithmetic":@"arithmetic" constantValues:constants error:&error];
        }
        if(!function)throw std::runtime_error("Metal specialization: "+error_message(error));
        id<MTLComputePipelineState> state=[device newComputePipelineStateWithFunction:function error:&error];
        if(!state)throw std::runtime_error("Metal pipeline: "+error_message(error));
        pipelines[key]=state;return state;
    }
    unsigned group_size(id<MTLComputePipelineState> state,int bits,int operation)const{
        NSUInteger width=state.threadExecutionWidth,maximum=state.maxTotalThreadsPerThreadgroup;
        NSUInteger preferred=options.threads_per_threadgroup?options.threads_per_threadgroup:((operation==100||(bits>=384&&(operation==5||operation==6)))?width:NSUInteger(128));
        if(preferred%width)throw std::invalid_argument("threadgroup size must be a multiple of pipeline SIMD width");
        NSUInteger result=std::min(preferred,maximum);result-=result%width;
        if(!result)throw std::runtime_error("pipeline cannot fit one SIMD group");
        return unsigned(result);
    }
    std::shared_ptr<detail::BufferStorage> allocate(std::size_t bytes){
        if(bytes>device.maxBufferLength)throw std::invalid_argument("buffer exceeds device maximum");
        auto r=std::make_shared<detail::BufferStorage>();r->owner=identity;r->bytes=bytes;
        r->buffer=[device newBufferWithLength:std::max(std::size_t(1),bytes) options:MTLResourceStorageModeShared];
        if(!r->buffer)throw std::runtime_error("Metal buffer allocation failed");return r;
    }
    void reserve(unsigned index,std::size_t bytes) {
        if(bytes>device.maxBufferLength)throw std::invalid_argument("buffer exceeds device maximum");
        if(!buffers[index]||buffers[index].length<bytes) {
            buffers[index]=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if(!buffers[index])throw std::runtime_error("Metal buffer allocation failed");
        }
    }
    Timing dispatch(int bits,int operation,const void* a,std::size_t a_bytes,const void* b,std::size_t b_bytes,
                    void* out,std::size_t out_bytes,std::size_t count,unsigned steps,unsigned states_per_weight=1) {
        @autoreleasepool {
            // Compilation is deliberately excluded from timings; cached per precision/operation.
            auto state=pipeline(bits,operation);
            auto start=std::chrono::steady_clock::now();
            reserve(0,a_bytes);if(b_bytes)reserve(1,b_bytes);reserve(2,out_bytes);
            std::memcpy(buffers[0].contents,a,a_bytes);if(b_bytes)std::memcpy(buffers[1].contents,b,b_bytes);
            id<MTLCommandBuffer> command=[queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            if(!command||!encoder)throw std::runtime_error("Metal command allocation failed");
            [encoder setComputePipelineState:state];
            for(unsigned i=0;i<3;++i)[encoder setBuffer:buffers[i==1&&!b_bytes?0:i] offset:0 atIndex:i];
            Params params={std::uint32_t(count),std::uint32_t(operation),steps,std::uint32_t(count/states_per_weight),states_per_weight};
            [encoder setBytes:&params length:sizeof(params) atIndex:3];
            NSUInteger group=group_size(state,bits,operation);
            [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(group,1,1)];
            [encoder endEncoding];[command commit];[command waitUntilCompleted];
            if(command.status==MTLCommandBufferStatusError)throw std::runtime_error("Metal execution: "+error_message(command.error));
            std::memcpy(out,buffers[2].contents,out_bytes);
            double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            return {command.GPUEndTime-command.GPUStartTime,wall};
        }
    }
};
struct Submission::Impl {
    id<MTLCommandBuffer> command;
    std::vector<std::shared_ptr<detail::BufferStorage>> resources;
    std::chrono::steady_clock::time_point start;
    std::once_flag finished;Timing timing{};std::string error;
    void finish(){std::call_once(finished,[&]{
        [command waitUntilCompleted];
        if(command.status==MTLCommandBufferStatusError)error=error_message(command.error);
        timing={command.GPUEndTime-command.GPUStartTime,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()};
        for(auto& r:resources)r->busy.store(false,std::memory_order_release);
    });}
    ~Impl(){finish();}
};
struct CommandBatch::Impl {
    std::shared_ptr<Engine::Impl> engine;
    id<MTLCommandBuffer> command;id<MTLComputeCommandEncoder> encoder;
    std::vector<std::shared_ptr<detail::BufferStorage>> resources;
    std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();bool submitted=false;
    explicit Impl(std::shared_ptr<Engine::Impl> e):engine(std::move(e)){
        command=[engine->queue commandBuffer];if(!command)throw std::runtime_error("Metal command allocation failed");
    }
    ~Impl(){if(encoder)[encoder endEncoding];}
    void retain(const std::shared_ptr<detail::BufferStorage>& r){
        if(!r||r->owner!=engine->identity)throw std::invalid_argument("buffer belongs to another engine");
        if(r->busy.load(std::memory_order_acquire))throw std::logic_error("buffer belongs to an unwaited submission");
        if(std::find(resources.begin(),resources.end(),r)==resources.end())resources.push_back(r);
    }
};
CommandBatch::CommandBatch(std::unique_ptr<Impl> p):impl_(std::move(p)){}
CommandBatch::~CommandBatch()=default;
CommandBatch::CommandBatch(CommandBatch&&)noexcept=default;
CommandBatch& CommandBatch::operator=(CommandBatch&&)noexcept=default;
void CommandBatch::encode(int bits,bool complex,Operation op,const std::shared_ptr<detail::BufferStorage>& a,
                         const std::shared_ptr<detail::BufferStorage>& b,const std::shared_ptr<detail::BufferStorage>& out,std::size_t count){
    if(!impl_||impl_->submitted)throw std::logic_error("batch already submitted or moved");
    validate(bits,count);int operation=int(op);
    if(!operation_is_valid(op)||operation_is_complex(op)!=complex)throw std::invalid_argument("operation and buffer format mismatch");
    impl_->retain(a);impl_->retain(b);impl_->retain(out);if(!count)return;
    auto state=impl_->engine->pipeline(bits,operation);
    if(!impl_->encoder){impl_->encoder=[impl_->command computeCommandEncoder];if(!impl_->encoder)throw std::runtime_error("Metal encoder allocation failed");}
    else [impl_->encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    auto encoder=impl_->encoder;[encoder setComputePipelineState:state];
    [encoder setBuffer:a->buffer offset:0 atIndex:0];[encoder setBuffer:b->buffer offset:0 atIndex:1];[encoder setBuffer:out->buffer offset:0 atIndex:2];
    Params params={std::uint32_t(count),std::uint32_t(operation),0,std::uint32_t(count),1};[encoder setBytes:&params length:sizeof(params) atIndex:3];
    NSUInteger group=impl_->engine->group_size(state,bits,operation);
    [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(group,1,1)];
}
void CommandBatch::encode_tree_sum(int bits,bool complex,const std::shared_ptr<detail::BufferStorage>& input,
                                   const std::shared_ptr<detail::BufferStorage>& out,std::size_t count){
    if(!impl_||impl_->submitted)throw std::logic_error("batch already submitted or moved");
    validate(bits,count);impl_->retain(input);impl_->retain(out);
    bool cooperative=impl_->engine->options.cooperative_reductions;
    int operation=(complex?102:101)+(cooperative?0:2);auto state=impl_->engine->pipeline(bits,operation);
    unsigned group=impl_->engine->group_size(state,bits,operation);
    if(cooperative){
        unsigned limit=std::min(group,complex?64u:128u);group=1;while(group<=limit/2)group*=2;
        if(group%state.threadExecutionWidth)throw std::runtime_error("reduction group cannot fit one SIMD group");
    }
    std::size_t span=cooperative?2*group:2;
    auto reduced=[&](std::size_t n){return std::max(std::size_t(1),n/span+(n%span!=0));};
    std::size_t stride=std::size_t(bits/8+12)*(complex?2:1),first=reduced(count),second=reduced(first);
    std::shared_ptr<detail::BufferStorage> scratch[2];
    if(first>1){scratch[0]=impl_->engine->allocate(checked_size(first,stride));impl_->retain(scratch[0]);}
    if(second>1){scratch[1]=impl_->engine->allocate(checked_size(second,stride));impl_->retain(scratch[1]);}
    auto source=input;unsigned level=0;
    do {
        std::size_t next=reduced(count);
        auto destination=next==1?out:scratch[level%2];
        if(!impl_->encoder){impl_->encoder=[impl_->command computeCommandEncoder];if(!impl_->encoder)throw std::runtime_error("Metal encoder allocation failed");}
        else [impl_->encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        auto encoder=impl_->encoder;[encoder setComputePipelineState:state];
        [encoder setBuffer:source->buffer offset:0 atIndex:0];[encoder setBuffer:destination->buffer offset:0 atIndex:2];
        Params params={std::uint32_t(count),std::uint32_t(operation),0,0,1};[encoder setBytes:&params length:sizeof(params) atIndex:3];
        if(cooperative)[encoder dispatchThreadgroups:MTLSizeMake(next,1,1) threadsPerThreadgroup:MTLSizeMake(group,1,1)];
        else [encoder dispatchThreads:MTLSizeMake(next,1,1) threadsPerThreadgroup:MTLSizeMake(group,1,1)];
        source=destination;count=next;++level;
    }while(count>1);
}
Submission CommandBatch::submit(){
    if(!impl_||impl_->submitted)throw std::logic_error("batch already submitted or moved");
    auto ticket=std::make_shared<Submission::Impl>();ticket->resources=impl_->resources;ticket->command=impl_->command;ticket->start=impl_->start;
    std::size_t claimed=0;
    for(auto& r:impl_->resources){bool expected=false;if(!r->busy.compare_exchange_strong(expected,true)){
        for(std::size_t i=0;i<claimed;++i)impl_->resources[i]->busy.store(false);
        ticket->resources.clear();ticket->command=nil;throw std::logic_error("buffer belongs to an unwaited submission");}
        ++claimed;
    }
    if(impl_->encoder){[impl_->encoder endEncoding];impl_->encoder=nil;}
    impl_->submitted=true;[impl_->command commit];return Submission(std::move(ticket));
}
bool Submission::ready()const{
    if(!impl_)throw std::logic_error("empty submission");
    return impl_->command.status==MTLCommandBufferStatusCompleted||impl_->command.status==MTLCommandBufferStatusError;
}
Timing Submission::wait(){if(!impl_)throw std::logic_error("empty submission");impl_->finish();if(!impl_->error.empty())throw std::runtime_error("Metal execution: "+impl_->error);return impl_->timing;}
std::shared_ptr<detail::BufferStorage> Engine::allocate(std::size_t bytes){return impl->allocate(bytes);}
CommandBatch Engine::batch(){return CommandBatch(std::make_unique<CommandBatch::Impl>(impl));}
Engine::Engine(EngineOptions options):impl(new Impl(options)){} Engine::~Engine()=default;
std::string Engine::device_name()const{return std::string([impl->device.name UTF8String]);}
PipelineInfo Engine::pipeline_info(int bits,Operation op){
    validate(bits,0);int operation=int(op);if(!operation_is_valid(op))throw std::invalid_argument("unknown arithmetic operation");
    auto state=impl->pipeline(bits,operation);return {unsigned(state.threadExecutionWidth),unsigned(state.maxTotalThreadsPerThreadgroup),impl->group_size(state,bits,operation)};
}
Timing Engine::run(int bits,Operation op,const void* a,const void* b,void* out,std::size_t count) {
    validate(bits,count);int operation=int(op);
    if(!operation_is_valid(op))throw std::invalid_argument("unknown arithmetic operation");
    if(!count)return {0,0};
    if(!a||(!b&&!operation_is_unary(op))||!out)throw std::invalid_argument("null arithmetic buffer");
    std::size_t stride=(bits/8+12)*(operation_is_complex(op)?2:1),bytes=checked_size(count,stride);
    return impl->dispatch(bits,operation,a,bytes,operation_is_unary(op)?a:b,operation_is_unary(op)?0:bytes,out,bytes,count,0);
}
Timing Engine::run_unary(int bits,Operation op,const void* a,void* out,std::size_t count){
    if(!operation_is_unary(op))throw std::invalid_argument("operation requires two inputs");
    return run(bits,op,a,nullptr,out,count);
}
Timing Engine::recurrence(int bits,const void* seeds,const void* weights,void* out,std::size_t count,unsigned steps,unsigned states_per_weight) {
    validate(bits,count);
    if(!states_per_weight||count%states_per_weight)throw std::invalid_argument("count must be divisible by states_per_weight");
    if(!count)return {0,0};
    if(!seeds||!out||(steps&&!weights))throw std::invalid_argument("null recurrence buffer");
    // Protect all shader index expressions, not just host byte arithmetic.
    if(count>std::numeric_limits<std::uint32_t>::max()/4||
       (steps&&std::uint64_t(steps)*4*(count/states_per_weight)>std::numeric_limits<std::uint32_t>::max()))throw std::invalid_argument("recurrence exceeds 32-bit indexing");
    std::size_t stride=2*(bits/8+12),bytes=checked_size(count,stride);
    // Metal requires a bound weights buffer even for a zero-step dispatch.
    auto seed_bytes=checked_size(bytes,4),weight_bytes=checked_size(bytes/states_per_weight,std::size_t(steps)*4);
    if(!steps)return impl->dispatch(bits,100,seeds,seed_bytes,seeds,stride,out,bytes,count,0,states_per_weight);
    return impl->dispatch(bits,100,seeds,seed_bytes,weights,weight_bytes,out,bytes,count,steps,states_per_weight);
}
}
