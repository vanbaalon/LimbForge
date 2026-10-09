#import <Foundation/Foundation.h>
#include "prototype.hpp"
#include "limbforge/linalg.hpp"
#include "engine_internal.hpp"
#include "exact_gemm4_source.hpp"
#include <chrono>
#include <climits>
#include <cstring>
#include <map>
#include <mutex>
namespace wolfnum::experimental {
using namespace limbforge;using I=detail::Internal;using O=detail::Operand;using S=std::shared_ptr<detail::BufferStorage>;
struct Params {std::uint64_t count,sa,sb,sc;std::uint32_t accumulate,negative;};
static std::size_t mul(std::size_t a,std::size_t b){if(b&&a>SIZE_MAX/b)throw std::invalid_argument("exact GEMM4: size overflow");return a*b;}
static std::size_t extent(std::size_t n,std::size_t stride){if(!n)return 0;auto v=mul(n-1,stride);if(v>SIZE_MAX-16)throw std::invalid_argument("exact GEMM4: stride overflow");return v+16;}
static void bits_ok(int bits){if(bits<64||bits>1024||bits%32)throw std::invalid_argument("exact GEMM4: unsupported precision");}
struct ExactGemm4State {std::atomic<bool> done{false};ExactGemm4Report report;};
bool ExactGemm4Ticket::resolved()const{return state_&&state_->done.load(std::memory_order_acquire);}
ExactGemm4Report ExactGemm4Ticket::report()const{if(!resolved())throw std::logic_error("exact GEMM4: wait for submission before reading report");return state_->report;}
struct ExactGemm4::Impl {
    id<MTLDevice> device;std::map<std::pair<int,bool>,id<MTLComputePipelineState>> cache;std::mutex mutex;
    explicit Impl(Engine& e):device(I::device(e)){}
    id<MTLComputePipelineState> pipeline(int bits,bool complex){std::lock_guard<std::mutex> lock(mutex);auto key=std::make_pair(bits,complex);auto it=cache.find(key);if(it!=cache.end())return it->second;
        @autoreleasepool {NSError* error=nil;auto options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion3_1;options.mathMode=MTLMathModeSafe;
            NSString* source=[NSString stringWithFormat:@"#define LF_BITS %d\n%s",bits,exact_gemm4_source];auto lib=[device newLibraryWithSource:source options:options error:&error];
            auto fail=[&]{throw std::runtime_error(std::string("WolfNum exact GEMM4 Metal: ")+(error?error.localizedDescription.UTF8String:"unknown error"));};if(!lib)fail();
            auto constants=[MTLFunctionConstantValues new];[constants setConstantValue:&complex type:MTLDataTypeBool atIndex:0];
            auto fn=[lib newFunctionWithName:@"exact_gemm4" constantValues:constants error:&error];if(!fn)fail();auto state=[device newComputePipelineStateWithFunction:fn error:&error];if(!state)fail();cache.emplace(key,state);return state;}
    }
};
ExactGemm4::ExactGemm4(Engine& e):engine_(&e),impl_(std::make_unique<Impl>(e)){}
ExactGemm4::~ExactGemm4()=default;
template<int N> static void repair(bool complex,const ExactGemm4Shape& shape,const S& a,const S& b,const S& old,const S& out,const S& flags,ExactGemm4Report& report){
    auto* ap=static_cast<const Number<N>*>(a->buffer.contents);auto* bp=static_cast<const Number<N>*>(b->buffer.contents);
    auto* cp=static_cast<Number<N>*>(out->buffer.contents);auto* initial=old?static_cast<const Number<N>*>(old->buffer.contents):nullptr;
    auto* mask=static_cast<const std::uint32_t*>(flags->buffer.contents);const unsigned components=complex?2:1,terms=complex?8:4;
    for(std::size_t t=0;t<shape.count;++t)for(unsigned cell=0;cell<16;++cell)for(unsigned part=0;part<components;++part){
        const auto index=components*(t*shape.stride_c+cell)+part;if(!mask[components*(t*16+cell)+part]){++report.gpu_components;continue;}
        Number<N> left[8],right[8];for(unsigned q=0;q<terms;++q){unsigned k=complex?q/2:q,sub=complex?q%2:0;
            left[q]=ap[components*(t*shape.stride_a+(cell/4)*4+k)+sub];right[q]=bp[components*(t*shape.stride_b+k*4+cell%4)+(complex?(sub?1-part:part):0)];
            if(complex&&!part&&sub)left[q]=negate(left[q]);}
        auto value=limbforge::exact_dot_add<N>(initial?initial+index:nullptr,false,left,1,right,1,terms);cp[index]=shape.negative?negate(value):value;++report.host_components;
    }
}
ExactGemm4Ticket ExactGemm4::encode(CommandBatch& batch,int bits,bool complex,const ExactGemm4Shape& shape,O a,O b,O c){
    bits_ok(bits);I::device(batch);auto na=extent(shape.count,shape.stride_a),nb=extent(shape.count,shape.stride_b),nc=extent(shape.count,shape.stride_c);
    if(shape.count>1&&shape.stride_c<16)throw std::invalid_argument("exact GEMM4: output matrices overlap");
    auto entries=mul(shape.count,16);if(entries>UINT32_MAX)throw std::invalid_argument("exact GEMM4: grid exceeds 32 bits");
    if(a.size<na||b.size<nb||c.size<nc)throw std::invalid_argument("exact GEMM4: undersized buffer");
    if(c.storage&&(c.storage==a.storage||c.storage==b.storage))throw std::invalid_argument("exact GEMM4: output aliases input");
    auto state=std::make_shared<ExactGemm4State>();if(!shape.count){I::on_completion(batch,[state]{state->done.store(true,std::memory_order_release);});return ExactGemm4Ticket(state);}
    I::require_final(batch,a.storage);I::require_final(batch,b.storage);if(shape.accumulate)I::require_final(batch,c.storage);
    I::retain(batch,a.storage);I::retain(batch,b.storage);I::retain(batch,c.storage,true);
    const auto element=std::size_t(bits/8+12)*(complex?2:1);const auto ab=mul(na,element),bb=mul(nb,element),cb=mul(nc,element);
    auto snap_a=I::scratch(batch,ab),snap_b=I::scratch(batch,bb),flags=I::scratch(batch,mul(entries,(complex?2:1)*sizeof(std::uint32_t)));S old;
    I::copy(batch,a.storage,snap_a,ab);I::copy(batch,b.storage,snap_b,bb);if(shape.accumulate){old=I::scratch(batch,cb);I::copy(batch,c.storage,old,cb);}
    auto pipeline=impl_->pipeline(bits,complex);I::keep(batch,pipeline);auto encoder=I::compute(batch);[encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a.storage->buffer offset:0 atIndex:0];[encoder setBuffer:b.storage->buffer offset:0 atIndex:1];[encoder setBuffer:c.storage->buffer offset:0 atIndex:2];[encoder setBuffer:flags->buffer offset:0 atIndex:4];
    Params p{shape.count,shape.stride_a,shape.stride_b,shape.stride_c,std::uint32_t(shape.accumulate),std::uint32_t(shape.negative)};[encoder setBytes:&p length:sizeof p atIndex:3];
    auto tg=std::min<NSUInteger>(pipeline.threadExecutionWidth,pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(entries,complex?2:1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];auto provisional=I::provisional(batch,c.storage);
    I::on_completion(batch,[=]{switch(bits/32){
#define CASE(N) case N:repair<N>(complex,shape,snap_a,snap_b,old,c.storage,flags,state->report);break;
CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9) CASE(10) CASE(11) CASE(12) CASE(13) CASE(14) CASE(15) CASE(16) CASE(17) CASE(18) CASE(19) CASE(20) CASE(21) CASE(22) CASE(23) CASE(24) CASE(25) CASE(26) CASE(27) CASE(28) CASE(29) CASE(30) CASE(31) CASE(32)
#undef CASE
    }state->report.provisional_reads=provisional->load();state->done.store(true,std::memory_order_release);});return ExactGemm4Ticket(state);
}
Timing ExactGemm4::gemm(int bits,bool complex,const ExactGemm4Shape& s,const void* a,const void* b,void* c,ExactGemm4Report* report){
    bits_ok(bits);auto start=std::chrono::steady_clock::now();auto batch=engine_->batch();auto element=std::size_t(bits/8+12)*(complex?2:1);
    std::size_t sizes[3]={extent(s.count,s.stride_a),extent(s.count,s.stride_b),extent(s.count,s.stride_c)};const void* pointers[3]={a,b,c};O ops[3];
    for(int i=0;i<3;++i){auto bytes=mul(sizes[i],element);if(bytes&&!pointers[i])throw std::invalid_argument("exact GEMM4: null host matrix");auto storage=I::scratch(batch,bytes);if(bytes)std::memcpy(storage->buffer.contents,pointers[i],bytes);ops[i]={storage,sizes[i]};}
    auto ticket=encode(batch,bits,complex,s,ops[0],ops[1],ops[2]);auto timing=batch.submit().wait();auto bytes=mul(sizes[2],element);if(bytes)std::memcpy(c,ops[2].storage->buffer.contents,bytes);
    if(report)*report=ticket.report();timing.wall_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return timing;
}
}
