#import <Foundation/Foundation.h>
#include "limbforge/batched_linalg.hpp"
#include "engine_internal.hpp"
#include "batched_linalg_source.hpp"
#include <cstring>
#include <map>
#include <chrono>
#include <unistd.h>
namespace limbforge {
namespace {
using I=detail::Internal;using O=detail::Operand;using S=std::shared_ptr<detail::BufferStorage>;
struct Params {std::uint64_t count,m,n,k,sa,sb,sc;std::int32_t n0;std::uint32_t flags;};
std::size_t mul_size(std::size_t a,std::size_t b){if(b&&a>std::size_t(-1)/b)throw std::invalid_argument("batched algebra: size overflow");return a*b;}
std::size_t extent(std::size_t count,std::size_t stride,std::size_t matrix){if(!count||!matrix)return 0;auto n=mul_size(count-1,stride);if(matrix>std::size_t(-1)-n)throw std::invalid_argument("batched algebra: stride overflow");return n+matrix;}
void bits_ok(int bits){if(bits<64||bits>1024||bits%32)throw std::invalid_argument("bits must be a multiple of 32 in [64,1024]");}
void grid_ok(std::size_t n){if(n>std::size_t(UINT32_MAX))throw std::invalid_argument("batched algebra: grid exceeds 32-bit indexing");}
void size_ok(O o,std::size_t n){if(o.size<n)throw std::invalid_argument("batched algebra: undersized buffer");}
void distinct(O out,std::initializer_list<O> in){for(auto a:in)if(out.storage&&out.storage==a.storage)throw std::invalid_argument("batched algebra: output aliases an input");}
S retain(CommandBatch& b,O o,std::size_t n,bool written=false,bool final=true){size_ok(o,n);if(!n&&!o.storage)return I::scratch(b,16);if(!written&&final)I::require_final(b,o.storage);I::retain(b,o.storage,written);return o.storage;}
std::string message(NSError* e){return e?std::string(e.localizedDescription.UTF8String):"unknown Metal error";}
}
struct BatchedLinalg::Impl {
    id<MTLDevice> device;std::map<int,id<MTLLibrary>> libs;
    std::map<std::tuple<int,std::string,bool>,id<MTLComputePipelineState>> pipelines;
    explicit Impl(Engine& e):device(I::device(e)){}
    id<MTLComputePipelineState> pipeline(int bits,const char* name,bool fused=false){
        auto key=std::make_tuple(bits,std::string(name),fused);auto it=pipelines.find(key);if(it!=pipelines.end())return it->second;
        @autoreleasepool {auto& lib=libs[bits];NSError* error=nil;if(!lib){auto options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion3_1;options.mathMode=MTLMathModeSafe;
            NSString* source=[NSString stringWithFormat:@"#define LF_BITS %d\n%s",bits,limbforge_batched_linalg_source];lib=[device newLibraryWithSource:source options:options error:&error];
            if(!lib)throw std::runtime_error("batched algebra Metal compilation: "+message(error));}
        auto constants=[MTLFunctionConstantValues new];[constants setConstantValue:&fused type:MTLDataTypeBool atIndex:0];
        auto f=[lib newFunctionWithName:[NSString stringWithUTF8String:name] constantValues:constants error:&error];
        if(!f)throw std::runtime_error("batched algebra Metal function: "+message(error));auto p=[device newComputePipelineStateWithFunction:f error:&error];
        if(!p)throw std::runtime_error("batched algebra Metal pipeline: "+message(error));pipelines.emplace(key,p);return p;}
    }
    void pass(CommandBatch& b,int bits,const char* name,const Params& p,std::initializer_list<std::pair<unsigned,S>> buffers,std::size_t count,bool fused=false,bool tiled=false,bool components=false){
        if(!count)return;if(I::device(b)!=device)throw std::invalid_argument("batched algebra: foreign batch device");
        auto state=pipeline(bits,name,fused);I::keep(b,state);auto encoder=I::compute(b);[encoder setComputePipelineState:state];
        for(auto& bind:buffers)[encoder setBuffer:bind.second->buffer offset:0 atIndex:bind.first];[encoder setBytes:&p length:sizeof p atIndex:3];
        if(tiled){if(state.maxTotalThreadsPerThreadgroup<32)throw std::runtime_error("batched GEMM cannot fit a SIMD group");
            auto grid=p.m==4&&p.n==4&&p.k==4?MTLSizeMake((p.count+1)/2,1,1):MTLSizeMake((p.n+3)/4,(p.m+7)/8,p.count);
            [encoder dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(32,1,1)];}
        else{NSUInteger tg=std::min<NSUInteger>(state.threadExecutionWidth,state.maxTotalThreadsPerThreadgroup);
            [encoder dispatchThreads:MTLSizeMake(count,components?2:1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];}
    }
};
BatchedLinalg::BatchedLinalg(Engine& e):engine_(&e),impl_(std::make_unique<Impl>(e)){}
BatchedLinalg::~BatchedLinalg()=default;
namespace {
Timing host_call(Engine& e,int bits,bool complex,const std::size_t* sizes,const void* const* pointers,bool accumulate,
                 const std::function<void(CommandBatch&,O,O,O,O)>& encode){
    bits_ok(bits);auto start=std::chrono::steady_clock::now();auto b=e.batch();S storage[4];O operands[4];auto element=std::size_t(bits/8+12)*(complex?2:1);
    for(int i=0;i<4;++i){auto bytes=mul_size(sizes[i],element);if(bytes&&!pointers[i])throw std::invalid_argument("batched algebra: null host operand");
        storage[i]=I::scratch(b,bytes);operands[i]={storage[i],sizes[i]};if(bytes&&(i<3||accumulate))std::memcpy(storage[i]->buffer.contents,pointers[i],bytes);}
    encode(b,operands[0],operands[1],operands[2],operands[3]);auto timing=b.submit().wait();if(sizes[3])std::memcpy(const_cast<void*>(pointers[3]),storage[3]->buffer.contents,mul_size(sizes[3],element));
    timing.wall_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return timing;
}
}
Timing BatchedLinalg::gemm(int bits,bool complex,const StridedGemm& s,const void* A,const void* B,void* C){
    std::size_t sizes[4]={extent(s.count,s.stride_a,mul_size(s.m,s.k)),extent(s.count,s.stride_b,mul_size(s.k,s.n)),0,extent(s.count,s.stride_c,mul_size(s.m,s.n))};const void* pointers[4]={A,B,nullptr,C};
    // Preserve padding between output matrices even in overwrite mode.
    return host_call(*engine_,bits,complex,sizes,pointers,true,[&](CommandBatch& b,O a,O bb,O,O c){encode_gemm(b,bits,complex,s,a,bb,c);});
}
Timing BatchedLinalg::power_moments(int bits,bool complex,const PowerMoments& s,const void* E,const void* y,const void* W,void* out){
    auto ek=mul_size(s.count,s.steps);std::size_t sizes[4]={ek,ek,mul_size(ek,s.ncols),mul_size(mul_size(s.count,std::size_t(s.nmax)+1),s.ncols)};const void* pointers[4]={E,y,W,out};
    return host_call(*engine_,bits,complex,sizes,pointers,s.accumulate,[&](CommandBatch& b,O e,O yy,O w,O c){encode_power(b,bits,complex,s,e,yy,w,c);});
}
void BatchedLinalg::encode_gemm(CommandBatch& b,int bits,bool complex,const StridedGemm& s,O A,O B,O C){
    bits_ok(bits);auto as=mul_size(s.m,s.k),bs=mul_size(s.k,s.n),cs=mul_size(s.m,s.n);grid_ok(mul_size(s.count,cs));
    if(s.count>1&&s.stride_c<cs)throw std::invalid_argument("batched GEMM: output matrices overlap");
    size_ok(A,extent(s.count,s.stride_a,as));size_ok(B,extent(s.count,s.stride_b,bs));size_ok(C,extent(s.count,s.stride_c,cs));distinct(C,{A,B});
    if(!s.count||!cs)return;
    auto a=retain(b,A,extent(s.count,s.stride_a,as)),bb=retain(b,B,extent(s.count,s.stride_b,bs)),c=retain(b,C,extent(s.count,s.stride_c,cs),true);
    Params p{s.count,s.m,s.n,s.k,s.stride_a,s.stride_b,s.stride_c,0,std::uint32_t(s.accumulate)|(std::uint32_t(s.negative)<<1)};
    const bool small=s.m==4&&s.n==4&&s.k==4;
    if(complex)impl_->pass(b,bits,"batch_private_complex",p,{{0,a},{1,bb},{2,c}},s.count*cs,s.fused,false,true);
    else impl_->pass(b,bits,small?"batch_gemm4_real":"batch_gemm_real",p,{{0,a},{1,bb},{2,c}},s.count*cs,s.fused,true);
}
void BatchedLinalg::encode_power(CommandBatch& b,int bits,bool complex,const PowerMoments& s,O E,O y,O W,O out){
    bits_ok(bits);auto rows=std::size_t(s.nmax)+1,ek=mul_size(s.count,s.steps),wk=mul_size(ek,s.ncols),pk=mul_size(ek,rows);
    grid_ok(ek);grid_ok(mul_size(mul_size(s.count,rows),s.ncols));size_ok(E,ek);size_ok(y,ek);size_ok(W,wk);distinct(out,{E,y,W});
    StridedGemm shape{s.count,rows,s.ncols,s.steps,rows*s.steps,std::size_t(s.steps)*s.ncols,rows*s.ncols,s.accumulate,s.fused};
    size_ok(out,mul_size(s.count,shape.stride_c));if(!s.count||!s.ncols)return;
    std::size_t bytes=mul_size(pk,std::size_t(bits/8+12)*(complex?2:1));auto powers=I::scratch(b,bytes);
    if(ek){auto e=retain(b,E,ek),yy=retain(b,y,ek);Params p{s.count,rows,s.ncols,s.steps,0,0,0,s.n0,0};
        impl_->pass(b,bits,complex?"batch_powers_complex":"batch_powers_real",p,{{0,e},{1,yy},{2,powers}},ek);}
    encode_gemm(b,bits,complex,shape,{powers,pk},W,out);
}
void BatchedLinalg::encode_normal(CommandBatch& b,int bits,std::size_t rows,std::size_t cols,O J,O g,O A,O rhs,bool fused){
    bits_ok(bits);auto jn=mul_size(rows,cols),an=mul_size(cols,cols),grid=mul_size(cols,cols+1);grid_ok(grid);
    size_ok(J,jn);size_ok(g,rows);size_ok(A,an);size_ok(rhs,cols);distinct(A,{J,g,rhs});distinct(rhs,{J,g});if(!cols)return;
    auto j=retain(b,J,jn),gg=retain(b,g,rows),a=retain(b,A,an,true),r=retain(b,rhs,cols,true);Params p{1,0,cols,rows,0,0,0,0,0};
    impl_->pass(b,bits,"batch_normal",p,{{0,j},{1,gg},{2,a},{4,r}},grid,fused);
}
void BatchedLinalg::encode_trials(CommandBatch& b,int bits,const CholeskyTrials& s,O A,O D,O mu,O L,O status){
    bits_ok(bits);auto n2=mul_size(s.n,s.n),ln=mul_size(s.count,n2);grid_ok(ln);size_ok(A,n2);size_ok(D,s.n);size_ok(mu,s.count);size_ok(L,ln);size_ok(status,s.count);
    distinct(L,{A,D,mu});if(!s.count)return;if(!s.n)throw std::invalid_argument("cholesky_trials: n must be positive");
    auto a=retain(b,A,n2),d=retain(b,D,s.n),u=retain(b,mu,s.count),l=retain(b,L,ln,true),st=retain(b,status,s.count,true);
    Params p{s.count,0,s.n,0,0,0,0,0,0};impl_->pass(b,bits,"batch_chol_init",p,{{0,a},{1,d},{2,l},{4,u},{5,st}},ln);
    for(std::size_t k=0;k<s.n;++k){p.k=k;impl_->pass(b,bits,"batch_chol_pivot",p,{{2,l},{5,st}},s.count);
        if(k+1<s.n){auto remaining=s.n-k-1;impl_->pass(b,bits,"batch_chol_column",p,{{2,l},{5,st}},s.count*remaining);impl_->pass(b,bits,"batch_chol_update",p,{{2,l},{5,st}},s.count*remaining*remaining);}}
    impl_->pass(b,bits,"batch_chol_finish",p,{{2,l},{5,st}},ln);
}
void BatchedLinalg::encode_augment(CommandBatch& b,int bits,std::size_t rows,std::size_t cols,O J,O g,O out){
    bits_ok(bits);auto jn=mul_size(rows,cols),on=mul_size(rows,cols+1);grid_ok(on);size_ok(J,jn);size_ok(g,rows);size_ok(out,on);if(!on)return;
    auto j=retain(b,J,jn),gg=retain(b,g,rows),o=retain(b,out,on,true);Params p{1,0,cols,rows,0,0,0,0,0};
    impl_->pass(b,bits,"batch_augment",p,{{0,j},{1,gg},{2,o}},on);
}
void BatchedLinalg::encode_extract(CommandBatch& b,int bits,std::size_t cols,O gram,O A,O rhs,const LinalgTicket& ticket){
    bits_ok(bits);auto n=mul_size(cols,cols);grid_ok(n);size_ok(A,n);size_ok(rhs,cols);distinct(A,{rhs});if(!cols)return;
    auto g=retain(b,gram,mul_size(cols+1,cols+1),false,false),a=retain(b,A,n,true),r=retain(b,rhs,cols,true);Params p{1,0,cols,0,0,0,0,0,0};
    impl_->pass(b,bits,"batch_extract",p,{{0,g},{2,a},{4,r}},n);
    I::provisional(b,a);I::provisional(b,r);
    // Linalg's completion repairs gram first. Copy again afterwards; never lose host-fallback outputs.
    I::on_completion(b,[g,a,r,cols,bits,ticket]{if(!ticket.report().fallback_outputs)return;
        auto bytes=std::size_t(bits/8+12);auto src=static_cast<const char*>(g->buffer.contents);auto dst=static_cast<char*>(a->buffer.contents);auto rr=static_cast<char*>(r->buffer.contents);
        for(std::size_t i=0;i<cols;++i){std::memcpy(dst+i*cols*bytes,src+i*(cols+1)*bytes,cols*bytes);std::memcpy(rr+i*bytes,src+(i*(cols+1)+cols)*bytes,bytes);}});
}
void BatchedLinalg::encode_solve(CommandBatch& b,int bits,std::size_t count,std::size_t n,std::size_t nrhs,O L,O status,O B,O X){
    bits_ok(bits);auto n2=mul_size(n,n),bn=mul_size(n,nrhs),xn=mul_size(count,bn);grid_ok(xn);size_ok(L,mul_size(count,n2));size_ok(status,count);size_ok(B,bn);size_ok(X,xn);distinct(X,{L,B});
    if(!xn)return;auto l=retain(b,L,count*n2),st=retain(b,status,count),bb=retain(b,B,bn),x=retain(b,X,xn,true);Params p{count,nrhs,n,0,0,0,0,0,0};
    impl_->pass(b,bits,"batch_chol_solve",p,{{0,l},{1,st},{2,x},{4,bb}},count*nrhs);
}
void BatchedLinalg::encode_inline(CommandBatch& b,int bits,const void* allocation,std::size_t bytes,const InlineComplexRecord* records,std::size_t count,O out){
    bits_ok(bits);grid_ok(count);size_ok(out,count);if(!count)return;std::size_t page=std::size_t(getpagesize());
    if(I::device(b)!=impl_->device)throw std::invalid_argument("inline staging: foreign batch device");
    if(!allocation||reinterpret_cast<std::uintptr_t>(allocation)%page||!bytes||bytes%page||!records)throw std::invalid_argument("inline staging: allocation must own complete aligned pages");
    const std::size_t source_bytes=8*((std::size_t(bits)+63)/64);
    for(std::size_t i=0;i<count;++i)for(auto offset:{records[i].real_offset,records[i].imag_offset})
        if(offset%8||offset>bytes||source_bytes>bytes-offset)throw std::invalid_argument("inline staging: significand out of bounds or unaligned");
    auto input=[impl_->device newBufferWithBytesNoCopy:const_cast<void*>(allocation) length:bytes options:MTLResourceStorageModeShared deallocator:nil];
    if(!input)throw std::runtime_error("inline staging: Metal wrapping failed");I::keep(b,input);
    auto meta=I::scratch(b,mul_size(count,sizeof(InlineComplexRecord)));std::memcpy(meta->buffer.contents,records,count*sizeof(InlineComplexRecord));
    auto dest=retain(b,out,count,true);auto state=impl_->pipeline(bits,"batch_inline_complex");I::keep(b,state);auto enc=I::compute(b);[enc setComputePipelineState:state];
    [enc setBuffer:input offset:0 atIndex:0];[enc setBuffer:meta->buffer offset:0 atIndex:1];[enc setBuffer:dest->buffer offset:0 atIndex:2];Params p{count,0,0,0,0,0,0,0,0};[enc setBytes:&p length:sizeof p atIndex:3];
    [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(state.threadExecutionWidth,1,1)];
}
void BatchedLinalg::encode_polynomial(CommandBatch& b,int bits,const PolynomialRecurrence& s,O start,O cp,O cq,O Y,O Ep,O Eq,O out){
    bits_ok(bits);if(!s.lanes_per_weight)throw std::invalid_argument("polynomial recurrence: zero sharing");
    auto groups=s.lanes/s.lanes_per_weight+(s.lanes%s.lanes_per_weight!=0);
    if(s.coefficient_sets!=1&&s.coefficient_sets!=groups)throw std::invalid_argument("polynomial recurrence: coefficient sets must be 1 or weight groups");
    auto coeff=mul_size(mul_size(s.coefficient_sets,4),s.terms),yn=mul_size(s.steps,groups),en=mul_size(yn,4),vn=mul_size(s.lanes,4),on=mul_size(vn,s.all_steps?std::size_t(s.steps)+1:1);
    grid_ok(s.lanes);size_ok(start,vn);size_ok(cp,coeff);size_ok(cq,coeff);size_ok(Y,yn);size_ok(Ep,en);size_ok(Eq,en);size_ok(out,on);distinct(out,{start,cp,cq,Y,Ep,Eq});if(!s.lanes)return;
    auto st=retain(b,start,vn),p=retain(b,cp,coeff),q=retain(b,cq,coeff),y=retain(b,Y,yn),ep=retain(b,Ep,en),eq=retain(b,Eq,en),o=retain(b,out,on,true);
    if(I::device(b)!=impl_->device)throw std::invalid_argument("polynomial recurrence: foreign batch device");
    struct PolyParams {std::uint64_t lanes,groups,sets;std::uint32_t steps,terms,share,flags;};
    PolyParams params{s.lanes,groups,s.coefficient_sets,s.steps,s.terms,s.lanes_per_weight,std::uint32_t(s.all_steps)|(std::uint32_t(s.reverse)<<1)};
    auto state=impl_->pipeline(bits,"batch_polynomial_recurrence",s.fused);I::keep(b,state);auto enc=I::compute(b);[enc setComputePipelineState:state];
    for(auto& v:std::initializer_list<std::pair<unsigned,S>>{{0,st},{1,p},{2,o},{4,q},{5,y},{6,ep},{7,eq}})[enc setBuffer:v.second->buffer offset:0 atIndex:v.first];
    [enc setBytes:&params length:sizeof params atIndex:3];[enc dispatchThreads:MTLSizeMake(s.lanes,1,1) threadsPerThreadgroup:MTLSizeMake(state.threadExecutionWidth,1,1)];
}
std::future<std::vector<Timing>> wait_all_async(std::vector<Submission> submissions){return std::async(std::launch::async,[s=std::move(submissions)]()mutable{
    std::vector<Timing> result(s.size());std::exception_ptr error;for(std::size_t i=0;i<s.size();++i)try{result[i]=s[i].wait();}catch(...){if(!error)error=std::current_exception();}
    if(error)std::rethrow_exception(error);return result;});}
}
