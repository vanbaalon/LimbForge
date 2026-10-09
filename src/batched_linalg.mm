#import <Foundation/Foundation.h>
#include "limbforge/batched_linalg.hpp"
#include "engine_internal.hpp"
#include "batched_linalg_source.hpp"
#include <cstring>
#include <array>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <chrono>
#include <unistd.h>
#include <mutex>
#include <algorithm>
namespace limbforge {
namespace {
using I=detail::Internal;using O=detail::Operand;using S=std::shared_ptr<detail::BufferStorage>;
struct Params {std::uint64_t count,m,n,k,sa,sb,sc;std::int32_t n0;std::uint32_t flags;std::uint64_t row_offset=0;};
std::size_t mul_size(std::size_t a,std::size_t b){if(b&&a>std::size_t(-1)/b)throw std::invalid_argument("batched algebra: size overflow");return a*b;}
std::size_t extent(std::size_t count,std::size_t stride,std::size_t matrix){if(!count||!matrix)return 0;auto n=mul_size(count-1,stride);if(matrix>std::size_t(-1)-n)throw std::invalid_argument("batched algebra: stride overflow");return n+matrix;}
void bits_ok(int bits){if(bits<64||bits>1024||bits%32)throw std::invalid_argument("bits must be a multiple of 32 in [64,1024]");}
void grid_ok(std::size_t n){if(n>std::size_t(UINT32_MAX))throw std::invalid_argument("batched algebra: grid exceeds 32-bit indexing");}
void size_ok(O o,std::size_t n){if(o.size<n)throw std::invalid_argument("batched algebra: undersized buffer");}
void distinct(O out,std::initializer_list<O> in){for(auto a:in)if(out.storage&&out.storage==a.storage)throw std::invalid_argument("batched algebra: output aliases an input");}
S retain(CommandBatch& b,O o,std::size_t n,bool written=false,bool final=true){size_ok(o,n);if(!n&&!o.storage)return I::scratch(b,16);if(!written&&final)I::require_final(b,o.storage);I::retain(b,o.storage,written);return o.storage;}
std::string message(NSError* e){return e?std::string(e.localizedDescription.UTF8String):"unknown Metal error";}
}
struct BatchedWorkspace {S first,second;std::atomic<std::size_t> bytes{0};};
struct BatchedPipelines {
    id<MTLDevice> device;std::map<int,id<MTLLibrary>> libs;
    std::map<std::tuple<int,std::string,bool>,id<MTLComputePipelineState>> pipelines;
    std::mutex mutex;
    explicit BatchedPipelines(id<MTLDevice> d):device(d){}
    id<MTLComputePipelineState> pipeline(int bits,const char* name,bool fused=false){
        std::lock_guard<std::mutex> lock(mutex);auto key=std::make_tuple(bits,std::string(name),fused);auto it=pipelines.find(key);if(it!=pipelines.end())return it->second;
        @autoreleasepool {auto& lib=libs[bits];NSError* error=nil;if(!lib){auto options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion3_1;options.mathMode=MTLMathModeSafe;
            NSString* source=[NSString stringWithFormat:@"#define LF_BITS %d\n%s",bits,limbforge_batched_linalg_source];lib=[device newLibraryWithSource:source options:options error:&error];
            if(!lib)throw std::runtime_error("batched algebra Metal compilation: "+message(error));}
        auto constants=[MTLFunctionConstantValues new];[constants setConstantValue:&fused type:MTLDataTypeBool atIndex:0];
        auto f=[lib newFunctionWithName:[NSString stringWithUTF8String:name] constantValues:constants error:&error];
        if(!f)throw std::runtime_error("batched algebra Metal function: "+message(error));auto p=[device newComputePipelineStateWithFunction:f error:&error];
        if(!p)throw std::runtime_error("batched algebra Metal pipeline: "+message(error));pipelines.emplace(key,p);return p;}
    }
};
struct BatchedLinalg::Impl {
    mutable std::mutex workspace_mutex;
    std::vector<std::shared_ptr<BatchedWorkspace>> workspaces;
    std::vector<std::weak_ptr<BatchedWorkspace>> detached;
    id<MTLDevice> device;std::shared_ptr<BatchedPipelines> cache;
    explicit Impl(Engine& e):device(I::device(e)),cache(std::make_shared<BatchedPipelines>(device)){}
    id<MTLComputePipelineState> pipeline(int bits,const char* name,bool fused=false){return cache->pipeline(bits,name,fused);}
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
std::pair<O,O> BatchedLinalg::workspace(CommandBatch& b,int bits,bool complex,std::size_t first,std::size_t second){
    bits_ok(bits);I::device(b);auto element=std::size_t(bits/8+12)*(complex?2:1);
    const auto first_bytes=mul_size(first,element),second_bytes=mul_size(second,element);
    std::shared_ptr<BatchedWorkspace> selected;
    {std::lock_guard<std::mutex> lock(impl_->workspace_mutex);
        for(auto& w:impl_->workspaces)if(w.use_count()==1&&(!w->first||!w->first->busy.load())&&(!w->second||!w->second->busy.load())){selected=w;break;}
        if(!selected){selected=std::make_shared<BatchedWorkspace>();impl_->workspaces.push_back(selected);}}
    // Allocate on this unit's Engine, retaining the previous foreign-engine rejection.
    auto reserve=[&](S& storage,std::size_t bytes){if(!storage||storage->bytes<bytes){
        auto buffer=engine_->make_buffer<std::uint32_t>(std::max<std::size_t>(bytes/4,1));storage=detail::Access::operand(buffer).storage;}};
    reserve(selected->first,first_bytes);if(second_bytes)reserve(selected->second,second_bytes);
    selected->bytes.store((selected->first?selected->first->bytes:0)+(selected->second?selected->second->bytes:0));
    I::retain(b,selected->first,true);if(second_bytes)I::retain(b,selected->second,true);
    // Each call needs its own gram until host fallback repair completes at wait().
    I::on_completion(b,[held=selected]{});
    return {{selected->first,first},{selected->second,second}};
}
BatchedWorkspaces BatchedLinalg::workspaces()const{
    BatchedWorkspaces result;std::lock_guard<std::mutex> lock(impl_->workspace_mutex);
    for(auto& w:impl_->workspaces){if(w.use_count()==1)result.idle_bytes+=w->bytes.load();
        else{result.busy_bytes+=w->bytes.load();++result.busy_workspaces;}}
    for(auto& weak:impl_->detached)if(auto w=weak.lock()){result.busy_bytes+=w->bytes.load();++result.busy_workspaces;}
    return result;
}
BatchedWorkspaces BatchedLinalg::release_workspaces(){
    BatchedWorkspaces result;std::vector<std::shared_ptr<BatchedWorkspace>> dropped;
    {std::lock_guard<std::mutex> lock(impl_->workspace_mutex);
        impl_->detached.erase(std::remove_if(impl_->detached.begin(),impl_->detached.end(),[](const auto& w){return w.expired();}),impl_->detached.end());
        for(auto& weak:impl_->detached)if(auto w=weak.lock()){result.busy_bytes+=w->bytes.load();++result.busy_workspaces;}
        for(auto& w:impl_->workspaces){if(w.use_count()==1)result.idle_bytes+=w->bytes.load();
            else{result.busy_bytes+=w->bytes.load();++result.busy_workspaces;impl_->detached.push_back(w);}}
        dropped.swap(impl_->workspaces);
    }
    return result;
}
namespace {
using WarmKeys=std::set<std::pair<std::string,bool>>;
WarmKeys prewarm_keys(const BatchedPrewarm& r){
    for(int bits:r.bits)bits_ok(bits);
    if(r.power_storage!=PowerStorage::full_table&&r.power_storage!=PowerStorage::compact)throw std::invalid_argument("prewarm: invalid power storage mode");
    WarmKeys keys;
    auto gemm=[&](const StridedGemm& s,bool complex){auto cs=mul_size(s.m,s.n);grid_ok(mul_size(s.count,cs));
        if(s.count>1&&s.stride_c<cs)throw std::invalid_argument("prewarm GEMM: output matrices overlap");
        extent(s.count,s.stride_a,mul_size(s.m,s.k));extent(s.count,s.stride_b,mul_size(s.k,s.n));extent(s.count,s.stride_c,cs);
        if(s.count&&cs)keys.emplace(complex?"batch_private_complex":s.m==4&&s.n==4&&s.k==4?"batch_gemm4_real":"batch_gemm_real",s.fused);
    };
    for(auto& s:r.real_gemm)gemm(s,false);for(auto& s:r.complex_gemm)gemm(s,true);
    auto power=[&](const PowerMoments& s,bool complex){auto rows=std::size_t(s.nmax)+1,ek=mul_size(s.count,s.steps);grid_ok(ek);mul_size(ek,rows);mul_size(ek,s.ncols);grid_ok(mul_size(mul_size(s.count,rows),s.ncols));
        if(!s.count||!s.ncols)return;bool compact=r.power_storage==PowerStorage::compact&&s.steps;
        if(ek){if(compact){keys.emplace(complex?"batch_panel_seed_complex":"batch_panel_seed_real",false);keys.emplace(complex?"batch_panel_powers_complex":"batch_panel_powers_real",false);}
            else keys.emplace(complex?"batch_powers_complex":"batch_powers_real",false);}
        auto add_gemm=[&](std::size_t m){StridedGemm g{s.count,m,s.ncols,s.steps,mul_size(m,s.steps),mul_size(s.steps,s.ncols),mul_size(m,s.ncols),s.accumulate,s.fused};gemm(g,complex);};
        if(compact){add_gemm(std::min<std::size_t>(8,rows));if(rows>8&&rows%8)add_gemm(rows%8);}else add_gemm(rows);
    };
    for(auto& s:r.real_power)power(s,false);for(auto& s:r.complex_power)power(s,true);
    for(bool fused:r.normal_fused)keys.emplace("batch_normal",fused);
    for(bool fused:r.polynomial_fused)keys.emplace("batch_polynomial_recurrence",fused);
    if(r.cholesky_trials)for(auto name:{"batch_chol_init","batch_chol_pivot","batch_chol_column","batch_chol_update","batch_chol_finish"})keys.emplace(name,false);
    if(r.cholesky_solve)keys.emplace("batch_chol_solve",false);
    if(r.exact_normal){keys.emplace("batch_augment",false);keys.emplace("batch_extract",false);}return keys;
}
void prepare(const std::shared_ptr<BatchedPipelines>& cache,const std::vector<int>& bits,const WarmKeys& keys){@autoreleasepool{for(int width:bits)for(auto& key:keys)cache->pipeline(width,key.first.c_str(),key.second);}}
}
void BatchedLinalg::prewarm(const BatchedPrewarm& r){auto keys=prewarm_keys(r);prepare(impl_->cache,r.bits,keys);}
std::future<void> BatchedLinalg::prewarm_async(BatchedPrewarm r){auto keys=prewarm_keys(r);auto cache=impl_->cache;return std::async(std::launch::async,[cache,bits=std::move(r.bits),keys=std::move(keys)]{prepare(cache,bits,keys);});}
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
    return power_moments(bits,complex,s,E,y,W,out,PowerStorage::full_table);
}
Timing BatchedLinalg::power_moments(int bits,bool complex,const PowerMoments& s,const void* E,const void* y,const void* W,void* out,PowerStorage storage){
    if(storage!=PowerStorage::full_table&&storage!=PowerStorage::compact)throw std::invalid_argument("power moments: invalid storage mode");
    auto ek=mul_size(s.count,s.steps);std::size_t sizes[4]={ek,ek,mul_size(ek,s.ncols),mul_size(mul_size(s.count,std::size_t(s.nmax)+1),s.ncols)};const void* pointers[4]={E,y,W,out};
    return host_call(*engine_,bits,complex,sizes,pointers,s.accumulate,[&](CommandBatch& b,O e,O yy,O w,O c){encode_power(b,bits,complex,s,e,yy,w,c,storage);});
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
void BatchedLinalg::encode_power(CommandBatch& b,int bits,bool complex,const PowerMoments& s,O E,O y,O W,O out,PowerStorage storage){
    if(storage!=PowerStorage::full_table&&storage!=PowerStorage::compact)throw std::invalid_argument("power moments: invalid storage mode");
    bits_ok(bits);auto rows=std::size_t(s.nmax)+1,ek=mul_size(s.count,s.steps),wk=mul_size(ek,s.ncols),pk=mul_size(ek,rows);
    grid_ok(ek);grid_ok(mul_size(mul_size(s.count,rows),s.ncols));size_ok(E,ek);size_ok(y,ek);size_ok(W,wk);distinct(out,{E,y,W});
    StridedGemm shape{s.count,rows,s.ncols,s.steps,rows*s.steps,std::size_t(s.steps)*s.ncols,rows*s.ncols,s.accumulate,s.fused};
    size_ok(out,mul_size(s.count,shape.stride_c));if(!s.count||!s.ncols)return;
    const bool panel=storage==PowerStorage::compact;
    auto element=std::size_t(bits/8+12)*(complex?2:1);
    if(!panel||!s.steps){auto powers=I::scratch(b,mul_size(pk,element));
        if(ek){auto e=retain(b,E,ek),yy=retain(b,y,ek);Params p{s.count,rows,s.ncols,s.steps,0,0,0,s.n0,0};
            impl_->pass(b,bits,complex?"batch_powers_complex":"batch_powers_real",p,{{0,e},{1,yy},{2,powers}},ek);}
        encode_gemm(b,bits,complex,shape,{powers,pk},W,out);return;}
    // The state is always unscaled. Segmenting the repeated multiplication adds no rounding.
    auto tile_rows=std::min<std::size_t>(8,rows),work_elements=mul_size(ek,tile_rows);
    auto state=I::scratch(b,mul_size(ek,element)),work=I::scratch(b,mul_size(work_elements,element));
    auto e=retain(b,E,ek),yy=retain(b,y,ek),w=retain(b,W,wk),c=retain(b,out,s.count*shape.stride_c,true);
    Params p{s.count,tile_rows,s.ncols,s.steps,rows,0,0,s.n0,0};
    impl_->pass(b,bits,complex?"batch_panel_seed_complex":"batch_panel_seed_real",p,{{0,yy},{2,state}},ek);
    for(std::size_t first=0;first<rows;first+=tile_rows){p.m=std::min(tile_rows,rows-first);p.row_offset=first;
        impl_->pass(b,bits,complex?"batch_panel_powers_complex":"batch_panel_powers_real",p,{{0,e},{1,yy},{2,work},{4,state}},ek);
        Params g{s.count,p.m,s.ncols,s.steps,p.m*s.steps,std::size_t(s.steps)*s.ncols,shape.stride_c,0,std::uint32_t(s.accumulate),first};
        if(complex)impl_->pass(b,bits,"batch_private_complex",g,{{0,work},{1,w},{2,c}},s.count*p.m*s.ncols,s.fused,false,true);
        else impl_->pass(b,bits,p.m==4&&s.ncols==4&&s.steps==4?"batch_gemm4_real":"batch_gemm_real",g,{{0,work},{1,w},{2,c}},s.count*p.m*s.ncols,s.fused,true);
    }
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
void BatchedLinalg::encode_sources(CommandBatch& b,int bits,const PolynomialSources& s,O cp,O cq,O Y,O Ep,O Eq,O P,O Q){
    bits_ok(bits);if(s.coefficient_sets!=1&&s.coefficient_sets!=s.groups)throw std::invalid_argument("polynomial sources: coefficient sets must be 1 or groups");
    auto coeff=mul_size(mul_size(s.coefficient_sets,4),s.terms),yn=mul_size(s.steps,s.groups),en=mul_size(yn,4);
    grid_ok(yn);size_ok(cp,coeff);size_ok(cq,coeff);size_ok(Y,yn);size_ok(Ep,en);size_ok(Eq,en);size_ok(P,en);size_ok(Q,en);distinct(P,{cp,cq,Y,Ep,Eq,Q});distinct(Q,{cp,cq,Y,Ep,Eq});if(!yn)return;
    auto cpbuf=retain(b,cp,coeff),cqbuf=retain(b,cq,coeff),ybuf=retain(b,Y,yn),ep=retain(b,Ep,en),eq=retain(b,Eq,en),p=retain(b,P,en,true),q=retain(b,Q,en,true);
    if(I::device(b)!=impl_->device)throw std::invalid_argument("polynomial sources: foreign batch device");
    struct PolyParams {std::uint64_t lanes,groups,sets;std::uint32_t steps,terms,share,flags;};PolyParams params{0,s.groups,s.coefficient_sets,s.steps,s.terms,0,0};
    auto state0=I::scratch(b,mul_size(en,std::size_t(bits/8+12)*2)),state1=I::scratch(b,mul_size(en,std::size_t(bits/8+12)*2));
    auto dispatch=[&](const char* name,bool fused,S current,S next,S coeff,S weight){
        auto pipeline=impl_->pipeline(bits,name,fused);I::keep(b,pipeline);auto enc=I::compute(b);[enc setComputePipelineState:pipeline];
        [enc setBuffer:current->buffer offset:0 atIndex:0];[enc setBuffer:coeff->buffer offset:0 atIndex:1];[enc setBuffer:next->buffer offset:0 atIndex:2];
        [enc setBuffer:ybuf->buffer offset:0 atIndex:5];[enc setBuffer:weight->buffer offset:0 atIndex:6];[enc setBytes:&params length:sizeof params atIndex:3];
        [enc dispatchThreads:MTLSizeMake(yn,8,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(pipeline.threadExecutionWidth,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
    };
    for(auto operands:{std::array<S,3>{cpbuf,ep,p},std::array<S,3>{cqbuf,eq,q}}){
        auto current=state0,next=state1;dispatch("batch_source_seed",false,current,current,operands[0],operands[1]);
        for(unsigned degree=s.terms?s.terms-1:0;degree-->0;){params.flags=degree;dispatch("batch_source_step",s.fused,current,next,operands[0],operands[1]);std::swap(current,next);}
        dispatch("batch_source_scale",false,current,operands[2],operands[0],operands[1]);
    }
}
void BatchedLinalg::encode_polynomial(CommandBatch& b,int bits,const PolynomialRecurrence& s,O start,O cp,O cq,O Y,O Ep,O Eq,O out,PolynomialEvaluation evaluation){
    if(evaluation!=PolynomialEvaluation::per_lane&&evaluation!=PolynomialEvaluation::shared_sources)throw std::invalid_argument("polynomial recurrence: invalid evaluation mode");
    bits_ok(bits);if(!s.lanes_per_weight)throw std::invalid_argument("polynomial recurrence: zero sharing");
    auto groups=s.lanes/s.lanes_per_weight+(s.lanes%s.lanes_per_weight!=0);
    if(s.coefficient_sets!=1&&s.coefficient_sets!=groups)throw std::invalid_argument("polynomial recurrence: coefficient sets must be 1 or weight groups");
    auto coeff=mul_size(mul_size(s.coefficient_sets,4),s.terms),yn=mul_size(s.steps,groups),en=mul_size(yn,4),vn=mul_size(s.lanes,4),on=mul_size(vn,s.all_steps?std::size_t(s.steps)+1:1);
    grid_ok(s.lanes);size_ok(start,vn);size_ok(cp,coeff);size_ok(cq,coeff);size_ok(Y,yn);size_ok(Ep,en);size_ok(Eq,en);size_ok(out,on);distinct(out,{start,cp,cq,Y,Ep,Eq});if(!s.lanes)return;
    auto st=retain(b,start,vn),p=retain(b,cp,coeff),q=retain(b,cq,coeff),y=retain(b,Y,yn),ep=retain(b,Ep,en),eq=retain(b,Eq,en),o=retain(b,out,on,true);
    if(I::device(b)!=impl_->device)throw std::invalid_argument("polynomial recurrence: foreign batch device");
    struct PolyParams {std::uint64_t lanes,groups,sets;std::uint32_t steps,terms,share,flags;};
    PolyParams params{s.lanes,groups,s.coefficient_sets,s.steps,s.terms,s.lanes_per_weight,std::uint32_t(s.all_steps)|(std::uint32_t(s.reverse)<<1)};
    if(evaluation==PolynomialEvaluation::shared_sources&&yn<=UINT32_MAX){
        auto element=std::size_t(bits/8+12)*2;
        auto weights_p=I::scratch(b,mul_size(en,element)),weights_q=I::scratch(b,mul_size(en,element));
        encode_sources(b,bits,{groups,s.coefficient_sets,s.steps,s.terms,s.fused},cp,cq,Y,Ep,Eq,{weights_p,en},{weights_q,en});
        // Rounded state stays in device memory. A dot pass completes before its four
        // independent component updates, including the final partial sharing group.
        auto dot=I::scratch(b,mul_size(s.lanes,element));I::copy(b,st,o,mul_size(vn,element));
        for(unsigned step=0;step<s.steps;++step){
            Params update{s.lanes,groups,s.lanes_per_weight,s.reverse?s.steps-1-step:step,
                          s.all_steps?mul_size(step,vn):0,0,s.all_steps?mul_size(std::size_t(step)+1,vn):0,0,0};
            impl_->pass(b,bits,"batch_source_dot",update,{{0,o},{1,weights_q},{2,dot}},s.lanes,s.fused,false,true);
            impl_->pass(b,bits,"batch_source_update",update,{{0,o},{1,weights_p},{2,o},{4,dot}},s.lanes,s.fused,false,true);
        }
        return;
    }
    auto state=impl_->pipeline(bits,"batch_polynomial_recurrence",s.fused);I::keep(b,state);auto enc=I::compute(b);[enc setComputePipelineState:state];
    for(auto& v:std::initializer_list<std::pair<unsigned,S>>{{0,st},{1,p},{2,o},{4,q},{5,y},{6,ep},{7,eq}})[enc setBuffer:v.second->buffer offset:0 atIndex:v.first];
    [enc setBytes:&params length:sizeof params atIndex:3];[enc dispatchThreads:MTLSizeMake(s.lanes,1,1) threadsPerThreadgroup:MTLSizeMake(state.threadExecutionWidth,1,1)];
}
std::future<std::vector<Timing>> wait_all_async(std::vector<Submission> submissions){return std::async(std::launch::async,[s=std::move(submissions)]()mutable{
    std::vector<Timing> result(s.size());std::exception_ptr error;for(std::size_t i=0;i<s.size();++i)try{result[i]=s[i].wait();}catch(...){if(!error)error=std::current_exception();}
    if(error)std::rethrow_exception(error);return result;});}
}
