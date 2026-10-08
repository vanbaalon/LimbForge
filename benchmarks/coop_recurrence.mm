// Experiment L4 (docs/optimization-plan.md §8, E1): G SIMD lanes cooperate on one recurrence trajectory.
// Each number is top-aligned in M=G*L limb slots (L=ceil(N/G) limbs per lane; the M-N low slots are zero).
// Exact products and sums live in a 2M-slot workspace: lane k holds block k (lo) and block k+G (hi).
// Carries use an associative generate/propagate scan over ballot masks; rounding is pack()'s RN-even.
// Results must equal the core (CPU) and the existing `recurrence` kernel bit for bit; then per-step latency
// is compared with interleaved A/B samples. Since round 22 the arithmetic is the library's src/cooperative.metal.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "reference.hpp"
#include "shader_source.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <map>
#include <sstream>
using namespace limbforge;
using Clock=std::chrono::steady_clock;
// The cooperative arithmetic itself is src/cooperative.metal (namespace coop, part of limbforge_shader_source,
// which also compiles recurrence_coop4/8/16/32); this adds the real and complex unit kernels.
const char* coop_source=R"METAL(
#define COOP_KERNELS(G) \
kernel void coop_real##G(device const Number<N>* a [[buffer(0)]],device const Number<N>* b [[buffer(1)]],device Number<N>* out [[buffer(2)]],constant Params& p [[buffer(3)]],\
    uint gid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){coop::real_body<G>(a,b,out,p,gid,lane);}\
kernel void coop_complex##G(device const Complex<N>* a [[buffer(0)]],device const Complex<N>* b [[buffer(1)]],device Complex<N>* out [[buffer(2)]],constant Params& p [[buffer(3)]],\
    uint gid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){coop::complex_body<G>(a,b,out,p,gid,lane);}
COOP_KERNELS(4)
COOP_KERNELS(8)
COOP_KERNELS(16)
COOP_KERNELS(32)
)METAL";
struct Params {std::uint32_t count,operation,steps,weight_count,states_per_weight,b_stride=0,b_period=0,c_stride=0,c_period=0;}; // matches kernels.metal
struct Options {std::vector<int> bits={256,384,1024};std::vector<unsigned> lanes={32,128,512,2048,8192},groups={4,8,16,32};
    unsigned steps=64,tg=32;int samples=9;double warm=0.2;bool time=true,check=true,keep_going=false;std::size_t failed=0,count=4096;};
Options opt;id<MTLDevice> device;id<MTLCommandQueue> queue;
void check(bool good,const std::string& message){if(!good)throw std::runtime_error(message);}
double median(std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];}
id<MTLBuffer> buffer(const void* data,std::size_t bytes){id<MTLBuffer> b=[device newBufferWithLength:std::max<std::size_t>(bytes,1) options:MTLResourceStorageModeShared];
    check(b!=nil,"buffer allocation failed");if(data)std::memcpy(b.contents,data,bytes);return b;}
struct Library {
    id<MTLLibrary> library;std::map<std::string,id<MTLComputePipelineState>> states;
    id<MTLComputePipelineState> get(const std::string& name){auto& s=states[name];if(s)return s;NSError* error=nil;
        id<MTLFunction> f=[library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];check(f!=nil,"missing kernel "+name);
        s=[device newComputePipelineStateWithFunction:f error:&error];check(s!=nil,"pipeline "+name+": "+(error?[[error localizedDescription] UTF8String]:""));
        check(s.threadExecutionWidth==32,"cooperative kernels assume 32-wide SIMD groups");return s;}
};
// One dispatch of `threads` threads; returns device seconds.
double run(id<MTLComputePipelineState> state,id<MTLBuffer> a,id<MTLBuffer> b,id<MTLBuffer> out,const Params& p,std::size_t threads,unsigned group){
    @autoreleasepool{id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:state];[encoder setBuffer:a offset:0 atIndex:0];[encoder setBuffer:b offset:0 atIndex:1];[encoder setBuffer:out offset:0 atIndex:2];
    [encoder setBytes:&p length:sizeof(p) atIndex:3];NSUInteger g=std::min<NSUInteger>(group,state.maxTotalThreadsPerThreadgroup);g-=g%32;
    [encoder dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(g,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];
    if(command.status==MTLCommandBufferStatusError)throw std::runtime_error(std::string("execution failed: ")+[[command.error localizedDescription] UTF8String]);
    return command.GPUEndTime-command.GPUStartTime;}
}
template<int N> struct Width {
    static constexpr int Bits=32*N;using F=Number<N>;using C=Complex<N>;Library lib;std::mt19937_64 rng{20261007u+Bits};
    std::size_t real_bad=0,complex_bad=0,recurrence_bad=0,checks=0;
    Width(){NSString* source=[NSString stringWithFormat:@"#define MP_BITS %d\n%s\n%s",Bits,limbforge_shader_source,coop_source];
        MTLCompileOptions* options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;options.languageVersion=MTLLanguageVersion3_1;
        NSError* error=nil;lib.library=[device newLibraryWithSource:source options:options error:&error];
        check(lib.library!=nil,std::string("Metal compilation: ")+(error?[[error localizedDescription] UTF8String]:""));}
    F rnd(int span){return reference::random_number<Bits>(rng,span);}
    static F ones(int e,int s){F x=zero<N>();for(auto& l:x.limb)l=~0u;x.exponent=e;x.sign=s;return x;}
    static F power(int e,int s){F x=zero<N>();x.limb[N-1]=0x80000000u;x.exponent=e;x.sign=s;return x;}
    // Random pairs plus cancellation, alignment, tie, carry, overflow, zero and status edge cases.
    void real_cases(std::vector<F>& a,std::vector<F>& b){
        for(std::size_t i=0;i<opt.count;++i){a.push_back(rnd(5));b.push_back(rnd(5));}
        auto pair=[&](F x,F y){a.push_back(x);b.push_back(y);};
        for(int r=0;r<16;++r){F x=rnd(5),y=x;pair(x,y);y.sign=-x.sign;pair(x,y);
            for(int keep=1;keep<N;++keep){y=x;y.sign=-x.sign;for(int j=0;j<N-keep;++j)y.limb[j]=std::uint32_t(rng());pair(x,y);}
            y=x;y.sign=-x.sign;y.limb[0]^=1;pair(x,y);y=x;y.sign=-x.sign;y.limb[0]+=1;pair(x,y);
            pair(power(x.exponent,1),ones(x.exponent-1,-1));pair(power(x.exponent,-1),ones(x.exponent-1,1));pair(power(x.exponent,1),rnd(0));
            for(int gap:{1,2,3,4,31,32,33,34,63,64,65,32*N-33,32*N-32,32*N-31,32*N-3,32*N-2,32*N-1,32*N,32*N+1,32*N+2,32*N+3,32*N+4,32*N+40})
                for(int s:{1,-1}){y=rnd(0);y.exponent=x.exponent-gap;y.sign=s*x.sign;pair(x,y);pair(y,x);
                    pair(x,power(x.exponent-gap,s));F z=power(x.exponent-gap,s);z.limb[0]=1;pair(x,z);pair(x,ones(x.exponent-gap,s));
                    pair(ones(x.exponent,1),power(x.exponent-gap,s));pair(ones(x.exponent,-1),ones(x.exponent-gap,s));}
            F o=ones(3,1);pair(o,o);pair(o,power(-29,1));pair(power(0,1),power(0,1));pair(o,rnd(5));
        }
        // Structured limbs (0, 1, all ones, ...) give long carry chains in block products and sums.
        const std::uint32_t pattern[]={0u,1u,~0u,~0u-1,0x80000000u,0x7fffffffu,~0u,~0u};
        for(int r=0;r<2048;++r){F x=rnd(5),y=rnd(5);for(int j=0;j<N;++j){x.limb[j]=pattern[rng()%8];y.limb[j]=pattern[rng()%8];}
            x.limb[N-1]|=0x80000000u;y.limb[N-1]|=0x80000000u;if(r%4==0)y.exponent=x.exponent-int(rng()%(32*N+4));pair(x,y);}
        pair(zero<N>(),rnd(5));pair(rnd(5),zero<N>());pair(zero<N>(),zero<N>());
        F big=rnd(0);big.exponent=1000000000;pair(big,big);F small=rnd(0);small.exponent=-1000000000;pair(small,small);
        F half=rnd(0);half.exponent=600000000;pair(half,half);half.exponent=-600000000;pair(half,half);pair(ones(1000000000,1),ones(1000000000,1));
        F bad=rnd(5);bad.status=invalid;pair(bad,rnd(5));pair(rnd(5),bad);
    }
    template<class T> std::size_t compare(const std::vector<T>& got,const std::vector<T>& want,const std::string& what,int G){
        std::size_t bad=0;long first=-1;for(std::size_t i=0;i<got.size();++i){bool same;
            if constexpr(std::is_same_v<T,F>)same=reference::equal<Bits>(got[i],want[i]);else same=reference::equal_complex<Bits>(got[i],want[i]);
            if(!same){if(first<0)first=long(i);++bad;}}
        checks+=got.size();if(bad)std::cerr<<Bits<<" bits G="<<G<<' '<<what<<": "<<bad<<'/'<<got.size()<<" mismatches, first case "<<first<<'\n';
        return bad;
    }
    void validate(int G){
        std::vector<F> a,b;real_cases(a,b);std::size_t n=a.size();std::vector<F> out(n),want(n);
        auto ba=buffer(a.data(),n*sizeof(F)),bb=buffer(b.data(),n*sizeof(F)),bo=buffer(nullptr,n*sizeof(F));
        const char* names[]={"add","sub","mul"};
        for(std::uint32_t op=0;op<3;++op){for(std::size_t i=0;i<n;++i)want[i]=op==0?add(a[i],b[i]):op==1?sub(a[i],b[i]):mul(a[i],b[i]);
            std::memset(bo.contents,0xa5,bo.length);run(lib.get("coop_real"+std::to_string(G)),ba,bb,bo,Params{std::uint32_t(n),op,0,0,1},n*G,opt.tg);
            std::memcpy(out.data(),bo.contents,n*sizeof(F));real_bad+=compare(out,want,names[op],G);}
        std::size_t m=opt.count;std::vector<C> x(m),y(m),z(m),zw(m);
        for(std::size_t i=0;i<m;++i){x[i]={rnd(5),rnd(5)};y[i]={rnd(5),rnd(5)};}
        for(std::size_t i=0;i<16;++i){y[i]={negate(x[i].re),x[i].im};}
        for(std::size_t i=16;i<m/2;++i)for(F* v:{&x[i].re,&x[i].im,&y[i].re,&y[i].im}){for(int j=0;j<N;++j)v->limb[j]=(rng()&1)?~0u:std::uint32_t(rng()%3);v->limb[N-1]|=0x80000000u;}
        auto bx=buffer(x.data(),m*sizeof(C)),by=buffer(y.data(),m*sizeof(C)),bz=buffer(nullptr,m*sizeof(C));
        for(std::uint32_t op:{4u,5u}){for(std::size_t i=0;i<m;++i)zw[i]=op==4?cadd(x[i],y[i]):cmul(x[i],y[i]);
            std::memset(bz.contents,0xa5,bz.length);run(lib.get("coop_complex"+std::to_string(G)),bx,by,bz,Params{std::uint32_t(m),op,0,0,1},m*G,opt.tg);
            std::memcpy(z.data(),bz.contents,m*sizeof(C));complex_bad+=compare(z,zw,op==4?"cadd":"cmul",G);}
        // Recurrences: tests/test_arithmetic.cpp data (per-lane weights), replicated shared weights, and a longer batch.
        for(auto [batch,steps,spw]:{std::tuple<unsigned,unsigned,unsigned>{17,31,1},{68,31,4},{256,64,1},{96,200,96}}){
            std::vector<C> seed(4*batch),weight(std::size_t(4)*(batch/spw)*steps),cpu(batch),gpu(batch),co(batch);
            for(auto& v:seed)v={rnd(5),rnd(5)};
            for(auto& v:weight){v={rnd(3),rnd(3)};v.re.exponent-=5;v.im.exponent-=5;}
            if(spw==4)for(unsigned j=0;j<4;++j)for(unsigned i=0;i<batch;++i)seed[j*batch+i]=seed[j*batch+i/4*4];
            unsigned wc=batch/spw,cpu_lanes=batch<=68?batch:12;
            for(unsigned i=0;i<cpu_lanes;++i){C ring[4];for(int j=0;j<4;++j)ring[j]=seed[j*batch+i];int head=0;
                for(unsigned t=0;t<steps;++t){C sum={zero<N>(),zero<N>()};for(int j=0;j<4;++j)sum=cadd(sum,cmul(weight[(t*4+j)*wc+i/spw],ring[(head+j)%4]));ring[head]=sum;head=(head+1)%4;}
                cpu[i]=ring[(head+3)%4];}
            auto bs=buffer(seed.data(),seed.size()*sizeof(C)),bw=buffer(weight.data(),weight.size()*sizeof(C)),bo1=buffer(nullptr,batch*sizeof(C)),bo2=buffer(nullptr,batch*sizeof(C));
            Params p{batch,0,steps,wc,spw};run(lib.get("recurrence"),bs,bw,bo1,p,batch,32);std::memcpy(gpu.data(),bo1.contents,batch*sizeof(C));
            std::memset(bo2.contents,0xa5,bo2.length);run(lib.get("recurrence_coop"+std::to_string(G)),bs,bw,bo2,p,std::size_t(batch)*G,opt.tg);std::memcpy(co.data(),bo2.contents,batch*sizeof(C));
            std::vector<C> gpu_head(gpu.begin(),gpu.begin()+cpu_lanes),co_head(co.begin(),co.begin()+cpu_lanes),cpu_head(cpu.begin(),cpu.begin()+cpu_lanes);
            std::string tag="recurrence "+std::to_string(batch)+"x"+std::to_string(steps)+" spw="+std::to_string(spw);
            check(compare(gpu_head,cpu_head,tag+" existing-vs-cpu",1)==0,"existing recurrence kernel disagrees with the core");
            recurrence_bad+=compare(co_head,cpu_head,tag+" coop-vs-cpu",G)+compare(co,gpu,tag+" coop-vs-existing",G);
        }
        std::cout<<"# "<<Bits<<" bits G="<<G<<" L="<<(N+G-1)/G<<": real "<<real_bad<<", complex "<<complex_bad<<", recurrence "<<recurrence_bad<<" mismatches (cumulative)"<<std::endl;
    }
    // Interleaved A/B: each sample round runs every variant once, in rotated order; median device time per step.
    void time(){
        std::vector<std::pair<std::string,unsigned>> variants={{"recurrence",1}};for(auto G:opt.groups)variants.push_back({"recurrence_coop"+std::to_string(G),G});
        for(unsigned n:opt.lanes){
            std::vector<C> seed(4*n),weight(4*opt.steps);for(auto& v:seed)v={rnd(5),rnd(5)};
            for(auto& v:weight){v={rnd(3),rnd(3)};v.re.exponent-=5;v.im.exponent-=5;}
            auto bs=buffer(seed.data(),seed.size()*sizeof(C)),bw=buffer(weight.data(),weight.size()*sizeof(C));
            std::vector<id<MTLBuffer>> outs;for(std::size_t v=0;v<variants.size();++v)outs.push_back(buffer(nullptr,n*sizeof(C)));
            Params p{n,0,opt.steps,1,n};std::vector<std::vector<double>> t(variants.size());
            // Every cooperative dispatch, warm-up included, is compared with the existing kernel's output.
            std::vector<std::size_t> bad(variants.size()),dispatches(variants.size());
            auto once=[&](std::size_t v){double t=run(lib.get(variants[v].first),bs,bw,outs[v],p,std::size_t(n)*variants[v].second,variants[v].second==1?32:opt.tg);
                if(v){++dispatches[v];bad[v]+=std::memcmp(outs[v].contents,outs[0].contents,n*sizeof(C))!=0;}return t;};
            once(0);
            for(auto start=Clock::now();std::chrono::duration<double>(Clock::now()-start).count()<opt.warm;)for(std::size_t v=0;v<variants.size();++v)once(v);
            for(int s=0;s<opt.samples;++s)for(std::size_t k=0;k<variants.size();++k){std::size_t v=(k+s)%variants.size();t[v].push_back(once(v)/opt.steps);}
            for(std::size_t v=1;v<variants.size();++v){std::cerr<<"# "<<Bits<<" bits "<<n<<" lanes G="<<variants[v].second<<": "<<bad[v]<<'/'<<dispatches[v]<<" dispatches differ\n";
                check(!bad[v]||opt.keep_going,"timed coop output differs from existing kernel");}
            double base=median(t[0]);
            for(std::size_t v=0;v<variants.size();++v){double m=median(t[v]);
                std::cout<<Bits<<','<<n<<','<<(v?"coop":"thread")<<','<<variants[v].second<<','<<(v?(N+variants[v].second-1)/variants[v].second:N)<<','
                    <<std::size_t(n)*variants[v].second<<','<<opt.steps<<','<<opt.samples<<','<<m*1e6<<','<<*std::min_element(t[v].begin(),t[v].end())*1e6<<','<<base/m<<std::endl;}
        }
    }
};
template<int N> void width(bool full){@autoreleasepool{
    bool listed=std::find(opt.bits.begin(),opt.bits.end(),32*N)!=opt.bits.end();
    if(listed||full){Width<N> w;if(opt.check){for(auto G:opt.groups)w.validate(int(G));
            bool good=!w.real_bad&&!w.complex_bad&&!w.recurrence_bad;opt.failed+=!good;check(good||opt.keep_going,"cooperative results differ at "+std::to_string(32*N)+" bits");}
        if(opt.time&&listed)w.time();}
    if constexpr(N<32)width<N+1>(full);
}}
std::vector<unsigned> list(const std::string& s){std::vector<unsigned> r;std::stringstream in(s);std::string x;while(std::getline(in,x,','))r.push_back(unsigned(std::stoul(x)));return r;}
int main(int argc,char** argv){try{
    bool full=false;
    for(int i=1;i<argc;++i){std::string arg=argv[i];auto next=[&]{check(i+1<argc,"missing value for "+arg);return std::string(argv[++i]);};
        if(arg=="--bits"){opt.bits.clear();for(auto b:list(next()))opt.bits.push_back(int(b));}else if(arg=="--lanes")opt.lanes=list(next());else if(arg=="--groups")opt.groups=list(next());
        else if(arg=="--steps")opt.steps=unsigned(std::stoul(next()));else if(arg=="--samples")opt.samples=std::stoi(next());else if(arg=="--warm")opt.warm=std::stod(next());
        else if(arg=="--threadgroup")opt.tg=unsigned(std::stoul(next()));else if(arg=="--count")opt.count=std::stoull(next());
        else if(arg=="--all-bits")full=true;else if(arg=="--no-time")opt.time=false;else if(arg=="--no-check")opt.check=false;else if(arg=="--keep-going")opt.keep_going=true;
        else throw std::invalid_argument("usage: coop_recurrence [--bits 256,384,1024] [--all-bits] [--groups 4,8,16,32] [--lanes 32,128,...] [--steps S] [--samples K] [--warm SECONDS] [--threadgroup T] [--count N] [--no-time] [--no-check]");}
    for(auto G:opt.groups)check(G==4||G==8||G==16||G==32,"groups must be 4, 8, 16 or 32");
    check(opt.tg>=32&&opt.tg%32==0&&opt.samples>0&&opt.steps>0,"invalid options");
    device=MTLCreateSystemDefaultDevice();check(device!=nil,"no Metal GPU");queue=[device newCommandQueue];
    std::cerr<<[device.name UTF8String]<<"; cooperative recurrence experiment (L4); every cooperative result compared bit for bit\n";
    std::cout<<std::setprecision(6)<<"bits,lanes,kernel,G,limbs_per_lane,threads,steps,samples,median_us_per_step,min_us_per_step,speedup_vs_thread\n";
    width<2>(full);if(opt.failed)std::cerr<<opt.failed<<" precision(s) with cooperative mismatches\n";return opt.failed?1:0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
