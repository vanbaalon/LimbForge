// GPU codegen probe (plan A2). Compiles alternative exact-product formulations into Metal and compares
// raw unrounded products, rounded multiplication, and complex multiplication with the CPU at every width.
// The same probe_variants.hpp source is compiled on both sides, so any mismatch is a GPU-only effect.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "reference.hpp"
#include "probe_variants.hpp"
#include "shader_source.hpp"
#include <fstream>
#include <sstream>
#include <chrono>
using Clock=std::chrono::steady_clock;
using namespace limbforge;
const char* probe_kernels=R"METAL(
constant uint probe_variant [[function_constant(1)]];
kernel void probe_raw(device const Number<N>* a [[buffer(0)]],device const Number<N>* b [[buffer(1)]],device uint* raw [[buffer(2)]],
                      constant uint& count [[buffer(3)]],uint i [[thread_position_in_grid]]){
    if(i>=count)return;Number<N> x=a[i],y=b[i];word p[2*N+1];probe::product<N>(int(probe_variant),x,y,p);
    for(int k=0;k<2*N+1;++k)raw[ulong(i)*(2*N+1)+k]=p[k];
}
kernel void probe_mul(device const Number<N>* a [[buffer(0)]],device const Number<N>* b [[buffer(1)]],device Number<N>* out [[buffer(2)]],
                      constant uint& count [[buffer(3)]],uint i [[thread_position_in_grid]]){
    if(i>=count)return;Number<N> x=a[i],y=b[i];out[i]=probe::mul<N>(int(probe_variant),x,y);
}
kernel void probe_cmul(device const Complex<N>* a [[buffer(0)]],device const Complex<N>* b [[buffer(1)]],device Complex<N>* out [[buffer(2)]],
                       constant uint& count [[buffer(3)]],uint i [[thread_position_in_grid]]){
    if(i>=count)return;Complex<N> x=a[i],y=b[i];out[i]=probe::cmul<N>(int(probe_variant),x,y);
}
)METAL";
const char* variant_names[]={"schoolbook","comba64","comba32","comba_mulhi","comba64_unrolled","comba64_rolled","schoolbook_mulhi","comba64_once","comba_mulhi_once","schoolbook_rolled","schoolbook_outer_rolled","schoolbook_padded","schoolbook_unrolled"};
struct Options {std::size_t count=4096;int only_bits=0,only_variant=-1;bool size_opt=false,dump=false,time=false;unsigned language=(3u<<16)|1u,group=128;std::string variants;};
Options opt;id<MTLDevice> device;id<MTLCommandQueue> queue;
std::string read_file(const char* path){std::ifstream in(path);if(!in)throw std::runtime_error(std::string("cannot read ")+path);std::stringstream s;s<<in.rdbuf();return s.str();}
id<MTLBuffer> buffer(const void* data,std::size_t bytes){id<MTLBuffer> b=[device newBufferWithLength:std::max<std::size_t>(bytes,1) options:MTLResourceStorageModeShared];
    if(!b)throw std::runtime_error("buffer allocation failed");if(data)std::memcpy(b.contents,data,bytes);return b;}
double last_seconds=0;
unsigned run(id<MTLLibrary> library,NSString* kernel,int variant,id<MTLBuffer> a,id<MTLBuffer> b,id<MTLBuffer> out,std::uint32_t count,int repeats=1){
    MTLFunctionConstantValues* constants=[MTLFunctionConstantValues new];std::uint32_t v=variant;[constants setConstantValue:&v type:MTLDataTypeUInt atIndex:1];
    NSError* error=nil;id<MTLFunction> function=[library newFunctionWithName:kernel constantValues:constants error:&error];
    if(!function)throw std::runtime_error("specialization failed: "+std::string([[error localizedDescription] UTF8String]));
    id<MTLComputePipelineState> state=[device newComputePipelineStateWithFunction:function error:&error];
    if(!state)throw std::runtime_error("pipeline failed: "+std::string([[error localizedDescription] UTF8String]));
    id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:state];[encoder setBuffer:a offset:0 atIndex:0];[encoder setBuffer:b offset:0 atIndex:1];[encoder setBuffer:out offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    NSUInteger group=std::min<NSUInteger>(opt.group,state.maxTotalThreadsPerThreadgroup);group-=group%state.threadExecutionWidth;
    for(int r=0;r<repeats;++r)[encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(group,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];
    last_seconds=(command.GPUEndTime-command.GPUStartTime)/repeats;
    if(command.status==MTLCommandBufferStatusError)throw std::runtime_error("execution failed: "+std::string([[command.error localizedDescription] UTF8String]));
    return unsigned(state.maxTotalThreadsPerThreadgroup);
}
void report(int bits,int variant,const char* kernel,std::size_t bad,long first,int limb,unsigned max_threads){
    std::cout<<bits<<','<<variant_names[variant]<<','<<kernel<<','<<bad<<','<<opt.count<<','<<first<<','<<limb<<','<<max_threads<<std::endl;}
template<int N> void show(const char* label,const Number<N>& x){std::cerr<<"  "<<label<<": sign="<<x.sign<<" exp="<<x.exponent<<" status="<<x.status<<" limbs(top..)=";
    for(int i=N-1;i>=0&&i>=N-4;--i)std::cerr<<std::hex<<x.limb[i]<<' ';std::cerr<<"... "<<x.limb[0]<<std::dec<<'\n';}
template<int N> void dump(const char* what,const Number<N>& got,const Number<N>& want,const Number<N>& x,const Number<N>& y){
    std::cerr<<32*N<<" bits "<<what<<" first mismatch\n";show("gpu",got);show("cpu",want);
    word p[2*N+1];probe::school<N>(x,y,p);std::cerr<<"  exact product top words: "<<std::hex<<p[2*N]<<' '<<p[2*N-1]<<' '<<p[2*N-2]<<std::dec<<'\n';}
template<int N> void probe_width(){@autoreleasepool{
    constexpr int Bits=32*N,W=2*N+1;const std::size_t n=opt.count;std::mt19937_64 rng(20261007+Bits);
    std::vector<Number<N>> a(n),b(n),out(n);std::vector<Complex<N>> ca(n),cb(n),cres(n);
    for(std::size_t i=0;i<n;++i){a[i]=reference::random_number<Bits>(rng,5);b[i]=reference::random_number<Bits>(rng,5);
        ca[i]={reference::random_number<Bits>(rng,5),reference::random_number<Bits>(rng,5)};cb[i]={reference::random_number<Bits>(rng,5),reference::random_number<Bits>(rng,5)};}
    // Trivial and extreme products first: one times one, and all-ones limbs.
    a[0]=b[0]=from_decimal<Bits>("1");for(auto& l:a[1].limb)l=~0u;for(auto& l:b[1].limb)l=~0u;
    std::vector<word> expected_raw(n*W);std::vector<Number<N>> expected_mul(n);std::vector<Complex<N>> expected_cmul(n);
    for(std::size_t i=0;i<n;++i){word p[W];probe::school<N>(a[i],b[i],p);std::copy(p,p+W,&expected_raw[i*W]);expected_mul[i]=limbforge::mul(a[i],b[i]);expected_cmul[i]=limbforge::cmul(ca[i],cb[i]);}
    // Host check: every variant is the same exact integer on the CPU.
    for(int v=0;v<probe::variant_count;++v)for(std::size_t i=0;i<n;++i){word p[W];probe::product<N>(v,a[i],b[i],p);
        if(!std::equal(p,p+W,&expected_raw[i*W]))throw std::runtime_error(std::string("CPU variant differs: ")+variant_names[v]+" bits="+std::to_string(Bits));}
    NSString* source=[NSString stringWithFormat:@"#define MP_BITS %d\n%s\n%s\n%s",Bits,limbforge_shader_source,opt.variants.c_str(),probe_kernels];
    MTLCompileOptions* options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;options.languageVersion=MTLLanguageVersion(opt.language);
    if(opt.size_opt)options.optimizationLevel=MTLLibraryOptimizationLevelSize;
    NSError* error=nil;id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
    if(!library)throw std::runtime_error("Metal compilation: "+std::string([[error localizedDescription] UTF8String]));
    auto ba=buffer(a.data(),n*sizeof(a[0])),bb=buffer(b.data(),n*sizeof(b[0])),bca=buffer(ca.data(),n*sizeof(ca[0])),bcb=buffer(cb.data(),n*sizeof(cb[0]));
    auto braw=buffer(nullptr,n*W*sizeof(word)),bmul=buffer(nullptr,n*sizeof(out[0])),bcmul=buffer(nullptr,n*sizeof(cres[0]));
    for(int v=0;v<probe::variant_count;++v){if(opt.only_variant>=0&&v!=opt.only_variant)continue;
        std::memset(braw.contents,0xa5,braw.length);
        unsigned threads=run(library,@"probe_raw",v,ba,bb,braw,std::uint32_t(n));const word* raw=static_cast<const word*>(braw.contents);
        std::size_t bad=0;long first=-1;int limb=-1;
        for(std::size_t i=0;i<n;++i)for(int k=0;k<W;++k)if(raw[i*W+k]!=expected_raw[i*W+k]){if(first<0){first=long(i);limb=k;}++bad;break;}
        report(Bits,v,"raw",bad,first,limb,threads);
        threads=run(library,@"probe_mul",v,ba,bb,bmul,std::uint32_t(n));std::memcpy(out.data(),bmul.contents,n*sizeof(out[0]));
        bad=0;first=-1;for(std::size_t i=0;i<n;++i)if(!reference::equal<Bits>(out[i],expected_mul[i])){if(first<0)first=long(i);++bad;}
        if(first>=0&&opt.dump)dump("mul",out[first],expected_mul[first],a[first],b[first]);
        report(Bits,v,"mul",bad,first,-1,threads);
        threads=run(library,@"probe_cmul",v,bca,bcb,bcmul,std::uint32_t(n));std::memcpy(cres.data(),bcmul.contents,n*sizeof(cres[0]));
        bad=0;first=-1;for(std::size_t i=0;i<n;++i)if(!reference::equal_complex<Bits>(cres[i],expected_cmul[i])){if(first<0)first=long(i);++bad;}
        if(first>=0&&opt.dump){dump("cmul.re",cres[first].re,expected_cmul[first].re,ca[first].re,cb[first].re);dump("cmul.im",cres[first].im,expected_cmul[first].im,ca[first].im,cb[first].im);}
        report(Bits,v,"cmul",bad,first,-1,threads);
        if(opt.time)for(auto kernel:{@"probe_raw",@"probe_mul",@"probe_cmul"}){bool real=![kernel isEqualToString:@"probe_cmul"],raw=[kernel isEqualToString:@"probe_raw"];
            auto x=real?ba:bca,y=real?bb:bcb,z=raw?braw:real?bmul:bcmul;std::vector<double> t;
            for(auto start=Clock::now();std::chrono::duration<double>(Clock::now()-start).count()<0.15;)run(library,kernel,v,x,y,z,std::uint32_t(n),32);
            for(int s=0;s<5;++s){run(library,kernel,v,x,y,z,std::uint32_t(n),32);t.push_back(last_seconds);}
            std::sort(t.begin(),t.end());std::cout<<Bits<<','<<variant_names[v]<<','<<(raw?"raw":real?"mul":"cmul")<<"_us,"<<t[2]*1e6<<','<<n<<",,,"<<std::endl;}
    }
}}
template<int N> void all(){if(!opt.only_bits||opt.only_bits==32*N)probe_width<N>();if constexpr(N<32)all<N+1>();}
int main(int argc,char** argv){try{
    for(int i=1;i<argc;++i){std::string arg=argv[i];auto next=[&]{if(i+1>=argc)throw std::invalid_argument("missing value for "+arg);return std::string(argv[++i]);};
        if(arg=="--bits")opt.only_bits=std::stoi(next());else if(arg=="--variant"){auto name=next();for(int v=0;v<probe::variant_count;++v)if(name==variant_names[v])opt.only_variant=v;if(opt.only_variant<0)throw std::invalid_argument("unknown variant");}
        else if(arg=="--count")opt.count=std::stoull(next());else if(arg=="--group")opt.group=unsigned(std::stoul(next()));else if(arg=="--size-opt")opt.size_opt=true;else if(arg=="--dump")opt.dump=true;else if(arg=="--time")opt.time=true;
        else if(arg=="--language"){auto v=next();auto dot=v.find('.');opt.language=(unsigned(std::stoul(v.substr(0,dot)))<<16)|unsigned(std::stoul(v.substr(dot+1)));}
        else throw std::invalid_argument("usage: gpu_codegen_probe [--bits B] [--variant NAME] [--count N] [--group G] [--size-opt] [--dump] [--time] [--language 3.1]");}
    if(opt.count<2||opt.count>1000000)throw std::invalid_argument("count must be in 2..1000000");
    opt.variants=read_file(LIMBFORGE_PROBE_VARIANTS);
    device=MTLCreateSystemDefaultDevice();if(!device)throw std::runtime_error("no Metal GPU");queue=[device newCommandQueue];
    std::cerr<<[device.name UTF8String]<<"; GPU codegen probe; mismatches are GPU-only (CPU variants verified identical)\n";
    std::cout<<"bits,variant,kernel,mismatches,count,first_index,first_limb,max_threads\n";all<2>();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
