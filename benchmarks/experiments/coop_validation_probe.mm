// Round 41-L4b diagnostic (not a CTest): bisects the G = 4/8 cooperative-recurrence mismatches that appear only
// under MTL_SHADER_VALIDATION=1 (docs/experiments.md, 17-L4 and 41-L4b). Kernels and switches are in
// coop_validation_probe.metal (a switchable copy of src/cooperative.metal). Every dispatch is compared bit for bit
// with the core (CPU). Build target: coop_validation_probe. Example:
//   MTL_SHADER_VALIDATION=1 ./build/coop_validation_probe --bits 800 --groups 4 --ops 0 --count 96 --steps 200 --repeats 10
//   ... --define PROBE_TG=1        (no SIMD intrinsics)     --pipeline-validation on|off (per-pipeline switch)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "reference.hpp"
#include "shader_source.hpp"
#include <fstream>
#include <map>
#include <set>
#include <sstream>
using namespace limbforge;
struct Params {std::uint32_t count,operation,steps,weight_count,states_per_weight,b_stride=0,b_period=0,c_stride=0,c_period=0;};
struct Options {std::vector<int> bits={800};std::vector<unsigned> groups={4},ops={0};unsigned count=96,steps=200,repeats=10,tg=32,spw=0;
    std::vector<std::pair<std::string,std::string>> defines;int pipeline_validation=0;bool detail=false;std::uint64_t seed=41;};
Options opt;id<MTLDevice> device;id<MTLCommandQueue> queue;std::string probe_source;
void check(bool good,const std::string& message){if(!good)throw std::runtime_error(message);}
id<MTLBuffer> buffer(const void* data,std::size_t bytes){id<MTLBuffer> b=[device newBufferWithLength:std::max<std::size_t>(bytes,1) options:MTLResourceStorageModeShared];
    check(b!=nil,"buffer allocation failed");if(data)std::memcpy(b.contents,data,bytes);return b;}
const char* op_name(unsigned op){static const char* n[]={"recurrence","cmul-chain","cadd-chain","mul-chain","add-chain"};return n[op];}
template<int N> struct Width {
    static constexpr int Bits=32*N;using F=Number<N>;using C=Complex<N>;id<MTLLibrary> library;
    Width(){
        std::string head="#define MP_BITS "+std::to_string(Bits)+"\n";
        NSString* source=[NSString stringWithFormat:@"%s%s\n#line 1\n%s",head.c_str(),limbforge_shader_source,probe_source.c_str()];
        MTLCompileOptions* options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;options.languageVersion=MTLLanguageVersion3_1;
        NSMutableDictionary* macros=[NSMutableDictionary dictionary];
        for(auto& [k,v]:opt.defines)macros[[NSString stringWithUTF8String:k.c_str()]]=[NSString stringWithUTF8String:v.c_str()];
        options.preprocessorMacros=macros;NSError* error=nil;library=[device newLibraryWithSource:source options:options error:&error];
        check(library!=nil,std::string("Metal compilation: ")+(error?[[error localizedDescription] UTF8String]:""));}
    id<MTLComputePipelineState> pipeline(const std::string& name){
        id<MTLFunction> f=[library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];check(f!=nil,"missing kernel "+name);
        MTLComputePipelineDescriptor* d=[MTLComputePipelineDescriptor new];d.computeFunction=f;
        if(@available(macOS 15.0,*))d.shaderValidation=opt.pipeline_validation==1?MTLShaderValidationEnabled:opt.pipeline_validation==2?MTLShaderValidationDisabled:MTLShaderValidationDefault;
        NSError* error=nil;id<MTLComputePipelineState> s=[device newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionNone reflection:nil error:&error];
        check(s!=nil,"pipeline "+name+": "+(error?[[error localizedDescription] UTF8String]:""));check(s.threadExecutionWidth==32,"32-wide SIMD groups assumed");return s;}
    // The CPU reference of each kernel (same composed roundings).
    void reference(unsigned op,const std::vector<C>& seed,const std::vector<C>& weight,unsigned wc,unsigned spw,std::vector<C>& out,std::vector<C>* trace){
        unsigned n=opt.count;
        for(unsigned i=0;i<n;++i){C q[4];for(int j=0;j<4;++j)q[j]=seed[j*n+i];unsigned wi=i/spw;
            for(unsigned t=0;t<opt.steps;++t){
                if(op==0){C sum={zero<N>(),zero<N>()};for(unsigned j=0;j<4;++j)sum=cadd(sum,cmul(weight[(t*4+j)*wc+wi],q[j]));q[0]=q[1];q[1]=q[2];q[2]=q[3];q[3]=sum;}
                else{const C& w=weight[t*wc+wi];
                    if(op==1)q[3]=cmul(w,q[3]);else if(op==2)q[3]=cadd(q[3],w);
                    else if(op==3){q[3].re=mul(w.re,q[3].re);q[3].im=mul(w.im,q[3].im);}else{q[3].re=add(q[3].re,w.re);q[3].im=add(q[3].im,w.im);}}
                if(trace)(*trace)[std::size_t(t)*n+i]=q[3];}
            out[i]=q[3];}
    }
    static std::string diff(const C& got,const C& want){
        std::ostringstream s;auto part=[&](const char* name,const F& g,const F& w){if(reference::equal<Bits>(g,w))return;
            s<<' '<<name<<": exp "<<g.exponent<<(g.exponent==w.exponent?"":"!="+std::to_string(w.exponent))<<" sign "<<g.sign<<(g.sign==w.sign?"":"!="+std::to_string(w.sign))
             <<" status "<<g.status<<" limbs differing (index from msb):";
            int lo=-1;for(int j=N-1;j>=0;--j)if(g.limb[j]!=w.limb[j]){s<<' '<<N-1-j;lo=j;}
            if(lo>=0)s<<std::hex<<" [lowest differing limb got 0x"<<g.limb[lo]<<" want 0x"<<w.limb[lo]<<']'<<std::dec;};
        part("re",got.re,want.re);part("im",got.im,want.im);return s.str();}
    void run(unsigned G,unsigned op){
        std::mt19937_64 rng(opt.seed*1000003u+Bits*31u+G*7u+op);unsigned n=opt.count,spw=opt.spw?opt.spw:n,wc=n/spw;
        check(n%spw==0,"count must be divisible by --spw");
        std::vector<C> seed(4*std::size_t(n)),weight(std::size_t(4)*wc*opt.steps),want(n),got(n);
        for(auto& v:seed)v={reference::random_number<Bits>(rng,5),reference::random_number<Bits>(rng,5)};
        for(auto& v:weight){v={reference::random_number<Bits>(rng,3),reference::random_number<Bits>(rng,3)};v.re.exponent-=5;v.im.exponent-=5;}
        bool tracing=false;for(auto& d:opt.defines)tracing|=d.first=="PROBE_TRACE"&&d.second!="0";
        std::vector<C> trace_want,trace_got;if(tracing){trace_want.resize(std::size_t(n)*opt.steps);trace_got.resize(trace_want.size());}
        reference(op,seed,weight,wc,spw,want,tracing?&trace_want:nullptr);
        // G = 1 is the one-thread kernel (probe_thread: kernels.metal `recurrence` plus the optional ballast; op 0 only).
        auto state=pipeline(G==1?std::string("probe_thread"):"probe_g"+std::to_string(G)+"_op"+std::to_string(op));check(G>1||op==0,"G=1 runs only op 0");
        auto bs=buffer(seed.data(),seed.size()*sizeof(C)),bw=buffer(weight.data(),weight.size()*sizeof(C)),bo=buffer(nullptr,n*sizeof(C)),
             bt=buffer(nullptr,tracing?trace_want.size()*sizeof(C):sizeof(C)),bd=buffer(nullptr,4096);std::map<unsigned,std::size_t> sites;
        Params p{n,0,opt.steps,wc,spw};std::size_t bad_dispatches=0,bad_values=0;std::map<unsigned,unsigned> groups;std::string first;double seconds=0;std::size_t inactive=0,inactive_dispatches=0,inactive_and_bad=0;
        for(unsigned r=0;r<opt.repeats;++r){
            std::memset(bo.contents,0xa5,bo.length);std::memset(bd.contents,0,bd.length);
            @autoreleasepool{id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
                [encoder setComputePipelineState:state];[encoder setBuffer:bs offset:0 atIndex:0];[encoder setBuffer:bw offset:0 atIndex:1];
                [encoder setBuffer:bo offset:0 atIndex:2];[encoder setBytes:&p length:sizeof(p) atIndex:3];[encoder setBuffer:bt offset:0 atIndex:4];[encoder setBuffer:bd offset:0 atIndex:5];
                [encoder dispatchThreads:MTLSizeMake(std::size_t(n)*G,1,1) threadsPerThreadgroup:MTLSizeMake(opt.tg,1,1)];[encoder endEncoding];
                [command commit];[command waitUntilCompleted];
                if(command.status==MTLCommandBufferStatusError)throw std::runtime_error(std::string("execution failed: ")+[[command.error description] UTF8String]);
                seconds+=command.GPUEndTime-command.GPUStartTime;}
            std::memcpy(got.data(),bo.contents,n*sizeof(C));std::size_t bad=0;
            for(unsigned i=0;i<n;++i)if(!reference::equal_complex<Bits>(got[i],want[i])){++bad;++groups[i*G/32];
                if(first.empty()){std::ostringstream s;s<<"repeat "<<r<<" trajectory "<<i<<" (SIMD group "<<i*G/32<<", lane group "<<i%(32/G)<<")"<<diff(got[i],want[i]);
                    if(tracing){std::memcpy(trace_got.data(),bt.contents,trace_got.size()*sizeof(C));
                        for(unsigned t=0;t<opt.steps;++t)if(!reference::equal_complex<Bits>(trace_got[std::size_t(t)*n+i],trace_want[std::size_t(t)*n+i])){
                            s<<"; first wrong step "<<t<<":"<<diff(trace_got[std::size_t(t)*n+i],trace_want[std::size_t(t)*n+i]);
                            if(t){const C& prev=trace_got[std::size_t(t-1)*n+i];s<<" (previous step exact: "<<(reference::equal_complex<Bits>(prev,trace_want[std::size_t(t-1)*n+i])?"yes":"no")<<")";}
                            break;}}
                    first=s.str();}}
            std::uint32_t d=0;for(unsigned l=0;l<1024;++l){auto v=static_cast<std::uint32_t*>(bd.contents)[l];d+=v;if(v)sites[l]+=v;}inactive+=d;inactive_dispatches+=d!=0;inactive_and_bad+=d!=0&&bad!=0;
            bad_values+=bad;bad_dispatches+=bad!=0;}
        std::cout<<Bits<<','<<G<<','<<(N+G-1)/G<<','<<op_name(op)<<','<<n<<','<<opt.steps<<','<<opt.repeats<<','<<bad_dispatches<<','<<bad_values<<','
                 <<groups.size()<<','<<seconds/opt.repeats*1e3<<','<<inactive_dispatches<<','<<inactive_and_bad<<','<<inactive<<std::endl;
        if(opt.detail&&!sites.empty()){std::cout<<"#  partial SIMD ops by probe source line:";for(auto [l,k]:sites)std::cout<<' '<<l<<':'<<k;std::cout<<std::endl;}
        if(opt.detail&&!first.empty()){std::cout<<"#  first: "<<first<<"\n#  SIMD groups hit (group:count):";for(auto [g,k]:groups)std::cout<<' '<<g<<':'<<k;std::cout<<std::endl;}
    }
};
template<int N> void width(){@autoreleasepool{
    if(std::find(opt.bits.begin(),opt.bits.end(),32*N)!=opt.bits.end()){Width<N> w;for(auto G:opt.groups)for(auto op:opt.ops)w.run(G,op);}
    if constexpr(N<32)width<N+1>();}}
std::vector<unsigned> list(const std::string& s){std::vector<unsigned> r;std::stringstream in(s);std::string x;while(std::getline(in,x,','))r.push_back(unsigned(std::stoul(x)));return r;}
int main(int argc,char** argv){try{
    std::string path=LIMBFORGE_PROBE_SOURCE;
    for(int i=1;i<argc;++i){std::string arg=argv[i];auto next=[&]{check(i+1<argc,"missing value for "+arg);return std::string(argv[++i]);};
        if(arg=="--bits"){opt.bits.clear();for(auto b:list(next()))opt.bits.push_back(int(b));}else if(arg=="--groups")opt.groups=list(next());else if(arg=="--ops")opt.ops=list(next());
        else if(arg=="--count")opt.count=unsigned(std::stoul(next()));else if(arg=="--steps")opt.steps=unsigned(std::stoul(next()));
        else if(arg=="--repeats")opt.repeats=unsigned(std::stoul(next()));else if(arg=="--threadgroup")opt.tg=unsigned(std::stoul(next()));
        else if(arg=="--spw")opt.spw=unsigned(std::stoul(next()));else if(arg=="--seed")opt.seed=std::stoull(next());else if(arg=="--detail")opt.detail=true;
        else if(arg=="--source")path=next();
        else if(arg=="--define"){std::string d=next();auto e=d.find('=');opt.defines.push_back({d.substr(0,e),e==std::string::npos?"1":d.substr(e+1)});}
        else if(arg=="--pipeline-validation"){std::string v=next();opt.pipeline_validation=v=="on"?1:v=="off"?2:0;}
        else throw std::invalid_argument("usage: coop_validation_probe [--bits 800] [--groups 4] [--ops 0..4] [--count 96] [--steps 200] [--repeats 10] [--spw S] [--threadgroup 32] [--seed K] [--define PROBE_X=1]... [--pipeline-validation default|on|off] [--detail] [--source file.metal]");}
    for(auto G:opt.groups)check(G==1||G==4||G==8||G==16||G==32,"groups must be 1 (one-thread kernel), 4, 8, 16 or 32");for(auto op:opt.ops)check(op<5,"ops are 0..4");
    for(auto& d:opt.defines)if(d.first=="PROBE_TG"&&d.second!="0")check(opt.tg==32,"PROBE_TG needs --threadgroup 32");
    std::ifstream in(path);check(bool(in),"cannot read "+path);std::stringstream s;s<<in.rdbuf();probe_source=s.str();
    device=MTLCreateSystemDefaultDevice();check(device!=nil,"no Metal GPU");queue=[device newCommandQueue];
    const char* v=getenv("MTL_SHADER_VALIDATION");std::cerr<<[device.name UTF8String]<<"; MTL_SHADER_VALIDATION="<<(v?v:"unset")<<"; defines:";
    for(auto& d:opt.defines)std::cerr<<' '<<d.first<<'='<<d.second;std::cerr<<"; pipeline validation "<<(opt.pipeline_validation==1?"on":opt.pipeline_validation==2?"off":"default")<<'\n';
    std::cout<<"bits,G,limbs_per_lane,kernel,trajectories,steps,repeats,bad_dispatches,bad_trajectories,simd_groups_hit,ms_per_dispatch,dispatches_with_partial_simd_ops,of_which_bad,partial_simd_ops"<<std::endl;
    width<2>();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
