// Round 20 (plan D6 via L1c): dense SYRK C = A^T A (A: rows = 2*cols, QSC normal-equation shape) and GEMM C = A^T B
// with one rounding per output (Linalg, residue GEMM over exponent bands), against (a) a straightforward LimbForge limb
// kernel, one thread per output with sequential composed mul+add (or fma), and (b) a consumer-style MPFR loop
// (sequential mpfr_mul + mpfr_add per output), serial and multithreaded. Optional CPU MPFR Cholesky of the result.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "limbforge/linalg.hpp"
#include "benchmark_support.hpp"
#include "shader_source.hpp"
#include <atomic>
#include <map>
const char* baseline_source=R"METAL(
// One thread per output of A^T B (A: K x M, B: K x Nn), sequential over k: composed (mul then add) or fused (fma).
// lower != 0 (SYRK, B = A): only outputs with j <= i.
kernel void limb_composed(device const Number<N>* A [[buffer(0)]],device const Number<N>* B [[buffer(1)]],device Number<N>* C [[buffer(2)]],
                          constant uint4& s [[buffer(3)]],uint2 g [[thread_position_in_grid]]){
    uint i=g.y,j=g.x;if(i>=s.x||j>=s.y||(s.w&&j>i))return;Number<N> r=zero<N>();
    for(uint k=0;k<s.z;++k){Number<N> a=A[k*s.x+i],b=B[k*s.y+j];r=add(r,mul(a,b));}
    C[i*s.y+j]=r;
}
kernel void limb_fma(device const Number<N>* A [[buffer(0)]],device const Number<N>* B [[buffer(1)]],device Number<N>* C [[buffer(2)]],
                     constant uint4& s [[buffer(3)]],uint2 g [[thread_position_in_grid]]){
    uint i=g.y,j=g.x;if(i>=s.x||j>=s.y||(s.w&&j>i))return;Number<N> r=zero<N>();
    for(uint k=0;k<s.z;++k){Number<N> a=A[k*s.x+i],b=B[k*s.y+j];r=limbforge::fma(a,b,r);}
    C[i*s.y+j]=r;
}
)METAL";
struct Baseline {
    id<MTLDevice> device;id<MTLCommandQueue> queue;std::map<int,std::pair<id<MTLComputePipelineState>,id<MTLComputePipelineState>>> states;
    Baseline(){device=MTLCreateSystemDefaultDevice();queue=[device newCommandQueue];}
    std::pair<id<MTLComputePipelineState>,id<MTLComputePipelineState>> get(int bits){
        auto f=states.find(bits);if(f!=states.end())return f->second;NSError* e=nil;MTLCompileOptions* o=[MTLCompileOptions new];o.languageVersion=MTLLanguageVersion3_1;o.mathMode=MTLMathModeSafe;
        id<MTLLibrary> lib=[device newLibraryWithSource:[NSString stringWithFormat:@"#define MP_BITS %d\n%s\n%s",bits,limbforge_shader_source,baseline_source] options:o error:&e];
        if(!lib)throw std::runtime_error("baseline library");
        auto make=[&](NSString* n){auto p=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:n] error:&e];if(!p)throw std::runtime_error("baseline pipeline");return p;};
        return states[bits]={make(@"limb_composed"),make(@"limb_fma")};}
    // Returns {device, wall} seconds; wall includes buffer copies.
    std::pair<double,double> run(int bits,bool fused,const void* A,const void* B,std::size_t M,std::size_t Nn,std::size_t K,void* C,bool lower){@autoreleasepool{
        auto start=Clock::now();std::size_t e=bits/8+12;auto st=get(bits);auto p=fused?st.second:st.first;
        id<MTLBuffer> a=[device newBufferWithBytes:A length:K*M*e options:MTLResourceStorageModeShared],b=lower?a:[device newBufferWithBytes:B length:K*Nn*e options:MTLResourceStorageModeShared];
        id<MTLBuffer> c=[device newBufferWithLength:M*Nn*e options:MTLResourceStorageModeShared];std::uint32_t s[4]={std::uint32_t(M),std::uint32_t(Nn),std::uint32_t(K),lower};
        id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];[enc setComputePipelineState:p];
        [enc setBuffer:a offset:0 atIndex:0];[enc setBuffer:b offset:0 atIndex:1];[enc setBuffer:c offset:0 atIndex:2];[enc setBytes:s length:16 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(Nn,M,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.status==MTLCommandBufferStatusError)throw std::runtime_error("baseline execution");std::memcpy(C,c.contents,M*Nn*e);
        return {cb.GPUEndTime-cb.GPUStartTime,std::chrono::duration<double>(Clock::now()-start).count()};}}
};
// QSC-like Jacobian columns: column scale in [-60,60], entries within 2^-25..2^5 of it, 25% zeros, a tiny outlier in 1/17 of columns.
template<int Bits> std::vector<Float<Bits>> jacobian(std::size_t rows,std::size_t cols,bool moderate,std::mt19937_64& rng){
    std::vector<Float<Bits>> A(rows*cols);
    for(std::size_t c=0;c<cols;++c){long scale=moderate?0:long(rng()%121)-60;
        for(std::size_t r=0;r<rows;++r){auto x=reference::random_number<Bits>(rng,moderate?4:15);x.exponent+=int(scale-(moderate?0:10));if(!moderate&&rng()%4==0)x=zero<Bits/32>();A[r*cols+c]=x;}
        if(!moderate&&c%17==5)A[(rng()%rows)*cols+c].exponent=int(scale-150);}
    return A;
}
// Consumer MPFR loop: s = 0; s += a_ki * b_kj (mpfr_mul, mpfr_add, each rounded), outputs of `rows_out` (sampled) rows.
double mpfr_product(int bits,MPArray& a,MPArray& b,std::size_t M,std::size_t Nn,std::size_t K,bool lower,const std::vector<std::size_t>& rows_out,unsigned threads){
    auto start=Clock::now();std::atomic<std::size_t> next{0};std::vector<std::thread> pool;
    for(unsigned t=0;t<threads;++t)pool.emplace_back([&]{mpfr_t s,p;mpfr_init2(s,bits);mpfr_init2(p,bits);
        for(std::size_t q;(q=next.fetch_add(1))<rows_out.size();){std::size_t i=rows_out[q];
            for(std::size_t j=0;j<(lower?i+1:Nn);++j){mpfr_set_zero(s,1);for(std::size_t k=0;k<K;++k){mpfr_mul(p,a[k*M+i],b[k*Nn+j],MPFR_RNDN);mpfr_add(s,s,p,MPFR_RNDN);}}}
        mpfr_clear(s);mpfr_clear(p);});
    for(auto& t:pool)t.join();return std::chrono::duration<double>(Clock::now()-start).count();
}
// Outputs in a set of rows, as a fraction of all outputs (lower triangle for SYRK).
double share(const std::vector<std::size_t>& rows,std::size_t M,std::size_t Nn,bool lower){double s=0,t=lower?double(M)*(M+1)/2:double(M)*Nn;for(auto i:rows)s+=lower?i+1:Nn;return s/t;}
std::vector<std::size_t> sample_rows(std::size_t M,std::size_t Nn,std::size_t K,bool lower,double budget){
    std::vector<std::size_t> r;double per=double(lower?(M+1)/2:Nn)*K;std::size_t want=std::min<std::size_t>(M,std::max<std::size_t>(1,std::size_t(budget/per)));
    for(std::size_t q=0;q<want;++q)r.push_back(q*M/want+(M/want)/2);return r;}
// CPU MPFR Cholesky (right-looking, lower), n^3/6 multiply-adds.
double cholesky(int bits,MPArray& C,std::size_t n){auto start=Clock::now();mpfr_t t;mpfr_init2(t,bits);
    for(std::size_t j=0;j<n;++j){mpfr_sqrt(C[j*n+j],C[j*n+j],MPFR_RNDN);for(std::size_t i=j+1;i<n;++i)mpfr_div(C[i*n+j],C[i*n+j],C[j*n+j],MPFR_RNDN);
        for(std::size_t k=j+1;k<n;++k)for(std::size_t i=k;i<n;++i){mpfr_mul(t,C[i*n+j],C[k*n+j],MPFR_RNDN);mpfr_sub(C[i*n+k],C[i*n+k],t,MPFR_RNDN);}}
    mpfr_clear(t);return std::chrono::duration<double>(Clock::now()-start).count();}
struct Options {int repeats=5;double warm=0.3,serial_budget=4e7,parallel_budget=4e8;std::size_t fma_max=400,chol_max=0;unsigned threads=std::thread::hardware_concurrency();int verify=64;};
template<int Bits> void bench(Linalg& gpu,Baseline& base,const Options& o,std::size_t n,bool syrk,bool moderate,std::mt19937_64& rng){
    std::size_t K=2*n,M=n,Nn=n;
    auto A=jacobian<Bits>(K,n,moderate,rng);auto B=syrk?A:jacobian<Bits>(K,n,moderate,rng);std::vector<Float<Bits>> C(M*Nn),D(M*Nn);
    auto call=[&]{return syrk?gpu.syrk(Bits,A.data(),K,n,C.data(),true):gpu.gemm(Bits,true,A.data(),B.data(),M,Nn,K,C.data());};
    auto warmup=[&](auto&& f){for(auto w=Clock::now();std::chrono::duration<double>(Clock::now()-w).count()<o.warm;)f();};
    call();warmup(call);Samples s;std::vector<double> analysis,upload,assembly;
    for(int r=0;r<o.repeats;++r){auto t=call();s.device.push_back(t.gpu_seconds);s.wall.push_back(t.wall_seconds);
        analysis.push_back(gpu.report().analysis_seconds);upload.push_back(gpu.report().upload_seconds);assembly.push_back(gpu.report().assembly_seconds);}
    auto rep=gpu.report();
    // Spot check against exact MPFR products + mpfr_sum.
    int bad=0;for(int v=0;v<o.verify;++v){std::size_t i=rng()%M,j=rng()%Nn;if(syrk&&j>i)std::swap(i,j);
        if(!reference::equal<Bits>(C[i*Nn+j],reference::dot<Bits>(A.data()+i,std::ptrdiff_t(M),B.data()+j,std::ptrdiff_t(Nn),K)))++bad;}
    if(bad)throw std::runtime_error("Linalg result differs from MPFR");
    std::vector<double> lc,lcw,lf;warmup([&]{base.run(Bits,false,A.data(),B.data(),M,Nn,K,D.data(),syrk);});
    for(int r=0;r<o.repeats;++r){auto t=base.run(Bits,false,A.data(),B.data(),M,Nn,K,D.data(),syrk);lc.push_back(t.first);lcw.push_back(t.second);}
    if(n<=o.fma_max)for(int r=0;r<std::max(1,o.repeats/2);++r)lf.push_back(base.run(Bits,true,A.data(),B.data(),M,Nn,K,D.data(),syrk).first);
    // CPU MPFR consumer loop on sampled rows, extrapolated by output count.
    MPArray ma(K*M,Bits),mb(syrk?1:K*Nn,Bits);for(std::size_t q=0;q<K*M;++q)to_mpfr<Bits>(ma[q],A[q]);if(!syrk)for(std::size_t q=0;q<K*Nn;++q)to_mpfr<Bits>(mb[q],B[q]);
    MPArray& bref=syrk?ma:mb;
    auto rs=sample_rows(M,Nn,K,syrk,o.serial_budget),rp=sample_rows(M,Nn,K,syrk,o.parallel_budget);
    double ser=mpfr_product(Bits,ma,bref,M,Nn,K,syrk,rs,1)/share(rs,M,Nn,syrk),par=mpfr_product(Bits,ma,bref,M,Nn,K,syrk,rp,o.threads)/share(rp,M,Nn,syrk);
    double chol=0;if(syrk&&n<=o.chol_max){MPArray mc(n*n,Bits);for(std::size_t q=0;q<n*n;++q)if(q/n>=q%n)to_mpfr<Bits>(mc[q],C[q]);chol=cholesky(Bits,mc,n);}
    unsigned moduli=0;for(auto& p:rep.pairs)moduli=std::max(moduli,p.moduli);std::string bands;for(std::size_t b=1;b<rep.left_bands.size();++b)bands+=(b>1?";":"")+std::to_string(rep.left_bands[b]);
    std::cout<<std::setprecision(5)<<Bits<<','<<(moderate?"moderate":"qsc")<<','<<(syrk?"syrk_lower":"gemm_AtB")<<','<<n<<','<<K<<','<<o.repeats<<','
        <<quantile(s.device,.5)*1e3<<','<<quantile(s.wall,.5)*1e3<<','<<quantile(analysis,.5)*1e3<<','<<quantile(upload,.5)*1e3<<','<<quantile(assembly,.5)*1e3<<','
        <<bands<<','<<rep.left_fallback+rep.right_fallback<<','<<rep.pairs.size()<<','<<moduli<<','<<rep.int8_gemms<<','<<rep.multi_band_outputs<<','<<rep.fallback_outputs<<','
        <<quantile(lc,.5)*1e3<<','<<quantile(lcw,.5)*1e3<<','<<(lf.empty()?0:quantile(lf,.5)*1e3)<<','<<ser<<','<<par<<','<<o.threads<<','
        <<share(rs,M,Nn,syrk)<<','<<share(rp,M,Nn,syrk)<<','<<chol<<','<<o.verify<<std::endl;
}
int main(int argc,char** argv){try{
    Options o;std::vector<int> bits={224,256};std::vector<std::size_t> sizes={200,400,800,1000};bool gemm=true,moderate=false;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto next=[&]{if(i+1>=argc)throw std::invalid_argument("missing value");return std::string(argv[++i]);};
        if(a=="--bits")bits={std::stoi(next())};else if(a=="--size")sizes={std::size_t(std::stoul(next()))};else if(a=="--repeats")o.repeats=std::stoi(next());else if(a=="--gpu-warm")o.warm=std::stod(next());
        else if(a=="--threads")o.threads=unsigned(std::stoul(next()));else if(a=="--serial-budget")o.serial_budget=std::stod(next());
        else if(a=="--parallel-budget")o.parallel_budget=std::stod(next());else if(a=="--fma-max")o.fma_max=std::stoul(next());
        else if(a=="--cholesky-max")o.chol_max=std::stoul(next());else if(a=="--no-gemm")gemm=false;else if(a=="--moderate")moderate=true;
        else throw std::invalid_argument("usage: linalg_limbforge [--bits 224|256] [--size n] [--repeats r] [--gpu-warm seconds] [--threads t] [--serial-budget MACs] [--parallel-budget MACs] [--fma-max n] [--cholesky-max n] [--no-gemm] [--moderate]");}
    Linalg gpu;Baseline base;std::mt19937_64 rng(20261007);
    std::cerr<<gpu.device_name()<<"; load average sampled by the caller; times: device = GPU command buffer, wall = host call\n";
    std::cout<<"bits,data,op,n,k,repeats,residue_device_ms,residue_wall_ms,analysis_ms,upload_ms,assembly_ms,lines_by_bands,fallback_lines,pairs,moduli_max,int8_gemms,"
               "multi_band_outputs,fallback_outputs,limb_composed_device_ms,limb_composed_wall_ms,limb_fma_device_ms,mpfr_serial_s,mpfr_parallel_s,mpfr_threads,"
               "serial_sampled_share,parallel_sampled_share,cholesky_mpfr_s,verified\n";
    for(int b:bits)for(auto n:sizes)for(bool syrk:{true,false}){if(!syrk&&!gemm)continue;
        if(b==224)bench<224>(gpu,base,o,n,syrk,moderate,rng);else if(b==256)bench<256>(gpu,base,o,n,syrk,moderate,rng);else throw std::invalid_argument("bits: 224 or 256");}
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
