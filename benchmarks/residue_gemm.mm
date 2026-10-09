// Experiment L1b (plan section 8): exact integer GEMM of P-bit signed integers through residues on
// Metal TensorOps int8 matmul, reconstructed by Garner's algorithm on the GPU and checked against GMP.
// Baselines: a WolfNum floating GEMM per output, with sequential fma and with composed mul+add.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "reference.hpp"
#include "shader_source.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
using namespace limbforge;
const char* residue_source=R"METAL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct Shape { uint M,N,K,W,L,Wc,modulus,j; };
// Balanced residue v in (-m/2, m/2] as two int8 digits v = 256*d1 + d0 (m <= 65279 keeps |d1| <= 127).
inline int2 digits(device const uint* limbs,char sign,uint W,uint m){
    uint r=0;for(int w=int(W)-1;w>=0;--w){uint x=limbs[w];r=((r<<16)|(x>>16))%m;r=((r<<16)|(x&0xffffu))%m;}
    int v=int(r);if(sign<0)v=v?int(m)-v:0;if(v>int(m/2))v-=int(m);
    int d0=((v+128)&255)-128;return int2((v-d0)/256,d0);
}
// A (M x K) -> cat = [d1 | d0] (M x 2K), a1 (M x K), a0 (M x K).
kernel void residues_a(device const uint* limbs [[buffer(0)]],device const char* sign [[buffer(1)]],device char* cat [[buffer(2)]],
                       device char* a1 [[buffer(3)]],device char* a0 [[buffer(4)]],constant Shape& s [[buffer(5)]],uint idx [[thread_position_in_grid]]){
    if(idx>=s.M*s.K)return;uint row=idx/s.K,col=idx%s.K;int2 d=digits(limbs+ulong(idx)*s.W,sign[idx],s.W,s.modulus);
    cat[ulong(row)*2*s.K+col]=char(d.x);cat[ulong(row)*2*s.K+s.K+col]=char(d.y);a1[idx]=char(d.x);a0[idx]=char(d.y);
}
// B (K x N) -> cat = [d0 ; d1] (2K x N): rows 0..K-1 hold d0, rows K..2K-1 hold d1.
kernel void residues_b(device const uint* limbs [[buffer(0)]],device const char* sign [[buffer(1)]],device char* cat [[buffer(2)]],
                       constant Shape& s [[buffer(5)]],uint idx [[thread_position_in_grid]]){
    if(idx>=s.K*s.N)return;int2 d=digits(limbs+ulong(idx)*s.W,sign[idx],s.W,s.modulus);
    cat[idx]=char(d.y);cat[ulong(s.K)*s.N+idx]=char(d.x);
}
// C (M x N) = A (M x K) * B (K x N), int8 -> exact int32 for K*2^14 < 2^31.
kernel void product(device int8_t* A [[buffer(0)]],device int8_t* B [[buffer(1)]],device int32_t* C [[buffer(2)]],
                    constant uint3& mnk [[buffer(3)]],uint2 tgid [[threadgroup_position_in_grid]]){
    int M=int(mnk.x),N=int(mnk.y),K=int(mnk.z);
    auto tA=tensor<device int8_t,dextents<int32_t,2>,tensor_inline>(A,dextents<int32_t,2>(K,M));
    auto tB=tensor<device int8_t,dextents<int32_t,2>,tensor_inline>(B,dextents<int32_t,2>(N,K));
    auto tC=tensor<device int32_t,dextents<int32_t,2>,tensor_inline>(C,dextents<int32_t,2>(N,M));
    constexpr auto d=matmul2d_descriptor(64,32,static_cast<int>(dynamic_extent),false,false,false);
    matmul2d<d,execution_simdgroups<4>> op;
    auto mA=tA.slice(0,int(tgid.y)*64);auto mB=tB.slice(int(tgid.x)*32,0);auto mC=tC.slice(int(tgid.x)*32,int(tgid.y)*64);
    op.run(mA,mB,mC);
}
inline uint reduce(int g,uint m){int r=g%int(m);return uint(r<0?r+int(m):r);}
// residue_j = (65536*G11 + 256*Gmid + G00) mod m, all in 32-bit arithmetic.
kernel void combine(device const int* g11 [[buffer(0)]],device const int* gmid [[buffer(1)]],device const int* g00 [[buffer(2)]],
                    device ushort* res [[buffer(3)]],constant Shape& s [[buffer(5)]],uint idx [[thread_position_in_grid]]){
    if(idx>=s.M*s.N)return;uint m=s.modulus;
    uint r=(reduce(g11[idx],m)*65536u)%m;r=(r+(reduce(gmid[idx],m)*256u)%m)%m;r=(r+reduce(g00[idx],m))%m;
    res[ulong(s.j)*s.M*s.N+idx]=ushort(r);
}
// Garner mixed-radix digits, Horner reconstruction, symmetric range -> magnitude limbs and sign.
kernel void reconstruct(device const ushort* res [[buffer(0)]],device const uint* moduli [[buffer(1)]],device const uint* inv [[buffer(2)]],
                        device const uint* mprod [[buffer(3)]],device const uint* mhalf [[buffer(4)]],constant Shape& s [[buffer(5)]],
                        device uint* out [[buffer(6)]],device char* osign [[buffer(7)]],uint idx [[thread_position_in_grid]]){
    if(idx>=s.M*s.N)return;uint L=s.L,Wc=s.Wc;ulong MN=ulong(s.M)*s.N;uint v[64];
    for(uint j=0;j<L;++j){uint m=moduli[j],t=res[ulong(j)*MN+idx];
        for(uint i=0;i<j;++i){t=(t+m-v[i]%m)%m;t=(t*inv[i*L+j])%m;}v[j]=t;}
    uint X[48];for(uint k=0;k<Wc;++k)X[k]=0;X[0]=v[L-1];
    for(int j=int(L)-2;j>=0;--j){ulong carry=v[j],m=moduli[j];for(uint k=0;k<Wc;++k){ulong p=ulong(X[k])*m+carry;X[k]=uint(p);carry=p>>32;}}
    int cmp=0;for(int k=int(Wc)-1;k>=0;--k)if(X[k]!=mhalf[k]){cmp=X[k]>mhalf[k]?1:-1;break;}
    char sg=1;if(cmp>0){long borrow=0;for(uint k=0;k<Wc;++k){long d=long(mprod[k])-long(X[k])-borrow;X[k]=uint(d);borrow=d<0;}sg=-1;}
    bool nz=false;for(uint k=0;k<Wc;++k){out[ulong(idx)*Wc+k]=X[k];nz|=X[k]!=0;}osign[idx]=nz?sg:0;
}
)METAL";
const char* baseline_source=R"METAL(
// One thread per output, sequential over k: fused (fma) or composed (mul then add).
kernel void gemm_fma(device const Number<N>* A [[buffer(0)]],device const Number<N>* B [[buffer(1)]],device Number<N>* C [[buffer(2)]],
                     constant uint3& mnk [[buffer(3)]],uint2 g [[thread_position_in_grid]]){
    if(g.x>=mnk.y||g.y>=mnk.x)return;Number<N> s=zero<N>();
    for(uint k=0;k<mnk.z;++k){Number<N> a=A[g.y*mnk.z+k],b=B[k*mnk.y+g.x];s=limbforge::fma(a,b,s);}
    C[g.y*mnk.y+g.x]=s;
}
kernel void gemm_composed(device const Number<N>* A [[buffer(0)]],device const Number<N>* B [[buffer(1)]],device Number<N>* C [[buffer(2)]],
                          constant uint3& mnk [[buffer(3)]],uint2 g [[thread_position_in_grid]]){
    if(g.x>=mnk.y||g.y>=mnk.x)return;Number<N> s=zero<N>();
    for(uint k=0;k<mnk.z;++k){Number<N> a=A[g.y*mnk.z+k],b=B[k*mnk.y+g.x];s=add(s,mul(a,b));}
    C[g.y*mnk.y+g.x]=s;
}
)METAL";
struct Shape {std::uint32_t M,N,K,W,L,Wc,modulus,j;};
id<MTLDevice> device;id<MTLCommandQueue> queue;
std::string text(NSError* e){return e?std::string([[e localizedDescription] UTF8String]):"unknown";}
id<MTLComputePipelineState> pipeline(id<MTLLibrary> lib,NSString* name){NSError* e=nil;id<MTLFunction> f=[lib newFunctionWithName:name];
    if(!f)throw std::runtime_error("missing function");auto p=[device newComputePipelineStateWithFunction:f error:&e];if(!p)throw std::runtime_error("pipeline: "+text(e));return p;}
id<MTLBuffer> buffer(std::size_t bytes,const void* data=nullptr){id<MTLBuffer> b=[device newBufferWithLength:std::max<std::size_t>(bytes,4) options:MTLResourceStorageModeShared];
    if(!b)throw std::runtime_error("allocation failed");if(data)std::memcpy(b.contents,data,bytes);return b;}
void dispatch1(id<MTLComputeCommandEncoder> e,id<MTLComputePipelineState> p,std::size_t n){[e setComputePipelineState:p];
    [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256,p.maxTotalThreadsPerThreadgroup),1,1)];}
std::vector<std::uint32_t> primes_below(std::uint32_t limit,std::size_t count){std::vector<std::uint32_t> r;
    for(std::uint32_t p=limit;r.size()<count&&p>2;--p){bool prime=true;for(std::uint32_t d=2;d*d<=p;++d)if(p%d==0){prime=false;break;}if(prime)r.push_back(p);}return r;}
std::uint32_t inverse(std::uint64_t a,std::uint64_t m){std::int64_t t=0,nt=1,r=std::int64_t(m),nr=std::int64_t(a%m);
    while(nr){std::int64_t q=r/nr;std::tie(t,nt)=std::make_pair(nt,t-q*nt);std::tie(r,nr)=std::make_pair(nr,r-q*nr);}return std::uint32_t(t<0?t+std::int64_t(m):t);}
struct Stages {double total=0,residues=0,products=0,combine=0,reconstruct=0;};
double seconds(id<MTLCommandBuffer> c){[c commit];[c waitUntilCompleted];if(c.status==MTLCommandBufferStatusError)throw std::runtime_error("execution: "+text(c.error));return c.GPUEndTime-c.GPUStartTime;}
double median(std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];}
int main(int argc,char** argv){try{
    int P=256,verify=64,repeats=3;std::vector<int> sizes={128,256,512,1024,2048};int baseline_max=1024;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto next=[&]{if(i+1>=argc)throw std::invalid_argument("missing value");return std::string(argv[++i]);};
        if(a=="--bits")P=std::stoi(next());else if(a=="--size")sizes={std::stoi(next())};else if(a=="--verify")verify=std::stoi(next());
        else if(a=="--baseline-max")baseline_max=std::stoi(next());else if(a=="--repeats")repeats=std::stoi(next());
        else throw std::invalid_argument("usage: residue_gemm [--bits 256] [--size n] [--verify outputs] [--baseline-max n] [--repeats r]");}
    if(P!=256)throw std::invalid_argument("this prototype's baseline is compiled for 256 bits");
    device=MTLCreateSystemDefaultDevice();queue=[device newCommandQueue];NSError* e=nil;
    MTLCompileOptions* o4=[MTLCompileOptions new];o4.languageVersion=MTLLanguageVersion((4u<<16)|0u);
    id<MTLLibrary> rl=[device newLibraryWithSource:@(residue_source) options:o4 error:&e];if(!rl)throw std::runtime_error("residue library: "+text(e));
    MTLCompileOptions* o3=[MTLCompileOptions new];o3.languageVersion=MTLLanguageVersion3_1;o3.mathMode=MTLMathModeSafe;
    id<MTLLibrary> bl=[device newLibraryWithSource:[NSString stringWithFormat:@"#define MP_BITS 256\n%s\n%s",limbforge_shader_source,baseline_source] options:o3 error:&e];
    if(!bl)throw std::runtime_error("baseline library: "+text(e));
    auto pa=pipeline(rl,@"residues_a"),pb=pipeline(rl,@"residues_b"),pp=pipeline(rl,@"product"),pc=pipeline(rl,@"combine"),pr=pipeline(rl,@"reconstruct");
    auto pf=pipeline(bl,@"gemm_fma"),pm=pipeline(bl,@"gemm_composed");
    std::cerr<<[device.name UTF8String]<<"; exact "<<P<<"-bit integer GEMM via int8 TensorOps residues; baseline: WolfNum 256-bit floating GEMM\n";
    std::cout<<std::setprecision(5)<<"n,moduli,residue_total_ms,residues_ms,products_ms,combine_ms,reconstruct_ms,verified,baseline_fma_ms,baseline_composed_ms,fma_over_residue,composed_over_residue\n";
    const std::uint32_t W=P/32;std::mt19937_64 rng(20261007);
    for(int n:sizes){@autoreleasepool{
        const std::uint32_t M=n,N=n,K=n;
        // Modulus product must exceed 2*K*2^(2P): |C| < K*2^(2P).
        double need=2.0*P+std::log2(double(K))+2;std::vector<std::uint32_t> mod;double have=0;
        for(auto p:primes_below(65279,200)){mod.push_back(p);have+=std::log2(double(p));if(have>need)break;}
        const std::uint32_t L=std::uint32_t(mod.size()),Wc=std::uint32_t(std::ceil(have/32))+1;
        if(L>64||Wc>48)throw std::runtime_error("modulus set exceeds kernel limits");
        std::vector<std::uint32_t> inv(L*L,0);for(std::uint32_t i=0;i<L;++i)for(std::uint32_t j=i+1;j<L;++j)inv[i*L+j]=inverse(mod[i],mod[j]);
        mpz_t mp,mh;mpz_init_set_ui(mp,1);mpz_init(mh);for(auto m:mod)mpz_mul_ui(mp,mp,m);mpz_fdiv_q_2exp(mh,mp,1);
        std::vector<std::uint32_t> mprod(Wc,0),mhalf(Wc,0);std::size_t cnt;mpz_export(mprod.data(),&cnt,-1,4,0,0,mp);mpz_export(mhalf.data(),&cnt,-1,4,0,0,mh);
        std::vector<std::uint32_t> al(std::size_t(M)*K*W),bl_(std::size_t(K)*N*W);std::vector<char> as(std::size_t(M)*K),bs(std::size_t(K)*N);
        for(auto& x:al)x=std::uint32_t(rng());for(auto& x:bl_)x=std::uint32_t(rng());for(auto& x:as)x=(rng()&1)?1:-1;for(auto& x:bs)x=(rng()&1)?1:-1;
        auto A=buffer(al.size()*4,al.data()),B=buffer(bl_.size()*4,bl_.data()),SA=buffer(as.size(),as.data()),SB=buffer(bs.size(),bs.data());
        auto acat=buffer(std::size_t(M)*2*K),a1=buffer(std::size_t(M)*K),a0=buffer(std::size_t(M)*K),bcat=buffer(std::size_t(2)*K*N);
        auto g11=buffer(std::size_t(M)*N*4),gmid=buffer(std::size_t(M)*N*4),g00=buffer(std::size_t(M)*N*4),res=buffer(std::size_t(L)*M*N*2);
        auto bmod=buffer(L*4,mod.data()),binv=buffer(inv.size()*4,inv.data()),bmp=buffer(Wc*4,mprod.data()),bmh=buffer(Wc*4,mhalf.data());
        auto out=buffer(std::size_t(M)*N*Wc*4),osign=buffer(std::size_t(M)*N);
        std::uint32_t mnk[3]={M,N,K},mnk2[3]={M,N,2*K};
        auto product=[&](id<MTLComputeCommandEncoder> enc,id<MTLBuffer> a,id<MTLBuffer> b,std::size_t boff,id<MTLBuffer> c,std::uint32_t* shape){
            [enc setComputePipelineState:pp];[enc setBuffer:a offset:0 atIndex:0];[enc setBuffer:b offset:boff atIndex:1];[enc setBuffer:c offset:0 atIndex:2];
            [enc setBytes:shape length:12 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((N+31)/32,(M+63)/64,1) threadsPerThreadgroup:MTLSizeMake(pp.threadExecutionWidth*4,1,1)];};
        // Stage mask: 1 residues, 2 products, 4 combine, 8 reconstruct.
        auto run=[&](int mask){id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
            for(std::uint32_t j=0;j<L;++j){Shape s={M,N,K,W,L,Wc,mod[j],j};[enc setBytes:&s length:sizeof(s) atIndex:5];
                if(mask&1){[enc setBuffer:A offset:0 atIndex:0];[enc setBuffer:SA offset:0 atIndex:1];[enc setBuffer:acat offset:0 atIndex:2];
                    [enc setBuffer:a1 offset:0 atIndex:3];[enc setBuffer:a0 offset:0 atIndex:4];dispatch1(enc,pa,std::size_t(M)*K);
                    [enc setBuffer:B offset:0 atIndex:0];[enc setBuffer:SB offset:0 atIndex:1];[enc setBuffer:bcat offset:0 atIndex:2];dispatch1(enc,pb,std::size_t(K)*N);}
                if(mask&2){product(enc,a1,bcat,std::size_t(K)*N,g11,mnk);product(enc,a0,bcat,0,g00,mnk);product(enc,acat,bcat,0,gmid,mnk2);}
                if(mask&4){[enc setBytes:&s length:sizeof(s) atIndex:5];[enc setBuffer:g11 offset:0 atIndex:0];[enc setBuffer:gmid offset:0 atIndex:1];
                    [enc setBuffer:g00 offset:0 atIndex:2];[enc setBuffer:res offset:0 atIndex:3];dispatch1(enc,pc,std::size_t(M)*N);}
            }
            if(mask&8){Shape s={M,N,K,W,L,Wc,0,0};[enc setBytes:&s length:sizeof(s) atIndex:5];[enc setBuffer:res offset:0 atIndex:0];[enc setBuffer:bmod offset:0 atIndex:1];
                [enc setBuffer:binv offset:0 atIndex:2];[enc setBuffer:bmp offset:0 atIndex:3];[enc setBuffer:bmh offset:0 atIndex:4];
                [enc setBuffer:out offset:0 atIndex:6];[enc setBuffer:osign offset:0 atIndex:7];dispatch1(enc,pr,std::size_t(M)*N);}
            [enc endEncoding];return seconds(cb);};
        run(15);Stages st;std::vector<double> t[5];
        for(int r=0;r<repeats;++r){t[0].push_back(run(15));t[1].push_back(run(1));t[2].push_back(run(2));t[3].push_back(run(4));t[4].push_back(run(8));}
        st.total=median(t[0]);st.residues=median(t[1]);st.products=median(t[2]);st.combine=median(t[3]);st.reconstruct=median(t[4]);
        run(15);// final consistent result for verification
        int bad=0,checked=0;mpz_t x,y,acc,got;mpz_inits(x,y,acc,got,nullptr);
        const std::uint32_t* o=static_cast<const std::uint32_t*>(out.contents);const char* os=static_cast<const char*>(osign.contents);
        for(int v=0;v<verify;++v){std::uint32_t i=std::uint32_t(rng()%M),jj=std::uint32_t(rng()%N);if(v==0){i=0;jj=0;}mpz_set_ui(acc,0);
            for(std::uint32_t k=0;k<K;++k){mpz_import(x,W,-1,4,0,0,&al[(std::size_t(i)*K+k)*W]);if(as[std::size_t(i)*K+k]<0)mpz_neg(x,x);
                mpz_import(y,W,-1,4,0,0,&bl_[(std::size_t(k)*N+jj)*W]);if(bs[std::size_t(k)*N+jj]<0)mpz_neg(y,y);mpz_addmul(acc,x,y);}
            mpz_import(got,Wc,-1,4,0,0,o+(std::size_t(i)*N+jj)*Wc);if(os[std::size_t(i)*N+jj]<0)mpz_neg(got,got);
            ++checked;if(mpz_cmp(acc,got))++bad;}
        mpz_clears(x,y,acc,got,mp,mh,nullptr);
        if(bad)throw std::runtime_error("residue GEMM differs from GMP at "+std::to_string(bad)+" of "+std::to_string(checked)+" outputs, n="+std::to_string(n));
        double tf=0,tc=0;
        if(n<=baseline_max){std::vector<Float<256>> fa(std::size_t(M)*K),fb(std::size_t(K)*N);for(auto& v:fa)v=reference::random_number<256>(rng,5);for(auto& v:fb)v=reference::random_number<256>(rng,5);
            auto FA=buffer(fa.size()*sizeof(fa[0]),fa.data()),FB=buffer(fb.size()*sizeof(fb[0]),fb.data()),FC=buffer(std::size_t(M)*N*sizeof(fa[0]));
            auto gemm=[&](id<MTLComputePipelineState> p){id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                [enc setComputePipelineState:p];[enc setBuffer:FA offset:0 atIndex:0];[enc setBuffer:FB offset:0 atIndex:1];[enc setBuffer:FC offset:0 atIndex:2];[enc setBytes:mnk length:12 atIndex:3];
                [enc dispatchThreads:MTLSizeMake(N,M,1) threadsPerThreadgroup:MTLSizeMake(32,4,1)];[enc endEncoding];return seconds(cb);};
            gemm(pf);std::vector<double> f,c;for(int r=0;r<repeats;++r){f.push_back(gemm(pf));c.push_back(gemm(pm));}tf=median(f);tc=median(c);
            // Spot-check the baseline against the CPU core (same sequential order).
            const auto* fc=static_cast<const Float<256>*>(FC.contents);gemm(pf);
            for(int v=0;v<8;++v){std::uint32_t i=std::uint32_t(rng()%M),jj=std::uint32_t(rng()%N);Float<256> s=zero<8>();
                for(std::uint32_t k=0;k<K;++k)s=fma(fa[std::size_t(i)*K+k],fb[std::size_t(k)*N+jj],s);
                if(!reference::equal<256>(s,fc[std::size_t(i)*N+jj]))throw std::runtime_error("baseline fma GEMM differs from CPU core");}
        }
        std::cout<<n<<','<<L<<','<<st.total*1e3<<','<<st.residues*1e3<<','<<st.products*1e3<<','<<st.combine*1e3<<','<<st.reconstruct*1e3<<','<<checked
            <<','<<tf*1e3<<','<<tc*1e3<<','<<(tf?tf/st.total:0)<<','<<(tc?tc/st.total:0)<<std::endl;
    }}
    return 0;
}catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}}
