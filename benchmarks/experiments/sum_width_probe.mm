// GPU sweep: the update rounding of src/linalg.metal (rounded<W> with SUB) at every N and the two widths it is
// instantiated with (LF_TERM_WORDS, LF_ACC_WORDS), against MPFR. Argument: workspace rounding (1 = none, 4 = library).
// Run from the repository root (reads include/limbforge/core.hpp). Build:
//   c++ -std=c++17 -O2 -fobjc-arc -I/opt/homebrew/include benchmarks/experiments/sum_width_probe.mm -o sum_width_probe \
//       -L/opt/homebrew/lib -lmpfr -lgmp -framework Metal -framework Foundation
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mpfr.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#include <cstring>
const char* kernel_src=R"(
using namespace metal;using namespace limbforge;
constant int N=LF_N;constant int CW=scratch(N);
constexpr int sum_words(int w){return (scratch(w)+RW-1)/RW*RW;}
template<int A> void body(device const uint* xs,device const uint* cs,device uint* out,uint i){
    constexpr int S=A+2;Exact<A> x;for(int k=0;k<A;++k)x.w[k]=xs[i*S+k];x.scale=as_type<int>(xs[i*S+A]);x.sign=as_type<int>(xs[i*S+A+1]);
    Number<N> c;for(int k=0;k<N;++k)c.limb[k]=cs[i*(N+2)+k];c.exponent=as_type<int>(cs[i*(N+2)+N]);c.sign=as_type<int>(cs[i*(N+2)+N+1]);c.status=0;
    Exact<CW> y=exact_number<CW>(c);Number<N> r=round_exact<N>(exact_add<N,sum_words(exact_words(N,A,CW))>(x,y));
    for(int k=0;k<N;++k)out[i*(N+2)+k]=r.limb[k];out[i*(N+2)+N]=as_type<uint>(r.exponent);out[i*(N+2)+N+1]=as_type<uint>(r.sign);
}
kernel void term(device const uint* xs [[buffer(0)]],device const uint* cs [[buffer(1)]],device uint* out [[buffer(2)]],uint i [[thread_position_in_grid]]){body<LF_TERM>(xs,cs,out,i);}
kernel void acc(device const uint* xs [[buffer(0)]],device const uint* cs [[buffer(1)]],device uint* out [[buffer(2)]],uint i [[thread_position_in_grid]]){body<LF_ACC>(xs,cs,out,i);}
)";
int padded(int w){return w>=13&&w<33?33:w;}
int main(int argc,char** argv){@autoreleasepool{
    int rw=argc>1?std::atoi(argv[1]):4;
    std::ifstream f("include/limbforge/core.hpp");std::stringstream ss;ss<<f.rdbuf();std::string core=ss.str();core.replace(core.find("#pragma once"),12,"");
    std::vector<unsigned> primes;for(unsigned q=65279;primes.size()<1200;--q){bool p=true;for(unsigned d=2;d*d<=q;++d)if(q%d==0){p=false;break;}if(p)primes.push_back(q);}
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> queue=[dev newCommandQueue];int total_bad=0;
    for(int N=2;N<=32;++N){
        int bits=32*N;double need=2.0*(bits+64)+std::log2(65472.0)+2,have=0;int L=0;while(have<=need)have+=std::log2(double(primes[L++]));
        int ml=padded(L);for(int i=L;i<ml;++i)have+=std::log2(double(primes[i]));int TW=padded(int(std::ceil(have/32))+1),AW=padded(2*N+(2*512+18+31)/32+1);
        std::string src="#include <metal_stdlib>\n#define LF_N "+std::to_string(N)+"\n#define LF_TERM "+std::to_string(TW)+"\n#define LF_ACC "+std::to_string(AW)+"\n#define RW "+std::to_string(rw)+"\n"+core+kernel_src;
        NSError* e=nil;MTLCompileOptions* o=[MTLCompileOptions new];o.mathMode=MTLMathModeSafe;
        id<MTLLibrary> lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:o error:&e];if(!lib){std::cout<<[[e description] UTF8String]<<"\n";return 1;}
        for(int which=0;which<2;++which){int A=which?AW:TW;const int count=8192;std::mt19937_64 rng(N*100+which);
            id<MTLComputePipelineState> p=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:which?@"acc":@"term"] error:&e];
            std::vector<std::uint32_t> xs(count*(A+2)),cs(count*(N+2));
            for(int it=0;it<count;++it){std::uint32_t* w=&xs[it*(A+2)];int top=int(rng()%A);for(int i=0;i<A;++i)w[i]=i<=top?std::uint32_t(rng()):0;
                if(rng()%3==0)for(int i=0;i<top;++i)w[i]=0;if(!w[top])w[top]=1;int scale=int(rng()%2000)-1000,sign=rng()&1?1:-1;std::memcpy(&w[A],&scale,4);std::memcpy(&w[A+1],&sign,4);
                std::uint32_t* c=&cs[it*(N+2)];for(int i=0;i<N;++i)c[i]=std::uint32_t(rng());c[N-1]|=0x80000000u;
                int mode=int(rng()%4),ce;if(mode==0)ce=scale+32*top+int(rng()%64)-32;                  // near the top of x: cancellation when signs differ
                else if(mode==1){ce=scale+32*top+31;for(int i=0;i<N;++i)c[i]=w[top-N+1+i>=0&&top-N+1+i<A?top-N+1+i:0];c[N-1]|=0x80000000u;} // near-exact cancellation
                else ce=scale+32*top+int(rng()%(64*N+400))-32*N-200;
                int cs_=rng()&1?1:-1;std::memcpy(&c[N],&ce,4);std::memcpy(&c[N+1],&cs_,4);}
            id<MTLBuffer> bx=[dev newBufferWithBytes:xs.data() length:xs.size()*4 options:0],bc=[dev newBufferWithBytes:cs.data() length:cs.size()*4 options:0],bo=[dev newBufferWithLength:cs.size()*4 options:0];
            id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];[enc setComputePipelineState:p];
            [enc setBuffer:bx offset:0 atIndex:0];[enc setBuffer:bc offset:0 atIndex:1];[enc setBuffer:bo offset:0 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];
            auto* g=static_cast<std::uint32_t*>(bo.contents);int bad=0;mpfr_t X,C,R,G;mpfr_inits2(32*A+64,X,(mpfr_ptr)0);mpfr_init2(C,32*N);mpfr_init2(R,32*N);mpfr_init2(G,32*N);
            for(int it=0;it<count;++it){const std::uint32_t* w=&xs[it*(A+2)];const std::uint32_t* c=&cs[it*(N+2)];int scale,sign,ce,csg;std::memcpy(&scale,&w[A],4);std::memcpy(&sign,&w[A+1],4);std::memcpy(&ce,&c[N],4);std::memcpy(&csg,&c[N+1],4);
                mpfr_set_ui(X,0,MPFR_RNDN);for(int i=A-1;i>=0;--i){mpfr_mul_2ui(X,X,32,MPFR_RNDN);mpfr_add_ui(X,X,w[i],MPFR_RNDN);}mpfr_mul_2si(X,X,scale,MPFR_RNDN);if(sign<0)mpfr_neg(X,X,MPFR_RNDN);
                mpfr_set_ui(C,0,MPFR_RNDN);for(int i=N-1;i>=0;--i){mpfr_mul_2ui(C,C,32,MPFR_RNDN);mpfr_add_ui(C,C,c[i],MPFR_RNDN);}mpfr_mul_2si(C,C,ce-(32*N-1),MPFR_RNDN);if(csg<0)mpfr_neg(C,C,MPFR_RNDN);
                mpfr_add(R,X,C,MPFR_RNDN);const std::uint32_t* o=&g[it*(N+2)];int ge,gs;std::memcpy(&ge,&o[N],4);std::memcpy(&gs,&o[N+1],4);
                mpfr_set_ui(G,0,MPFR_RNDN);for(int i=N-1;i>=0;--i){mpfr_mul_2ui(G,G,32,MPFR_RNDN);mpfr_add_ui(G,G,o[i],MPFR_RNDN);}mpfr_mul_2si(G,G,ge-(32*N-1),MPFR_RNDN);if(gs<0)mpfr_neg(G,G,MPFR_RNDN);
                if(mpfr_cmp(G,R)!=0||(gs==0)!=(mpfr_zero_p(R)!=0))++bad;}
            mpfr_clears(X,C,R,G,(mpfr_ptr)0);total_bad+=bad;
            if(bad||which)std::cout<<"N="<<N<<" "<<(which?"acc":"term")<<" A="<<A<<" bad="<<bad<<"/"<<count<<(bad?"  <-- FAIL":"")<<"\n";
        }
    }
    std::cout<<"total bad "<<total_bad<<"\n";return total_bad?1:0;}}
