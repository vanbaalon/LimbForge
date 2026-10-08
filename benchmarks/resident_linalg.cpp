// Round 42: resident Linalg::syrk / gemm in an Engine's CommandBatch (operands already in Buffers; GPU band analysis and plan)
// versus the host-array calls (copies, host analysis, own command buffer) on the same data, bit-identical results checked.
// SYRK: QSC-like A (2n x n, column scales 2^+-60), lower triangle; GEMM: C (n x n) = A^T B, A and B (n x n) moderate.
// Repeats are interleaved (host, resident, alternating order) after a warm-up; medians. Differences under 15% are noise on a
// loaded host. Columns: wall host, wall resident (batch -> wait), GPU resident (command buffer), ratio host/resident.
// Usage: resident_linalg_limbforge [--bits B] [--repeats R] [--sizes n1,n2,...]
#include "reference.hpp"
#include "limbforge/linalg.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <sstream>
using namespace limbforge;
using Clock=std::chrono::steady_clock;
double median(std::vector<double> v){std::sort(v.begin(),v.end());return v.empty()?0:v[v.size()/2];}
template<int B> std::vector<Float<B>> fixture(std::size_t rows,std::size_t cols,bool qsc,std::mt19937_64& rng){
    std::vector<Float<B>> A(rows*cols);
    for(std::size_t c=0;c<cols;++c){long scale=qsc?long(rng()%121)-60:0;
        for(std::size_t r=0;r<rows;++r){auto x=reference::random_number<B>(rng,qsc?15:4);x.exponent+=int(scale);if(qsc&&rng()%4==0)x=zero<B/32>();A[r*cols+c]=x;}}
    return A;
}
template<int B> void run(Engine& e,Linalg& la,const std::vector<std::size_t>& sizes,int repeats){
    using T=Float<B>;std::mt19937_64 rng(42);
    std::printf("%-5s %5s %5s %12s %12s %12s %12s %8s %s\n","op","bits","n","host_ms","host_gpu_ms","resident_ms","res_gpu_ms","ratio","note");
    for(std::size_t n:sizes)for(int op=0;op<2;++op){const bool syrk=op==0;const std::size_t rows=syrk?2*n:n,cols=n;
        auto A=fixture<B>(rows,cols,syrk,rng),Bm=fixture<B>(n,n,false,rng);std::vector<T> host(n*n,zero<B/32>()),res(n*n);
        auto Ab=e.make_buffer<T>(A.size()),Bb=e.make_buffer<T>(Bm.size()),Cb=e.make_buffer<T>(n*n);Ab.upload(A.data(),A.size());Bb.upload(Bm.data(),Bm.size());
        auto host_call=[&]{auto t=Clock::now();if(syrk)la.syrk(B,A.data(),rows,cols,host.data(),true);else la.gemm(B,true,A.data(),Bm.data(),n,n,n,host.data());
            return std::chrono::duration<double>(Clock::now()-t).count();};
        double gpu=0;auto resident_call=[&]{auto t=Clock::now();auto batch=e.batch();
            if(syrk)la.syrk(batch,Ab,rows,cols,Cb,true);else la.gemm(batch,true,Ab,Bb,n,n,n,Cb);
            gpu=batch.submit().wait().gpu_seconds;return std::chrono::duration<double>(Clock::now()-t).count();};
        host_call();resident_call(); // warm-up: pipelines, workspaces
        Cb.download(res.data(),res.size());
        for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<=(syrk?i:n-1);++j)if(std::memcmp(&res[i*n+j],&host[i*n+j],sizeof(T)))
            throw std::runtime_error("resident result differs at ("+std::to_string(i)+","+std::to_string(j)+")");
        std::vector<double> th,tr,tg,hg;
        for(int r=0;r<repeats;++r){if(r%2){tr.push_back(resident_call());tg.push_back(gpu);th.push_back(host_call());hg.push_back(la.report().gpu_seconds);}else{th.push_back(host_call());hg.push_back(la.report().gpu_seconds);tr.push_back(resident_call());tg.push_back(gpu);}}
        const double h=median(th),s=median(tr),ratio=h/s;
        std::printf("%-5s %5d %5zu %12.3f %12.3f %12.3f %12.3f %8.2f %s\n",syrk?"syrk":"gemm",B,n,1e3*h,1e3*median(hg),1e3*s,1e3*median(tg),ratio,std::abs(ratio-1)<0.15?"noise (<15%)":"");
        std::fflush(stdout);
    }
}
int main(int argc,char** argv){try{
    int bits=256,repeats=9;std::vector<std::size_t> sizes={200,400,600,800,1000};
    for(int i=1;i<argc;++i){std::string a=argv[i];
        if(a=="--bits"&&i+1<argc)bits=std::stoi(argv[++i]);else if(a=="--repeats"&&i+1<argc)repeats=std::stoi(argv[++i]);
        else if(a=="--sizes"&&i+1<argc){sizes.clear();std::stringstream s(argv[++i]);std::string x;while(std::getline(s,x,','))sizes.push_back(std::stoul(x));}
        else throw std::invalid_argument("usage: resident_linalg_limbforge [--bits 224|256|384] [--repeats R] [--sizes n1,n2,...]");}
    Engine e;Linalg la(e);std::printf("%s, repeats %d (interleaved, medians)\n",e.device_name().c_str(),repeats);
    if(bits==224)run<224>(e,la,sizes,repeats);else if(bits==256)run<256>(e,la,sizes,repeats);else if(bits==384)run<384>(e,la,sizes,repeats);
    else throw std::invalid_argument("bits: 224, 256 or 384");
    return 0;}
catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
