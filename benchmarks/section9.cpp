// Offline calibration for the new complex GEMM/power-moment paths. No timing in dispatch policy.
#include "limbforge/batched_linalg.hpp"
#include "limbforge/dispatch_policy.hpp"
#include "benchmark_support.hpp"
#include <random>
#include <cstdlib>
#include <cmath>
struct Options {
    int bits=352;std::size_t count=2,m=4,n=4,k=4;unsigned workers=18,repeats=3;
    std::string operation="gemm";bool resident=false,compare_power=false;double warm=0;
};
// The four products are separately rounded, exactly as the composed GPU complex multiply.
void product(mpfr_ptr r,mpfr_ptr i,mpfr_srcptr ar,mpfr_srcptr ai,mpfr_srcptr br,mpfr_srcptr bi,MPArray& t){
    mpfr_mul(t[4],ar,br,MPFR_RNDN);mpfr_mul(t[5],ai,bi,MPFR_RNDN);
    mpfr_mul(t[6],ar,bi,MPFR_RNDN);mpfr_mul(t[7],ai,br,MPFR_RNDN);
    mpfr_sub(r,t[4],t[5],MPFR_RNDN);mpfr_add(i,t[6],t[7],MPFR_RNDN);
}
std::size_t checked(std::size_t a,std::size_t b){if(b&&a>std::size_t(-1)/b)throw std::invalid_argument("shape overflow");return a*b;}
template<int Bits> void run(const Options& o){
    using C=Complex<Bits/32>;Engine e;BatchedLinalg la(e);Workers pool(o.workers);
    auto left=checked(checked(o.count,o.m),o.k),right=checked(checked(o.count,o.k),o.n),outputs=checked(checked(o.count,o.m),o.n),ek=checked(o.count,o.k);
    MPArray a(checked(left,2),Bits),b(checked(right,2),Bits),result(checked(outputs,2),Bits),ep(checked(ek,2),Bits),yy(checked(ek,2),Bits);
    std::vector<C> A(o.operation=="gemm"?left:0),B(right),E(ek),Y(ek),out(outputs);std::mt19937_64 rng(9+Bits);
    auto fill=[&](std::vector<C>& x,MPArray& dest){for(std::size_t j=0;j<x.size();++j){x[j]={reference::random_number<Bits>(rng,2),reference::random_number<Bits>(rng,2)};
        to_mpfr<Bits>(dest[2*j],x[j].re);to_mpfr<Bits>(dest[2*j+1],x[j].im);}};
    fill(A,a);fill(B,b);fill(E,ep);fill(Y,yy);
    std::vector<std::unique_ptr<MPArray>> scratch;for(unsigned w=0;w<o.workers;++w)scratch.emplace_back(std::make_unique<MPArray>(8,Bits));
    auto cpu=[&]{
        if(o.operation=="power")pool.run(o.workers,[&](std::size_t first,std::size_t last){for(auto w=first;w<last;++w){auto& t=*scratch[w];
            for(auto idx=ek*w/o.workers;idx<ek*(w+1)/o.workers;++idx){auto batch=idx/o.k,k=idx%o.k;mpfr_set_ui(t[0],1,MPFR_RNDN);mpfr_set_zero(t[1],1);
                for(std::size_t row=0;row<o.m;++row){auto dst=(batch*o.m+row)*o.k+k;product(a[2*dst],a[2*dst+1],ep[2*idx],ep[2*idx+1],t[0],t[1],t);
                    if(row+1<o.m)product(t[0],t[1],t[0],t[1],yy[2*idx],yy[2*idx+1],t);}}}});
        pool.run(o.workers,[&](std::size_t first,std::size_t last){for(auto w=first;w<last;++w){auto& t=*scratch[w];
            for(auto idx=outputs*w/o.workers;idx<outputs*(w+1)/o.workers;++idx){auto batch=idx/(o.m*o.n),row=(idx/o.n)%o.m,col=idx%o.n;
                mpfr_set_zero(result[2*idx],1);mpfr_set_zero(result[2*idx+1],1);
                for(std::size_t k=0;k<o.k;++k){auto ai=(batch*o.m+row)*o.k+k,bi=(batch*o.k+k)*o.n+col;
                    product(t[2],t[3],a[2*ai],a[2*ai+1],b[2*bi],b[2*bi+1],t);
                    mpfr_add(result[2*idx],result[2*idx],t[2],MPFR_RNDN);mpfr_add(result[2*idx+1],result[2*idx+1],t[3],MPFR_RNDN);}}}});
    };
    StridedGemm shape{o.count,o.m,o.n,o.k,o.m*o.k,o.k*o.n,o.m*o.n};
    PowerMoments powers{o.count,unsigned(o.k),unsigned(o.m-1),unsigned(o.n),0};
    auto put=[&](const std::vector<C>& v){auto buffer=e.make_buffer<C>(v.size());buffer.upload(v.data(),v.size());return buffer;};
    Buffer<C> ab,bb,eb,yb,cb;if(o.resident){ab=put(A);bb=put(B);eb=put(E);yb=put(Y);cb=e.make_buffer<C>(outputs);}
    PowerStorage storage=PowerStorage::full_table;
    auto gpu=[&]{if(!o.resident)return o.operation=="gemm"?la.gemm(Bits,true,shape,A.data(),B.data(),out.data()):la.power_moments(Bits,true,powers,E.data(),Y.data(),B.data(),out.data(),storage);
        auto batch=e.batch();if(o.operation=="gemm")la.gemm(batch,shape,ab,bb,cb);else la.power_moments(batch,powers,eb,yb,bb,cb,storage);return batch.submit().wait();};
    auto verify=[&]{if(o.resident)cb.download(out.data(),outputs);for(std::size_t i=0;i<outputs;++i){C want{from_mpfr<Bits>(result[2*i]),from_mpfr<Bits>(result[2*i+1])};
        if(!reference::equal_complex<Bits>(out[i],want))throw std::runtime_error("CPU/GPU mismatch at "+std::to_string(i));}};
    std::vector<std::string> paths=o.compare_power?std::vector<std::string>{"table","panel"}:std::vector<std::string>{"table"};
    auto select=[&](std::size_t p){storage=paths[p]=="panel"?PowerStorage::compact:PowerStorage::full_table;};
    std::vector<double> ct;std::vector<std::vector<double>> gt(paths.size()),device(paths.size());cpu();
    for(std::size_t p=0;p<paths.size();++p){select(p);gpu();verify();} // compilation, workers and page faults excluded
    gpu_warm_seconds=o.warm;
    for(unsigned r=0;r<o.repeats;++r){auto begin=Clock::now();cpu();ct.push_back(std::chrono::duration<double>(Clock::now()-begin).count());
        for(std::size_t j=0;j<paths.size();++j){auto p=(j+r)%paths.size();select(p);keep_gpu_busy(gpu);auto t=gpu();gt[p].push_back(t.wall_seconds);device[p].push_back(t.gpu_seconds);verify();}}
    std::cerr<<"Device: "<<e.device_name()<<"; composed complex arithmetic; n0=0; checked initial and every timed sample ("<<outputs<<" outputs); "<<(o.resident?"resident, transfers outside timing":"host-array, transfers included")<<"; GPU warm-up "<<o.warm<<" s\n";
    std::cout<<"operation,bits,count,m,n,k,cpu_workers,fused,repeats,cpu_wall,gpu_wall,gpu_device,recommendation,profile,resident,power_path,clock_mode,power_scratch_bytes\n";
    for(std::size_t p=0;p<paths.size();++p){DispatchKey key{o.operation=="gemm"?"gemm_complex":"power_moments_complex",Bits,o.count,o.m,o.n,o.k,o.workers,false};
        key.profile=e.device_name()+" / section9-panel / " LIMBFORGE_VERSION_STRING " / "+LIMBFORGE_CALIBRATION_BUILD_TYPE+" / "+paths[p]+(o.warm>0?" / warm":" / cpu-interleaved");key.resident=o.resident;
        DispatchSample sample{quantile(ct,.5),quantile(gt[p],.5)};BreakEvenTable table;table.record(key,sample);auto rec=table.recommend(key);
        auto scratch_bytes=o.operation=="gemm"?0:checked(checked(ek,paths[p]=="panel"?1+std::min<std::size_t>(8,o.m):o.m),sizeof(C));
        std::cout<<key.operation<<','<<Bits<<','<<o.count<<','<<o.m<<','<<o.n<<','<<o.k<<','<<o.workers<<",0,"<<o.repeats<<','<<std::setprecision(9)<<sample.cpu_wall<<','<<sample.gpu_wall<<','<<quantile(device[p],.5)<<','<<(rec==BackendRecommendation::gpu?"gpu":rec==BackendRecommendation::cpu?"cpu":"unknown")<<','<<key.profile<<','<<o.resident<<','<<paths[p]<<','<<(o.warm>0?"warm":"cpu-interleaved")<<','<<scratch_bytes<<'\n';}

}
template<int Bits=64> void width(const Options& o){if(o.bits==Bits)run<Bits>(o);else if constexpr(Bits<1024)width<Bits+32>(o);}
int main(int argc,char** argv){try{Options o;
    for(int i=1;i<argc;++i){std::string arg=argv[i];if(arg=="--help"){std::cout<<"section9_limbforge --operation gemm|power --bits B --count C --m M --n N --k K --workers W --repeats R [--resident] [--gpu-warm seconds] [--compare-power]\n";return 0;}
        if(arg=="--resident"){o.resident=true;continue;}if(arg=="--compare-power"){o.compare_power=true;continue;}
        if(i+1==argc)throw std::invalid_argument("missing option value");std::string value=argv[++i];if(arg=="--operation")o.operation=value;
        else if(arg=="--gpu-warm"){std::size_t end;o.warm=std::stod(value,&end);if(end!=value.size()||!std::isfinite(o.warm)||o.warm<0||o.warm>10)throw std::invalid_argument("invalid warm-up seconds");}
        else {if(value.empty()||value[0]=='-')throw std::invalid_argument("expected positive integer");std::size_t end=0;auto v=std::stoull(value,&end);if(end!=value.size()||v>UINT32_MAX||!v)throw std::invalid_argument("invalid integer");
            if(arg=="--bits")o.bits=int(v);else if(arg=="--count")o.count=v;else if(arg=="--m")o.m=v;else if(arg=="--n")o.n=v;else if(arg=="--k")o.k=v;else if(arg=="--workers")o.workers=unsigned(v);else if(arg=="--repeats")o.repeats=unsigned(v);else throw std::invalid_argument("unknown option");}}
    if(o.bits<64||o.bits>1024||o.bits%32||(o.operation!="gemm"&&o.operation!="power")||o.workers>256||o.repeats>1000)throw std::invalid_argument("invalid calibration options");
    if(o.compare_power&&o.operation!="power")throw std::invalid_argument("--compare-power requires --operation power");
    checked(checked(o.count,o.m),o.n);width(o);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
