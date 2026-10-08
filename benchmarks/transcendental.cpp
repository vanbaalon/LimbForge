// Round 35 (plan D7): element-wise transcendental functions on the GPU (host arrays, correctly rounded with host
// retries) versus MPFR/MPC serial and an 18-worker pool. Medians of `repeats` samples after one warm-up call (the
// first call per width and function compiles its pipeline: reported separately). GPU results are checked bit for bit
// against the MPFR/MPC results of the same run.
// Usage: transcendental_limbforge [--repeats r] [--sizes n ...] [--bits b ...] [--functions f ...] [--serial-cap n]
//
// Round 44 (--retries): GPU-side retries. Host-array and resident (one pass per CommandBatch, Submission timing) wall and
// device times without MPFR, for random inputs and inputs with a fraction --hard p of hard cases (short dyadics such as
// exp(2^-32N), cos(2^-16N), log1p(2^-32N), complex exp/log near 1, resolved at the second or third retry rung). The resident
// result is checked bitwise against the host-array call. Compiles against the round-35/39 API as well (host_retried falls
// back to `retried`), so the same source measures the previous library (host retries) for before/after comparisons.
// Each row first keeps the GPU busy with its own work for --gpu-warm seconds (default 0.3) so that it sees warm clocks.
// Usage: transcendental_limbforge --retries [--repeats r] [--sizes n ...] [--bits b ...] [--functions f ...] [--hard p ...]
//        [--gpu-warm s] [--retry-threshold t] (TranscendentalOptions::gpu_retry_threshold of the host-array calls)
#include "benchmark_support.hpp"
#include "limbforge/transcendental.hpp"
#include <mpc.h>
#include <map>
#include <sstream>
namespace {
const char* fname(Function f){static const char* n[]={"exp","expm1","log","log1p","sin","cos","atan2","complex_exp","complex_log","complex_powi"};return n[int(f)];}
template<int Bits> Float<Bits> value(std::mt19937_64& g,int emin,int emax,int sign=0){
    Float<Bits> x=zero<Bits/32>();for(auto& l:x.limb)l=std::uint32_t(g());x.limb[Bits/32-1]|=0x80000000u;
    x.exponent=emin+int(g()%std::uint64_t(emax-emin+1));x.sign=sign?sign:(g()&1?1:-1);return x;
}
struct Row { int bits;std::string f;std::size_t n;double compile,gpu_dev,gpu_wall,gpu_wall_min,serial,pool;std::size_t retried,mismatches;int repeats; };
template<int Bits> Row measure(Transcendentals& tr,Workers& w,Function f,std::size_t n,int repeats,std::size_t serial_cap){
    constexpr int N=Bits/32;std::mt19937_64 g(n*31+Bits+int(f));const bool cx=function_is_complex(f);
    // Inputs: exp/expm1 |x| < 2^6 (no overflow), log/log1p positive, sin/cos |x| < 2^8, complex exp re in [-2^5, 2^5],
    // complex log anywhere, powi |z| ~ 1 with k = 10.
    std::vector<Float<Bits>> a(n),b(n);std::vector<Complex<N>> z(cx?n:0);std::vector<std::int32_t> k(1,10);
    for(std::size_t i=0;i<n;++i){
        if(f==Function::exp)a[i]=value<Bits>(g,-10,5);else if(f==Function::log)a[i]=value<Bits>(g,-60,60,1);else a[i]=value<Bits>(g,-10,7);
        b[i]=value<Bits>(g,-10,7);
        if(f==Function::complex_exp)z[i]={value<Bits>(g,-10,4),value<Bits>(g,-10,7)};
        else if(f==Function::complex_log)z[i]={value<Bits>(g,-60,60),value<Bits>(g,-60,60)};
        else if(f==Function::complex_powi)z[i]={value<Bits>(g,-1,-1),value<Bits>(g,-3,-1)};
    }
    std::vector<Float<Bits>> out(n);std::vector<Complex<N>> zout(cx?n:0);
    auto gpu=[&]{if(f==Function::complex_powi)return tr.powi(Bits,z.data(),k.data(),1,zout.data(),n);
        return cx?tr.run(Bits,f,z.data(),zout.data(),n):tr.run(Bits,f,a.data(),out.data(),n,b.data());};
    double compile=tr.prewarm(Bits,f);
    gpu();
    std::vector<double> dev,wall;std::size_t retried=0;
    for(int r=0;r<repeats;++r){auto t=gpu();dev.push_back(t.gpu_seconds);wall.push_back(t.wall_seconds);retried+=tr.report().retried;}
    // MPFR/MPC on preallocated variables: serial over min(n, serial_cap) values (scaled), pool over all n.
    std::size_t ns=std::min(n,serial_cap);
    const std::size_t nc=cx?n:0;std::unique_ptr<mpc_t[]> mz(new mpc_t[nc]),mo(new mpc_t[nc]);MPArray ma(cx?0:n,Bits),mb(cx?0:n,Bits),mo_r(cx?0:n,Bits);
    for(std::size_t i=0;i<n;++i){if(cx){mpc_init2(mz[i],Bits);mpc_init2(mo[i],Bits);to_mpc<Bits>(mz[i],z[i]);}else{to_mpfr<Bits>(ma[i],a[i]);to_mpfr<Bits>(mb[i],b[i]);}}
    auto cpu=[&](std::size_t lo,std::size_t hi){for(std::size_t i=lo;i<hi;++i)switch(f){
        case Function::exp:mpfr_exp(mo_r[i],ma[i],MPFR_RNDN);break;case Function::log:mpfr_log(mo_r[i],ma[i],MPFR_RNDN);break;
        case Function::sin:mpfr_sin(mo_r[i],ma[i],MPFR_RNDN);break;case Function::cos:mpfr_cos(mo_r[i],ma[i],MPFR_RNDN);break;
        case Function::atan2:mpfr_atan2(mo_r[i],ma[i],mb[i],MPFR_RNDN);break;case Function::expm1:mpfr_expm1(mo_r[i],ma[i],MPFR_RNDN);break;
        case Function::log1p:mpfr_log1p(mo_r[i],ma[i],MPFR_RNDN);break;
        case Function::complex_exp:mpc_exp(mo[i],mz[i],MPC_RNDNN);break;case Function::complex_log:mpc_log(mo[i],mz[i],MPC_RNDNN);break;
        default:mpc_pow_si(mo[i],mz[i],10,MPC_RNDNN);break;}};
    std::vector<double> serial,pool;
    for(int r=0;r<std::max(1,repeats/2);++r){auto t0=Clock::now();cpu(0,ns);serial.push_back(std::chrono::duration<double>(Clock::now()-t0).count()*double(n)/double(ns));}
    for(int r=0;r<repeats;++r){auto t0=Clock::now();w.run(n,cpu);pool.push_back(std::chrono::duration<double>(Clock::now()-t0).count());}
    std::size_t bad=0;
    if(f!=Function::complex_powi)for(std::size_t i=0;i<n;++i){
        if(cx){Complex<N> r=from_mpc<Bits>(mo[i]);if(!reference::equal_complex<Bits>(r,zout[i]))++bad;}
        else if(!reference::equal<Bits>(from_mpfr<Bits>(mo_r[i]),out[i]))++bad;}
    for(std::size_t i=0;i<nc;++i){mpc_clear(mz[i]);mpc_clear(mo[i]);}
    return {Bits,fname(f),n,compile,quantile(dev,.5),quantile(wall,.5),quantile(wall,0),quantile(serial,.5),quantile(pool,.5),retried,bad,repeats};
}
// ---- --retries: host-array and resident timings, random and hard-case-heavy inputs ----
template<class R> auto host_count(const R& r,int)->decltype(r.host_retried()){return r.host_retried();}
template<class O> auto set_threshold(O& o,std::size_t t,int)->decltype(void(o.gpu_retry_threshold=t)){o.gpu_retry_threshold=t;}
template<class O> void set_threshold(O&,std::size_t,long){} // before round 44: no threshold
template<class R> std::size_t host_count(const R& r,long){return r.retried;} // before round 44: every retry ran on the host
// Hard inputs (deterministic cycle): short dyadics whose results lie within ~|x|^3 of a rounding midpoint.
template<int Bits> Float<Bits> hard_real(Function f,std::size_t j){
    constexpr int N=Bits/32;static const int e_exp[]={-32*N,1-32*N,-32*N-1},e_cos[]={-16*N,-16*N-1,-16*N+1};
    Float<Bits> x=zero<N>();x.limb[N-1]=0x80000000u;x.limb[0]=word(j/6%3);x.sign=j%2?-1:1;
    x.exponent=(f==Function::cos||f==Function::sin)?e_cos[j/2%3]:e_exp[j/2%3];
    if(f==Function::log){x.exponent=0;x.sign=1;x.limb[0]=word(1+j%3);} // 1 + small: log close to x - 1
    if(f==Function::log1p)x.sign=1;
    return x;
}
template<int Bits> Complex<Bits/32> hard_complex(Function f,std::size_t j){
    constexpr int N=Bits/32;Float<Bits> one=zero<N>(),tiny=zero<N>();one.limb[N-1]=tiny.limb[N-1]=0x80000000u;one.sign=tiny.sign=1;
    tiny.exponent=f==Function::complex_exp?-32*N:-16*N;tiny.limb[0]=word(j%3);if(j%2)tiny.sign=-1;
    if(f==Function::complex_exp){Float<Bits> t2=tiny;t2.exponent=-16*N;return j%4<2?Complex<N>{tiny,zero<N>()}:Complex<N>{tiny,t2};}
    return {one,tiny}; // log|z| of (1, 2^-16N): log1p of 2^-32N
}
struct RetryRow { double host_dev=0,host_wall=0,res_dev=0,res_wall=0;std::size_t retried=0,host=0,rung[3]={};std::size_t mismatches=0; };
template<int Bits> RetryRow measure_retries(Engine& e,Transcendentals& tr,Function f,std::size_t n,double hard,int repeats){
    constexpr int N=Bits/32;std::mt19937_64 g(n*17+Bits+int(f));const bool cx=function_is_complex(f);
    std::vector<Float<Bits>> a(cx?0:n),b(cx?0:n),out(cx?0:n),res(cx?0:n);std::vector<Complex<N>> z(cx?n:0),zout(cx?n:0),zres(cx?n:0);
    const std::size_t stride=hard>0?std::max<std::size_t>(1,std::size_t(1/hard)):0;std::size_t h=0;
    for(std::size_t i=0;i<n;++i){bool hc=stride&&i%stride==0;
        if(cx){z[i]=f==Function::complex_exp?Complex<N>{value<Bits>(g,-10,4),value<Bits>(g,-10,7)}:Complex<N>{value<Bits>(g,-60,60),value<Bits>(g,-60,60)};if(hc)z[i]=hard_complex<Bits>(f,h++);}
        else{a[i]=f==Function::exp?value<Bits>(g,-10,5):(f==Function::log||f==Function::log1p)?value<Bits>(g,-60,60,1):value<Bits>(g,-10,7);b[i]=value<Bits>(g,-10,7);if(hc)a[i]=hard_real<Bits>(f,h++);}}
    auto host=[&]{return cx?tr.run(Bits,f,z.data(),zout.data(),n):tr.run(Bits,f,a.data(),out.data(),n,b.data());};
    auto A=cx?Buffer<Float<Bits>>():e.make_buffer<Float<Bits>>(n),B=cx?Buffer<Float<Bits>>():e.make_buffer<Float<Bits>>(n),O=cx?Buffer<Float<Bits>>():e.make_buffer<Float<Bits>>(n);
    auto Z=cx?e.make_buffer<Complex<N>>(n):Buffer<Complex<N>>(),ZO=cx?e.make_buffer<Complex<N>>(n):Buffer<Complex<N>>();
    if(cx)Z.upload(z.data(),n);else{A.upload(a.data(),n);B.upload(b.data(),n);}
    auto resident=[&]{auto batch=e.batch();auto t=cx?tr.run(batch,f,Z,ZO):f==Function::atan2?tr.run(batch,f,A,B,O):tr.run(batch,f,A,O);auto s=batch.submit();return s.wait();};
    tr.prewarm(Bits,f);host();resident();keep_gpu_busy([&]{host();resident();}); // warm clocks (docs/gpu-codegen.md section 1)
    RetryRow r;std::vector<double> hd,hw,rd,rw;
    for(int k=0;k<repeats;++k){auto t=host();hd.push_back(t.gpu_seconds);hw.push_back(t.wall_seconds);auto u=resident();rd.push_back(u.gpu_seconds);rw.push_back(u.wall_seconds);}
    const auto& rep=tr.report();r.retried=rep.retried;r.host=host_count(rep,0);for(int k=0;k<3;++k)r.rung[k]=rep.resolved[k];
    if(cx){ZO.download(zres.data(),n);for(std::size_t i=0;i<n;++i)if(!reference::equal_complex<Bits>(zres[i],zout[i]))++r.mismatches;}
    else{O.download(res.data(),n);for(std::size_t i=0;i<n;++i)if(!reference::equal<Bits>(res[i],out[i]))++r.mismatches;}
    r.host_dev=quantile(hd,.5);r.host_wall=quantile(hw,.5);r.res_dev=quantile(rd,.5);r.res_wall=quantile(rw,.5);return r;
}
template<int Bits=64,class F> void by_bits(int bits,F&& f){if constexpr(Bits<=1024){if(bits==Bits){f(std::integral_constant<int,Bits>{});return;}by_bits<Bits+32>(bits,f);}}
}
int main(int argc,char** argv){try{
    int repeats=5;std::size_t serial_cap=100000;std::vector<std::size_t> sizes={10000,100000,1000000};std::vector<int> widths={224,256,384};
    std::vector<Function> fs={Function::exp,Function::log,Function::sin,Function::complex_exp,Function::complex_log,Function::complex_powi};
    bool retries=false;std::vector<double> hard={0,0.01};long threshold=-1;
    for(int i=1;i<argc;++i){std::string s=argv[i];
        if(s=="--retries")retries=true;else if(s=="--retry-threshold"&&i+1<argc)threshold=std::stol(argv[++i]);else if(s=="--gpu-warm"&&i+1<argc)gpu_warm_seconds=std::stod(argv[++i]);else if(s=="--hard"){hard.clear();while(i+1<argc&&argv[i+1][0]!='-')hard.push_back(std::stod(argv[++i]));}
        else if(s=="--repeats"&&i+1<argc)repeats=std::stoi(argv[++i]);else if(s=="--serial-cap"&&i+1<argc)serial_cap=std::stoul(argv[++i]);
        else if(s=="--sizes"){sizes.clear();while(i+1<argc&&argv[i+1][0]!='-')sizes.push_back(std::stoul(argv[++i]));}
        else if(s=="--bits"){widths.clear();while(i+1<argc&&argv[i+1][0]!='-')widths.push_back(std::stoi(argv[++i]));}
        else if(s=="--functions"){fs.clear();while(i+1<argc&&argv[i+1][0]!='-'){std::string n=argv[++i];for(int q=0;q<10;++q)if(n==fname(Function(q)))fs.push_back(Function(q));}}
        else{std::cerr<<"unknown argument "<<s<<'\n';return 2;}}
    if(retries){if(gpu_warm_seconds<=0)gpu_warm_seconds=0.3;
        Engine e;TranscendentalOptions o;if(threshold>=0)set_threshold(o,std::size_t(threshold),0);Transcendentals tr(e,o);
        std::cerr<<tr.device_name()<<"; host-array and resident passes, medians of "<<repeats<<"\n";
        std::cout<<std::setprecision(5)<<"bits,function,n,hard_fraction,repeats,host_device_ms,host_wall_ms,resident_device_ms,resident_wall_ms,retried,rung1,rung2,rung3,host_retried,resident_mismatches\n";
        for(int b:widths)for(Function f:fs)for(double hf:hard)for(std::size_t n:sizes)by_bits(b,[&](auto tag){constexpr int B=decltype(tag)::value;if(f==Function::complex_powi)return;
            RetryRow r=measure_retries<B>(e,tr,f,n,hf,repeats);
            std::cout<<B<<','<<fname(f)<<','<<n<<','<<hf<<','<<repeats<<','<<1e3*r.host_dev<<','<<1e3*r.host_wall<<','<<1e3*r.res_dev<<','<<1e3*r.res_wall<<','
                <<r.retried<<','<<r.rung[0]<<','<<r.rung[1]<<','<<r.rung[2]<<','<<r.host<<','<<r.mismatches<<std::endl;});
        return 0;
    }
    Transcendentals tr;Workers w(18);
    std::cerr<<tr.device_name()<<"; MPFR "<<mpfr_get_version()<<", MPC "<<mpc_get_version()<<"; pool 18 workers; serial scaled from <= "<<serial_cap<<" values\n";
    std::cout<<std::setprecision(5)<<"bits,function,n,repeats,compile_s,gpu_device_ms,gpu_wall_ms,gpu_wall_min_ms,mpfr_serial_ms,mpfr_pool18_ms,serial_over_gpu_wall,pool_over_gpu_wall,retried_total,mismatches\n";
    for(int b:widths)for(Function f:fs)for(std::size_t n:sizes)by_bits(b,[&](auto tag){constexpr int B=decltype(tag)::value;
        Row r=measure<B>(tr,w,f,n,repeats,serial_cap);
        std::cout<<r.bits<<','<<r.f<<','<<r.n<<','<<r.repeats<<','<<r.compile<<','<<1e3*r.gpu_dev<<','<<1e3*r.gpu_wall<<','<<1e3*r.gpu_wall_min<<','<<1e3*r.serial<<','<<1e3*r.pool<<','
            <<r.serial/r.gpu_wall<<','<<r.pool/r.gpu_wall<<','<<r.retried<<','<<r.mismatches<<std::endl;});
    return 0;}
catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}}
