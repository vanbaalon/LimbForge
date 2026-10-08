// Round 35 (plan D7): element-wise transcendental functions on the GPU (host arrays, correctly rounded with host
// retries) versus MPFR/MPC serial and an 18-worker pool. Medians of `repeats` samples after one warm-up call (the
// first call per width and function compiles its pipeline: reported separately). GPU results are checked bit for bit
// against the MPFR/MPC results of the same run.
// Usage: transcendental_limbforge [--repeats r] [--sizes n ...] [--bits b ...] [--functions f ...] [--serial-cap n]
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
template<int Bits=64,class F> void by_bits(int bits,F&& f){if constexpr(Bits<=1024){if(bits==Bits){f(std::integral_constant<int,Bits>{});return;}by_bits<Bits+32>(bits,f);}}
}
int main(int argc,char** argv){try{
    int repeats=5;std::size_t serial_cap=100000;std::vector<std::size_t> sizes={10000,100000,1000000};std::vector<int> widths={224,256,384};
    std::vector<Function> fs={Function::exp,Function::log,Function::sin,Function::complex_exp,Function::complex_log,Function::complex_powi};
    for(int i=1;i<argc;++i){std::string s=argv[i];
        if(s=="--repeats"&&i+1<argc)repeats=std::stoi(argv[++i]);else if(s=="--serial-cap"&&i+1<argc)serial_cap=std::stoul(argv[++i]);
        else if(s=="--sizes"){sizes.clear();while(i+1<argc&&argv[i+1][0]!='-')sizes.push_back(std::stoul(argv[++i]));}
        else if(s=="--bits"){widths.clear();while(i+1<argc&&argv[i+1][0]!='-')widths.push_back(std::stoi(argv[++i]));}
        else if(s=="--functions"){fs.clear();while(i+1<argc&&argv[i+1][0]!='-'){std::string n=argv[++i];for(int q=0;q<10;++q)if(n==fname(Function(q)))fs.push_back(Function(q));}}
        else{std::cerr<<"unknown argument "<<s<<'\n';return 2;}}
    Transcendentals tr;Workers w(18);
    std::cerr<<tr.device_name()<<"; MPFR "<<mpfr_get_version()<<", MPC "<<mpc_get_version()<<"; pool 18 workers; serial scaled from <= "<<serial_cap<<" values\n";
    std::cout<<std::setprecision(5)<<"bits,function,n,repeats,compile_s,gpu_device_ms,gpu_wall_ms,gpu_wall_min_ms,mpfr_serial_ms,mpfr_pool18_ms,serial_over_gpu_wall,pool_over_gpu_wall,retried_total,mismatches\n";
    for(int b:widths)for(Function f:fs)for(std::size_t n:sizes)by_bits(b,[&](auto tag){constexpr int B=decltype(tag)::value;
        Row r=measure<B>(tr,w,f,n,repeats,serial_cap);
        std::cout<<r.bits<<','<<r.f<<','<<r.n<<','<<r.repeats<<','<<r.compile<<','<<1e3*r.gpu_dev<<','<<1e3*r.gpu_wall<<','<<1e3*r.gpu_wall_min<<','<<1e3*r.serial<<','<<1e3*r.pool<<','
            <<r.serial/r.gpu_wall<<','<<r.pool/r.gpu_wall<<','<<r.retried<<','<<r.mismatches<<std::endl;});
    return 0;}
catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}}
