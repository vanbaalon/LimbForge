// ns/value of the MPFR/MPC bridge: mpz reference path (detail::*_slow) vs direct limb copy, scalar and array,
// single and multithreaded. Every new result is checked against the old one. Usage: bridge_limbforge [count] [samples]
#include "reference.hpp"
#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
using namespace limbforge;
using Clock=std::chrono::steady_clock;
static std::size_t count=65536;static int samples=9;
struct Time { double median,min; };
static Time median_ns(const std::function<void()>& run){
    run();std::vector<double> t;
    for(int s=0;s<samples;++s){auto a=Clock::now();run();t.push_back(std::chrono::duration<double,std::nano>(Clock::now()-a).count()/double(count));}
    std::sort(t.begin(),t.end());return {t[t.size()/2],t[0]};
}
struct MPArray {
    std::unique_ptr<mpfr_t[]> a;std::size_t n;
    MPArray(std::size_t n,mpfr_prec_t p):a(new mpfr_t[n]),n(n){for(std::size_t i=0;i<n;++i)mpfr_init2(a[i],p);}
    ~MPArray(){for(std::size_t i=0;i<n;++i)mpfr_clear(a[i]);}
};
static void row(int bits,const char* format,const char* direction,mpfr_prec_t prec,const char* method,unsigned threads,Time ns,Time old){
    std::cout<<bits<<','<<format<<','<<direction<<','<<prec<<','<<method<<','<<threads<<','<<ns.median<<','<<ns.min<<','<<old.median/ns.median<<std::endl;
}
template<int Bits> static void run(gmp_randstate_t rand,std::mt19937_64& rng){
    unsigned hw=std::max(1u,std::thread::hardware_concurrency());
    auto auto_threads=[&](std::size_t grain){return count<2*grain?1u:unsigned(std::min<std::size_t>(hw,count/grain));};
    constexpr std::size_t real_grain=(std::size_t(1)<<22)/Bits,complex_grain=real_grain/2;
    std::vector<Float<Bits>> values(count),out(count),check(count);
    for(auto& v:values)v=reference::random_number<Bits>(rng);
    for(mpfr_prec_t prec:{mpfr_prec_t(Bits),mpfr_prec_t(Bits+64),mpfr_prec_t(Bits-32)}){
        MPArray in(count,prec),dst(count,prec),ref(count,prec);
        for(std::size_t i=0;i<count;++i){mpfr_urandomb(in.a[i],rand);mpfr_mul_2si(in.a[i],in.a[i],long(rng()%2001)-1000,MPFR_RNDN);if(rng()&1)mpfr_neg(in.a[i],in.a[i],MPFR_RNDN);}
        auto from_old=[&](std::size_t b,std::size_t e,detail::Range){for(std::size_t i=b;i<e;++i)check[i]=detail::from_mpfr_slow<Bits>(in.a[i]);};
        auto to_old=[&](std::size_t b,std::size_t e,detail::Range){for(std::size_t i=b;i<e;++i)detail::to_mpfr_slow<Bits>(ref.a[i],values[i]);};
        auto verify=[&]{for(std::size_t i=0;i<count;++i)if(!reference::equal<Bits>(out[i],check[i])||!mpfr_equal_p(dst.a[i],ref.a[i]))throw std::runtime_error("bridge mismatch at "+std::to_string(Bits));};
        Time fo=median_ns([&]{from_old(0,count,{});}),to=median_ns([&]{to_old(0,count,{});});
        row(Bits,"real","from",prec,"old",1,fo,fo);
        row(Bits,"real","from",prec,"new",1,median_ns([&]{for(std::size_t i=0;i<count;++i)out[i]=from_mpfr<Bits>(in.a[i]);}),fo);
        row(Bits,"real","from",prec,"new_array",1,median_ns([&]{from_mpfr_array<Bits>(in.a.get(),out.data(),count,1);}),fo);
        row(Bits,"real","from",prec,"old_threads",auto_threads(real_grain),median_ns([&]{detail::parallel_chunks(count,0,real_grain,from_old);}),fo);
        row(Bits,"real","from",prec,"new_array_threads",auto_threads(real_grain),median_ns([&]{from_mpfr_array<Bits>(in.a.get(),out.data(),count,0);}),fo);
        row(Bits,"real","to",prec,"old",1,to,to);
        row(Bits,"real","to",prec,"new",1,median_ns([&]{for(std::size_t i=0;i<count;++i)to_mpfr<Bits>(dst.a[i],values[i]);}),to);
        row(Bits,"real","to",prec,"new_array",1,median_ns([&]{to_mpfr_array<Bits>(dst.a.get(),values.data(),count,1);}),to);
        row(Bits,"real","to",prec,"old_threads",auto_threads(real_grain),median_ns([&]{detail::parallel_chunks(count,0,real_grain,to_old);}),to);
        row(Bits,"real","to",prec,"new_array_threads",auto_threads(real_grain),median_ns([&]{to_mpfr_array<Bits>(dst.a.get(),values.data(),count,0);}),to);
        verify();
    }
#if defined(LIMBFORGE_HAS_MPC)&&defined(LIMBFORGE_TEST_MPC)
    using C=Complex<Bits/32>;std::unique_ptr<mpc_t[]> in(new mpc_t[count]),dst(new mpc_t[count]);std::vector<C> cv(count),cout_(count),cref(count);
    for(std::size_t i=0;i<count;++i){mpc_init2(in[i],Bits);mpc_init2(dst[i],Bits);mpc_urandom(in[i],rand);cv[i]={values[i],reference::random_number<Bits>(rng)};}
    auto from_old=[&](std::size_t b,std::size_t e,detail::Range){for(std::size_t i=b;i<e;++i)cref[i]={detail::from_mpfr_slow<Bits>(mpc_realref(in[i])),detail::from_mpfr_slow<Bits>(mpc_imagref(in[i]))};};
    auto to_old=[&](std::size_t b,std::size_t e,detail::Range){for(std::size_t i=b;i<e;++i){detail::to_mpfr_slow<Bits>(mpc_realref(dst[i]),cv[i].re);detail::to_mpfr_slow<Bits>(mpc_imagref(dst[i]),cv[i].im);}};
    Time fo=median_ns([&]{from_old(0,count,{});}),to=median_ns([&]{to_old(0,count,{});});
    row(Bits,"complex","from",Bits,"old",1,fo,fo);
    row(Bits,"complex","from",Bits,"new_array",1,median_ns([&]{from_mpc_array<Bits>(in.get(),cout_.data(),count,1);}),fo);
    row(Bits,"complex","from",Bits,"old_threads",auto_threads(complex_grain),median_ns([&]{detail::parallel_chunks(count,0,complex_grain,from_old);}),fo);
    row(Bits,"complex","from",Bits,"new_array_threads",auto_threads(complex_grain),median_ns([&]{from_mpc_array<Bits>(in.get(),cout_.data(),count,0);}),fo);
    for(std::size_t i=0;i<count;++i)if(!reference::equal_complex<Bits>(cout_[i],cref[i]))throw std::runtime_error("mpc mismatch");
    row(Bits,"complex","to",Bits,"old",1,to,to);
    row(Bits,"complex","to",Bits,"new_array",1,median_ns([&]{to_mpc_array<Bits>(dst.get(),cv.data(),count,1);}),to);
    row(Bits,"complex","to",Bits,"old_threads",auto_threads(complex_grain),median_ns([&]{detail::parallel_chunks(count,0,complex_grain,to_old);}),to);
    row(Bits,"complex","to",Bits,"new_array_threads",auto_threads(complex_grain),median_ns([&]{to_mpc_array<Bits>(dst.get(),cv.data(),count,0);}),to);
    for(std::size_t i=0;i<count;++i){if(!reference::equal_complex<Bits>(from_mpc<Bits>(dst[i]),cv[i]))throw std::runtime_error("mpc round trip");mpc_clear(in[i]);mpc_clear(dst[i]);}
#endif
}
int main(int argc,char** argv){
    try{
        if(argc>1)count=std::stoul(argv[1]);if(argc>2)samples=std::stoi(argv[2]);
        gmp_randstate_t rand;gmp_randinit_default(rand);gmp_randseed_ui(rand,20261007);std::mt19937_64 rng(20261007);
        std::cout<<"bits,format,direction,mpfr_prec,method,threads,median_ns_per_value,min_ns_per_value,median_speedup_vs_old"<<std::endl;
        run<224>(rand,rng);run<256>(rand,rng);run<384>(rand,rng);run<1024>(rand,rng);
        gmp_randclear(rand);return 0;
    }catch(const std::exception& e){std::cerr<<"FAILED: "<<e.what()<<std::endl;return 1;}
}
