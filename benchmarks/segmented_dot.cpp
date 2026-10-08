// Segmented complex dot products (plan D3) at QSC P-series shapes versus the consumer's MPC loop.
#include "benchmark_support.hpp"
#include <mpc.h>
template<int Bits> void shape(Engine& e,Workers& w,std::size_t S,std::size_t K,bool shared,int repeats){
    using C=Complex<Bits/32>;std::mt19937_64 rng(S+K);auto rc=[&]{return C{reference::random_number<Bits>(rng,8),reference::random_number<Bits>(rng,8)};};
    std::vector<C> a(S*K),b(shared?K:S*K),out(S);for(auto& z:a)z=rc();for(auto& z:b)z=rc();SegmentedDot s{S,K,shared};
    e.segmented_dot(Bits,true,s,a.data(),b.data(),out.data());std::vector<double> dev,wall;
    for(int r=0;r<repeats;++r){auto t=e.segmented_dot(Bits,true,s,a.data(),b.data(),out.data());dev.push_back(t.gpu_seconds);wall.push_back(t.wall_seconds);}
    std::size_t n=std::min<std::size_t>(S,4096);std::unique_ptr<mpc_t[]> A(new mpc_t[n*K]),B(new mpc_t[n*K]),O(new mpc_t[n]);
    for(std::size_t i=0;i<n*K;++i){mpc_init2(A[i],Bits);mpc_init2(B[i],Bits);to_mpfr<Bits>(mpc_realref(A[i]),a[i].re);to_mpfr<Bits>(mpc_imagref(A[i]),a[i].im);
        auto& y=b[shared?i%K:i];to_mpfr<Bits>(mpc_realref(B[i]),y.re);to_mpfr<Bits>(mpc_imagref(B[i]),y.im);}
    for(std::size_t i=0;i<n;++i)mpc_init2(O[i],Bits);
    auto cpu=[&](std::size_t lo,std::size_t hi){mpc_t t;mpc_init2(t,Bits);for(std::size_t i=lo;i<hi;++i){mpc_set_ui(O[i],0,MPC_RNDNN);
        for(std::size_t k=0;k<K;++k){mpc_mul(t,A[i*K+k],B[i*K+k],MPC_RNDNN);mpc_add(O[i],O[i],t,MPC_RNDNN);}}mpc_clear(t);};
    std::vector<double> pool;for(int r=0;r<repeats;++r){auto st=Clock::now();w.run(n,cpu);pool.push_back(std::chrono::duration<double>(Clock::now()-st).count()*double(S)/n);}
    std::cout<<Bits<<','<<S<<','<<K<<','<<(shared?"shared":"separate")<<','<<quantile(dev,.5)<<','<<quantile(wall,.5)<<','<<quantile(pool,.5)<<','<<quantile(pool,.5)/quantile(wall,.5)<<std::endl;
    for(std::size_t i=0;i<n*K;++i){mpc_clear(A[i]);mpc_clear(B[i]);}for(std::size_t i=0;i<n;++i)mpc_clear(O[i]);
}
int main(){try{Engine e;Workers w(std::max(1u,std::thread::hardware_concurrency()));std::cerr<<e.device_name()<<"; CPU: 18-worker MPC pool (mpc_mul+mpc_add), scaled from 4096 segments\n";
    std::cout<<std::setprecision(5)<<"bits,segments,length,table,gpu_device_s,gpu_wall_s,cpu_pool_s,pool_over_gpu_wall\n";
    for(std::size_t K:{24ul,32ul,62ul})for(bool sh:{false,true})shape<256>(e,w,56000,K,sh,3);shape<224>(e,w,56000,32,false,3);return 0;}
catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}}
