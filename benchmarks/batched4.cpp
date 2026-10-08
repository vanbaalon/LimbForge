// Batched 4x4 complex inverse + determinant (plan D5) versus an MPC elimination loop (18-worker pool).
#include "benchmark_support.hpp"
#include <mpc.h>
template<int Bits> void shape(Engine& e,Workers& w,std::size_t M,bool fused,int repeats){
    using C=Complex<Bits/32>;std::mt19937_64 rng(M+Bits);std::vector<C> A(M*16),X(M*16),det(M);std::vector<std::uint32_t> st(M);
    for(auto& z:A)z={reference::random_number<Bits>(rng,4),reference::random_number<Bits>(rng,4)};
    Batched4 s;s.count=M;s.inverse=true;s.determinant=true;s.fused=fused;e.lu4(Bits,s,A.data(),nullptr,X.data(),det.data(),st.data());
    std::vector<double> dev,wall;for(int r=0;r<repeats;++r){auto t=e.lu4(Bits,s,A.data(),nullptr,X.data(),det.data(),st.data());dev.push_back(t.gpu_seconds);wall.push_back(t.wall_seconds);}
    std::size_t n=std::min<std::size_t>(M,8192);std::unique_ptr<mpc_t[]> a(new mpc_t[n*16]);
    for(std::size_t i=0;i<n*16;++i){mpc_init2(a[i],Bits);to_mpfr<Bits>(mpc_realref(a[i]),A[i].re);to_mpfr<Bits>(mpc_imagref(a[i]),A[i].im);}
    auto cpu=[&](std::size_t lo,std::size_t hi){mpc_t m[16],y[4],l,t,d;for(auto& v:m)mpc_init2(v,Bits);for(auto& v:y)mpc_init2(v,Bits);mpc_init2(l,Bits);mpc_init2(t,Bits);mpc_init2(d,Bits);
        for(std::size_t q=lo;q<hi;++q){for(int i=0;i<16;++i)mpc_set(m[i],a[q*16+i],MPC_RNDNN);
            for(int k=0;k<4;++k)for(int r=k+1;r<4;++r){mpc_div(l,m[r*4+k],m[k*4+k],MPC_RNDNN);for(int c=k+1;c<4;++c){mpc_mul(t,l,m[k*4+c],MPC_RNDNN);mpc_sub(m[r*4+c],m[r*4+c],t,MPC_RNDNN);}mpc_set(m[r*4+k],l,MPC_RNDNN);}
            mpc_set(d,m[0],MPC_RNDNN);for(int k=1;k<4;++k)mpc_mul(d,d,m[k*5],MPC_RNDNN);
            for(int j=0;j<4;++j){for(int i=0;i<4;++i)mpc_set_ui(y[i],i==j,MPC_RNDNN);
                for(int i=1;i<4;++i)for(int u=0;u<i;++u){mpc_mul(t,m[i*4+u],y[u],MPC_RNDNN);mpc_sub(y[i],y[i],t,MPC_RNDNN);}
                for(int i=3;i>=0;--i){for(int u=i+1;u<4;++u){mpc_mul(t,m[i*4+u],y[u],MPC_RNDNN);mpc_sub(y[i],y[i],t,MPC_RNDNN);}mpc_div(y[i],y[i],m[i*5],MPC_RNDNN);}}}
        for(auto& v:m)mpc_clear(v);for(auto& v:y)mpc_clear(v);mpc_clear(l);mpc_clear(t);mpc_clear(d);};
    std::vector<double> pool;for(int r=0;r<repeats;++r){auto t0=Clock::now();w.run(n,cpu);pool.push_back(std::chrono::duration<double>(Clock::now()-t0).count()*double(M)/n);}
    std::cout<<Bits<<','<<M<<','<<(fused?"fused":"composed")<<','<<quantile(dev,.5)<<','<<quantile(wall,.5)<<','<<quantile(pool,.5)<<','<<quantile(pool,.5)/quantile(wall,.5)<<std::endl;
    for(std::size_t i=0;i<n*16;++i)mpc_clear(a[i]);
}
int main(){try{Engine e;Workers w(std::max(1u,std::thread::hardware_concurrency()));std::cerr<<e.device_name()<<"; CPU: 18-worker MPC pool, scaled from 8192 matrices\n";
    std::cout<<std::setprecision(5)<<"bits,matrices,mode,gpu_device_s,gpu_wall_s,cpu_pool_s,pool_over_gpu_wall\n";
    for(std::size_t M:{10000ul,100000ul}){shape<224>(e,w,M,false,3);shape<256>(e,w,M,false,3);}shape<256>(e,w,100000,true,3);return 0;}
catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}}
