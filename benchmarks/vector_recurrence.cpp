// Vector recurrence (plan D4) at QSC shapes: GPU host-array and device time versus the consumer's
// current CPU arithmetic (MPC mpc_mul/mpc_add: q.v then v += p*s per lane-step), serial and pooled.
// The GPU follows the fused contract of docs/numerics.md and is validated separately in
// tests/test_vector_recurrence.cpp; the CPU loop uses composed MPC roundings (timing only).
#include "benchmark_support.hpp"
#include <mpc.h>
struct MPCArray {
    std::unique_ptr<mpc_t[]> v;std::size_t n;MPCArray(std::size_t count,int bits):v(new mpc_t[count]),n(count){for(std::size_t i=0;i<n;++i)mpc_init2(v[i],bits);}
    ~MPCArray(){for(std::size_t i=0;i<n;++i)mpc_clear(v[i]);} mpc_ptr operator[](std::size_t i){return v[i];}
};
template<int Bits> void shape(Engine& engine,Workers& workers,std::size_t lanes,unsigned steps,bool tangent,int repeats,std::size_t cpu_lanes,bool fused=false){
    using C=Complex<Bits/32>;std::mt19937_64 rng(Bits+lanes+steps);const unsigned per=4;const std::size_t G=lanes/per;
    auto rc=[&](int span,int shift){C z={reference::random_number<Bits>(rng,span),reference::random_number<Bits>(rng,span)};z.re.exponent+=shift;z.im.exponent+=shift;return z;};
    std::vector<C> start(4*lanes),p(std::size_t(steps)*4*G),q(p.size()),out(4*lanes);
    for(auto& z:start)z=rc(3,0);for(auto& z:p)z=rc(1,-1);for(auto& z:q)z=rc(1,-1);
    VectorRecurrence s;s.lanes=lanes;s.steps=steps;s.lanes_per_weight=per;s.fused=fused;
    std::vector<C> base,dp,dq,none(1);
    if(tangent){VectorRecurrence b;b.lanes=lanes/4;b.steps=steps;b.all_steps=true;b.fused=fused;
        // Base chains for lanes/4 trajectories; every 4 tangent lanes share one base and one dp/dq set.
        std::vector<C> bstart(4*b.lanes),bp(std::size_t(steps)*4*b.lanes),bq(bp.size());
        for(std::size_t k=0;k<std::size_t(steps)*4;++k)for(std::size_t i=0;i<b.lanes;++i){bp[k*b.lanes+i]=p[k*G+i];bq[k*b.lanes+i]=q[k*G+i];}
        for(std::size_t j=0;j<4;++j)for(std::size_t i=0;i<b.lanes;++i)bstart[j*b.lanes+i]=start[j*lanes+i];
        base.resize(std::size_t(steps+1)*4*b.lanes);engine.vector_recurrence(Bits,b,bstart.data(),bp.data(),bq.data(),nullptr,base.data());
        s.tangent=true;s.lanes_per_base=4;s.lanes_per_tangent=4;dp.resize(std::size_t(steps)*4*G);dq.resize(dp.size());for(auto& z:dp)z=rc(1,-3);for(auto& z:dq)z=rc(1,-3);}
    auto gpu=[&]{return engine.vector_recurrence(Bits,s,start.data(),p.data(),q.data(),none.data(),out.data(),tangent?base.data():nullptr,tangent?dp.data():nullptr,tangent?dq.data():nullptr);};
    gpu();std::vector<double> device,wall;for(int r=0;r<repeats;++r){auto t=gpu();device.push_back(t.gpu_seconds);wall.push_back(t.wall_seconds);}
    // CPU: consumer arithmetic on cpu_lanes lanes, timed serially and with the worker pool; scaled to all lanes.
    std::size_t n=std::min(cpu_lanes,lanes);const std::size_t Gn=n/per+1;MPCArray v(4*n,Bits),P(std::size_t(steps)*4*Gn,Bits),Q(P.n,Bits);
    for(std::size_t j=0;j<4;++j)for(std::size_t i=0;i<n;++i){to_mpfr<Bits>(mpc_realref(v[j*n+i]),start[j*lanes+i].re);to_mpfr<Bits>(mpc_imagref(v[j*n+i]),start[j*lanes+i].im);}
    for(std::size_t k=0;k<std::size_t(steps)*4;++k)for(std::size_t g=0;g<Gn&&g<G;++g){
        to_mpfr<Bits>(mpc_realref(P[k*Gn+g]),p[k*G+g].re);to_mpfr<Bits>(mpc_imagref(P[k*Gn+g]),p[k*G+g].im);
        to_mpfr<Bits>(mpc_realref(Q[k*Gn+g]),q[k*G+g].re);to_mpfr<Bits>(mpc_imagref(Q[k*Gn+g]),q[k*G+g].im);}
    // Per lane-step: a dot product (4 mul + 4 add) and 4 updates (mul + add); the tangent form adds
    // two more dot products and two more multiply-adds per component, as in tips.md.
    auto cpu=[&](std::size_t lo,std::size_t hi){mpc_t s0,s1,s2,t;mpc_init2(s0,Bits);mpc_init2(s1,Bits);mpc_init2(s2,Bits);mpc_init2(t,Bits);
        for(std::size_t i=lo;i<hi;++i){std::size_t g=i/per;
            for(unsigned k=0;k<steps;++k){std::size_t w=std::size_t(k)*4*Gn+g;
                auto dot=[&](mpc_ptr out,MPCArray& a){mpc_set_ui(out,0,MPC_RNDNN);for(int j=0;j<4;++j){mpc_mul(t,a[w+j*Gn],v[j*n+i],MPC_RNDNN);mpc_add(out,out,t,MPC_RNDNN);}};
                dot(s0,Q);if(tangent){dot(s1,Q);dot(s2,P);}
                for(int j=0;j<4;++j){mpc_mul(t,P[w+j*Gn],s0,MPC_RNDNN);mpc_add(v[j*n+i],v[j*n+i],t,MPC_RNDNN);
                    if(tangent){mpc_mul(t,Q[w+j*Gn],s1,MPC_RNDNN);mpc_add(v[j*n+i],v[j*n+i],t,MPC_RNDNN);mpc_mul(t,P[w+j*Gn],s2,MPC_RNDNN);mpc_add(v[j*n+i],v[j*n+i],t,MPC_RNDNN);}}}}
        mpc_clear(s0);mpc_clear(s1);mpc_clear(s2);mpc_clear(t);};
    std::vector<double> serial,pooled;
    for(int r=0;r<repeats;++r){auto a=Clock::now();cpu(0,std::min<std::size_t>(n,256));serial.push_back(std::chrono::duration<double>(Clock::now()-a).count()*double(lanes)/std::min<std::size_t>(n,256));
        a=Clock::now();workers.run(n,cpu);pooled.push_back(std::chrono::duration<double>(Clock::now()-a).count()*double(lanes)/n);}
    std::cout<<Bits<<','<<lanes<<','<<steps<<','<<(tangent?"tangent":"base")<<(fused?"_fused":"")<<','<<quantile(device,.5)<<','<<quantile(wall,.5)<<','<<quantile(serial,.5)<<','
        <<quantile(pooled,.5)<<','<<quantile(pooled,.5)/quantile(wall,.5)<<','<<quantile(device,.5)/(double(lanes)*steps)*1e9<<std::endl;
}
int main(int argc,char** argv){try{
    int repeats=3;std::size_t cpu_lanes=4096;unsigned workers_n=std::max(1u,std::thread::hardware_concurrency());
    for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--repeats"&&i+1<argc)repeats=std::stoi(argv[++i]);else if(a=="--cpu-lanes"&&i+1<argc)cpu_lanes=std::stoull(argv[++i]);
        else if(a=="--workers"&&i+1<argc)workers_n=unsigned(std::stoul(argv[++i]));else throw std::invalid_argument("usage: vector_recurrence_limbforge [--repeats R] [--cpu-lanes N] [--workers W]");}
    Engine engine;Workers workers(workers_n);std::cerr<<engine.device_name()<<"; workers="<<workers_n<<"; CPU times scaled from "<<cpu_lanes<<" lanes (serial from 256)\n";
    std::cout<<std::setprecision(5)<<"bits,lanes,steps,mode,gpu_device_s,gpu_wall_s,cpu_serial_s,cpu_pool_s,pool_over_gpu_wall,gpu_ns_per_lane_step\n";
    for(auto lanes:{20000ul,70000ul,140000ul})for(unsigned steps:{40u,150u}){shape<224>(engine,workers,lanes,steps,false,repeats,cpu_lanes);shape<256>(engine,workers,lanes,steps,false,repeats,cpu_lanes);}
    shape<256>(engine,workers,70000,150,true,repeats,cpu_lanes);shape<256>(engine,workers,70000,150,false,repeats,cpu_lanes,true);shape<256>(engine,workers,70000,150,true,repeats,cpu_lanes,true);
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
