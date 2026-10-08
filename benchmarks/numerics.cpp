// Round 32 (plan S1/S4): complex polynomial values and jets at BSolver/QSC-like shapes (one shared coefficient set)
// versus CPU loops of the same composed sequence in MPFR (bit-identical contract) and the consumer-style MPC loop
// (mpc_mul + mpc_add, a different rounding), serial and on all workers; then norms of complex vectors.
// CPU columns are measured on min(points, 4096) points and scaled. Usage: numerics_limbforge [--quick]
#include "benchmark_support.hpp"
#include "limbforge/numerics.hpp"
#ifdef LIMBFORGE_TEST_MPC
#include <mpc.h>
#endif
template<class F> double median_time(int repeats,F&& f){std::vector<double> t;for(int r=0;r<repeats;++r){auto s=Clock::now();f();t.push_back(std::chrono::duration<double>(Clock::now()-s).count());}return quantile(t,.5);}
template<int Bits> void poly_shape(Numerics& nm,Workers& w,std::size_t points,std::size_t degree,unsigned order,bool fused,int repeats){
    using C=Complex<Bits/32>;std::mt19937_64 rng(points*7+degree);const std::size_t terms=degree+1;
    std::vector<C> c(terms),x(points),jets(points*(order+1));for(auto& z:c)z={reference::random_number<Bits>(rng,4),reference::random_number<Bits>(rng,4)};
    for(auto& z:x)z={reference::random_number<Bits>(rng,1),reference::random_number<Bits>(rng,1)};
    Polynomial shape{points,terms,points,true,fused};nm.poly_eval_jet(Bits,shape,order,c.data(),x.data(),jets.data());
    std::vector<double> dev,wall;for(int r=0;r<repeats;++r){auto t=nm.poly_eval_jet(Bits,shape,order,c.data(),x.data(),jets.data());dev.push_back(t.gpu_seconds);wall.push_back(t.wall_seconds);}
    // CPU: the same composed sequence with mpfr_t (re = RN(RN(a.re x.re) - RN(a.im x.im)) + b.re etc.), one jet ring per point.
    const std::size_t n=std::min<std::size_t>(points,4096);const double scale=double(points)/n;
    MPArray cre(terms,Bits),cim(terms,Bits),xre(n,Bits),xim(n,Bits);
    for(std::size_t k=0;k<terms;++k){to_mpfr<Bits>(cre[k],c[k].re);to_mpfr<Bits>(cim[k],c[k].im);}
    for(std::size_t i=0;i<n;++i){to_mpfr<Bits>(xre[i],x[i].re);to_mpfr<Bits>(xim[i],x[i].im);}
    double serial=0,pool=0,mpc_pool=0;
    if(!fused){
        auto mpfr_loop=[&](std::size_t lo,std::size_t hi){MPArray r(6,Bits),t(4,Bits);
            for(std::size_t i=lo;i<hi;++i){for(int j=0;j<6;++j)mpfr_set_zero(r[j],1);
                for(std::size_t k=terms;k-->0;)for(int j=int(order);j>=0;--j){mpfr_ptr ar=r[2*j],ai=r[2*j+1];mpfr_srcptr br=j?r[2*j-2]:cre[k],bi=j?r[2*j-1]:cim[k];
                    mpfr_mul(t[0],ar,xre[i],MPFR_RNDN);mpfr_mul(t[1],ai,xim[i],MPFR_RNDN);mpfr_mul(t[2],ar,xim[i],MPFR_RNDN);mpfr_mul(t[3],ai,xre[i],MPFR_RNDN);
                    mpfr_sub(t[0],t[0],t[1],MPFR_RNDN);mpfr_add(t[2],t[2],t[3],MPFR_RNDN);mpfr_add(ar,t[0],br,MPFR_RNDN);mpfr_add(ai,t[2],bi,MPFR_RNDN);}}};
        serial=median_time(repeats,[&]{mpfr_loop(0,n);})*scale;pool=median_time(repeats,[&]{w.run(n,mpfr_loop);})*scale;
#ifdef LIMBFORGE_TEST_MPC
        std::unique_ptr<mpc_t[]> cc(new mpc_t[terms]),xx(new mpc_t[n]);for(std::size_t k=0;k<terms;++k){mpc_init2(cc[k],Bits);to_mpc<Bits>(cc[k],c[k]);}
        for(std::size_t i=0;i<n;++i){mpc_init2(xx[i],Bits);to_mpc<Bits>(xx[i],x[i]);}
        auto mpc_loop=[&](std::size_t lo,std::size_t hi){mpc_t r[3],t;for(auto& z:r)mpc_init2(z,Bits);mpc_init2(t,Bits);
            for(std::size_t i=lo;i<hi;++i){for(auto& z:r)mpc_set_ui(z,0,MPC_RNDNN);
                for(std::size_t k=terms;k-->0;)for(int j=int(order);j>=0;--j){mpc_mul(t,r[j],xx[i],MPC_RNDNN);mpc_add(r[j],t,j?r[j-1]:cc[k],MPC_RNDNN);}}
            for(auto& z:r)mpc_clear(z);mpc_clear(t);};
        mpc_pool=median_time(repeats,[&]{w.run(n,mpc_loop);})*scale;
        for(std::size_t k=0;k<terms;++k)mpc_clear(cc[k]);for(std::size_t i=0;i<n;++i)mpc_clear(xx[i]);
#endif
    }
    double gw=quantile(wall,.5);
    std::cout<<"poly,"<<Bits<<','<<points<<','<<degree<<','<<order<<','<<(fused?"fused":"composed")<<','<<quantile(dev,.5)<<','<<gw<<','<<serial<<','<<pool<<','<<mpc_pool
             <<','<<(serial?serial/gw:0)<<','<<(pool?pool/gw:0)<<std::endl;
}
template<int Bits> void norm_shape(Numerics& nm,Workers& w,std::size_t segments,std::size_t length,int repeats){
    using C=Complex<Bits/32>;std::mt19937_64 rng(segments+length);const std::size_t total=segments*length;
    std::vector<C> x(total);for(auto& z:x)z={reference::random_number<Bits>(rng,40),reference::random_number<Bits>(rng,40)};
    std::vector<Float<Bits>> sc(total),v(segments);for(auto& s:sc)s=reference::random_number<Bits>(rng,40);std::vector<NormInfo> info(segments);Segments seg{segments,length,true};
    const char* names[]={"norm_inf","norm_max","norm2","scaled_residual","summarize_status"};
    for(int op=0;op<5;++op){auto call=[&]{switch(op){case 0:return nm.norm_inf(Bits,seg,x.data(),v.data(),info.data());case 1:return nm.norm_max(Bits,seg,x.data(),v.data(),info.data());
        case 2:return nm.norm2(Bits,seg,x.data(),v.data(),info.data());case 3:return nm.scaled_residual(Bits,seg,x.data(),sc.data(),v.data(),info.data());default:return nm.summarize_status(Bits,seg,x.data(),info.data());}};
        call();std::vector<double> dev,wall;for(int r=0;r<repeats;++r){auto t=call();dev.push_back(t.gpu_seconds);wall.push_back(t.wall_seconds);}
        double serial=0,pool=0;
        if(op==0){ // consumer-style CPU max |z| = max sqrt(re^2 + im^2) with mpfr (mpfr_hypot is correctly rounded: one rounding)
            const std::size_t n=std::min<std::size_t>(total,1<<16);const double scale=double(total)/n;MPArray re(n,Bits),im(n,Bits);
            for(std::size_t i=0;i<n;++i){to_mpfr<Bits>(re[i],x[i].re);to_mpfr<Bits>(im[i],x[i].im);}
            auto loop=[&](std::size_t lo,std::size_t hi){MPArray t(2,Bits);mpfr_set_zero(t[1],1);for(std::size_t i=lo;i<hi;++i){mpfr_hypot(t[0],re[i],im[i],MPFR_RNDN);if(mpfr_cmp(t[0],t[1])>0)mpfr_set(t[1],t[0],MPFR_RNDN);}};
            serial=median_time(repeats,[&]{loop(0,n);})*scale;pool=median_time(repeats,[&]{w.run(n,loop);})*scale;}
        std::cout<<names[op]<<','<<Bits<<','<<segments<<','<<length<<",,,"<<quantile(dev,.5)<<','<<quantile(wall,.5)<<','<<serial<<','<<pool<<",,"
                 <<(serial?serial/quantile(wall,.5):0)<<','<<(pool?pool/quantile(wall,.5):0)<<std::endl;}
}
int main(int argc,char** argv){try{bool quick=argc>1&&std::string(argv[1])=="--quick";Numerics nm;Workers w(std::max(1u,std::thread::hardware_concurrency()));
    std::cerr<<nm.device_name()<<"; CPU pool: "<<w.size()<<" workers; CPU columns scaled from <= 4096 points (polynomials) / 65536 entries (norms)\n";
    std::cout<<std::setprecision(4)<<"op,bits,points_or_segments,degree_or_length,order,mode,gpu_device_s,gpu_wall_s,mpfr_serial_s,mpfr_pool_s,mpc_pool_s,serial_over_gpu_wall,pool_over_gpu_wall\n";
    const int repeats=quick?2:5;
    std::vector<std::size_t> pts=quick?std::vector<std::size_t>{1000,100000}:std::vector<std::size_t>{1000,10000,100000,1000000};
    for(std::size_t points:pts)for(std::size_t degree:{4ul,16ul,64ul})for(unsigned order:{0u,2u}){poly_shape<224>(nm,w,points,degree,order,false,repeats);poly_shape<256>(nm,w,points,degree,order,false,repeats);poly_shape<384>(nm,w,points,degree,order,false,repeats);}
    for(std::size_t points:pts)for(unsigned order:{0u,2u})poly_shape<256>(nm,w,points,32,order,true,repeats);
    norm_shape<256>(nm,w,1,1000000,repeats);norm_shape<256>(nm,w,10000,100,repeats);norm_shape<384>(nm,w,1,1000000,repeats);
    return 0;}
catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}}
