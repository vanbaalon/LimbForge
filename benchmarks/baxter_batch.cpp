// An explicit experiment using the existing solver as an independent reference.
// Seeds / coefficient preparation remain on the CPU; only propagation is GPU.
#include "limbforge/engine.hpp"
#include "limbforge/mpfr_bridge.hpp"
#include "baxter.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
namespace {
constexpr int bits=384;using GPUC=limbforge::Complex<bits/32>;
GPUC pack(const bs4::C& x){return {limbforge::from_mpfr<bits>(x.re.backend().data()),limbforge::from_mpfr<bits>(x.im.backend().data())};}
bs4::C unpack(const GPUC& x){bs4::C z;limbforge::to_mpfr<bits>(z.re.backend().data(),x.re);limbforge::to_mpfr<bits>(z.im.backend().data(),x.im);return z;}
double elapsed(std::chrono::steady_clock::time_point start){return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
}
int main(int argc,char** argv) {
    try {
        int points=argc>1?std::stoi(argv[1]):16,steps=argc>2?std::stoi(argv[2]):600;
        if(points<1||points>1024||steps<4||steps>10000)throw std::invalid_argument("points 1..1024, steps 4..10000");
        bs4::Real::default_precision(110);
        auto model=bs4::Model::lengthOne(bs4::Real(5)/6,bs4::Real(1)/6,bs4::C(bs4::Real("0.95")),bs4::C(bs4::Real("0.01")),0);
        bs4::Solver solver(model,60,steps);limbforge::Engine engine;
        int weight_count=points*2,count=weight_count*4;
        std::vector<GPUC> seeds(4*count),weights(std::size_t(steps)*4*weight_count),result(count);
        std::vector<bs4::C> us(points);
        for(int point=0;point<points;++point)us[point]=bs4::C(bs4::Real("0.17")+bs4::Real(point)/1000,bs4::Real("0.07"));
        auto start=std::chrono::steady_clock::now();
        for(int w=0;w<weight_count;++w){bool down=(w%2)==0;int direction=down?1:-1;auto u=us[w/2];
            // Oldest-to-newest ring in propagation order: reverse the solver's seed array.
            for(int j=0;j<4;++j){auto q=solver.series.largeAll(u+bs4::I(direction*(steps+3-j)));
                for(int a=0;a<4;++a)seeds[j*count+w*4+a]=pack(q[a]);}
            for(int t=0;t<steps;++t){auto v=u+bs4::I(direction*(steps+1-t));
                std::array<bs4::C,5>A;for(int k=0;k<5;++k)A[k]=bs4::eval(model.A[k],v);
                auto inv=bs4::reciprocal(A[down?0:4]);
                for(int j=0;j<4;++j)weights[(std::size_t(t)*4+j)*weight_count+w]=pack(-A[down?4-j:j]*inv);
            }
        }
        double preparation=elapsed(start);
        engine.recurrence(bits,seeds.data(),weights.data(),result.data(),count,steps,4); // compile / warm up
        auto timing=engine.recurrence(bits,seeds.data(),weights.data(),result.data(),count,steps,4);
        bs4::Real max_error=0;start=std::chrono::steady_clock::now();
        std::vector<bs4::Q> expected(weight_count);
        for(int w=0;w<weight_count;++w)expected[w]=solver.q(us[w/2],(w%2)==0);
        double cpu=elapsed(start);
        for(int w=0;w<weight_count;++w)for(int a=0;a<4;++a){auto z=unpack(result[w*4+a]);
            bs4::Real error=bs4::abs(z-expected[w][a])/(1+bs4::abs(expected[w][a]));max_error=std::max(max_error,error);}
        std::cout<<std::setprecision(10)<<"device="<<engine.device_name()<<"\npoints="<<points<<"\nsteps="<<steps
            <<"\nbits="<<bits<<"\ncpu_reference_decimal_digits=110\npreparation_s="<<preparation
            <<"\ngpu_s="<<timing.gpu_seconds<<"\ngpu_wall_s="<<timing.wall_seconds
            <<"\npreparation_plus_gpu_wall_s="<<preparation+timing.wall_seconds<<"\nmpfr_solver_propagation_s="<<cpu
            <<"\nmax_error_over_1_plus_abs_q="<<max_error<<"\n";
        if(max_error>bs4::tiny(80))throw std::runtime_error("Baxter comparison failed at 80-digit scaled error threshold");
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
