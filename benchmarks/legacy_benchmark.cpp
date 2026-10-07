#include "reference.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
using namespace limbforge;
double median(std::vector<double> values){std::sort(values.begin(),values.end());return values[values.size()/2];}
template<int Bits> void benchmark(Engine& engine,int count) {
    using F=Float<Bits>;std::mt19937_64 rng(Bits);
    std::vector<F>a(count),b(count),out(count);
    for(int i=0;i<count;++i){a[i]=reference::random_number<Bits>(rng,5);b[i]=reference::random_number<Bits>(rng,5);}
    // Reuse MPFR variables: conversion, allocation, and data generation excluded
    // from the CPU arithmetic-only baseline; GPU wall includes host copies.
    std::unique_ptr<mpfr_t[]> ma(new mpfr_t[count]),mb(new mpfr_t[count]),mc(new mpfr_t[count]);
    for(int i=0;i<count;++i){mpfr_init2(ma[i],Bits);mpfr_init2(mb[i],Bits);mpfr_init2(mc[i],Bits);
        to_mpfr<Bits>(ma[i],a[i]);to_mpfr<Bits>(mb[i],b[i]);}
    for(Operation op:{Operation::add,Operation::mul,Operation::div}) {
        engine.run(Bits,op,a.data(),b.data(),out.data(),count);
        std::vector<double> gpu,wall,cpu;
        for(int repeat=0;repeat<5;++repeat){
            auto start=std::chrono::steady_clock::now();
            for(int i=0;i<count;++i){if(op==Operation::add)mpfr_add(mc[i],ma[i],mb[i],MPFR_RNDN);
                else if(op==Operation::mul)mpfr_mul(mc[i],ma[i],mb[i],MPFR_RNDN);else mpfr_div(mc[i],ma[i],mb[i],MPFR_RNDN);}
            cpu.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
            auto t=engine.run(Bits,op,a.data(),b.data(),out.data(),count);gpu.push_back(t.gpu_seconds);wall.push_back(t.wall_seconds);
        }
        for(int i=0;i<count;++i)if(!reference::equal<Bits>(out[i],from_mpfr<Bits>(mc[i])))throw std::runtime_error("benchmark output mismatch");
        std::cout<<Bits<<","<<(op==Operation::add?"add":op==Operation::mul?"mul":"div")<<","<<count<<","<<median(cpu)<<","<<median(gpu)<<","<<median(wall)<<","<<median(cpu)/median(wall)<<std::endl;
    }
    for(int i=0;i<count;++i){mpfr_clear(ma[i]);mpfr_clear(mb[i]);mpfr_clear(mc[i]);}
}
int main(int argc,char** argv){
    try{int count=argc>1?std::stoi(argv[1]):65536;if(count<1)throw std::invalid_argument("count must be positive");
        Engine engine;std::cerr<<"Device: "<<engine.device_name()<<"; median of 5 warmed runs; serial MPFR baseline\n";
        std::cout<<std::setprecision(8)<<"bits,operation,count,mpfr_cpu_s,gpu_s,gpu_wall_s,cpu_over_gpu_wall\n";
        benchmark<256>(engine,count);benchmark<384>(engine,count);benchmark<1024>(engine,count);
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
