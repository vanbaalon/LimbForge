#include "reference.hpp"
#include <algorithm>
#include <array>
#include <iomanip>
#include <memory>
using namespace limbforge;
constexpr std::array<unsigned,5> groups={32,64,128,256,512};
constexpr int samples=9;
double percentile(std::vector<double> a,double p){std::sort(a.begin(),a.end());return a[std::size_t(p*(a.size()-1))];}
const char* op_name(Operation op){const char* names[]={"add","sub","mul","div","complex_add","complex_mul","complex_div"};return names[int(op)];}
template<int Bits,bool IsComplex> void cases(std::array<std::unique_ptr<Engine>,5>& engines,std::size_t count){
    using T=std::conditional_t<IsComplex,Complex<Bits/32>,Float<Bits>>;
    std::mt19937_64 rng(20261007+Bits);std::vector<T>a(count),b(count),expected(count),out(count);
    for(std::size_t i=0;i<count;++i){if constexpr(IsComplex){a[i]={reference::random_number<Bits>(rng,5),reference::random_number<Bits>(rng,5)};b[i]={reference::random_number<Bits>(rng,5),reference::random_number<Bits>(rng,5)};}
        else {a[i]=reference::random_number<Bits>(rng,5);b[i]=reference::random_number<Bits>(rng,5);}}
    std::array<Buffer<T>,5>x,y,z;
    for(int j=0;j<5;++j){x[j]=engines[j]->template make_buffer<T>(count);y[j]=engines[j]->template make_buffer<T>(count);z[j]=engines[j]->template make_buffer<T>(count);x[j].upload(a.data(),count);y[j].upload(b.data(),count);}
    for(int which=IsComplex?4:0;which<(IsComplex?7:4);++which){auto op=Operation(which);
        for(std::size_t i=0;i<count;++i){if constexpr(IsComplex)expected[i]=reference::complex<Bits>(op,a[i],b[i]);else expected[i]=reference::real<Bits>(op,a[i],b[i]);}
        std::array<std::vector<double>,5>gpu,wall;
        for(int repeat=-2;repeat<samples;++repeat)for(int k=0;k<5;++k){int j=(k+repeat+2)%5;auto batch=engines[j]->batch();batch.run(op,x[j],y[j],z[j]);auto t=batch.submit().wait();
            if(repeat>=0){gpu[j].push_back(t.gpu_seconds);wall[j].push_back(t.wall_seconds);}
            z[j].download(out.data(),count);
            for(std::size_t i=0;i<count;++i){bool equal;if constexpr(IsComplex)equal=reference::equal_complex<Bits>(out[i],expected[i]);else equal=reference::equal<Bits>(out[i],expected[i]);
                if(!equal)throw std::runtime_error("tuning MPFR mismatch bits="+std::to_string(Bits)+" op="+op_name(op)+" group="+std::to_string(groups[j])+" index="+std::to_string(i));}
        }
        for(int j=0;j<5;++j){auto p=engines[j]->pipeline_info(Bits,op);
            std::cout<<Bits<<','<<op_name(op)<<','<<count<<','<<groups[j]<<','<<p.threads_per_threadgroup<<','<<p.simd_width<<','<<p.max_threads<<','<<samples<<','<<percentile(gpu[j],.5)<<','<<percentile(wall[j],.5)<<','<<percentile(wall[j],.9)<<std::endl;}
    }
}
template<int Bits>void suite(std::array<std::unique_ptr<Engine>,5>& engines){for(auto n:{257u,4096u,65536u}){cases<Bits,false>(engines,n);cases<Bits,true>(engines,n);}}
int main(){try{std::array<std::unique_ptr<Engine>,5> engines;for(int j=0;j<5;++j)engines[j]=std::make_unique<Engine>(EngineOptions{groups[j]});
    std::cerr<<engines[0]->device_name()<<"; interleaved workgroup sweep; every dispatch checked against MPFR\n";
    std::cout<<std::setprecision(10)<<"bits,operation,count,requested_threads,actual_threads,simd_width,max_threads,samples,gpu_median_s,wall_median_s,wall_p90_s\n";
    suite<256>(engines);suite<384>(engines);suite<1024>(engines);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
