#include "reference.hpp"
#include <algorithm>
#include <iomanip>
using namespace limbforge;
double median(std::vector<double> a){std::sort(a.begin(),a.end());return a[a.size()/2];}
template<int Bits>void cases(Engine& e,std::size_t count){using F=Float<Bits>;
    std::mt19937_64 rng(20261007+Bits);std::vector<F>a(count),expected(count),out(count);for(std::size_t i=0;i<count;++i){a[i]=reference::random_number<Bits>(rng,5);expected[i]=reference::real<Bits>(Operation::square,a[i],a[i]);}
    auto x=e.make_buffer<F>(count),z=e.make_buffer<F>(count);x.upload(a.data(),count);std::vector<double>gpu[2],wall[2];
    for(int repeat=-2;repeat<9;++repeat)for(int k=0;k<2;++k){int method=(repeat+2+k)%2;auto batch=e.batch();if(method)batch.run(Operation::square,x,z);else batch.run(Operation::mul,x,x,z);auto t=batch.submit().wait();
        if(repeat>=0){gpu[method].push_back(t.gpu_seconds);wall[method].push_back(t.wall_seconds);}z.download(out.data(),count);
        for(std::size_t i=0;i<count;++i)if(!reference::equal<Bits>(out[i],expected[i]))throw std::runtime_error("square comparison MPFR mismatch bits="+std::to_string(Bits));}
    for(int method=0;method<2;++method)std::cout<<Bits<<','<<count<<','<<(method?"square":"mul_self")<<",9,"<<median(gpu[method])<<','<<median(wall[method])<<std::endl;
}
int main(){try{Engine e;std::cerr<<e.device_name()<<"; interleaved resident square versus mul(a,a); every dispatch checked against MPFR\n";
    std::cout<<std::setprecision(10)<<"bits,count,method,samples,gpu_median_s,wall_median_s\n";for(auto n:{257u,4096u,65536u}){cases<256>(e,n);cases<384>(e,n);cases<1024>(e,n);}return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
