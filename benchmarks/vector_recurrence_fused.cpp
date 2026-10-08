// Fused vector recurrence (round 46-fused-wide): GPU device time per lane-step for the rank-one, tangent and
// matrix forms at a few widths, composed mode for scale. Pipelines are warmed before timing; results are
// validated in tests/test_vector_recurrence.cpp. Output CSV: bits,mode,fused,lanes,steps,median_device_s,ns_per_lane_step.
// Usage: vector_recurrence_fused [--repeats R] [--bits B]... [--modes rank,tangent,matrix] [--composed]
#include "reference.hpp"
#include <algorithm>
#include <iostream>
#include <set>
#include <string>
using namespace limbforge;
template<int Bits> void shape(Engine& e,const std::string& mode,bool fused,int repeats){
    using C=Complex<Bits/32>;std::mt19937_64 rng(Bits);
    auto rc=[&](int span,int shift){C z={reference::random_number<Bits>(rng,span),reference::random_number<Bits>(rng,span)};z.re.exponent+=shift;z.im.exponent+=shift;return z;};
    const std::size_t lanes=16384;const unsigned steps=Bits>512?8:24;
    VectorRecurrence s;s.lanes=lanes;s.steps=steps;s.lanes_per_weight=4;s.fused=fused;s.matrix=mode=="matrix";s.tangent=mode=="tangent";
    std::size_t G=lanes/4;std::vector<C> start(4*lanes),p(std::size_t(steps)*(s.matrix?16:4)*G),q(s.matrix?1:std::size_t(steps)*4*G),out(4*lanes),base,dp,dq,none(1);
    for(auto& z:start)z=rc(3,0);for(auto& z:p)z=rc(1,s.matrix?-2:-1);for(auto& z:q)z=rc(1,-1);
    if(s.tangent){s.lanes_per_base=4;s.lanes_per_tangent=4;base.resize(std::size_t(steps+1)*4*G);dp.resize(std::size_t(steps)*4*G);dq.resize(dp.size());
        for(auto& z:base)z=rc(3,0);for(auto& z:dp)z=rc(1,-3);for(auto& z:dq)z=rc(1,-3);}
    auto run=[&]{return e.vector_recurrence(Bits,s,start.data(),p.data(),q.data(),none.data(),out.data(),s.tangent?base.data():nullptr,s.tangent?dp.data():nullptr,s.tangent?dq.data():nullptr);};
    run();std::vector<double> d;for(int r=0;r<repeats;++r)d.push_back(run().gpu_seconds);std::sort(d.begin(),d.end());double m=d[d.size()/2];
    std::cout<<Bits<<','<<mode<<','<<(fused?1:0)<<','<<lanes<<','<<steps<<','<<m<<','<<m/(double(lanes)*steps)*1e9<<std::endl;
}
template<int Bits> void widths(Engine& e,const std::set<int>& bits,const std::vector<std::string>& modes,bool composed,int repeats){
    if(bits.count(Bits))for(auto& m:modes){shape<Bits>(e,m,true,repeats);if(composed)shape<Bits>(e,m,false,repeats);}
    if constexpr(Bits<1024)widths<Bits+32>(e,bits,modes,composed,repeats);
}
int main(int argc,char** argv){try{
    int repeats=5;bool composed=false;std::set<int> bits;std::vector<std::string> modes;
    for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--repeats"&&i+1<argc)repeats=std::stoi(argv[++i]);else if(a=="--bits"&&i+1<argc)bits.insert(std::stoi(argv[++i]));
        else if(a=="--composed")composed=true;else if(a=="--modes"&&i+1<argc){std::string m=argv[++i];for(std::size_t p=0;p<=m.size();){auto q=m.find(',',p);if(q==std::string::npos)q=m.size();modes.push_back(m.substr(p,q-p));p=q+1;}}
        else throw std::invalid_argument("usage: vector_recurrence_fused [--repeats R] [--bits B]... [--modes rank,tangent,matrix] [--composed]");}
    if(bits.empty())bits={256,384,1024};if(modes.empty())modes={"rank","tangent","matrix"};
    Engine e;std::cout<<"bits,mode,fused,lanes,steps,median_device_s,ns_per_lane_step\n";widths<64>(e,bits,modes,composed,repeats);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
