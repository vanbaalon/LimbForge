// On-device acceptance check (plan S4): exact comparison of values with a threshold, host and resident.
#include "reference.hpp"
#include "limbforge/numerics.hpp"
#include <stdexcept>
using namespace limbforge;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
template<int N> int compare(const Number<N>& a,const Number<N>& b){if(a.sign!=b.sign)return a.sign<b.sign?-1:1;int m=magnitude_compare(a,b);return a.sign>=0?m:-m;}
template<int Bits> ThresholdInfo expected(const std::vector<Float<Bits>>& v,const Float<Bits>& t){
    ThresholdInfo r;r.compared=std::uint32_t(v.size());
    for(std::size_t i=0;i<v.size();++i){word st=v[i].status|t.status;r.status|=st;bool pass=!st&&compare(v[i],t)<=0;if(!pass){++r.failing;if(r.first_failing==no_index)r.first_failing=std::uint32_t(i);}}
    return r;}
bool same(const ThresholdInfo& a,const ThresholdInfo& b){return a.failing==b.failing&&a.first_failing==b.first_failing&&a.status==b.status&&a.compared==b.compared;}
template<int Bits> void run(Engine& e,Numerics& nm){
    using F=Float<Bits>;std::mt19937_64 rng(Bits+3);F t=reference::random_number<Bits>(rng,3);t.sign=1;
    for(std::size_t n:{0ul,1ul,7ul,1000ul,70000ul}){
        std::vector<F> v(n);for(auto& x:v){x=reference::random_number<Bits>(rng,4);}
        auto above=t;above.limb[0]+=1;auto below=t;below.limb[0]-=1;
        if(n>=7){v[0]=t;v[1]=above;v[2]=below;v[3]=zero<Bits/32>();v[4]=negate(t);v[5].status=invalid;v[6]=t;v[6].exponent+=1;}
        for(int variant=0;variant<2;++variant){F th=t;if(variant)th.status=exponent_overflow;
            ThresholdInfo host;nm.check_threshold(Bits,n,v.data(),&th,&host);auto want=expected<Bits>(v,th);
            require(same(host,want),"host threshold bits="+std::to_string(Bits)+" n="+std::to_string(n));
            auto bv=e.make_buffer<F>(std::max<std::size_t>(n,1)),bt=e.make_buffer<F>(1);auto bi=e.make_buffer<ThresholdInfo>(1);if(n)bv.upload(v.data(),n);bt.upload(&th,1);
            {auto b=e.batch();nm.check_threshold(b,bv,n,bt,bi);b.submit().wait();}ThresholdInfo res;bi.download(&res,1);
            require(same(res,want),"resident threshold bits="+std::to_string(Bits)+" n="+std::to_string(n));}
    }
    // Chain in one batch: residual r = a - b, scaled_residual per segment, then the acceptance check.
    const std::size_t S=64,L=33;std::vector<F> a(S*L),b(S*L),sc(S*L),vals(S);for(auto& x:a)x=reference::random_number<Bits>(rng,2);for(auto& x:b)x=reference::random_number<Bits>(rng,2);
    for(auto& x:sc){x=reference::random_number<Bits>(rng,2);x.sign=1;}
    auto A=e.make_buffer<F>(S*L),B=e.make_buffer<F>(S*L),R=e.make_buffer<F>(S*L),SC=e.make_buffer<F>(S*L),V=e.make_buffer<F>(S),T=e.make_buffer<F>(1);auto I=e.make_buffer<ThresholdInfo>(1);
    A.upload(a.data(),a.size());B.upload(b.data(),b.size());SC.upload(sc.data(),sc.size());T.upload(&t,1);Segments seg{S,L,false};
    {auto batch=e.batch();batch.run(Operation::sub,A,B,R);nm.scaled_residual(batch,seg,R,SC,V);nm.check_threshold(batch,V,S,T,I);batch.submit().wait();}
    V.download(vals.data(),S);ThresholdInfo got;I.download(&got,1);require(same(got,expected<Bits>(vals,t)),"chained threshold bits="+std::to_string(Bits));
    std::cout<<Bits<<" bits: threshold checks match (host, resident, chained)"<<std::endl;
}
int main(){try{Engine e;Numerics nm(e);run<64>(e,nm);run<224>(e,nm);run<256>(e,nm);run<384>(e,nm);run<1024>(e,nm);std::cout<<"All threshold checks passed."<<std::endl;return 0;}
catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}}
