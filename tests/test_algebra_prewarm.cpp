#include "section9_cases.hpp"
#include <future>
#include <limits>

template<int B> void focus(){
    Engine e;BatchedLinalg batch(e);Linalg exact(e);
    BatchedPrewarm request;request.bits={B};request.complex_gemm={{2,9,5,17,153,0,48},{2,4,4,4,16,16,16}};
    request.real_gemm={{2,4,4,4,16,16,16}};request.complex_power={{2,7,3,4,-2}};
    // 20 rows produces an 8-row panel and a final 4-row panel at compact storage.
    request.real_power={{2,4,19,4,0}};request.normal_fused={false,true};request.polynomial_fused={false,true};
    request.cholesky_trials=true;request.cholesky_solve=true;request.exact_normal=true;
    auto compact=request;compact.power_storage=PowerStorage::compact;
    auto first=batch.prewarm_async(request),second=batch.prewarm_async(compact);
    auto dense=exact.prewarm_async({{B},{0,9,32,1100},true});
    // Owner-thread work overlaps background cache preparation. Same keys must be safe.
    products<B>(e,batch);normal<B>(e,batch);polynomial<B>(e,batch);first.get();second.get();dense.get();
    batch.prewarm(request);batch.prewarm(compact);exact.prewarm({{B},{9},true});
    auto one=from_decimal<B>("1"),two=from_decimal<B>("2");std::vector<F<B>> E(8,one),Y(8,two),W(32,one),powers(160),expected_powers(160);
    auto value=from_decimal<B>("4");for(int n=0;n<20;++n){for(int trial=0;trial<2;++trial)for(int j=0;j<4;++j)expected_powers[(trial*20+n)*4+j]=value;value=rm<B>(value,two);}
    batch.power_moments(B,false,request.real_power[0],E.data(),Y.data(),W.data(),powers.data(),PowerStorage::compact);check<B>(powers,expected_powers,"prewarmed compact real 4-row tail");
    constexpr int rows=9,cols=4;std::mt19937_64 rng(B+117);std::vector<F<B>> J(rows*cols),gram(cols*cols),want(cols*cols);
    for(auto& x:J)x=reference::random_number<B>(rng,2);
    for(int i=0;i<cols;++i)for(int j=0;j<cols;++j)want[i*cols+j]=reference::dot<B>(J.data()+i,cols,J.data()+j,cols,rows);
    exact.syrk(B,J.data(),rows,cols,gram.data(),false);check<B>(gram,want,"prewarmed host SYRK");
    auto jb=put(e,J),gb=e.make_buffer<F<B>>(gram.size());auto encoded=e.batch();auto ticket=exact.syrk(encoded,jb,rows,cols,gb,false);encoded.submit().wait();
    require(ticket.resolved(),"prewarmed resident ticket unresolved");check<B>(get(gb),want,"prewarmed resident SYRK");
    std::vector<F<B>> A,L(9),expected;for(auto x:{4,2,0,2,10,3,0,3,10})A.push_back(from_decimal<B>(std::to_string(x)));
    for(auto x:{2,0,0,1,3,0,0,1,3})expected.push_back(from_decimal<B>(std::to_string(x)));
    FactorOptions options;options.block=1;options.host_macs=0;
    auto factor=exact.cholesky(B,A.data(),3,L.data(),options);require(factor.pivot==3&&factor.gpu_updates>0,"prewarmed Cholesky failed or skipped GPU updates");check<B>(L,expected,"prewarmed GPU-updated Cholesky");
    std::cout<<B<<" bits: prewarm/encode overlap, repeat requests, compact tails, host/resident exact and Cholesky passed\n";
}
template<class F> void reject(F f){bool rejected=false;try{f();}catch(const std::invalid_argument&){rejected=true;}require(rejected,"bad prewarm request accepted");}
int main(){try{
    check_library_version();
    {Engine e;BatchedLinalg b(e);Linalg l(e);BatchedPrewarm bad;bad.bits={63};reject([&]{b.prewarm_async(bad);});
        bad.bits={64};bad.power_storage=static_cast<PowerStorage>(99);reject([&]{b.prewarm(bad);});
        bad.power_storage=PowerStorage::full_table;bad.real_gemm={{2,1,1,1,1,1,0}};reject([&]{b.prewarm_async(bad);});
        bad.real_gemm={{1,std::numeric_limits<std::size_t>::max(),2,1,0,0,0}};reject([&]{b.prewarm(bad);});
        reject([&]{l.prewarm_async({{65},{},false});});reject([&]{l.prewarm({{64},{65473},true});});
        b.prewarm({});l.prewarm_async({}).get();}
    focus<64>();focus<352>();
    std::future<void> a,b;
    {Engine e;BatchedLinalg batched(e);Linalg exact(e);BatchedPrewarm r;r.bits={224};r.complex_power={{1,3,3,2,0}};a=batched.prewarm_async(r);b=exact.prewarm_async({{224},{9},true});}
    a.get();b.get(); // caches/device outlive their unit and Engine, without host scratch.
    std::cout<<"Algebra prewarm focused references, lifetime and validation passed\n";return 0;
}catch(const std::exception& e){std::cerr<<"algebra prewarm: "<<e.what()<<'\n';return 1;}}
