// Left-looking independent MPFR replay of shifted Linalg blocked factors.
#include "reference.hpp"
#include "limbforge/batched_linalg.hpp"
#include <limits>
#include <iostream>
using namespace limbforge;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<class F> void reject(F f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}require(bad,"invalid blocked-trial call accepted");}
template<int B> using F=Float<B>;
template<int B> std::uint32_t factor_reference(const std::vector<F<B>>& A,std::size_t n,const std::vector<F<B>>& diagonal,const F<B>& mu,std::size_t block,std::vector<F<B>>& out){
    auto nb=block?std::min(block,n):n;out.assign(n*n,zero<B/32>());std::vector<F<B>> shifted=A;
    for(std::size_t i=0;i<n;++i)shifted[i*n+i]=reference::real<B>(Operation::add,A[i*n+i],reference::real<B>(Operation::mul,mu,diagonal[i]));
    auto value=[&](std::size_t i,std::size_t j){auto v=shifted[i*n+j];auto kb=j/nb*nb;
        for(std::size_t k=0;k<kb;k+=nb)v=reference::dot_sub<B>(v,out.data()+i*n+k,1,out.data()+j*n+k,1,nb);
        return reference::dot_sub<B>(v,out.data()+i*n+kb,1,out.data()+j*n+kb,1,j-kb);};
    for(std::size_t j=0;j<n;++j){auto v=value(j,j);if(v.status||v.sign<=0){for(std::size_t i=j;i<n;++i)for(std::size_t k=j;k<=i;++k)out[i*n+k]=zero<B/32>(invalid);return std::uint32_t(j+1);}
        out[j*n+j]=reference::real<B>(Operation::sqrt,v,v);for(std::size_t i=j+1;i<n;++i)out[i*n+j]=reference::real<B>(Operation::div,value(i,j),out[j*n+j]);}
    return 0;
}
template<class T> Buffer<T> put(Engine& e,const std::vector<T>& v){auto b=e.make_buffer<T>(v.size());b.upload(v.data(),v.size());return b;}
template<int B> void fixture(std::size_t n,std::size_t block,bool dense,bool force_fallback,bool gpu,bool spread=false){
    constexpr auto N=B/32;Engine e;BatchedLinalg batch(e);LinalgOptions lo;lo.host_threads=8;lo.force_fallback=force_fallback;Linalg exact(e,lo);
    std::vector<F<B>> A(n*n,zero<N>()),diagonal(n,from_decimal<B>("1")),shifts;std::mt19937_64 rng(B+21+block);
    // Mixed trial outcomes, including failures on both sides of a panel boundary.
    for(const auto* s:{"0","1","-1","-3","-40"})shifts.push_back(from_decimal<B>(s));
    shifts.push_back(zero<N>(invalid));
    if(spread){std::vector<F<B>> base(n*n,zero<N>());for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<=i;++j){auto v=reference::random_number<B>(rng,0);
            if(i==j)v=from_decimal<B>("1");else v.exponent=-100-int(rng()%701);base[i*n+j]=v;}
        for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<=i;++j)A[i*n+j]=reference::dot<B>(base.data()+i*n,1,base.data()+j*n,1,n);}
    else if(dense){std::vector<F<B>> J((n+5)*n);for(auto& x:J)x=reference::random_number<B>(rng,1);
        for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<=i;++j)A[i*n+j]=reference::dot<B>(J.data()+i,n,J.data()+j,n,n+5);
        for(std::size_t i=0;i<n;++i)A[i*n+i]=reference::real<B>(Operation::add,A[i*n+i],from_decimal<B>("2"));}
    else for(std::size_t i=0;i<n;++i){A[i*n+i]=from_decimal<B>(i==(block?std::min(block,n-1):n-1)?"2":"4");if(i)A[i*n+i-1]=from_decimal<B>("0.5");}
    // Upper triangle is deliberately invalid; the factor must never read it.
    for(std::size_t i=0;i<n;++i)for(std::size_t j=i+1;j<n;++j)A[i*n+j]=zero<N>(invalid);
    auto a=put(e,A),d=put(e,diagonal),mu=put(e,shifts),l=e.make_buffer<F<B>>(shifts.size()*n*n+3);auto status=e.make_buffer<std::uint32_t>(shifts.size()+3);
    auto sentinel=from_decimal<B>("777");std::fill(l.mapped(),l.mapped()+l.size(),sentinel);std::fill(status.mapped(),status.mapped()+status.size(),123456);
    FactorOptions o;o.block=block;o.host_macs=0;o.gpu=gpu;
    auto info=batch.cholesky_trials_blocked(exact,{n,shifts.size()},a,d,mu,l,status,o);
    for(std::size_t t=0;t<shifts.size();++t){std::vector<F<B>> expected;auto p=factor_reference<B>(A,n,diagonal,shifts[t],block,expected);
        require(status.mapped()[t]==p,"blocked-trial pivot differs from MPFR");for(std::size_t i=0;i<n*n;++i)if(!reference::equal<B>(l.mapped()[t*n*n+i],expected[i]))throw std::runtime_error("blocked trial MPFR factor mismatch at trial "+std::to_string(t)+", entry "+std::to_string(i));
        // Secondary production oracle: no substituted rounding sequence.
        std::vector<F<B>> shifted=A,one(n*n);for(std::size_t i=0;i<n;++i)shifted[i*n+i]=reference::real<B>(Operation::add,A[i*n+i],reference::real<B>(Operation::mul,shifts[t],diagonal[i]));
        auto cpu=o;cpu.gpu=false;auto result=exact.cholesky(B,shifted.data(),n,one.data(),cpu);require((result.pivot==n?0:result.pivot+1)==p,"single factor pivot mismatch");
        for(std::size_t i=0;i<n*n;++i)require(reference::equal<B>(l.mapped()[t*n*n+i],one[i]),"single Linalg factor differs");}
    for(std::size_t i=shifts.size()*n*n;i<l.size();++i)require(reference::equal<B>(l.mapped()[i],sentinel),"output factor tail overwritten");
    for(std::size_t i=shifts.size();i<status.size();++i)require(status.mapped()[i]==123456,"output status tail overwritten");
    if(gpu&&block&&block<n){require(info.submissions>0&&info.gpu_updates>=info.submissions,"shared exact submissions missing");
        if(force_fallback||spread)require(info.fallback_outputs>0,"exact repair not exercised");else require(info.gpu_outputs>0,"physical GPU exact output not exercised");
        const char* mode=std::getenv("WOLFNUM_EXACT_TRIAL_SCHEDULE");if(dense&&!spread&&!force_fallback&&mode&&std::string(mode)=="grouped")require(info.grouped_updates>0,"dense factor did not exercise grouped dispatches");}
    else require(!info.submissions&&!info.gpu_updates,"host/single-block factor dispatched GPU updates");
    auto w=batch.workspaces();require(!w.busy_workspaces,"factor left busy scratch");batch.release_workspaces();require(!batch.workspaces().idle_bytes,"factor scratch release failed");
}
// Guaranteed failure at the first column of the prefetched second panel, while
// another shifted trial succeeds. Dense completed columns exercise the live SYRK.
template<int B> void late_panel_failure(bool host_only){
    constexpr auto N=B/32;const std::size_t n=65,count=3;Engine e;BatchedLinalg batch(e);LinalgOptions lo;lo.host_threads=8;Linalg exact(e,lo);
    std::vector<F<B>> base(n*n,zero<N>()),A(n*n,zero<N>()),diagonal(n,from_decimal<B>("1"));
    for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<=i;++j){
        if(i==j&&i!=32)base[i*n+j]=from_decimal<B>("1");
        else if(i!=j&&j<32)base[i*n+j]=from_decimal<B>("0.125");}
    for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<=i;++j)A[i*n+j]=reference::dot<B>(base.data()+i*n,1,base.data()+j*n,1,n);
    for(std::size_t i=0;i<n;++i)for(std::size_t j=i+1;j<n;++j)A[i*n+j]=zero<N>(invalid);
    std::vector<F<B>> shifts={zero<N>(),from_decimal<B>("0.0625"),from_decimal<B>("-0.0009765625")};
    auto a=put(e,A),d=put(e,diagonal),mu=put(e,shifts),l=e.make_buffer<F<B>>(count*n*n);auto status=e.make_buffer<std::uint32_t>(count);
    FactorOptions o;o.block=32;o.host_macs=0;o.gpu=!host_only;
    auto info=batch.cholesky_trials_blocked(exact,{n,count},a,d,mu,l,status,o);
    for(std::size_t t=0;t<count;++t){std::vector<F<B>> expected;auto pivot=factor_reference<B>(A,n,diagonal,shifts[t],32,expected);
        require(pivot==(t==1?0:33),"late-failure fixture lost its known pivot");require(status.mapped()[t]==pivot,"prefetched pivot status wrong");
        for(std::size_t q=0;q<n*n;++q)require(reference::equal<B>(l.mapped()[t*n*n+q],expected[q]),"prefetched prefix or invalid tail overwritten");}
    require(info.panels==3,"look-ahead panel accounting wrong");
    if(!host_only)require(info.submissions==2&&info.gpu_outputs>0,"late failure did not span actual shared GPU updates");
    require(!batch.workspaces().busy_workspaces,"look-ahead left busy scratch");
}
template<int B> void width(bool host_only){fixture<B>(13,4,false,false,!host_only);fixture<B>(17,8,true,false,!host_only);fixture<B>(9,0,true,false,!host_only);fixture<B>(13,4,false,true,!host_only);fixture<B>(13,4,false,false,false);fixture<B>(13,4,false,false,!host_only,true);fixture<B>(65,32,true,false,!host_only);late_panel_failure<B>(host_only);std::cout<<B<<" bits: blocked trial factors, mixed pivots and "<<(host_only?"host-only scheduling":"shared GPU/exact repair")<<" match MPFR\n";}
void validation(bool host_only){Engine e,foreign;BatchedLinalg batch(e);Linalg exact(e);using T=F<64>;auto a=e.make_buffer<T>(9),d=e.make_buffer<T>(3),mu=e.make_buffer<T>(2),l=e.make_buffer<T>(18),small=e.make_buffer<T>(1);auto s=e.make_buffer<std::uint32_t>(2);
    reject([&]{batch.cholesky_trials_blocked(exact,{3,2},a,d,mu,small,s);});reject([&]{batch.cholesky_trials_blocked(exact,{3,2},a,d,mu,a,s);});
    auto wrong=foreign.make_buffer<T>(9);reject([&]{batch.cholesky_trials_blocked(exact,{3,2},wrong,d,mu,l,s);});
    // A late invalid operand must not leave earlier operands claimed.
    a.mapped();d.mapped();mu.mapped();l.mapped();s.mapped();
    reject([&]{batch.cholesky_trials_blocked(exact,{std::numeric_limits<std::size_t>::max(),2},a,d,mu,l,s);});
    reject([&]{batch.cholesky_trials_blocked(exact,{0,2},a,d,mu,l,s);});FactorOptions bad;bad.host_macs=std::numeric_limits<double>::quiet_NaN();reject([&]{batch.cholesky_trials_blocked(exact,{3,2},a,d,mu,l,s,bad);});
    if(!host_only){auto b=e.batch();b.run(Operation::square,a,a);auto submitted=b.submit();bool busy=false;try{batch.cholesky_trials_blocked(exact,{3,2},a,d,mu,l,s);}catch(const std::logic_error&){busy=true;}
    require(busy,"busy trial input accepted");submitted.wait();l.mapped();s.mapped();} // partial claims were released.
    Buffer<T> empty;Buffer<std::uint32_t> no_status;auto no_trials=batch.cholesky_trials_blocked(exact,{0,0},empty,empty,empty,empty,no_status);require(!no_trials.submissions,"zero trials dispatched GPU");
    std::fill(a.mapped(),a.mapped()+9,zero<2>());for(int i=0;i<3;++i)a.mapped()[i*3+i]=from_decimal<64>("1");std::fill(d.mapped(),d.mapped()+3,from_decimal<64>("1"));
    FactorOptions only_host;only_host.gpu=false;batch.cholesky_trials_blocked(exact,{3,2},a,d,d,l,s,only_host);
    auto want=reference::real<64>(Operation::sqrt,from_decimal<64>("2"),zero<2>());
    for(int t=0;t<2;++t){require(!s.mapped()[t],"aliased read-only inputs failed");for(int i=0;i<3;++i)require(reference::equal<64>(l.mapped()[t*9+i*3+i],want),"aliased input factor wrong");}
}
template<int B=64> void select(int bits,bool all,bool host_only){if(all||bits==B)width<B>(host_only);if constexpr(B<1024)select<B+32>(bits,all,host_only);}
int main(int argc,char** argv){try{check_library_version();int bits=352;bool all=false,host_only=false;for(int i=1;i<argc;++i){std::string arg=argv[i];if(arg=="--all-widths")all=true;else if(arg=="--host-only")host_only=true;else if(arg=="--bits"&&i+1<argc)bits=std::stoi(argv[++i]);else throw std::invalid_argument("--all-widths, --bits B or --host-only expected");}
    if(bits<64||bits>1024||bits%32)throw std::invalid_argument("invalid width");validation(host_only);select(bits,all,host_only);std::cout<<"Blocked exact damping trials reference run passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
