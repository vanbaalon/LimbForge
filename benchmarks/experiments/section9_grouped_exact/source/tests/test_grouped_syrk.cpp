// Independent MPFR checks of the private host-packed grouped exact SYRK fast path.
#include "reference.hpp"
#include "linalg_internal.hpp"
#include <iostream>
using namespace limbforge;
using H=detail::LinalgHooks;
void require(bool b,const char* why){if(!b)throw std::runtime_error(why);}
template<int B> void fixture(std::size_t M,std::size_t K,std::size_t count){
    using T=Float<B>;constexpr int N=B/32;Engine e;LinalgOptions options;options.host_threads=4;Linalg la(e,options);
    const auto ac=count*K*M,cc=count*M*M;auto A=e.make_buffer<T>(ac+5),C=e.make_buffer<T>(cc+5),C2=e.make_buffer<T>(cc+5);
    std::mt19937_64 rng(19017+B+M+K);std::vector<T> input(ac),old(cc+5),expected(cc+5);
    for(std::size_t q=0;q<count;++q)for(std::size_t k=0;k<K;++k)for(std::size_t i=0;i<M;++i){auto x=reference::random_number<B>(rng,0);
        // Every line is single-band, but each trial/row uses a distinct exponent
        // range. Some trial sums alternate signs; old entries force cancellation.
        x.exponent+=int(q)*700-1100+int(i%7)*50+int(k%17);input[q*K*M+k*M+i]=x;}
    const auto sentinel=from_decimal<B>("777");std::fill(A.mapped(),A.mapped()+A.size(),sentinel);A.upload(input.data(),ac);
    for(auto& x:old)x=reference::random_number<B>(rng,3);
    for(std::size_t q=0;q<count;++q)for(std::size_t i=0;i<M;++i)for(std::size_t j=0;j<=i;++j){auto idx=q*M*M+i*M+j;
        if((i+j)%7==0)old[idx]=reference::dot<B>(input.data()+q*K*M+i,M,input.data()+q*K*M+j,M,K);
        if((i+j)%19==0)old[idx]=zero<N>(invalid);}
    for(std::size_t i=cc;i<old.size();++i)old[i]=sentinel;expected=old;
    for(std::size_t q=0;q<count;++q)for(std::size_t i=0;i<M;++i)for(std::size_t j=0;j<=i;++j){auto idx=q*M*M+i*M+j;
        expected[idx]=reference::dot_sub<B>(old[idx],input.data()+q*K*M+i,M,input.data()+q*K*M+j,M,K);}
    C.upload(old.data(),old.size());C2.upload(old.data(),old.size());auto batch=e.batch();
    auto first=H::syrk_group(la,batch,B,detail::Access::operand(A),detail::Access::operand(C),count,K,M);
    auto second=H::syrk_group(la,batch,B,detail::Access::operand(A),detail::Access::operand(C2),count,K,M);
    require(bool(first)&&bool(second),"valid grouped fast path declined");require(!first->done.load()&&!second->done.load(),"grouped output resolved before wait");
    auto memory=la.workspaces();require(memory.busy_workspaces>=2&&memory.busy_bytes,"grouped workspace not retained by live calls");
    la.release_workspaces();require(la.workspaces().busy_workspaces>=2,"busy grouped scratch freed early");
    auto submission=batch.submit();bool busy=false;try{C.mapped();}catch(const std::logic_error&){busy=true;}require(busy,"in-flight grouped output mappable");submission.wait();
    require(first->done.load()&&second->done.load()&&!first->provisional_reads&&!second->provisional_reads,"grouped completion missing");
    require(first->gpu_outputs==count*M*(M+1)/2,"grouped output accounting wrong");
    for(std::size_t i=0;i<old.size();++i)if(!reference::equal<B>(C.mapped()[i],expected[i])||!reference::equal<B>(C2.mapped()[i],expected[i]))
        throw std::runtime_error("grouped exact SYRK MPFR mismatch at "+std::to_string(B)+" bits, entry "+std::to_string(i));
    for(std::size_t i=ac;i<A.size();++i)require(reference::equal<B>(A.mapped()[i],sentinel),"grouped input tail changed");
    require(!la.workspaces().busy_workspaces,"grouped scratch still busy after wait");
    // All-or-nothing fast-path rejection: no output changes before/after an empty submission.
    const auto saved=input[0];
    for(int mode=0;mode<3;++mode){input[0]=saved;if(mode==0)input[0]=zero<N>(invalid);
        else if(mode==1)input[0].exponent+=1000;
        else for(std::size_t k=0;k<K;++k)input[k*M]=zero<N>();
        A.upload(input.data(),ac);C.upload(old.data(),old.size());auto fallback=e.batch();
        require(!H::syrk_group(la,fallback,B,detail::Access::operand(A),detail::Access::operand(C),count,K,M),"unsupported input incorrectly accepted");fallback.submit().wait();
        for(std::size_t i=0;i<old.size();++i)require(reference::equal<B>(C.mapped()[i],old[i]),"declined fast path changed output");}
    bool alias=false;try{auto b=e.batch();H::syrk_group(la,b,B,detail::Access::operand(C),detail::Access::operand(C),count,K,M);}catch(const std::invalid_argument&){alias=true;}
    require(alias,"aliased grouped output accepted");
}
template<int B=64> void select(int bits,bool all){if(all||bits==B){fixture<B>(67,33,3);fixture<B>(9,7,8);fixture<B>(131,65,3);std::cout<<B<<" bits: grouped exact SYRK MPFR, cancellation, statuses, tails and busy release passed\n";}if constexpr(B<1024)select<B+32>(bits,all);}
int main(int argc,char** argv){try{check_library_version();int bits=352;bool all=false;for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--all-widths")all=true;else if(a=="--bits"&&i+1<argc)bits=std::stoi(argv[++i]);else throw std::invalid_argument("--bits B or --all-widths expected");}
    if(bits<64||bits>1024||bits%32)throw std::invalid_argument("invalid width");select(bits,all);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
