#include "prototype.hpp"
#include "reference.hpp"
#include <algorithm>
#include <type_traits>
using namespace limbforge;using namespace wolfnum::experimental;
static void need(bool value,const char* msg){if(!value)throw std::runtime_error(msg);}
template<class T> Buffer<T> upload(Engine& e,const std::vector<T>& v){auto b=e.make_buffer<T>(v.size());b.upload(v.data(),v.size());return b;}
template<class T> std::vector<T> download(const Buffer<T>& b){std::vector<T> v(b.size());b.download(v.data(),v.size());return v;}
// Independent oracle: exact 2B-bit products and a single MPFR sum at B bits.
// No arithmetic core/fallback routine participates in reference evaluation.
template<int B> Float<B> component(const Float<B>* a,const Float<B>* b,unsigned n,const Float<B>* old,bool negative){
    word status=old?old->status:0;for(unsigned i=0;i<n;++i)status|=a[i].status|b[i].status;if(status)return zero<B/32>(status);
    reference::ExponentRange range;reference::MP x(B),y(B),result(B);mpfr_t products[9];mpfr_ptr terms[9];
    for(unsigned i=0;i<n;++i){mpfr_init2(products[i],2*B);terms[i]=products[i];to_mpfr<B>(x.x,a[i]);to_mpfr<B>(y.x,b[i]);need(mpfr_mul(products[i],x.x,y.x,MPFR_RNDN)==0,"oracle product inexact");}
    if(old){mpfr_init2(products[n],B);to_mpfr<B>(products[n],*old);terms[n]=products[n];}
    mpfr_sum(result.x,terms,n+unsigned(old!=nullptr),MPFR_RNDN);if(negative)mpfr_neg(result.x,result.x,MPFR_RNDN);auto out=from_mpfr<B>(result.x);
    for(unsigned i=0;i<n+unsigned(old!=nullptr);++i)mpfr_clear(products[i]);return out;
}
template<int B,class T> std::vector<T> oracle(const ExactGemm4Shape& s,const std::vector<T>& a,const std::vector<T>& b,std::vector<T> out){
    for(std::size_t t=0;t<s.count;++t)for(unsigned i=0;i<4;++i)for(unsigned j=0;j<4;++j){auto o=t*s.stride_c+i*4+j;
        if constexpr(std::is_same_v<T,Float<B>>){Float<B> x[4],y[4];for(unsigned k=0;k<4;++k){x[k]=a[t*s.stride_a+i*4+k];y[k]=b[t*s.stride_b+k*4+j];}out[o]=component<B>(x,y,4,s.accumulate?&out[o]:nullptr,s.negative);}
        else for(unsigned part=0;part<2;++part){Float<B> x[8],y[8];for(unsigned k=0;k<4;++k){const auto& l=a[t*s.stride_a+i*4+k];const auto& r=b[t*s.stride_b+k*4+j];
            x[2*k]=l.re;y[2*k]=part?r.im:r.re;x[2*k+1]=part?l.im:negate(l.im);y[2*k+1]=part?r.re:r.im;}
            auto v=component<B>(x,y,8,s.accumulate?(part?&out[o].im:&out[o].re):nullptr,s.negative);if(part)out[o].im=v;else out[o].re=v;}
    }return out;
}
template<int B,class T> void compare(const std::vector<T>& got,const std::vector<T>& want,const char* context){need(got.size()==want.size(),"reference size");
    for(std::size_t i=0;i<got.size();++i){bool equal;if constexpr(std::is_same_v<T,Float<B>>)equal=reference::equal<B>(got[i],want[i]);else equal=reference::equal_complex<B>(got[i],want[i]);
        if(!equal)throw std::runtime_error(std::string(context)+" mismatch at "+std::to_string(i)+" bits "+std::to_string(B));}}
template<int B,class T> T random_value(std::mt19937_64& rng,int span){if constexpr(std::is_same_v<T,Float<B>>)return reference::random_number<B>(rng,span);else return {reference::random_number<B>(rng,span),reference::random_number<B>(rng,span)};}
template<class T> T negative(T v){if constexpr(detail::Format<T>::complex)return {negate(v.re),negate(v.im)};else return negate(v);}
template<int B,class T> void cases(Engine& e,ExactGemm4& unit,bool dense){
    std::mt19937_64 rng(B+sizeof(T));const auto count=dense?4097u:5u;
    for(bool wide:{false,true})for(bool accumulate:{false,true})for(bool broadcast:{false,true}){
        ExactGemm4Shape s{count,broadcast?0u:19u,broadcast?21u:0u,23,accumulate,accumulate!=broadcast};
        auto extent=[&](std::size_t stride){return (s.count-1)*stride+16;};std::vector<T> a(extent(s.stride_a)),b(extent(s.stride_b)),initial(s.count*s.stride_c);
        for(auto* v:{&a,&b,&initial})for(auto& z:*v)z=random_value<B,T>(rng,wide?4096:4);
        // Exact cancelling products plus a small surviving term test deep cancellation.
        a[1]=negative(a[0]);for(unsigned j=0;j<4;++j)b[4+j]=b[j];
        // Component-local statuses: one input component is invalid, another row stays valid.
        if constexpr(detail::Format<T>::complex)a[4].im=zero<B/32>(invalid);else a[4]=zero<B/32>(invalid);
        auto expected=oracle<B>(s,a,b,initial);auto ab=upload(e,a),bb=upload(e,b),out=upload(e,initial);
        for(unsigned repeat=0;repeat<(dense?3u:1u);++repeat){out.upload(initial.data(),initial.size());auto batch=e.batch();auto ticket=unit.gemm(batch,s,ab,bb,out);
            need(!ticket.resolved(),"premature ticket resolution");bool rejected=false;try{ticket.report();}catch(const std::logic_error&){rejected=true;}need(rejected,"premature report accepted");
            batch.submit().wait();need(ticket.resolved(),"ticket not resolved");auto report=ticket.report();need(report.gpu_components+report.host_components==s.count*16*(detail::Format<T>::complex?2:1),"component accounting");
            if(!wide)need(report.host_components==0,"bounded fixture unexpectedly repaired");else need(report.host_components>0,"wide fixture did not exercise fallback");
            need(!report.provisional_reads,"unexpected provisional read");compare<B>(download(out),expected,"resident");}
        auto host=initial;ExactGemm4Report report;unit.gemm(B,detail::Format<T>::complex,s,a.data(),b.data(),host.data(),&report);compare<B>(host,expected,"host");
        // The fallback reads snapshots, even if the original inputs are overwritten later.
        if(wide&&!dense&&!broadcast&&!accumulate){out.upload(initial.data(),initial.size());auto batch=e.batch();auto ticket=unit.gemm(batch,s,ab,bb,out);
            batch.run(detail::Format<T>::complex?Operation::complex_add:Operation::add,ab,ab,ab);batch.submit().wait();compare<B>(download(out),expected,"fallback snapshot");}
    }
    // Generated input in the same batch, unit destruction and destructor wait.
    ExactGemm4Shape shape{3,16,16,16};std::vector<T> a(48),b(48),initial(48);for(auto* v:{&a,&b})for(auto& z:*v)z=random_value<B,T>(rng,4);
    auto ab=upload(e,a),bb=upload(e,b),out=upload(e,initial);auto expected=oracle<B>(shape,a,b,initial);
    {auto batch=e.batch();ExactGemm4Ticket ticket=[&]{ExactGemm4 temporary(e);return temporary.gemm(batch,shape,ab,bb,out);}();auto submission=batch.submit();}
    compare<B>(download(out),expected,"unit destruction/destructor wait");
    {auto generated=e.make_buffer<T>(48),zeros=upload(e,initial);auto batch=e.batch();
        batch.run(detail::Format<T>::complex?Operation::complex_add:Operation::add,ab,zeros,generated);
        auto ticket=unit.gemm(batch,shape,generated,bb,out);batch.submit().wait();need(!ticket.report().provisional_reads,"generated input provisional");}
    compare<B>(download(out),expected,"earlier device producer");
    expected=oracle<B>(shape,a,a,initial);
    {auto batch=e.batch();unit.gemm(batch,shape,ab,ab,out);batch.submit().wait();}
    compare<B>(download(out),expected,"identical input buffers");
}
template<int B> void boundaries(Engine& e,ExactGemm4& unit){using T=Float<B>;ExactGemm4Shape s{4,16,16,16};std::vector<T> a(64,zero<B/32>()),b(64,zero<B/32>()),out(64,zero<B/32>());
    auto one=from_decimal<B>("1");for(unsigned t=0;t<4;++t)for(unsigned k=0;k<4;++k)b[t*16+k*4]=one;
    a[0]=one;a[1]=one;a[1].exponent=-B; // half-ulp tie, even result
    a[16]=one;a[16].limb[0]|=1;a[17]=a[1]; // half-ulp tie, odd result
    a[32]=one;a[32].exponent=1000000000;b[32]=a[32]; // exponent overflow
    a[48]=one;a[48].exponent=-1000000000;b[48]=a[48]; // exponent underflow
    auto expected=oracle<B>(s,a,b,out);unit.gemm(B,false,s,a.data(),b.data(),out.data());compare<B>(out,expected,"ties/exponent limits");
    // Three widely separated clusters, cancellation reveals the lowest one.
    for(auto& v:a)v=zero<B/32>();for(unsigned t=0;t<4;++t){a[t*16]=one;a[t*16].exponent=10000;a[t*16+1]=negate(a[t*16]);a[t*16+2]=one;a[t*16+2].exponent=-10000;a[t*16+3]=one;a[t*16+3].exponent=-10001;
        for(unsigned k=0;k<4;++k)b[t*16+k*4]=one;}
    expected=oracle<B>(s,a,b,out);unit.gemm(B,false,s,a.data(),b.data(),out.data());compare<B>(out,expected,"wide-cluster cancellation");
}
static void validation(Engine& e,ExactGemm4& unit){using T=Float<352>;std::vector<T> v(48,from_decimal<352>("1"));auto a=upload(e,v),b=upload(e,v),c=upload(e,v);
    auto throws=[&](auto action){bool yes=false;try{action();}catch(const std::exception&){yes=true;}need(yes,"invalid call accepted");};
    throws([&]{auto batch=e.batch();unit.gemm(batch,{3,16,16,16},a,b,a);});
    throws([&]{auto batch=e.batch();unit.gemm(batch,{3,16,16,15},a,b,c);});
    throws([&]{auto batch=e.batch();unit.gemm(batch,{4,16,16,16},a,b,c);});
    throws([&]{auto batch=e.batch();unit.gemm(batch,{SIZE_MAX,16,16,16},a,b,c);});
    throws([&]{unit.gemm(353,false,{0},nullptr,nullptr,nullptr);});
    Engine foreign;auto other=upload(foreign,v);throws([&]{auto batch=e.batch();unit.gemm(batch,{3,16,16,16},other,b,c);});
    {auto batch=e.batch();unit.gemm(batch,{3,16,16,16},a,b,c);throws([&]{unit.gemm(batch,{3,16,16,16},a,b,c);});throws([&]{unit.gemm(batch,{3,16,16,16},c,b,a);});}
    {auto batch=e.batch();auto ticket=unit.gemm(batch,{3,16,16,16},a,b,c);auto submission=batch.submit();throws([&]{c.mapped();});throws([&]{auto next=e.batch();unit.gemm(next,{3,16,16,16},a,b,c);});submission.wait();need(ticket.resolved(),"busy ticket resolution");}
    {auto batch=e.batch();auto ticket=unit.gemm(batch,{3,16,16,16},a,b,c);auto unused=e.make_buffer<T>(48);batch.run(Operation::add,c,c,unused);batch.submit().wait();need(ticket.report().provisional_reads,"provisional read not reported");}
    {Buffer<T> empty;auto batch=e.batch();auto ticket=unit.gemm(batch,{0},empty,empty,empty);batch.submit().wait();need(ticket.report().gpu_components==0,"empty output");}
    std::cout<<"ownership/validation PASS\n";
}
template<int B> void run(Engine& e,ExactGemm4& unit,bool dense){cases<B,Float<B>>(e,unit,dense);cases<B,Complex<B/32>>(e,unit,dense);boundaries<B>(e,unit);std::cout<<B<<" bits real+complex exact4 host/resident "<<(dense?"dense3":"small")<<" cancellation/wide/status/ties/boundaries PASS\n"<<std::flush;}
int main(int argc,char** argv){try{bool all=false,dense=false;int bits=352;for(int i=1;i<argc;++i){std::string x=argv[i];if(x=="--all-widths")all=true;else if(x=="--dense")dense=true;else if(x=="--bits"&&i+1<argc)bits=std::stoi(argv[++i]);else throw std::invalid_argument("unknown option");}if(bits<64||bits>1024||bits%32)throw std::invalid_argument("unsupported precision");Engine e;ExactGemm4 unit(e);validation(e,unit);
    for(int width=64;width<=1024;width+=32)if(all||bits==width)switch(width/32){
#define CASE(N) case N:run<32*N>(e,unit,dense);break;
CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9) CASE(10) CASE(11) CASE(12) CASE(13) CASE(14) CASE(15) CASE(16) CASE(17) CASE(18) CASE(19) CASE(20) CASE(21) CASE(22) CASE(23) CASE(24) CASE(25) CASE(26) CASE(27) CASE(28) CASE(29) CASE(30) CASE(31) CASE(32)
#undef CASE
    }std::cout<<"PASS exact GEMM4 independent MPFR gate\n";return 0;}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
