// Precision casts (plan S6): exact widening, RN-even narrowing, against MPFR rounding to the target width.
#include "reference.hpp"
#include <stdexcept>
using namespace limbforge;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
template<int From,int To> Float<To> expected(const Float<From>& x){
    if(x.status)return zero<To/32>(x.status);
    reference::ExponentRange range;reference::MP a(From),b(To);to_mpfr<From>(a.x,x);mpfr_set(b.x,a.x,MPFR_RNDN);return from_mpfr<To>(b.x);
}
template<int From,int To> void pair(Engine& e){
    std::mt19937_64 rng(From*7+To);const std::size_t n=777;std::vector<Float<From>> x(n);std::vector<Float<To>> y(n),z(n);
    for(auto& v:x)v=reference::random_number<From>(rng,100);
    x[0]=zero<From/32>();x[1].status=invalid;x[2].exponent=1000000000;x[3].exponent=-1000000000;
    if(To<From){// exact tie: keep only the bit just below the target ulp (odd and even targets), then all ones (rounds up into the next binade)
        for(int k=4;k<8;++k){auto& v=x[k];for(auto& l:v.limb)l=0;v.limb[From/32-1]=0x80000000u;int bit=From-To-1;v.limb[bit/32]|=1u<<(bit%32);if(k%2)v.limb[(From-To)/32]|=1u<<((From-To)%32);}
        for(auto& l:x[8].limb)l=~0u;x[8].exponent=1000000000;}
    e.cast(From,To,false,x.data(),y.data(),n);
    auto bx=e.make_buffer<Float<From>>(n);auto by=e.make_buffer<Float<To>>(n);bx.upload(x.data(),n);{auto b=e.batch();b.cast(bx,by);b.submit().wait();}by.download(z.data(),n);
    for(std::size_t i=0;i<n;++i){auto want=expected<From,To>(x[i]);std::string label=std::to_string(From)+"->"+std::to_string(To)+" i="+std::to_string(i);
        require(reference::equal<To>(y[i],want),"cast "+label);require(reference::equal<To>(z[i],want),"resident cast "+label);}
    // Complex: componentwise, host path.
    std::vector<Complex<From/32>> cx(n);std::vector<Complex<To/32>> cy(n);for(std::size_t i=0;i<n;++i)cx[i]={x[i],x[n-1-i]};e.cast(From,To,true,cx.data(),cy.data(),n);
    for(std::size_t i=0;i<n;++i)require(reference::equal<To>(cy[i].re,expected<From,To>(cx[i].re))&&reference::equal<To>(cy[i].im,expected<From,To>(cx[i].im)),"complex cast");
}
int main(){try{Engine e;
    pair<256,224>(e);pair<224,256>(e);pair<384,256>(e);pair<256,384>(e);pair<1024,64>(e);pair<64,1024>(e);pair<288,96>(e);pair<96,288>(e);pair<1024,992>(e);pair<256,256>(e);
    std::cout<<"All cast checks passed."<<std::endl;return 0;}
catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}}
