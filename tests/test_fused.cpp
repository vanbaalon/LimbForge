// Fused multiply-add: real fma/fms and complex cfma/cfms, each component one rounding of the exact
// value. Independent references: mpfr_fma/mpfr_fms, and exact products plus mpfr_sum.
#include "reference.hpp"
#include <stdexcept>
using namespace limbforge;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
template<int N> Number<N> half_ulp(const Number<N>& x,int sign){Number<N> h=zero<N>();h.limb[N-1]=0x80000000u;h.sign=sign;h.exponent=x.exponent-32*N;return h;}
// Real cases: random spans, cancellation against the rounded product, far-apart terms, ties and
// near-ties, zeros, statuses, and exponents whose intermediate product leaves the supported range.
template<int Bits> void real_cases(std::vector<Float<Bits>>& a,std::vector<Float<Bits>>& b,std::vector<Float<Bits>>& c,std::mt19937_64& rng){
    constexpr int N=Bits/32;std::size_t n=a.size();
    for(std::size_t i=0;i<n;++i){a[i]=reference::random_number<Bits>(rng,40);b[i]=reference::random_number<Bits>(rng,40);c[i]=reference::random_number<Bits>(rng,80);
        switch(i%23){
        case 0:c[i]=negate(mul(a[i],b[i]));break;                                   // leaves the product's rounding error
        case 1:c[i]=negate(mul(a[i],b[i]));c[i].limb[0]^=word(rng()&15);break;       // deep partial cancellation
        case 2:c[i].exponent=a[i].exponent+b[i].exponent-int(rng()%(3*Bits+80));break; // addend near/below the window
        case 3:c[i].exponent=a[i].exponent+b[i].exponent+int(rng()%(2*Bits+80));break; // product near/below the addend
        case 4:a[i]=from_decimal<Bits>("1");c[i]=half_ulp(b[i],(rng()&1)?1:-1);break; // exact tie
        case 5:a[i]=from_decimal<Bits>("1");c[i]=half_ulp(b[i],(rng()&1)?1:-1);c[i].limb[0]|=1;break; // just past a tie
        case 6:a[i]=zero<N>();break;case 7:c[i]=zero<N>();break;case 8:b[i]=zero<N>();c[i]=zero<N>();break;
        case 9:c[i].status=invalid;break;
        case 10:a[i].exponent=600000000;b[i].exponent=600000000;c[i]=negate(mul(a[i],b[i]));break; // product overflow
        case 11:a[i].exponent=600000000;b[i].exponent=600000000;c[i].exponent=999999990;break;
        case 12:a[i].exponent=-600000000;b[i].exponent=-500000000;c[i].exponent=-999999999;break; // near emin
        case 13:{a[i].exponent=b[i].exponent=0;c[i]=negate(a[i]);c[i].exponent=b[i].exponent+a[i].exponent;break;}
        default:break;}
    }
}
Engine* gpu=nullptr;
template<int Bits> void test_real(){
    std::mt19937_64 rng(77+Bits);const std::size_t n=2300;std::vector<Float<Bits>>a(n),b(n),c(n),out(n);real_cases<Bits>(a,b,c,rng);
    for(bool subtract:{false,true}){
        if(gpu)gpu->run_ternary(Bits,subtract?Operation::fms:Operation::fma,a.data(),b.data(),c.data(),out.data(),n);
        for(std::size_t i=0;i<n;++i){
            auto expected=reference::fused<Bits>(a[i],b[i],c[i],subtract);auto host=subtract?fms(a[i],b[i],c[i]):fma(a[i],b[i],c[i]);
            std::string label=std::string(subtract?"fms":"fma")+" vs MPFR bits="+std::to_string(Bits)+" case="+std::to_string(i);
            require(reference::equal<Bits>(host,expected),"CPU "+label);if(gpu)require(reference::equal<Bits>(out[i],expected),"GPU "+label);}
    }
}
// Complex cases add exact and near cancellation between the two products, and a large product
// cancelled by the addend so that the other, tiny product decides the result.
template<int Bits> void test_complex(){
    using C=Complex<Bits/32>;std::mt19937_64 rng(91+Bits);const std::size_t n=1500;std::vector<C> as(n),bs(n),cs(n),out(n);
    for(std::size_t i=0;i<n;++i){
        C a={reference::random_number<Bits>(rng,30),reference::random_number<Bits>(rng,30)},b={reference::random_number<Bits>(rng,30),reference::random_number<Bits>(rng,30)},
          c={reference::random_number<Bits>(rng,60),reference::random_number<Bits>(rng,60)};
        switch(i%11){
        case 0:b={a.im,a.re};break;                                     // re: a.re*a.im - a.im*a.re = 0 exactly
        case 1:b={a.im,a.re};b.re.limb[0]^=1;break;                     // near-total cancellation
        case 2:a.re=from_decimal<Bits>("1");c.re=negate(b.re);a.im.exponent=b.im.exponent=-3*Bits;break; // tiny product exposed
        case 3:c.re=negate(cmul(a,b).re);c.im=negate(cmul(a,b).im);break;
        case 4:a.im.exponent-=int(rng()%(4*Bits));break;                // products far apart
        case 5:c.re.exponent-=int(rng()%(4*Bits));c.im.exponent+=int(rng()%(4*Bits));break;
        case 6:a.im=zero<Bits/32>();break;case 7:c=C{zero<Bits/32>(),zero<Bits/32>()};break;
        case 8:a.re=from_decimal<Bits>("1");a.im=zero<Bits/32>();c.re=half_ulp(b.re,-1);break; // tie through the 3-term path
        default:break;}
        as[i]=a;bs[i]=b;cs[i]=c;
    }
    for(bool subtract:{false,true}){
        if(gpu)gpu->run_ternary(Bits,subtract?Operation::complex_fms:Operation::complex_fma,as.data(),bs.data(),cs.data(),out.data(),n);
        for(std::size_t i=0;i<n;++i){auto expected=reference::complex_fused<Bits>(as[i],bs[i],cs[i],subtract);auto host=subtract?cfms(as[i],bs[i],cs[i]):cfma(as[i],bs[i],cs[i]);
            std::string label=std::string(subtract?"cfms":"cfma")+" vs MPFR bits="+std::to_string(Bits)+" case="+std::to_string(i);
            require(reference::equal_complex<Bits>(host,expected),"CPU "+label);if(gpu)require(reference::equal_complex<Bits>(out[i],expected),"GPU "+label);}
    }
}
// Resident chain: x <- x*y + x in place (out aliases an input), then x*y - z; one command buffer.
template<int Bits> void test_resident(){
    using C=Complex<Bits/32>;std::mt19937_64 rng(5+Bits);constexpr std::size_t n=257;std::vector<C> a(n),b(n),z(n),out(n),expected(n);
    for(std::size_t i=0;i<n;++i){a[i]={reference::random_number<Bits>(rng,3),reference::random_number<Bits>(rng,3)};b[i]={reference::random_number<Bits>(rng,0),reference::random_number<Bits>(rng,0)};
        b[i].re.exponent-=2;b[i].im.exponent-=2;z[i]={reference::random_number<Bits>(rng,3),reference::random_number<Bits>(rng,3)};expected[i]=a[i];}
    auto x=gpu->make_buffer<C>(n),y=gpu->make_buffer<C>(n),w=gpu->make_buffer<C>(n),r=gpu->make_buffer<C>(n);x.upload(a.data(),n);y.upload(b.data(),n);w.upload(z.data(),n);
    auto batch=gpu->batch();for(int s=0;s<8;++s)batch.run(Operation::complex_fma,x,y,x,x);batch.run(Operation::complex_fms,x,y,w,r);batch.submit().wait();r.download(out.data(),n);
    for(std::size_t i=0;i<n;++i){for(int s=0;s<8;++s)expected[i]=reference::complex_fused<Bits>(expected[i],b[i],expected[i]);
        require(reference::equal_complex<Bits>(out[i],reference::complex_fused<Bits>(expected[i],b[i],z[i],true)),"resident fused chain bits="+std::to_string(Bits)+" case="+std::to_string(i));}
}
template<int Bits> void all(){test_real<Bits>();test_complex<Bits>();if(gpu)test_resident<Bits>();if constexpr(Bits<1024)all<Bits+32>();}
int main(int argc,char** argv){try{
    bool cpu_only=argc==2&&std::string(argv[1])=="--cpu-only";if(argc>1&&!cpu_only)throw std::invalid_argument("usage: test_limbforge_fused [--cpu-only]");
    std::unique_ptr<Engine> engine;if(!cpu_only){engine=std::make_unique<Engine>();gpu=engine.get();}
    all<64>();std::cout<<"fused fma/fms/cfma/cfms: all 31 precisions match MPFR on the CPU"<<(gpu?", GPU, and resident chains":"")<<"\n";return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
