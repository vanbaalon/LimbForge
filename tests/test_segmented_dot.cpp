// Segmented dot products (plan D3) against an MPFR sequential fused chain.
#include "reference.hpp"
#include <stdexcept>
using namespace limbforge;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
template<int Bits,bool IsComplex> void run(Engine& e){
    using T=std::conditional_t<IsComplex,Complex<Bits/32>,Float<Bits>>;std::mt19937_64 rng(Bits*3+IsComplex);
    auto rnd=[&](int span){if constexpr(IsComplex)return T{reference::random_number<Bits>(rng,span),reference::random_number<Bits>(rng,span)};else return reference::random_number<Bits>(rng,span);};
    auto zero_t=[]{if constexpr(IsComplex)return T{zero<Bits/32>(),zero<Bits/32>()};else return zero<Bits/32>();};
    auto step=[](const T& a,const T& b,const T& s){if constexpr(IsComplex)return reference::complex_fused<Bits>(a,b,s);else return reference::fused<Bits>(a,b,s);};
    auto equal=[](const T& x,const T& y){if constexpr(IsComplex)return reference::equal_complex<Bits>(x,y);else return reference::equal<Bits>(x,y);};
    for(std::size_t K:{0ul,1ul,2ul,10ul,57ul,120ul})for(bool shared:{false,true}){
        const std::size_t S=33;SegmentedDot shape{S,K,shared};
        std::vector<T> a(S*K),b(shared?K:S*K),out(S),expected(S),resident(S);for(auto& x:a)x=rnd(20);for(auto& x:b)x=rnd(20);
        if(K>=2&&!shared){for(std::size_t k=0;k+1<K;k+=2){a[k+1]=a[k];if constexpr(IsComplex){b[k+1]={negate(b[k].re),negate(b[k].im)};}else b[k+1]=negate(b[k]);}} // segment 0: exact pair cancellation
        if(K>=1){if constexpr(IsComplex)a[K].re.status=invalid;else a[K].status=invalid;} // segment 1 (or later): status
        for(std::size_t i=0;i<S;++i){T s=zero_t();for(std::size_t k=0;k<K;++k)s=step(a[i*K+k],b[shared?k:i*K+k],s);expected[i]=s;}
        e.segmented_dot(Bits,IsComplex,shape,a.data(),b.data(),out.data());
        auto A=e.make_buffer<T>(a.size()),Bb=e.make_buffer<T>(b.size()),O=e.make_buffer<T>(S);A.upload(a.data(),a.size());Bb.upload(b.data(),b.size());
        auto batch=e.batch();batch.segmented_dot(shape,A,Bb,O);batch.submit().wait();O.download(resident.data(),S);
        for(std::size_t i=0;i<S;++i){std::string label=" bits="+std::to_string(Bits)+(IsComplex?" complex":" real")+" K="+std::to_string(K)+(shared?" shared":"")+" segment="+std::to_string(i);
            require(equal(out[i],expected[i]),"GPU segmented dot vs MPFR"+label);require(equal(resident[i],expected[i]),"resident segmented dot vs MPFR"+label);}
    }
    std::cout<<Bits<<" bits "<<(IsComplex?"complex":"real")<<": segmented dots match MPFR"<<std::endl;
}
int main(){try{Engine e;
    run<224,false>(e);run<224,true>(e);run<256,false>(e);run<256,true>(e);run<384,false>(e);run<384,true>(e);run<64,false>(e);run<64,true>(e);run<1024,false>(e);run<1024,true>(e);
    bool rejected=false;try{auto x=e.make_buffer<Float<64>>(3);auto b=e.batch();b.segmented_dot(SegmentedDot{2,2,false},x,x,x);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"undersized segmented dot buffer accepted");std::cout<<"All segmented dot checks passed."<<std::endl;return 0;}
catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}}
