// Broadcast operands (plan C1): b (and c) read at (i / stride) % period, against MPFR.
#include "reference.hpp"
#include <stdexcept>
using namespace limbforge;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
std::size_t at(std::size_t i,Broadcast b){std::size_t j=i/b.stride;return b.period?j%b.period:j;}
template<int Bits> void run(Engine& e){
    using F=Float<Bits>;using C=Complex<Bits/32>;std::mt19937_64 rng(Bits+9);const std::size_t n=1001;
    for(Broadcast bb:{Broadcast{4,0},Broadcast{1,7},Broadcast{3,5},Broadcast{n,0}}){
        std::size_t m=broadcast_elements(n,bb);std::vector<F> a(n),b(m),out(n),res(n);for(auto& x:a)x=reference::random_number<Bits>(rng,9);for(auto& x:b)x=reference::random_number<Bits>(rng,9);
        e.run(Bits,Operation::mul,a.data(),b.data(),out.data(),n,bb);
        auto A=e.make_buffer<F>(n),B=e.make_buffer<F>(m),O=e.make_buffer<F>(n);A.upload(a.data(),n);B.upload(b.data(),m);
        {auto batch=e.batch();batch.run(Operation::div,A,B,O,bb);batch.submit().wait();}O.download(res.data(),n);
        for(std::size_t i=0;i<n;++i){require(reference::equal<Bits>(out[i],reference::real<Bits>(Operation::mul,a[i],b[at(i,bb)])),"broadcast mul bits="+std::to_string(Bits));
            require(reference::equal<Bits>(res[i],reference::real<Bits>(Operation::div,a[i],b[at(i,bb)])),"resident broadcast div bits="+std::to_string(Bits));}
        // Complex multiply with broadcast b; fused a*b+c with broadcast b and a differently broadcast c.
        Broadcast cb{2,3};std::size_t mc=broadcast_elements(n,cb);
        std::vector<C> x(n),y(m),z(mc),cout(n);for(auto& v:x)v={reference::random_number<Bits>(rng,5),reference::random_number<Bits>(rng,5)};
        for(auto& v:y)v={reference::random_number<Bits>(rng,5),reference::random_number<Bits>(rng,5)};for(auto& v:z)v={reference::random_number<Bits>(rng,8),reference::random_number<Bits>(rng,8)};
        e.run(Bits,Operation::complex_mul,x.data(),y.data(),cout.data(),n,bb);
        for(std::size_t i=0;i<n;++i)require(reference::equal_complex<Bits>(cout[i],reference::complex<Bits>(Operation::complex_mul,x[i],y[at(i,bb)])),"broadcast complex_mul bits="+std::to_string(Bits));
        e.run_ternary(Bits,Operation::complex_fma,x.data(),y.data(),z.data(),cout.data(),n,bb,cb);
        for(std::size_t i=0;i<n;++i)require(reference::equal_complex<Bits>(cout[i],reference::complex_fused<Bits>(x[i],y[at(i,bb)],z[at(i,cb)])),"broadcast complex_fma bits="+std::to_string(Bits));
        auto X=e.make_buffer<C>(n),Y=e.make_buffer<C>(m),Z=e.make_buffer<C>(mc),R=e.make_buffer<C>(n);X.upload(x.data(),n);Y.upload(y.data(),m);Z.upload(z.data(),mc);
        {auto batch=e.batch();batch.run(Operation::complex_fms,X,Y,Z,R,bb,cb);batch.submit().wait();}R.download(cout.data(),n);
        for(std::size_t i=0;i<n;++i)require(reference::equal_complex<Bits>(cout[i],reference::complex_fused<Bits>(x[i],y[at(i,bb)],z[at(i,cb)],true)),"resident broadcast complex_fms bits="+std::to_string(Bits));
    }
    bool rejected=false;try{auto A=e.make_buffer<F>(8),B=e.make_buffer<F>(1),O=e.make_buffer<F>(8);auto batch=e.batch();batch.run(Operation::add,A,B,O,Broadcast{4,0});}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"short broadcast operand accepted");
    std::cout<<Bits<<" bits: broadcast operands match MPFR"<<std::endl;
}
int main(){try{Engine e;run<64>(e);run<224>(e);run<256>(e);run<384>(e);run<1024>(e);std::cout<<"All broadcast checks passed."<<std::endl;return 0;}
catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}}
