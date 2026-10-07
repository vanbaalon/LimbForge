#include "reference.hpp"
using namespace limbforge;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<class Fn>void rejects(Fn fn){bool rejected=false;try{fn();}catch(const std::exception&){rejected=true;}require(rejected,"invalid reduction accepted");}
template<int Bits,bool IsComplex=false>void check(Engine& e,std::size_t count,bool errors=false){
    using T=std::conditional_t<IsComplex,Complex<Bits/32>,Float<Bits>>;
    std::mt19937_64 rng(20261007+Bits+count);std::vector<T> input(count);
    for(auto& x:input){
        if constexpr(IsComplex)x={reference::random_number<Bits>(rng,5),reference::random_number<Bits>(rng,5)};
        else x=reference::random_number<Bits>(rng,5);
    }
    if(errors&&count>3){
        if constexpr(IsComplex){input[1].re=zero<Bits/32>(division_by_zero);input[count-1].re=zero<Bits/32>(exponent_overflow);}
        else {input[1]=zero<Bits/32>(division_by_zero);input[count-1]=zero<Bits/32>(exponent_overflow);}
    }
    auto expected=reference::tree_sum<Bits>(input);auto x=e.make_buffer<T>(count),out=e.make_buffer<T>(1);
    x.upload(input.data(),count);auto batch=e.batch();batch.tree_sum(x,out);auto pending=batch.submit();
    rejects([&]{out.mapped();});rejects([&]{x.mapped();});pending.wait();T actual;out.download(&actual,1);
    if constexpr(IsComplex)require(reference::equal_complex<Bits>(actual,expected),"complex tree sum differs from MPFR");
    else require(reference::equal<Bits>(actual,expected),"real tree sum differs from MPFR");
}
template<int Bits>void widths(Engine& e){
    for(auto n:{0u,1u,2u,3u,31u,32u,33u,257u,4097u})check<Bits>(e,n);
    check<Bits>(e,257,true);check<Bits,true>(e,257);check<Bits,true>(e,33,true);
    std::cout<<Bits<<"-bit reduction passed\n";
    if constexpr(Bits<1024)widths<Bits+32>(e);
}
void ordering_and_lifetime(Engine& e){
    using F=Float<384>;F one=from_decimal<384>("1"),big=one;big.exponent=386;auto negative=big;negative.sign=-1;
    // The specified tree rounds away each unit before opposite large values cancel.
    std::vector<F> input={big,one,negative,one};auto x=e.make_buffer<F>(4),out=e.make_buffer<F>(1);x.upload(input.data(),4);
    auto batch=e.batch();batch.tree_sum(x,out);batch.submit().wait();F actual;out.download(&actual,1);
    require(reference::equal<384>(actual,zero<12>()),"reduction changed its rounding order");
    auto expected=input;for(auto& v:expected)v=reference::real<384>(Operation::square,v,v);
    auto sum=reference::tree_sum<384>(expected);sum=reference::real<384>(Operation::sqrt,sum,sum);
    auto dependent=e.batch();dependent.run(Operation::square,x,x);dependent.tree_sum(x,out);dependent.run(Operation::sqrt,out,out);
    dependent.submit().wait();out.download(&actual,1);require(reference::equal<384>(actual,sum),"reduction dependency barrier failure");
    auto wrong=e.make_buffer<F>(2);rejects([&]{auto b=e.batch();b.tree_sum(x,wrong);});
    Engine other;auto foreign=other.make_buffer<F>(1);rejects([&]{auto b=e.batch();b.tree_sum(x,foreign);});
    out.upload(&one,1);auto copy=e.batch();copy.tree_sum(out,out);copy.submit().wait();out.download(&actual,1);
    require(reference::equal<384>(actual,one),"one-element alias failure");
    Submission pending;Buffer<F> survivor;
    {Engine temporary;auto source=temporary.make_buffer<F>(4);source.upload(input.data(),4);survivor=temporary.make_buffer<F>(1);
        auto b=temporary.batch();b.tree_sum(source,survivor);pending=b.submit();}
    pending.wait();survivor.download(&actual,1);require(reference::equal<384>(actual,zero<12>()),"reduction scratch lifetime failure");
}
int main(){try{Engine e;widths<64>(e);check<1024>(e,65537);check<1024,true>(e,65537);ordering_and_lifetime(e);
    std::cout<<"All reductions passed.\n";return 0;
}catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}}
