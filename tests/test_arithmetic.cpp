#include "reference.hpp"
#include <stdexcept>
using namespace limbforge;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
void test_word_division(){
    std::mt19937_64 rng(20261007);constexpr dword B=dword(1)<<32;
    for(int i=0;i<1000000;++i){word d=word(rng())|0x80000000u;
        if(i%7==0)d=0x80000000u;if(i%11==0)d=0xffffffffu;
        word high=word(rng()%d),low=word(rng());
        if(i%13==0)high=d-1;if(i%17==0)high=0;if(i%19==0)low=0;if(i%23==0)low=0xffffffffu;
        auto result=divide_word(high,low,d,word(~dword(0)/d-B));dword numerator=(dword(high)<<32)|low;
        require(result.quotient==numerator/d&&result.remainder==numerator%d,"reciprocal quotient/remainder case="+std::to_string(i));
    }
    std::cout<<"1000000 exact reciprocal quotient/remainder checks passed"<<std::endl;
}
template<int Bits> void test_real(Engine* engine) {
    constexpr int N=Bits/32;using F=Float<Bits>;
    static_assert(sizeof(F)==Bits/8+12,"Metal struct layout");
    std::mt19937_64 rng(20261007+Bits);constexpr int count=4096;
    std::vector<F>a(count),b(count),gpu(count);
    for(int i=0;i<count;++i){a[i]=reference::random_number<Bits>(rng);b[i]=reference::random_number<Bits>(rng);
        // Exercise cancellation and every relevant alignment around limb / ulp boundaries.
        if(i%3==0){b[i].exponent=a[i].exponent-int(rng()%(Bits+80));}
        if(i%11==0){b[i]=negate(a[i]);b[i].limb[0]^=word(rng()&255);}
        if(i%17==0){a[i]=zero<N>();}
        if(i%23==0){b[i]=zero<N>();}
        if(i%101==0){a[i].status=invalid;}
    }
    F one=from_decimal<Bits>("1"),half_ulp=one;half_ulp.exponent=-Bits;
    a[1]=one;b[1]=half_ulp;a[2]=one;a[2].limb[0]|=1;b[2]=half_ulp;
    a[3]=one;b[3]=negate(half_ulp); // rounding at a binade boundary
    a[4]=one;b[4]=one;a[4].limb[0]=1;b[4]=negate(b[4]); // exact deep cancellation
    a[5]=one;a[5].exponent=1000000000;b[5]=from_decimal<Bits>("2");
    a[6]=one;a[6].exponent=-1000000000;b[6]=from_decimal<Bits>("0.5");
    a[7]=one;b[7]=one;for(auto& l:a[7].limb)l=0xffffffffu;for(auto& l:b[7].limb)l=0xffffffffu;
    // Exact products just below a binade boundary: rounding can rescue emin-1.
    mpz_t za,zb,target,remainder,half;mpz_inits(za,zb,target,remainder,half,nullptr);
    mpz_set_ui(target,1);mpz_mul_2exp(target,target,2*Bits-1);
    mpz_set_ui(half,1);mpz_mul_2exp(half,half,Bits-2);
    for(int fixture=8;fixture<40;++fixture){
        do {a[fixture]=reference::random_number<Bits>(rng,0);
            mpz_import(za,N,-1,4,0,0,a[fixture].limb);mpz_fdiv_qr(zb,remainder,target,za);
        }while(mpz_sgn(remainder)==0||mpz_cmp(remainder,half)>=0);
        b[fixture]=zero<N>();std::size_t exported;mpz_export(b[fixture].limb,&exported,-1,4,0,0,zb);
        a[fixture].sign=b[fixture].sign=fixture%2?-1:1;
        a[fixture].exponent=fixture%3? -1000000000:1000000000;
        b[fixture].exponent=fixture%3? -1:0;
    }
    mpz_clears(za,zb,target,remainder,half,nullptr);
    for(Operation op:{Operation::add,Operation::sub,Operation::mul,Operation::div}) {
        if(engine)engine->run(Bits,op,a.data(),b.data(),gpu.data(),count);
        for(int i=0;i<count;++i){auto expected=reference::real<Bits>(op,a[i],b[i]);F host;
            switch(op){case Operation::add:host=add(a[i],b[i]);break;case Operation::sub:host=sub(a[i],b[i]);break;
            case Operation::mul:host=mul(a[i],b[i]);break;default:host=div(a[i],b[i]);}
            std::string label="bits="+std::to_string(Bits)+" op="+std::to_string(int(op))+" case="+std::to_string(i);
            require(reference::equal<Bits>(host,expected),"CPU vs MPFR "+label);
            if(engine)require(reference::equal<Bits>(gpu[i],expected),"GPU vs MPFR "+label);
        }
    }
    std::cout<<Bits<<" bits: 16384 real MPFR comparisons passed"<<std::endl;
}
template<int Bits> void test_complex_and_recurrence(Engine* engine) {
    using C=Complex<Bits/32>;std::mt19937_64 rng(Bits);constexpr int count=257;
    std::vector<C>a(count),b(count),out(count);
    for(int i=0;i<count;++i){a[i]={reference::random_number<Bits>(rng),reference::random_number<Bits>(rng)};
        b[i]={reference::random_number<Bits>(rng),reference::random_number<Bits>(rng)};}
    b[0]={zero<Bits/32>(),zero<Bits/32>()};
    for(Operation op:{Operation::complex_add,Operation::complex_mul,Operation::complex_div}) {
        if(engine)engine->run(Bits,op,a.data(),b.data(),out.data(),count);
        for(int i=0;i<count;++i){
            auto expected=reference::complex<Bits>(op,a[i],b[i]);
            auto host=op==Operation::complex_add?cadd(a[i],b[i]):op==Operation::complex_mul?cmul(a[i],b[i]):cdiv(a[i],b[i]);
            require(reference::equal_complex<Bits>(host,expected),"CPU complex MPFR mismatch bits="+std::to_string(Bits));
            if(engine)require(reference::equal_complex<Bits>(out[i],expected),"GPU complex MPFR mismatch bits="+std::to_string(Bits));
        }
    }
    constexpr int batch=17,steps=31;
    std::vector<C>seed(4*batch),weight(4*batch*steps),result(batch),expected(batch);
    for(auto& z:seed)z={reference::random_number<Bits>(rng,5),reference::random_number<Bits>(rng,5)};
    for(auto& z:weight){z={reference::random_number<Bits>(rng,3),reference::random_number<Bits>(rng,3)};z.re.exponent-=5;z.im.exponent-=5;}
    if(engine)engine->recurrence(Bits,seed.data(),weight.data(),result.data(),batch,steps);
    for(int i=0;i<batch;++i){C ring[4],cpu_ring[4];for(int j=0;j<4;++j)ring[j]=cpu_ring[j]=seed[j*batch+i];int head=0;
        for(int t=0;t<steps;++t){C sum={zero<Bits/32>(),zero<Bits/32>()};for(int j=0;j<4;++j)
            sum=reference::complex<Bits>(Operation::complex_add,sum,reference::complex<Bits>(Operation::complex_mul,weight[(t*4+j)*batch+i],ring[(head+j)%4]));
            C cpu_sum={zero<Bits/32>(),zero<Bits/32>()};
            for(int j=0;j<4;++j)cpu_sum=cadd(cpu_sum,cmul(weight[(t*4+j)*batch+i],cpu_ring[(head+j)%4]));
            cpu_ring[head]=cpu_sum;ring[head]=sum;head=(head+1)%4;}
        expected[i]=ring[(head+3)%4];require(reference::equal_complex<Bits>(cpu_ring[(head+3)%4],expected[i]),"CPU recurrence MPFR mismatch");if(engine)require(reference::equal_complex<Bits>(result[i],expected[i]),"recurrence MPFR mismatch");
    }
    if(engine)engine->recurrence(Bits,seed.data(),nullptr,result.data(),batch,0);
    if(engine)for(int i=0;i<batch;++i)require(reference::equal_complex<Bits>(result[i],seed[3*batch+i]),"zero-step recurrence");
    // Replicated seeds must follow the same trajectory with one shared weight set.
    constexpr int lanes=4;std::vector<C> shared_seed(4*batch*lanes),shared_result(batch*lanes);
    for(int j=0;j<4;++j)for(int i=0;i<batch;++i)for(int a=0;a<lanes;++a)shared_seed[j*batch*lanes+i*lanes+a]=seed[j*batch+i];
    if(engine)engine->recurrence(Bits,shared_seed.data(),weight.data(),shared_result.data(),batch*lanes,steps,lanes);
    if(engine)for(int i=0;i<batch;++i)for(int a=0;a<lanes;++a)require(reference::equal_complex<Bits>(shared_result[i*lanes+a],expected[i]),"shared-weight recurrence");
    std::cout<<Bits<<" bits: complex arithmetic and fused recurrence passed"<<std::endl;
}
// Dense batches stress private GPU storage and close-exponent cancellation.
// In particular, a single-value check does not detect cross-thread corruption.
void test_dense_complex(Engine* engine){
    constexpr int Bits=1024;using C=Complex<32>;const int count=engine?65536:4096;
    std::mt19937_64 rng(20261007+Bits);std::vector<C>a(count),b(count),expected(count),out(count);
    for(int i=0;i<count;++i){a[i]={reference::random_number<Bits>(rng,5),reference::random_number<Bits>(rng,5)};
        b[i]={reference::random_number<Bits>(rng,5),reference::random_number<Bits>(rng,5)};}
    for(Operation op:{Operation::complex_add,Operation::complex_mul,Operation::complex_div}){
        for(int i=0;i<count;++i){expected[i]=reference::complex<Bits>(op,a[i],b[i]);
            auto host=op==Operation::complex_add?cadd(a[i],b[i]):op==Operation::complex_mul?cmul(a[i],b[i]):cdiv(a[i],b[i]);
            require(reference::equal_complex<Bits>(host,expected[i]),"dense CPU complex case="+std::to_string(i));}
        if(engine)for(int repeat=0;repeat<3;++repeat){engine->run(Bits,op,a.data(),b.data(),out.data(),count);
            for(int i=0;i<count;++i)require(reference::equal_complex<Bits>(out[i],expected[i]),
                "dense GPU complex op="+std::to_string(int(op))+" case="+std::to_string(i)+" repeat="+std::to_string(repeat));}
    }
    std::cout<<"1024 bits: dense complex batches passed ("<<count<<" values, three GPU repeats)"<<std::endl;
}
template<int Bits> void all_precisions(Engine* engine){test_real<Bits>(engine);if constexpr(Bits<1024)all_precisions<Bits+32>(engine);}
int main(int argc,char** argv) {
    try {
        bool cpu_only=argc==2&&std::string(argv[1])=="--cpu-only";
        if(argc>1&&!cpu_only)throw std::invalid_argument("usage: test_limbforge [--cpu-only]");
        std::unique_ptr<Engine> engine;
        if(!cpu_only){engine=std::make_unique<Engine>();std::cout<<"Device: "<<engine->device_name()<<std::endl;}
        else std::cout<<"CPU/MPFR validation (no GPU device created)"<<std::endl;
        test_word_division();
        all_precisions<64>(engine.get());
        test_complex_and_recurrence<128>(engine.get());test_complex_and_recurrence<384>(engine.get());test_complex_and_recurrence<1024>(engine.get());
        test_dense_complex(engine.get());
        if(engine){
            bool rejected=false;try{engine->run(80,Operation::add,nullptr,nullptr,nullptr,1);}catch(const std::invalid_argument&){rejected=true;}
            require(rejected,"invalid precision rejection");
            require(engine->run(256,Operation::add,nullptr,nullptr,nullptr,0).wall_seconds==0,"empty batch");
        }
        std::cout<<"All tests passed."<<std::endl;return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
