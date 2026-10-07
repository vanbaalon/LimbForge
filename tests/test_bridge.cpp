// CPU-only checks of the MPFR/MPC bridge: every fast path must equal the mpz reference (detail::*_slow) bit for bit,
// including the MPFR inexact flag, under several MPFR exponent ranges, and from_mpfr must equal mpfr_set rounding.
#include "reference.hpp"
#include <stdexcept>
#include <utility>
using namespace limbforge;
static std::size_t checks=0;
static void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);++checks;}
// Strict identity: precision, kind, sign, exponent and every significand bit (including the bits below prec).
static bool same(mpfr_srcptr a,mpfr_srcptr b){
    if(mpfr_get_prec(a)!=mpfr_get_prec(b)||mpfr_custom_get_kind(a)!=mpfr_custom_get_kind(b))return false;
    return !mpfr_regular_p(a)||(mpfr_get_exp(a)==mpfr_get_exp(b)&&!std::memcmp(mpfr_custom_get_significand(a),mpfr_custom_get_significand(b),mpfr_custom_get_size(mpfr_get_prec(a))));
}
struct Custom { // mpfr_custom_init number with inline limbs, as in the qscmx host type
    std::vector<mp_limb_t> limbs;mpfr_t x;
    explicit Custom(mpfr_prec_t p):limbs(mpfr_custom_get_size(p)/sizeof(mp_limb_t),~mp_limb_t(0)){mpfr_custom_init(limbs.data(),p);mpfr_custom_init_set(x,MPFR_NAN_KIND,0,p,limbs.data());}
    Custom(const Custom&)=delete;
};
struct Range { mpfr_exp_t emin,emax;const char* name; };
struct SetRange {
    mpfr_exp_t emin=mpfr_get_emin(),emax=mpfr_get_emax();
    explicit SetRange(Range r){if(mpfr_set_emin(r.emin)||mpfr_set_emax(r.emax))throw std::runtime_error("exponent range unsupported");}
    ~SetRange(){mpfr_set_emin(emin);mpfr_set_emax(emax);}
};
static std::vector<Range> ranges;
struct Mpz { mpz_t z;Mpz(){mpz_init(z);}~Mpz(){mpz_clear(z);}Mpz(const Mpz&)=delete; };
struct Rand { gmp_randstate_t s;explicit Rand(unsigned long seed){gmp_randinit_default(s);gmp_randseed_ui(s,seed);}~Rand(){gmp_randclear(s);} };
// A p-bit integer (top bit set) whose rounding to `keep` bits is interesting: random, all ones (carry into the next
// binade), power of two, exact, ties with even and odd kept lsb, just above / below a tie, all-ones tie.
static void significand(mpz_t z,Rand& rand,long p,long keep,int pattern){
    mpz_urandomb(z,rand.s,p);mpz_setbit(z,p-1);long d=p-keep;
    if(pattern==1||pattern==7){mpz_set_ui(z,0);mpz_setbit(z,p);mpz_sub_ui(z,z,1);if(pattern==1)return;}
    if(pattern==6){mpz_set_ui(z,0);mpz_setbit(z,p-1);return;}
    if(d<=0||pattern==0)return;
    mpz_fdiv_q_2exp(z,z,d);mpz_mul_2exp(z,z,d);
    switch(pattern){case 2:if(d<p-1)mpz_clrbit(z,d);mpz_setbit(z,d-1);break;
    case 3:case 7:mpz_setbit(z,d);mpz_setbit(z,d-1);break;
    case 4:mpz_setbit(z,d-1);mpz_setbit(z,0);break;
    case 5:for(long i=0;i<d-1;++i)mpz_setbit(z,i);break;}
}
static std::vector<mpfr_prec_t> precisions(int bits){
    std::vector<mpfr_prec_t> list;
    for(long p:{long(MPFR_PREC_MIN),2L,24L,53L,63L,64L,65L,100L,bits-65L,bits-64L,bits-33L,bits-32L,bits-31L,bits-1L,long(bits),bits+1L,bits+31L,bits+32L,bits+33L,bits+63L,bits+64L,bits+65L,2L*bits+7,1100L,2100L})
        if(p>=MPFR_PREC_MIN&&std::find(list.begin(),list.end(),p)==list.end())list.push_back(p);
    return list;
}
// MPFR exponents near LimbForge's ±1e9 limit and near each tested range's emin / emax and the Bits scaling bound.
static mpfr_exp_t pick_exponent(std::mt19937_64& rng,int bits){
    static const long fixed[]={1000000001,1000000002,1000000000,999999999,-999999999,-1000000000,-1000000001,-998999999};
    switch(rng()%4){case 0:return long(rng()%601)-300;case 1:return fixed[rng()%8];
    case 2:{auto& r=ranges[rng()%ranges.size()];long base=(rng()&1)?r.emin:r.emax;return base+long(rng()%5)-2;}
    default:return bits+long(rng()%5)-2;}
}
static void garbage(mpfr_ptr x,Rand& rand,std::mt19937_64& rng){
    switch(rng()%4){case 0:mpfr_set_nan(x);break;case 1:mpfr_set_inf(x,-1);break;case 2:mpfr_set_zero(x,-1);break;
    default:mpfr_urandomb(x,rand.s);mpfr_mul_2si(x,x,long(rng()%200)-100,MPFR_RNDN);mpfr_neg(x,x,MPFR_RNDN);}
}
template<int Bits> static void test_from(Rand& rand,std::mt19937_64& rng){
    reference::MP rounded(Bits),back(Bits);Mpz z;
    for(mpfr_prec_t p:precisions(Bits)){
        reference::MP x(p);Custom c(p);
        for(int k=0;k<5;++k){ // specials
            if(k==0)mpfr_set_nan(x.x);else if(k<3)mpfr_set_inf(x.x,k==1?1:-1);else mpfr_set_zero(x.x,k==3?1:-1);
            require(reference::equal<Bits>(from_mpfr<Bits>(x.x),detail::from_mpfr_slow<Bits>(x.x)),"from special "+std::to_string(k));
        }
        for(auto range:ranges)for(int i=0;i<120;++i){
            mpfr_exp_t e=pick_exponent(rng,Bits);int pattern=int(rng()%9);
            {reference::ExponentRange wide;significand(z.z,rand,p,Bits,pattern);mpfr_set_z_2exp(x.x,z.z,e-p,MPFR_RNDN);if(rng()&1)mpfr_neg(x.x,x.x,MPFR_RNDN);mpfr_set(c.x,x.x,MPFR_RNDN);}
            SetRange set(range);std::string where=std::to_string(Bits)+" prec "+std::to_string(p)+" exp "+std::to_string(e)+" pattern "+std::to_string(pattern)+" range "+range.name;
            mpfr_clear_flags();auto slow=detail::from_mpfr_slow<Bits>(x.x);auto slow_flags=mpfr_flags_save();
            mpfr_clear_flags();auto fast=from_mpfr<Bits>(x.x);auto fast_flags=mpfr_flags_save();
            require(reference::equal<Bits>(fast,slow),"from_mpfr "+where);require(fast_flags==slow_flags,"from_mpfr flags "+where);
            require(reference::equal<Bits>(from_mpfr<Bits>(c.x),slow),"from_mpfr custom "+where);
            if(!fast.status&&mpfr_get_emin()<=e&&e<mpfr_get_emax()&&mpfr_get_emax()>Bits){ // independent: equals mpfr_set RN to Bits
                mpfr_set(rounded.x,x.x,MPFR_RNDN);detail::to_mpfr_slow<Bits>(back.x,fast);
                require(mpfr_equal_p(rounded.x,back.x)&&mpfr_signbit(rounded.x)==mpfr_signbit(back.x),"from_mpfr vs mpfr_set "+where);
            }
        }
    }
}
template<int Bits> static void test_to(Rand& rand,std::mt19937_64& rng){
    Mpz z;std::size_t count=0;
    for(mpfr_prec_t p:precisions(Bits)){
        reference::MP fast(p),slow(p);Custom c(p);
        {auto x=zero<Bits/32>();garbage(fast.x,rand,rng);to_mpfr<Bits>(fast.x,x);detail::to_mpfr_slow<Bits>(slow.x,x);require(same(fast.x,slow.x),"to zero");}
        for(word status:{word(division_by_zero),word(exponent_overflow),word(invalid)}){auto x=reference::random_number<Bits>(rng);x.status=status;
            bool a=false,b=false;try{to_mpfr<Bits>(fast.x,x);}catch(const std::runtime_error&){a=true;}try{detail::to_mpfr_slow<Bits>(slow.x,x);}catch(const std::runtime_error&){b=true;}
            require(a&&b,"to_mpfr status must throw");}
        for(auto range:ranges)for(int i=0;i<120;++i){
            int pattern=int(rng()%9);auto x=reference::random_number<Bits>(rng);significand(z.z,rand,Bits,p,pattern);
            for(auto& l:x.limb)l=0;std::size_t words=0;mpz_export(x.limb,&words,-1,sizeof(word),0,0,z.z);
            long e=long(pick_exponent(rng,Bits))-1;x.exponent=int(std::max(-1000000000L,std::min(1000000000L,e)));
            SetRange set(range);std::string where=std::to_string(Bits)+" prec "+std::to_string(p)+" exp "+std::to_string(x.exponent)+" pattern "+std::to_string(pattern)+" range "+range.name;
            garbage(fast.x,rand,rng);garbage(slow.x,rand,rng);garbage(c.x,rand,rng);
            mpfr_clear_flags();detail::to_mpfr_slow<Bits>(slow.x,x);auto slow_flags=mpfr_flags_save();
            mpfr_clear_flags();to_mpfr<Bits>(fast.x,x);auto fast_flags=mpfr_flags_save();to_mpfr<Bits>(c.x,x);
            require(same(fast.x,slow.x),"to_mpfr "+where);require(fast_flags==slow_flags,"to_mpfr flags "+where);require(same(c.x,slow.x),"to_mpfr custom "+where);
            if(p>=Bits&&mpfr_regular_p(fast.x)&&mpfr_get_exp(fast.x)==x.exponent+1){++count;require(reference::equal<Bits>(from_mpfr<Bits>(fast.x),x),"round trip "+where);}
        }
    }
    require(count>0,"no exact round trips");
}
struct Array { // mpfr_t[] with mixed precisions
    std::size_t n;mpfr_t* a;
    Array(std::size_t n,const std::vector<mpfr_prec_t>& precs,std::mt19937_64& rng):n(n),a(new mpfr_t[n]){for(std::size_t i=0;i<n;++i)mpfr_init2(a[i],precs[rng()%precs.size()]);}
    explicit Array(const Array& like,int):n(like.n),a(new mpfr_t[n]){for(std::size_t i=0;i<n;++i)mpfr_init2(a[i],mpfr_get_prec(like.a[i]));}
    ~Array(){for(std::size_t i=0;i<n;++i)mpfr_clear(a[i]);delete[] a;}
    Array(const Array&)=delete;
};
template<int Bits> static void test_arrays(Rand& rand,std::mt19937_64& rng){
    constexpr std::size_t n=17000; // threads=0 auto-threads from 512 bits (grain 2^22/Bits); threads=5 always splits
    std::vector<mpfr_prec_t> precs={mpfr_prec_t(Bits),mpfr_prec_t(Bits),mpfr_prec_t(Bits+64),mpfr_prec_t(Bits-31),53};
    Array in(n,precs,rng),out(n,precs,rng),ref(out,0);std::vector<Float<Bits>> lf(n),expect(n),values(n);
    {reference::ExponentRange wide;
        for(std::size_t i=0;i<n;++i){
            if(i%97==0)mpfr_set_nan(in.a[i]);else if(i%89==0)mpfr_set_inf(in.a[i],-1);else if(i%83==0)mpfr_set_zero(in.a[i],1);
            else{mpfr_urandomb(in.a[i],rand.s);if(mpfr_zero_p(in.a[i]))mpfr_set_ui(in.a[i],1,MPFR_RNDN);mpfr_mul_2si(in.a[i],in.a[i],pick_exponent(rng,Bits),MPFR_RNDN);if(rng()&1)mpfr_neg(in.a[i],in.a[i],MPFR_RNDN);}
            values[i]=reference::random_number<Bits>(rng);values[i].exponent=int(std::max(-1000000000L,std::min(1000000000L,long(pick_exponent(rng,Bits)))));
            if(i%101==0)values[i]=zero<Bits/32>();
        }}
    for(auto range:ranges){SetRange set(range); // workers must see the caller's exponent range
        for(std::size_t i=0;i<n;++i){expect[i]=detail::from_mpfr_slow<Bits>(in.a[i]);detail::to_mpfr_slow<Bits>(ref.a[i],values[i]);}
        for(unsigned threads:{0u,1u,5u}){
            from_mpfr_array<Bits>(in.a,lf.data(),n,threads);
            for(std::size_t i=0;i<n;++i)require(reference::equal<Bits>(lf[i],expect[i]),"from_mpfr_array "+std::to_string(Bits)+" "+range.name+" i="+std::to_string(i));
            for(std::size_t i=0;i<n;++i)garbage(out.a[i],rand,rng);
            to_mpfr_array<Bits>(out.a,values.data(),n,threads);
            for(std::size_t i=0;i<n;++i)require(same(out.a[i],ref.a[i]),"to_mpfr_array "+std::to_string(Bits)+" "+range.name+" i="+std::to_string(i));
        }
    }
    auto bad=values;bad[n/2].status=invalid;
    for(unsigned threads:{0u,1u,4u}){bool thrown=false;try{to_mpfr_array<Bits>(out.a,bad.data(),n,threads);}catch(const std::runtime_error&){thrown=true;}require(thrown,"to_mpfr_array status must throw");}
}
#if defined(LIMBFORGE_HAS_MPC)&&defined(LIMBFORGE_TEST_MPC)
template<int Bits> static void test_mpc(Rand& rand,std::mt19937_64& rng){
    using C=Complex<Bits/32>;constexpr std::size_t n=20000;auto precs=precisions(Bits);
    mpc_t* in=new mpc_t[n];mpc_t* out=new mpc_t[n];std::vector<C> lf(n),values(n);
    for(std::size_t i=0;i<n;++i){mpc_init3(in[i],precs[rng()%precs.size()],precs[rng()%precs.size()]);mpc_init3(out[i],precs[rng()%precs.size()],precs[rng()%precs.size()]);
        for(mpfr_ptr part:{mpc_realref(in[i]),mpc_imagref(in[i])}){mpfr_urandomb(part,rand.s);mpfr_mul_2si(part,part,long(rng()%2001)-1000,MPFR_RNDN);if(rng()&1)mpfr_neg(part,part,MPFR_RNDN);}
        if(i%53==0)mpfr_set_nan(mpc_imagref(in[i]));if(i%59==0)mpfr_set_zero(mpc_realref(in[i]),-1);
        values[i]={reference::random_number<Bits>(rng),reference::random_number<Bits>(rng)};if(i%61==0)values[i].im=zero<Bits/32>();}
    reference::MP re(Bits),im(Bits);
    for(std::size_t i=0;i<n;++i){
        C a=from_mpc<Bits>(in[i]);require(reference::equal<Bits>(a.re,detail::from_mpfr_slow<Bits>(mpc_realref(in[i])))&&reference::equal<Bits>(a.im,detail::from_mpfr_slow<Bits>(mpc_imagref(in[i]))),"from_mpc");
        to_mpc<Bits>(out[i],values[i]);
        reference::MP sr(mpfr_get_prec(mpc_realref(out[i]))),si(mpfr_get_prec(mpc_imagref(out[i])));detail::to_mpfr_slow<Bits>(sr.x,values[i].re);detail::to_mpfr_slow<Bits>(si.x,values[i].im);
        require(same(mpc_realref(out[i]),sr.x)&&same(mpc_imagref(out[i]),si.x),"to_mpc");
    }
    for(unsigned threads:{0u,1u,6u}){
        from_mpc_array<Bits>(in,lf.data(),n,threads);
        for(std::size_t i=0;i<n;++i)require(reference::equal_complex<Bits>(lf[i],from_mpc<Bits>(in[i])),"from_mpc_array");
        mpc_t* again=new mpc_t[n];for(std::size_t i=0;i<n;++i){mpc_init3(again[i],mpfr_get_prec(mpc_realref(out[i])),mpfr_get_prec(mpc_imagref(out[i])));mpc_set_nan(again[i]);}
        to_mpc_array<Bits>(again,values.data(),n,threads);
        for(std::size_t i=0;i<n;++i)require(same(mpc_realref(again[i]),mpc_realref(out[i]))&&same(mpc_imagref(again[i]),mpc_imagref(out[i])),"to_mpc_array");
        for(std::size_t i=0;i<n;++i)mpc_clear(again[i]);delete[] again;
    }
    C bad=values[1];bad.im.status=invalid;mpc_set_nan(out[0]);bool thrown=false;
    try{to_mpc<Bits>(out[0],bad);}catch(const std::runtime_error&){thrown=true;}
    require(thrown&&mpfr_nan_p(mpc_realref(out[0])),"to_mpc must throw before writing");
    for(std::size_t i=0;i<n;++i){mpc_clear(in[i]);mpc_clear(out[i]);}delete[] in;delete[] out;
}
#endif
template<int Bits> static void test_precision(){
    std::mt19937_64 rng(20261007+Bits);Rand rand(1307+Bits);std::size_t before=checks;
    test_from<Bits>(rand,rng);test_to<Bits>(rand,rng);test_arrays<Bits>(rand,rng);
#if defined(LIMBFORGE_HAS_MPC)&&defined(LIMBFORGE_TEST_MPC)
    test_mpc<Bits>(rand,rng);
#endif
    std::cout<<Bits<<" bits: "<<checks-before<<" bridge checks passed"<<std::endl;
}
template<int... I> static void test_all(std::integer_sequence<int,I...>){(test_precision<64+32*I>(),...);}
int main(){
    try{
        ranges={{mpfr_get_emin(),mpfr_get_emax(),"default"},{-3000000000L,3000000000L,"wide"},{-300,700,"narrow"},{-1100,1100,"medium"}};
#if defined(LIMBFORGE_HAS_MPC)&&defined(LIMBFORGE_TEST_MPC)
        std::cout<<"MPC "<<MPC_VERSION_STRING<<" checks enabled"<<std::endl;
#else
        std::cout<<"MPC not found: mpc_t checks skipped"<<std::endl;
#endif
        test_all(std::make_integer_sequence<int,31>{});
        std::cout<<checks<<" bridge checks passed at 31 precisions"<<std::endl;return 0;
    }catch(const std::exception& e){std::cerr<<"FAILED: "<<e.what()<<std::endl;return 1;}
}
