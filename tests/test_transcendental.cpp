// Transcendental functions (plan D7) against MPFR/MPC: correct rounding of exp, expm1, log, log1p, sin, cos, atan2
// (mpfr_*) and complex exp/log (mpc_exp/mpc_log, or an MPFR evaluation of each exact component), bit for bit, on the
// GPU and the CPU path; powi against an MPFR replay of its documented fused sequence (bit for bit) and mpc_pow_si (ulp
// statistics). White-box check of the error bounds of the W-word approximations against MPFR at 32W+256 bits.
// GPU retry rungs (round 44): --force-level k makes GPU levels below k (0 = first pass, 1-3 = rungs) treat every element as
// undecided, so rung k evaluates every element (k beyond the last GPU rung: the host final step); the report must account
// for every element. Run with k = 1, 2, 3 over all widths to validate the rung kernels at their working widths.
// Usage: test_limbforge_transcendental [--cpu-only|--gpu-only] [--points n] [--cpu-points n] [--bound-points n] [--bits b ...]
//        [--all-widths n] [--no-hard] [--no-bounds] [--verbose-bounds] [--seed s] [--force-level k] [--retry-threshold t]
// The GPU object uses gpu_retry_threshold = 0 (GPU rungs for every retry) unless --retry-threshold is given; the default
// threshold (host ladder for few retries, GPU rungs above it) is checked separately at 256 bits (check_threshold).
#include "reference.hpp"
#include "limbforge/transcendental.hpp"
#include "transcendental_core.hpp"
#ifdef LIMBFORGE_TEST_MPC
#include <mpc.h>
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <thread>
using namespace limbforge;
namespace tr=limbforge::transcendental;
namespace {
struct Wide { // the widest MPFR exponent range: the reference never overflows where WolfNum's range check decides
    mpfr_exp_t emin=mpfr_get_emin(),emax=mpfr_get_emax();
    Wide(){mpfr_set_emin(mpfr_get_emin_min());mpfr_set_emax(mpfr_get_emax_max());}
    ~Wide(){mpfr_set_emin(emin);mpfr_set_emax(emax);}
};
template<class F> void parallel(std::size_t n,F f){
    unsigned t=std::max(1u,std::min(18u,std::thread::hardware_concurrency()));if(n<256)t=1;
    std::vector<std::thread> w;for(unsigned i=1;i<t;++i)w.emplace_back([&,i]{Wide r;f(n*i/t,n*(i+1)/t);mpfr_free_cache2(MPFR_FREE_LOCAL_CACHE);});
    {Wide r;f(std::size_t(0),n/t);}for(auto& x:w)x.join();
}
const char* name(Function f){static const char* n[]={"exp","expm1","log","log1p","sin","cos","atan2","complex_exp","complex_log","complex_powi"};return n[int(f)];}
// MPFR value -> WolfNum with the contract's range rule (MPFR infinities and underflows to zero: exponent_overflow).
template<int Bits> Float<Bits> convert(mpfr_srcptr z,bool exact_zero_possible=false){
    if(mpfr_inf_p(z))return zero<Bits/32>(exponent_overflow);
    if(mpfr_zero_p(z)&&!exact_zero_possible)return zero<Bits/32>(exponent_overflow);
    return from_mpfr<Bits>(z);
}
// Correctly rounded reference of a real function.
template<int Bits> Float<Bits> ref_real(Function f,const Float<Bits>& a,const Float<Bits>& b){
    constexpr int N=Bits/32;word st=a.status|(f==Function::atan2?b.status:0);if(st)return zero<N>(st);
    reference::MP x(Bits),y(Bits),z(Bits);to_mpfr<Bits>(x.x,a);if(f==Function::atan2)to_mpfr<Bits>(y.x,b);
    // |x| >= 2^31: e^x lies beyond 2^(+-3e9) (decided here: MPFR would need ~|x| bits to reduce the argument).
    if((f==Function::exp||f==Function::expm1)&&a.sign&&a.exponent>=31){
        if(a.sign>0||f==Function::exp)return zero<N>(exponent_overflow);return tr::unit<N>(0,-1);}
    switch(f){
    case Function::exp:mpfr_exp(z.x,x.x,MPFR_RNDN);return convert<Bits>(z.x);
    case Function::expm1:if(!a.sign)return zero<N>();mpfr_expm1(z.x,x.x,MPFR_RNDN);return convert<Bits>(z.x);
    case Function::log:if(a.sign<=0)return zero<N>(invalid);mpfr_log(z.x,x.x,MPFR_RNDN);return convert<Bits>(z.x,true);
    case Function::log1p:{if(a.sign<0&&mpfr_cmp_si(x.x,-1)<=0)return zero<N>(invalid);mpfr_log1p(z.x,x.x,MPFR_RNDN);return convert<Bits>(z.x,true);}
    case Function::sin:if(a.sign&&a.exponent>=tr::TRIG_EMAX)return zero<N>(invalid);mpfr_sin(z.x,x.x,MPFR_RNDN);return convert<Bits>(z.x,!a.sign);
    case Function::cos:if(a.sign&&a.exponent>=tr::TRIG_EMAX)return zero<N>(invalid);mpfr_cos(z.x,x.x,MPFR_RNDN);return convert<Bits>(z.x);
    case Function::atan2:if(!a.sign&&!b.sign)return zero<N>(invalid);mpfr_atan2(z.x,x.x,y.x,MPFR_RNDN);return convert<Bits>(z.x,!a.sign);
    default:throw std::logic_error("ref_real");}
}
// Correctly rounded components of exp(a) cos(b), exp(a) sin(b) by an MPFR Ziv loop (relative error <= 3 2^-P at
// precision P, mpfr_can_round decides). Used for complex exp when MPC is unavailable  or a part outside [2^-64, 2^8) (mpc_exp raises its
// working precision with the exponents of the parts: re ~ 2^30, re ~ 2^-1e9 or im ~ 2^8000 need gigabytes).
template<int Bits> Float<Bits> ziv_exp_trig(const Float<Bits>& a,const Float<Bits>& b,bool imag){
    for(int P=4*Bits;;P*=2){if(P>(1<<24))throw std::runtime_error("ziv_exp_trig: no decision");
        reference::MP x(Bits),y(Bits),e(P),c(P),z(Bits);to_mpfr<Bits>(x.x,a);to_mpfr<Bits>(y.x,b);
        mpfr_exp(e.x,x.x,MPFR_RNDN);if(imag)mpfr_sin(c.x,y.x,MPFR_RNDN);else mpfr_cos(c.x,y.x,MPFR_RNDN);mpfr_mul(e.x,e.x,c.x,MPFR_RNDN);
        if(mpfr_inf_p(e.x)||mpfr_zero_p(e.x)||mpfr_can_round(e.x,P-2,MPFR_RNDN,MPFR_RNDN,Bits)){mpfr_set(z.x,e.x,MPFR_RNDN);return convert<Bits>(z.x);}
    }
}
// Correctly rounded components of complex exp/log: mpc_exp / mpc_log (complex exp with |re| >= 2^8, or without MPC:
// ziv_exp_trig; complex log without MPC: hypot/log and atan2 at 8x precision).
template<int Bits> Complex<Bits/32> ref_complex(Function f,const Complex<Bits/32>& w){
    constexpr int N=Bits/32;word st=w.re.status|w.im.status;if(st)return {zero<N>(st),zero<N>(st)};
    if(f==Function::complex_exp){
        if(w.im.sign&&w.im.exponent>=tr::TRIG_EMAX)return {zero<N>(invalid),zero<N>(invalid)};
        if(w.re.sign&&w.re.exponent>=31&&w.im.sign)return {zero<N>(exponent_overflow),zero<N>(exponent_overflow)};
        if(!w.im.sign)return {ref_real<Bits>(Function::exp,w.re,w.re),zero<N>()};
#ifdef LIMBFORGE_TEST_MPC
        auto moderate=[](const Float<Bits>& v){return !v.sign||(v.exponent>=-64&&v.exponent<8);};
        if(!moderate(w.re)||!moderate(w.im))
#endif
        return {ziv_exp_trig<Bits>(w.re,w.im,false),ziv_exp_trig<Bits>(w.re,w.im,true)};
    }else if(!w.re.sign&&!w.im.sign)return {zero<N>(invalid),zero<N>(invalid)};
#ifdef LIMBFORGE_TEST_MPC
    mpc_t x,z;mpc_init2(x,Bits);mpc_init2(z,Bits);to_mpc<Bits>(x,w);
    if(f==Function::complex_exp)mpc_exp(z,x,MPC_RNDNN);else mpc_log(z,x,MPC_RNDNN);
    Complex<N> r;
    if(f==Function::complex_exp){r.re=convert<Bits>(mpc_realref(z));r.im=convert<Bits>(mpc_imagref(z));}
    else{r.re=convert<Bits>(mpc_realref(z),true);r.im=convert<Bits>(mpc_imagref(z),true);}
    mpc_clear(x);mpc_clear(z);return r;
#else
    reference::MP a(Bits),b(Bits),p(8*Bits),q(8*Bits),z(Bits);to_mpfr<Bits>(a.x,w.re);to_mpfr<Bits>(b.x,w.im);Complex<N> r;
    mpfr_hypot(p.x,a.x,b.x,MPFR_RNDN);mpfr_log(p.x,p.x,MPFR_RNDN);mpfr_set(z.x,p.x,MPFR_RNDN);r.re=convert<Bits>(z.x,true);
    mpfr_atan2(q.x,b.x,a.x,MPFR_RNDN);mpfr_set(z.x,q.x,MPFR_RNDN);r.im=convert<Bits>(z.x,true);
    return r;
#endif
}
// MPFR replay of tr::cpowi (fused products, binary powering, reciprocal), with WolfNum's range check after each
// rounded component and status propagation (a status in any input component reaches both output components).
template<int Bits> struct Replay {
    using C=Complex<Bits/32>;static constexpr int N=Bits/32;
    static C prod(const C& a,const C& b){
        word st=a.re.status|a.im.status|b.re.status|b.im.status;if(st)return {zero<N>(st),zero<N>(st)};
        reference::MP ar(Bits),ai(Bits),br(Bits),bi(Bits),z(Bits);to_mpfr<Bits>(ar.x,a.re);to_mpfr<Bits>(ai.x,a.im);to_mpfr<Bits>(br.x,b.re);to_mpfr<Bits>(bi.x,b.im);
        C r;mpfr_fmms(z.x,ar.x,br.x,ai.x,bi.x,MPFR_RNDN);r.re=convert<Bits>(z.x,true);
        mpfr_fmma(z.x,ar.x,bi.x,ai.x,br.x,MPFR_RNDN);r.im=convert<Bits>(z.x,true);return r;
    }
    static C reciprocal(const C& w){
        word st=w.re.status|w.im.status;if(st)return {zero<N>(st),zero<N>(st)};
        if(!w.re.sign&&!w.im.sign)return {zero<N>(division_by_zero),zero<N>(division_by_zero)};
        reference::MP a(Bits),b(Bits),d(Bits),z(Bits);to_mpfr<Bits>(a.x,w.re);to_mpfr<Bits>(b.x,w.im);
        mpfr_fmma(d.x,a.x,a.x,b.x,b.x,MPFR_RNDN);C r;
        mpfr_div(z.x,a.x,d.x,MPFR_RNDN);r.re=convert<Bits>(z.x,!w.re.sign);
        mpfr_neg(b.x,b.x,MPFR_RNDN);mpfr_div(z.x,b.x,d.x,MPFR_RNDN);r.im=convert<Bits>(z.x,!w.im.sign);return r;
    }
    static C power(const C& z,int k){
        word st=z.re.status|z.im.status;if(st)return {zero<N>(st),zero<N>(st)};
        if(!k){C one;one.re=tr::unit<N>();one.im=zero<N>();return one;}
        std::uint32_t n=k<0?std::uint32_t(-(std::int64_t(k))):std::uint32_t(k);C p=z,acc=z;bool have=false;
        for(;;){if(n&1){acc=have?prod(acc,p):p;have=true;}n>>=1;if(!n)break;p=prod(p,p);}
        return k<0?reciprocal(acc):acc;
    }
};
// ---- inputs ----
std::mt19937_64 rng_global;
template<int Bits> Float<Bits> random_value(std::mt19937_64& g,int emin,int emax,int sign=0){
    Float<Bits> x=zero<Bits/32>();for(auto& l:x.limb)l=std::uint32_t(g());x.limb[Bits/32-1]|=0x80000000u;
    // Some short significands (exact small dyadics, frequent in applications).
    if(g()%8==0){int keep=int(g()%40);for(int i=0;i<Bits/32;++i){int lo=32*i,hi=lo+31;int cut=Bits-keep;if(hi<cut)x.limb[i]=0;else if(lo<cut)x.limb[i]&=~((std::uint32_t(1)<<(cut-lo))-1);}x.limb[Bits/32-1]|=0x80000000u;}
    x.exponent=emin+int(g()%std::uint64_t(emax-emin+1));x.sign=sign?sign:(g()&1?1:-1);return x;
}
template<int Bits> Float<Bits> from_double(double v){reference::MP m(Bits);mpfr_set_d(m.x,v,MPFR_RNDN);return from_mpfr<Bits>(m.x);}
template<int Bits> Float<Bits> pow2(long e,int sign=1){Float<Bits> x=tr::unit<Bits/32>(int(e),sign);return x;}
// x * (1 + j ulp) neighbours.
template<int Bits> Float<Bits> step(Float<Bits> x,int j){
    constexpr int N=Bits/32;if(!x.sign)return x;
    reference::MP m(Bits),u(Bits);to_mpfr<Bits>(m.x,x);for(int i=0;i<std::abs(j);++i){if(j>0)mpfr_nextabove(m.x);else mpfr_nextbelow(m.x);}
    (void)N;(void)u;return from_mpfr<Bits>(m.x);
}
// Hard and boundary inputs per function (deterministic). Values whose results lie within 2^-(32N+...) ulp of a
// rounding boundary: exp(+-2^-32N), expm1(2^(1-32N)), log1p(2^-32N), cos(2^-16N), complex log of (1, 2^-16N), ...
template<int Bits> std::vector<Float<Bits>> hard_real(Function f){
    constexpr int N=Bits/32;std::vector<Float<Bits>> v;Wide wide;
    auto both=[&](Float<Bits> x){v.push_back(x);x.sign=-x.sign;v.push_back(x);};
    for(long e:{0L,-1L,-2L,-5L,-31L,-32L*N,-32L*N-1,-32L*N+1,1L-32L*N,-16L*N,-16L*N-1,-16L*N-2,-32L*N-2,-32L*N-3,-32L*N-10,-64L*N,-1000L,-1000000L,-1000000000L,1L,2L,5L,10L,20L,29L,30L,31L,40L,1000L,8191L,8192L,1000000000L})
        both(pow2<Bits>(e));
    for(int j=-3;j<=3;++j){both(step<Bits>(pow2<Bits>(0),j));both(step<Bits>(pow2<Bits>(-32*N),j));both(step<Bits>(pow2<Bits>(-16*N),j));both(step<Bits>(pow2<Bits>(1-32*N),j));}
    reference::MP c(Bits+64),t(Bits);
    if(f==Function::exp||f==Function::expm1){
        // Near multiples of ln 2 (reduction cancellation), overflow and underflow thresholds.
        for(long k:{1L,2L,3L,7L,100L,1000L,123456L,1L<<20,1L<<28,(1L<<31)-1}){mpfr_const_log2(c.x,MPFR_RNDN);mpfr_mul_si(c.x,c.x,k,MPFR_RNDN);mpfr_set(t.x,c.x,MPFR_RNDN);
            auto x=from_mpfr<Bits>(t.x);for(int j=-2;j<=2;++j)both(step<Bits>(x,j));}
        for(double s:{1000000000.0,1000000001.0,1000000002.0,-1000000000.0,-1000000001.0,-999999999.0}){
            mpfr_const_log2(c.x,MPFR_RNDN);mpfr_mul_d(c.x,c.x,s,MPFR_RNDN);mpfr_set(t.x,c.x,MPFR_RNDN);auto x=from_mpfr<Bits>(t.x);for(int j=-2;j<=2;++j)v.push_back(step<Bits>(x,j));}
        for(double d:{-23.0*(N+2)-3,-23.0*(N+2)-4,-0.5,0.5,0.3465735902799726,-0.3465735902799726,0.6931471805599453})both(from_double<Bits>(d));
    }
    if(f==Function::log||f==Function::log1p){
        for(double d:{0.70710678118654752,1.41421356237309505,0.25,0.75,1.25,0.2,0.4142135623730950,0.2928932188134524,0.29289321881345254,0.999,1.001,2.0,3.0,0.5,-0.5,-0.75,-0.999999,-0.25})v.push_back(from_double<Bits>(d));
        mpfr_const_euler(c.x,MPFR_RNDN);mpfr_set(t.x,c.x,MPFR_RNDN);v.push_back(from_mpfr<Bits>(t.x));
        for(int j=1;j<=4;++j){v.push_back(step<Bits>(pow2<Bits>(0),j));v.push_back(step<Bits>(pow2<Bits>(0),-j));v.push_back(step<Bits>(pow2<Bits>(-1,-1),j));}
        v.push_back(pow2<Bits>(0,-1));v.push_back(zero<N>());v.push_back(zero<N>(division_by_zero));
    }
    if(f==Function::sin||f==Function::cos){
        // Near multiples of pi/2 (reduction cancellation), large arguments, the pi/4 switch, the domain limit.
        for(long k:{1L,2L,3L,4L,5L,100L,355L,710L,1L<<20,103993L,833719L,(1L<<40)+1}){mpfr_const_pi(c.x,MPFR_RNDN);mpfr_mul_si(c.x,c.x,k,MPFR_RNDN);mpfr_div_2ui(c.x,c.x,1,MPFR_RNDN);
            mpfr_set(t.x,c.x,MPFR_RNDN);auto x=from_mpfr<Bits>(t.x);for(int j=-2;j<=2;++j)both(step<Bits>(x,j));}
        mpfr_const_pi(c.x,MPFR_RNDN);mpfr_div_2ui(c.x,c.x,2,MPFR_RNDN);mpfr_set(t.x,c.x,MPFR_RNDN);{auto x=from_mpfr<Bits>(t.x);for(int j=-3;j<=3;++j)both(step<Bits>(x,j));}
        for(long e:{100L,1000L,4000L,8000L,8190L,8191L})both(step<Bits>(pow2<Bits>(e),1));
        // Integers and the largest mantissa just below the domain limit.
        auto top=pow2<Bits>(8191);for(auto& l:top.limb)l=0xffffffffu;both(top);
    }
    return v;
}
template<int Bits> std::vector<std::pair<Float<Bits>,Float<Bits>>> hard_pairs(){ // (y, x) for atan2, (re, im) for complex
    constexpr int N=Bits/32;std::vector<std::pair<Float<Bits>,Float<Bits>>> v;auto z=zero<N>();
    std::vector<Float<Bits>> s={z,pow2<Bits>(0),pow2<Bits>(0,-1),pow2<Bits>(-16*N),pow2<Bits>(-16*N,-1),pow2<Bits>(-32*N),pow2<Bits>(-1),pow2<Bits>(-2,-1),
        step<Bits>(pow2<Bits>(0),1),step<Bits>(pow2<Bits>(0),-1),pow2<Bits>(1000),pow2<Bits>(-1000,-1),pow2<Bits>(1000000000),pow2<Bits>(-1000000000),pow2<Bits>(-999999999,-1),
        from_double<Bits>(0.6),from_double<Bits>(0.8),from_double<Bits>(-0.6),from_double<Bits>(3.0),from_double<Bits>(-4.0),pow2<Bits>(30),pow2<Bits>(31,-1),pow2<Bits>(8191),
        zero<N>(invalid)};
    for(auto& a:s)for(auto& b:s)v.push_back({a,b});
    return v;
}
// ---- comparisons ----
struct Tally { std::size_t points=0,mismatches=0,retried=0,unresolved=0;std::size_t resolved[4]={};double gpu=0,wall=0; };
int forced=0;std::map<int,std::string> rung_info; // bits -> GPU rung words per function
std::map<int,std::array<std::size_t,5>> width_tally; // bits -> retried, GPU rungs 1-3, host final step (all functions)
// Report consistency: every retried element is resolved at exactly one rung (or unresolved); under --force-level k every
// element is retried and none is resolved below rung k.
void check_report(const TranscendentalReport& r,std::size_t n,int bits,Function f,bool gpu);
template<int Bits> bool same(const Float<Bits>& a,const Float<Bits>& b){return reference::equal<Bits>(a,b);}
template<int Bits> std::string show(const Float<Bits>& x){
    if(x.status)return "status "+std::to_string(x.status);if(!x.sign)return "0";
    if((x.sign!=1&&x.sign!=-1)||!(x.limb[Bits/32-1]>>31)||x.exponent>(1<<30)||x.exponent< -(1<<30)) // not a valid value (e.g. a miscompiled kernel)
        return "invalid number (sign "+std::to_string(x.sign)+", exponent "+std::to_string(x.exponent)+", top limb "+std::to_string(x.limb[Bits/32-1])+")";
    reference::MP m(Bits);to_mpfr<Bits>(m.x,x);char buf[400];mpfr_snprintf(buf,sizeof buf,"%.40Re (e=%d)",m.x,x.exponent);return buf;
}
int failures=0;bool verbose_bounds=false;
void fail_header(const char* what,int bits,Function f){std::printf("FAIL %s %d-bit %s\n",what,bits,name(f));++failures;}
void check_report(const TranscendentalReport& r,std::size_t n,int bits,Function f,bool gpu){
    std::size_t sum=r.unresolved;for(int k=0;k<4;++k)sum+=r.resolved[k];bool good=sum==r.retried&&r.count==n&&(gpu||!r.resolved[3]);
    if(gpu&&forced>0){good=good&&r.retried==n;for(int k=0;k<forced-1&&k<3;++k)good=good&&!r.resolved[k];}
    if(!good){fail_header("report",bits,f);std::printf("  count %zu retried %zu resolved %zu/%zu/%zu/%zu unresolved %zu (n %zu, forced %d)\n",r.count,r.retried,
        r.resolved[0],r.resolved[1],r.resolved[2],r.resolved[3],r.unresolved,n,forced);}
    if(gpu){auto& w=width_tally[bits];w[0]+=r.retried;for(int k=0;k<3;++k)w[k+1]+=r.resolved[k];w[4]+=r.host_retried();}
    if(gpu){std::string& s=rung_info[bits];char b[96];std::snprintf(b,sizeof b," %s %d/%d/%d",name(f),r.rung_words[0],r.rung_words[1],r.rung_words[2]);if(s.find(b)==std::string::npos)s+=b;}
}
// Runs function f on GPU (or CPU) and compares with the reference.
template<int Bits> void check_real(Transcendentals* gpu,Function f,const std::vector<Float<Bits>>& a,const std::vector<Float<Bits>>& b,Tally& t,const char* tag){
    std::size_t n=a.size();std::vector<Float<Bits>> out(n),ref(n);
    TranscendentalReport rep;
    if(gpu){gpu->run(Bits,f,a.data(),out.data(),n,f==Function::atan2?b.data():nullptr);rep=gpu->report();}
    else transcendental_cpu(Bits,f,a.data(),out.data(),n,f==Function::atan2?b.data():nullptr,0,&rep);
    parallel(n,[&](std::size_t lo,std::size_t hi){for(std::size_t i=lo;i<hi;++i)ref[i]=ref_real<Bits>(f,a[i],f==Function::atan2?b[i]:a[i]);});
    std::size_t bad=0;
    for(std::size_t i=0;i<n;++i)if(!same<Bits>(out[i],ref[i])){if(bad<5){if(!bad)fail_header(tag,Bits,f);
        std::printf("  i=%zu a=%s%s%s\n   got %s\n   ref %s\n",i,show<Bits>(a[i]).c_str(),f==Function::atan2?" b=":"",f==Function::atan2?show<Bits>(b[i]).c_str():"",show<Bits>(out[i]).c_str(),show<Bits>(ref[i]).c_str());}++bad;}
    check_report(rep,n,Bits,f,gpu!=nullptr);
    t.points+=n;t.mismatches+=bad;t.retried+=rep.retried;t.unresolved+=rep.unresolved;for(int r=0;r<4;++r)t.resolved[r]+=rep.resolved[r];t.gpu+=rep.gpu_seconds;t.wall+=rep.wall_seconds;
}
template<int Bits> void check_complex(Transcendentals* gpu,Function f,const std::vector<Complex<Bits/32>>& z,Tally& t,const char* tag){
    std::size_t n=z.size();std::vector<Complex<Bits/32>> out(n),ref(n);TranscendentalReport rep;
    if(gpu){gpu->run(Bits,f,z.data(),out.data(),n);rep=gpu->report();}else transcendental_cpu(Bits,f,z.data(),out.data(),n,nullptr,0,&rep);
    parallel(n,[&](std::size_t lo,std::size_t hi){for(std::size_t i=lo;i<hi;++i)ref[i]=ref_complex<Bits>(f,z[i]);});
    std::size_t bad=0;
    for(std::size_t i=0;i<n;++i)if(!reference::equal_complex<Bits>(out[i],ref[i])){if(bad<5){if(!bad)fail_header(tag,Bits,f);
        std::printf("  i=%zu z=(%s, %s)\n   got (%s, %s)\n   ref (%s, %s)\n",i,show<Bits>(z[i].re).c_str(),show<Bits>(z[i].im).c_str(),show<Bits>(out[i].re).c_str(),show<Bits>(out[i].im).c_str(),
            show<Bits>(ref[i].re).c_str(),show<Bits>(ref[i].im).c_str());}++bad;}
    check_report(rep,n,Bits,f,gpu!=nullptr);
    t.points+=n;t.mismatches+=bad;t.retried+=rep.retried;t.unresolved+=rep.unresolved;for(int r=0;r<4;++r)t.resolved[r]+=rep.resolved[r];t.gpu+=rep.gpu_seconds;t.wall+=rep.wall_seconds;
}
// Random inputs per function.
template<int Bits> std::vector<Float<Bits>> random_real(Function f,std::size_t n,std::mt19937_64& g){
    std::vector<Float<Bits>> v(n);
    for(auto& x:v){int r=int(g()%16);
        switch(f){
        case Function::exp:case Function::expm1:x=r<12?random_value<Bits>(g,-40,30):r<14?random_value<Bits>(g,-1000000000,-41):random_value<Bits>(g,31,1000000000);break;
        case Function::log:x=r<10?random_value<Bits>(g,-1000000000,1000000000,1):r<14?random_value<Bits>(g,-60,60,1):random_value<Bits>(g,-60,60,-1);break;
        case Function::log1p:x=r<8?random_value<Bits>(g,-300,20):r<12?random_value<Bits>(g,-3,-1):random_value<Bits>(g,-1000000000,1000000000);break;
        default:x=r<12?random_value<Bits>(g,-80,80):r<15?random_value<Bits>(g,81,8200):random_value<Bits>(g,-1000000000,1000000000);break;}
        if(g()%1000==0)x=zero<Bits/32>();}
    return v;
}
template<int Bits> void run_width(Transcendentals* gpu,std::size_t points,std::mt19937_64& g,std::map<std::string,Tally>& tallies,bool hard){
    constexpr int N=Bits/32;const char* tag=gpu?"gpu":"cpu";
    for(Function f:{Function::exp,Function::expm1,Function::log,Function::log1p,Function::sin,Function::cos}){
        auto a=random_real<Bits>(f,points,g);if(hard){auto h=hard_real<Bits>(f);a.insert(a.end(),h.begin(),h.end());}
        check_real<Bits>(gpu,f,a,a,tallies[name(f)],tag);
    }
    { // atan2
        std::vector<Float<Bits>> y(points),x(points);
        for(std::size_t i=0;i<points;++i){int r=int(g()%8);int e1=r<5?-100:-1000000000,e2=r<5?100:1000000000;y[i]=random_value<Bits>(g,e1,e2);x[i]=random_value<Bits>(g,e1,e2);
            if(r==7){x[i]=y[i];x[i].sign=g()&1?1:-1;x[i]=step<Bits>(x[i],int(g()%5)-2);}if(g()%500==0)y[i]=zero<N>();if(g()%500==0)x[i]=zero<N>();}
        if(hard)for(auto& p:hard_pairs<Bits>()){y.push_back(p.first);x.push_back(p.second);}
        check_real<Bits>(gpu,Function::atan2,y,x,tallies["atan2"],tag);
    }
    for(Function f:{Function::complex_exp,Function::complex_log}){
        std::vector<Complex<N>> z(points);
        for(auto& w:z){int r=int(g()%8);
            if(f==Function::complex_exp){w.re=r<6?random_value<Bits>(g,-40,29):random_value<Bits>(g,-1000,40);w.im=r<6?random_value<Bits>(g,-80,60):random_value<Bits>(g,-2000,8200);}
            else if(r<5){w.re=random_value<Bits>(g,-1000000000,1000000000);w.im=random_value<Bits>(g,-1000000000,1000000000);}
            else if(r<7){w.re=random_value<Bits>(g,-3,1);w.im=random_value<Bits>(g,-3,1);}
            else{w.re=random_value<Bits>(g,-1,-1);w.re.limb[N-1]=0xffffffffu;w.im=random_value<Bits>(g,-40-int(g()%(16*N)),-20);} // |z| near 1
            if(g()%300==0)w.re=zero<N>();if(g()%300==0)w.im=zero<N>();}
        if(hard)for(auto& p:hard_pairs<Bits>())z.push_back({p.first,p.second});
        check_complex<Bits>(gpu,f,z,tallies[name(f)],tag);
    }
}
// powi: MPFR replay (bit for bit) and MPC ulp statistics.
template<int Bits> void check_powi(Transcendentals* gpu,std::size_t points,std::mt19937_64& g,std::map<std::string,Tally>& tallies,double& max_ulps){
    constexpr int N=Bits/32;std::vector<Complex<N>> z(points);std::vector<std::int32_t> k(points);
    for(std::size_t i=0;i<points;++i){z[i]={random_value<Bits>(g,-4,4),random_value<Bits>(g,-4,4)};int r=int(g()%10);
        k[i]=r<6?int(g()%41)-20:r<8?int(g()%2001)-1000:r==8?(g()&1?INT32_MAX:INT32_MIN):int(std::int32_t(g()));
        if(g()%200==0)z[i].im=zero<N>();if(g()%500==0)z[i]={zero<N>(),zero<N>()};if(g()%500==0)k[i]=0;}
    std::vector<Complex<N>> out(points),ref(points);
    if(gpu)gpu->powi(Bits,z.data(),k.data(),points,out.data(),points);else transcendental_cpu(Bits,Function::complex_powi,z.data(),out.data(),points,k.data());
    std::vector<double> ulps(points,0);
    parallel(points,[&](std::size_t lo,std::size_t hi){for(std::size_t i=lo;i<hi;++i){ref[i]=Replay<Bits>::power(z[i],k[i]);
#ifdef LIMBFORGE_TEST_MPC
        const auto& o=out[i];if(o.re.status||o.im.status||std::abs(k[i])>64)continue;
        mpc_t x,p;mpc_init2(x,Bits);mpc_init2(p,2*Bits);to_mpc<Bits>(x,z[i]);mpc_pow_si(p,x,k[i],MPC_RNDNN);
        reference::MP e(2*Bits);double worst=0;long top=std::max(mpfr_get_exp(mpc_realref(p)),mpfr_get_exp(mpc_imagref(p)));
        for(int c=0;c<2;++c){const auto& comp=c?o.im:o.re;mpfr_srcptr pv=c?mpc_imagref(p):mpc_realref(p); // normwise: ulps of the larger part
            to_mpfr<Bits>(e.x,comp);mpfr_sub(e.x,e.x,pv,MPFR_RNDN);mpfr_mul_2si(e.x,e.x,Bits-top,MPFR_RNDN);worst=std::max(worst,std::fabs(mpfr_get_d(e.x,MPFR_RNDN)));}
        ulps[i]=worst;mpc_clear(x);mpc_clear(p);
#endif
    }});
    std::size_t bad=0;for(std::size_t i=0;i<points;++i){max_ulps=std::max(max_ulps,ulps[i]);if(!reference::equal_complex<Bits>(out[i],ref[i])){if(bad<5){if(!bad)fail_header(gpu?"gpu":"cpu",Bits,Function::complex_powi);
        std::printf("  i=%zu k=%d z=(%s, %s)\n   got (%s, %s)\n   ref (%s, %s)\n",i,k[i],show<Bits>(z[i].re).c_str(),show<Bits>(z[i].im).c_str(),show<Bits>(out[i].re).c_str(),show<Bits>(out[i].im).c_str(),
            show<Bits>(ref[i].re).c_str(),show<Bits>(ref[i].im).c_str());}++bad;}}
    auto& t=tallies["complex_powi"];t.points+=points;t.mismatches+=bad;
}
// ---- white-box: the error bounds of the W-word approximations ----
struct BoundStats { std::size_t checked=0,violations=0,exact=0,rejected=0;double max_ratio=0,max_err=0,max_bound=0; };
// Exact value of a W-word number (W may exceed 32 words) times 2^shift.
template<int W> void set_number(mpfr_ptr out,const Number<W>& x,exponent_type shift){
    if(!x.sign){mpfr_set_zero(out,1);return;}mpz_t z;mpz_init(z);mpz_import(z,W,-1,sizeof(word),0,0,x.limb);
    mpfr_set_z(out,z,MPFR_RNDN);mpfr_mul_2si(out,out,long(x.exponent)+long(shift)-(32*W-1),MPFR_RNDN);if(x.sign<0)mpfr_neg(out,out,MPFR_RNDN);mpz_clear(z);
}
template<int W> double actual_ulps(const tr::Approx<W>& a,mpfr_srcptr v){ // |y 2^shift - v| / (ulp_W(y) 2^shift)
    constexpr int P=32*W+256;reference::MP y(P),d(P);set_number<W>(y.x,a.y,a.shift);
    mpfr_sub(d.x,y.x,v,MPFR_RNDN);if(mpfr_zero_p(d.x))return 0;
    long ulp=long(a.y.exponent)+long(a.shift)+1-32*W;mpfr_mul_2si(d.x,d.x,-ulp,MPFR_RNDN);return std::fabs(mpfr_get_d(d.x,MPFR_RNDN));
}
template<int W> void bound_check(Function f,std::size_t points,std::mt19937_64& g,BoundStats& s){
    constexpr int P=32*W+256;const auto& t=*static_cast<const tr::Tables<W>*>(detail::transcendental_tables(W));const word* bits=detail::two_over_pi_bits();
    constexpr int NB=32*(W-2); // inputs carry N = W-2 words, as on the GPU
    std::vector<Float<NB>> a=random_real<NB>(f==Function::atan2||f==Function::complex_exp||f==Function::complex_log?Function::sin:f,points,g),b=random_real<NB>(Function::sin,points,g);
    if(f!=Function::atan2&&!function_is_complex(f)){auto h=hard_real<NB>(f);a.insert(a.end(),h.begin(),h.end());b.resize(a.size(),a[0]);}
    if(f==Function::complex_exp)for(std::size_t i=0;i<points;++i)a[i]=random_value<NB>(g,-40,29);
    std::mutex m;
    parallel(a.size(),[&](std::size_t lo,std::size_t hi){BoundStats local;reference::MP x(P),y(P),v(P),w(P);
        for(std::size_t i=lo;i<hi;++i){
            if(a[i].status||b[i].status)continue;
            Number<W> X=tr::widen<W>(a[i]),Y=tr::widen<W>(b[i]);tr::Approx<W> r[2];int parts=1;
            switch(f){case Function::exp:r[0]=tr::exp_approx(X,t,false);break;case Function::expm1:r[0]=tr::exp_approx(X,t,true);break;
                case Function::log:r[0]=tr::log_approx(X,t);break;case Function::log1p:r[0]=tr::log1p_approx(X,t);break;
                case Function::sin:r[0]=tr::sin_approx(X,bits,t,false);break;case Function::cos:r[0]=tr::sin_approx(X,bits,t,true);break;
                case Function::atan2:r[0]=tr::atan2_approx(X,Y,t);break;case Function::complex_exp:tr::cexp_approx(X,Y,bits,t,r[0],r[1]);parts=2;break;
                default:tr::clog_approx(X,Y,t,r[0],r[1]);parts=2;break;}
            to_mpfr<NB>(x.x,a[i]);to_mpfr<NB>(y.x,b[i]);
            for(int c=0;c<parts;++c){
                const auto& q=r[c];if(q.y.status)continue;
                switch(f){case Function::exp:mpfr_exp(v.x,x.x,MPFR_RNDN);break;case Function::expm1:mpfr_expm1(v.x,x.x,MPFR_RNDN);break;
                    case Function::log:mpfr_log(v.x,x.x,MPFR_RNDN);break;case Function::log1p:mpfr_log1p(v.x,x.x,MPFR_RNDN);break;
                    case Function::sin:mpfr_sin(v.x,x.x,MPFR_RNDN);break;case Function::cos:mpfr_cos(v.x,x.x,MPFR_RNDN);break;
                    case Function::atan2:mpfr_atan2(v.x,x.x,y.x,MPFR_RNDN);break;
                    case Function::complex_exp:mpfr_exp(v.x,x.x,MPFR_RNDN);if(c)mpfr_sin(w.x,y.x,MPFR_RNDN);else mpfr_cos(w.x,y.x,MPFR_RNDN);mpfr_mul(v.x,v.x,w.x,MPFR_RNDN);break;
                    default:if(c)mpfr_atan2(v.x,y.x,x.x,MPFR_RNDN);else{mpfr_hypot(v.x,x.x,y.x,MPFR_RNDN);mpfr_log(v.x,v.x,MPFR_RNDN);}break;}
                if(mpfr_inf_p(v.x))continue;
                if(q.err<0){reference::MP e(P);set_number<W>(e.x,q.y,q.shift);
                    ++local.exact;if(!mpfr_equal_p(e.x,v.x)){++local.violations;std::lock_guard<std::mutex> l(m);std::printf("  exact claim wrong: W=%d %s i=%zu\n",W,name(f),i);}continue;}
                if(!q.y.sign||q.err>1e15){++local.rejected;continue;} // rejected outright
                double act=actual_ulps<W>(q,v.x);++local.checked;local.max_bound=std::max(local.max_bound,double(q.err));
                if(act>1e3&&verbose_bounds){std::lock_guard<std::mutex> l(m);std::printf("  large W=%d %s comp %d: actual %.3g bound %.3g a=%s b=%s\n",W,name(f),c,act,double(q.err),show<NB>(a[i]).c_str(),show<NB>(b[i]).c_str());}local.max_err=std::max(local.max_err,act);local.max_ratio=std::max(local.max_ratio,act/double(q.err));
                if(act>double(q.err)){++local.violations;std::lock_guard<std::mutex> l(m);
                    std::printf("  BOUND VIOLATION W=%d %s comp %d: actual %.3g > bound %.3g, x=%s\n",W,name(f),c,act,double(q.err),show<NB>(a[i]).c_str());}
            }
        }
        std::lock_guard<std::mutex> l(m);s.checked+=local.checked;s.rejected+=local.rejected;s.max_bound=std::max(s.max_bound,local.max_bound);s.violations+=local.violations;s.exact+=local.exact;s.max_ratio=std::max(s.max_ratio,local.max_ratio);s.max_err=std::max(s.max_err,local.max_err);});
}
template<int W> void bounds_for(std::size_t points,std::mt19937_64& g){
    for(Function f:{Function::exp,Function::expm1,Function::log,Function::log1p,Function::sin,Function::cos,Function::atan2,Function::complex_exp,Function::complex_log}){
        BoundStats s;bound_check<W>(f,points,g,s);
        std::printf("bounds W=%2d %-12s checked %7zu exact %5zu rejected %3zu max actual %.3g ulp_W, max bound %.3g, max actual/bound %.3f%s\n",W,name(f),s.checked,s.exact,s.rejected,s.max_err,s.max_bound,s.max_ratio,s.violations?"  VIOLATIONS":"");
        if(s.violations){++failures;}
    }
}
}
// ---- per-rung GPU counts: a host replica of the GPU decisions (same approximations and certify<N,W> at the reported
// first-pass and rung widths) predicts how many elements each level decides; the report must match exactly. ----
template<int N,int W> bool decided(Function f,const Number<N>& a,const Number<N>& b){
    const auto& t=*static_cast<const tr::Tables<W>*>(detail::transcendental_tables(W));const word* bits=detail::two_over_pi_bits();
    Number<W> X=tr::widen<W>(a),Y=tr::widen<W>(b);tr::Approx<W> r[2];int parts=1;
    switch(f){case Function::exp:r[0]=tr::exp_approx(X,t,false);break;case Function::expm1:r[0]=tr::exp_approx(X,t,true);break;
        case Function::log:r[0]=tr::log_approx(X,t);break;case Function::log1p:r[0]=tr::log1p_approx(X,t);break;
        case Function::sin:r[0]=tr::sin_approx(X,bits,t,false);break;case Function::cos:r[0]=tr::sin_approx(X,bits,t,true);break;
        case Function::atan2:r[0]=tr::atan2_approx(X,Y,t);break;case Function::complex_exp:tr::cexp_approx(X,Y,bits,t,r[0],r[1]);parts=2;break;
        default:tr::clog_approx(X,Y,t,r[0],r[1]);parts=2;break;}
    Number<N> o;bool ok=true;for(int c=0;c<parts;++c)ok=tr::certify<N,W>(r[c].y,r[c].shift,r[c].err,o)&&ok;return ok;
}
template<int N,int... Ws> bool decided_at(int W,Function f,const Number<N>& a,const Number<N>& b){
    bool r=false,found=false;((W==Ws?(found=true,r=decided<N,Ws>(f,a,b)):false),...);
    if(!found)throw std::runtime_error("rung-count check: width "+std::to_string(W)+" not instantiated (update check_rung_counts)");return r;
}
template<int Bits,int... Ws> void check_rung_counts(Transcendentals& gpu,std::size_t points,std::mt19937_64& g){
    constexpr int N=Bits/32;std::size_t total[4]={};
    for(Function f:{Function::exp,Function::expm1,Function::log,Function::log1p,Function::sin,Function::cos,Function::atan2,Function::complex_exp,Function::complex_log}){
        std::vector<Float<Bits>> a,b;
        if(f==Function::atan2||function_is_complex(f)){for(std::size_t i=0;i<points;++i){a.push_back(random_value<Bits>(g,-3,3));b.push_back(random_value<Bits>(g,-3,3));}
            for(auto& p:hard_pairs<Bits>()){a.push_back(p.first);b.push_back(p.second);}}
        else{a=random_real<Bits>(f,points,g);auto h=hard_real<Bits>(f);a.insert(a.end(),h.begin(),h.end());b=a;}
        const std::size_t n=a.size();TranscendentalReport rep;
        if(function_is_complex(f)){std::vector<Complex<N>> z(n),o(n);for(std::size_t i=0;i<n;++i)z[i]={a[i],b[i]};gpu.run(Bits,f,z.data(),o.data(),n);}
        else{std::vector<Float<Bits>> o(n);gpu.run(Bits,f,a.data(),o.data(),n,b.data());}
        rep=gpu.report();int words[4]={rep.first_words,rep.rung_words[0],rep.rung_words[1],rep.rung_words[2]};
        std::vector<int> level(n,4);
        parallel(n,[&](std::size_t lo,std::size_t hi){for(std::size_t i=lo;i<hi;++i)for(int l=0;l<4&&words[l];++l)if(decided_at<N,Ws...>(words[l],f,a[i],b[i])){level[i]=l;break;}});
        std::size_t want[5]={};for(int l:level)++want[l];int last=1;while(last<4&&words[last])++last;
        std::size_t host=0;for(int l=last;l<5;++l)host+=want[l];
        bool ok=rep.retried==n-want[0]&&rep.host_retried()==host;for(int l=1;l<last;++l)ok=ok&&rep.resolved[l-1]==want[l];
        for(int l=1;l<last;++l)total[l-1]+=want[l];total[3]+=host;
        if(!ok){fail_header("rung counts",Bits,f);std::printf("  predicted retried %zu rungs %zu/%zu/%zu host %zu; reported %zu, %zu/%zu/%zu, host %zu\n",n-want[0],want[1],want[2],want[3],host,
            rep.retried,rep.resolved[0],rep.resolved[1],rep.resolved[2],rep.host_retried());}
    }
    std::printf("per-rung GPU counts at %d bits equal the host replica (all functions; rung 1/2/3/host: %zu/%zu/%zu/%zu)\n",Bits,total[0],total[1],total[2],total[3]);
}
// Host-array retry threshold: few retries go to the host ladder (no GPU rung, report.resolved[3]), many to the GPU rungs
// (second command buffer); both give the results of the threshold-0 object bit for bit.
void check_threshold(Transcendentals& gpu0,std::mt19937_64& g){
    constexpr int Bits=256;Transcendentals def;std::size_t fails=0;
    for(Function f:{Function::exp,Function::cos,Function::log1p}){
        auto hard=hard_real<Bits>(f);auto few=random_real<Bits>(f,2000,g);few.insert(few.end(),hard.begin(),hard.end());
        std::vector<Float<Bits>> many;for(int r=0;r<40;++r)many.insert(many.end(),hard.begin(),hard.end());
        for(auto* in:{&few,&many}){std::size_t n=in->size();std::vector<Float<Bits>> o0(n),o1(n);
            gpu0.run(Bits,f,in->data(),o0.data(),n);auto r0=gpu0.report();def.run(Bits,f,in->data(),o1.data(),n);auto r1=def.report();
            bool host=r1.retried<=TranscendentalOptions{}.gpu_retry_threshold,ok=r0.retried==r1.retried&&r1.retried>0;
            for(std::size_t i=0;i<n;++i)ok=ok&&reference::equal<Bits>(o0[i],o1[i]);
            if(host)ok=ok&&r1.resolved[3]+r1.unresolved==r1.retried&&!r1.rung_words[0];
            else{ok=ok&&r1.rung_words[0]==r0.rung_words[0];for(int k=0;k<4;++k)ok=ok&&r1.resolved[k]==r0.resolved[k];}
            if(!ok){++fails;fail_header("retry threshold",Bits,f);std::printf("  retried %zu/%zu host route %d\n",r0.retried,r1.retried,int(host));}
            else std::printf("retry threshold %s: %zu retries -> %s, results equal the GPU-rung object\n",name(f),r1.retried,host?"host ladder":"GPU rungs");}
    }
    (void)fails;
}
template<int B=64,class F> void by_bits(int bits,F&& f){
    if constexpr(B<=1024){if(bits==B){f(std::integral_constant<int,B>{});return;}by_bits<B+32>(bits,f);}else throw std::invalid_argument("bits must be a multiple of 32 in [64,1024]");
}
int main(int argc,char** argv){
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    bool cpu_only=false;std::size_t threshold=0;std::size_t points=20000,cpu_points=3000,bound_points=3000,all_points=0;std::vector<int> widths={64,224,256,384,1024};unsigned seed=35;bool bounds=true,gpu_only=false,hard=true;
    for(int i=1;i<argc;++i){std::string s=argv[i];
        if(s=="--cpu-only")cpu_only=true;else if(s=="--points"&&i+1<argc)points=std::stoul(argv[++i]);else if(s=="--cpu-points"&&i+1<argc)cpu_points=std::stoul(argv[++i]);
        else if(s=="--bound-points"&&i+1<argc)bound_points=std::stoul(argv[++i]);else if(s=="--all-widths"&&i+1<argc)all_points=std::stoul(argv[++i]);
        else if(s=="--force-level"&&i+1<argc)forced=std::stoi(argv[++i]);
        else if(s=="--retry-threshold"&&i+1<argc)threshold=std::stoul(argv[++i]);
        else if(s=="--seed"&&i+1<argc)seed=unsigned(std::stoul(argv[++i]));else if(s=="--no-bounds")bounds=false;else if(s=="--verbose-bounds")verbose_bounds=true;else if(s=="--gpu-only")gpu_only=true;else if(s=="--no-hard")hard=false;
        else if(s=="--bits"){widths.clear();while(i+1<argc&&argv[i+1][0]!='-')widths.push_back(std::stoi(argv[++i]));}
        else{std::fprintf(stderr,"unknown argument %s\n",argv[i]);return 2;}}
    std::mt19937_64 g(seed);
    if(bounds){std::printf("== error-bound check (W-word approximations vs MPFR at 32W+256 bits)\n");
        bounds_for<4>(bound_points,g);bounds_for<9>(bound_points,g);bounds_for<10>(bound_points,g);bounds_for<14>(bound_points,g);bounds_for<34>(bound_points/3,g);}
    std::unique_ptr<Transcendentals> gpu;if(!cpu_only){TranscendentalOptions o;o.gpu_retry_threshold=threshold;gpu=std::make_unique<Transcendentals>(o);std::printf("device: %s\n",gpu->device_name().c_str());}
    if(forced){detail::transcendental_force_level(forced);std::printf("GPU levels below %d treat every element as undecided\n",forced);}
    auto run=[&](Transcendentals* dev,std::size_t n,bool hard,const std::vector<int>& ws,const char* label){
        std::map<std::string,Tally> tallies;double powi_ulps=0;width_tally.clear();
        for(int b:ws){auto t0=std::chrono::steady_clock::now();
            by_bits(b,[&](auto tag){constexpr int Bits=decltype(tag)::value;run_width<Bits>(dev,n,g,tallies,hard);check_powi<Bits>(dev,n,g,tallies,powi_ulps);});
            auto& w=width_tally[b];
            std::printf("  %s %4d bits done in %.1f s%s%s",label,b,std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count(),dev?"; GPU rung words:":"",dev?rung_info[b].c_str():"");
            if(dev)std::printf("; retried %zu (rungs %zu/%zu/%zu, host %zu)",w[0],w[1],w[2],w[3],w[4]);std::printf("\n");w={};std::fflush(stdout);}
        std::printf("== %s summary (%zu random points per function per width%s)\n",label,n,hard?" + hard cases":"");
        for(auto& [k,t]:tallies)std::printf("  %-12s points %9zu mismatches %zu retried %zu (rung1 %zu, rung2 %zu, rung3 %zu, host %zu) unresolved %zu retry rate %.2e\n",k.c_str(),t.points,t.mismatches,t.retried,
            t.resolved[0],t.resolved[1],t.resolved[2],t.resolved[3],t.unresolved,t.points?double(t.retried)/double(t.points):0.0);
        std::printf("  complex_powi max error vs mpc_pow_si (|k| <= 64, normwise): %.3g ulp\n",powi_ulps);
    };
    if(!gpu_only)run(nullptr,cpu_points,hard,widths,"cpu");
    if(gpu&&!forced&&!std::getenv("LIMBFORGE_TRANSCENDENTAL_RUNG_WORDS")&&!std::getenv("LIMBFORGE_TRANSCENDENTAL_GUARD_WORDS"))
        for(int b:widths){if(b==256){check_rung_counts<256,10,14,22,35>(*gpu,2000,g);check_threshold(*gpu,g);}if(b==384)check_rung_counts<384,14,18,29,35>(*gpu,2000,g);if(b==1024)check_rung_counts<1024,34,35>(*gpu,500,g);}
    if(gpu){if(points)run(gpu.get(),points,hard,widths,"gpu");
        if(all_points){std::vector<int> all;for(int b=64;b<=1024;b+=32)all.push_back(b);run(gpu.get(),all_points,hard,all,"gpu all widths");}}
    std::printf(failures?"FAILED (%d)\n":"PASSED\n",failures);return failures?1:0;
}
