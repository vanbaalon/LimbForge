// Element-wise transcendental functions (plan D7; docs/numerics.md, "Transcendental functions"). The GPU evaluates
// at N+2 words and certifies the rounding (src/transcendental_core.hpp); the compacted list of undecided elements is
// resolved here on the host by the same code at more words. Constants come from exact integer series computed here.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "limbforge/transcendental.hpp"
#include "transcendental_core.hpp"
#include "transcendental_source.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace limbforge {
namespace tr=transcendental;
namespace {
using Clock=std::chrono::steady_clock;
double since(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
std::string message(NSError* e){return e?std::string([[e localizedDescription] UTF8String]):"unknown Metal error";}
// ---- exact fixed-point series for the constants: value = integer / 2^B (little-endian words) ----
using Big=std::vector<word>;
constexpr long two_over_pi_length=((tr::TRIG_EMAX+32L*136+320)/32)*32; // bits; the table has L/32 + 2 words
word divs(Big& x,word d){dword r=0;for(std::size_t i=x.size();i-->0;){dword c=(r<<32)|x[i];x[i]=word(c/d);r=c%d;}return word(r);}
void addb(Big& x,const Big& y){dword c=0;for(std::size_t i=0;i<x.size();++i){dword v=dword(x[i])+y[i]+c;x[i]=word(v);c=v>>32;}}
void subb(Big& x,const Big& y){dword b=0;for(std::size_t i=0;i<x.size();++i){dword v=dword(y[i])+b;b=dword(x[i])<v;x[i]=word(dword(x[i])-v);}}
void mulb(Big& x,word m){dword c=0;for(auto& w:x){dword v=dword(w)*m+c;w=word(v);c=v>>32;}}
bool zerob(const Big& x){for(word w:x)if(w)return false;return true;}
int cmpb(const Big& x,const Big& y){for(std::size_t i=x.size();i-->0;)if(x[i]!=y[i])return x[i]>y[i]?1:-1;return 0;}
// sum_{k>=0} s_k 2^-(first+step*k) / (mul*k+add), s_k = (-1)^k when alternating; each term truncated (error < 1 unit).
Big series(std::size_t words,long B,long first,long step,word mul,word add,bool alternate){
    Big pos(words,0),neg(words,0);
    for(long k=0;first+step*k<=B;++k){Big t(words,0);long bit=B-(first+step*k);t[bit/32]=word(1)<<(bit%32);divs(t,word(mul*k+add));
        if(zerob(t))break;addb(alternate&&(k&1)?neg:pos,t);}
    subb(pos,neg);return pos;
}
// atan(1/m) = sum (-1)^k m^-(2k+1)/(2k+1) (Machin).
Big atan_inverse(std::size_t words,long B,word m){
    Big pos(words,0),neg(words,0),p(words,0);p[B/32]=word(1)<<(B%32);divs(p,m);
    for(word k=0;!zerob(p);++k){Big t=p;divs(t,2*k+1);addb(k&1?neg:pos,t);divs(p,m*m);}
    subb(pos,neg);return pos;
}
Big pi_fixed(std::size_t words,long B){Big a=atan_inverse(words,B,5),b=atan_inverse(words,B,239);mulb(a,16);mulb(b,4);subb(a,b);return a;}
// RN of integer(v) * 2^-B to M words (ties to even), sign applied.
template<int M> Number<M> to_number(const Big& v,long B,int sign){
    std::size_t top=v.size();while(top&&!v[top-1])--top;if(!top)return zero<M>();
    long h=32*long(top-1)+31-__builtin_clz(v[top-1]),low=h-32*M+1;
    auto bit=[&](long p)->word{return p<0||p>=32*long(v.size())?0u:(v[std::size_t(p/32)]>>(p%32))&1u;};
    Number<M> r=zero<M>();r.sign=sign;r.exponent=int(h-B);
    for(int i=0;i<M;++i)for(int j=0;j<32;++j)r.limb[i]|=bit(low+32*i+j)<<j;
    bool round=bit(low-1),sticky=false;for(long p=low-2;p>=0&&!sticky;--p)sticky=bit(p);
    if(round&&(sticky||(r.limb[0]&1)))increment(r);
    return r;
}
int floor_log2_factorial(int n){int s=0;for(int j=2;j<=n;++j)s+=31-__builtin_clz(unsigned(j));return s;}
int isqrt(int x){int r=0;while((r+1)*(r+1)<=x)++r;return r;}
template<int W> std::shared_ptr<void> make_tables(){
    auto t=std::make_shared<tr::Tables<W>>();std::memset(t.get(),0,sizeof(tr::Tables<W>));
    const int need=32*W+4;int best=1<<30;
    // exp: s halvings and K Taylor terms with S*K + log2((K+1)!) >= 32W+4 (truncation < u/16), minimising S+K.
    for(int S=1;S<=64;++S){int K=1;while(S*K+floor_log2_factorial(K+1)<need)++K;if(S+K<best){best=S+K;t->s_exp=S;t->k_exp=K;}}
    // sin/vers: first omitted term w^(K+1)/(2K+3)! with w < 2^-2S below u/16.
    best=1<<30;for(int S=1;S<=64;++S){int K=1;while(2*S*(K+1)+floor_log2_factorial(2*K+3)<need)++K;if(S+K<best){best=S+K;t->s_trig=S;t->k_trig=K;}}
    t->steps=std::min(int(tr::TMAX),std::max(8,isqrt(24*W)));t->k_series=(need+2*t->steps-1)/(2*t->steps)+1;
    const long B=32L*(W+6);const std::size_t words=std::size_t(B/32+2);
    Big ln2=series(words,B,1,1,1,1,false),pi=pi_fixed(words,B);
    t->ln2x=to_number<W+2>(ln2,B,1);t->ln2=to_number<W>(ln2,B,1);t->pi=to_number<W>(pi,B,1);t->half_pi=to_number<W>(pi,B+1,1);
    t->up[0]=t->ln2;t->down[0]=zero<W>(invalid);t->atn[0]=to_number<W>(pi,B+2,1);
    for(int i=1;i<=tr::TMAX;++i){
        t->up[i]=to_number<W>(series(words,B,i,i,1,1,true),B,1);
        t->down[i]=to_number<W>(series(words,B,i,i,1,1,false),B,-1);
        t->atn[i]=to_number<W>(series(words,B,i,2L*i,2,1,true),B,1);
    }
    return t;
}
// Bits of 2/pi, MSB first (bit j has weight 2^-j), floor(2^L * 2/pi) up to +1: L covers every window of
// reduce_half_pi up to 136 working words with 64 spare bits, plus two zero words for the shifted reads.
std::vector<word> make_two_over_pi(){
    const long L=two_over_pi_length,Bp=L+160;const std::size_t pw=std::size_t(Bp/32+2);
    Big P=pi_fixed(pw,Bp);                 // ~ pi 2^Bp
    // Q = floor(2^(Bp+2) * 2^(L-1) / P): remainder R = 2^(Bp+2) - P (the leading quotient bit is 1), then L-1 bits.
    Big R(pw+1,0),Pw=P;Pw.push_back(0);R[std::size_t((Bp+2)/32)]=word(1)<<((Bp+2)%32);subb(R,Pw);
    std::vector<word> q(std::size_t(L/32)+2,0);
    auto set=[&](long bitpos){long j=L-bitpos;if(j>=1&&j<=L)q[std::size_t((j-1)/32)]|=word(1)<<(31-(j-1)%32);};
    set(L-1);
    for(long b=L-2;b>=0;--b){word c=0;for(auto& w:R){word n=(w<<1)|c;c=w>>31;w=n;}if(cmpb(R,Pw)>=0){subb(R,Pw);set(b);}}
    return q;
}
struct TableCache { std::mutex m;std::map<int,std::shared_ptr<void>> tables;std::vector<word> bits;bool have_bits=false; };
TableCache& cache(){static TableCache c;return c;}
#define LF_TABLE_CASE(X) case X:p=make_tables<X>();break;
std::shared_ptr<void> build(int W){std::shared_ptr<void> p;
    switch(W){LF_TABLE_CASE(4)LF_TABLE_CASE(5)LF_TABLE_CASE(6)LF_TABLE_CASE(7)LF_TABLE_CASE(8)LF_TABLE_CASE(9)LF_TABLE_CASE(10)LF_TABLE_CASE(11)
        LF_TABLE_CASE(12)LF_TABLE_CASE(13)LF_TABLE_CASE(14)LF_TABLE_CASE(15)LF_TABLE_CASE(16)LF_TABLE_CASE(17)LF_TABLE_CASE(18)LF_TABLE_CASE(19)
        LF_TABLE_CASE(20)LF_TABLE_CASE(21)LF_TABLE_CASE(22)LF_TABLE_CASE(23)LF_TABLE_CASE(24)LF_TABLE_CASE(25)LF_TABLE_CASE(26)LF_TABLE_CASE(27)
        LF_TABLE_CASE(28)LF_TABLE_CASE(29)LF_TABLE_CASE(30)LF_TABLE_CASE(31)LF_TABLE_CASE(32)LF_TABLE_CASE(33)LF_TABLE_CASE(34)LF_TABLE_CASE(48)
        LF_TABLE_CASE(68)LF_TABLE_CASE(136)
        default:throw std::invalid_argument("transcendental tables: unsupported working width");}
    return p;}
#undef LF_TABLE_CASE
// sizeof(Tables<W>) = 4*(8 + (W+5) + 3(W+3) + 3(TMAX+1)(W+3)); Number<M> is M+3 words.
std::size_t table_bytes(int W){return 4*std::size_t(8+(W+5)+3*(W+3)+3*(tr::TMAX+1)*(W+3));}
// Working words of the GPU kernels: the smallest validated width >= N+2. All-width GPU runs (round 35) gave wrong
// results, while the CPU was exact, for W = 11, 12, 13, 15, 16 (exp, expm1, complex exp/log: their W+2-word fma) and
// for every function at W = 17, 19-21, 23-25, 27, 28; W = 4-10, 14, 18, 22, 26 and 29-34 passed every test
// (docs/gpu-codegen.md rule 5: probe every instantiated width). The extra guard words only cost time.
// Diagnostic override (probing other widths): LIMBFORGE_TRANSCENDENTAL_GUARD_WORDS=g selects W = N+g, g in [2, 8].
int gpu_words(int N){
    if(const char* g=std::getenv("LIMBFORGE_TRANSCENDENTAL_GUARD_WORDS")){int v=std::atoi(g);if(v>=2&&v<=8&&N+v<=34)return N+v;}
    static const int ok[]={4,5,6,7,8,9,10,14,18,22,26,29,30,31,32,33,34};for(int w:ok)if(w>=N+2)return w;return 34;}
// ---- host evaluation ladder ----
constexpr int ladder[]={4,6,8,10,12,16,20,24,34,48,68,136};
constexpr int ladder_size=int(sizeof(ladder)/sizeof(ladder[0]));
int ladder_at_least(int w){for(int L:ladder)if(L>=w)return L;return 136;}
template<int W> __attribute__((noinline)) bool evaluate(Function f,const Number<W>& x,const Number<W>& y,tr::Approx<W>& r0,tr::Approx<W>& r1){
    const auto& t=*static_cast<const tr::Tables<W>*>(detail::transcendental_tables(W));const word* bits=detail::two_over_pi_bits();
    switch(f){
    case Function::exp:r0=tr::exp_approx(x,t,false);return false;
    case Function::expm1:r0=tr::exp_approx(x,t,true);return false;
    case Function::log:r0=tr::log_approx(x,t);return false;
    case Function::log1p:r0=tr::log1p_approx(x,t);return false;
    case Function::sin:r0=tr::sin_approx(x,bits,t,false);return false;
    case Function::cos:r0=tr::sin_approx(x,bits,t,true);return false;
    case Function::atan2:r0=tr::atan2_approx(x,y,t);return false;
    case Function::complex_exp:tr::cexp_approx(x,y,bits,t,r0,r1);return true;
    case Function::complex_log:tr::clog_approx(x,y,t,r0,r1);return true;
    default:throw std::invalid_argument("transcendental: not a ladder function");}
}
// One attempt at W words; on failure o[] holds RN of the approximation (the unresolved fallback).
template<int N,int W> bool attempt(Function f,const Number<N>& a,const Number<N>& b,Number<N>* o){
    tr::Approx<W> r0,r1;bool two=evaluate<W>(f,tr::widen<W>(a),tr::widen<W>(b),r0,r1);
    bool ok=tr::certify<N,W>(r0.y,r0.shift,r0.err,o[0]);if(!ok)tr::certify<N,W>(r0.y,r0.shift,-1.f,o[0]);
    if(two){bool ok1=tr::certify<N,W>(r1.y,r1.shift,r1.err,o[1]);if(!ok1)tr::certify<N,W>(r1.y,r1.shift,-1.f,o[1]);ok=ok&&ok1;}
    return ok;
}
template<int N,int I=0> bool attempt_at(int W,Function f,const Number<N>& a,const Number<N>& b,Number<N>* o){
    if constexpr(I<ladder_size){constexpr int L=ladder[I];
        if(W==L){if constexpr(L>=N+2)return attempt<N,L>(f,a,b,o);else return false;}
        return attempt_at<N,I+1>(W,f,a,b,o);
    }else return false;
}
template<int N=2,class F> auto by_words(int words,F&& f){if constexpr(N<32){if(words!=N)return by_words<N+1>(words,f);}return f(std::integral_constant<int,N>{});}
void check_bits(int bits){if(bits<64||bits>1024||bits%32)throw std::invalid_argument("bits must be a multiple of 32 in [64,1024]");}
template<class F> void parallel_for(std::size_t n,unsigned threads,std::size_t grain,F f){
    if(!threads)threads=std::max(1u,std::thread::hardware_concurrency());
    unsigned t=unsigned(std::min<std::size_t>(threads,(n+grain-1)/std::max<std::size_t>(grain,1)));
    if(t<=1){if(n)f(std::size_t(0),n);return;}
    std::vector<std::thread> w;for(unsigned i=1;i<t;++i)w.emplace_back(f,n*i/t,n*(i+1)/t);f(std::size_t(0),n/t);for(auto& x:w)x.join();
}
struct Counters { std::atomic<std::size_t> resolved[4]{},unresolved{}; };
// Element i of a real (one Number) or complex (two Numbers) array.
template<int N> void operands(Function f,const Number<N>* A,const Number<N>* Bv,std::size_t i,Number<N>& a,Number<N>& b){
    if(function_is_complex(f)){a=A[2*i];b=A[2*i+1];}else{a=A[i];b=f==Function::atan2?Bv[i]:zero<N>();}
}
// Retry ladder for the listed elements: widths >= N+4, 2N+4, 4N+8 words.
template<int N> void resolve(Function f,const Number<N>* A,const Number<N>* Bv,Number<N>* out,const std::uint32_t* list,std::size_t n,unsigned threads,Counters& c){
    const int rungs[3]={ladder_at_least(N+4),ladder_at_least(2*N+4),ladder_at_least(4*N+8)};const std::size_t stride=function_is_complex(f)?2:1;
    parallel_for(n,threads,16,[&](std::size_t lo,std::size_t hi){for(std::size_t q=lo;q<hi;++q){std::size_t i=list[q];Number<N> a,b,o[2];operands(f,A,Bv,i,a,b);
        int r=0,last=0;bool done=false;for(;r<3&&!done;++r){if(r&&rungs[r]==last)continue;last=rungs[r];done=attempt_at<N>(rungs[r],f,a,b,o);if(done)++c.resolved[r];}
        if(!done)++c.unresolved;
        out[stride*i]=o[0];if(stride==2)out[stride*i+1]=o[1];}});
}
template<int N> void powi_cpu(const Complex<N>* z,const std::int32_t* k,std::size_t k_count,Complex<N>* out,std::size_t n,unsigned threads){
    parallel_for(n,threads,256,[&](std::size_t lo,std::size_t hi){for(std::size_t i=lo;i<hi;++i)out[i]=tr::cpowi<N>(z[i],k[k_count==1?0:i]);});
}
}
namespace detail {
const void* transcendental_tables(int W){
    auto& c=cache();{std::lock_guard<std::mutex> l(c.m);auto it=c.tables.find(W);if(it!=c.tables.end())return it->second.get();}
    auto p=build(W);std::lock_guard<std::mutex> l(c.m);return c.tables.emplace(W,p).first->second.get();
}
const std::uint32_t* two_over_pi_bits(){
    auto& c=cache();{std::lock_guard<std::mutex> l(c.m);if(c.have_bits)return c.bits.data();}
    auto q=make_two_over_pi();std::lock_guard<std::mutex> l(c.m);if(!c.have_bits){c.bits=std::move(q);c.have_bits=true;}return c.bits.data();
}
}
void transcendental_cpu(int bits,Function f,const void* a,void* out,std::size_t count,const void* b,unsigned threads,TranscendentalReport* report){
    check_bits(bits);if(int(f)<0||int(f)>int(Function::complex_powi))throw std::invalid_argument("transcendental: invalid function");
    if(f==Function::atan2&&!b&&count)throw std::invalid_argument("atan2 needs the x operand");
    if(f==Function::complex_powi&&!b&&count)throw std::invalid_argument("complex_powi needs the k array");
    auto start=Clock::now();TranscendentalReport rep;rep.count=count;
    by_words(bits/32,[&](auto tag){constexpr int N=decltype(tag)::value;
        if(f==Function::complex_powi){powi_cpu<N>(static_cast<const Complex<N>*>(a),static_cast<const std::int32_t*>(b),count,static_cast<Complex<N>*>(out),count,threads);return;}
        const auto* A=static_cast<const Number<N>*>(a);const auto* Bv=static_cast<const Number<N>*>(b);auto* O=static_cast<Number<N>*>(out);
        const std::size_t stride=function_is_complex(f)?2:1;const int first=ladder_at_least(N+2);
        detail::transcendental_tables(first);detail::two_over_pi_bits();
        // Undecided elements are not written in the first pass, so an in-place call keeps their operands for the retry.
        std::vector<std::uint32_t> failed;std::mutex m;
        parallel_for(count,threads,64,[&](std::size_t lo,std::size_t hi){std::vector<std::uint32_t> mine;
            for(std::size_t i=lo;i<hi;++i){Number<N> x,y,o[2];operands(f,A,Bv,i,x,y);
                if(!attempt_at<N>(first,f,x,y,o)){mine.push_back(std::uint32_t(i));continue;}
                O[stride*i]=o[0];if(stride==2)O[stride*i+1]=o[1];}
            if(!mine.empty()){std::lock_guard<std::mutex> l(m);failed.insert(failed.end(),mine.begin(),mine.end());}});
        if(failed.empty())return;
        rep.retried=failed.size();std::sort(failed.begin(),failed.end());
        auto rt=Clock::now();Counters c;resolve<N>(f,A,Bv,O,failed.data(),failed.size(),threads,c);
        for(int r=0;r<3;++r)rep.resolved[r]=c.resolved[r];rep.unresolved=c.unresolved;rep.retry_seconds=since(rt);
    });
    rep.wall_seconds=since(start);if(report)*report=rep;
}
struct Library { id<MTLLibrary> lib;std::map<int,id<MTLComputePipelineState>> pipes;id<MTLBuffer> tables; };
struct Transcendentals::Impl {
    TranscendentalOptions options;TranscendentalReport report;id<MTLDevice> device;id<MTLCommandQueue> queue;
    std::map<int,Library> libs;id<MTLBuffer> bits_buffer;std::map<std::string,id<MTLBuffer>> pool;
    id<MTLBuffer> buffer(const std::string& role,std::size_t bytes){
        auto& b=pool[role];if(!b||b.length<bytes){b=nil;b=[device newBufferWithLength:std::max<std::size_t>(bytes,16) options:MTLResourceStorageModeShared];}
        if(!b)throw std::runtime_error("Metal allocation failed ("+role+")");return b;}
    Library& library(int bits){
        auto found=libs.find(bits);if(found!=libs.end())return found->second;
        const int W=gpu_words(bits/32);Library L;
        NSString* source=[NSString stringWithFormat:@"#define LF_BITS %d\n#define LF_W %d\n%s",bits,W,limbforge_transcendental_source];
        MTLCompileOptions* o=[MTLCompileOptions new];o.languageVersion=MTLLanguageVersion((4u<<16)|0u);o.mathMode=MTLMathModeSafe;
        NSError* e=nil;L.lib=[device newLibraryWithSource:source options:o error:&e];
        if(!L.lib)throw std::runtime_error("Metal compilation (transcendental): "+message(e));
        const void* t=detail::transcendental_tables(W);
        L.tables=[device newBufferWithBytes:t length:table_bytes(W) options:MTLResourceStorageModeShared];
        if(!bits_buffer){const word* q=detail::two_over_pi_bits();std::size_t n=std::size_t(two_over_pi_length/32+2);
            bits_buffer=[device newBufferWithBytes:q length:4*n options:MTLResourceStorageModeShared];}
        if(!L.tables||!bits_buffer)throw std::runtime_error("Metal allocation failed (transcendental tables)");
        return libs.emplace(bits,L).first->second;
    }
    id<MTLComputePipelineState> pipeline(int bits,Function f){
        Library& L=library(bits);auto it=L.pipes.find(int(f));if(it!=L.pipes.end())return it->second;
        NSString* name=f==Function::atan2?@"lf_atan2":f==Function::complex_powi?@"lf_powi":function_is_complex(f)?@"lf_complex":@"lf_unary";
        MTLFunctionConstantValues* v=[MTLFunctionConstantValues new];int op=int(f);[v setConstantValue:&op type:MTLDataTypeInt atIndex:0];
        NSError* e=nil;id<MTLFunction> fn=[L.lib newFunctionWithName:name constantValues:v error:&e];
        if(!fn)throw std::runtime_error("missing Metal function: "+message(e));
        id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:fn error:&e];
        if(!p)throw std::runtime_error("Metal pipeline: "+message(e));
        return L.pipes.emplace(int(f),p).first->second;
    }
    Timing dispatch(int bits,Function f,const void* a,const void* b,const std::int32_t* k,std::size_t k_count,void* out,std::size_t count);
};
static_assert(sizeof(tr::Tables<10>)==4*(8+15+3*13+3*65*13),"table layout");
Transcendentals::Transcendentals(TranscendentalOptions options):impl(std::make_unique<Impl>()){
    impl->options=options;impl->device=MTLCreateSystemDefaultDevice();if(!impl->device)throw std::runtime_error("no Metal GPU available");
    impl->queue=[impl->device newCommandQueue];if(!impl->queue)throw std::runtime_error("cannot create Metal command queue");
}
Transcendentals::~Transcendentals()=default;
std::string Transcendentals::device_name()const{return [[impl->device name] UTF8String];}
const TranscendentalReport& Transcendentals::report()const{return impl->report;}
double Transcendentals::prewarm(int bits,Function f){@autoreleasepool{check_bits(bits);auto s=Clock::now();impl->pipeline(bits,f);return since(s);}}
Timing Transcendentals::run(int bits,Function f,const void* a,void* out,std::size_t count,const void* b){
    if(f==Function::complex_powi)throw std::invalid_argument("use Transcendentals::powi");
    if(f==Function::atan2&&!b&&count)throw std::invalid_argument("atan2 needs the x operand");
    return impl->dispatch(bits,f,a,b,nullptr,0,out,count);
}
Timing Transcendentals::powi(int bits,const void* z,const std::int32_t* k,std::size_t k_count,void* out,std::size_t count){
    if(count&&(!k||(k_count!=1&&k_count!=count)))throw std::invalid_argument("powi: k_count must be 1 or count");
    return impl->dispatch(bits,Function::complex_powi,z,nullptr,k,k_count,out,count);
}
Timing Transcendentals::Impl::dispatch(int bits,Function f,const void* a,const void* b,const std::int32_t* k,std::size_t k_count,void* out,std::size_t count){@autoreleasepool{
    check_bits(bits);if(int(f)<0||int(f)>int(Function::complex_powi))throw std::invalid_argument("transcendental: invalid function");
    if(count>=(std::size_t(1)<<31))throw std::invalid_argument("transcendental: count exceeds 32-bit indexing");
    auto start=Clock::now();report=TranscendentalReport{};report.count=count;if(!count)return {0,0};
    auto cs=Clock::now();id<MTLComputePipelineState> p=pipeline(bits,f);report.compile_seconds=since(cs);
    const std::size_t elem=std::size_t(bits/8+12)*(function_is_complex(f)?2:1),bytes=elem*count;
    id<MTLBuffer> A=buffer("a",bytes),O=buffer("out",bytes),R=buffer("retry",4*(count+1)),Bb=nil,K=nil;
    std::memcpy(A.contents,a,bytes);std::memset(R.contents,0,4);
    if(f==Function::atan2){Bb=buffer("b",bytes);std::memcpy(Bb.contents,b,bytes);}
    if(f==Function::complex_powi){K=buffer("k",4*k_count);std::memcpy(K.contents,k,4*k_count);}
    Library& L=library(bits);
    id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:p];std::uint32_t n32=std::uint32_t(count),ks=k_count==1?0u:1u;
    [enc setBuffer:A offset:0 atIndex:0];[enc setBuffer:(Bb?Bb:A) offset:0 atIndex:1];[enc setBuffer:O offset:0 atIndex:2];
    [enc setBuffer:L.tables offset:0 atIndex:3];[enc setBuffer:bits_buffer offset:0 atIndex:4];
    [enc setBuffer:R offset:0 atIndex:5];[enc setBuffer:R offset:4 atIndex:6];[enc setBytes:&n32 length:4 atIndex:7];
    [enc setBuffer:(K?K:A) offset:0 atIndex:8];[enc setBytes:&ks length:4 atIndex:9];
    NSUInteger tg=std::min<NSUInteger>(std::max(1u,options.threads_per_threadgroup),p.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];[enc endEncoding];
    [cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("transcendental: Metal command failed: "+message(cb.error));
    report.gpu_seconds=cb.GPUEndTime-cb.GPUStartTime;
    std::memcpy(out,O.contents,bytes);
    std::uint32_t failed;std::memcpy(&failed,R.contents,4);
    if(failed){
        auto rt=Clock::now();report.retried=failed;std::vector<std::uint32_t> list(failed);std::memcpy(list.data(),static_cast<const char*>(R.contents)+4,4*std::size_t(failed));
        std::sort(list.begin(),list.end());Counters c;
        by_words(bits/32,[&](auto tag){constexpr int N=decltype(tag)::value;
            resolve<N>(f,static_cast<const Number<N>*>(A.contents),Bb?static_cast<const Number<N>*>(Bb.contents):nullptr,static_cast<Number<N>*>(out),list.data(),list.size(),options.host_threads,c);});
        for(int r=0;r<3;++r)report.resolved[r]=c.resolved[r];report.unresolved=c.unresolved;report.retry_seconds=since(rt);
    }
    report.wall_seconds=since(start);return {report.gpu_seconds,report.wall_seconds};
}}
}
