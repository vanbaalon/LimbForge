// Element-wise transcendental functions (plan D7; docs/numerics.md, "Transcendental functions"). The GPU evaluates
// at W >= N+2 words and certifies the rounding (src/transcendental_core.hpp); the compacted list of undecided elements is
// re-evaluated on the GPU by up to three retry rungs at more words in the same command buffer (round 44), and the few
// elements left after them by the host ladder here. Constants come from exact integer series computed here.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "limbforge/transcendental.hpp"
#include "engine_internal.hpp"
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
template<int W=4> std::shared_ptr<void> build(int w){
    if constexpr(W<=136){if(w==W)return make_tables<W>();return build<W+1>(w);}
    else throw std::invalid_argument("transcendental tables: unsupported working width");
}
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
// ---- GPU retry rungs (round 44) ----
// Rung r = 1, 2, 3 evaluates the elements left by the previous level at rung_table[N-2][r-1] working words (0: the rung
// and the later ones run on the host, in the final step). Each entry is the smallest width >= N+4, 2N+4, 4N+8 (and above
// the previous level) whose rung kernel (LF_RUNG) matched MPFR/MPC for every element forced through it (all functions,
// random and hard inputs, test_transcendental --force-level r at all 31 precisions), capped at the widest validated
// width, 35 words: every rung kernel at W >= 36 words gave wrong results for every function (core add after a product
// or a division miscompiles in these kernels; docs/gpu-codegen.md section 11). A capped rung is still useful: a width
// is only a heuristic for how many elements a rung decides; certify keeps every decided result correctly rounded.
// The section-9 widths (11-13, 15-17, 19-21, 23-25, 27, 28) are avoided; every other width used passed at every N.
// A rung that does not compile (e.g. complex log at 136 words via the override: "exceeds available stack space") ends
// the GPU levels: it and the later rungs run on the host.
// Diagnostic override (probing): LIMBFORGE_TRANSCENDENTAL_RUNG_WORDS="N:w1/w2/w3,..." replaces the entries of those N
// (increasing widths above the first pass, <= 136; a 0 sends that rung and the later ones to the host, which isolates
// the rung under test with --force-level in tests/test_transcendental.cpp; invalid entries are ignored).
const unsigned char rung_table[31][3]={ // N = 2..32: rungs 1, 2, 3
    {6,8,18},{7,10,22},{8,14,26},{9,14,29},{10,18,32},{14,18,35},{14,22,35},{18,22,35},{18,26,35},{18,26,35},
    {18,29,35},{22,30,35},{22,32,35},{22,34,35},{22,35,0},{26,35,0},{26,35,0},{26,35,0},{26,35,0},{29,35,0},{29,35,0},
    {29,35,0},{29,35,0},{30,35,0},{30,35,0},{31,35,0},{32,35,0},{33,35,0},{34,35,0},{35,0,0},{35,0,0}};
// Words of GPU rungs 1-3 at N words (0: the rung runs on the host).
void rung_words(int N,Function f,int words[3]){
    int w[3]={rung_table[N-2][0],rung_table[N-2][1],rung_table[N-2][2]};
    if(const char* s=std::getenv("LIMBFORGE_TRANSCENDENTAL_RUNG_WORDS"))
        for(const char* p=s;*p;){char* e;long n=std::strtol(p,&e,10);if(e==p||*e!=':')break;long v[3];p=e+1;
            for(int r=0;r<3;++r){v[r]=std::strtol(p,&e,10);p=e;if(r<2&&*p=='/')++p;}
            bool ok=v[0]>gpu_words(N)&&v[0]<=136;long prev=v[0];
            for(int r=1;r<3;++r){if(!v[r])prev=999;else if(prev==999||v[r]<=prev||v[r]>136)ok=false;else prev=v[r];}
            if(n==N&&ok)for(int r=0;r<3;++r)w[r]=int(v[r]);
            if(*p==',')++p;}
    bool gpu=f!=Function::complex_powi;
    for(int r=0;r<3;++r){if(!w[r])gpu=false;words[r]=gpu?w[r]:0;}
}
std::atomic<int> forced_level{0};
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
void host_rungs(int N,int words[3]){words[0]=ladder_at_least(N+4);words[1]=ladder_at_least(2*N+4);words[2]=ladder_at_least(4*N+8);}
// Element i of a real (one Number) or complex (two Numbers) array.
template<int N> void operands(Function f,const Number<N>* A,const Number<N>* Bv,std::size_t i,Number<N>& a,Number<N>& b){
    if(function_is_complex(f)){a=A[2*i];b=A[2*i+1];}else{a=A[i];b=f==Function::atan2?Bv[i]:zero<N>();}
}
// Host retry ladder for the listed elements (widths >= N+4, 2N+4, 4N+8 words); get(q, a, b) gives the operands of the
// q-th listed element. Writes every listed element (the RN fallback when no rung decides).
template<int N,class Get> void resolve(Function f,Get get,Number<N>* out,const std::uint32_t* list,std::size_t n,unsigned threads,Counters& c){
    int rungs[3];host_rungs(N,rungs);const std::size_t stride=function_is_complex(f)?2:1;
    parallel_for(n,threads,16,[&](std::size_t lo,std::size_t hi){for(std::size_t q=lo;q<hi;++q){std::size_t i=list[q];Number<N> a,b,o[2];get(q,a,b);
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
void transcendental_force_level(int level){forced_level.store(std::max(0,level));}
}
void transcendental_cpu(int bits,Function f,const void* a,void* out,std::size_t count,const void* b,unsigned threads,TranscendentalReport* report){
    check_bits(bits);if(int(f)<0||int(f)>int(Function::complex_powi))throw std::invalid_argument("transcendental: invalid function");
    if(f==Function::atan2&&!b&&count)throw std::invalid_argument("atan2 needs the x operand");
    if(f==Function::complex_powi&&!b&&count)throw std::invalid_argument("complex_powi needs the k array");
    auto start=Clock::now();TranscendentalReport rep;rep.count=count;
    by_words(bits/32,[&](auto tag){constexpr int N=decltype(tag)::value;
        if(f==Function::complex_powi){powi_cpu<N>(static_cast<const Complex<N>*>(a),static_cast<const std::int32_t*>(b),count,static_cast<Complex<N>*>(out),count,threads);return;}
        host_rungs(N,rep.rung_words);rep.first_words=ladder_at_least(N+2);
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
        auto rt=Clock::now();Counters c;
        resolve<N>(f,[&](std::size_t q,Number<N>& x,Number<N>& y){operands(f,A,Bv,failed[q],x,y);},O,failed.data(),failed.size(),threads,c);
        for(int r=0;r<3;++r)rep.resolved[r]=c.resolved[r];rep.unresolved=c.unresolved;rep.retry_seconds=since(rt);
    });
    rep.wall_seconds=since(start);if(report)*report=rep;
}
// ---- GPU passes ----
// Retry-list buffer: counters[4] (undecided after levels 0-3), indirect dispatch arguments of rungs 1-3 (3 words each, at
// byte 16+12(r-1)), then from byte 64 lists A and B of count entries; level l appends to list l % 2 and reads (l-1) % 2.
constexpr std::size_t list_header=64;
std::size_t list_offset(int level,std::size_t count){return list_header+std::size_t(level%2)*4*count;}
std::size_t list_bytes(std::size_t count){return list_header+8*count;}
struct Level { std::uint32_t count,level,force,last; }; // = Level of src/transcendental.metal
// Levels of one function at one precision: 0 = first pass, 1.. = GPU rungs (words[l] working words).
struct Plan { int levels=0;int words[4]={};id<MTLComputePipelineState> pipe[4],prepare;id<MTLBuffer> tables[4]; };
struct PassBuffers { id<MTLBuffer> x,y,out,lists,saved; };
// Report of the GPU levels from the counters; returns the number of elements left for the host final step.
std::size_t gpu_report(TranscendentalReport& rep,const Plan& P,const std::uint32_t* c){
    rep.retried=c[0];rep.first_words=P.words[0];for(int r=0;r<3;++r)rep.rung_words[r]=r+1<P.levels?P.words[r+1]:0;
    for(int l=1;l<P.levels;++l)rep.resolved[l-1]=c[l-1]-c[l];
    return c[P.levels-1];
}
// Host final step: the host ladder for the listed elements, operands from `saved` (by list slot; elements left after the
// last GPU level) or, with saved = nullptr, from the operand arrays A, B (by element index; host-array calls).
void host_final(int bits,Function f,const void* saved,const void* A,const void* B,const std::uint32_t* list,std::size_t n,void* out,unsigned threads,TranscendentalReport& rep){
    if(!n)return;auto rt=Clock::now();std::vector<std::uint32_t> sorted(list,list+n);std::vector<std::uint32_t> order(n);
    for(std::size_t q=0;q<n;++q)order[q]=std::uint32_t(q);std::sort(order.begin(),order.end(),[&](std::uint32_t x,std::uint32_t y){return list[x]<list[y];});
    for(std::size_t q=0;q<n;++q)sorted[q]=list[order[q]];
    const bool two=function_is_complex(f)||f==Function::atan2;Counters c;
    by_words(bits/32,[&](auto tag){constexpr int N=decltype(tag)::value;const auto* S=static_cast<const Number<N>*>(saved);
        const auto* X=static_cast<const Number<N>*>(A);const auto* Y=static_cast<const Number<N>*>(B);
        resolve<N>(f,[&](std::size_t q,Number<N>& a,Number<N>& b){std::size_t s=order[q];
                if(!S)operands(f,X,Y,sorted[q],a,b);else if(two){a=S[2*s];b=S[2*s+1];}else{a=S[s];b=zero<N>();}},
                   static_cast<Number<N>*>(out),sorted.data(),n,threads,c);});
    for(int r=0;r<3;++r)rep.resolved[3]+=c.resolved[r];rep.unresolved=c.unresolved;rep.retry_seconds=since(rt);
}
struct Library { id<MTLLibrary> lib;std::map<int,id<MTLComputePipelineState>> pipes;id<MTLBuffer> tables; };
struct Transcendentals::Impl {
    TranscendentalOptions options;TranscendentalReport report;id<MTLDevice> device;id<MTLCommandQueue> queue;
    std::map<std::pair<int,int>,Library> libs;std::map<std::pair<int,int>,Plan> plans;id<MTLBuffer> bits_buffer;std::map<std::string,id<MTLBuffer>> pool;
    // As Engine::Impl: lookups hold the lock, compilation does not; the first stored object is kept (map nodes are stable).
    std::mutex cache_mutex;
    id<MTLBuffer> buffer(const std::string& role,std::size_t bytes){
        auto& b=pool[role];if(!b||b.length<bytes){b=nil;b=[device newBufferWithLength:std::max<std::size_t>(bytes,16) options:MTLResourceStorageModeShared];}
        if(!b)throw std::runtime_error("Metal allocation failed ("+role+")");return b;}
    // One library per (bits, working words) with its constant tables.
    Library& library(int bits,int W){
        {std::lock_guard<std::mutex> lock(cache_mutex);auto found=libs.find({bits,W});if(found!=libs.end())return found->second;}
        Library L;
        NSString* source=[NSString stringWithFormat:@"#define LF_BITS %d\n#define LF_W %d\n%s",bits,W,limbforge_transcendental_source];
        MTLCompileOptions* o=[MTLCompileOptions new];o.languageVersion=MTLLanguageVersion((4u<<16)|0u);o.mathMode=MTLMathModeSafe;
        NSError* e=nil;L.lib=[device newLibraryWithSource:source options:o error:&e];
        if(!L.lib)throw std::runtime_error("Metal compilation (transcendental): "+message(e));
        const void* t=detail::transcendental_tables(W);
        L.tables=[device newBufferWithBytes:t length:table_bytes(W) options:MTLResourceStorageModeShared];
        const word* q=detail::two_over_pi_bits();std::size_t n=std::size_t(two_over_pi_length/32+2);
        id<MTLBuffer> bits_copy=[device newBufferWithBytes:q length:4*n options:MTLResourceStorageModeShared];
        if(!L.tables||!bits_copy)throw std::runtime_error("Metal allocation failed (transcendental tables)");
        std::lock_guard<std::mutex> lock(cache_mutex);if(!bits_buffer)bits_buffer=bits_copy;
        return libs.emplace(std::make_pair(bits,W),L).first->second;
    }
    // The kernel of f in the (bits, W) library: first pass, or retry rung (LF_RUNG). Throws if it does not compile.
    id<MTLComputePipelineState> pipeline(int bits,int W,Function f,bool rung){
        Library& L=library(bits,W);const int key=2*int(f)+(rung?1:0);
        {std::lock_guard<std::mutex> lock(cache_mutex);auto it=L.pipes.find(key);if(it!=L.pipes.end())return it->second;}
        NSString* name=f==Function::atan2?@"lf_atan2":f==Function::complex_powi?@"lf_powi":function_is_complex(f)?@"lf_complex":@"lf_unary";
        MTLFunctionConstantValues* v=[MTLFunctionConstantValues new];int op=int(f);bool list=rung;
        [v setConstantValue:&op type:MTLDataTypeInt atIndex:0];[v setConstantValue:&list type:MTLDataTypeBool atIndex:1];
        NSError* e=nil;id<MTLFunction> fn=[L.lib newFunctionWithName:name constantValues:v error:&e];
        if(!fn)throw std::runtime_error("missing Metal function: "+message(e));
        id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:fn error:&e];
        if(!p)throw std::runtime_error("Metal pipeline (transcendental, "+std::to_string(W)+" words): "+message(e));
        std::lock_guard<std::mutex> lock(cache_mutex);return L.pipes.emplace(key,p).first->second;
    }
    // lf_prepare of the (bits, W) library (indirect dispatch arguments of a rung).
    id<MTLComputePipelineState> prepare_pipeline(int bits,int W){
        Library& L=library(bits,W);const int key=-1;
        {std::lock_guard<std::mutex> lock(cache_mutex);auto it=L.pipes.find(key);if(it!=L.pipes.end())return it->second;}
        NSError* e=nil;id<MTLFunction> fn=[L.lib newFunctionWithName:@"lf_prepare"];
        id<MTLComputePipelineState> p=fn?[device newComputePipelineStateWithFunction:fn error:&e]:nil;
        if(!p)throw std::runtime_error("Metal pipeline (transcendental prepare): "+message(e));
        std::lock_guard<std::mutex> lock(cache_mutex);return L.pipes.emplace(key,p).first->second;
    }
    // First pass and GPU rungs of f at bits, compiled concurrently on first use. A rung whose pipeline does not compile
    // ends the GPU levels (it and later rungs run on the host).
    Plan plan(int bits,Function f){
        {std::lock_guard<std::mutex> lock(cache_mutex);auto it=plans.find({bits,int(f)});if(it!=plans.end())return it->second;}
        Plan P;const int N=bits/32;int rw[3];rung_words(N,f,rw);P.words[0]=gpu_words(N);int want=1;
        for(int r=0;r<3&&rw[r];++r)P.words[want++]=rw[r];
        std::string errors[4];std::vector<std::thread> compile;
        auto job=[&](int l){@autoreleasepool{try{P.pipe[l]=pipeline(bits,P.words[l],f,l>0);P.tables[l]=library(bits,P.words[l]).tables;}
            catch(const std::exception& ex){errors[l]=ex.what();}}};
        for(int l=1;l<want;++l)compile.emplace_back(job,l);
        job(0);for(auto& t:compile)t.join();
        if(!errors[0].empty())throw std::runtime_error(errors[0]);
        if(want>1)P.prepare=prepare_pipeline(bits,P.words[0]);
        P.levels=1;while(P.levels<want&&errors[P.levels].empty())++P.levels;
        for(int l=P.levels;l<4;++l){P.pipe[l]=nil;P.tables[l]=nil;P.words[l]=0;}
        std::lock_guard<std::mutex> lock(cache_mutex);return plans.emplace(std::make_pair(bits,int(f)),P).first->second;
    }
    NSUInteger group(id<MTLComputePipelineState> p)const{return std::min<NSUInteger>(std::max(1u,options.threads_per_threadgroup),p.maxTotalThreadsPerThreadgroup);}
    // Encodes the levels of P; next() returns the encoder for one dispatch (ordered after the previous one, with a barrier).
    // Levels [lo, hi) of P (default: all).
    template<class Next> void encode_levels(const Plan& P,Next next,const PassBuffers& b,std::size_t count,int lo=0,int hi=-1){
        id<MTLBuffer> bits;{std::lock_guard<std::mutex> lock(cache_mutex);bits=bits_buffer;}
        const std::uint32_t force=std::uint32_t(std::min(forced_level.load(),4));if(hi<0)hi=P.levels;
        for(int l=lo;l<hi;++l){
            const NSUInteger tg=group(P.pipe[l]);const std::size_t args=16+12*std::size_t(l>0?l-1:0);
            if(l){id<MTLComputeCommandEncoder> enc=next();std::uint32_t pp[2]={std::uint32_t(l-1),std::uint32_t(tg)};
                [enc setComputePipelineState:P.prepare];[enc setBuffer:b.lists offset:0 atIndex:0];[enc setBuffer:b.lists offset:args atIndex:1];
                [enc setBytes:pp length:sizeof pp atIndex:2];[enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];}
            id<MTLComputeCommandEncoder> enc=next();Level lv{std::uint32_t(count),std::uint32_t(l),force,l==P.levels-1?1u:0u};
            [enc setComputePipelineState:P.pipe[l]];
            [enc setBuffer:b.x offset:0 atIndex:0];[enc setBuffer:(b.y?b.y:b.x) offset:0 atIndex:1];[enc setBuffer:b.out offset:0 atIndex:2];
            [enc setBuffer:P.tables[l] offset:0 atIndex:3];[enc setBuffer:bits offset:0 atIndex:4];
            [enc setBuffer:b.lists offset:0 atIndex:5];[enc setBuffer:b.lists offset:list_offset(l,count) atIndex:6];
            [enc setBytes:&lv length:sizeof lv atIndex:7];[enc setBuffer:b.saved offset:0 atIndex:8];
            [enc setBuffer:b.lists offset:list_offset(l?l-1:0,count) atIndex:9];
            if(l)[enc dispatchThreadgroupsWithIndirectBuffer:b.lists indirectBufferOffset:args threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
            else [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        }
    }
    Timing dispatch(int bits,Function f,const void* a,const void* b,const std::int32_t* k,std::size_t k_count,void* out,std::size_t count);
};
static_assert(sizeof(tr::Tables<10>)==4*(8+15+3*13+3*65*13),"table layout");
static_assert(sizeof(Level)==16,"Level layout");
Transcendentals::Transcendentals(TranscendentalOptions options):impl(std::make_unique<Impl>()){
    impl->options=options;impl->device=MTLCreateSystemDefaultDevice();if(!impl->device)throw std::runtime_error("no Metal GPU available");
    impl->queue=[impl->device newCommandQueue];if(!impl->queue)throw std::runtime_error("cannot create Metal command queue");
}
Transcendentals::Transcendentals(Engine& engine,TranscendentalOptions options):impl(std::make_unique<Impl>()){
    impl->options=options;impl->device=detail::Internal::device(engine);
    impl->queue=[impl->device newCommandQueue];if(!impl->queue)throw std::runtime_error("cannot create Metal command queue");
}
Transcendentals::~Transcendentals()=default;
std::string Transcendentals::device_name()const{return [[impl->device name] UTF8String];}
const TranscendentalReport& Transcendentals::report()const{return impl->report;}
double Transcendentals::prewarm(int bits,Function f){@autoreleasepool{check_bits(bits);if(int(f)<0||int(f)>int(Function::complex_powi))throw std::invalid_argument("transcendental: invalid function");
    auto s=Clock::now();impl->plan(bits,f);return since(s);}}
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
    auto cs=Clock::now();Plan P=plan(bits,f);report.compile_seconds=since(cs);
    const bool powi=f==Function::complex_powi,two=function_is_complex(f)||f==Function::atan2;
    const std::size_t number=std::size_t(bits/8+12),bytes=number*(function_is_complex(f)?2:1)*count;
    id<MTLBuffer> A=buffer("a",bytes),O=buffer("out",bytes),Bb=nil,K=nil,R=nil,S=nil;
    std::memcpy(A.contents,a,bytes);
    if(f==Function::atan2){Bb=buffer("b",bytes);std::memcpy(Bb.contents,b,bytes);}
    id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    if(powi){
        K=buffer("k",4*k_count);std::memcpy(K.contents,k,4*k_count);std::uint32_t n32=std::uint32_t(count),ks=k_count==1?0u:1u;
        [enc setComputePipelineState:P.pipe[0]];[enc setBuffer:A offset:0 atIndex:0];[enc setBuffer:O offset:0 atIndex:2];
        [enc setBytes:&n32 length:4 atIndex:7];[enc setBuffer:K offset:0 atIndex:8];[enc setBytes:&ks length:4 atIndex:9];
        [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(group(P.pipe[0]),1,1)];
    }
    // With a retry threshold the first pass runs alone: up to `gpu_retry_threshold` undecided elements are resolved by the
    // host ladder (lower latency than a GPU rung), more by the GPU rungs in a second command buffer.
    const bool split=!powi&&P.levels>1&&options.gpu_retry_threshold>0;
    auto run=[&](id<MTLCommandBuffer> c,id<MTLComputeCommandEncoder> e){[e endEncoding];[c commit];[c waitUntilCompleted];
        if(c.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("transcendental: Metal command failed: "+message(c.error));
        report.gpu_seconds+=c.GPUEndTime-c.GPUStartTime;};
    if(!powi){
        R=buffer("lists",list_bytes(count));S=buffer("saved",number*(two?2:1)*count);std::memset(R.contents,0,16);
        bool first=true;encode_levels(P,[&]{if(!first)[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];first=false;return enc;},PassBuffers{A,Bb,O,R,S},count,0,split?1:-1);
    }
    run(cb,enc);
    const auto* lists=static_cast<const char*>(R?R.contents:nullptr);std::uint32_t c[4]={};if(R)std::memcpy(c,lists,16);
    bool host_only=split&&c[0]<=options.gpu_retry_threshold;
    if(split&&!host_only){
        id<MTLCommandBuffer> cb2=[queue commandBuffer];id<MTLComputeCommandEncoder> enc2=[cb2 computeCommandEncoder];bool first=true;
        encode_levels(P,[&]{if(!first)[enc2 memoryBarrierWithScope:MTLBarrierScopeBuffers];first=false;return enc2;},PassBuffers{A,Bb,O,R,S},count,1,-1);
        run(cb2,enc2);std::memcpy(c,lists,16);
    }
    std::memcpy(out,O.contents,bytes);
    if(host_only){report.retried=c[0];report.first_words=P.words[0];
        host_final(bits,f,nullptr,A.contents,Bb?Bb.contents:nullptr,reinterpret_cast<const std::uint32_t*>(lists+list_offset(0,count)),c[0],out,options.host_threads,report);}
    else if(!powi){std::size_t left=gpu_report(report,P,c);
        host_final(bits,f,S.contents,nullptr,nullptr,reinterpret_cast<const std::uint32_t*>(lists+list_offset(P.levels-1,count)),left,out,options.host_threads,report);}
    report.wall_seconds=since(start);return {report.gpu_seconds,report.wall_seconds};
}}
// ---- Resident passes ----
namespace detail { struct TranscendentalPass { TranscendentalReport report;std::atomic<bool> done{false}; }; }
bool TranscendentalTicket::resolved()const{return pass_&&pass_->done.load(std::memory_order_acquire);}
const TranscendentalReport& TranscendentalTicket::report()const{
    if(!resolved())throw std::logic_error("transcendental pass not resolved: wait for the submission that contains it");return pass_->report;}
TranscendentalTicket Transcendentals::encode(CommandBatch& batch,int bits,bool complex,Function f,const detail::Operand& a,const detail::Operand& b,
                                             const detail::Operand& k,const detail::Operand& out){@autoreleasepool{
    using detail::Internal;
    if(Internal::device(batch)!=impl->device)throw std::invalid_argument("unit and batch use different Metal devices (construct the unit from the batch's Engine)");
    check_bits(bits);if(int(f)<0||int(f)>int(Function::complex_powi))throw std::invalid_argument("transcendental: invalid function");
    if(function_is_complex(f)!=complex)throw std::invalid_argument("transcendental: function and buffer format mismatch");
    const bool two=f==Function::atan2,powi=f==Function::complex_powi;
    if(two!=bool(b.storage))throw std::invalid_argument(two?"atan2 needs the x operand":"only atan2 takes a second operand");
    if(!a.storage||!out.storage||(powi&&a.size&&!k.storage))throw std::invalid_argument("transcendental: missing buffer");
    const std::size_t count=a.size;
    if(out.size!=count||(two&&b.size!=count))throw std::invalid_argument("transcendental: buffer counts must match");
    if(powi&&count&&k.size!=1&&k.size!=count)throw std::invalid_argument("powi: k must hold 1 or count values");
    if(count>=(std::size_t(1)<<31))throw std::invalid_argument("transcendental: count exceeds 32-bit indexing");
    Internal::retain(batch,a.storage);if(two)Internal::retain(batch,b.storage);if(powi&&k.storage)Internal::retain(batch,k.storage);Internal::retain(batch,out.storage,true);
    auto pass=std::make_shared<detail::TranscendentalPass>();pass->report.count=count;
    auto cs=Clock::now();Plan P;if(count)P=impl->plan(bits,f);pass->report.compile_seconds=count?since(cs):0;
    if(!count||powi){ // no retries: final when the GPU completes
        if(count){Internal::keep(batch,P.pipe[0]);auto enc=Internal::compute(batch);std::uint32_t n32=std::uint32_t(count),ks=k.size==1?0u:1u;
            [enc setComputePipelineState:P.pipe[0]];[enc setBuffer:a.storage->buffer offset:0 atIndex:0];[enc setBuffer:out.storage->buffer offset:0 atIndex:2];
            [enc setBytes:&n32 length:4 atIndex:7];[enc setBuffer:k.storage->buffer offset:0 atIndex:8];[enc setBytes:&ks length:4 atIndex:9];
            [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(impl->group(P.pipe[0]),1,1)];}
        Internal::on_completion(batch,[pass]{pass->done.store(true,std::memory_order_release);});return TranscendentalTicket(pass);
    }
    // The levels read the operands directly: they run before any later operation of the batch, a level writes only decided
    // elements (an aliased input of an undecided element stays intact), and the last level saves the operands of the
    // elements it leaves for the host.
    const std::size_t number=std::size_t(bits/8+12);
    auto R=Internal::scratch(batch,list_bytes(count));std::memset(R->buffer.contents,0,16);
    auto S=Internal::scratch(batch,number*(complex||two?2:1)*count);
    for(int l=0;l<P.levels;++l){Internal::keep(batch,P.pipe[l]);Internal::keep(batch,P.tables[l]);}Internal::keep(batch,P.prepare);
    {std::lock_guard<std::mutex> lock(impl->cache_mutex);Internal::keep(batch,impl->bits_buffer);}
    impl->encode_levels(P,[&]{return Internal::compute(batch);},PassBuffers{a.storage->buffer,two?b.storage->buffer:nil,out.storage->buffer,R->buffer,S->buffer},count);
    // `out` stays write-protected in the batch: the host final step may still patch it at wait().
    auto flag=Internal::provisional(batch,out.storage);auto O=out.storage;const unsigned threads=impl->options.host_threads;
    Internal::on_completion(batch,[pass,flag,R,S,O,P,f,bits,count,threads]{
        TranscendentalReport& rep=pass->report;std::uint32_t c[4];std::memcpy(c,R->buffer.contents,16);std::size_t left=gpu_report(rep,P,c);
        host_final(bits,f,S->buffer.contents,nullptr,nullptr,reinterpret_cast<const std::uint32_t*>(static_cast<const char*>(R->buffer.contents)+list_offset(P.levels-1,count)),left,
                   O->buffer.contents,threads,rep);
        rep.in_batch_reads=flag->load();rep.provisional_reads=rep.in_batch_reads&&rep.host_retried()>0;pass->done.store(true,std::memory_order_release);});
    return TranscendentalTicket(pass);
}}
}
