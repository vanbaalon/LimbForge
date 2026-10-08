// Dense real GEMM/SYRK with one rounding per output (plan D6 via L1c). Exact integer GEMMs of exponent bands
// through int8 TensorOps residues (benchmarks/residue_gemm.mm), Garner reconstruction and one rounding on the
// GPU; lines whose exponents need too many bands or span too much use the exact host path (exact_dot).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "limbforge/linalg.hpp"
#include "linalg_source.hpp"
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <unistd.h>
#include <stdexcept>
#include <thread>

namespace limbforge {
namespace {
using Clock=std::chrono::steady_clock;
double since(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
std::string message(NSError* e){return e?std::string([[e localizedDescription] UTF8String]):"unknown Metal error";}
struct Params {std::uint32_t rows,cols,K,Kp,Mp,Np,j,L,Wc,tri,lower,fold,ls_line,ls_k,rs_line,rs_k,out_cols,multi_rows,multi_cols,both,j0,group,upper,row0,col0,out_stride;};
struct Line {std::int32_t bands,low,slot;};
struct Member {std::int32_t line,lo,hi;};
constexpr std::size_t max_k=65472; // concatenated digit product: 2*Kp*2^14 < 2^31
constexpr int group=4;            // moduli per digit pass (each input entry is read once per group)
// Bytes of digit planes per modulus batch; LIMBFORGE_LINALG_PLANE_BUDGET overrides it (testing).
std::size_t plane_budget(){static const std::size_t b=[]{const char* e=std::getenv("LIMBFORGE_LINALG_PLANE_BUDGET");return e?std::size_t(std::strtoull(e,nullptr,10)):std::size_t(64)<<20;}();return b;}
std::size_t up(std::size_t x,std::size_t m){return (x+m-1)/m*m;}
int padded(int words){return words>=13&&words<33?33:words;} // docs/gpu-codegen.md rule 1
// Primes <= 65279, descending: balanced residues then split into two int8 digits.
const std::vector<std::uint32_t>& primes(){static const std::vector<std::uint32_t> p=[]{std::vector<std::uint32_t> r;
    for(std::uint32_t q=65279;r.size()<1200;--q){bool prime=true;for(std::uint32_t d=2;d*d<=q;++d)if(q%d==0){prime=false;break;}if(prime)r.push_back(q);}return r;}();return p;}
std::uint32_t inverse(std::uint64_t a,std::uint64_t m){std::int64_t t=0,nt=1,r=std::int64_t(m),nr=std::int64_t(a%m);
    while(nr){std::int64_t q=r/nr,x=t-q*nt;t=nt;nt=x;x=r-q*nr;r=nr;nr=x;}return std::uint32_t(t<0?t+std::int64_t(m):t);}
// Persistent host workers. run(n, f) calls f(i, worker) once for every item i < n. Items are claimed by whichever threads
// are running (the caller is worker 0), and run returns as soon as every item is done: a worker that the (often loaded) host
// does not schedule in time does not delay the call, it later finds no item left. The batch is shared state, so a late
// worker never touches the caller's frame; f is called only for claimed items, all finished before run returns.
class Pool {
    struct Batch {std::function<void(std::size_t,unsigned)> f;std::size_t n=0;std::atomic<std::size_t> next{0},done{0};};
    std::vector<std::thread> workers;std::mutex m;std::condition_variable wake,finished;std::shared_ptr<Batch> batch;std::size_t generation=0;bool stop=false;
    void work(Batch& b,unsigned id){for(std::size_t i;(i=b.next.fetch_add(1))<b.n;){b.f(i,id);if(b.done.fetch_add(1)+1==b.n){std::lock_guard<std::mutex> l(m);finished.notify_all();}}}
public:
    explicit Pool(unsigned n){for(unsigned i=1;i<n;++i)workers.emplace_back([this,i]{std::size_t seen=0;std::unique_lock<std::mutex> l(m);
        for(;;){wake.wait(l,[&]{return stop||generation!=seen;});if(stop)return;seen=generation;auto b=batch;l.unlock();work(*b,i);l.lock();}});}
    ~Pool(){{std::lock_guard<std::mutex> l(m);stop=true;}wake.notify_all();for(auto& w:workers)w.join();}
    unsigned size()const{return unsigned(workers.size())+1;}
    void run(std::size_t n,std::function<void(std::size_t,unsigned)> f){
        if(!n)return;if(workers.empty()||n==1){for(std::size_t i=0;i<n;++i)f(i,0);return;}
        auto b=std::make_shared<Batch>();b->f=std::move(f);b->n=n;
        {std::lock_guard<std::mutex> l(m);batch=b;++generation;}
        if(n-1<workers.size())for(std::size_t i=0;i+1<n;++i)wake.notify_one();else wake.notify_all();
        work(*b,0);
        if(b->done.load()<n){std::unique_lock<std::mutex> l(m);finished.wait(l,[&]{return b->done.load()==n;});}}
};
thread_local Pool* current_pool=nullptr; // the calling Linalg's workers (one Linalg per host thread)
thread_local bool side_thread=false;  // this thread runs a factorization's trailing update (Linalg::Impl::pool_workers)
template<class F> void parallel(unsigned threads,std::size_t n,F f){
    unsigned t=unsigned(std::min<std::size_t>(threads?threads:1,n));if(t<=1){if(n)f(0,n);return;}
    if(current_pool){t=std::min(t,current_pool->size());current_pool->run(t,[&](std::size_t q,unsigned){f(n*q/t,n*(q+1)/t);});return;}
    std::vector<std::thread> w;for(unsigned i=0;i<t;++i)w.emplace_back(f,n*i/t,n*(i+1)/t);for(auto& x:w)x.join();}
// Strided view of a host matrix: element (line, k) at base[line*line_stride + k*k_stride], each `bytes` long.
struct View {
    const unsigned char* base;std::size_t bytes,line_stride,k_stride;int words;
    const unsigned char* at(std::size_t line,std::size_t k)const{return base+(line*line_stride+k*k_stride)*bytes;}
    void fields(std::size_t line,std::size_t k,int& exponent,int& sign,word& status)const{
        const unsigned char* p=at(line,k)+4*words;std::memcpy(&exponent,p,4);std::memcpy(&sign,p+4,4);std::memcpy(&status,p+8,4);}
};
// Line classes: 0 GPU bands, 1 zero or status (trivial result), 2 exact host fallback.
struct Side {
    std::vector<Line> line;std::vector<word> status;std::vector<std::uint8_t> cls;std::vector<std::vector<Member>> band;
    std::size_t multi=0,single=0,fallback=0;std::vector<std::size_t> histogram;
};
Side analyze(const View& v,std::size_t lines,std::size_t K,const LinalgOptions& o,bool force,unsigned threads){
    Side s;s.line.assign(lines,{0,0,0});s.status.assign(lines,0);s.cls.assign(lines,1);
    std::vector<std::vector<std::pair<long,long>>> bands(lines);const long G=o.band_bits;
    // Pass 1 in storage order: status and the largest and smallest nonzero exponent of every line.
    std::vector<long> top(lines,LONG_MIN),bottom(lines,LONG_MAX);std::mutex merge;
    auto scan=[&](std::size_t l0,std::size_t l1,std::size_t k0,std::size_t k1,word* st,long* hi,long* lo){int e,sg;word x;
        for(std::size_t k=k0;k<k1;++k)for(std::size_t i=l0;i<l1;++i){v.fields(i,k,e,sg,x);st[i]|=x;if(!x&&sg){hi[i]=std::max(hi[i],long(e));lo[i]=std::min(lo[i],long(e));}}};
    if(v.k_stride==1)parallel(threads,lines,[&](std::size_t a,std::size_t b){for(std::size_t i=a;i<b;++i)scan(i,i+1,0,K,s.status.data(),top.data(),bottom.data());});
    else parallel(threads,K,[&](std::size_t a,std::size_t b){std::vector<word> st(lines,0);std::vector<long> hi(lines,LONG_MIN),lo(lines,LONG_MAX);scan(0,lines,a,b,st.data(),hi.data(),lo.data());
        std::lock_guard<std::mutex> lock(merge);for(std::size_t i=0;i<lines;++i){s.status[i]|=st[i];top[i]=std::max(top[i],hi[i]);bottom[i]=std::min(bottom[i],lo[i]);}});
    // Greedy bands [hi-G, hi] from the largest remaining exponent down (lo: smallest exponent inside); one band needs no further pass.
    parallel(threads,lines,[&](std::size_t first,std::size_t last){for(std::size_t i=first;i<last;++i){
        if(s.status[i]||top[i]==LONG_MIN)continue;
        if(force){s.cls[i]=2;continue;}
        auto& b=bands[i];if(o.max_bands>=1&&top[i]-bottom[i]<=G){b.push_back({bottom[i],top[i]});s.cls[i]=0;continue;}
        long hi=top[i];bool fallback=top[i]-bottom[i]>o.max_spread;int e,sg;word x;
        while(!fallback){long lo=hi,next=LONG_MIN;
            for(std::size_t k=0;k<K;++k){v.fields(i,k,e,sg,x);if(x||!sg||e>hi)continue;if(e>=hi-G)lo=std::min(lo,long(e));else next=std::max(next,long(e));}
            b.push_back({lo,hi});if(int(b.size())>o.max_bands)fallback=true;if(next==LONG_MIN)break;hi=next;}
        s.cls[i]=fallback?2:0;
    }});
    s.histogram.assign(std::size_t(std::max(o.max_bands,0))+1,0);
    for(std::size_t i=0;i<lines;++i){
        if(s.cls[i]==1){++s.histogram[0];continue;}if(s.cls[i]==2){++s.fallback;continue;}
        int nb=int(bands[i].size());s.line[i]={nb,int(bands[i].back().first),nb>1?int(s.multi++):-1-int(s.single++)};++s.histogram[nb];
        if(s.band.size()<std::size_t(nb))s.band.resize(nb);
        for(int b=0;b<nb;++b)s.band[b].push_back({int(i),int(bands[i][b].first),int(bands[i][b].second)});
    }
    return s;
}
// out = exact_dot(a, b), or with sub the update RN(out - dot) (out is read first).
using DotFn=void(*)(const void*,std::ptrdiff_t,const void*,std::ptrdiff_t,std::size_t,void*,bool);
template<int N> void dot_n(const void* a,std::ptrdiff_t sa,const void* b,std::ptrdiff_t sb,std::size_t K,void* out,bool sub){
    auto* o=static_cast<Number<N>*>(out);Number<N> c=*o;
    *o=exact_dot_add<N>(sub?&c:nullptr,sub,static_cast<const Number<N>*>(a),sa,static_cast<const Number<N>*>(b),sb,K);}
template<int N> void dot_table(DotFn* t){t[N]=dot_n<N>;if constexpr(N<32)dot_table<N+1>(t);}
DotFn dot_function(int words){static DotFn table[33]={};if(!table[2])dot_table<2>(table);return table[words];}
// f(std::integral_constant<int, words>) for words in [2, 32].
template<int N=2,class F> auto by_words(int words,F&& f){if constexpr(N<32){if(words!=N)return by_words<N+1>(words,f);}return f(std::integral_constant<int,N>{});}
}
// Factor storage: zeroed, page-aligned and a whole number of pages, so the GPU uses it in place (wrap).
namespace detail {
struct PageBlock {
    void* p=nullptr;std::size_t bytes=0;
    PageBlock()=default;
    explicit PageBlock(std::size_t n){if(!n)return;std::size_t page=std::size_t(getpagesize());bytes=up(n,page);if(posix_memalign(&p,page,bytes))throw std::bad_alloc();std::memset(p,0,bytes);}
    ~PageBlock(){std::free(p);}
    PageBlock(PageBlock&& o)noexcept:p(o.p),bytes(o.bytes){o.p=nullptr;o.bytes=0;}
    PageBlock& operator=(PageBlock&& o)noexcept{std::swap(p,o.p);std::swap(bytes,o.bytes);return *this;}
};
struct QRData {Linalg* owner=nullptr;int bits=0;std::size_t m=0,n=0,nb=0;QRInfo info;QROptions options;PageBlock v,r,t,tau;std::vector<std::size_t> perm;};
}
struct Kernels {
    id<MTLComputePipelineState> left,right,product_res,reconstruct,finish,reconstruct_sub,finish_sub;int max_moduli,term_words,acc_words;
    id<MTLBuffer> moduli,recips,inv;std::map<int,std::vector<std::uint32_t>> bounds; // per modulus count: [M/2 | M], Wc words each
};
struct Linalg::Impl {
    LinalgOptions options;LinalgReport report;id<MTLDevice> device;id<MTLCommandQueue> queue;
    std::map<int,Kernels> kernels;std::map<std::string,id<MTLBuffer>> pool;std::unique_ptr<Pool> workers;
    // Buffers that the factorization drivers keep on the GPU: views and outputs inside them are used in place.
    std::vector<id<MTLBuffer>> resident;
    unsigned threads()const{return options.host_threads?options.host_threads:std::max(1u,std::thread::hardware_concurrency());}
    // The factorization runs trailing updates on a second host thread (side_thread) while the caller's thread factors
    // the next panel; each thread then uses its own worker pool.
    std::unique_ptr<Pool> side_workers;
    Pool& pool_workers(){auto& w=side_thread?side_workers:workers;if(!w)w=std::make_unique<Pool>(side_thread?std::max(1u,threads()/4):threads());current_pool=w.get();return *w;}
    // f(i) for i in [0, n), dynamically scheduled over the persistent workers.
    template<class F> void each(std::size_t n,F f){if(!n)return;if(n==1){f(std::size_t(0));return;}
        pool_workers().run(n,[&](std::size_t i,unsigned){f(i);});}
    id<MTLBuffer> find(const void* p,std::size_t& offset){auto c=static_cast<const unsigned char*>(p);
        for(id<MTLBuffer> b:resident){auto base=static_cast<const unsigned char*>(b.contents);if(c>=base&&c<base+b.length){offset=std::size_t(c-base);return b;}}return nil;}
    // Page-aligned caller memory is wrapped without a copy (the call waits for the GPU before returning).
    id<MTLBuffer> wrap(const void* p,std::size_t bytes){std::size_t page=std::size_t(getpagesize());if(!p||!bytes||reinterpret_cast<std::uintptr_t>(p)%page)return nil;
        return [device newBufferWithBytesNoCopy:const_cast<void*>(p) length:up(bytes,page) options:MTLResourceStorageModeShared deallocator:nil];}
    id<MTLBuffer> buffer(const std::string& role,std::size_t bytes){
        auto& b=pool[role];if(!b||b.length<bytes){b=nil;b=[device newBufferWithLength:std::max<std::size_t>(bytes,16) options:MTLResourceStorageModeShared];}
        if(!b)throw std::runtime_error("Metal allocation failed ("+role+")");return b;}
    id<MTLComputePipelineState> state(id<MTLLibrary> lib,NSString* name,int sub=-1){NSError* e=nil;id<MTLFunction> f=nil;
        if(sub<0)f=[lib newFunctionWithName:name];
        else{MTLFunctionConstantValues* v=[MTLFunctionConstantValues new];bool s=sub;[v setConstantValue:&s type:MTLDataTypeBool atIndex:0];f=[lib newFunctionWithName:name constantValues:v error:&e];}
        if(!f)throw std::runtime_error("missing Metal function");auto p=[device newComputePipelineStateWithFunction:f error:&e];
        if(!p)throw std::runtime_error("Metal pipeline: "+message(e));return p;}
    // Moduli: enough for two (bits+G)-bit integers summed over max_k terms.
    Kernels& get(int bits){
        auto found=kernels.find(bits);if(found!=kernels.end())return found->second;
        Kernels k;const int N=bits/32;double need=2.0*(bits+options.band_bits)+std::log2(double(max_k))+2,have=0;int L=0;
        while(have<=need)have+=std::log2(double(primes()[L++]));
        k.max_moduli=padded(L);for(int i=L;i<k.max_moduli;++i)have+=std::log2(double(primes()[i]));
        k.term_words=padded(int(std::ceil(have/32))+1);k.acc_words=padded(2*N+(2*options.max_spread+18+31)/32+1);
        // Garner reduces v_i mod m_j with one conditional subtraction: every modulus must exceed half the largest.
        if(2*primes()[k.max_moduli+group]<=primes()[0])throw std::logic_error("linalg: modulus range too wide");
        NSString* source=[NSString stringWithFormat:@"#define LF_BITS %d\n#define LF_TERM_WORDS %d\n#define LF_ACC_WORDS %d\n#define LF_MAX_MODULI %d\n#define LF_GROUP %d\n%s",
                          bits,k.term_words,k.acc_words,k.max_moduli,group,limbforge_linalg_source];
        MTLCompileOptions* o=[MTLCompileOptions new];o.languageVersion=MTLLanguageVersion((4u<<16)|0u);o.mathMode=MTLMathModeSafe;
        NSError* e=nil;id<MTLLibrary> lib=[device newLibraryWithSource:source options:o error:&e];
        if(!lib)throw std::runtime_error("Metal compilation (linalg): "+message(e));
        k.left=state(lib,@"digits_left");k.right=state(lib,@"digits_right");k.product_res=state(lib,@"product_residue");
        k.reconstruct=state(lib,@"reconstruct",0);k.finish=state(lib,@"finish",0);k.reconstruct_sub=state(lib,@"reconstruct",1);k.finish_sub=state(lib,@"finish",1);
        std::vector<std::uint32_t> inv(std::size_t(k.max_moduli)*k.max_moduli,0);const auto& m=primes();
        for(int i=0;i<k.max_moduli;++i)for(int j=i+1;j<k.max_moduli;++j)inv[std::size_t(i)*k.max_moduli+j]=inverse(m[i],m[j]);
        std::vector<std::uint32_t> c(k.max_moduli+group);for(std::size_t i=0;i<c.size();++i)c[i]=std::uint32_t((std::uint64_t(1)<<32)/m[i]);
        k.moduli=[device newBufferWithBytes:m.data() length:4*c.size() options:MTLResourceStorageModeShared];
        k.recips=[device newBufferWithBytes:c.data() length:4*c.size() options:MTLResourceStorageModeShared];
        k.inv=[device newBufferWithBytes:inv.data() length:4*inv.size() options:MTLResourceStorageModeShared];
        if(!k.moduli||!k.recips||!k.inv)throw std::runtime_error("Metal allocation failed (moduli)");
        return kernels.emplace(bits,std::move(k)).first->second;
    }
    static const std::vector<std::uint32_t>& bounds(Kernels& k,int L,int Wc){
        auto& b=k.bounds[L];if(!b.empty())return b;std::vector<std::uint32_t> x(Wc,0);x[0]=1;
        for(int j=0;j<L;++j){std::uint64_t c=0;for(auto& w:x){std::uint64_t t=std::uint64_t(w)*primes()[j]+c;w=std::uint32_t(t);c=t>>32;}}
        b.assign(2*Wc,0);for(int i=0;i<Wc;++i){b[i]=(x[i]>>1)|(i+1<Wc?x[i+1]<<31:0);b[Wc+i]=x[i];}return b;
    }
    // C[i*stride + j] (stride 0: Nc) = RN(dot), or RN(C - dot) with sub. Views and C inside a resident buffer are used in place.
    Timing run(int bits,View lv,View rv,std::size_t M,std::size_t Nc,std::size_t K,void* C,bool syrk,bool lower_only,bool sub=false,std::size_t stride=0);
    template<int N> CholeskyInfo cholesky_n(const Number<N>* A,std::size_t n,Number<N>* L,const FactorOptions& fo);
    template<int N> Timing solve_n(const Number<N>* L,std::size_t n,const Number<N>* B,std::size_t nrhs,Number<N>* X,const FactorOptions& fo,int passes,bool upper=false);
    // Householder QR (qr_n), one compact-WY block applied to columns [c0, c1) of X (qr_block), Q or Q^T applied to C (qr_apply_n).
    template<int N> void qr_n(const Number<N>* A,std::size_t m,std::size_t n,detail::QRData& f);
    template<int N> Timing qr_block(const detail::QRData& f,std::size_t b,Number<N>* X,std::size_t ld,std::size_t c0,std::size_t c1,bool qt,Number<N>* Wb,Number<N>* Yb,double threshold,bool gpu,bool& on_gpu,bool have_y=false);
    template<int N> Timing qr_apply_n(const detail::QRData& f,Number<N>* C,std::size_t nrhs,bool qt);
    template<int N> Timing qr_solve_n(const detail::QRData& f,const Number<N>* B,std::size_t nrhs,Number<N>* X);
    ~Impl(){if((workers&&current_pool==workers.get())||(side_workers&&current_pool==side_workers.get()))current_pool=nullptr;}
    // Restores the resident list on scope exit.
    struct Residency {Impl& im;std::size_t size;explicit Residency(Impl& i):im(i),size(i.resident.size()){} ~Residency(){im.resident.resize(size);}};
};
Linalg::Linalg(LinalgOptions options):impl(std::make_unique<Impl>()){
    if(options.band_bits<0||options.band_bits>1024||options.max_bands<0||options.max_spread<0||options.max_spread>8192)
        throw std::invalid_argument("linalg options: band_bits in [0,1024], max_bands >= 0, max_spread in [0,8192]");
    impl->options=options;impl->device=MTLCreateSystemDefaultDevice();if(!impl->device)throw std::runtime_error("no Metal GPU available");
    impl->queue=[impl->device newCommandQueue];if(!impl->queue)throw std::runtime_error("cannot create Metal command queue");
}
Linalg::~Linalg()=default;
std::string Linalg::device_name()const{return [[impl->device name] UTF8String];}
const LinalgReport& Linalg::report()const{return impl->report;}
const LinalgOptions& Linalg::options()const{return impl->options;}
static void check_bits(int bits){if(bits<64||bits>1024||bits%32)throw std::invalid_argument("bits must be a multiple of 32 in [64,1024]");}
Timing Linalg::syrk(int bits,const void* A,std::size_t rows,std::size_t cols,void* C,bool lower_only,bool subtract){
    check_bits(bits);if(subtract&&!lower_only)throw std::invalid_argument("syrk: subtract updates the lower triangle only (lower_only)");
    std::size_t e=bits/8+12;auto a=static_cast<const unsigned char*>(A);
    return impl->run(bits,{a,e,1,cols,bits/32},{a,e,1,cols,bits/32},cols,cols,rows,C,true,lower_only,subtract);
}
Timing Linalg::gemm(int bits,bool transpose_a,const void* A,const void* B,std::size_t m,std::size_t n,std::size_t k,void* C,bool subtract){
    check_bits(bits);std::size_t e=bits/8+12;
    View lv=transpose_a?View{static_cast<const unsigned char*>(A),e,1,m,bits/32}:View{static_cast<const unsigned char*>(A),e,k,1,bits/32};
    return impl->run(bits,lv,{static_cast<const unsigned char*>(B),e,1,n,bits/32},m,n,k,C,false,false,subtract);
}
Timing Linalg::Impl::run(int bits,View lv,View rv,std::size_t M,std::size_t Nc,std::size_t K,void* C,bool syrk,bool lower_only,bool sub,std::size_t stride){@autoreleasepool{
    auto start=Clock::now();report=LinalgReport{};report.m=M;report.n=Nc;report.k=K;if(!stride)stride=Nc;pool_workers();
    const std::size_t limit=std::size_t(1)<<31;if(M>=limit||Nc>=limit||K>=limit||M*stride>=limit)throw std::invalid_argument("matrix dimensions exceed 32-bit indexing");
    if(M&&Nc&&!C)throw std::invalid_argument("null output");if(M*Nc&&K&&(!lv.base||!rv.base))throw std::invalid_argument("null input");
    const int N=bits/32;const std::size_t eb=lv.bytes;const unsigned nt=threads();
    auto t=Clock::now();bool force=options.force_fallback||K>max_k;
    Side ls=analyze(lv,M,K,options,force,nt),rs_own;if(!syrk)rs_own=analyze(rv,Nc,K,options,force,nt);
    const Side& rs=syrk?ls:rs_own;
    report.left_bands=ls.histogram;report.right_bands=rs.histogram;report.left_fallback=ls.fallback;report.right_fallback=rs.fallback;
    report.analysis_seconds=since(t);
    Timing timing{0,0};id<MTLBuffer> out=nil;std::size_t out_offset=0;
    if(!ls.band.empty()&&!rs.band.empty()){
        Kernels& kern=get(bits);t=Clock::now();
        // Inputs (one copy for SYRK), band members (segments aligned to 64 members), line tables, output, accumulators.
        // Resident (factorization) operands are strided views of a larger buffer; public operands are contiguous.
        std::size_t a_offset=0,b_offset=0;id<MTLBuffer> in_a=find(lv.base,a_offset),in_b=nil;
        if(!in_a&&!(in_a=wrap(lv.base,M*K*eb))){in_a=buffer("a",M*K*eb);std::memcpy(in_a.contents,lv.base,M*K*eb);}
        if(syrk){in_b=in_a;b_offset=a_offset;}
        else if(!(in_b=find(rv.base,b_offset))&&!(in_b=wrap(rv.base,K*Nc*eb))){in_b=buffer("b",K*Nc*eb);std::memcpy(in_b.contents,rv.base,K*Nc*eb);}
        // Extended operands: all bands of a side concatenated (band b of the left starts at row loff[b]); one integer GEMM per
        // modulus then holds every band pair as a sub-block. SYRK shares one extended operand.
        std::vector<Member> members;std::vector<std::size_t> loff,roff;
        for(auto& b:ls.band){loff.push_back(members.size());members.insert(members.end(),b.begin(),b.end());}
        const std::size_t Me=members.size(),rbase=syrk?0:up(Me,64);
        if(!syrk){members.resize(rbase);for(auto& b:rs.band){roff.push_back(members.size()-rbase);members.insert(members.end(),b.begin(),b.end());}}else roff=loff;
        const std::size_t Ne=syrk?Me:members.size()-rbase;
        id<MTLBuffer> mem=buffer("members",members.size()*sizeof(Member));std::memcpy(mem.contents,members.data(),members.size()*sizeof(Member));
        id<MTLBuffer> lline=buffer("lline",M*sizeof(Line)),rline=lline;std::memcpy(lline.contents,ls.line.data(),M*sizeof(Line));
        if(!syrk){rline=buffer("rline",Nc*sizeof(Line));std::memcpy(rline.contents,rs.line.data(),Nc*sizeof(Line));}
        // An update reads the old C on the GPU: a copied output starts as a copy of C.
        if((out=find(C,out_offset))||(stride==Nc&&(out=wrap(C,M*Nc*eb))))report.zero_copy_output=true;
        else if(stride!=Nc)throw std::logic_error("linalg: strided output outside a resident buffer");
        else{out=buffer("out",M*Nc*eb);if(sub)std::memcpy(out.contents,C,M*Nc*eb);}
        std::size_t slots=ls.multi*Nc+ls.single*rs.multi;id<MTLBuffer> acc=buffer("acc",slots*kern.acc_words*4);
        // Moduli for |sum| < K * 2^(Pb+Pc) over all pairs, P = bits + widest band. SYRK reconstructs pairs b >= c only:
        // an off-diagonal pair also supplies the transposed (c, b) terms (fold).
        const std::size_t Kp=up(std::max<std::size_t>(K,1),64),Mp=up(Me,64),Np=up(Ne,64);int logk=0;while((std::size_t(1)<<logk)<K)++logk;
        int pmax=0;for(const Side* side:{static_cast<const Side*>(&ls),&rs})for(auto& b:side->band)for(auto& x:b)pmax=std::max(pmax,x.hi-x.lo);
        double need=2.0*bits+2*pmax+logk+2,have=0;unsigned L=0;while(have<=need)have+=std::log2(double(primes()[L++]));
        if(int(L)>kern.max_moduli)throw std::logic_error("linalg: modulus count exceeds the compiled limit");
        const unsigned Wc=unsigned(std::ceil(have/32))+1;auto& bw=bounds(kern,int(L),int(Wc));report.gemm_rows=Me;report.gemm_cols=Ne;report.int8_gemms=3*L;
        struct Pair{unsigned b,c;};std::vector<Pair> plan;
        for(unsigned b=0;b<ls.band.size();++b)for(unsigned c=0;c<(syrk?b+1:rs.band.size());++c){plan.push_back({b,c});
            report.pairs.push_back({b,c,unsigned(ls.band[b].size()),unsigned(rs.band[c].size()),L});}
        id<MTLBuffer> bnd=buffer("bounds",bw.size()*4);std::memcpy(bnd.contents,bw.data(),bw.size()*4);
        // Moduli are processed in batches of digit planes (per batch member: left cat = [d1 | d0] (Mp x 2Kp), d1 and d0 (Mp x Kp);
        // right [d0 ; d1] (2Kp x Np)); one product_residue dispatch per batch forms the three int8 products of every tile and
        // modulus in registers and writes only the residues. Small products take every modulus in one batch; large ones are
        // limited by plane_budget(). The residues, and hence the results, do not depend on the batching.
        const std::size_t hp=Mp*Kp,rp=2*Kp*Np,gp=Mp*Np;
        const unsigned B=std::min<unsigned>(L,std::max<unsigned>(group,unsigned(plane_budget()/(4*hp+rp))/group*group));
        id<MTLBuffer> lcat=buffer("lcat",B*2*hp),a1=buffer("a1",B*hp),a0=buffer("a0",B*hp),rcat=buffer("rcat",B*rp),res=buffer("res",std::size_t(L)*gp*2);
        report.upload_seconds=since(t);
        id<MTLCommandBuffer> cb=[queue commandBuffer];
        if(slots){id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];[blit fillBuffer:acc range:NSMakeRange(0,slots*kern.acc_words*4) value:0];[blit endEncoding];}
        id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
        [enc setBuffer:kern.moduli offset:0 atIndex:11];[enc setBuffer:kern.recips offset:0 atIndex:12];
        auto grid=[&](id<MTLComputePipelineState> p,std::size_t w,std::size_t h,std::size_t d=1){if(!w||!h||!d)return;[enc setComputePipelineState:p];
            [enc dispatchThreads:MTLSizeMake(w,h,d) threadsPerThreadgroup:MTLSizeMake(32,std::min<NSUInteger>(8,p.maxTotalThreadsPerThreadgroup/32),1)];};
        Params p{};p.rows=std::uint32_t(Me);p.cols=std::uint32_t(Ne);p.K=std::uint32_t(K);p.Kp=std::uint32_t(Kp);p.Mp=std::uint32_t(Mp);p.Np=std::uint32_t(Np);
        p.L=L;p.Wc=Wc;p.tri=syrk;p.lower=syrk;p.both=syrk;p.ls_line=std::uint32_t(lv.line_stride);p.ls_k=std::uint32_t(lv.k_stride);
        p.rs_line=std::uint32_t(rv.line_stride);p.rs_k=std::uint32_t(rv.k_stride);p.out_cols=std::uint32_t(Nc);p.multi_rows=std::uint32_t(ls.multi);p.multi_cols=std::uint32_t(rs.multi);p.out_stride=std::uint32_t(stride);
        const std::uint32_t st[4]={std::uint32_t(2*hp),std::uint32_t(hp),std::uint32_t(rp),0};
        for(unsigned j0=0;j0<L;j0+=B){
            // Digits of the extended operands for moduli j0 .. j0+count-1 (grid z: groups of `group` moduli), then the residues.
            const unsigned count=std::min(B,L-j0),groups=(count+group-1)/group;p.j0=j0;p.group=count;p.j=j0;[enc setBytes:&p length:sizeof(p) atIndex:5];
            [enc setBuffer:in_a offset:a_offset atIndex:0];[enc setBuffer:mem offset:0 atIndex:1];[enc setBuffer:lcat offset:0 atIndex:2];[enc setBuffer:a1 offset:0 atIndex:3];
            [enc setBuffer:a0 offset:0 atIndex:4];[enc setBuffer:rcat offset:0 atIndex:6];grid(kern.left,Kp,Mp,groups);
            if(!syrk){[enc setBuffer:in_b offset:b_offset atIndex:0];[enc setBuffer:mem offset:rbase*sizeof(Member) atIndex:1];[enc setBuffer:rcat offset:0 atIndex:2];grid(kern.right,Np,Kp,groups);}
            [enc setComputePipelineState:kern.product_res];
            [enc setBuffer:lcat offset:0 atIndex:0];[enc setBuffer:a1 offset:0 atIndex:1];[enc setBuffer:a0 offset:0 atIndex:2];[enc setBuffer:rcat offset:0 atIndex:3];
            [enc setBuffer:res offset:0 atIndex:4];[enc setBytes:st length:16 atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(Np/32,Mp/64,count) threadsPerThreadgroup:MTLSizeMake(kern.product_res.threadExecutionWidth*4,1,1)];
        }
        [enc setBuffer:res offset:0 atIndex:0];[enc setBuffer:kern.inv offset:0 atIndex:2];[enc setBuffer:bnd offset:0 atIndex:3];[enc setBuffer:mem offset:0 atIndex:4];
        [enc setBuffer:mem offset:rbase*sizeof(Member) atIndex:6];[enc setBuffer:lline offset:0 atIndex:7];[enc setBuffer:rline offset:0 atIndex:8];
        [enc setBuffer:out offset:out_offset atIndex:9];[enc setBuffer:acc offset:0 atIndex:10];
        for(auto& pr:plan){Params q=p;q.rows=std::uint32_t(ls.band[pr.b].size());q.cols=std::uint32_t(rs.band[pr.c].size());q.row0=std::uint32_t(loff[pr.b]);
            q.col0=std::uint32_t(roff[pr.c]);q.tri=syrk&&pr.b==pr.c;q.fold=syrk&&pr.b>pr.c;q.upper=0;[enc setBytes:&q length:sizeof(q) atIndex:5];grid(sub?kern.reconstruct_sub:kern.reconstruct,q.cols,q.rows);
            if(q.fold){q.upper=1;[enc setBytes:&q length:sizeof(q) atIndex:5];grid(sub?kern.reconstruct_sub:kern.reconstruct,q.cols,q.rows);}}
        if(slots){Params q=p;q.rows=std::uint32_t(M);q.cols=std::uint32_t(Nc);[enc setBytes:&q length:sizeof(q) atIndex:5];grid(sub?kern.finish_sub:kern.finish,Nc,M);}
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.status==MTLCommandBufferStatusError)throw std::runtime_error("Metal execution (linalg): "+message(cb.error));
        timing.gpu_seconds=report.gpu_seconds=cb.GPUEndTime-cb.GPUStartTime;
    }
    // Assembly: GPU outputs, trivial zeros with statuses, and exact host dots for fallback lines.
    t=Clock::now();std::atomic<std::size_t> next{0},gpu_count{0},multi_count{0},fallback_count{0},trivial_count{0};std::atomic<double> fallback_time{0};
    const unsigned char* o=out?static_cast<const unsigned char*>(out.contents)+out_offset:nullptr;auto* c=static_cast<unsigned char*>(C);DotFn dot=dot_function(N);
    auto worker=[&](std::size_t,std::size_t){std::size_t g=0,mu=0,fb=0,tr=0;double ft=0;
        for(std::size_t i;(i=next.fetch_add(1))<M;)for(std::size_t j=0;j<(syrk?i+1:Nc);++j){
            unsigned char* dst=c+(i*stride+j)*eb;std::uint8_t ci=ls.cls[i],cj=rs.cls[j];
            // A zero line leaves an update unchanged; a status line gives zero with the statuses (and the old entry's).
            if(ci==1||cj==1){word st=ls.status[i]|rs.status[j],old=0;std::memcpy(&old,dst+4*N+8,4);if(sub)st|=old;
                if(!sub||st){std::memset(dst,0,eb);std::memcpy(dst+4*N+8,&st,4);}++tr;}
            else if(ci==2||cj==2){auto f=Clock::now();dot(lv.at(i,0),std::ptrdiff_t(lv.k_stride),rv.at(j,0),std::ptrdiff_t(rv.k_stride),K,dst,sub);ft+=since(f);++fb;}
            else{if(o!=c)std::memcpy(dst,o+(i*stride+j)*eb,eb);++g;mu+=ls.line[i].bands>1||rs.line[j].bands>1;}
        }
        gpu_count+=g;multi_count+=mu;fallback_count+=fb;trivial_count+=tr;double old=fallback_time.load();while(!fallback_time.compare_exchange_weak(old,old+ft)){}};
    parallel(nt,nt,worker);
    if(syrk&&!lower_only)parallel(nt,M,[&](std::size_t first,std::size_t last){for(std::size_t i=first;i<last;++i)for(std::size_t j=i+1;j<Nc;++j)std::memcpy(c+(i*Nc+j)*eb,c+(j*Nc+i)*eb,eb);});
    report.gpu_outputs=gpu_count;report.multi_band_outputs=multi_count;report.fallback_outputs=fallback_count;report.trivial_outputs=trivial_count;
    report.fallback_seconds=fallback_time;report.assembly_seconds=since(t);timing.wall_seconds=since(start);return timing;
}}
// ---- Blocked Cholesky and triangular solves (docs/numerics.md, "Cholesky factorization") ----
// D(c; x, y; K) = RN(c - sum_{k<K} x_k y_k) with one rounding, on the host (exact_dot_add) or the GPU (run with sub);
// the two give identical results, so the split below only changes speed.
template<int N> CholeskyInfo Linalg::Impl::cholesky_n(const Number<N>* A,std::size_t n,Number<N>* L,const FactorOptions& fo){@autoreleasepool{
    using T=Number<N>;auto start=Clock::now();CholeskyInfo info;info.pivot=n;const std::size_t nb=fo.block?std::min(fo.block,n):n;
    Residency keep(*this);T* W=L;pool_workers();
    // GPU work matrix: L itself when page aligned, otherwise a shared scratch copy.
    const bool gpu=fo.gpu&&n>nb&&double(n-nb)*double(n-nb+1)/2*double(nb)>=fo.host_macs;
    if(gpu){id<MTLBuffer> b=wrap(L,n*n*sizeof(T));if(!b){b=buffer("factor",n*n*sizeof(T));W=static_cast<T*>(b.contents);}resident.push_back(b);}
    each(n,[&](std::size_t i){for(std::size_t j=0;j<n;++j)W[i*n+j]=j>i?zero<N>():A[i*n+j];});
    auto D=[](const T& c,const T* x,std::ptrdiff_t sx,const T* y,std::ptrdiff_t sy,std::size_t K){return exact_dot_add<N>(&c,true,x,sx,y,sy,K);};
    // Columns [k0, ke) of row i: l_ij = RN(D(a_ij; l_i, l_j; [k0, j)) / l_jj), rows j < i of the block already final.
    auto panel=[&](std::size_t i,std::size_t k0,std::size_t ke){T* r=W+i*n;for(std::size_t j=k0;j<ke;++j)r[j]=div(D(r[j],r+k0,1,W+j*n+k0,1,j-k0),W[j*n+j]);};
    // Panel [k0, k1) on the host: the diagonal block row by row, then the rows below it (independent given the
    // diagonal block, in parallel). Returns the first failing pivot, or k1.
    auto factor_panel=[&](std::size_t k0,std::size_t k1){auto t=Clock::now();std::size_t p=k1;++info.blocks;
        for(std::size_t i=k0;i<k1&&p==k1;++i){panel(i,k0,i);T s=D(W[i*n+i],W+i*n+k0,1,W+i*n+k0,1,i-k0);
            if(s.status||s.sign<=0){p=i;info.pivot=i;info.pivot_sign=s.status?0:s.sign;info.pivot_status=s.status;}else W[i*n+i]=sqrt(s);}
        const std::size_t first=p<k1?p+1:k1;each(n-first,[&](std::size_t r){panel(first+r,k0,p);});
        info.panel_seconds+=since(t);return p;};
    // Update by block [k0, k1): a_ij = D(a_ij; l_i, l_j; [k0, k1)) for rows i >= r0 and columns j in [r0, c1), j <= i.
    // r0 = c1's block start with c1 < n: the next panel (a GEMM; its strict upper part is restored to zero); otherwise
    // the lower triangle from r0 (a SYRK). Returns {GPU seconds, wall seconds}.
    auto update=[&](std::size_t k0,std::size_t k1,std::size_t r0,std::size_t c1,bool& on_gpu){
        auto t=Clock::now();const std::size_t m=n-r0,w=c1-r0,K=k1-k0;const bool square=c1==n;Timing tm{0,0};
        on_gpu=gpu&&(square?double(m)*double(m+1)/2:double(m)*double(w))*double(K)>=fo.host_macs;
        if(on_gpu){auto e=reinterpret_cast<const unsigned char*>(W+r0*n+k0);View v{e,sizeof(T),n,1,N};
            tm=run(32*N,v,v,m,square?m:w,K,W+r0*n+r0,square,square,true,n);
            if(!square)for(std::size_t i=r0;i<c1;++i)for(std::size_t j=i+1;j<c1;++j)W[i*n+j]=zero<N>();}
        else each(m,[&](std::size_t r){std::size_t i=r0+r;for(std::size_t j=r0;j<=i&&j<c1;++j)W[i*n+j]=D(W[i*n+j],W+i*n+k0,1,W+j*n+k0,1,K);});
        tm.wall_seconds=since(t);return tm;};
    auto account=[&](Timing tm,bool on_gpu){info.timing.gpu_seconds+=tm.gpu_seconds;info.update_seconds+=tm.wall_seconds;++(on_gpu?info.gpu_updates:info.host_updates);};
    // Look-ahead: after panel [k0, k1), update the next panel's columns [k1, k2) first, then factor that panel on this
    // thread while the rest of the trailing matrix (from k2) is updated on a second thread (GPU). The regions are disjoint
    // and every update is one exact rounding, so the overlap does not change any result.
    std::size_t k0=0,k1=std::min(n,nb),p=n?factor_panel(0,k1):0;
    while(p==k1&&k1<n){const std::size_t k2=std::min(n,k1+nb);bool on_gpu=false;
        if(k2<n){account(update(k0,k1,k1,k2,on_gpu),on_gpu);
            Timing rest{0,0};bool rest_gpu=false,async=gpu;std::thread side;std::exception_ptr failed;
            if(async)side=std::thread([&]{@autoreleasepool{side_thread=true;try{rest=update(k0,k1,k2,n,rest_gpu);}catch(...){failed=std::current_exception();}side_thread=false;}});
            try{p=factor_panel(k1,k2);}catch(...){if(async)side.join();throw;}
            if(async){side.join();if(failed)std::rethrow_exception(failed);}else rest=update(k0,k1,k2,n,rest_gpu);
            account(rest,rest_gpu);}
        else{account(update(k0,k1,k1,n,on_gpu),on_gpu);p=factor_panel(k1,k2);}
        k0=k1;k1=k2;}
    if(p<n)each(n-p,[&](std::size_t r){std::size_t i=p+r;for(std::size_t j=p;j<=i;++j)W[i*n+j]=zero<N>(invalid);});
    if(W!=L)each(n,[&](std::size_t i){std::memcpy(static_cast<void*>(L+i*n),W+i*n,n*sizeof(T));});
    info.timing.wall_seconds=since(start);return info;
}}
// passes: 1 = L X = B (forward), 2 = L^T X = B (backward), 3 = both (forward first). upper (backward only): L holds an upper
// triangular R = L^T itself (row i of R in place of column i of L; the same sequence).
template<int N> Timing Linalg::Impl::solve_n(const Number<N>* L,std::size_t n,const Number<N>* B,std::size_t nrhs,Number<N>* X,const FactorOptions& fo,int passes,bool upper){@autoreleasepool{
    using T=Number<N>;auto start=Clock::now();Timing timing{0,0};if(!n||!nrhs)return timing;
    const std::size_t nb=fo.block?std::min(fo.block,n):n;Residency keep(*this);const T* F=L;T* W=X;pool_workers();
    const bool gpu=fo.gpu&&n>nb&&double(n-nb)*double(nrhs)*double(nb)>=fo.solve_host_macs;
    if(gpu){id<MTLBuffer> b=wrap(L,n*n*sizeof(T));if(!b){b=buffer("lfactor",n*n*sizeof(T));std::memcpy(b.contents,L,n*n*sizeof(T));F=static_cast<const T*>(b.contents);}resident.push_back(b);
        id<MTLBuffer> x=wrap(X,n*nrhs*sizeof(T));if(!x){x=buffer("rhs",n*nrhs*sizeof(T));W=static_cast<T*>(x.contents);}resident.push_back(x);}
    if(W!=B)std::memcpy(static_cast<void*>(W),B,n*nrhs*sizeof(T));
    auto D=[](const T& c,const T* x,std::ptrdiff_t sx,const T* y,std::ptrdiff_t sy,std::size_t K){return exact_dot_add<N>(&c,true,x,sx,y,sy,K);};
    // Rows out[i] (M x nrhs) = D(out[i]; line i of the factor view, X rows [r0, r0+K)).
    auto update=[&](const T* base,std::size_t line_stride,std::size_t k_stride,std::size_t M,std::size_t r0,std::size_t K,T* out){
        if(!M||!K)return;
        if(gpu&&double(M)*double(nrhs)*double(K)>=fo.solve_host_macs){
            Timing tm=run(32*N,View{reinterpret_cast<const unsigned char*>(base),sizeof(T),line_stride,k_stride,N},View{reinterpret_cast<const unsigned char*>(W+r0*nrhs),sizeof(T),1,nrhs,N},
                          M,nrhs,K,out,false,false,true,nrhs);timing.gpu_seconds+=tm.gpu_seconds;}
        else each(M,[&](std::size_t i){for(std::size_t c=0;c<nrhs;++c)out[i*nrhs+c]=D(out[i*nrhs+c],base+i*line_stride,std::ptrdiff_t(k_stride),W+r0*nrhs+c,std::ptrdiff_t(nrhs),K);});};
    // Forward: x_i = RN(D(b_i; l_i, x; [r0, i)) / l_ii) within block [r0, r1), then b_i = D(b_i; l_i, x; [r0, r1)) for i >= r1.
    if(passes&1)for(std::size_t r0=0;r0<n;r0+=nb){const std::size_t r1=std::min(n,r0+nb);
        each(nrhs,[&](std::size_t c){for(std::size_t i=r0;i<r1;++i){T& x=W[i*nrhs+c];x=div(D(x,F+i*n+r0,1,W+r0*nrhs+c,std::ptrdiff_t(nrhs),i-r0),F[i*n+i]);}});
        update(F+r1*n+r0,n,1,n-r1,r0,r1-r0,W+r1*nrhs);}
    // Backward (L^T): x_i = RN(D(b_i; column i of L, x; (i, r1)) / l_ii) from the last block, then b_i = D(b_i; column i, x; [r0, r1)) for i < r0.
    if(passes&2)for(std::size_t r0=(n-1)/nb*nb;;r0-=nb){const std::size_t r1=std::min(n,r0+nb);
        each(nrhs,[&](std::size_t c){for(std::size_t i=r1;i-->r0;){T& x=W[i*nrhs+c];
            x=div(i+1<r1?D(x,upper?F+i*n+i+1:F+(i+1)*n+i,std::ptrdiff_t(upper?1:n),W+(i+1)*nrhs+c,std::ptrdiff_t(nrhs),r1-i-1):x,F[i*n+i]);}});
        if(upper)update(F+r0,n,1,r0,r0,r1-r0,W);else update(F+r0*n,1,n,r0,r0,r1-r0,W);if(!r0)break;}
    if(W!=X)std::memcpy(static_cast<void*>(X),W,n*nrhs*sizeof(T));
    timing.wall_seconds=since(start);return timing;
}}
// ---- Householder QR (docs/numerics.md, "QR factorization and least squares") ----
// P(x, y; S) = RN(sum x_k y_k) and D(c; x, y; S) = RN(c - sum x_k y_k), one rounding each (exact_dot / exact_dot_add or
// the GPU products: identical bits). Block b holds reflectors [k0, k0+kb), k0 = b nb; V_b, T_b as stored in the factor.
namespace {
// x * 2^-e exactly (an exponent outside the range gives exponent_overflow).
template<int N> Number<N> scaled(Number<N> x,long e){if(!x.sign)return x;long r=long(x.exponent)-e;if(r>1000000000L||r< -1000000000L)return zero<N>(x.status|exponent_overflow);x.exponent=int(r);return x;}
template<int N> Number<N> power2(int e){Number<N> x=zero<N>();x.limb[N-1]=0x80000000u;x.sign=1;x.exponent=e;return x;}
// Split exact dot products (the QR panel): the rows of one dot product are divided into chunks; each chunk adds its products
// exactly into a window anchored at its lowest term (positive and negative parts), and merge_parts aligns the windows and
// rounds the exact total once. That is exact_dot's value (P is one rounding of the exact sum, whatever the grouping). A
// chunk or total wider than the window, or wider than merge's limit, is reported to the caller, which then uses exact_dot.
constexpr int part_cap=128;  // words per chunk window
struct Part {word status;long lo,hi;std::size_t terms;int words;bool wide;word* pos;word* neg;};
// Exact 2N-word integer product of the mantissas (as exact_dot_add's terms).
template<int N> void mantissa_product(const Number<N>& x,const Number<N>& y,word* p){detail::mantissa_mul<N>(x.limb,y.limb,p);}
// Chunk partial of sum_k a[k*sa]*b[k*sb] (k < K): status OR, range of the term exponent sums, exact parts at scale lo.
template<int N> void part_dot(Part& p,const Number<N>* a,std::ptrdiff_t sa,const Number<N>* b,std::ptrdiff_t sb,std::size_t K){
    p.status=0;p.lo=LONG_MAX;p.hi=LONG_MIN;p.terms=0;p.words=0;p.wide=false;
    for(std::size_t k=0;k<K;++k){const auto &x=a[std::ptrdiff_t(k)*sa],&y=b[std::ptrdiff_t(k)*sb];p.status|=x.status|y.status;
        if(x.sign&&y.sign){long e=long(x.exponent)+y.exponent;p.lo=std::min(p.lo,e);p.hi=std::max(p.hi,e);++p.terms;}}
    if(p.status||!p.terms)return;
    int logc=0;while((std::size_t(1)<<logc)<p.terms)++logc;
    const long need=p.hi-p.lo+64*N+logc+1;if(need>32L*(part_cap-2)){p.wide=true;return;}
    p.words=int(need/32)+2;std::fill(p.pos,p.pos+p.words,0);std::fill(p.neg,p.neg+p.words,0);word t[64];
    for(std::size_t k=0;k<K;++k){const auto &x=a[std::ptrdiff_t(k)*sa],&y=b[std::ptrdiff_t(k)*sb];if(!x.sign||!y.sign)continue;
        mantissa_product<N>(x,y,t);detail::add_shifted(x.sign==y.sign?p.pos:p.neg,t,2*N,long(x.exponent)+y.exponent-p.lo);}
}
// out = RN(exact total of parts [0, count)), as exact_dot of the whole range; false when a part or the total is too wide.
// With exact, also the total: magnitude words (at exponent sum *lo), *sign (0 for zero).
template<int N> bool merge_parts(const Part* p,std::size_t count,Number<N>& out,std::vector<word>* exact=nullptr,long* lo_out=nullptr,int* sign_out=nullptr){
    word status=0;bool wide=false;long lo=LONG_MAX,hi=LONG_MIN;std::size_t terms=0;
    for(std::size_t i=0;i<count;++i){status|=p[i].status;wide|=p[i].wide;if(p[i].terms){lo=std::min(lo,p[i].lo);hi=std::max(hi,p[i].hi);terms+=p[i].terms;}}
    if(status){out=zero<N>(status);if(exact){exact->clear();*lo_out=0;*sign_out=0;}return true;}
    if(wide)return false;
    if(!terms){out=zero<N>();if(exact){exact->clear();*lo_out=0;*sign_out=0;}return true;}
    int logc=0;while((std::size_t(1)<<logc)<terms)++logc;
    std::size_t W=std::size_t((hi-lo+64*N+logc+1)/32)+2;
    for(std::size_t i=0;i<count;++i)if(p[i].terms)W=std::max(W,std::size_t((p[i].lo-lo)/32)+std::size_t(p[i].words)+2);
    constexpr std::size_t cap=256;if(W>cap)return false;
    word pos[cap],neg[cap];std::fill(pos,pos+W,0);std::fill(neg,neg+W,0);
    for(std::size_t i=0;i<count;++i)if(p[i].terms){detail::add_shifted(pos,p[i].pos,p[i].words,p[i].lo-lo);detail::add_shifted(neg,p[i].neg,p[i].words,p[i].lo-lo);}
    int cmp=0;for(std::size_t i=W;i-->0;)if(pos[i]!=neg[i]){cmp=pos[i]>neg[i]?1:-1;break;}
    if(!cmp){out=zero<N>();if(exact){exact->clear();*lo_out=0;*sign_out=0;}return true;}
    word* x=cmp>0?pos:neg;const word* y=cmp>0?neg:pos;dword borrow=0;
    for(std::size_t i=0;i<W;++i){dword v=dword(y[i])+borrow;borrow=dword(x[i])<v;x[i]=word(dword(x[i])-v);}
    out=detail::pack_words<N>(x,W,lo-2*(32*N-1),cmp,ok);
    if(exact){exact->assign(x,x+W);*lo_out=lo;*sign_out=cmp;}
    return true;
}
// RN(pos - neg) for exact nonnegative W-word accumulators at exponent sum lo (both are modified); zero(status) for a status.
template<int N> Number<N> round_acc(word* pos,word* neg,std::size_t W,long lo,word status){
    if(status)return zero<N>(status);int cmp=0;for(std::size_t i=W;i-->0;)if(pos[i]!=neg[i]){cmp=pos[i]>neg[i]?1:-1;break;}
    if(!cmp)return zero<N>();word* x=cmp>0?pos:neg;const word* y=cmp>0?neg:pos;dword borrow=0;
    for(std::size_t i=0;i<W;++i){dword d=dword(y[i])+borrow;borrow=dword(x[i])<d;x[i]=word(dword(x[i])-d);}
    return detail::pack_words<N>(x,W,lo-2*(32*N-1),cmp,ok);}
// Exact order of two values for the pivot choice: +1 if a > b, -1 if a < b, 0 if equal; a status ranks above every value.
template<int N> int compare_magnitude(const Number<N>& a,const Number<N>& b){
    if(a.status||b.status)return (a.status!=0)-(b.status!=0);
    if(a.sign!=b.sign)return a.sign>b.sign?1:-1;if(!a.sign)return 0;
    int cmp=a.exponent!=b.exponent?(a.exponent>b.exponent?1:-1):0;
    for(int i=N-1;i>=0&&!cmp;--i)if(a.limb[i]!=b.limb[i])cmp=a.limb[i]>b.limb[i]?1:-1;
    return a.sign>0?cmp:-cmp;}
// P(v, v) for v = 2^-e (v0, x_1, x_2, ...) without forming v: RN(2^-2e (v0^2 + S - x0^2)) for the exact S = sum_{r>=0} x_r^2
// (magnitude words at exponent sum lo). Requires every x_r 2^-e in range (scaled() without a status).
template<int N> Number<N> reflector_norm(const std::vector<word>& S,long lo,const Number<N>& x0,const Number<N>& v0,long e){
    const long sv=2L*v0.exponent,sx=x0.sign?2L*x0.exponent:sv,low=std::min({lo,sv,sx});
    std::vector<word> acc(std::max(S.size()+std::size_t((lo-low)/32),std::size_t((std::max(sv,sx)-low)/32)+2*N)+3,0);word t[64];
    detail::accumulate(acc,S.data(),int(S.size()),lo-low,1);mantissa_product<N>(v0,v0,t);detail::accumulate(acc,t,2*N,sv-low,1);
    if(x0.sign){mantissa_product<N>(x0,x0,t);detail::accumulate(acc,t,2*N,sx-low,-1);}
    return detail::pack_words<N>(acc,low-2*(32*N-1)-2*e,1,ok);
}
}
// Block b applied to columns [c0, c1) of X (rows [k0, m), row stride ld): W = P(V_b^T X), Y = P(T_b^T W) (qt: Q^T) or
// P(T_b W) (Q), X = D(X; V_b, Y). Wb and Yb (nb rows, stride ld) are scratch indexed by the column of X; with have_y,
// Yb already holds Y (the pivoted factorization forms W and Y during its panel) and only X is updated.
template<int N> Timing Linalg::Impl::qr_block(const detail::QRData& f,std::size_t b,Number<N>* X,std::size_t ld,std::size_t c0,std::size_t c1,bool qt,Number<N>* Wb,Number<N>* Yb,double threshold,bool gpu,bool& on_gpu,bool have_y){
    using T=Number<N>;Timing tm{0,0};on_gpu=false;const std::size_t m=f.m,n=f.n,nb=f.nb,k0=b*nb,kb=std::min(n,k0+nb)-k0,c=c1-c0,mr=m-k0;if(!c||!kb)return tm;
    const T* V=static_cast<const T*>(f.v.p)+k0*n+k0;const T* Tb=static_cast<const T*>(f.t.p)+b*nb*nb;T* Xb=X+k0*ld;constexpr std::size_t e=sizeof(T);
    // Each product is placed by its own size (the small Y = P(T^T W) of a narrow update then avoids a GPU round trip).
    auto big=[&](std::size_t rows,std::size_t k){return gpu&&double(rows)*double(c)*double(k)>=threshold;};
    const bool gw=big(kb,mr)&&!have_y,gy=big(kb,kb)&&!have_y,gx=big(mr,kb);on_gpu=gw||gy||gx;
    auto at=[](const void* p){return static_cast<const unsigned char*>(p);};
    if(have_y){}
    else if(gw)tm.gpu_seconds+=run(32*N,View{at(V),e,1,n,N},View{at(Xb+c0),e,1,ld,N},kb,c,mr,Wb+c0,false,false,false,ld).gpu_seconds;
    else each(kb*c,[&](std::size_t q){std::size_t i=q/c,j=c0+q%c;Wb[i*ld+j]=exact_dot<N>(V+i,std::ptrdiff_t(n),Xb+j,std::ptrdiff_t(ld),mr);});
    if(have_y){}
    else if(gy)tm.gpu_seconds+=run(32*N,qt?View{at(Tb),e,1,nb,N}:View{at(Tb),e,nb,1,N},View{at(Wb+c0),e,1,ld,N},kb,c,kb,Yb+c0,false,false,false,ld).gpu_seconds;
    else each(kb*c,[&](std::size_t q){std::size_t i=q/c,j=c0+q%c;
        Yb[i*ld+j]=qt?exact_dot<N>(Tb+i,std::ptrdiff_t(nb),Wb+j,std::ptrdiff_t(ld),kb):exact_dot<N>(Tb+i*nb,1,Wb+j,std::ptrdiff_t(ld),kb);});
    if(gx)tm.gpu_seconds+=run(32*N,View{at(V),e,n,1,N},View{at(Yb+c0),e,1,ld,N},mr,c,kb,Xb+c0,false,false,true,ld).gpu_seconds;
    else each(mr,[&](std::size_t r){for(std::size_t j=c0;j<c1;++j)Xb[r*ld+j]=exact_dot_add<N>(&Xb[r*ld+j],true,V+r*n,1,Yb+j,std::ptrdiff_t(ld),kb);});
    return tm;
}
template<int N> void Linalg::Impl::qr_n(const Number<N>* A,std::size_t m,std::size_t n,detail::QRData& f){@autoreleasepool{
    using T=Number<N>;auto start=Clock::now();QRInfo& info=f.info;info=QRInfo{};info.rank=n;const QROptions& fo=f.options;const std::size_t nb=f.nb;
    Residency keep(*this);pool_workers();T* V=static_cast<T*>(f.v.p);T* Tm=static_cast<T*>(f.t.p);T* tau=static_cast<T*>(f.tau.p);
    // Work matrix (m x n) and the W, Y scratch of the trailing updates (nb x n); resident (with V and T) when the GPU may be used.
    const bool gpu=fo.gpu&&n>nb;std::vector<T> host;T *Wk,*Wb,*Yb;
    if(gpu){id<MTLBuffer> w=buffer("qr_work",m*n*sizeof(T)),wb=buffer("qr_w",nb*n*sizeof(T)),yb=buffer("qr_y",nb*n*sizeof(T));
        for(id<MTLBuffer> x:{w,wb,yb})resident.push_back(x);
        for(detail::PageBlock* p:{&f.v,&f.t}){id<MTLBuffer> x=wrap(p->p,p->bytes);if(!x)throw std::runtime_error("Metal buffer wrap failed (qr)");resident.push_back(x);}
        Wk=static_cast<T*>(w.contents);Wb=static_cast<T*>(wb.contents);Yb=static_cast<T*>(yb.contents);}
    else{host.resize(m*n+2*nb*n);Wk=host.data();Wb=Wk+m*n;Yb=Wb+nb*n;}
    each(m,[&](std::size_t i){std::memcpy(static_cast<void*>(Wk+i*n),A+i*n,n*sizeof(T));});
    // Panel scratch: the panel's columns (Pt) and reflectors (Vt) contiguous per column (rows [k0, m)), panel W rows (Wp,
    // nb x nb: Wp[i][c] = P(v_i, a_c) with a_c as at the block start), Gram column G and y.
    std::vector<T> Pt(nb*m),Vt(nb*m),Wp(nb*nb),G(nb),y(nb);const T two=power2<N>(1),one=power2<N>(0);long top=LONG_MIN;
    // Row blocks of the panel phases (claimed dynamically: the cores differ in speed): norm parts per row block, per-thread
    // accumulators of the Gram/W dots (anchored at a common bound, so a thread can add any rows), exponent ranges of the
    // panel columns (block-start values) and reflectors, notes on x below the diagonal per row block, the exact column norm.
    struct Tail {bool below;long emin,emax;};struct Acc {word status;bool used;word* pos;word* neg;};struct Range {long lo,hi;};
    const std::size_t lanes=pool_workers().size(),max_blocks=m/8+1;
    std::vector<Part> parts(max_blocks);std::vector<word> arena(max_blocks*2*part_cap);std::vector<Tail> tails(max_blocks);std::vector<word> sum;
    for(std::size_t i=0;i<parts.size();++i){parts[i].pos=arena.data()+i*2*part_cap;parts[i].neg=parts[i].pos+part_cap;}
    std::vector<Acc> accs(lanes*nb);std::vector<word> acc_arena(accs.size()*2*part_cap);std::vector<Range> prange(nb),vrange(nb);
    std::vector<long> dlo(nb);std::vector<int> dwords(nb);
    for(std::size_t i=0;i<accs.size();++i){accs[i].pos=acc_arena.data()+i*2*part_cap;accs[i].neg=accs[i].pos+part_cap;}
    auto range_of=[](const T* x,std::size_t len){Range g{LONG_MAX,LONG_MIN};for(std::size_t r=0;r<len;++r)if(x[r].sign&&!x[r].status){g.lo=std::min(g.lo,long(x[r].exponent));g.hi=std::max(g.hi,long(x[r].exponent));}return g;};
    // Pivoting: the permutation, squared column norms nu (downdated after every step) and nuref (their values at the last
    // exact computation), exponent ranges of the trailing columns at the block start (rows [k0, m)).
    const bool pivot=fo.pivot;auto& perm=f.perm;perm.resize(n);for(std::size_t k=0;k<n;++k)perm[k]=k;
    std::vector<T> nu(pivot?n:0),nuref(pivot?n:0);std::vector<Range> crange(pivot?n:0);std::vector<std::uint8_t> flag(pivot?n:0);
    if(pivot)each(n,[&](std::size_t k){nu[k]=nuref[k]=exact_dot<N>(Wk+k,std::ptrdiff_t(n),Wk+k,std::ptrdiff_t(n),m);});
    constexpr std::size_t TW=16; // trailing columns per item of the pivoted W/Y rows
    // Pivoting: W row c of the trailing columns [a, e): Wb[c*n+k] = P(v_j, a_k) over local rows [c, mr) of the block-start
    // columns (Wk, read by rows), each column in an exact window anchored at the product of the exponent ranges (else exact_dot).
    auto w_tile=[&](std::size_t k0,std::size_t mr,std::size_t c,std::size_t a,std::size_t e,const T* vc,Range vr,int logl){
        word acc[TW][2][part_cap];long lo[TW];int words[TW];word status[TW],tw[64];
        for(std::size_t k=a;k<e;++k){const std::size_t i=k-a;status[i]=0;words[i]=0;lo[i]=0;if(vr.lo>vr.hi||crange[k].lo>crange[k].hi)continue;
            lo[i]=vr.lo+crange[k].lo;const long need=vr.hi+crange[k].hi-lo[i]+64*N+logl+1;
            if(need>32L*(part_cap-2)){words[i]=-1;continue;}words[i]=int(need/32)+2;std::fill(acc[i][0],acc[i][0]+words[i],0);std::fill(acc[i][1],acc[i][1]+words[i],0);}
        for(std::size_t r=c;r<mr;++r){const T& xa=vc[r];const T* row=Wk+(k0+r)*n;
            for(std::size_t k=a;k<e;++k){const std::size_t i=k-a;const T& xb=row[k];const word st=xa.status|xb.status;status[i]|=st;
                if(st||!xa.sign||!xb.sign||words[i]<=0)continue;
                mantissa_product<N>(xa,xb,tw);detail::add_shifted(acc[i][xa.sign==xb.sign?0:1],tw,2*N,long(xa.exponent)+xb.exponent-lo[i]);}}
        for(std::size_t k=a;k<e;++k){const std::size_t i=k-a;T& out=Wb[c*n+k];
            if(words[i]<0)out=exact_dot<N>(vc+c,1,Wk+(k0+c)*n+k,std::ptrdiff_t(n),mr-c);
            else out=words[i]?round_acc<N>(acc[i][0],acc[i][1],std::size_t(words[i]),lo[i],status[i]):zero<N>(status[i]);}};
    auto factor_panel=[&](std::size_t k0,std::size_t k1){auto t=Clock::now();++info.blocks;
        const std::size_t kb=k1-k0,mr=m-k0;T* Tb=Tm+(k0/nb)*nb*nb;
        if(!pivot)each(kb,[&](std::size_t c){T* x=Pt.data()+c*mr;for(std::size_t r=0;r<mr;++r)x[r]=Wk[(k0+r)*n+k0+c];std::fill(Vt.data()+c*mr,Vt.data()+(c+1)*mr,zero<N>());prange[c]=range_of(x,mr);});
        else{each(kb,[&](std::size_t c){std::fill(Vt.data()+c*mr,Vt.data()+(c+1)*mr,zero<N>());});
            each((n-k0+TW-1)/TW,[&](std::size_t t){const std::size_t a=k0+t*TW,e=std::min(n,a+TW);for(std::size_t k=a;k<e;++k)crange[k]={LONG_MAX,LONG_MIN};
                for(std::size_t r=k0;r<m;++r)for(std::size_t k=a;k<e;++k){const T& z=Wk[r*n+k];if(z.sign&&!z.status){crange[k].lo=std::min(crange[k].lo,long(z.exponent));crange[k].hi=std::max(crange[k].hi,long(z.exponent));}}});}
        std::size_t done=kb; // columns of this panel that end with a reflector or the remainder
        // Row blocks of rows [a, mr): BR rows each (about four per lane).
        const std::size_t BR=std::clamp<std::size_t>(mr/(4*lanes),8,256);
        auto blocks=[&](std::size_t a){return (mr-a+BR-1)/BR;};
        for(std::size_t c=0;c<kb;++c){const std::size_t j=k0+c;T* x=Pt.data()+c*mr;
            if(pivot){
                // Pivot: the largest nu over columns [j, n) (a status above every value), ties to the lowest original index;
                // columns j and p are swapped everywhere (A P), then column j is loaded.
                std::size_t p=j;for(std::size_t k=j+1;k<n;++k){const int cmp=compare_magnitude(nu[k],nu[p]);if(cmp>0||(cmp==0&&perm[k]<perm[p]))p=k;}
                if(p!=j){for(std::size_t r=0;r<m;++r)std::swap(Wk[r*n+j],Wk[r*n+p]);for(std::size_t i=0;i<c;++i){std::swap(Wb[i*n+j],Wb[i*n+p]);std::swap(Yb[i*n+j],Yb[i*n+p]);}
                    std::swap(nu[j],nu[p]);std::swap(nuref[j],nuref[p]);std::swap(crange[j],crange[p]);std::swap(perm[j],perm[p]);}
                for(std::size_t r=0;r<mr;++r)x[r]=Wk[(k0+r)*n+j];prange[c]=crange[j];}
            const std::size_t q=std::min(j,info.rank)-k0;const bool reflect=info.rank==n;
            // Column k0+c by the reflectors [k0, k0+q) of this block: y_i = P(T_b[0..i][i], w[0..i]); x = D(x; V_b, y). In the
            // same pass every row block sums its part of x[c..mr)^2 exactly (s below) and notes nonzeros below row c and their exponents.
            // Pivoting formed this column's y with the other trailing columns (Yb).
            if(q)for(std::size_t i=0;i<q;++i)y[i]=pivot?Yb[i*n+j]:exact_dot<N>(Tb+i,std::ptrdiff_t(nb),Wp.data()+c,std::ptrdiff_t(nb),i+1);
            const std::size_t nbk=blocks(0);
            each(nbk,[&](std::size_t k){const std::size_t r0=k*BR,r1=std::min(mr,r0+BR);
                if(q)for(std::size_t r=r0;r<r1;++r)x[r]=exact_dot_add<N>(&x[r],true,Vt.data()+r,std::ptrdiff_t(mr),y.data(),1,q);
                if(!reflect)return;
                const std::size_t a=std::max(r0,c);part_dot<N>(parts[k],x+a,1,x+a,1,r1>a?r1-a:0);
                Tail tl{false,LONG_MAX,LONG_MIN};for(std::size_t r=std::max(r0,c+1);r<r1;++r)if(x[r].sign){tl.below=true;tl.emin=std::min(tl.emin,long(x[r].exponent));tl.emax=std::max(tl.emax,long(x[r].exponent));}
                tails[k]=tl;});
            if(!reflect)continue;  // after a failing column: the remainder only
            // Reflector j = k0+c from x[c..mr): s = P(x, x); beta = -sgn(x_0) sqrt(s); v0 = RN(x_0 - beta); v = 2^-e (v0, x_1, ...)
            // with e = exponent(v0); tau = RN(2 / P(v, v)); r_jj = beta. Nothing below the diagonal: tau = 0, v = e_j, r_jj = x_0.
            // s and P(v, v) come from the exact block sums (P(v, v) = RN(2^-2e (v0^2 + S - x_0^2)), S the exact sum), else exact_dot.
            const std::size_t len=mr-c;T* xc=x+c;T s;long slo=0;int ssign=0;const bool exact=merge_parts<N>(parts.data(),nbk,s,&sum,&slo,&ssign);
            if(!exact)s=exact_dot<N>(xc,1,xc,1,len);
            bool below=false;long emin=LONG_MAX,emax=LONG_MIN;for(std::size_t k=0;k<nbk;++k){below|=tails[k].below;emin=std::min(emin,tails[k].emin);emax=std::max(emax,tails[k].emax);}
            int reason=s.status?QRInfo::status_column:(!xc[0].sign&&!below)?QRInfo::zero_column:QRInfo::full_rank;T beta=xc[0],v0=one;
            if(!reason&&below){beta=sqrt(s);if(xc[0].sign>=0)beta=negate(beta);v0=sub(xc[0],beta);}
            if(!reason&&fo.rank_bits>0&&top!=LONG_MIN&&long(beta.exponent)+fo.rank_bits<top)reason=QRInfo::small_column;
            if(reason){info.rank=j;info.reason=reason;info.status=s.status;if(pivot){done=c;break;}continue;}
            T* vc=Vt.data()+c*mr;T* v=vc+c;T tj=zero<N>();long e=0;bool scale_rows=false;vrange[c]={0,0};
            if(below){e=v0.exponent;v[0]=scaled(v0,e);
                if(exact&&emin-e>=-1000000000L&&emax-e<=1000000000L){tj=div(two,reflector_norm<N>(sum,slo,xc[0],v0,e));scale_rows=true;vrange[c]={std::min(0L,emin-e),std::max(0L,emax-e)};}
                else{for(std::size_t r=1;r<len;++r)v[r]=scaled(xc[r],e);tj=div(two,exact_dot<N>(v,1,v,1,len));vrange[c]=range_of(v,len);}}
            else v[0]=one;
            xc[0]=beta;tau[k0+c]=tj;top=std::max(top,long(beta.exponent));
            // Gram column G_i = P(v_i, v_j) (i < c) and panel W row c: Wp[c][c'] = P(v_j, a_c') for c' > c (not yet reduced by this
            // block), over rows [c, mr). Dot u's terms lie within the product of the two exponent ranges: every thread adds the rows
            // it claims into its accumulator anchored at that bound (when the window fits; otherwise exact_dot below), after
            // scaling those rows of v (when not done above).
            // Pivoting: only the Gram column here; W row c of every trailing column k > j, P(v_j, a_k) over rows [j, m), is formed
            // in the same dispatch by column tiles (items after the row blocks), each column in its own exact window.
            int logl=0;while((std::size_t(1)<<logl)<=len)++logl;const std::size_t ndots=pivot?c:kb-1,ntiles=pivot?(n-j-1+TW-1)/TW:0;
            for(std::size_t u=0;u<ndots;++u){const Range &a=u<c?vrange[u]:vrange[c],&b=u<c?vrange[c]:prange[u+1];dwords[u]=-1;
                if(a.lo>a.hi||b.lo>b.hi){dwords[u]=0;continue;}
                const long lo=a.lo+b.lo,need=a.hi+b.hi-lo+64*N+logl+1;if(need<=32L*(part_cap-2)){dlo[u]=lo;dwords[u]=int(need/32)+2;}}
            const std::size_t nbc=blocks(c);for(auto& g:accs)g.used=false;
            if(ntiles&&scale_rows)each(nbc,[&](std::size_t k){const std::size_t r0=c+k*BR,r1=std::min(mr,r0+BR);for(std::size_t r=std::max(r0,c+1);r<r1;++r)vc[r]=scaled(x[r],e);});
            if(ntiles)scale_rows=false;
            pool_workers().run(nbc+ntiles,[&](std::size_t k,unsigned id){
                if(k>=nbc){w_tile(k0,mr,c,j+1+(k-nbc)*TW,std::min(n,j+1+(k-nbc+1)*TW),vc,vrange[c],logl);return;}
                Acc* ac=accs.data()+std::size_t(id)*nb;{const std::size_t r0=c+k*BR,r1=std::min(mr,r0+BR);
                    if(scale_rows)for(std::size_t r=std::max(r0,c+1);r<r1;++r)vc[r]=scaled(x[r],e);
                    for(std::size_t u=0;u<ndots;++u){if(dwords[u]<0)continue;Acc& g=ac[u];
                        if(!g.used){g.used=true;g.status=0;std::fill(g.pos,g.pos+dwords[u],0);std::fill(g.neg,g.neg+dwords[u],0);}
                        const T* pa=u<c?Vt.data()+u*mr:vc;const T* pb=u<c?vc:Pt.data()+(u+1)*mr;word tw[64];
                        for(std::size_t r=r0;r<r1;++r){const T &xa=pa[r],&xb=pb[r];const word st=xa.status|xb.status;g.status|=st;
                            if(st||!xa.sign||!xb.sign)continue;
                            mantissa_product<N>(xa,xb,tw);detail::add_shifted(xa.sign==xb.sign?g.pos:g.neg,tw,2*N,long(xa.exponent)+xb.exponent-dlo[u]);}}}});
            for(std::size_t u=0;u<ndots;++u){T& out=u<c?G[u]:Wp[c*nb+u+1];
                if(dwords[u]<0){out=u<c?exact_dot<N>(Vt.data()+u*mr+c,1,v,1,len):exact_dot<N>(v,1,Pt.data()+(u+1)*mr+c,1,len);continue;}
                // Merge: one rounding of the exact total (statuses first, as exact_dot).
                const std::size_t W=std::size_t(dwords[u]);word status=0;word pos[part_cap+1],neg[part_cap+1];std::fill(pos,pos+W,0);std::fill(neg,neg+W,0);
                for(std::size_t id=0;id<lanes;++id){const Acc& g=accs[id*nb+u];if(!g.used)continue;status|=g.status;dword cp=0,cn=0;
                    for(std::size_t i=0;i<W;++i){cp+=dword(pos[i])+g.pos[i];pos[i]=word(cp);cp>>=32;cn+=dword(neg[i])+g.neg[i];neg[i]=word(cn);cn>>=32;}}
                out=round_acc<N>(pos,neg,W,dlo[u],status);}
            // T_b column c: T_cc = tau_j, T_lc = -RN(tau_j * P(T_b[l][l..c), G[l..c))).
            Tb[c*nb+c]=tj;for(std::size_t l=0;l<c;++l)Tb[l*nb+c]=negate(mul(tj,exact_dot<N>(Tb+l*nb+l,1,G.data()+l,1,c-l)));
            if(pivot&&j+1<n){
                // Every trailing column k > j: y_k(c) = P(T_b[0..c][c], w_k[0..c]) (row c of Y), r_jk = D(a_jk; V[j][k0..j], y_k)
                // (its entry in row j of R), nu_k = D(nu_k; r_jk, r_jk). A nu_k that drops to <= 0 or more than bits/2 binades
                // below nuref_k (nuref_k != 0, no status) is recomputed: nu_k = nuref_k = P(x_k, x_k) over rows (j, m) of the
                // reduced column x_rk = D(a_rk; V[r][k0..j], y_k).
                each((n-j-1+TW-1)/TW,[&](std::size_t t){for(std::size_t k=j+1+t*TW;k<std::min(n,j+1+(t+1)*TW);++k){
                    Yb[c*n+k]=exact_dot<N>(Tb+c,std::ptrdiff_t(nb),Wb+k,std::ptrdiff_t(n),c+1);
                    const T r=exact_dot_add<N>(&Wk[j*n+k],true,Vt.data()+c,std::ptrdiff_t(mr),Yb+k,std::ptrdiff_t(n),c+1);
                    nu[k]=exact_dot_add<N>(&nu[k],true,&r,1,&r,1,1);
                    flag[k]=nuref[k].sign&&!nuref[k].status&&!nu[k].status&&(nu[k].sign<=0||long(nu[k].exponent)+32*N/2<long(nuref[k].exponent));}});
                std::vector<std::size_t> redo;for(std::size_t k=j+1;k<n;++k)if(flag[k])redo.push_back(k);info.norm_recomputations+=redo.size();
                each(redo.size(),[&](std::size_t i){const std::size_t k=redo[i];std::vector<T> z(mr-c-1);
                    for(std::size_t r=c+1;r<mr;++r)z[r-c-1]=exact_dot_add<N>(&Wk[(k0+r)*n+k],true,Vt.data()+r,std::ptrdiff_t(mr),Yb+k,std::ptrdiff_t(n),c+1);
                    nu[k]=nuref[k]=exact_dot<N>(z.data(),1,z.data(),1,z.size());});}
        }
        each(done,[&](std::size_t c){const T *x=Pt.data()+c*mr,*v=Vt.data()+c*mr;for(std::size_t r=0;r<mr;++r){Wk[(k0+r)*n+k0+c]=x[r];V[(k0+r)*n+k0+c]=v[r];}});
        info.panel_seconds+=since(t);return std::min(k1,info.rank);};
    auto update=[&](std::size_t b,std::size_t c0,std::size_t c1,bool& on_gpu,bool have_y=false){auto t=Clock::now();
        Timing tm=qr_block<N>(f,b,Wk,n,c0,c1,true,Wb,Yb,fo.host_macs,gpu,on_gpu,have_y);tm.wall_seconds=since(t);return tm;};
    auto account=[&](Timing tm,bool on_gpu){info.timing.gpu_seconds+=tm.gpu_seconds;info.update_seconds+=tm.wall_seconds;++(on_gpu?info.gpu_updates:info.host_updates);};
    // Right-looking with one block of look-ahead (as cholesky_n): after panel b, block b updates the next panel's columns,
    // then the rest of the trailing matrix on a second thread while this thread factors the next panel. After a failing
    // column p in block b, block b (reflectors < p; the others are zero) is applied to the remaining columns and the loop stops.
    // Pivoting: no look-ahead (the next panel's columns are chosen during it); the panel formed W and Y of the trailing
    // columns, so the update is X = D(X; V_b, Y) only. After a failing column p, block b is applied to columns [p, n).
    if(pivot)for(std::size_t k0=0;k0<n;k0+=nb){const std::size_t k1=std::min(n,k0+nb),b=k0/nb,p=factor_panel(k0,k1);bool on_gpu=false;
        if(p<k1){if(p>k0)account(update(b,p,n,on_gpu),on_gpu);break;}
        if(k1<n)account(update(b,k1,n,on_gpu,true),on_gpu);}
    std::size_t k0=0,k1=std::min(n,nb),p=n&&!pivot?factor_panel(0,k1):0;
    while(!pivot&&k1<n){const std::size_t b=k0/nb;bool on_gpu=false;
        if(p<k1){if(p>k0)account(update(b,k1,n,on_gpu),on_gpu);break;}
        const std::size_t k2=std::min(n,k1+nb);account(update(b,k1,k2,on_gpu),on_gpu);
        if(k2<n){Timing rest{0,0};bool rest_gpu=false,async=gpu;std::thread side;std::exception_ptr failed;
            if(async)side=std::thread([&]{@autoreleasepool{side_thread=true;try{rest=update(b,k2,n,rest_gpu);}catch(...){failed=std::current_exception();}side_thread=false;}});
            try{p=factor_panel(k1,k2);}catch(...){if(async)side.join();throw;}
            if(async){side.join();if(failed)std::rethrow_exception(failed);}else rest=update(b,k2,n,rest_gpu);
            account(rest,rest_gpu);}
        else p=factor_panel(k1,k2);
        k0=k1;k1=k2;}
    // R: the upper triangle; with rank p < n, rows i >= p keep the remainder in columns j >= p.
    T* R=static_cast<T*>(f.r.p);const std::size_t rank=info.rank;
    each(n,[&](std::size_t i){for(std::size_t j=0;j<n;++j)R[i*n+j]=j>=std::min(i,rank)?Wk[i*n+j]:zero<N>();});
    info.timing.wall_seconds=since(start);
}}
// C (m x nrhs) <- Q^T C (qt: blocks in order) or Q C (blocks in reverse), over the blocks holding reflectors [0, rank).
template<int N> Timing Linalg::Impl::qr_apply_n(const detail::QRData& f,Number<N>* C,std::size_t nrhs,bool qt){@autoreleasepool{
    using T=Number<N>;Timing tm{0,0};auto start=Clock::now();const std::size_t m=f.m,nb=f.nb,blocks=nb?(f.info.rank+nb-1)/nb:0;
    if(!blocks||!nrhs){tm.wall_seconds=since(start);return tm;}
    Residency keep(*this);pool_workers();const bool gpu=f.options.gpu&&double(std::min(nb,f.n))*double(nrhs)*double(m)>=f.options.solve_host_macs;
    std::vector<T> host;T *W=C,*Wb,*Yb;
    if(gpu){id<MTLBuffer> x=wrap(C,m*nrhs*sizeof(T));if(!x){x=buffer("qr_rhs",m*nrhs*sizeof(T));W=static_cast<T*>(x.contents);std::memcpy(static_cast<void*>(W),C,m*nrhs*sizeof(T));}
        id<MTLBuffer> wb=buffer("qr_w",nb*nrhs*sizeof(T)),yb=buffer("qr_y",nb*nrhs*sizeof(T));for(id<MTLBuffer> b:{x,wb,yb})resident.push_back(b);
        for(const detail::PageBlock* p:{&f.v,&f.t}){id<MTLBuffer> b=wrap(p->p,p->bytes);if(!b)throw std::runtime_error("Metal buffer wrap failed (qr)");resident.push_back(b);}
        Wb=static_cast<T*>(wb.contents);Yb=static_cast<T*>(yb.contents);}
    else{host.resize(2*nb*nrhs);Wb=host.data();Yb=Wb+nb*nrhs;}
    for(std::size_t q=0;q<blocks;++q){bool on_gpu;tm.gpu_seconds+=qr_block<N>(f,qt?q:blocks-1-q,W,nrhs,0,nrhs,qt,Wb,Yb,f.options.solve_host_macs,gpu,on_gpu).gpu_seconds;}
    if(W!=C)std::memcpy(static_cast<void*>(C),W,m*nrhs*sizeof(T));
    tm.wall_seconds=since(start);return tm;
}}
// X = R^-1 (Q^T B)[0, n): Q^T B by blocks, then the backward substitution of cholesky_solve with R (upper).
template<int N> Timing Linalg::Impl::qr_solve_n(const detail::QRData& f,const Number<N>* B,std::size_t nrhs,Number<N>* X){
    using T=Number<N>;auto start=Clock::now();Timing tm{0,0};const std::size_t m=f.m,n=f.n;if(!n||!nrhs)return tm;
    const std::size_t p=f.info.rank;const bool pivoted=f.options.pivot;
    if(p<n&&(!pivoted||f.info.reason==QRInfo::status_column)){const word st=invalid|(pivoted?f.info.status:0);pool_workers();
        each(n,[&](std::size_t i){for(std::size_t c=0;c<nrhs;++c)X[i*nrhs+c]=zero<N>(st);});tm.wall_seconds=since(start);return tm;}
    std::vector<T> C(B,B+m*nrhs);tm.gpu_seconds+=qr_apply_n<N>(f,C.data(),nrhs,true).gpu_seconds;
    if(pivoted){
        // z = R[0:p,0:p]^-1 (Q^T B)[0:p) (the same backward substitution), x[perm[i]] = z_i, and 0 for i >= p (basic solution).
        const T* R=static_cast<const T*>(f.r.p);std::vector<T> R11(p<n?p*p:0),Z(p*nrhs);for(std::size_t i=0;i<p&&p<n;++i)std::copy(R+i*n,R+i*n+p,R11.begin()+i*p);
        if(p)tm.gpu_seconds+=solve_n<N>(p<n?R11.data():R,p,C.data(),nrhs,Z.data(),f.options,2,true).gpu_seconds;
        for(std::size_t i=0;i<n;++i)for(std::size_t c=0;c<nrhs;++c)X[f.perm[i]*nrhs+c]=i<p?Z[i*nrhs+c]:zero<N>();
        tm.wall_seconds=since(start);return tm;}
    tm.gpu_seconds+=solve_n<N>(static_cast<const T*>(f.r.p),n,C.data(),nrhs,X,f.options,2,true).gpu_seconds;
    tm.wall_seconds=since(start);return tm;
}
static void check_size(std::size_t n,std::size_t nrhs){if(n>=46341||n*nrhs>=(std::size_t(1)<<31))throw std::invalid_argument("matrix dimensions exceed 32-bit indexing");}
CholeskyInfo Linalg::cholesky(int bits,const void* A,std::size_t n,void* L,const FactorOptions& o){
    check_bits(bits);check_size(n,1);if(n&&(!A||!L))throw std::invalid_argument("null matrix");
    return by_words(bits/32,[&](auto w){constexpr int N=decltype(w)::value;return impl->cholesky_n<N>(static_cast<const Number<N>*>(A),n,static_cast<Number<N>*>(L),o);});
}
Timing Linalg::trsm(int bits,bool transpose,const void* L,std::size_t n,const void* B,std::size_t nrhs,void* X,const FactorOptions& o){
    check_bits(bits);check_size(n,nrhs);if(n&&nrhs&&(!L||!B||!X))throw std::invalid_argument("null matrix");
    return by_words(bits/32,[&](auto w){constexpr int N=decltype(w)::value;
        return impl->solve_n<N>(static_cast<const Number<N>*>(L),n,static_cast<const Number<N>*>(B),nrhs,static_cast<Number<N>*>(X),o,transpose?2:1);});
}
Timing Linalg::cholesky_solve(int bits,const void* L,std::size_t n,const void* B,std::size_t nrhs,void* X,const FactorOptions& o){
    check_bits(bits);check_size(n,nrhs);if(n&&nrhs&&(!L||!B||!X))throw std::invalid_argument("null matrix");
    return by_words(bits/32,[&](auto w){constexpr int N=decltype(w)::value;
        return impl->solve_n<N>(static_cast<const Number<N>*>(L),n,static_cast<const Number<N>*>(B),nrhs,static_cast<Number<N>*>(X),o,3);});
}
QRFactor Linalg::factor_qr(int bits,const void* A,std::size_t m,std::size_t n,const QROptions& o){
    check_bits(bits);if(m<n)throw std::invalid_argument("factor_qr: rows >= columns required");check_size(n,m);if(m*n&&!A)throw std::invalid_argument("null matrix");
    if(o.rank_bits<0)throw std::invalid_argument("factor_qr: rank_bits >= 0");
    QRFactor q;q.d=std::make_unique<detail::QRData>();auto& f=*q.d;f.owner=this;f.bits=bits;f.m=m;f.n=n;f.nb=o.block?std::min(o.block,n):n;f.options=o;
    const std::size_t e=std::size_t(bits/8+12),blocks=f.nb?(n+f.nb-1)/f.nb:0;
    f.v=detail::PageBlock(m*n*e);f.r=detail::PageBlock(n*n*e);f.t=detail::PageBlock(blocks*f.nb*f.nb*e);f.tau=detail::PageBlock(n*e);
    by_words(bits/32,[&](auto w){constexpr int N=decltype(w)::value;impl->qr_n<N>(static_cast<const Number<N>*>(A),m,n,f);});
    return q;
}
QRFactor Linalg::factor_qr_augmented(int bits,const void* J,std::size_t m,std::size_t n,const void* d,const QROptions& o){
    check_bits(bits);check_size(n,m+n);if(n&&(!d||(m&&!J)))throw std::invalid_argument("null matrix");
    const std::size_t e=std::size_t(bits/8+12);std::vector<unsigned char> S((m+n)*n*e,0);if(m*n)std::memcpy(S.data(),J,m*n*e);
    for(std::size_t i=0;i<n;++i)std::memcpy(S.data()+((m+i)*n+i)*e,static_cast<const unsigned char*>(d)+i*e,e);
    return factor_qr(bits,S.data(),m+n,n,o);
}
QRFactor::QRFactor()=default;QRFactor::~QRFactor()=default;QRFactor::QRFactor(QRFactor&&)noexcept=default;QRFactor& QRFactor::operator=(QRFactor&&)noexcept=default;
static const detail::QRData& qr_data(const std::unique_ptr<detail::QRData>& d){if(!d)throw std::logic_error("empty QR factor");return *d;}
bool QRFactor::empty()const{return !d;}
int QRFactor::bits()const{return qr_data(d).bits;}
std::size_t QRFactor::rows()const{return qr_data(d).m;}
std::size_t QRFactor::cols()const{return qr_data(d).n;}
std::size_t QRFactor::block()const{return qr_data(d).nb;}
const QRInfo& QRFactor::info()const{return qr_data(d).info;}
bool QRFactor::full_rank()const{return qr_data(d).info.rank==qr_data(d).n;}
const void* QRFactor::r()const{return qr_data(d).r.p;}
const void* QRFactor::v()const{return qr_data(d).v.p;}
const void* QRFactor::tau()const{return qr_data(d).tau.p;}
const void* QRFactor::t()const{return qr_data(d).t.p;}
bool QRFactor::pivoted()const{return qr_data(d).options.pivot;}
const std::size_t* QRFactor::permutation()const{return qr_data(d).perm.data();}
Timing QRFactor::solve(const void* B,std::size_t nrhs,void* X)const{const auto& f=qr_data(d);check_size(f.n,nrhs);check_size(1,f.m*nrhs);if(f.n&&nrhs&&(!B||!X))throw std::invalid_argument("null matrix");
    return by_words(f.bits/32,[&](auto w){constexpr int N=decltype(w)::value;return f.owner->impl->qr_solve_n<N>(f,static_cast<const Number<N>*>(B),nrhs,static_cast<Number<N>*>(X));});}
Timing QRFactor::apply_q(void* B,std::size_t nrhs,bool transpose)const{const auto& f=qr_data(d);check_size(f.n,nrhs);check_size(1,f.m*nrhs);if(f.m&&nrhs&&!B)throw std::invalid_argument("null matrix");
    return by_words(f.bits/32,[&](auto w){constexpr int N=decltype(w)::value;return f.owner->impl->qr_apply_n<N>(f,static_cast<Number<N>*>(B),nrhs,transpose);});}
}
