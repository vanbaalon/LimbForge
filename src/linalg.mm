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
#include <cstring>
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
struct Params {std::uint32_t rows,cols,K,Kp,Mp,Np,j,L,Wc,tri,lower,fold,ls_line,ls_k,rs_line,rs_k,out_cols,multi_rows,multi_cols,both,j0,group,upper,row0,col0,pad;};
struct Line {std::int32_t bands,low,slot;};
struct Member {std::int32_t line,lo,hi;};
constexpr std::size_t max_k=65472; // concatenated digit product: 2*Kp*2^14 < 2^31
constexpr int group=4;            // moduli per digit pass (each input entry is read once per group)
std::size_t up(std::size_t x,std::size_t m){return (x+m-1)/m*m;}
int padded(int words){return words>=13&&words<33?33:words;} // docs/gpu-codegen.md rule 1
// Primes <= 65279, descending: balanced residues then split into two int8 digits.
const std::vector<std::uint32_t>& primes(){static const std::vector<std::uint32_t> p=[]{std::vector<std::uint32_t> r;
    for(std::uint32_t q=65279;r.size()<1200;--q){bool prime=true;for(std::uint32_t d=2;d*d<=q;++d)if(q%d==0){prime=false;break;}if(prime)r.push_back(q);}return r;}();return p;}
std::uint32_t inverse(std::uint64_t a,std::uint64_t m){std::int64_t t=0,nt=1,r=std::int64_t(m),nr=std::int64_t(a%m);
    while(nr){std::int64_t q=r/nr,x=t-q*nt;t=nt;nt=x;x=r-q*nr;r=nr;nr=x;}return std::uint32_t(t<0?t+std::int64_t(m):t);}
template<class F> void parallel(unsigned threads,std::size_t n,F f){
    unsigned t=unsigned(std::min<std::size_t>(threads?threads:1,n));if(t<=1){if(n)f(0,n);return;}
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
using DotFn=void(*)(const void*,std::ptrdiff_t,const void*,std::ptrdiff_t,std::size_t,void*);
template<int N> void dot_n(const void* a,std::ptrdiff_t sa,const void* b,std::ptrdiff_t sb,std::size_t K,void* out){
    *static_cast<Number<N>*>(out)=exact_dot<N>(static_cast<const Number<N>*>(a),sa,static_cast<const Number<N>*>(b),sb,K);}
template<int N> void dot_table(DotFn* t){t[N]=dot_n<N>;if constexpr(N<32)dot_table<N+1>(t);}
DotFn dot_function(int words){static DotFn table[33]={};if(!table[2])dot_table<2>(table);return table[words];}
}
struct Kernels {
    id<MTLComputePipelineState> left,right,product,combine,reconstruct,finish;int max_moduli,term_words,acc_words;
    id<MTLBuffer> moduli,recips,inv;std::map<int,std::vector<std::uint32_t>> bounds; // per modulus count: [M/2 | M], Wc words each
};
struct Linalg::Impl {
    LinalgOptions options;LinalgReport report;id<MTLDevice> device;id<MTLCommandQueue> queue;
    std::map<int,Kernels> kernels;std::map<std::string,id<MTLBuffer>> pool;
    unsigned threads()const{return options.host_threads?options.host_threads:std::max(1u,std::thread::hardware_concurrency());}
    // Page-aligned caller memory is wrapped without a copy (the call waits for the GPU before returning).
    id<MTLBuffer> wrap(const void* p,std::size_t bytes){std::size_t page=std::size_t(getpagesize());if(!p||!bytes||reinterpret_cast<std::uintptr_t>(p)%page)return nil;
        return [device newBufferWithBytesNoCopy:const_cast<void*>(p) length:up(bytes,page) options:MTLResourceStorageModeShared deallocator:nil];}
    id<MTLBuffer> buffer(const std::string& role,std::size_t bytes){
        auto& b=pool[role];if(!b||b.length<bytes){b=nil;b=[device newBufferWithLength:std::max<std::size_t>(bytes,16) options:MTLResourceStorageModeShared];}
        if(!b)throw std::runtime_error("Metal allocation failed ("+role+")");return b;}
    id<MTLComputePipelineState> state(id<MTLLibrary> lib,NSString* name){NSError* e=nil;id<MTLFunction> f=[lib newFunctionWithName:name];
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
        k.left=state(lib,@"digits_left");k.right=state(lib,@"digits_right");k.product=state(lib,@"product");k.combine=state(lib,@"combine");
        k.reconstruct=state(lib,@"reconstruct");k.finish=state(lib,@"finish");
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
    Timing run(int bits,View lv,View rv,std::size_t M,std::size_t Nc,std::size_t K,void* C,bool syrk,bool lower_only);
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
Timing Linalg::syrk(int bits,const void* A,std::size_t rows,std::size_t cols,void* C,bool lower_only){
    if(bits<64||bits>1024||bits%32)throw std::invalid_argument("bits must be a multiple of 32 in [64,1024]");
    std::size_t e=bits/8+12;auto a=static_cast<const unsigned char*>(A);
    return impl->run(bits,{a,e,1,cols,bits/32},{a,e,1,cols,bits/32},cols,cols,rows,C,true,lower_only);
}
Timing Linalg::gemm(int bits,bool transpose_a,const void* A,const void* B,std::size_t m,std::size_t n,std::size_t k,void* C){
    if(bits<64||bits>1024||bits%32)throw std::invalid_argument("bits must be a multiple of 32 in [64,1024]");
    std::size_t e=bits/8+12;
    View lv=transpose_a?View{static_cast<const unsigned char*>(A),e,1,m,bits/32}:View{static_cast<const unsigned char*>(A),e,k,1,bits/32};
    return impl->run(bits,lv,{static_cast<const unsigned char*>(B),e,1,n,bits/32},m,n,k,C,false,false);
}
Timing Linalg::Impl::run(int bits,View lv,View rv,std::size_t M,std::size_t Nc,std::size_t K,void* C,bool syrk,bool lower_only){@autoreleasepool{
    auto start=Clock::now();report=LinalgReport{};report.m=M;report.n=Nc;report.k=K;
    const std::size_t limit=std::size_t(1)<<31;if(M>=limit||Nc>=limit||K>=limit)throw std::invalid_argument("matrix dimensions exceed 32-bit indexing");
    if(M&&Nc&&!C)throw std::invalid_argument("null output");if(M*Nc&&K&&(!lv.base||!rv.base))throw std::invalid_argument("null input");
    const int N=bits/32;const std::size_t eb=lv.bytes;const unsigned nt=threads();
    auto t=Clock::now();bool force=options.force_fallback||K>max_k;
    Side ls=analyze(lv,M,K,options,force,nt),rs_own;if(!syrk)rs_own=analyze(rv,Nc,K,options,force,nt);
    const Side& rs=syrk?ls:rs_own;
    report.left_bands=ls.histogram;report.right_bands=rs.histogram;report.left_fallback=ls.fallback;report.right_fallback=rs.fallback;
    report.analysis_seconds=since(t);
    Timing timing{0,0};id<MTLBuffer> out=nil;
    if(!ls.band.empty()&&!rs.band.empty()){
        Kernels& kern=get(bits);t=Clock::now();
        // Inputs (one copy for SYRK), band members (segments aligned to 64 members), line tables, output, accumulators.
        id<MTLBuffer> in_a=wrap(lv.base,M*K*eb),in_b=in_a;
        if(!in_a){in_a=buffer("a",M*K*eb);std::memcpy(in_a.contents,lv.base,M*K*eb);}
        if(!syrk&&!(in_b=wrap(rv.base,K*Nc*eb))){in_b=buffer("b",K*Nc*eb);std::memcpy(in_b.contents,rv.base,K*Nc*eb);}
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
        if(!(out=wrap(C,M*Nc*eb)))out=buffer("out",M*Nc*eb);else report.zero_copy_output=true;
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
        // Digit planes per group member: left cat = [d1 | d0] (Mp x 2Kp), d1 and d0 (Mp x Kp); right [d0 ; d1] (2Kp x Np).
        const std::size_t hp=Mp*Kp,rp=2*Kp*Np;
        id<MTLBuffer> lcat=buffer("lcat",group*2*hp),a1=buffer("a1",group*hp),a0=buffer("a0",group*hp),rcat=buffer("rcat",group*rp);
        id<MTLBuffer> g11=buffer("g11",Mp*Np*4),gmid=buffer("gmid",Mp*Np*4),g00=buffer("g00",Mp*Np*4),res=buffer("res",std::size_t(L)*Mp*Np*2);
        report.upload_seconds=since(t);
        id<MTLCommandBuffer> cb=[queue commandBuffer];
        if(slots){id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];[blit fillBuffer:acc range:NSMakeRange(0,slots*kern.acc_words*4) value:0];[blit endEncoding];}
        id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
        [enc setBuffer:kern.moduli offset:0 atIndex:11];[enc setBuffer:kern.recips offset:0 atIndex:12];
        auto grid=[&](id<MTLComputePipelineState> p,std::size_t w,std::size_t h){if(!w||!h)return;[enc setComputePipelineState:p];
            [enc dispatchThreads:MTLSizeMake(w,h,1) threadsPerThreadgroup:MTLSizeMake(32,std::min<NSUInteger>(8,p.maxTotalThreadsPerThreadgroup/32),1)];};
        auto product=[&](id<MTLBuffer> a,std::size_t aoff,id<MTLBuffer> b,std::size_t boff,id<MTLBuffer> c,std::size_t k){
            std::uint32_t mnk[4]={std::uint32_t(Mp),std::uint32_t(Np),std::uint32_t(k),syrk};[enc setComputePipelineState:kern.product];
            [enc setBuffer:a offset:aoff atIndex:0];[enc setBuffer:b offset:boff atIndex:1];[enc setBuffer:c offset:0 atIndex:2];[enc setBytes:mnk length:16 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(Np/32,Mp/64,1) threadsPerThreadgroup:MTLSizeMake(kern.product.threadExecutionWidth*4,1,1)];};
        Params p{};p.rows=std::uint32_t(Me);p.cols=std::uint32_t(Ne);p.K=std::uint32_t(K);p.Kp=std::uint32_t(Kp);p.Mp=std::uint32_t(Mp);p.Np=std::uint32_t(Np);
        p.L=L;p.Wc=Wc;p.tri=syrk;p.lower=syrk;p.both=syrk;p.ls_line=std::uint32_t(lv.line_stride);p.ls_k=std::uint32_t(lv.k_stride);
        p.rs_line=std::uint32_t(rv.line_stride);p.rs_k=std::uint32_t(rv.k_stride);p.out_cols=std::uint32_t(Nc);p.multi_rows=std::uint32_t(ls.multi);p.multi_cols=std::uint32_t(rs.multi);
        [enc setBuffer:mem offset:0 atIndex:1];[enc setBuffer:lcat offset:0 atIndex:2];[enc setBuffer:a1 offset:0 atIndex:3];[enc setBuffer:a0 offset:0 atIndex:4];
        for(unsigned j0=0;j0<L;j0+=group){
            // Digits of the extended operands for moduli j0 .. j0+group-1 (each input entry is read once per group).
            p.j0=j0;p.group=std::min<unsigned>(group,L-j0);[enc setBytes:&p length:sizeof(p) atIndex:5];
            [enc setBuffer:in_a offset:0 atIndex:0];[enc setBuffer:mem offset:0 atIndex:1];[enc setBuffer:lcat offset:0 atIndex:2];[enc setBuffer:a1 offset:0 atIndex:3];
            [enc setBuffer:a0 offset:0 atIndex:4];[enc setBuffer:rcat offset:0 atIndex:6];grid(kern.left,Kp,Mp);
            if(!syrk){[enc setBuffer:in_b offset:0 atIndex:0];[enc setBuffer:mem offset:rbase*sizeof(Member) atIndex:1];[enc setBuffer:rcat offset:0 atIndex:2];grid(kern.right,Np,Kp);}
            for(unsigned j=j0;j<std::min(L,j0+group);++j){std::size_t q=j-j0;p.j=j;
                product(a1,q*hp,rcat,q*rp+Kp*Np,g11,Kp);product(a0,q*hp,rcat,q*rp,g00,Kp);product(lcat,q*2*hp,rcat,q*rp,gmid,2*Kp);
                [enc setBytes:&p length:sizeof(p) atIndex:5];[enc setBuffer:g11 offset:0 atIndex:0];[enc setBuffer:gmid offset:0 atIndex:1];[enc setBuffer:g00 offset:0 atIndex:2];
                [enc setBuffer:res offset:0 atIndex:3];grid(kern.combine,Ne,Me);}
        }
        [enc setBuffer:res offset:0 atIndex:0];[enc setBuffer:kern.inv offset:0 atIndex:2];[enc setBuffer:bnd offset:0 atIndex:3];[enc setBuffer:mem offset:0 atIndex:4];
        [enc setBuffer:mem offset:rbase*sizeof(Member) atIndex:6];[enc setBuffer:lline offset:0 atIndex:7];[enc setBuffer:rline offset:0 atIndex:8];
        [enc setBuffer:out offset:0 atIndex:9];[enc setBuffer:acc offset:0 atIndex:10];
        for(auto& pr:plan){Params q=p;q.rows=std::uint32_t(ls.band[pr.b].size());q.cols=std::uint32_t(rs.band[pr.c].size());q.row0=std::uint32_t(loff[pr.b]);
            q.col0=std::uint32_t(roff[pr.c]);q.tri=syrk&&pr.b==pr.c;q.fold=syrk&&pr.b>pr.c;q.upper=0;[enc setBytes:&q length:sizeof(q) atIndex:5];grid(kern.reconstruct,q.cols,q.rows);
            if(q.fold){q.upper=1;[enc setBytes:&q length:sizeof(q) atIndex:5];grid(kern.reconstruct,q.cols,q.rows);}}
        if(slots){Params q=p;q.rows=std::uint32_t(M);q.cols=std::uint32_t(Nc);[enc setBytes:&q length:sizeof(q) atIndex:5];grid(kern.finish,Nc,M);}
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.status==MTLCommandBufferStatusError)throw std::runtime_error("Metal execution (linalg): "+message(cb.error));
        timing.gpu_seconds=report.gpu_seconds=cb.GPUEndTime-cb.GPUStartTime;
    }
    // Assembly: GPU outputs, trivial zeros with statuses, and exact host dots for fallback lines.
    t=Clock::now();std::atomic<std::size_t> next{0},gpu_count{0},multi_count{0},fallback_count{0},trivial_count{0};std::atomic<double> fallback_time{0};
    const unsigned char* o=out?static_cast<const unsigned char*>(out.contents):nullptr;auto* c=static_cast<unsigned char*>(C);DotFn dot=dot_function(N);
    auto worker=[&](std::size_t,std::size_t){std::size_t g=0,mu=0,fb=0,tr=0;double ft=0;
        for(std::size_t i;(i=next.fetch_add(1))<M;)for(std::size_t j=0;j<(syrk?i+1:Nc);++j){
            unsigned char* dst=c+(i*Nc+j)*eb;std::uint8_t ci=ls.cls[i],cj=rs.cls[j];
            if(ci==1||cj==1){std::memset(dst,0,eb);word st=ls.status[i]|rs.status[j];std::memcpy(dst+4*N+8,&st,4);++tr;}
            else if(ci==2||cj==2){auto f=Clock::now();dot(lv.at(i,0),std::ptrdiff_t(lv.k_stride),rv.at(j,0),std::ptrdiff_t(rv.k_stride),K,dst);ft+=since(f);++fb;}
            else{if(o!=c)std::memcpy(dst,o+(i*Nc+j)*eb,eb);++g;mu+=ls.line[i].bands>1||rs.line[j].bands>1;}
        }
        gpu_count+=g;multi_count+=mu;fallback_count+=fb;trivial_count+=tr;double old=fallback_time.load();while(!fallback_time.compare_exchange_weak(old,old+ft)){}};
    parallel(nt,nt,worker);
    if(syrk&&!lower_only)parallel(nt,M,[&](std::size_t first,std::size_t last){for(std::size_t i=first;i<last;++i)for(std::size_t j=i+1;j<Nc;++j)std::memcpy(c+(i*Nc+j)*eb,c+(j*Nc+i)*eb,eb);});
    report.gpu_outputs=gpu_count;report.multi_band_outputs=multi_count;report.fallback_outputs=fallback_count;report.trivial_outputs=trivial_count;
    report.fallback_seconds=fallback_time;report.assembly_seconds=since(t);timing.wall_seconds=since(start);return timing;
}}
}
