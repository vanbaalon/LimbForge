// Resident dense linear algebra (docs/numerics.md, "Resident products"): Linalg::syrk/gemm encoded into an Engine's CommandBatch
// on Buffer operands, bitwise against the host-array calls (themselves MPFR-validated by test_linalg / test_cholesky / test_qr) at
// 64/224/256/384 bits, including QSC-like 800 x 400, multi-band, host-fallback and workspace-overflow paths, statuses, report
// fields, chains with engine and Numerics operations in one batch, ownership errors and lifetimes; and the synchronous Buffer
// forms of cholesky, trsm, cholesky_solve, factor_qr, QRFactor::solve and apply_q; Linalg::workspaces / release_workspaces with idle,
// pending (submitted, unwaited) and unsubmitted batches. Usage: test_limbforge_resident_linalg [--quick]
#include "reference.hpp"
#include "limbforge/linalg.hpp"
#include "limbforge/numerics.hpp"
#include <cstring>
#include <functional>
#include <stdexcept>
using namespace limbforge;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
template<class E,class Fn> void rejects(Fn f,const std::string& what){bool ok=false;try{f();}catch(const E&){ok=true;}require(ok,"not rejected: "+what);}
template<class T> bool same(const T& a,const T& b){return std::memcmp(&a,&b,sizeof(T))==0;}
template<class T> void same_all(const std::vector<T>& a,const std::vector<T>& b,const std::string& what){
    require(a.size()==b.size(),what+": size");std::size_t bad=0,first=0;for(std::size_t i=0;i<a.size();++i)if(!same(a[i],b[i])&&!bad++)first=i;
    require(!bad,what+": "+std::to_string(bad)+" mismatches, first at "+std::to_string(first));}
template<class T> Buffer<T> put(Engine& e,const std::vector<T>& v,std::size_t extra=0){auto b=e.make_buffer<T>(v.size()+extra);if(!v.empty())b.upload(v.data(),v.size());return b;}
template<class T> std::vector<T> get(const Buffer<T>& b,std::size_t n){std::vector<T> v(n);if(n)b.download(v.data(),n);return v;}
template<int B> using F=Float<B>;
template<int B> F<B> power(long e,int sign=1){F<B> x=zero<B/32>();x.limb[B/32-1]=0x80000000u;x.sign=sign;x.exponent=int(e);return x;}
// Fixtures as test_linalg (structure per column = per line of A^T A): moderate, wide (multi-band, 5-band and spread fallbacks,
// exponent extremes), qsc (column scales 2^+-60), sparse (zero lines and status entries).
enum Kind {moderate,wide,qsc,sparse};
const char* kind_name(Kind k){static const char* n[]={"moderate","wide","qsc","sparse"};return n[k];}
template<int B> std::vector<F<B>> matrix(std::size_t rows,std::size_t cols,Kind kind,std::mt19937_64& rng){
    constexpr int N=B/32;std::vector<F<B>> A(rows*cols,zero<N>());auto at=[&](std::size_t r,std::size_t c)->F<B>&{return A[r*cols+c];};
    for(std::size_t c=0;c<cols;++c){
        long scale=kind==wide?long(rng()%2001)-1000:kind==qsc?long(rng()%121)-60:0;
        for(std::size_t r=0;r<rows;++r){F<B> x=reference::random_number<B>(rng,kind==qsc?15:kind==wide?30:4);x.exponent+=int(scale+(kind==qsc?-10:0));
            if(rng()%(kind==sparse?3:kind==qsc?4:10)==0)x=zero<N>();at(r,c)=x;}
        if(kind==wide){std::size_t r=rng()%rows;
            switch(c%9){case 1:case 2:at(r,c).exponent=int(scale-100-long(rng()%350));break;               // second band
                case 3:at(r,c).exponent=int(scale-5000);break;                                               // spread -> fallback
                case 4:for(std::size_t q=0;q<rows;q+=2)at(q,c).exponent=int(scale-70*long(q/2%5));break;     // 5 bands -> fallback
                case 7:for(std::size_t q=0;q<rows;q+=2)at(q,c).exponent=int(scale-100*long(q/2%3));break;    // 3 bands
                case 8:for(std::size_t q=1;q<rows;q+=2)at(q,c).exponent=int(scale-90*long(q/2%4));break;     // 4 bands
                case 5:at(r,c).exponent=600000000;break;case 6:for(std::size_t q=0;q<rows;++q)at(q,c).exponent=-600000000+int(q%5);break;
                default:break;}}
        if(kind==qsc&&c%17==5)at(rng()%rows,c).exponent=int(scale-150);
        if(kind==sparse){if(c%11==3)for(std::size_t r=0;r<rows;++r)at(r,c)=zero<N>();
            if(c%13==7)at(rng()%rows,c).status=rng()&1?invalid:exponent_overflow;}
    }
    return A;
}
template<int B> std::vector<F<B>> old_values(std::size_t n,std::mt19937_64& rng){std::vector<F<B>> C(n);
    for(auto& x:C){x=reference::random_number<B>(rng,40);if(rng()%29==0)x=zero<B/32>(rng()&1?invalid:division_by_zero);else if(rng()%23==0)x=zero<B/32>();}return C;}
// The resident report equals the host call's analysis and placement (no timings).
void same_report(const LinalgReport& h,const LinalgReport& r,const std::string& what){
    require(h.m==r.m&&h.n==r.n&&h.k==r.k,what+": report shape");
    require(h.left_bands==r.left_bands&&h.right_bands==r.right_bands,what+": band histograms");
    require(h.left_fallback==r.left_fallback&&h.right_fallback==r.right_fallback,what+": fallback lines "+std::to_string(h.left_fallback)+"/"+std::to_string(r.left_fallback));
    require(h.gpu_outputs==r.gpu_outputs&&h.multi_band_outputs==r.multi_band_outputs&&h.fallback_outputs==r.fallback_outputs&&h.trivial_outputs==r.trivial_outputs,
            what+": output counts gpu "+std::to_string(h.gpu_outputs)+"/"+std::to_string(r.gpu_outputs)+" multi "+std::to_string(h.multi_band_outputs)+"/"+
            std::to_string(r.multi_band_outputs)+" fallback "+std::to_string(h.fallback_outputs)+"/"+std::to_string(r.fallback_outputs)+" trivial "+
            std::to_string(h.trivial_outputs)+"/"+std::to_string(r.trivial_outputs));
    require(h.gemm_rows==r.gemm_rows&&h.gemm_cols==r.gemm_cols&&h.int8_gemms==r.int8_gemms,what+": extended sizes / moduli "+std::to_string(h.int8_gemms)+"/"+std::to_string(r.int8_gemms));
    require(h.pairs.size()==r.pairs.size(),what+": band pairs");
    for(std::size_t i=0;i<h.pairs.size();++i){auto &a=h.pairs[i],&b=r.pairs[i];
        require(a.left_band==b.left_band&&a.right_band==b.right_band&&a.rows==b.rows&&a.cols==b.cols&&a.moduli==b.moduli,what+": band pair "+std::to_string(i));}
    require(!r.resident_overflow,what+": unexpected workspace overflow");
}
struct Stats {std::size_t calls=0,multi=0,fallback=0,overflow=0;};
// One resident SYRK (and its update) against the host call with the same Linalg.
template<int B> void check_syrk(Engine& e,Linalg& la,const std::vector<F<B>>& A,std::size_t rows,std::size_t cols,bool lower_only,bool update,
                                const std::string& what,std::mt19937_64& rng,Stats& s,bool overflow=false){
    const std::size_t n=cols*cols;std::vector<F<B>> C0=update?old_values<B>(n,rng):std::vector<F<B>>(n,power<B>(9)),host=C0;
    la.syrk(B,A.data(),rows,cols,host.data(),lower_only,update);const LinalgReport hr=la.report();
    auto Ab=put(e,A,3),Cb=put(e,C0,5);auto batch=e.batch();auto t=la.syrk(batch,Ab,rows,cols,Cb,lower_only,update);
    require(!t.resolved(),what+": resolved before submission");auto sub=batch.submit();rejects<std::logic_error>([&]{t.report();},what+": report before wait");sub.wait();
    require(t.resolved()&&!t.provisional_reads(),what+": ticket after wait");
    same_all(get(Cb,n),host,what);if(!overflow||!t.report().resident_overflow)same_report(hr,t.report(),what);
    ++s.calls;s.multi+=hr.multi_band_outputs>0;s.fallback+=hr.fallback_outputs>0;s.overflow+=t.report().resident_overflow;
}
template<int B> void check_gemm(Engine& e,Linalg& la,bool ta,const std::vector<F<B>>& A,const std::vector<F<B>>& Bm,std::size_t m,std::size_t n,std::size_t k,
                                bool update,const std::string& what,std::mt19937_64& rng,Stats& s,bool overflow=false){
    std::vector<F<B>> C0=update?old_values<B>(m*n,rng):std::vector<F<B>>(m*n,power<B>(9)),host=C0;
    la.gemm(B,ta,A.data(),Bm.data(),m,n,k,host.data(),update);const LinalgReport hr=la.report();
    auto Ab=put(e,A),Bb=put(e,Bm,1),Cb=put(e,C0);auto batch=e.batch();auto t=la.gemm(batch,ta,Ab,Bb,m,n,k,Cb,update);batch.submit().wait();
    same_all(get(Cb,m*n),host,what);if(!overflow||!t.report().resident_overflow)same_report(hr,t.report(),what);
    ++s.calls;s.multi+=hr.multi_band_outputs>0;s.fallback+=hr.fallback_outputs>0;s.overflow+=t.report().resident_overflow;
}
template<int B> std::vector<F<B>> transpose(const std::vector<F<B>>& A,std::size_t rows,std::size_t cols){
    std::vector<F<B>> T(A.size());for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<cols;++c)T[c*rows+r]=A[r*cols+c];return T;}
struct Config {const char* name;LinalgOptions options;bool overflow;};
std::vector<Config> configs(){LinalgOptions narrow;narrow.band_bits=16;narrow.max_bands=8;narrow.resident_bands=8;LinalgOptions zero_width;zero_width.band_bits=0;zero_width.max_bands=3;zero_width.resident_bands=3;
    LinalgOptions fallback;fallback.force_fallback=true;LinalgOptions tight;tight.resident_bands=1;
    return {{"default",{},false},{"narrow bands",narrow,false},{"zero-width bands",zero_width,false},{"forced fallback",fallback,false},{"resident_bands=1",tight,true}};}
template<int B> void products(Engine& e,bool quick){
    std::mt19937_64 rng(7000+B);Stats s;const auto cfg=configs();
    for(std::size_t ci=0;ci<cfg.size();++ci){Linalg la(e,cfg[ci].options);const bool main=ci==0,ov=cfg[ci].overflow;
        for(Kind kind:{moderate,wide,qsc,sparse}){
            std::vector<std::size_t> sizes=main&&!quick?std::vector<std::size_t>{1,7,33,128}:std::vector<std::size_t>{7,33};
            for(std::size_t sz:sizes){const std::string what=std::string(cfg[ci].name)+" "+kind_name(kind)+" bits="+std::to_string(B)+" size="+std::to_string(sz);
                const std::size_t rows=kind==qsc?2*sz+1:sz+2;auto A=matrix<B>(rows,sz,kind,rng);
                check_syrk<B>(e,la,A,rows,sz,(sz+ci)%2==0,false,"syrk "+what,rng,s,ov);
                check_syrk<B>(e,la,A,rows,sz,true,true,"syrk update "+what,rng,s,ov);
                const std::size_t m=sz,n=sz+3,k=rows;auto L=matrix<B>(k,m,kind,rng),R=matrix<B>(k,n,kind,rng);
                check_gemm<B>(e,la,true,L,R,m,n,k,false,"gemm^T "+what,rng,s,ov);
                check_gemm<B>(e,la,false,transpose<B>(L,k,m),R,m,n,k,sz%2==1,"gemm "+what,rng,s,ov);
                check_gemm<B>(e,la,true,L,R,m,n,k,true,"gemm^T update "+what,rng,s,ov);
            }
        }
    }
    // QSC-like normal equations A^T A of 800 x 400 (and the wide fixture of that shape: multi-band and fallback lines).
    {Linalg la(e);auto A=matrix<B>(800,400,qsc,rng);check_syrk<B>(e,la,A,800,400,true,false,"QSC-like 800x400 bits="+std::to_string(B),rng,s);
     if(!quick){auto W=matrix<B>(800,400,wide,rng);check_syrk<B>(e,la,W,800,400,false,false,"wide 800x400 bits="+std::to_string(B),rng,s);
        auto G=matrix<B>(400,300,wide,rng),H=matrix<B>(400,200,qsc,rng);check_gemm<B>(e,la,true,G,H,300,200,400,true,"wide gemm^T update 300x200x400",rng,s);}
     // More than 1024 lines per side: several lines per thread in the plan's scans.
     if(B==256){auto G=matrix<B>(20,1300,wide,rng),H=matrix<B>(20,1030,sparse,rng);check_gemm<B>(e,la,true,G,H,1300,1030,20,true,"gemm^T update 1300x1030x20",rng,s);
        auto W=matrix<B>(24,1100,wide,rng);check_syrk<B>(e,la,W,24,1100,false,false,"wide syrk 24x1100",rng,s);}}
    // Degenerate shapes: K = 0 (zeros; an update leaves C unchanged) and empty outputs.
    {Linalg la(e);std::vector<F<B>> none;auto C0=std::vector<F<B>>(9,power<B>(3));auto Ab=e.make_buffer<F<B>>(0);auto Cb=put(e,C0),Cu=put(e,C0);
     auto batch=e.batch();la.syrk(batch,Ab,0,3,Cb,false);la.syrk(batch,Ab,0,3,Cu,true,true);auto E=e.make_buffer<F<B>>(0);auto t=la.gemm(batch,false,Ab,Ab,0,0,5,E);batch.submit().wait();
     for(auto& x:get(Cb,9))require(same(x,zero<B/32>()),"syrk K=0");same_all(get(Cu,9),C0,"syrk update K=0");require(t.resolved()&&t.report().gpu_outputs==0,"empty gemm");}
    require(s.multi&&s.fallback&&s.overflow,"fixtures must reach multi-band, fallback and overflow paths");
    std::cout<<B<<" bits: "<<s.calls<<" resident syrk/gemm calls equal the host-array calls ("<<s.multi<<" with multi-band outputs, "<<s.fallback
             <<" with host-fallback outputs, "<<s.overflow<<" workspace overflows)"<<std::endl;
}
// Every width (docs/gpu-codegen.md rule 3): the wide and QSC fixtures through syrk, the syrk update and a gemm update.
template<int B> void all_widths(Engine& e,std::size_t& calls){
    std::mt19937_64 rng(B);Stats s;Linalg la(e);
    for(Kind kind:{wide,qsc}){auto A=matrix<B>(41,33,kind,rng),R=matrix<B>(41,30,kind,rng);const std::string w=std::string(kind_name(kind))+" bits="+std::to_string(B);
        check_syrk<B>(e,la,A,41,33,false,false,"syrk "+w,rng,s);check_syrk<B>(e,la,A,41,33,true,true,"syrk update "+w,rng,s);
        check_gemm<B>(e,la,true,A,R,33,30,41,true,"gemm^T update "+w,rng,s);}
    calls+=s.calls;if constexpr(B<1024)all_widths<B+32>(e,calls);
}
// ---------------- chains, workspace sharing, ownership, lifetime ----------------
void chains(Engine& e){
    constexpr int B=256;using T=F<B>;std::mt19937_64 rng(99);Linalg la(e);Numerics nm(e);
    // mul (engine) -> syrk -> norm2 (Numerics) in one batch; norm2 reads the provisional C (flagged).
    {const std::size_t rows=300,cols=120;auto A=matrix<B>(rows,cols,qsc,rng),S=matrix<B>(rows,cols,moderate,rng);for(auto& x:S)if(!x.sign)x=power<B>(0);
     std::vector<T> A2(A.size()),C(cols*cols,zero<B/32>()),norm(cols);e.run(B,Operation::mul,A.data(),S.data(),A2.data(),A.size());
     la.syrk(B,A2.data(),rows,cols,C.data(),false);nm.norm2(B,Segments{cols,cols,false},C.data(),norm.data());
     auto Ab=put(e,A),Sb=put(e,S),A2b=e.make_buffer<T>(A.size()),Cb=e.make_buffer<T>(C.size()),Nb=e.make_buffer<T>(cols);
     auto batch=e.batch();batch.run(Operation::mul,Ab,Sb,A2b);auto t=la.syrk(batch,A2b,rows,cols,Cb,false);nm.norm2(batch,Segments{cols,cols,false},Cb,Nb);
     rejects<std::logic_error>([&]{batch.run(Operation::add,Cb,Cb,Cb);},"write to a provisional product output in the same batch");
     batch.submit().wait();same_all(get(A2b,A.size()),A2,"chain mul");same_all(get(Cb,C.size()),C,"chain mul->syrk");same_all(get(Nb,cols),norm,"chain syrk->norm2");
     require(t.provisional_reads()&&t.report().fallback_outputs==0,"chain: provisional read flagged, no fallback outputs");}
    // gemm -> syrk of its output (no fallback lines: the provisional read sees final values) and a gemm update with the wide fixture
    // (host fallback at wait), sharing the batch workspace; a second batch pending at the same time (its own workspace).
    {const std::size_t k=90,m=70,n=60;auto Q=matrix<B>(k,m,qsc,rng),L=matrix<B>(k,m,wide,rng),R=matrix<B>(k,n,moderate,rng);
     std::vector<T> C1(m*n),C2(n*n,zero<B/32>()),C3=old_values<B>(m*n,rng),C4=C3,D1(m*n),D2(n*n,zero<B/32>());
     la.gemm(B,true,Q.data(),R.data(),m,n,k,C1.data());require(la.report().fallback_outputs==0,"qsc gemm without fallback");la.syrk(B,C1.data(),m,n,C2.data(),true);
     la.gemm(B,true,L.data(),R.data(),m,n,k,C3.data(),true);require(la.report().fallback_outputs>0,"wide gemm must use the host fallback");
     la.gemm(B,true,L.data(),R.data(),m,n,k,D1.data());la.syrk(B,D1.data(),m,n,D2.data(),true);
     auto Qb=put(e,Q),Lb=put(e,L),Rb=put(e,R),L2=put(e,L),R2=put(e,R),C1b=e.make_buffer<T>(m*n),C2b=put(e,std::vector<T>(n*n,zero<B/32>())),C3b=put(e,C4),C4b=put(e,C4);
     auto D1b=e.make_buffer<T>(m*n),D2b=put(e,std::vector<T>(n*n,zero<B/32>()));
     auto batch=e.batch();auto t1=la.gemm(batch,true,Qb,Rb,m,n,k,C1b);auto t2=la.syrk(batch,C1b,m,n,C2b,true);auto t3=la.gemm(batch,true,Lb,Rb,m,n,k,C3b,true);
     auto t6=la.gemm(batch,true,Lb,Rb,m,n,k,D1b);la.syrk(batch,D1b,m,n,D2b,true); // reads fallback outputs before wait: flagged, recomputed below
     auto other=e.batch();auto t4=la.gemm(other,true,Lb,Rb,m,n,k,C4b,true); // encoded before the first submission
     auto first=batch.submit();rejects<std::logic_error>([&]{other.submit();},"second batch reading busy buffers");
     {auto busy=e.batch();rejects<std::logic_error>([&]{la.gemm(busy,true,Lb,Rb,m,n,k,C4b,true);},"encoding with busy buffers");}
     auto again=e.batch();auto t5=la.gemm(again,true,L2,R2,m,n,k,C4b,true);auto second=again.submit(); // pending together with the first
     second.wait();first.wait();
     same_all(get(C1b,m*n),C1,"chain gemm");same_all(get(C2b,n*n),C2,"chain gemm->syrk");same_all(get(C3b,m*n),C3,"chain gemm update with fallback");
     same_all(get(C4b,m*n),C3,"gemm update in a later batch");same_all(get(D1b,m*n),D1,"gemm with fallback");
     require(t1.provisional_reads()&&!t2.provisional_reads()&&!t3.provisional_reads()&&t3.report().fallback_outputs>0&&t5.resolved()&&!t4.resolved(),"chain tickets");
     require(t6.provisional_reads()&&t6.report().fallback_outputs>0,"provisional read of fallback outputs is reported");
     {auto redo=e.batch();la.syrk(redo,D1b,m,n,D2b,true);redo.submit().wait();same_all(get(D2b,n*n),D2,"recomputed syrk after a flagged read");}}
    std::cout<<"chains: engine mul -> syrk -> norm2, gemm -> syrk -> gemm update (host fallback at wait) in one batch, concurrent batches"<<std::endl;
}
void ownership(Engine& e){
    constexpr int B=224;using T=F<B>;std::mt19937_64 rng(5);Linalg la(e);Engine other;
    auto A=put(e,matrix<B>(20,10,moderate,rng)),C=e.make_buffer<T>(100),small=e.make_buffer<T>(99),foreign=other.make_buffer<T>(200);
    {auto b=e.batch();rejects<std::invalid_argument>([&]{la.syrk(b,foreign,20,10,C);},"foreign A");}
    {auto b=e.batch();rejects<std::invalid_argument>([&]{la.syrk(b,A,20,10,foreign);},"foreign C");}
    {auto b=e.batch();rejects<std::invalid_argument>([&]{la.syrk(b,A,20,10,small);},"C too small");}
    {auto b=e.batch();rejects<std::invalid_argument>([&]{la.syrk(b,A,21,10,C);},"A too small");}
    {auto b=e.batch();rejects<std::invalid_argument>([&]{la.syrk(b,A,10,10,A);},"C aliases A");}
    {auto b=e.batch();rejects<std::invalid_argument>([&]{la.syrk(b,A,20,10,C,false,true);},"full-matrix update");}
    {auto b=e.batch();rejects<std::invalid_argument>([&]{la.gemm(b,false,A,A,10,20,20,A);},"gemm C aliases A");}
    {auto b=e.batch();auto t=la.syrk(b,A,20,10,C);rejects<std::logic_error>([&]{la.syrk(b,A,20,10,C);},"second write of a provisional output");}
    {auto b=e.batch();b.run(Operation::square,A,A);auto s=b.submit();auto b2=e.batch();rejects<std::logic_error>([&]{la.syrk(b2,A,20,10,C);},"busy A");
     auto L=e.make_buffer<T>(100);rejects<std::logic_error>([&]{la.cholesky(A,10,L);},"busy buffer in a synchronous factorization");s.wait();}
    LinalgTicket empty;require(!empty.resolved(),"empty ticket");rejects<std::logic_error>([&]{empty.report();},"empty ticket report");
    std::cout<<"ownership: foreign, small, aliased, busy and provisional buffers are rejected"<<std::endl;
}
void lifetime(Engine& e){
    constexpr int B=256;using T=F<B>;std::mt19937_64 rng(6);const std::size_t rows=60,cols=40;auto A=matrix<B>(rows,cols,wide,rng);
    std::vector<T> want(cols*cols,zero<B/32>());{Linalg la(e);la.syrk(B,A.data(),rows,cols,want.data(),false);}
    auto Ab=put(e,A);
    // The batch, submission and ticket outlive the Linalg; the completion step (host fallback) runs at wait.
    {auto Cb=e.make_buffer<T>(cols*cols);LinalgTicket t;CommandBatch batch=e.batch();{Linalg la(e);t=la.syrk(batch,Ab,rows,cols,Cb,false);}
     auto s=batch.submit();s.wait();require(t.resolved()&&t.report().fallback_outputs>0,"ticket after the Linalg is gone");same_all(get(Cb,want.size()),want,"syrk after Linalg destruction");}
    // A discarded batch leaves its ticket unresolved and frees the workspace; a destroyed submission resolves its ticket.
    {Linalg la(e);auto Cb=e.make_buffer<T>(cols*cols);LinalgTicket t;{auto batch=e.batch();t=la.syrk(batch,Ab,rows,cols,Cb,false);}require(!t.resolved(),"discarded batch");
     LinalgTicket u;{auto batch=e.batch();u=la.syrk(batch,Ab,rows,cols,Cb,false);auto s=batch.submit();}require(u.resolved(),"submission destructor resolves");
     same_all(get(Cb,want.size()),want,"syrk after a discarded batch");}
    std::cout<<"lifetime: tickets and batches outlive the Linalg; discarded batches and destroyed submissions"<<std::endl;
}
// ---------------- workspace release ----------------
// release_workspaces frees the host scratch and idle resident workspaces at once and detaches busy ones: a pending submission (whose
// host fallback at wait reads its call workspace) and an unsubmitted batch keep theirs until the completion step has run or the batch
// is destroyed, and their results are unchanged. Later calls (also into a batch that held a released workspace) allocate afresh.
void release(Engine& e){
    constexpr int B=256;using T=F<B>;constexpr int N=B/32;std::mt19937_64 rng(77);const std::size_t rows=120,cols=80,nrhs=3;
    auto A=matrix<B>(rows,cols,wide,rng),Q=matrix<B>(rows,cols,qsc,rng);std::vector<T> want(cols*cols,zero<N>()),wantq(cols*cols,zero<N>());
    {Linalg ref(e);ref.syrk(B,A.data(),rows,cols,want.data(),false);require(ref.report().fallback_outputs>0,"release fixture: host fallback");ref.syrk(B,Q.data(),rows,cols,wantq.data(),false);}
    Linalg la(e);{const auto w=la.workspaces();require(!w.scratch_bytes&&!w.idle_bytes&&!w.busy_bytes&&!w.busy_workspaces,"a fresh Linalg retains nothing");}
    // Host scratch (QR work matrix, W/Y, staging of the products, right-hand sides): released, and a factor made before still solves.
    QROptions qo;qo.host_macs=0;qo.solve_host_macs=0;QRFactor f=la.factor_qr(B,Q.data(),rows,cols,qo);std::vector<T> b(rows*nrhs),x0(cols*nrhs),x1(cols*nrhs);
    for(auto& v:b)v=reference::random_number<B>(rng,3);f.solve(b.data(),nrhs,x0.data());
    const auto w0=la.workspaces();require(w0.scratch_bytes>0&&!w0.idle_bytes&&!w0.busy_bytes,"scratch after host calls");
    {const auto r=la.release_workspaces();require(r.scratch_bytes==w0.scratch_bytes&&!r.idle_bytes&&!r.busy_bytes,"released scratch");require(!la.workspaces().scratch_bytes,"no scratch after release");}
    f.solve(b.data(),nrhs,x1.data());same_all(x1,x0,"QR solve after release");
    {QRFactor g=la.factor_qr(B,Q.data(),rows,cols,qo);require(std::memcmp(g.r(),f.r(),cols*cols*sizeof(T))==0,"QR factor after release");}
    la.release_workspaces();
    // Resident: idle after wait.
    auto Ab=put(e,A),Qb=put(e,Q);auto C1=e.make_buffer<T>(cols*cols),C2=e.make_buffer<T>(cols*cols),C3=e.make_buffer<T>(cols*cols),C4=e.make_buffer<T>(cols*cols);
    {auto batch=e.batch();la.syrk(batch,Qb,rows,cols,C1,false);batch.submit().wait();same_all(get(C1,want.size()),wantq,"syrk before release");}
    const auto w1=la.workspaces();require(w1.idle_bytes>0&&!w1.busy_bytes&&!w1.busy_workspaces,"idle resident workspaces after wait");
    // A pending submission with host-fallback outputs (computed at wait from the snapshots in its call workspace).
    auto batch=e.batch();auto t=la.syrk(batch,Ab,rows,cols,C2,false);
    const auto w2=la.workspaces();require(w2.busy_workspaces==2&&w2.busy_bytes>0,"call and batch workspaces busy after encoding");
    {const auto r=la.release_workspaces();require(r.busy_workspaces==2&&r.busy_bytes==w2.busy_bytes&&r.idle_bytes==w2.idle_bytes,"release detaches busy workspaces");
     const auto w=la.workspaces();require(w.busy_workspaces==2&&w.busy_bytes==w2.busy_bytes&&!w.idle_bytes&&!w.scratch_bytes,"detached workspaces stay alive while held");}
    // A later call into the same (unsubmitted) batch gets new workspaces; then the batch is submitted and released again before wait.
    auto t2=la.syrk(batch,Qb,rows,cols,C3,false);require(la.workspaces().busy_workspaces==4,"new workspaces for a later call of the batch");
    auto sub=batch.submit();{const auto r=la.release_workspaces();require(r.busy_workspaces==2,"second release with the submission pending");}
    // A second, unsubmitted batch released and destroyed: its workspaces are freed with it and C4 is untouched.
    {auto other=e.batch();std::vector<T> marker(cols*cols,power<B>(5));C4.upload(marker.data(),marker.size());auto Q2=put(e,Q);la.syrk(other,Q2,rows,cols,C4,false);
     require(la.workspaces().busy_workspaces==6,"pending and unsubmitted workspaces");la.release_workspaces();
     require(la.workspaces().busy_workspaces==6,"released workspaces of an unsubmitted batch stay until it is destroyed");}
    require(la.workspaces().busy_workspaces==4,"a destroyed batch frees its released workspaces");
    for(auto& x:get(C4,cols*cols))require(same(x,power<B>(5)),"a destroyed batch leaves its output untouched");
    sub.wait();require(t.resolved()&&t2.resolved()&&t.report().fallback_outputs>0,"tickets after wait");
    same_all(get(C2,want.size()),want,"syrk with host fallback, workspaces released before wait");same_all(get(C3,want.size()),wantq,"later call after a release");
    {const auto w=la.workspaces();require(!w.busy_bytes&&!w.busy_workspaces&&!w.idle_bytes,"released workspaces freed after wait");}
    // Calls after the release allocate afresh and give the same bits.
    {auto again=e.batch();la.syrk(again,Ab,rows,cols,C4,false);again.submit().wait();same_all(get(C4,want.size()),want,"syrk after release");
     require(la.workspaces().idle_bytes>0,"new idle workspaces");}
    std::cout<<"workspace release: host scratch, idle, pending and unsubmitted resident workspaces (results unchanged)"<<std::endl;
}
// ---------------- synchronous Buffer forms of the factorizations ----------------
template<int B> void factor_forms(Engine& e){
    using T=F<B>;constexpr int N=B/32;std::mt19937_64 rng(31+B);Linalg la(e);const std::size_t m=160,n=96,nrhs=3;
    auto J=matrix<B>(m,n,qsc,rng);std::vector<T> A(n*n,zero<N>());la.syrk(B,J.data(),m,n,A.data(),false);
    for(std::size_t i=0;i<n;++i)A[i*n+i]=reference::real<B>(Operation::add,A[i*n+i],power<B>(-20));
    std::vector<T> rhs(n*nrhs);for(auto& x:rhs)x=reference::random_number<B>(rng,3);FactorOptions fo;fo.host_macs=1e3;fo.solve_host_macs=1e3; // GPU updates
    std::vector<T> L(n*n),X(n*nrhs),Y(n*nrhs);auto info=la.cholesky(B,A.data(),n,L.data(),fo);require(info.pivot==n,"cholesky fixture");
    la.cholesky_solve(B,L.data(),n,rhs.data(),nrhs,X.data(),fo);la.trsm(B,true,L.data(),n,rhs.data(),nrhs,Y.data(),fo);
    auto Ab=put(e,A),Lb=e.make_buffer<T>(n*n),Bb=put(e,rhs),Xb=e.make_buffer<T>(n*nrhs+1),Yb=e.make_buffer<T>(n*nrhs);
    auto ri=la.cholesky(Ab,n,Lb,fo);require(ri.pivot==n&&ri.gpu_updates==info.gpu_updates&&ri.gpu_updates>0,"buffer cholesky info");same_all(get(Lb,n*n),L,"buffer cholesky");
    la.cholesky_solve(Lb,n,Bb,nrhs,Xb,fo);same_all(get(Xb,n*nrhs),X,"buffer cholesky_solve");la.trsm(true,Lb,n,Bb,nrhs,Yb,fo);same_all(get(Yb,n*nrhs),Y,"buffer trsm^T");
    {auto In=put(e,A);la.cholesky(In,n,In,fo);same_all(get(In,n*n),L,"buffer cholesky in place");auto Z=put(e,rhs);la.cholesky_solve(Lb,n,Z,nrhs,Z,fo);same_all(get(Z,n*nrhs),X,"buffer solve in place");}
    rejects<std::invalid_argument>([&]{la.cholesky_solve(Lb,n,Bb,nrhs,Lb,fo);},"solve X aliases L");
    // QR: the factor of a buffer equals the host factor; solve and apply_q on buffers equal the host calls.
    QROptions qo;qo.host_macs=1e3;qo.solve_host_macs=1e3;for(bool pivot:{false,true}){qo.pivot=pivot;
        auto Jb=put(e,J);QRFactor h=la.factor_qr(B,J.data(),m,n,qo),r=la.factor_qr(Jb,m,n,qo);
        require(r.info().rank==h.info().rank&&r.info().gpu_updates==h.info().gpu_updates,"buffer QR info");
        for(auto f:{&QRFactor::r,&QRFactor::v,&QRFactor::t,&QRFactor::tau}){std::size_t bytes=f==&QRFactor::r?n*n:f==&QRFactor::v?m*n:f==&QRFactor::tau?n:((n+h.block()-1)/h.block())*h.block()*h.block();
            require(std::memcmp((h.*f)(),(r.*f)(),bytes*sizeof(T))==0,"buffer QR factor arrays");}
        std::vector<T> b(m*nrhs),x(n*nrhs),q(m*nrhs);for(auto& v:b)v=reference::random_number<B>(rng,3);h.solve(b.data(),nrhs,x.data());q=b;h.apply_q(q.data(),nrhs,true);
        auto bb=put(e,b),xb=e.make_buffer<T>(n*nrhs),qb=put(e,b);r.solve(bb,nrhs,xb);r.apply_q(qb,nrhs,true);
        same_all(get(xb,n*nrhs),x,"buffer QR solve");same_all(get(qb,m*nrhs),q,"buffer QR apply_q");rejects<std::invalid_argument>([&]{r.solve(bb,nrhs,bb);},"QR solve X aliases B");}
    std::cout<<B<<" bits: buffer cholesky / cholesky_solve / trsm / factor_qr / QR solve / apply_q equal the host-array calls"<<std::endl;
}
int main(int argc,char** argv){try{
    bool quick=argc==2&&std::string(argv[1])=="--quick";if(argc>1&&!quick)throw std::invalid_argument("usage: test_limbforge_resident_linalg [--quick]");
    Engine e;std::cout<<e.device_name()<<std::endl;
    products<224>(e,quick);products<256>(e,quick);products<64>(e,quick);products<384>(e,quick);
    if(!quick){std::size_t calls=0;all_widths<64>(e,calls);std::cout<<"all 31 widths: "<<calls<<" resident syrk/gemm calls (wide and QSC fixtures) equal the host-array calls"<<std::endl;}
    chains(e);ownership(e);lifetime(e);release(e);
    factor_forms<256>(e);factor_forms<224>(e);
    return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
