// Dense products with one rounding per output: syrk / gemm on the GPU (residue GEMM over exponent bands, exact
// host fallback) and the CPU exact_dot, against exact MPFR products summed by one mpfr_sum (reference::dot).
#include "reference.hpp"
#include "limbforge/linalg.hpp"
#include <atomic>
#include <mutex>
#include <thread>
#include <unistd.h>
using namespace limbforge;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
template<int Bits> using F=Float<Bits>;
template<int Bits> F<Bits> power(long e,int sign=1){F<Bits> x=zero<Bits/32>();x.limb[Bits/32-1]=0x80000000u;x.sign=sign;x.exponent=int(e);return x;}
template<int Bits> F<Bits> with_exponent(F<Bits> x,long e){x.exponent=int(e);return x;}
template<int Bits> std::string text(const F<Bits>& x){std::string s="sign="+std::to_string(x.sign)+" exp="+std::to_string(x.exponent)+" status="+std::to_string(x.status)+" limbs=";
    for(int i=Bits/32-1;i>=0;--i){char b[9];std::snprintf(b,9,"%08x",x.limb[i]);s+=b;}return s;}
template<class Job> void parallel_rows(std::size_t n,Job job){std::atomic<std::size_t> next{0};std::vector<std::thread> w;
    for(unsigned t=0;t<std::max(1u,std::thread::hardware_concurrency());++t)w.emplace_back([&]{for(std::size_t i;(i=next.fetch_add(1))<n;)job(i);});for(auto& x:w)x.join();}
struct Failures {std::mutex m;std::size_t count=0;std::string first;
    void add(const std::string& s){std::lock_guard<std::mutex> l(m);if(!count++)first=s;}
    void check(const std::string& label){require(!count,label+": "+std::to_string(count)+" mismatches; first "+first);}};
// ---- CPU exact_dot ----
template<int Bits> void dot_cases(){
    constexpr int N=Bits/32;std::mt19937_64 rng(31+Bits);
    auto check=[&](const std::vector<F<Bits>>& a,const std::vector<F<Bits>>& b,const std::string& label){
        auto want=reference::dot<Bits>(a.data(),1,b.data(),1,a.size());auto got=exact_dot<N>(a.data(),1,b.data(),1,a.size());
        require(reference::equal<Bits>(got,want),"exact_dot bits="+std::to_string(Bits)+" "+label+"\n got  "+text<Bits>(got)+"\n want "+text<Bits>(want));
        // Updates RN(c - dot) and RN(c + dot): c = RN(dot) leaves only the rounding error; c far above/below; statuses.
        auto sub=[&](F<Bits> c,const std::string& l){auto w=reference::dot_sub<Bits>(c,a.data(),1,b.data(),1,a.size());auto g=exact_dot_add<N>(&c,true,a.data(),1,b.data(),1,a.size());
            require(reference::equal<Bits>(g,w),"exact_dot_add bits="+std::to_string(Bits)+" "+label+" "+l+"\n got  "+text<Bits>(g)+"\n want "+text<Bits>(w));
            auto plus=exact_dot_add<N>(&c,false,a.data(),1,b.data(),1,a.size());c=negate(c);w=negate(reference::dot_sub<Bits>(c,a.data(),1,b.data(),1,a.size()));
            require(reference::equal<Bits>(plus,w),"exact_dot_add (add) bits="+std::to_string(Bits)+" "+label+" "+l);};
        sub(zero<N>(),"zero addend");sub(want,"c = RN(dot)");if(want.sign){F<Bits> next=want;next.limb[0]^=1;sub(next,"c = RN(dot) +- ulp");}
        if(want.sign&&!want.status){sub(with_exponent<Bits>(want,want.exponent+Bits+3),"c far above");sub(with_exponent<Bits>(negate(want),want.exponent-Bits-3),"c far below");
            sub(with_exponent<Bits>(want,want.exponent-2*Bits-70),"c very far below");}
        sub(reference::random_number<Bits>(rng,40),"random c");F<Bits> bad=power<Bits>(3);bad.status=exponent_overflow;sub(bad,"c with status");};
    check({},{},"empty");
    for(int it=0;it<60;++it){std::size_t K=1+rng()%40;std::vector<F<Bits>> a(K),b(K);
        int span=it%4==0?5:it%4==1?300:it%4==2?3000:900000000;
        for(std::size_t k=0;k<K;++k){a[k]=reference::random_number<Bits>(rng,span);b[k]=reference::random_number<Bits>(rng,span/2+1);if(rng()%9==0)a[k]=zero<N>();}
        if(it%5==1){a[K-1]=negate(a[0]);b[K-1]=b[0];}                                   // exact cancellation of the first product
        if(it%5==2&&K>2){a[1]=negate(a[0]);b[1]=b[0];a[2]=with_exponent<Bits>(a[2],a[0].exponent-4*Bits);} // tiny term exposed
        if(it%7==3)a[rng()%K].status=invalid;
        check(a,b,"random it="+std::to_string(it));}
    // Ties: 1 + 2^-bits (even: down), (1+ulp) + 2^-bits (odd: up), negated; through cancellation of large and far terms.
    F<Bits> one=power<Bits>(0),odd=one;odd.limb[0]|=1;
    for(int s:{1,-1})for(int p:{1,Bits/2,Bits-1})for(int extra:{0,1,-1}){
        std::vector<F<Bits>> a={one,power<Bits>(-p,s),power<Bits>(700),power<Bits>(700,-1),odd},b={power<Bits>(0,s),power<Bits>(-(Bits-p)-extra),power<Bits>(-3),power<Bits>(-3),zero<N>()};
        check(a,b,"tie p="+std::to_string(p)+" extra="+std::to_string(extra));
        a[0]=odd;check(a,b,"odd tie p="+std::to_string(p));
        a[2]=power<Bits>(9000);a[3]=power<Bits>(9000,-1);check(a,b,"far tie p="+std::to_string(p));
        b[4]=power<Bits>(-(2*Bits+500),-s);check(a,b,"tie broken far below p="+std::to_string(p));}
    // (1+u)^2 - (1+2u) = u^2 = 2^(2-2*Bits): cancellation down to the last product bit, then a term d bits below it.
    F<Bits> two=one;two.limb[0]|=2;
    for(int s:{1,-1})for(int d:{1,20,Bits-2,Bits-1,Bits,Bits+1,Bits+6,Bits+40,3*Bits}){
        std::vector<F<Bits>> a={odd,one,power<Bits>(2-2*Bits-d)},b={odd,negate(two),power<Bits>(0,s)};check(a,b,"last-bit cancellation d="+std::to_string(d));
        a.push_back(power<Bits>(2-2*Bits-d-Bits-9));b.push_back(power<Bits>(0,-s));check(a,b,"last-bit cancellation, two far terms d="+std::to_string(d));}
    // Ties created by the addend: c + 2^-Bits with c = 1 (even: stays) and c = 1 + ulp (odd: up), and the same far below.
    {auto tie=[&](F<Bits> c,const std::vector<F<Bits>>& a,const std::vector<F<Bits>>& b,const std::string& l){
        auto w=reference::dot_sub<Bits>(c,a.data(),1,b.data(),1,a.size()),g=exact_dot_add<N>(&c,true,a.data(),1,b.data(),1,a.size());
        require(reference::equal<Bits>(g,w),"exact_dot_add tie bits="+std::to_string(Bits)+" "+l+"\n got  "+text<Bits>(g)+"\n want "+text<Bits>(w));};
     for(int s:{1,-1})for(F<Bits> c:{one,odd}){if(s<0)c=negate(c);
        tie(c,{power<Bits>(-Bits)},{power<Bits>(0,-s)},"half ulp");tie(c,{power<Bits>(-Bits),power<Bits>(-3*Bits)},{power<Bits>(0,-s),power<Bits>(0,s)},"half ulp and sticky");
        tie(c,{power<Bits>(-Bits),power<Bits>(-3*Bits)},{power<Bits>(0,-s),power<Bits>(0,-s)},"half ulp and opposite sticky");
        tie(c,{power<Bits>(500),power<Bits>(500),power<Bits>(-Bits)},{power<Bits>(2),power<Bits>(2,-1),power<Bits>(0,-s)},"half ulp after cancellation");}}
    // Near the exponent limits: products and sums whose rounded result leaves the range.
    {std::vector<F<Bits>> a={power<Bits>(600000000),power<Bits>(-600000000)},b={power<Bits>(500000000),power<Bits>(-500000000)};check(a,b,"overflow");
     a[0]=power<Bits>(-600000000);b[0]=power<Bits>(-400000000);check(a,b,"near emin");a[0].limb[0]=0xffffffffu;check(a,b,"near emin raw");}
}
template<int Bits> void all_dots(){dot_cases<Bits>();if constexpr(Bits<1024)all_dots<Bits+32>();}
// ---- Matrix fixtures (row-major rows x cols; structure per column = per line of A^T A) ----
enum Kind {moderate,wide,cancel,sparse,qsc,ties,saturated};
const char* kind_name(Kind k){static const char* n[]={"moderate","wide","cancel","sparse","qsc","ties","saturated"};return n[k];}
template<int Bits> std::vector<F<Bits>> matrix(std::size_t rows,std::size_t cols,Kind kind,std::mt19937_64& rng){
    constexpr int N=Bits/32;std::vector<F<Bits>> A(rows*cols,zero<N>());auto at=[&](std::size_t r,std::size_t c)->F<Bits>&{return A[r*cols+c];};
    for(std::size_t c=0;c<cols;++c){
        long scale=kind==wide?long(rng()%2001)-1000:kind==qsc?long(rng()%121)-60:0;
        for(std::size_t r=0;r<rows;++r){F<Bits> x=reference::random_number<Bits>(rng,kind==qsc?15:kind==wide?30:4);x.exponent+=int(scale+(kind==qsc?-10:0));
            if(rng()%(kind==sparse?3:kind==qsc?4:10)==0)x=zero<N>();at(r,c)=x;}
        if(kind==wide){std::size_t r=rng()%rows;
            switch(c%9){case 1:case 2:at(r,c).exponent=int(scale-100-long(rng()%350));break;   // second band
                case 3:at(r,c).exponent=int(scale-5000);break;                                   // spread -> fallback
                case 4:for(std::size_t q=0;q<rows;q+=2)at(q,c).exponent=int(scale-70*long(q/2%5));break; // 5 bands
                case 7:for(std::size_t q=0;q<rows;q+=2)at(q,c).exponent=int(scale-100*long(q/2%3));break; // 3 bands
                case 8:for(std::size_t q=1;q<rows;q+=2)at(q,c).exponent=int(scale-90*long(q/2%4));break; // 4 bands
                case 5:at(r,c).exponent=600000000;break;case 6:for(std::size_t q=0;q<rows;++q)at(q,c).exponent=-600000000+int(q%5);break;
                default:break;}}
        if(kind==qsc&&c%17==5)at(rng()%rows,c).exponent=int(scale-150);
        if(kind==sparse){if(c%11==3)for(std::size_t r=0;r<rows;++r)at(r,c)=zero<N>();
            if(c%13==7)at(rng()%rows,c).status=rng()&1?invalid:exponent_overflow;}
    }
    if(kind==cancel)for(std::size_t c=0;c+1<cols;c+=2){
        // Column c: halves equal; column c+1: same with the second half negated, so the dot product cancels exactly.
        std::size_t h=rows/2;for(std::size_t r=0;r<h;++r){at(r+h,c)=at(r,c);at(r,c+1)=at(r,c);at(r+h,c+1)=negate(at(r,c));}
        if(rows%2&&c%4==0){at(rows-1,c)=power<Bits>(-400-long(c%3));at(rows-1,c+1)=power<Bits>(-401,(c%8)?1:-1);} // tiny remainder in a far band
        if(c%6==2&&h)at(0,c+1).limb[0]^=1;                                                                         // near cancellation
    }
    if(kind==ties&&rows>=4)for(std::size_t c=0;c<cols;++c){
        // C_ij = u_i u_j + 2^-(p_i+p_j) + X_i X_j (1 + s_i s_j); p_i+p_j = Bits gives exact ties.
        static const int p[]={1,Bits/2,Bits/2-1,Bits/2+1,Bits-1,3};F<Bits> u=power<Bits>(0);if(c%3==1)u.limb[0]|=1;if(c%5==4)u=negate(u);
        for(std::size_t r=0;r<rows;++r)at(r,c)=zero<N>();
        at(0,c)=u;at(1,c)=power<Bits>(-p[c%6]);at(2,c)=power<Bits>(200+long(c%2));at(3,c)=power<Bits>(200+long(c%2),c%4<2?1:-1);
    }
    // Largest magnitudes: all-ones significands, equal exponents (and one wider band), one sign: |sum| near K*2^(2P).
    if(kind==saturated)for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<cols;++c){F<Bits> x=power<Bits>(c%5==4&&r%2?-9:0);for(auto& l:x.limb)l=0xffffffffu;at(r,c)=x;}
    return A;
}
template<int Bits> std::vector<F<Bits>> transpose(const std::vector<F<Bits>>& A,std::size_t rows,std::size_t cols){
    std::vector<F<Bits>> T(A.size());for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<cols;++c)T[c*rows+r]=A[r*cols+c];return T;}
std::string bands(const LinalgReport& r){std::string s=r.zero_copy_output?"zero-copy bands":"bands";for(auto x:r.left_bands)s+=" "+std::to_string(x);
    return s+" fallback lines "+std::to_string(r.left_fallback)+"/"+std::to_string(r.right_fallback)+" pairs "+std::to_string(r.pairs.size())+" multi outputs "+std::to_string(r.multi_band_outputs)+" fallback outputs "+std::to_string(r.fallback_outputs);}
template<int Bits> void check_syrk(Linalg& g,const std::vector<F<Bits>>& A,std::size_t rows,std::size_t cols,bool lower,const std::string& label,bool verbose=false){
    F<Bits> sentinel=power<Bits>(12345);sentinel.status=4;std::vector<F<Bits>> C(cols*cols,sentinel);
    g.syrk(Bits,A.data(),rows,cols,C.data(),lower);Failures f;
    parallel_rows(cols,[&](std::size_t i){for(std::size_t j=0;j<=i;++j){
        auto want=reference::dot<Bits>(A.data()+i,std::ptrdiff_t(cols),A.data()+j,std::ptrdiff_t(cols),rows);
        if(!reference::equal<Bits>(C[i*cols+j],want))f.add("("+std::to_string(i)+","+std::to_string(j)+")\n got  "+text<Bits>(C[i*cols+j])+"\n want "+text<Bits>(want));
        if(j<i&&!reference::equal<Bits>(C[j*cols+i],lower?sentinel:want))f.add("upper ("+std::to_string(j)+","+std::to_string(i)+")");}});
    std::string l="syrk bits="+std::to_string(Bits)+" "+std::to_string(rows)+"x"+std::to_string(cols)+(lower?" lower ":" full ")+label;
    f.check(l+" ["+bands(g.report())+"]");if(verbose)std::cout<<l<<": "<<bands(g.report())<<"\n";
}
template<int Bits> void check_gemm(Linalg& g,bool ta,const std::vector<F<Bits>>& A,const std::vector<F<Bits>>& B,std::size_t m,std::size_t n,std::size_t k,const std::string& label){
    std::vector<F<Bits>> C(m*n,power<Bits>(1));g.gemm(Bits,ta,A.data(),B.data(),m,n,k,C.data());Failures f;
    parallel_rows(m,[&](std::size_t i){for(std::size_t j=0;j<n;++j){
        auto want=ta?reference::dot<Bits>(A.data()+i,std::ptrdiff_t(m),B.data()+j,std::ptrdiff_t(n),k):reference::dot<Bits>(A.data()+i*k,1,B.data()+j,std::ptrdiff_t(n),k);
        if(!reference::equal<Bits>(C[i*n+j],want))f.add("("+std::to_string(i)+","+std::to_string(j)+")\n got  "+text<Bits>(C[i*n+j])+"\n want "+text<Bits>(want));}});
    f.check("gemm bits="+std::to_string(Bits)+(ta?" AT":" A")+" m="+std::to_string(m)+" n="+std::to_string(n)+" k="+std::to_string(k)+" "+label+" ["+bands(g.report())+"]");
}
// Old entries for an update C <- RN(C - product): random values, RN(product) (exact cancellation down to its rounding
// error), the same far above / below, zeros and statuses.
template<int Bits> std::vector<F<Bits>> old_entries(const std::vector<F<Bits>>& product,std::mt19937_64& rng){
    std::vector<F<Bits>> C(product.size());
    for(std::size_t q=0;q<C.size();++q){const F<Bits>& p=product[q];switch(rng()%8){
        case 0:case 1:C[q]=p;break;case 2:C[q]=p.sign?with_exponent<Bits>(p,p.exponent+Bits+2):p;break;
        case 3:C[q]=p.sign?with_exponent<Bits>(negate(p),p.exponent-Bits-5):zero<Bits/32>();break;case 4:C[q]=zero<Bits/32>();break;
        case 5:C[q]=power<Bits>(9);C[q].status=rng()&1?invalid:division_by_zero;break;
        default:C[q]=reference::random_number<Bits>(rng,p.sign?3:300);if(p.sign)C[q].exponent+=p.exponent;}}
    return C;
}
template<int Bits> void check_syrk_update(Linalg& g,const std::vector<F<Bits>>& A,std::size_t rows,std::size_t cols,const std::string& label,std::mt19937_64& rng){
    std::vector<F<Bits>> P(cols*cols,zero<Bits/32>());g.syrk(Bits,A.data(),rows,cols,P.data(),false);
    auto C=old_entries<Bits>(P,rng),C0=C;g.syrk(Bits,A.data(),rows,cols,C.data(),true,true);Failures f;
    parallel_rows(cols,[&](std::size_t i){for(std::size_t j=0;j<cols;++j){
        auto want=j<=i?reference::dot_sub<Bits>(C0[i*cols+j],A.data()+i,std::ptrdiff_t(cols),A.data()+j,std::ptrdiff_t(cols),rows):C0[i*cols+j];
        if(!reference::equal<Bits>(C[i*cols+j],want))f.add("("+std::to_string(i)+","+std::to_string(j)+")\n got  "+text<Bits>(C[i*cols+j])+"\n want "+text<Bits>(want));}});
    f.check("syrk update bits="+std::to_string(Bits)+" "+std::to_string(rows)+"x"+std::to_string(cols)+" "+label+" ["+bands(g.report())+"]");
}
template<int Bits> void check_gemm_update(Linalg& g,const std::vector<F<Bits>>& A,const std::vector<F<Bits>>& B,std::size_t m,std::size_t n,std::size_t k,const std::string& label,std::mt19937_64& rng){
    std::vector<F<Bits>> P(m*n);g.gemm(Bits,true,A.data(),B.data(),m,n,k,P.data());
    auto C=old_entries<Bits>(P,rng),C0=C;g.gemm(Bits,true,A.data(),B.data(),m,n,k,C.data(),true);Failures f;
    parallel_rows(m,[&](std::size_t i){for(std::size_t j=0;j<n;++j){
        auto want=reference::dot_sub<Bits>(C0[i*n+j],A.data()+i,std::ptrdiff_t(m),B.data()+j,std::ptrdiff_t(n),k);
        if(!reference::equal<Bits>(C[i*n+j],want))f.add("("+std::to_string(i)+","+std::to_string(j)+")\n got  "+text<Bits>(C[i*n+j])+"\n want "+text<Bits>(want));}});
    f.check("gemm update bits="+std::to_string(Bits)+" m="+std::to_string(m)+" n="+std::to_string(n)+" k="+std::to_string(k)+" "+label+" ["+bands(g.report())+"]");
}
struct Config {const char* name;LinalgOptions options;};
std::vector<Config> configs(){LinalgOptions narrow;narrow.band_bits=16;narrow.max_bands=8;LinalgOptions zero_width;zero_width.band_bits=0;zero_width.max_bands=3;
    LinalgOptions fallback;fallback.force_fallback=true;return {{"default",{}},{"narrow bands",narrow},{"zero-width bands",zero_width},{"forced fallback",fallback}};}
template<int Bits> void matrices(std::vector<std::unique_ptr<Linalg>>& gpus,bool thorough){
    std::mt19937_64 rng(1000+Bits);
    std::vector<std::size_t> sizes=thorough?std::vector<std::size_t>{1,7,33,128,257}:std::vector<std::size_t>{7,33};
    for(std::size_t ci=0;ci<gpus.size();++ci){Linalg& g=*gpus[ci];std::string cfg=configs()[ci].name;bool main=ci==0;
        for(Kind kind:{moderate,wide,cancel,sparse,qsc,ties,saturated}){
            for(std::size_t s:sizes){if(!main&&s>33)continue;if(kind==ties&&s<4)continue;
                std::size_t rows=kind==qsc||kind==cancel?2*s+1:s;auto A=matrix<Bits>(rows,s,kind,rng);
                check_syrk<Bits>(g,A,rows,s,(s+ci)%2==0,cfg+" "+kind_name(kind));
                if(s>128&&!main)continue;
                std::size_t m=s,n=s+(s>1?3:0),k=rows;auto L=matrix<Bits>(k,m,kind,rng),R=matrix<Bits>(k,n,kind,rng);
                check_gemm<Bits>(g,true,L,R,m,n,k,cfg+" "+kind_name(kind));check_gemm<Bits>(g,false,transpose<Bits>(L,k,m),R,m,n,k,cfg+" "+kind_name(kind));
                if(s<=128){check_syrk_update<Bits>(g,A,rows,s,cfg+" "+kind_name(kind),rng);check_gemm_update<Bits>(g,L,R,m,n,k,cfg+" "+kind_name(kind),rng);}
            }
        }
    }
    if(thorough){auto A=matrix<Bits>(400,200,qsc,rng);check_syrk<Bits>(*gpus[0],A,400,200,true,"QSC-like",true);
        auto W=matrix<Bits>(400,200,wide,rng);check_syrk<Bits>(*gpus[0],W,400,200,false,"wide QSC-shape",true);}
    // Saturated sums over many K: the modulus product must exceed 2*K*2^(Pb+Pc) for every log2(K).
    for(std::size_t K=1;K<=(thorough?4096u:64u);K=K*2+(K%3==1)){auto A=matrix<Bits>(K,5,saturated,rng);check_syrk<Bits>(*gpus[0],A,K,5,false,"saturated K="+std::to_string(K));
        auto B=matrix<Bits>(K,2,saturated,rng);for(auto& x:B)x=negate(x);check_gemm<Bits>(*gpus[0],true,A,B,5,2,K,"saturated");}
    // Tight bound: a band width w and K = 2^t for which a product of primes (<= 65279, descending, as in the library)
    // exceeds the K*2^(2*(Bits+w)) bound by less than one bit, so a modulus set without the factor 2 for the sign fails.
    {std::vector<double> lg;for(unsigned q=65279;lg.size()<300;--q){bool p=true;for(unsigned d=2;d*d<=q;++d)if(q%d==0){p=false;break;}if(p)lg.push_back(std::log2(double(q)));}
     for(int w=0;w<=64;++w)for(int t=4;t<=(thorough?12:7);++t){double need=2.0*(Bits+w)+t,have=0;for(std::size_t i=0;have<=need;++i)have+=lg[i];
        if(have-need>0.8)continue;std::size_t K=std::size_t(1)<<t;std::vector<F<Bits>> A(K*2);
        for(auto& x:A){x=power<Bits>(0);for(auto& l:x.limb)l=0xffffffffu;}A[0].exponent=-w;
        check_syrk<Bits>(*gpus[0],A,K,2,false,"tight modulus bound w="+std::to_string(w)+" K="+std::to_string(K));w=65;break;}}
    // Page-aligned (wrapped without a copy) and misaligned (copied) inputs and outputs give the verified result.
    {const std::size_t rows=150,cols=70,e=sizeof(F<Bits>),page=std::size_t(getpagesize());auto A=matrix<Bits>(rows,cols,wide,rng);
     std::vector<F<Bits>> want(cols*cols,power<Bits>(7));gpus[0]->syrk(Bits,A.data(),rows,cols,want.data(),true);
     for(std::size_t ai:{0,1})for(std::size_t ci:{0,1}){void *pa=nullptr,*pc=nullptr;
        if(posix_memalign(&pa,page,(rows*cols+1)*e)||posix_memalign(&pc,page,(cols*cols+1)*e))throw std::bad_alloc();
        auto* a=static_cast<F<Bits>*>(pa)+ai;auto* c=static_cast<F<Bits>*>(pc)+ci;std::copy(A.begin(),A.end(),a);std::fill(c,c+cols*cols,power<Bits>(7));
        gpus[0]->syrk(Bits,a,rows,cols,c,true);bool same=std::equal(want.begin(),want.end(),c,[](const F<Bits>& x,const F<Bits>& y){return reference::equal<Bits>(x,y);});
        require(gpus[0]->report().zero_copy_output==(ci==0),"zero-copy output selection");std::free(pa);std::free(pc);
        require(same,"syrk with input offset "+std::to_string(ai)+" output offset "+std::to_string(ci)+" bits="+std::to_string(Bits));}}
    // Degenerate shapes: K = 0 and empty outputs.
    std::vector<F<Bits>> none,C(9,power<Bits>(3));gpus[0]->syrk(Bits,none.data(),0,3,C.data(),false);
    for(auto& x:C)require(reference::equal<Bits>(x,zero<Bits/32>()),"syrk with K=0 must give zeros");
    gpus[0]->gemm(Bits,false,none.data(),none.data(),0,0,5,none.data());
}
int main(int argc,char** argv){try{
    bool cpu_only=argc==2&&std::string(argv[1])=="--cpu-only";if(argc>1&&!cpu_only)throw std::invalid_argument("usage: test_limbforge_linalg [--cpu-only]");
    all_dots<64>();std::cout<<"exact_dot and exact_dot_add: all 31 precisions match exact MPFR products + mpfr_sum\n";
    if(cpu_only)return 0;
    std::vector<std::unique_ptr<Linalg>> gpus;for(auto& c:configs())gpus.push_back(std::make_unique<Linalg>(c.options));
    std::cout<<gpus[0]->device_name()<<"\n";
    matrices<224>(gpus,true);std::cout<<"syrk/gemm 224 bits: all outputs match MPFR\n";
    matrices<256>(gpus,true);std::cout<<"syrk/gemm 256 bits: all outputs match MPFR\n";
    matrices<64>(gpus,false);matrices<128>(gpus,false);matrices<160>(gpus,false);matrices<288>(gpus,false);
    matrices<384>(gpus,false);matrices<512>(gpus,false);matrices<1024>(gpus,false);
    std::cout<<"syrk/gemm 64,128,160,288,384,512,1024 bits: all outputs match MPFR\n";return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
