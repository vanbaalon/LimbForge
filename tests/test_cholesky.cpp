// Blocked Cholesky and triangular solves (Linalg::cholesky, trsm, cholesky_solve) against an independent MPFR
// implementation of the documented rounding sequence (docs/numerics.md, "Cholesky factorization"), bit for bit.
// The reference is left-looking and column by column: entry (i, j) is A_ij updated by one exact-dot rounding per
// earlier column block (reference::dot_sub), then the in-block dot, then sqrt or a division (MPFR).
#include "reference.hpp"
#include "limbforge/linalg.hpp"
#include <atomic>
#include <map>
#include <mutex>
#include <thread>
#include <unistd.h>
using namespace limbforge;
using limbforge::Operation;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
template<int Bits> using F=Float<Bits>;
template<int Bits> F<Bits> power(long e,int sign=1){F<Bits> x=zero<Bits/32>();x.limb[Bits/32-1]=0x80000000u;x.sign=sign;x.exponent=int(e);return x;}
template<int Bits> std::string text(const F<Bits>& x){std::string s="sign="+std::to_string(x.sign)+" exp="+std::to_string(x.exponent)+" status="+std::to_string(x.status)+" limbs=";
    for(int i=Bits/32-1;i>=0;--i){char b[9];std::snprintf(b,9,"%08x",x.limb[i]);s+=b;}return s;}
template<class Job> void parallel_rows(std::size_t n,Job job){if(n<2){if(n)job(std::size_t(0));return;}std::atomic<std::size_t> next{0};std::vector<std::thread> w;
    for(unsigned t=0;t<std::min<std::size_t>(n,std::max(1u,std::thread::hardware_concurrency()));++t)w.emplace_back([&]{for(std::size_t i;(i=next.fetch_add(1))<n;)job(i);});for(auto& x:w)x.join();}
// Page-aligned storage (GPU in place) or offset by one element (copied by the library).
template<int Bits> struct Mat {
    void* raw=nullptr;F<Bits>* p=nullptr;std::size_t size=0;
    Mat(std::size_t count,bool aligned):size(count){std::size_t page=std::size_t(getpagesize());if(posix_memalign(&raw,page,(count+1)*sizeof(F<Bits>)+page))throw std::bad_alloc();
        p=static_cast<F<Bits>*>(raw)+(aligned?0:1);for(std::size_t i=0;i<count;++i)p[i]=power<Bits>(777,-1);}
    explicit Mat(const std::vector<F<Bits>>& v,bool aligned):Mat(v.size(),aligned){std::copy(v.begin(),v.end(),p);}
    ~Mat(){std::free(raw);} Mat(const Mat&)=delete;
    std::vector<F<Bits>> vec()const{return std::vector<F<Bits>>(p,p+size);}
};
// ---- Reference (independent MPFR implementation of the documented sequence) ----
struct Info {std::size_t pivot;int sign;word status;};
template<int Bits> Info ref_cholesky(const std::vector<F<Bits>>& A,std::size_t n,std::size_t nb,std::vector<F<Bits>>& L){
    constexpr int N=Bits/32;L.assign(n*n,zero<N>());if(!nb||nb>n)nb=n;Info info{n,0,0};
    // s_ij: A_ij, minus one rounded exact dot per complete column block before j's block, minus the in-block dot over [kb, j).
    auto value=[&](std::size_t i,std::size_t j){F<Bits> v=A[i*n+j];std::size_t kb=j/nb*nb;
        for(std::size_t b=0;b<kb;b+=nb)v=reference::dot_sub<Bits>(v,L.data()+i*n+b,1,L.data()+j*n+b,1,nb);
        return reference::dot_sub<Bits>(v,L.data()+i*n+kb,1,L.data()+j*n+kb,1,j-kb);};
    for(std::size_t j=0;j<n;++j){F<Bits> s=value(j,j);
        if(s.status||s.sign<=0){info={j,s.status?0:s.sign,s.status};for(std::size_t i=j;i<n;++i)for(std::size_t k=j;k<=i;++k)L[i*n+k]=zero<N>(invalid);return info;}
        L[j*n+j]=reference::real<Bits>(Operation::sqrt,s,s);
        parallel_rows(n-j-1,[&](std::size_t r){std::size_t i=j+1+r;L[i*n+j]=reference::real<Bits>(Operation::div,value(i,j),L[j*n+j]);});
    }
    return info;
}
// passes: 1 forward (L X = B), 2 backward (L^T X = B), 3 both. X holds B on entry.
template<int Bits> void ref_solve(const std::vector<F<Bits>>& L,std::size_t n,std::vector<F<Bits>>& X,std::size_t nrhs,std::size_t nb,int passes){
    if(!nb||nb>n)nb=n;const F<Bits>* l=L.data();
    if(passes&1)for(std::size_t i=0;i<n;++i)parallel_rows(nrhs,[&](std::size_t c){F<Bits> v=X[i*nrhs+c];std::size_t r0=i/nb*nb;
        for(std::size_t b=0;b<r0;b+=nb)v=reference::dot_sub<Bits>(v,l+i*n+b,1,X.data()+b*nrhs+c,std::ptrdiff_t(nrhs),nb);
        v=reference::dot_sub<Bits>(v,l+i*n+r0,1,X.data()+r0*nrhs+c,std::ptrdiff_t(nrhs),i-r0);X[i*nrhs+c]=reference::real<Bits>(Operation::div,v,l[i*n+i]);});
    if(passes&2)for(std::size_t i=n;i-->0;)parallel_rows(nrhs,[&](std::size_t c){F<Bits> v=X[i*nrhs+c];std::size_t r0=i/nb*nb,r1=std::min(n,r0+nb);
        for(std::size_t b=(n-1)/nb*nb;b>=r1&&b>r0;b-=nb)v=reference::dot_sub<Bits>(v,l+b*n+i,std::ptrdiff_t(n),X.data()+b*nrhs+c,std::ptrdiff_t(nrhs),std::min(n,b+nb)-b);
        if(i+1<r1)v=reference::dot_sub<Bits>(v,l+(i+1)*n+i,std::ptrdiff_t(n),X.data()+(i+1)*nrhs+c,std::ptrdiff_t(nrhs),r1-i-1);
        X[i*nrhs+c]=reference::real<Bits>(Operation::div,v,l[i*n+i]);});
}
// ---- Inputs ----
enum Kind {moderate,qsc,wide,banded,graded,rank_deficient,spread};
const char* kind_name(Kind k){static const char* s[]={"moderate","qsc","wide","banded","graded","rank-deficient","spread"};return s[k];}
// A = J^T J + damping (Marquardt: a_ii += 2^(e_ii - 10)) for J (2n x n), or a banded diagonally dominant matrix.
template<int Bits> std::vector<F<Bits>> spd(Linalg& la,std::size_t n,Kind kind,std::mt19937_64& rng){
    constexpr int N=Bits/32;std::vector<F<Bits>> A(n*n,zero<N>());
    if(kind==banded){for(std::size_t i=0;i<n;++i){A[i*n+i]=reference::random_number<Bits>(rng,0);A[i*n+i].sign=1;A[i*n+i].exponent=3;
        for(std::size_t d=1;d<=3&&d<=i;++d)if(d!=2){F<Bits> x=reference::random_number<Bits>(rng,0);x.exponent=-1;A[i*n+i-d]=A[(i-d)*n+i]=x;}}return A;}
    // A = L0 L0^T (one rounding per entry) for unit-diagonal L0 with off-diagonal exponents in [-700, 0]: lines of the
    // trailing updates then span more than max_spread (exact host fallback with an addend).
    if(kind==spread){std::vector<F<Bits>> L0(n*n,zero<N>()),T(n*n,zero<N>());
        for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<=i;++j){F<Bits> x=reference::random_number<Bits>(rng,0);
            if(i==j){x.sign=1;x.exponent=0;}else if(rng()%10<3)x=zero<N>();else x.exponent=-int(rng()%701);L0[i*n+j]=T[j*n+i]=x;}
        la.gemm(Bits,false,L0.data(),T.data(),n,n,n,A.data());return A;}
    std::size_t m=2*n+1;std::vector<F<Bits>> J(m*n);
    for(std::size_t c=0;c<n;++c){long scale=kind==qsc?long(rng()%121)-60:kind==wide?long(rng()%601)-300:kind==graded?long(c%2?20:-20):0;
        for(std::size_t r=0;r<m;++r){F<Bits> x=reference::random_number<Bits>(rng,kind==qsc?15:4);x.exponent+=int(scale);if(kind==qsc&&rng()%4==0)x=zero<N>();J[r*n+c]=x;}
        if(kind==qsc&&c%17==5)J[(rng()%m)*n+c].exponent=int(scale-150);}
    if(kind==rank_deficient&&n>=3)for(std::size_t r=0;r<m;++r){J[r*n+n-1]=J[r*n+1];J[r*n+n/2]=negate(J[r*n+0]);}
    la.syrk(Bits,J.data(),m,n,A.data(),false);
    if(kind!=rank_deficient)for(std::size_t i=0;i<n;++i){F<Bits>& a=A[i*n+i];if(a.sign)a=reference::real<Bits>(Operation::add,a,power<Bits>(a.exponent-10));}
    return A;
}
template<int Bits> std::vector<F<Bits>> rhs(std::size_t n,std::size_t nrhs,std::mt19937_64& rng){std::vector<F<Bits>> B(n*nrhs);
    for(auto& x:B){x=reference::random_number<Bits>(rng,8);if(rng()%11==0)x=zero<Bits/32>();}return B;}
// ---- Checks ----
template<int Bits> void same(const std::vector<F<Bits>>& got,const std::vector<F<Bits>>& want,std::size_t cols,const std::string& label){
    for(std::size_t q=0;q<want.size();++q)require(reference::equal<Bits>(got[q],want[q]),label+": entry ("+std::to_string(q/cols)+","+std::to_string(q%cols)+")\n got  "+text<Bits>(got[q])+"\n want "+text<Bits>(want[q]));}
struct Run {const char* name;FactorOptions options;};
std::vector<Run> runs(std::size_t n){FactorOptions def,gpu16,host16,single,gpu1;gpu16.block=16;gpu16.host_macs=0;gpu16.solve_host_macs=0;host16.block=16;host16.gpu=false;single.block=0;gpu1.block=1;gpu1.host_macs=0;gpu1.solve_host_macs=0;
    std::vector<Run> r={{"default",def},{"gpu block 16",gpu16},{"host block 16",host16}};if(n<=130)r.push_back({"single block",single});if(n<=40)r.push_back({"gpu block 1",gpu1});
    FactorOptions gpu7=gpu16;gpu7.block=7;if(n<=140)r.push_back({"gpu block 7",gpu7});return r;}
bool expect_spd=true;std::size_t cases_run=0,solves_run=0,failed_pivots=0,gpu_updates=0,host_updates=0;
template<int Bits> void factor(Linalg& la,const std::vector<F<Bits>>& A,std::size_t n,const std::string& label,std::mt19937_64& rng,bool solve=true,std::vector<Run> rs={}){
    if(rs.empty())rs=runs(n);std::map<std::size_t,std::pair<Info,std::vector<F<Bits>>>> refs;
    for(std::size_t q=0;q<rs.size();++q){const Run& run=rs[q];std::size_t nb=run.options.block?std::min(run.options.block,n):n;
        auto found=refs.find(nb);if(found==refs.end()){std::vector<F<Bits>> L;Info info=ref_cholesky<Bits>(A,n,nb,L);found=refs.emplace(nb,std::make_pair(info,std::move(L))).first;}
        const Info& want=found->second.first;const auto& WL=found->second.second;require(!expect_spd||want.pivot==n,"SPD fixture failed at pivot "+std::to_string(want.pivot)+": "+label);
        std::string l="cholesky bits="+std::to_string(Bits)+" n="+std::to_string(n)+" "+label+" ["+run.name+"]";
        bool aligned=q%2==0,inplace=q==2;Mat<Bits> a(A,aligned),out(n*n,!aligned);
        CholeskyInfo info=la.cholesky(Bits,a.p,n,inplace?a.p:out.p,run.options);const Mat<Bits>& L=inplace?a:out;
        require(info.pivot==want.pivot&&info.pivot_sign==want.sign&&info.pivot_status==want.status,l+": pivot "+std::to_string(info.pivot)+" sign "+std::to_string(info.pivot_sign)+
            " status "+std::to_string(info.pivot_status)+", want "+std::to_string(want.pivot)+" "+std::to_string(want.sign)+" "+std::to_string(want.status));
        same<Bits>(L.vec(),WL,n,l);++cases_run;failed_pivots+=want.pivot<n;gpu_updates+=info.gpu_updates;host_updates+=info.host_updates;
        if(!solve||want.pivot<n)continue;
        // Solves with this factor: L L^T X = B (multiple right-hand sides), forward and transposed trsm, in place.
        std::size_t nrhs=q%3==0?1:q%3==1?3:5;auto B=rhs<Bits>(n,nrhs,rng);if(q==1&&n>3)B[nrhs+1].status=invalid;
        for(int passes:{3,1,2}){auto X=B;ref_solve<Bits>(WL,n,X,nrhs,nb,passes);Mat<Bits> b(B,!aligned),x(n*nrhs,aligned);bool in=passes==2&&q%2==1;
            if(passes==3)la.cholesky_solve(Bits,L.p,n,b.p,nrhs,in?b.p:x.p,run.options);else la.trsm(Bits,passes==2,L.p,n,b.p,nrhs,in?b.p:x.p,run.options);
            same<Bits>((in?b:x).vec(),X,nrhs,(passes==3?"cholesky_solve":passes==1?"trsm":"trsm transposed")+std::string(" nrhs=")+std::to_string(nrhs)+" "+l);++solves_run;}
    }
}
template<int Bits> void failures(Linalg& la,std::mt19937_64& rng){
    constexpr int N=Bits/32;auto one=power<Bits>(0),two=power<Bits>(1);
    factor<Bits>(la,{one,one,one,one},2,"exact zero pivot",rng);
    factor<Bits>(la,{one,two,two,one},2,"indefinite",rng);
    factor<Bits>(la,{negate(one)},1,"negative 1x1",rng);
    factor<Bits>(la,std::vector<F<Bits>>(9,zero<N>()),3,"zero matrix",rng);
    {auto A=spd<Bits>(la,40,moderate,rng);A[37*40+37]=power<Bits>(40,-1);factor<Bits>(la,A,40,"negative pivot 37",rng);
     A=spd<Bits>(la,40,moderate,rng);A[20*40+5].status=invalid;factor<Bits>(la,A,40,"status at (20,5)",rng);
     A=spd<Bits>(la,40,qsc,rng);A[0].status=exponent_overflow;factor<Bits>(la,A,40,"status at (0,0)",rng);
     A=spd<Bits>(la,40,moderate,rng);for(std::size_t j=0;j<40;++j)A[std::max<std::size_t>(j,23)*40+std::min<std::size_t>(j,23)]=zero<N>();factor<Bits>(la,A,40,"zero row 23",rng);
     A=spd<Bits>(la,40,moderate,rng);for(std::size_t j=0;j<17;++j)A[17*40+j]=A[16*40+j];A[17*40+17]=A[16*40+16];A[17*40+16]=A[16*40+16];  // rows 16 and 17 equal
     for(std::size_t i=18;i<40;++i)A[i*40+17]=A[i*40+16];factor<Bits>(la,A,40,"duplicated row 17",rng);}
    // trsm with a zero diagonal entry: division_by_zero propagates through later rows.
    {std::size_t n=20,nrhs=2;std::vector<F<Bits>> L(n*n,zero<N>());for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<=i;++j)L[i*n+j]=reference::random_number<Bits>(rng,3);
     L[9*n+9]=zero<N>();auto B=rhs<Bits>(n,nrhs,rng);FactorOptions o;o.block=4;o.host_macs=0;o.solve_host_macs=0;
     for(int t:{1,2}){auto X=B;ref_solve<Bits>(L,n,X,nrhs,4,t);std::vector<F<Bits>> got(n*nrhs);la.trsm(Bits,t==2,L.data(),n,B.data(),nrhs,got.data(),o);same<Bits>(got,X,nrhs,"trsm zero diagonal bits="+std::to_string(Bits));}}
    // Empty problems.
    {std::vector<F<Bits>> none;CholeskyInfo i=la.cholesky(Bits,none.data(),0,none.data());require(i.pivot==0,"n=0");la.cholesky_solve(Bits,none.data(),0,none.data(),3,none.data());
     auto A=spd<Bits>(la,5,moderate,rng);std::vector<F<Bits>> L(25);la.cholesky(Bits,A.data(),5,L.data());la.cholesky_solve(Bits,L.data(),5,none.data(),0,none.data());}
}
template<int Bits> void suite(Linalg& la,bool thorough){
    std::mt19937_64 rng(4242+Bits);expect_spd=false;failures<Bits>(la,rng);expect_spd=true;
    std::vector<std::pair<std::size_t,Kind>> sizes={{1,moderate},{2,qsc},{7,graded},{33,qsc},{128,wide},{257,qsc}};
    for(auto [n,kind]:sizes){if(n==257&&!thorough)continue;factor<Bits>(la,spd<Bits>(la,n,kind,rng),n,kind_name(kind),rng);}
    factor<Bits>(la,spd<Bits>(la,48,banded,rng),48,"banded",rng);factor<Bits>(la,spd<Bits>(la,70,spread,rng),70,"spread",rng);expect_spd=false;factor<Bits>(la,spd<Bits>(la,33,rank_deficient,rng),33,"rank-deficient",rng);expect_spd=true;
    if(!thorough)return;
    // Block boundaries for blocks 16 and 64.
    FactorOptions b16;b16.block=16;b16.host_macs=0;b16.solve_host_macs=0;FactorOptions b64;b64.host_macs=0;b64.solve_host_macs=0;FactorOptions h64;h64.gpu=false;
    for(std::size_t n:{15,16,17,32,47,63,64,65,129}){std::vector<Run> rs={{"gpu block 16",b16},{"gpu block 64",b64},{"host block 64",h64}};
        factor<Bits>(la,spd<Bits>(la,n,n%2?qsc:moderate,rng),n,"block boundary",rng,true,rs);}
    factor<Bits>(la,spd<Bits>(la,140,wide,rng),140,"wide",rng,true,{{"default",FactorOptions{}},{"gpu block 16",b16}});
}
int main(){try{
    Linalg la;std::cout<<la.device_name()<<"\n";
    suite<224>(la,true);std::cout<<"cholesky/solve 224 bits: match the MPFR sequence\n";
    suite<256>(la,true);std::cout<<"cholesky/solve 256 bits: match the MPFR sequence\n";
    suite<64>(la,false);suite<384>(la,false);std::cout<<"cholesky/solve 64, 384 bits: match the MPFR sequence\n";
    suite<384>(la,true);std::cout<<"cholesky/solve 384 bits (thorough): match the MPFR sequence\n";
    // QSC-like normal equations: A = J^T J + mu I, J 800 x 400 with column scales 2^[-60, 60].
    for(int bits:{224,256}){std::mt19937_64 rng(400+bits);FactorOptions b16;b16.block=16;b16.host_macs=0;b16.solve_host_macs=0;std::vector<Run> rs={{"default",FactorOptions{}},{"gpu block 16",b16}};
        if(bits==224)factor<224>(la,spd<224>(la,400,qsc,rng),400,"QSC n=400",rng,true,rs);else factor<256>(la,spd<256>(la,400,qsc,rng),400,"QSC n=400",rng,true,rs);
        std::cout<<"QSC-like n=400 at "<<bits<<" bits: match the MPFR sequence\n";}
    std::cout<<cases_run<<" factorizations ("<<failed_pivots<<" with a failing pivot; "<<gpu_updates<<" GPU and "<<host_updates<<" host trailing updates) and "<<solves_run<<" solves bit-identical to the reference\n";return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
