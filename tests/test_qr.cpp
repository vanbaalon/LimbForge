// Householder QR (Linalg::factor_qr, QRFactor::solve / apply_q) against an independent MPFR implementation of the
// documented rounding sequence (docs/numerics.md, "QR factorization and least squares"), bit for bit. The reference is
// left-looking and column by column: column j is reduced by one W/Y/update group per block of earlier reflectors
// (reference::dot / dot_sub, one mpfr_sum each), then its reflector is formed with mpfr_sqrt, mpfr_sub, mpfr_div, mpfr_mul.
#include "reference.hpp"
#include "limbforge/linalg.hpp"
#include <atomic>
#include <climits>
#include <map>
#include <thread>
#include <unistd.h>
using namespace limbforge;
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
template<int Bits> F<Bits> mp(Operation op,const F<Bits>& a,const F<Bits>& b){return reference::real<Bits>(op,a,b);}
template<int Bits> F<Bits> times_power(const F<Bits>& x,long e){reference::MP y(Bits);to_mpfr<Bits>(y.x,x);mpfr_mul_2si(y.x,y.x,e,MPFR_RNDN);return from_mpfr<Bits>(y.x);}
template<int Bits> struct RefQR {std::size_t m=0,n=0,nb=0,rank=0;int reason=0;word status=0;std::vector<F<Bits>> R,V,T,tau;};
template<int Bits> RefQR<Bits> ref_qr(const std::vector<F<Bits>>& A,std::size_t m,std::size_t n,std::size_t nb,int rank_bits){
    constexpr int N=Bits/32;RefQR<Bits> q;q.m=m;q.n=n;q.nb=nb=nb&&nb<n?nb:n;q.rank=n;std::size_t blocks=nb?(n+nb-1)/nb:0;
    std::vector<F<Bits>> a=A;q.V.assign(m*n,zero<N>());q.T.assign(blocks*nb*nb,zero<N>());q.tau.assign(n,zero<N>());long top=LONG_MIN;F<Bits>* V=q.V.data();
    for(std::size_t j=0;j<n;++j){
        // Reflectors [0, min(j, rank)) in groups of whole blocks (the last one possibly partial): w, y, one update rounding.
        const std::size_t qn=std::min(j,q.rank);
        for(std::size_t b0=0;b0<qn;b0+=nb){const std::size_t kb=std::min(b0+nb,qn)-b0;const F<Bits>* Tb=q.T.data()+(b0/nb)*nb*nb;std::vector<F<Bits>> w(kb),y(kb);
            parallel_rows(kb,[&](std::size_t i){w[i]=reference::dot<Bits>(V+(b0+i)*n+b0+i,std::ptrdiff_t(n),a.data()+(b0+i)*n+j,std::ptrdiff_t(n),m-b0-i);});
            for(std::size_t i=0;i<kb;++i)y[i]=reference::dot<Bits>(Tb+i,std::ptrdiff_t(nb),w.data(),1,i+1);
            parallel_rows(m-b0,[&](std::size_t r){F<Bits>& x=a[(b0+r)*n+j];x=reference::dot_sub<Bits>(x,V+(b0+r)*n+b0,1,y.data(),1,std::min(kb,r+1));});}
        if(q.rank<n)continue;
        const std::size_t len=m-j;const F<Bits> alpha=a[j*n+j];F<Bits> s=reference::dot<Bits>(a.data()+j*n+j,std::ptrdiff_t(n),a.data()+j*n+j,std::ptrdiff_t(n),len);
        bool below=false;for(std::size_t r=j+1;r<m;++r)below|=a[r*n+j].sign!=0;
        int reason=s.status?QRInfo::status_column:(!alpha.sign&&!below)?QRInfo::zero_column:0;F<Bits> beta=alpha,v0=power<Bits>(0);
        if(!reason&&below){beta=mp<Bits>(Operation::sqrt,s,s);if(alpha.sign>=0)beta=negate(beta);v0=mp<Bits>(Operation::sub,alpha,beta);}
        if(!reason&&rank_bits>0&&top!=LONG_MIN&&long(beta.exponent)+rank_bits<top)reason=QRInfo::small_column;
        if(reason){q.rank=j;q.reason=reason;q.status=s.status;continue;}
        F<Bits> t=zero<N>();
        if(below){long e=v0.exponent;V[j*n+j]=times_power<Bits>(v0,-e);for(std::size_t r=j+1;r<m;++r)V[r*n+j]=times_power<Bits>(a[r*n+j],-e);
            t=mp<Bits>(Operation::div,power<Bits>(1),reference::dot<Bits>(V+j*n+j,std::ptrdiff_t(n),V+j*n+j,std::ptrdiff_t(n),len));}
        else V[j*n+j]=power<Bits>(0);
        a[j*n+j]=beta;q.tau[j]=t;top=std::max(top,long(beta.exponent));
        // T column: Gram entries over the block's rows, then T_lc = -RN(tau * RN(sum_p T_lp G_pc)).
        const std::size_t b0=j/nb*nb,c=j-b0;F<Bits>* Tb=q.T.data()+(b0/nb)*nb*nb;std::vector<F<Bits>> G(c);
        parallel_rows(c,[&](std::size_t i){G[i]=reference::dot<Bits>(V+b0*n+b0+i,std::ptrdiff_t(n),V+b0*n+j,std::ptrdiff_t(n),m-b0);});
        Tb[c*nb+c]=t;for(std::size_t l=0;l<c;++l)Tb[l*nb+c]=negate(mp<Bits>(Operation::mul,t,reference::dot<Bits>(Tb+l*nb+l,1,G.data()+l,1,c-l)));
    }
    q.R.assign(n*n,zero<N>());for(std::size_t i=0;i<n;++i)for(std::size_t j=std::min(i,q.rank);j<n;++j)q.R[i*n+j]=a[i*n+j];
    return q;
}
// C (m x nrhs) <- Q^T C or Q C by the blocks of reflectors [0, rank).
template<int Bits> void ref_apply(const RefQR<Bits>& q,std::vector<F<Bits>>& C,std::size_t nrhs,bool qt){
    const std::size_t m=q.m,n=q.n,nb=q.nb,blocks=nb?(q.rank+nb-1)/nb:0;const F<Bits>* V=q.V.data();
    for(std::size_t s=0;s<blocks;++s){const std::size_t b=qt?s:blocks-1-s,b0=b*nb,kb=std::min(b0+nb,q.rank)-b0;const F<Bits>* Tb=q.T.data()+b*nb*nb;
        parallel_rows(nrhs,[&](std::size_t c){std::vector<F<Bits>> w(kb),y(kb);
            for(std::size_t i=0;i<kb;++i)w[i]=reference::dot<Bits>(V+(b0+i)*n+b0+i,std::ptrdiff_t(n),C.data()+(b0+i)*nrhs+c,std::ptrdiff_t(nrhs),m-b0-i);
            for(std::size_t i=0;i<kb;++i)y[i]=qt?reference::dot<Bits>(Tb+i,std::ptrdiff_t(nb),w.data(),1,i+1):reference::dot<Bits>(Tb+i*nb+i,1,w.data()+i,1,kb-i);
            for(std::size_t r=b0;r<m;++r){F<Bits>& x=C[r*nrhs+c];x=reference::dot_sub<Bits>(x,V+r*n+b0,1,y.data(),1,std::min(kb,r-b0+1));}});}
}
template<int Bits> std::vector<F<Bits>> ref_solve(const RefQR<Bits>& q,std::vector<F<Bits>> C,std::size_t nrhs){
    const std::size_t n=q.n,nb=q.nb;std::vector<F<Bits>> X(n*nrhs,zero<Bits/32>(invalid));if(q.rank<n)return X;
    ref_apply<Bits>(q,C,nrhs,true);std::copy(C.begin(),C.begin()+n*nrhs,X.begin());const F<Bits>* R=q.R.data();
    for(std::size_t i=n;i-->0;)parallel_rows(nrhs,[&](std::size_t c){F<Bits> v=X[i*nrhs+c];std::size_t r0=i/nb*nb,r1=std::min(n,r0+nb);
        for(std::size_t b=(n-1)/nb*nb;b>=r1&&b>r0;b-=nb)v=reference::dot_sub<Bits>(v,R+i*n+b,1,X.data()+b*nrhs+c,std::ptrdiff_t(nrhs),std::min(n,b+nb)-b);
        if(i+1<r1)v=reference::dot_sub<Bits>(v,R+i*n+i+1,1,X.data()+(i+1)*nrhs+c,std::ptrdiff_t(nrhs),r1-i-1);
        X[i*nrhs+c]=mp<Bits>(Operation::div,v,R[i*n+i]);});
    return X;
}
// ---- Inputs ----
enum Kind {moderate,qsc,wide,spread,graded,dependent,triangular};
const char* kind_name(Kind k){static const char* s[]={"moderate","qsc","wide","spread","graded","dependent","upper triangular"};return s[k];}
template<int Bits> std::vector<F<Bits>> matrix(std::size_t m,std::size_t n,Kind kind,std::mt19937_64& rng){
    constexpr int N=Bits/32;std::vector<F<Bits>> A(m*n,zero<N>());
    for(std::size_t c=0;c<n;++c){long scale=kind==qsc?long(rng()%121)-60:kind==wide?long(rng()%601)-300:kind==graded?long(c%2?20:-20):0;
        for(std::size_t r=0;r<m;++r){if(kind==triangular&&r>c)continue;
            F<Bits> x=reference::random_number<Bits>(rng,kind==qsc?15:kind==spread?400:4);x.exponent+=int(scale);if(kind==qsc&&rng()%4==0)x=zero<N>();A[r*n+c]=x;}
        if(kind==qsc&&c%17==5)A[(rng()%m)*n+c].exponent=int(scale-150);}
    if(kind==dependent&&n>=3)for(std::size_t r=0;r<m;++r){A[r*n+n-1]=A[r*n+1];A[r*n+n/2]=negate(A[r*n]);}
    return A;
}
template<int Bits> std::vector<F<Bits>> rhs(std::size_t m,std::size_t nrhs,std::mt19937_64& rng){std::vector<F<Bits>> B(m*nrhs);
    for(auto& x:B){x=reference::random_number<Bits>(rng,8);if(rng()%11==0)x=zero<Bits/32>();}return B;}
// ---- Checks ----
template<int Bits> void same(const F<Bits>* got,const std::vector<F<Bits>>& want,std::size_t cols,const std::string& label){
    for(std::size_t q=0;q<want.size();++q)require(reference::equal<Bits>(got[q],want[q]),label+": entry ("+std::to_string(q/cols)+","+std::to_string(q%cols)+")\n got  "+text<Bits>(got[q])+"\n want "+text<Bits>(want[q]));}
struct Run {const char* name;QROptions options;};
std::vector<Run> runs(std::size_t n){QROptions def,gpu16,host16,single,gpu1,gpu7;gpu16.block=16;gpu16.host_macs=0;gpu16.solve_host_macs=0;host16.block=16;host16.gpu=false;
    single.block=0;single.solve_host_macs=0;gpu1.block=1;gpu1.host_macs=0;gpu1.solve_host_macs=0;gpu7=gpu16;gpu7.block=7;
    std::vector<Run> r={{"default",def},{"gpu block 16",gpu16},{"host block 16",host16}};if(n<=130)r.push_back({"single block",single});
    if(n<=40)r.push_back({"gpu block 1",gpu1});if(n<=140)r.push_back({"gpu block 7",gpu7});return r;}
std::size_t factors_run=0,solves_run=0,applies_run=0,deficient=0,gpu_updates=0,host_updates=0;
template<int Bits> void factor(Linalg& la,const std::vector<F<Bits>>& A,std::size_t m,std::size_t n,const std::string& label,std::mt19937_64& rng,
                               std::vector<Run> rs={},int rank_bits=0,std::size_t expect_rank=SIZE_MAX){
    if(rs.empty())rs=runs(n);std::map<std::size_t,RefQR<Bits>> refs;
    for(std::size_t k=0;k<rs.size();++k){Run run=rs[k];run.options.rank_bits=rank_bits;const std::size_t nb=run.options.block&&run.options.block<n?run.options.block:n;
        auto found=refs.find(nb);if(found==refs.end())found=refs.emplace(nb,ref_qr<Bits>(A,m,n,nb,rank_bits)).first;const RefQR<Bits>& want=found->second;
        std::string l="qr bits="+std::to_string(Bits)+" "+std::to_string(m)+"x"+std::to_string(n)+" "+label+" ["+run.name+"]";
        if(expect_rank!=SIZE_MAX)require(want.rank==expect_rank,l+": reference rank "+std::to_string(want.rank)+", fixture expects "+std::to_string(expect_rank));
        bool aligned=k%2==0;Mat<Bits> a(A,aligned);QRFactor qr=la.factor_qr(Bits,a.p,m,n,run.options);const QRInfo& info=qr.info();
        require(info.rank==want.rank&&info.reason==want.reason&&info.status==want.status,l+": rank "+std::to_string(info.rank)+" reason "+std::to_string(info.reason)+
            " status "+std::to_string(info.status)+", want "+std::to_string(want.rank)+" "+std::to_string(want.reason)+" "+std::to_string(want.status));
        require(qr.rows()==m&&qr.cols()==n&&qr.block()==nb&&qr.bits()==Bits&&qr.full_rank()==(want.rank==n),l+": shape");
        same<Bits>(static_cast<const F<Bits>*>(qr.r()),want.R,n,l+" R");same<Bits>(static_cast<const F<Bits>*>(qr.v()),want.V,n,l+" V");
        same<Bits>(static_cast<const F<Bits>*>(qr.tau()),want.tau,1,l+" tau");same<Bits>(static_cast<const F<Bits>*>(qr.t()),want.T,nb?nb:1,l+" T");
        ++factors_run;deficient+=want.rank<n;gpu_updates+=info.gpu_updates;host_updates+=info.host_updates;
        // Least squares (multiple right-hand sides; a status in B on one run) and Q / Q^T applied in place.
        std::size_t nrhs=k%3==0?1:k%3==1?3:5;auto B=rhs<Bits>(m,nrhs,rng);if(k==1&&m>3)B[nrhs+1].status=invalid;
        {Mat<Bits> b(B,!aligned),x(n*nrhs,aligned);qr.solve(b.p,nrhs,x.p);same<Bits>(x.p,ref_solve<Bits>(want,B,nrhs),nrhs,"solve nrhs="+std::to_string(nrhs)+" "+l);++solves_run;}
        for(bool qt:{true,false}){auto C=B;ref_apply<Bits>(want,C,nrhs,qt);Mat<Bits> b(B,aligned!=qt);qr.apply_q(b.p,nrhs,qt);
            same<Bits>(b.p,C,nrhs,std::string(qt?"apply Q^T":"apply Q")+" nrhs="+std::to_string(nrhs)+" "+l);++applies_run;}
    }
}
template<int Bits> void special(Linalg& la,std::mt19937_64& rng){
    constexpr int N=Bits/32;
    // Zero column 3 (zero_column at 3), all-zero matrix (rank 0), zero first column.
    {auto A=matrix<Bits>(20,9,moderate,rng);for(std::size_t r=0;r<20;++r)A[r*9+3]=zero<N>();factor<Bits>(la,A,20,9,"zero column 3",rng,{},0,3);}
    factor<Bits>(la,std::vector<F<Bits>>(12,zero<N>()),4,3,"zero matrix",rng,{},0,0);
    {auto A=matrix<Bits>(40,20,qsc,rng);for(std::size_t r=0;r<40;++r)A[r*20]=zero<N>();factor<Bits>(la,A,40,20,"zero column 0",rng,{},0,0);}
    // A column that becomes exactly zero below the diagonal after reduction is still full rank (tau = 0); upper triangular input.
    factor<Bits>(la,matrix<Bits>(12,12,triangular,rng),12,12,"upper triangular square",rng,{},0,12);
    factor<Bits>(la,matrix<Bits>(30,18,triangular,rng),30,18,"upper triangular tall",rng,{},0,18);
    // Statuses: in column 4 (status_column at 4), in column 0, and in a column past a zero column (the zero column fails first).
    {auto A=matrix<Bits>(25,10,moderate,rng);A[7*10+4].status=invalid;factor<Bits>(la,A,25,10,"status at (7,4)",rng,{},0,4);
     A=matrix<Bits>(25,10,moderate,rng);A[20*10+0].status=exponent_overflow;factor<Bits>(la,A,25,10,"status at (20,0)",rng,{},0,0);
     A=matrix<Bits>(25,10,moderate,rng);A[3*10+8].status=invalid;for(std::size_t r=0;r<25;++r)A[r*10+6]=zero<N>();factor<Bits>(la,A,25,10,"zero column 6, status in 8",rng,{},0,6);}
    // Exactly dependent columns: full rank at rank_bits = 0 (rounding residue), detected with rank_bits = bits/2 (column 3 = -column 0 at n/2 = 3).
    {auto A=matrix<Bits>(41,7,dependent,rng);factor<Bits>(la,A,41,7,"dependent, no rank test",rng);factor<Bits>(la,A,41,7,"dependent, rank_bits",rng,{},Bits/2,3);
     auto B=matrix<Bits>(90,40,dependent,rng);factor<Bits>(la,B,90,40,"dependent 90x40, rank_bits",rng,{},Bits/2,20);}
    // Augmented Levenberg-Marquardt matrix [J; diag(d)]: identical to factoring the stacked matrix.
    {std::size_t m=37,n=21;auto J=matrix<Bits>(m,n,qsc,rng);std::vector<F<Bits>> d(n),S((m+n)*n,zero<N>());for(auto& x:d){x=reference::random_number<Bits>(rng,6);x.sign=1;}
     std::copy(J.begin(),J.end(),S.begin());for(std::size_t i=0;i<n;++i)S[(m+i)*n+i]=d[i];QROptions o;o.block=8;o.host_macs=0;o.solve_host_macs=0;
     QRFactor a=la.factor_qr_augmented(Bits,J.data(),m,n,d.data(),o),b=la.factor_qr(Bits,S.data(),m+n,n,o);
     auto want=ref_qr<Bits>(S,m+n,n,8,0);same<Bits>(static_cast<const F<Bits>*>(a.r()),want.R,n,"augmented R");same<Bits>(static_cast<const F<Bits>*>(a.v()),want.V,n,"augmented V");
     same<Bits>(static_cast<const F<Bits>*>(b.r()),want.R,n,"stacked R");auto B=rhs<Bits>(m+n,2,rng);std::vector<F<Bits>> X(2*n);a.solve(B.data(),2,X.data());same<Bits>(X.data(),ref_solve<Bits>(want,B,2),2,"augmented solve");}
    // Empty problems and a moved factor.
    {std::vector<F<Bits>> none;QRFactor e=la.factor_qr(Bits,none.data(),0,0);require(e.full_rank()&&e.info().rank==0,"0x0");e.solve(none.data(),3,none.data());
     auto A=matrix<Bits>(5,0,moderate,rng);QRFactor z=la.factor_qr(Bits,A.data(),5,0);require(z.info().rank==0,"5x0");
     A=matrix<Bits>(6,4,moderate,rng);QRFactor f=la.factor_qr(Bits,A.data(),6,4);QRFactor g=std::move(f);require(f.empty()&&!g.empty(),"move");
     std::vector<F<Bits>> X;g.solve(none.data(),0,X.data());bool threw=false;try{la.factor_qr(Bits,A.data(),3,4);}catch(const std::invalid_argument&){threw=true;}require(threw,"m < n must throw");}
}
template<int Bits> void suite(Linalg& la,bool thorough){
    std::mt19937_64 rng(5151+Bits);special<Bits>(la,rng);
    std::vector<std::tuple<std::size_t,std::size_t,Kind>> shapes={{1,1,moderate},{3,2,qsc},{33,7,graded},{128,64,wide},{70,40,spread},{257,129,qsc}};
    for(auto [m,n,kind]:shapes){if(m==257&&!thorough)continue;factor<Bits>(la,matrix<Bits>(m,n,kind,rng),m,n,kind_name(kind),rng);}
    if(!thorough)return;
    // Block boundaries for blocks 16 and 32 (square and tall).
    QROptions b16;b16.block=16;b16.host_macs=0;b16.solve_host_macs=0;QROptions b32=b16;b32.block=32;QROptions h32;h32.gpu=false;
    for(std::size_t n:{15,16,17,31,32,33,63,64,65}){std::vector<Run> rs={{"gpu block 16",b16},{"gpu block 32",b32},{"host block 32",h32}};
        factor<Bits>(la,matrix<Bits>(n+n/2+1,n,n%2?qsc:moderate,rng),n+n/2+1,n,"block boundary",rng,rs);
        if(n%16==0)factor<Bits>(la,matrix<Bits>(n,n,graded,rng),n,n,"square block boundary",rng,rs);}
}
int main(){try{
    Linalg la;std::cout<<la.device_name()<<"\n";
    suite<224>(la,true);std::cout<<"qr 224 bits: match the MPFR sequence\n";
    suite<256>(la,true);std::cout<<"qr 256 bits: match the MPFR sequence\n";
    suite<64>(la,false);suite<384>(la,false);std::cout<<"qr 64, 384 bits: match the MPFR sequence\n";
    // QSC-like least squares: J 800 x 400 with column scales 2^[-60, 60].
    for(int bits:{224,256}){std::mt19937_64 rng(800+bits);QROptions b16;b16.block=16;b16.host_macs=0;b16.solve_host_macs=0;std::vector<Run> rs={{"default",QROptions{}},{"gpu block 16",b16}};
        if(bits==224)factor<224>(la,matrix<224>(800,400,qsc,rng),800,400,"QSC-like",rng,rs);else factor<256>(la,matrix<256>(800,400,qsc,rng),800,400,"QSC-like",rng,rs);
        std::cout<<"QSC-like 800x400 at "<<bits<<" bits: match the MPFR sequence\n";}
    std::cout<<factors_run<<" factorizations ("<<deficient<<" rank-deficient; "<<gpu_updates<<" GPU and "<<host_updates<<" host block updates), "<<solves_run<<" solves and "
             <<applies_run<<" Q applications bit-identical to the reference\n";return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
