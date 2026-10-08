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
template<int Bits> struct RefQR {std::size_t m=0,n=0,nb=0,rank=0;int reason=0;word status=0;std::vector<F<Bits>> R,V,T,tau;std::vector<std::size_t> perm;std::size_t recomputed=0;};
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
// Column-pivoted QR (QROptions::pivot): a right-looking MPFR program of the pivoted sequence. Before step j the remaining
// column with the largest nu (status first; exact comparison; ties to the lowest original index) is swapped into place;
// column j is reduced by the block's reflectors so far (one rounding) and gives reflector j as in ref_qr; then every
// trailing column k gets w_k = P(v_j, a_k) (rows [j, m), a_k as at the block start), y_k = P(T[.][j], w_k), its row-j entry
// r_jk = D(a_jk; V[j][block], y_k) and nu_k = D(nu_k; r_jk, r_jk), recomputed from the reduced column when it falls to <= 0
// or more than bits/2 binades below its last exact value. After the block, X = D(X; V_b, Y) with these W and Y.
template<int Bits> RefQR<Bits> ref_qrp(const std::vector<F<Bits>>& A,std::size_t m,std::size_t n,std::size_t nb,int rank_bits){
    constexpr int N=Bits/32;RefQR<Bits> q;q.m=m;q.n=n;q.nb=nb=nb&&nb<n?nb:n;q.rank=n;std::size_t blocks=nb?(n+nb-1)/nb:0;
    std::vector<F<Bits>> a=A;q.V.assign(m*n,zero<N>());q.T.assign(blocks*nb*nb,zero<N>());q.tau.assign(n,zero<N>());long top=LONG_MIN;F<Bits>* V=q.V.data();
    q.perm.resize(n);for(std::size_t k=0;k<n;++k)q.perm[k]=k;
    std::vector<F<Bits>> nu(n),nuref(n),Wr(nb*n,zero<N>()),Yr(nb*n,zero<N>());
    parallel_rows(n,[&](std::size_t k){nu[k]=nuref[k]=reference::dot<Bits>(a.data()+k,std::ptrdiff_t(n),a.data()+k,std::ptrdiff_t(n),m);});
    auto greater=[](const F<Bits>& x,const F<Bits>& y){ // exact order, a status above every value
        if(x.status||y.status)return x.status&&!y.status;reference::MP u(Bits),v(Bits);to_mpfr<Bits>(u.x,x);to_mpfr<Bits>(v.x,y);return mpfr_cmp(u.x,v.x)>0;};
    // Applies reflectors [b0, b0+g) of block b0/nb to columns [c0, n) with Wr/Yr rows [0, g) (fresh = form them first).
    auto apply=[&](std::size_t b0,std::size_t g,std::size_t c0,bool fresh){const F<Bits>* Tb=q.T.data()+(b0/nb)*nb*nb;
        if(fresh)parallel_rows(n-c0,[&](std::size_t u){const std::size_t k=c0+u;
            for(std::size_t i=0;i<g;++i)Wr[i*n+k]=reference::dot<Bits>(V+(b0+i)*n+b0+i,std::ptrdiff_t(n),a.data()+(b0+i)*n+k,std::ptrdiff_t(n),m-b0-i);
            for(std::size_t i=0;i<g;++i)Yr[i*n+k]=reference::dot<Bits>(Tb+i,std::ptrdiff_t(nb),Wr.data()+k,std::ptrdiff_t(n),i+1);});
        parallel_rows(m-b0,[&](std::size_t r){for(std::size_t k=c0;k<n;++k){F<Bits>& x=a[(b0+r)*n+k];x=reference::dot_sub<Bits>(x,V+(b0+r)*n+b0,1,Yr.data()+k,std::ptrdiff_t(n),std::min(g,r+1));}});};
    for(std::size_t b0=0;b0<n&&q.rank==n;b0+=nb){const std::size_t kb=std::min(n,b0+nb)-b0;F<Bits>* Tb=q.T.data()+(b0/nb)*nb*nb;
        for(std::size_t c=0;c<kb;++c){const std::size_t j=b0+c;
            std::size_t p=j;for(std::size_t k=j+1;k<n;++k)if(greater(nu[k],nu[p])||(!greater(nu[p],nu[k])&&q.perm[k]<q.perm[p]))p=k;
            if(p!=j){for(std::size_t r=0;r<m;++r)std::swap(a[r*n+j],a[r*n+p]);for(std::size_t i=0;i<c;++i){std::swap(Wr[i*n+j],Wr[i*n+p]);std::swap(Yr[i*n+j],Yr[i*n+p]);}
                std::swap(nu[j],nu[p]);std::swap(nuref[j],nuref[p]);std::swap(q.perm[j],q.perm[p]);}
            // Column j by the block's reflectors so far (its y from the steps before), then reflector j as in ref_qr.
            std::vector<F<Bits>> x(m-b0);for(std::size_t r=0;r<m-b0;++r)x[r]=c?reference::dot_sub<Bits>(a[(b0+r)*n+j],V+(b0+r)*n+b0,1,Yr.data()+j,std::ptrdiff_t(n),std::min(c,r+1)):a[(b0+r)*n+j];
            const std::size_t len=m-j;const F<Bits> alpha=x[c];F<Bits> s=reference::dot<Bits>(x.data()+c,1,x.data()+c,1,len);
            bool below=false;for(std::size_t r=c+1;r<m-b0;++r)below|=x[r].sign!=0;
            int reason=s.status?QRInfo::status_column:(!alpha.sign&&!below)?QRInfo::zero_column:0;F<Bits> beta=alpha,v0=power<Bits>(0);
            if(!reason&&below){beta=mp<Bits>(Operation::sqrt,s,s);if(alpha.sign>=0)beta=negate(beta);v0=mp<Bits>(Operation::sub,alpha,beta);}
            if(!reason&&rank_bits>0&&top!=LONG_MIN&&long(beta.exponent)+rank_bits<top)reason=QRInfo::small_column;
            if(reason){q.rank=j;q.reason=reason;q.status=s.status;if(c)apply(b0,c,j,true);break;}
            for(std::size_t r=0;r<m-b0;++r)a[(b0+r)*n+j]=x[r];
            F<Bits> t=zero<N>();
            if(below){long e=v0.exponent;V[j*n+j]=times_power<Bits>(v0,-e);for(std::size_t r=j+1;r<m;++r)V[r*n+j]=times_power<Bits>(a[r*n+j],-e);
                t=mp<Bits>(Operation::div,power<Bits>(1),reference::dot<Bits>(V+j*n+j,std::ptrdiff_t(n),V+j*n+j,std::ptrdiff_t(n),len));}
            else V[j*n+j]=power<Bits>(0);
            a[j*n+j]=beta;q.tau[j]=t;top=std::max(top,long(beta.exponent));
            std::vector<F<Bits>> G(c);parallel_rows(c,[&](std::size_t i){G[i]=reference::dot<Bits>(V+b0*n+b0+i,std::ptrdiff_t(n),V+b0*n+j,std::ptrdiff_t(n),m-b0);});
            Tb[c*nb+c]=t;for(std::size_t l=0;l<c;++l)Tb[l*nb+c]=negate(mp<Bits>(Operation::mul,t,reference::dot<Bits>(Tb+l*nb+l,1,G.data()+l,1,c-l)));
            // Trailing columns: W and Y rows c, row j of R, downdated norms (recomputed when they collapse).
            std::vector<char> redo(n,0);
            parallel_rows(n-j-1,[&](std::size_t u){const std::size_t k=j+1+u;
                Wr[c*n+k]=reference::dot<Bits>(V+j*n+j,std::ptrdiff_t(n),a.data()+j*n+k,std::ptrdiff_t(n),len);
                Yr[c*n+k]=reference::dot<Bits>(Tb+c,std::ptrdiff_t(nb),Wr.data()+k,std::ptrdiff_t(n),c+1);
                F<Bits> r=reference::dot_sub<Bits>(a[j*n+k],V+j*n+b0,1,Yr.data()+k,std::ptrdiff_t(n),c+1);nu[k]=reference::dot_sub<Bits>(nu[k],&r,1,&r,1,1);
                redo[k]=nuref[k].sign&&!nuref[k].status&&!nu[k].status&&(nu[k].sign<=0||long(nu[k].exponent)+Bits/2<long(nuref[k].exponent));
                if(redo[k]){std::vector<F<Bits>> z;for(std::size_t i=j+1;i<m;++i)z.push_back(reference::dot_sub<Bits>(a[i*n+k],V+i*n+b0,1,Yr.data()+k,std::ptrdiff_t(n),c+1));
                    nu[k]=nuref[k]=reference::dot<Bits>(z.data(),1,z.data(),1,z.size());}});
            for(char x:redo)q.recomputed+=x!=0;
        }
        if(q.rank==n&&b0+kb<n)apply(b0,kb,b0+kb,false);
    }
    q.R.assign(n*n,zero<N>());for(std::size_t i=0;i<n;++i)for(std::size_t j=std::min(i,q.rank);j<n;++j)q.R[i*n+j]=a[i*n+j];
    return q;
}
// Pivoted solve: z = R[0:p,0:p]^-1 (Q^T b)[0:p) by the backward substitution (blocks min(nb, p)), x[perm[i]] = z_i, 0 beyond p.
template<int Bits> std::vector<F<Bits>> ref_solve_pivoted(const RefQR<Bits>& q,std::vector<F<Bits>> C,std::size_t nrhs){
    const std::size_t n=q.n,p=q.rank,nb=std::min(q.nb,p);std::vector<F<Bits>> X(n*nrhs,zero<Bits/32>());
    if(p<n&&q.reason==QRInfo::status_column){std::fill(X.begin(),X.end(),zero<Bits/32>(invalid|q.status));return X;}
    ref_apply<Bits>(q,C,nrhs,true);std::vector<F<Bits>> Z(C.begin(),C.begin()+p*nrhs);const F<Bits>* R=q.R.data();
    for(std::size_t i=p;i-->0;)parallel_rows(nrhs,[&](std::size_t c){F<Bits> v=Z[i*nrhs+c];std::size_t r0=i/nb*nb,r1=std::min(p,r0+nb);
        for(std::size_t b=(p-1)/nb*nb;b>=r1&&b>r0;b-=nb)v=reference::dot_sub<Bits>(v,R+i*n+b,1,Z.data()+b*nrhs+c,std::ptrdiff_t(nrhs),std::min(p,b+nb)-b);
        if(i+1<r1)v=reference::dot_sub<Bits>(v,R+i*n+i+1,1,Z.data()+(i+1)*nrhs+c,std::ptrdiff_t(nrhs),r1-i-1);
        Z[i*nrhs+c]=mp<Bits>(Operation::div,v,R[i*n+i]);});
    for(std::size_t i=0;i<p;++i)for(std::size_t c=0;c<nrhs;++c)X[q.perm[i]*nrhs+c]=Z[i*nrhs+c];
    return X;
}
// ---- Inputs ----
enum Kind {moderate,qsc,wide,spread,graded,dependent,triangular,extreme};
const char* kind_name(Kind k){static const char* s[]={"moderate","qsc","wide","spread","graded","dependent","upper triangular","extreme spread"};return s[k];}
template<int Bits> std::vector<F<Bits>> matrix(std::size_t m,std::size_t n,Kind kind,std::mt19937_64& rng){
    constexpr int N=Bits/32;std::vector<F<Bits>> A(m*n,zero<N>());
    for(std::size_t c=0;c<n;++c){long scale=kind==qsc?long(rng()%121)-60:kind==wide?long(rng()%601)-300:kind==graded?long(c%2?20:-20):0;
        for(std::size_t r=0;r<m;++r){if(kind==triangular&&r>c)continue;
            F<Bits> x=reference::random_number<Bits>(rng,kind==qsc?15:kind==spread?400:kind==extreme?3000:4);x.exponent+=int(scale);if(kind==qsc&&rng()%4==0)x=zero<N>();A[r*n+c]=x;}
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
     A=matrix<Bits>(25,10,moderate,rng);A[3*10+8].status=invalid;for(std::size_t r=0;r<25;++r)A[r*10+6]=zero<N>();factor<Bits>(la,A,25,10,"zero column 6, status in 8",rng,{},0,6);
     // A failure at the first column of a block leaves later columns unchanged (no reflectors to apply), statuses included.
     A=matrix<Bits>(40,30,moderate,rng);A[5*30+20].status=invalid;for(std::size_t r=0;r<40;++r)A[r*30]=zero<N>();factor<Bits>(la,A,40,30,"zero column 0, status in 20",rng,{},0,0);}
    // Exactly dependent columns: full rank at rank_bits = 0 (rounding residue), detected with rank_bits = bits/2 (column 3 = -column 0 at n/2 = 3).
    {auto A=matrix<Bits>(41,7,dependent,rng);factor<Bits>(la,A,41,7,"dependent, no rank test",rng);factor<Bits>(la,A,41,7,"dependent, rank_bits",rng,{},Bits/2,3);
     auto B=matrix<Bits>(90,40,dependent,rng);factor<Bits>(la,B,90,40,"dependent 90x40, rank_bits",rng,{},Bits/2,20);}
    // Exponent spreads beyond the panel's exact windows (its chunked sums fall back to exact_dot), several row blocks per column.
    factor<Bits>(la,matrix<Bits>(300,24,extreme,rng),300,24,"extreme spread",rng);
    // An entry 2^-999999995 next to entries near 2^10: scaling v by 2^-exponent(v0) leaves the range (exponent_overflow in v).
    {auto A=matrix<Bits>(150,12,moderate,rng);for(std::size_t r=0;r<150;++r)A[r*12].exponent+=10;A[77*12].exponent=-999999995;factor<Bits>(la,A,150,12,"tiny entry in column 0",rng);}
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
// Pivoted factor against ref_qrp (permutation, rank, R, V, tau, T, recomputed norms), its basic least-squares solution and Q
// applications; full-rank status-free inputs also against ref_qr of A P (the pivoted sequence is the unpivoted one of A P).
std::size_t pivoted_run=0,pivoted_deficient=0,recomputations=0;
template<int Bits> void factor_pivoted(Linalg& la,const std::vector<F<Bits>>& A,std::size_t m,std::size_t n,const std::string& label,std::mt19937_64& rng,
                                       int rank_bits=0,std::size_t expect_rank=SIZE_MAX,bool expect_recompute=false){
    QROptions def,gpu16,host7,single,gpu1;def.pivot=gpu16.pivot=host7.pivot=single.pivot=gpu1.pivot=true;gpu16.block=16;gpu16.host_macs=0;gpu16.solve_host_macs=0;
    host7.block=7;host7.gpu=false;single.block=0;single.solve_host_macs=0;gpu1.block=1;gpu1.host_macs=0;gpu1.solve_host_macs=0;
    std::vector<Run> rs={{"default",def},{"gpu block 16",gpu16},{"host block 7",host7}};if(n<=100)rs.push_back({"single block",single});if(n<=24)rs.push_back({"gpu block 1",gpu1});
    std::map<std::size_t,RefQR<Bits>> refs;bool crossed=false;
    for(std::size_t k=0;k<rs.size();++k){Run run=rs[k];run.options.rank_bits=rank_bits;const std::size_t nb=run.options.block&&run.options.block<n?run.options.block:n;
        auto found=refs.find(nb);if(found==refs.end())found=refs.emplace(nb,ref_qrp<Bits>(A,m,n,nb,rank_bits)).first;const RefQR<Bits>& want=found->second;
        std::string l="pivoted qr bits="+std::to_string(Bits)+" "+std::to_string(m)+"x"+std::to_string(n)+" "+label+" ["+run.name+"]";
        if(expect_rank!=SIZE_MAX)require(want.rank==expect_rank,l+": reference rank "+std::to_string(want.rank)+", fixture expects "+std::to_string(expect_rank));
        if(expect_recompute)require(want.recomputed>0,l+": fixture expects recomputed norms");
        bool aligned=k%2==0;Mat<Bits> a(A,aligned);QRFactor qr=la.factor_qr(Bits,a.p,m,n,run.options);const QRInfo& info=qr.info();
        require(qr.pivoted()&&info.rank==want.rank&&info.reason==want.reason&&info.status==want.status&&info.norm_recomputations==want.recomputed,l+": rank "+std::to_string(info.rank)+
            " reason "+std::to_string(info.reason)+" status "+std::to_string(info.status)+" recomputed "+std::to_string(info.norm_recomputations)+", want "+std::to_string(want.rank)+" "+
            std::to_string(want.reason)+" "+std::to_string(want.status)+" "+std::to_string(want.recomputed));
        for(std::size_t j=0;j<n;++j)require(qr.permutation()[j]==want.perm[j],l+": permutation entry "+std::to_string(j)+" is "+std::to_string(qr.permutation()[j])+", want "+std::to_string(want.perm[j]));
        same<Bits>(static_cast<const F<Bits>*>(qr.r()),want.R,n,l+" R");same<Bits>(static_cast<const F<Bits>*>(qr.v()),want.V,n,l+" V");
        same<Bits>(static_cast<const F<Bits>*>(qr.tau()),want.tau,1,l+" tau");same<Bits>(static_cast<const F<Bits>*>(qr.t()),want.T,nb?nb:1,l+" T");
        if(!crossed&&want.rank==n&&!want.status&&n<=150){crossed=true;std::vector<F<Bits>> AP(m*n);for(std::size_t r=0;r<m;++r)for(std::size_t j=0;j<n;++j)AP[r*n+j]=A[r*n+want.perm[j]];
            auto u=ref_qr<Bits>(AP,m,n,nb,rank_bits);require(u.rank==n,l+": A P reference rank");
            same<Bits>(u.R.data(),want.R,n,l+" R of A P");same<Bits>(u.V.data(),want.V,n,l+" V of A P");same<Bits>(u.T.data(),want.T,nb?nb:1,l+" T of A P");same<Bits>(u.tau.data(),want.tau,1,l+" tau of A P");}
        ++pivoted_run;pivoted_deficient+=want.rank<n;recomputations+=want.recomputed;
        std::size_t nrhs=k%3==0?1:k%3==1?3:5;auto B=rhs<Bits>(m,nrhs,rng);if(k==1&&m>3)B[nrhs+1].status=invalid;
        {Mat<Bits> b(B,!aligned),x(n*nrhs,aligned);qr.solve(b.p,nrhs,x.p);same<Bits>(x.p,ref_solve_pivoted<Bits>(want,B,nrhs),nrhs,"pivoted solve nrhs="+std::to_string(nrhs)+" "+l);}
        for(bool qt:{true,false}){auto C=B;ref_apply<Bits>(want,C,nrhs,qt);Mat<Bits> b(B,aligned!=qt);qr.apply_q(b.p,nrhs,qt);same<Bits>(b.p,C,nrhs,std::string(qt?"pivoted Q^T":"pivoted Q")+" "+l);}
    }
}
template<int Bits> void pivoted(Linalg& la,std::mt19937_64& rng){
    constexpr int N=Bits/32;
    for(auto [m,n,kind]:std::vector<std::tuple<std::size_t,std::size_t,Kind>>{{1,1,moderate},{5,3,qsc},{33,7,graded},{60,30,qsc},{64,32,wide},{70,40,spread},{150,60,moderate}})
        factor_pivoted<Bits>(la,matrix<Bits>(m,n,kind,rng),m,n,kind_name(kind),rng);
    // Rank deficiency: exactly dependent columns (found with rank_bits), zero columns (moved last), duplicate columns (ties go to
    // the lower original index), a status (chosen first: rank 0), and nearly parallel columns (downdated norms recomputed).
    factor_pivoted<Bits>(la,matrix<Bits>(41,7,dependent,rng),41,7,"dependent",rng,Bits/2,5);
    factor_pivoted<Bits>(la,matrix<Bits>(90,40,dependent,rng),90,40,"dependent 90x40",rng,Bits/2,38);
    {auto A=matrix<Bits>(30,12,moderate,rng);for(std::size_t r=0;r<30;++r){A[r*12+2]=zero<N>();A[r*12+7]=zero<N>();}factor_pivoted<Bits>(la,A,30,12,"zero columns 2, 7",rng,0,10);}
    {auto A=matrix<Bits>(25,9,moderate,rng);for(std::size_t r=0;r<25;++r){A[r*9+6]=A[r*9+2];A[r*9+8]=A[r*9+2];}factor_pivoted<Bits>(la,A,25,9,"duplicate columns",rng,Bits/2,7);}
    {auto A=matrix<Bits>(25,10,moderate,rng);A[7*10+4].status=invalid;factor_pivoted<Bits>(la,A,25,10,"status at (7,4)",rng,0,0);}
    {auto A=matrix<Bits>(80,20,moderate,rng);for(std::size_t r=0;r<80;++r)for(std::size_t j:{5,13}){F<Bits> t=reference::random_number<Bits>(rng,4);t.exponent-=Bits/4+10;A[r*20+j]=reference::real<Bits>(Operation::add,A[r*20+j-1],t);}
     factor_pivoted<Bits>(la,A,80,20,"nearly parallel columns",rng,0,SIZE_MAX,true);}
    factor_pivoted<Bits>(la,matrix<Bits>(300,150,qsc,rng),300,150,"QSC-like",rng);
}
template<int Bits> void suite(Linalg& la,bool thorough){
    std::mt19937_64 rng(5151+Bits);special<Bits>(la,rng);pivoted<Bits>(la,rng);
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
    std::cout<<pivoted_run<<" pivoted factorizations ("<<pivoted_deficient<<" rank-deficient, "<<recomputations<<" recomputed norms) bit-identical to the pivoted reference\n";
    std::cout<<factors_run<<" factorizations ("<<deficient<<" rank-deficient; "<<gpu_updates<<" GPU and "<<host_updates<<" host block updates), "<<solves_run<<" solves and "
             <<applies_run<<" Q applications bit-identical to the reference\n";return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
