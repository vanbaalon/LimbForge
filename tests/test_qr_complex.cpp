// Complex Householder QR (Linalg::factor_qr_complex, ComplexQRFactor::solve / apply_q) against an independent MPFR replay of
// the documented rounding sequence (docs/numerics.md, "Complex QR factorization and least squares"), bit for bit. The replay is
// left-looking and column by column on complex entries: every complex dot product or update is two mpfr_sum calls (one per
// real component) over the exact products of the components (reference::dot / dot_sub on the component lists), and the
// reflector uses mpfr_sqrt, mpfr_sub, mpfr_mul_2si, mpfr_div and reference::dot2_add. Also: a real matrix gives the real QR
// (Linalg::factor_qr) bit for bit in the real parts and zero imaginary parts. Column pivoting (QROptions::pivot) against a separate
// right-looking replay of the pivoted sequence (ref_qrp: permutation, rank, recomputed norms, R, V, tau, T, basic solutions), and
// full-rank status-free pivoted factors against the unpivoted replay of A P.
// Usage: test_limbforge_qr_complex [--quick] [--all-widths] (every 32-bit width from 64 to 1024 bits on small GPU shapes).
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
template<int Bits> using CF=Complex<Bits/32>;
template<int Bits> F<Bits> power(long e,int sign=1){F<Bits> x=zero<Bits/32>();x.limb[Bits/32-1]=0x80000000u;x.sign=sign;x.exponent=int(e);return x;}
template<int Bits> CF<Bits> czero(word status=0){return {zero<Bits/32>(status),zero<Bits/32>(status)};}
template<int Bits> std::string text(const F<Bits>& x){std::string s="sign="+std::to_string(x.sign)+" exp="+std::to_string(x.exponent)+" status="+std::to_string(x.status)+" limbs=";
    for(int i=Bits/32-1;i>=0;--i){char b[9];std::snprintf(b,9,"%08x",x.limb[i]);s+=b;}return s;}
template<class Job> void parallel_rows(std::size_t n,Job job){if(n<2){if(n)job(std::size_t(0));return;}std::atomic<std::size_t> next{0};std::vector<std::thread> w;
    for(unsigned t=0;t<std::min<std::size_t>(n,std::max(1u,std::thread::hardware_concurrency()));++t)w.emplace_back([&]{for(std::size_t i;(i=next.fetch_add(1))<n;)job(i);});for(auto& x:w)x.join();}
// Page-aligned storage (GPU in place) or offset by one element (copied by the library).
template<class E> struct Mat {
    void* raw=nullptr;E* p=nullptr;std::size_t size=0;
    Mat(std::size_t count,bool aligned,const E& fill):size(count){std::size_t page=std::size_t(getpagesize());if(posix_memalign(&raw,page,(count+1)*sizeof(E)+page))throw std::bad_alloc();
        p=static_cast<E*>(raw)+(aligned?0:1);for(std::size_t i=0;i<count;++i)p[i]=fill;}
    Mat(const std::vector<E>& v,bool aligned):Mat(v.size(),aligned,E{}){std::copy(v.begin(),v.end(),p);}
    ~Mat(){std::free(raw);} Mat(const Mat&)=delete;
};
// ---- Reference: complex operations from MPFR sums of exact component products ----
template<int Bits> F<Bits> mp(Operation op,const F<Bits>& a,const F<Bits>& b){return reference::real<Bits>(op,a,b);}
template<int Bits> F<Bits> times_power(const F<Bits>& x,long e){if(x.status||!x.sign)return x;reference::MP y(Bits);to_mpfr<Bits>(y.x,x);mpfr_mul_2si(y.x,y.x,e,MPFR_RNDN);return from_mpfr<Bits>(y.x);}
// Real sum of products a[k] b[k] (optionally c - sum): one MPFR rounding of the exact value.
template<int Bits> struct Terms {std::vector<F<Bits>> a,b;void add(const F<Bits>& x,const F<Bits>& y,int sign=1){a.push_back(sign>0?x:negate(x));b.push_back(y);}
    F<Bits> sum()const{return reference::dot<Bits>(a.data(),1,b.data(),1,a.size());}
    F<Bits> sub_from(const F<Bits>& c)const{return reference::dot_sub<Bits>(c,a.data(),1,b.data(),1,a.size());}};
// sum_k conj(u_k) w_k (Hermitian) over K entries with strides su, sw.
template<int Bits> CF<Bits> hdot(const CF<Bits>* u,std::ptrdiff_t su,const CF<Bits>* w,std::ptrdiff_t sw,std::size_t K){Terms<Bits> re,im;
    for(std::size_t k=0;k<K;++k){const CF<Bits> &x=u[std::ptrdiff_t(k)*su],&y=w[std::ptrdiff_t(k)*sw];re.add(x.re,y.re);re.add(x.im,y.im);im.add(x.re,y.im);im.add(x.im,y.re,-1);}
    return {re.sum(),im.sum()};}
// sum_k u_k w_k (no conjugate).
template<int Bits> CF<Bits> pdot(const CF<Bits>* u,std::ptrdiff_t su,const CF<Bits>* w,std::ptrdiff_t sw,std::size_t K){Terms<Bits> re,im;
    for(std::size_t k=0;k<K;++k){const CF<Bits> &x=u[std::ptrdiff_t(k)*su],&y=w[std::ptrdiff_t(k)*sw];re.add(x.re,y.re);re.add(x.im,y.im,-1);im.add(x.re,y.im);im.add(x.im,y.re);}
    return {re.sum(),im.sum()};}
// c - sum_k u_k w_k, one rounding per component; extra terms (x, y) join the real part only (the solve's zero coupling term).
template<int Bits> CF<Bits> dsub(const CF<Bits>& c,const CF<Bits>* u,std::ptrdiff_t su,const CF<Bits>* w,std::ptrdiff_t sw,std::size_t K,const F<Bits>* ex=nullptr,const F<Bits>* ey=nullptr){
    Terms<Bits> re,im;for(std::size_t k=0;k<K;++k){const CF<Bits> &x=u[std::ptrdiff_t(k)*su],&y=w[std::ptrdiff_t(k)*sw];re.add(x.re,y.re);re.add(x.im,y.im,-1);im.add(x.re,y.im);im.add(x.im,y.re);}
    if(ex)re.add(*ex,*ey);return {re.sub_from(c.re),im.sub_from(c.im)};}
template<int Bits> struct RefQR {std::size_t m=0,n=0,nb=0,rank=0;int reason=0;word status=0;std::vector<CF<Bits>> R,V,T,tau;std::vector<std::size_t> perm;std::size_t recomputed=0;};
template<int Bits> RefQR<Bits> ref_qr(const std::vector<CF<Bits>>& A,std::size_t m,std::size_t n,std::size_t nb,int rank_bits){
    constexpr int N=Bits/32;RefQR<Bits> q;q.m=m;q.n=n;q.nb=nb=nb&&nb<n?nb:n;q.rank=n;std::size_t blocks=nb?(n+nb-1)/nb:0;
    std::vector<CF<Bits>> a=A;q.V.assign(m*n,czero<Bits>());q.T.assign(blocks*nb*nb,czero<Bits>());q.tau.assign(n,czero<Bits>());long top=LONG_MIN;CF<Bits>* V=q.V.data();
    for(std::size_t j=0;j<n;++j){
        // Reflectors [0, min(j, rank)) in groups of whole blocks (the last one possibly partial): w = V^H a, y = T^H w, a -= V y.
        const std::size_t qn=std::min(j,q.rank);
        for(std::size_t b0=0;b0<qn;b0+=nb){const std::size_t kb=std::min(b0+nb,qn)-b0;const CF<Bits>* Tb=q.T.data()+(b0/nb)*nb*nb;std::vector<CF<Bits>> w(kb),y(kb);
            parallel_rows(kb,[&](std::size_t i){w[i]=hdot<Bits>(V+(b0+i)*n+b0+i,std::ptrdiff_t(n),a.data()+(b0+i)*n+j,std::ptrdiff_t(n),m-b0-i);});
            for(std::size_t i=0;i<kb;++i)y[i]=hdot<Bits>(Tb+i,std::ptrdiff_t(nb),w.data(),1,i+1);
            parallel_rows(m-b0,[&](std::size_t r){CF<Bits>& x=a[(b0+r)*n+j];x=dsub<Bits>(x,V+(b0+r)*n+b0,1,y.data(),1,std::min(kb,r+1));});}
        if(q.rank<n)continue;
        const std::size_t len=m-j;const CF<Bits> alpha=a[j*n+j];
        Terms<Bits> ss;bool below=false;for(std::size_t r=j;r<m;++r){ss.add(a[r*n+j].re,a[r*n+j].re);ss.add(a[r*n+j].im,a[r*n+j].im);if(r>j)below|=a[r*n+j].re.sign||a[r*n+j].im.sign;}
        const F<Bits> s=ss.sum();const bool needs=below||alpha.im.sign;
        int reason=s.status?QRInfo::status_column:(!alpha.re.sign&&!alpha.im.sign&&!below)?QRInfo::zero_column:0;F<Bits> beta=alpha.re,v0=power<Bits>(0);
        if(!reason&&needs){beta=mp<Bits>(Operation::sqrt,s,s);if(alpha.re.sign>=0)beta=negate(beta);v0=mp<Bits>(Operation::sub,alpha.re,beta);}
        if(!reason&&rank_bits>0&&top!=LONG_MIN&&long(beta.exponent)+rank_bits<top)reason=QRInfo::small_column;
        if(reason){q.rank=j;q.reason=reason;q.status=s.status;continue;}
        CF<Bits> t=czero<Bits>();
        if(needs){const long e=v0.exponent;V[j*n+j]={times_power<Bits>(v0,-e),times_power<Bits>(alpha.im,-e)};
            for(std::size_t r=j+1;r<m;++r)V[r*n+j]={times_power<Bits>(a[r*n+j].re,-e),times_power<Bits>(a[r*n+j].im,-e)};
            Terms<Bits> vv;for(std::size_t r=j;r<m;++r){vv.add(V[r*n+j].re,V[r*n+j].re);vv.add(V[r*n+j].im,V[r*n+j].im);}const F<Bits> sv=vv.sum(),vi=V[j*n+j].im;
            if(!vi.sign&&!vi.status)t={mp<Bits>(Operation::div,power<Bits>(1),sv),zero<N>()};
            else{const F<Bits> h=times_power<Bits>(sv,-1),bb=mp<Bits>(Operation::mul,times_power<Bits>(beta,-e),vi),d=reference::dot2_add<Bits>(h,h,bb,bb,zero<N>());
                t={mp<Bits>(Operation::div,h,d),negate(mp<Bits>(Operation::div,bb,d))};}}
        else V[j*n+j]={power<Bits>(0),zero<N>()};
        a[j*n+j]={beta,zero<N>()};q.tau[j]=t;top=std::max(top,long(beta.exponent));
        // T column: Gram entries over the block's rows, then T_lc = -RN(tau * P(T_b[l][l..c) G[l..c))) per component.
        const std::size_t b0=j/nb*nb,c=j-b0;CF<Bits>* Tb=q.T.data()+(b0/nb)*nb*nb;std::vector<CF<Bits>> G(c);
        parallel_rows(c,[&](std::size_t i){G[i]=hdot<Bits>(V+b0*n+b0+i,std::ptrdiff_t(n),V+b0*n+j,std::ptrdiff_t(n),m-b0);});
        Tb[c*nb+c]=t;for(std::size_t l=0;l<c;++l){CF<Bits> z=reference::complex_fused<Bits>(t,pdot<Bits>(Tb+l*nb+l,1,G.data()+l,1,c-l),czero<Bits>());Tb[l*nb+c]={negate(z.re),negate(z.im)};}
    }
    q.R.assign(n*n,czero<Bits>());for(std::size_t i=0;i<n;++i)for(std::size_t j=std::min(i,q.rank);j<n;++j)q.R[i*n+j]=a[i*n+j];
    return q;
}
// C (m x nrhs) <- Q^H C or Q C by the blocks of reflectors [0, rank).
template<int Bits> void ref_apply(const RefQR<Bits>& q,std::vector<CF<Bits>>& C,std::size_t nrhs,bool adjoint){
    const std::size_t m=q.m,n=q.n,nb=q.nb,blocks=nb?(q.rank+nb-1)/nb:0;const CF<Bits>* V=q.V.data();
    for(std::size_t s=0;s<blocks;++s){const std::size_t b=adjoint?s:blocks-1-s,b0=b*nb,kb=std::min(b0+nb,q.rank)-b0;const CF<Bits>* Tb=q.T.data()+b*nb*nb;
        parallel_rows(nrhs,[&](std::size_t c){std::vector<CF<Bits>> w(kb),y(kb);
            for(std::size_t i=0;i<kb;++i)w[i]=hdot<Bits>(V+(b0+i)*n+b0+i,std::ptrdiff_t(n),C.data()+(b0+i)*nrhs+c,std::ptrdiff_t(nrhs),m-b0-i);
            for(std::size_t i=0;i<kb;++i)y[i]=adjoint?hdot<Bits>(Tb+i,std::ptrdiff_t(nb),w.data(),1,i+1):pdot<Bits>(Tb+i*nb+i,1,w.data()+i,1,kb-i);
            for(std::size_t r=b0;r<m;++r){CF<Bits>& x=C[r*nrhs+c];x=dsub<Bits>(x,V+r*n+b0,1,y.data(),1,std::min(kb,r-b0+1));}});}
}
// Backward substitution in blocks of nb: one rounding per component per later block, then within the block, then
// x_i = (RN(u_re / r_ii), RN(u_im / r_ii)) (real r_ii). The real part's update also carries the zero coupling term
// (-Im r_ii) x_i.im of the embedded solve (it can only add x_i.im's status).
template<int Bits> std::vector<CF<Bits>> ref_solve(const RefQR<Bits>& q,std::vector<CF<Bits>> C,std::size_t nrhs){
    const std::size_t n=q.n,nb=q.nb;std::vector<CF<Bits>> X(n*nrhs,czero<Bits>(invalid));if(q.rank<n)return X;
    ref_apply<Bits>(q,C,nrhs,true);std::copy(C.begin(),C.begin()+n*nrhs,X.begin());const CF<Bits>* R=q.R.data();
    for(std::size_t i=n;i-->0;)parallel_rows(nrhs,[&](std::size_t c){CF<Bits> v=X[i*nrhs+c];std::size_t r0=i/nb*nb,r1=std::min(n,r0+nb);
        for(std::size_t b=(n-1)/nb*nb;b>=r1&&b>r0;b-=nb)v=dsub<Bits>(v,R+i*n+b,1,X.data()+b*nrhs+c,std::ptrdiff_t(nrhs),std::min(n,b+nb)-b);
        // Imaginary part first (as the embedded rows 2i+1, 2i), then the real part with the coupling term on the final x_i.im.
        const F<Bits> xim=mp<Bits>(Operation::div,dsub<Bits>(v,R+i*n+i+1,1,X.data()+(i+1)*nrhs+c,std::ptrdiff_t(nrhs),r1-i-1).im,R[i*n+i].re),coupling=negate(R[i*n+i].im);
        const F<Bits> u=dsub<Bits>(v,R+i*n+i+1,1,X.data()+(i+1)*nrhs+c,std::ptrdiff_t(nrhs),r1-i-1,&coupling,&xim).re;
        X[i*nrhs+c]={mp<Bits>(Operation::div,u,R[i*n+i].re),xim};});
    return X;
}
// sum_k |x_k|^2 over K complex entries (stride sx): one MPFR rounding of the exact real value.
template<int Bits> F<Bits> norm2(const CF<Bits>* x,std::ptrdiff_t sx,std::size_t K){Terms<Bits> t;for(std::size_t k=0;k<K;++k){const CF<Bits>& z=x[std::ptrdiff_t(k)*sx];t.add(z.re,z.re);t.add(z.im,z.im);}return t.sum();}
// Column-pivoted complex QR (QROptions::pivot): a right-looking MPFR program of the pivoted sequence (docs/numerics.md, "Complex column
// pivoting"). nu_k = nuref_k = sum |a_rk|^2 (one rounding); before step j the remaining column with the largest nu (status first; exact
// comparison by mpfr_cmp; ties to the lowest original index) is swapped into place; column j is reduced by the block's reflectors so far
// (its y from the steps before) and gives reflector j as in ref_qr; then every trailing column k gets w_k = P(v_j^H a_k) (rows [j, m),
// a_k as at the block start), y_k = P(T[.][j]^H w_k), its row-j entry r_jk = D(a_jk; V[j][block], y_k) and nu_k = RN(nu_k - |r_jk|^2),
// recomputed from the reduced column (rows (j, m)) when it falls to <= 0 or more than bits/2 binades below its last exact value.
// After the block, X = D(X; V_b, Y) with these W and Y; after a failing column p, block b's reflectors [b0, p) with fresh W, Y.
template<int Bits> RefQR<Bits> ref_qrp(const std::vector<CF<Bits>>& A,std::size_t m,std::size_t n,std::size_t nb,int rank_bits){
    constexpr int N=Bits/32;RefQR<Bits> q;q.m=m;q.n=n;q.nb=nb=nb&&nb<n?nb:n;q.rank=n;std::size_t blocks=nb?(n+nb-1)/nb:0;
    std::vector<CF<Bits>> a=A;q.V.assign(m*n,czero<Bits>());q.T.assign(blocks*nb*nb,czero<Bits>());q.tau.assign(n,czero<Bits>());long top=LONG_MIN;CF<Bits>* V=q.V.data();
    q.perm.resize(n);for(std::size_t k=0;k<n;++k)q.perm[k]=k;
    std::vector<F<Bits>> nu(n),nuref(n);std::vector<CF<Bits>> Wr(nb*n,czero<Bits>()),Yr(nb*n,czero<Bits>());
    parallel_rows(n,[&](std::size_t k){nu[k]=nuref[k]=norm2<Bits>(a.data()+k,std::ptrdiff_t(n),m);});
    auto greater=[](const F<Bits>& x,const F<Bits>& y){ // exact order, a status above every value
        if(x.status||y.status)return x.status&&!y.status;reference::MP u(Bits),v(Bits);to_mpfr<Bits>(u.x,x);to_mpfr<Bits>(v.x,y);return mpfr_cmp(u.x,v.x)>0;};
    // Applies reflectors [b0, b0+g) of block b0/nb to columns [c0, n) with Wr/Yr rows [0, g) (fresh = form them first).
    auto apply=[&](std::size_t b0,std::size_t g,std::size_t c0,bool fresh){const CF<Bits>* Tb=q.T.data()+(b0/nb)*nb*nb;
        if(fresh)parallel_rows(n-c0,[&](std::size_t u){const std::size_t k=c0+u;
            for(std::size_t i=0;i<g;++i)Wr[i*n+k]=hdot<Bits>(V+(b0+i)*n+b0+i,std::ptrdiff_t(n),a.data()+(b0+i)*n+k,std::ptrdiff_t(n),m-b0-i);
            for(std::size_t i=0;i<g;++i)Yr[i*n+k]=hdot<Bits>(Tb+i,std::ptrdiff_t(nb),Wr.data()+k,std::ptrdiff_t(n),i+1);});
        parallel_rows(m-b0,[&](std::size_t r){for(std::size_t k=c0;k<n;++k){CF<Bits>& x=a[(b0+r)*n+k];x=dsub<Bits>(x,V+(b0+r)*n+b0,1,Yr.data()+k,std::ptrdiff_t(n),std::min(g,r+1));}});};
    for(std::size_t b0=0;b0<n&&q.rank==n;b0+=nb){const std::size_t kb=std::min(n,b0+nb)-b0;CF<Bits>* Tb=q.T.data()+(b0/nb)*nb*nb;
        for(std::size_t c=0;c<kb;++c){const std::size_t j=b0+c;
            std::size_t p=j;for(std::size_t k=j+1;k<n;++k)if(greater(nu[k],nu[p])||(!greater(nu[p],nu[k])&&q.perm[k]<q.perm[p]))p=k;
            if(p!=j){for(std::size_t r=0;r<m;++r)std::swap(a[r*n+j],a[r*n+p]);for(std::size_t i=0;i<c;++i){std::swap(Wr[i*n+j],Wr[i*n+p]);std::swap(Yr[i*n+j],Yr[i*n+p]);}
                std::swap(nu[j],nu[p]);std::swap(nuref[j],nuref[p]);std::swap(q.perm[j],q.perm[p]);}
            std::vector<CF<Bits>> x(m-b0);for(std::size_t r=0;r<m-b0;++r)x[r]=c?dsub<Bits>(a[(b0+r)*n+j],V+(b0+r)*n+b0,1,Yr.data()+j,std::ptrdiff_t(n),std::min(c,r+1)):a[(b0+r)*n+j];
            const std::size_t len=m-j;const CF<Bits> alpha=x[c];const F<Bits> s=norm2<Bits>(x.data()+c,1,len);
            bool below=false;for(std::size_t r=c+1;r<m-b0;++r)below|=x[r].re.sign||x[r].im.sign;const bool needs=below||alpha.im.sign;
            int reason=s.status?QRInfo::status_column:(!alpha.re.sign&&!alpha.im.sign&&!below)?QRInfo::zero_column:0;F<Bits> beta=alpha.re,v0=power<Bits>(0);
            if(!reason&&needs){beta=mp<Bits>(Operation::sqrt,s,s);if(alpha.re.sign>=0)beta=negate(beta);v0=mp<Bits>(Operation::sub,alpha.re,beta);}
            if(!reason&&rank_bits>0&&top!=LONG_MIN&&long(beta.exponent)+rank_bits<top)reason=QRInfo::small_column;
            if(reason){q.rank=j;q.reason=reason;q.status=s.status;if(c)apply(b0,c,j,true);break;}
            for(std::size_t r=0;r<m-b0;++r)a[(b0+r)*n+j]=x[r];
            CF<Bits> t=czero<Bits>();
            if(needs){const long e=v0.exponent;V[j*n+j]={times_power<Bits>(v0,-e),times_power<Bits>(alpha.im,-e)};
                for(std::size_t r=j+1;r<m;++r)V[r*n+j]={times_power<Bits>(a[r*n+j].re,-e),times_power<Bits>(a[r*n+j].im,-e)};
                const F<Bits> sv=norm2<Bits>(V+j*n+j,std::ptrdiff_t(n),len),vi=V[j*n+j].im;
                if(!vi.sign&&!vi.status)t={mp<Bits>(Operation::div,power<Bits>(1),sv),zero<N>()};
                else{const F<Bits> h=times_power<Bits>(sv,-1),bb=mp<Bits>(Operation::mul,times_power<Bits>(beta,-e),vi),d=reference::dot2_add<Bits>(h,h,bb,bb,zero<N>());
                    t={mp<Bits>(Operation::div,h,d),negate(mp<Bits>(Operation::div,bb,d))};}}
            else V[j*n+j]={power<Bits>(0),zero<N>()};
            a[j*n+j]={beta,zero<N>()};q.tau[j]=t;top=std::max(top,long(beta.exponent));
            std::vector<CF<Bits>> G(c);parallel_rows(c,[&](std::size_t i){G[i]=hdot<Bits>(V+b0*n+b0+i,std::ptrdiff_t(n),V+b0*n+j,std::ptrdiff_t(n),m-b0);});
            Tb[c*nb+c]=t;for(std::size_t l=0;l<c;++l){CF<Bits> z=reference::complex_fused<Bits>(t,pdot<Bits>(Tb+l*nb+l,1,G.data()+l,1,c-l),czero<Bits>());Tb[l*nb+c]={negate(z.re),negate(z.im)};}
            // Trailing columns: W and Y rows c, row j of R, downdated norms (recomputed when they collapse).
            std::vector<char> redo(n,0);
            parallel_rows(n-j-1,[&](std::size_t u){const std::size_t k=j+1+u;
                Wr[c*n+k]=hdot<Bits>(V+j*n+j,std::ptrdiff_t(n),a.data()+j*n+k,std::ptrdiff_t(n),len);
                Yr[c*n+k]=hdot<Bits>(Tb+c,std::ptrdiff_t(nb),Wr.data()+k,std::ptrdiff_t(n),c+1);
                const CF<Bits> r=dsub<Bits>(a[j*n+k],V+j*n+b0,1,Yr.data()+k,std::ptrdiff_t(n),c+1);
                const F<Bits> rr[2]={r.re,r.im};nu[k]=reference::dot_sub<Bits>(nu[k],rr,1,rr,1,2);
                redo[k]=nuref[k].sign&&!nuref[k].status&&!nu[k].status&&(nu[k].sign<=0||long(nu[k].exponent)+Bits/2<long(nuref[k].exponent));
                if(redo[k]){std::vector<CF<Bits>> z;for(std::size_t i=j+1;i<m;++i)z.push_back(dsub<Bits>(a[i*n+k],V+i*n+b0,1,Yr.data()+k,std::ptrdiff_t(n),c+1));
                    nu[k]=nuref[k]=norm2<Bits>(z.data(),1,z.size());}});
            for(char x:redo)q.recomputed+=x!=0;
        }
        if(q.rank==n&&b0+kb<n)apply(b0,kb,b0+kb,false);
    }
    q.R.assign(n*n,czero<Bits>());for(std::size_t i=0;i<n;++i)for(std::size_t j=std::min(i,q.rank);j<n;++j)q.R[i*n+j]=a[i*n+j];
    return q;
}
// Pivoted solve: z = R[0:p,0:p]^-1 (Q^H b)[0:p) by the backward substitution of ref_solve (blocks min(nb, p)), x[perm[i]] = z_i, 0 beyond p.
template<int Bits> std::vector<CF<Bits>> ref_solve_pivoted(const RefQR<Bits>& q,std::vector<CF<Bits>> C,std::size_t nrhs){
    const std::size_t n=q.n,p=q.rank,nb=std::min(q.nb,p);std::vector<CF<Bits>> X(n*nrhs,czero<Bits>());
    if(p<n&&q.reason==QRInfo::status_column){std::fill(X.begin(),X.end(),czero<Bits>(invalid|q.status));return X;}
    ref_apply<Bits>(q,C,nrhs,true);std::vector<CF<Bits>> Z(C.begin(),C.begin()+p*nrhs);const CF<Bits>* R=q.R.data();
    for(std::size_t i=p;i-->0;)parallel_rows(nrhs,[&](std::size_t c){CF<Bits> v=Z[i*nrhs+c];std::size_t r0=i/nb*nb,r1=std::min(p,r0+nb);
        for(std::size_t b=(p-1)/nb*nb;b>=r1&&b>r0;b-=nb)v=dsub<Bits>(v,R+i*n+b,1,Z.data()+b*nrhs+c,std::ptrdiff_t(nrhs),std::min(p,b+nb)-b);
        const F<Bits> xim=mp<Bits>(Operation::div,dsub<Bits>(v,R+i*n+i+1,1,Z.data()+(i+1)*nrhs+c,std::ptrdiff_t(nrhs),r1-i-1).im,R[i*n+i].re),coupling=negate(R[i*n+i].im);
        const F<Bits> u=dsub<Bits>(v,R+i*n+i+1,1,Z.data()+(i+1)*nrhs+c,std::ptrdiff_t(nrhs),r1-i-1,&coupling,&xim).re;
        Z[i*nrhs+c]={mp<Bits>(Operation::div,u,R[i*n+i].re),xim};});
    for(std::size_t i=0;i<p;++i)for(std::size_t c=0;c<nrhs;++c)X[q.perm[i]*nrhs+c]=Z[i*nrhs+c];
    return X;
}
// ---- Inputs ----
enum Kind {moderate,qsc,wide,spread,graded,dependent,triangular,extreme,realonly};
const char* kind_name(Kind k){static const char* s[]={"moderate","qsc","wide","spread","graded","dependent","upper triangular","extreme spread","real"};return s[k];}
template<int Bits> std::vector<CF<Bits>> matrix(std::size_t m,std::size_t n,Kind kind,std::mt19937_64& rng){
    constexpr int N=Bits/32;std::vector<CF<Bits>> A(m*n,czero<Bits>());
    for(std::size_t c=0;c<n;++c){long scale=kind==qsc?long(rng()%121)-60:kind==wide?long(rng()%601)-300:kind==graded?long(c%2?20:-20):0;
        for(std::size_t r=0;r<m;++r){if(kind==triangular&&r>c)continue;const int span=kind==qsc?15:kind==spread?400:kind==extreme?3000:4;
            CF<Bits> z{reference::random_number<Bits>(rng,span),reference::random_number<Bits>(rng,span)};z.re.exponent+=int(scale);z.im.exponent+=int(scale);
            const unsigned u=unsigned(rng()%16);if(kind==realonly||u==0)z.im=zero<N>();else if(u==1)z.re=zero<N>();  // some purely real / imaginary entries
            if(kind==qsc&&rng()%4==0)z=czero<Bits>();A[r*n+c]=z;}
        if(kind==qsc&&c%17==5){CF<Bits>& z=A[(rng()%m)*n+c];z.re.exponent=int(scale-150);z.im.exponent=int(scale-170);}}
    if(kind==dependent&&n>=3)for(std::size_t r=0;r<m;++r){A[r*n+n-1]=A[r*n+1];const CF<Bits> z=A[r*n];A[r*n+n/2]={negate(z.im),z.re};} // column n/2 = i * column 0
    return A;
}
template<int Bits> std::vector<CF<Bits>> rhs(std::size_t m,std::size_t nrhs,std::mt19937_64& rng){std::vector<CF<Bits>> B(m*nrhs);
    for(auto& z:B){z={reference::random_number<Bits>(rng,8),reference::random_number<Bits>(rng,8)};if(rng()%11==0)z=czero<Bits>();else if(rng()%9==0)z.im=zero<Bits/32>();}return B;}
// ---- Checks ----
template<int Bits> void same(const CF<Bits>* got,const std::vector<CF<Bits>>& want,std::size_t cols,const std::string& label){
    for(std::size_t q=0;q<want.size();++q)for(int part=0;part<2;++part){const F<Bits> &g=part?got[q].im:got[q].re,&w=part?want[q].im:want[q].re;
        require(reference::equal<Bits>(g,w),label+": entry ("+std::to_string(q/cols)+","+std::to_string(q%cols)+")."+(part?"im":"re")+"\n got  "+text<Bits>(g)+"\n want "+text<Bits>(w));}}
struct Run {const char* name;QROptions options;};
std::vector<Run> runs(std::size_t n){QROptions def,gpu16,host16,single,gpu1,gpu7;gpu16.block=16;gpu16.host_macs=0;gpu16.solve_host_macs=0;host16.block=16;host16.gpu=false;
    single.block=0;single.solve_host_macs=0;gpu1.block=1;gpu1.host_macs=0;gpu1.solve_host_macs=0;gpu7=gpu16;gpu7.block=7;
    std::vector<Run> r={{"default",def},{"gpu block 16",gpu16},{"host block 16",host16}};if(n<=130)r.push_back({"single block",single});
    if(n<=40)r.push_back({"gpu block 1",gpu1});if(n<=140)r.push_back({"gpu block 7",gpu7});return r;}
std::size_t factors_run=0,solves_run=0,applies_run=0,deficient=0,gpu_updates=0,host_updates=0,real_matches=0;
template<int Bits> void factor(Linalg& la,const std::vector<CF<Bits>>& A,std::size_t m,std::size_t n,const std::string& label,std::mt19937_64& rng,
                               std::vector<Run> rs={},int rank_bits=0,std::size_t expect_rank=SIZE_MAX){
    if(rs.empty())rs=runs(n);std::map<std::size_t,RefQR<Bits>> refs;
    for(std::size_t k=0;k<rs.size();++k){Run run=rs[k];run.options.rank_bits=rank_bits;const std::size_t nb=run.options.block&&run.options.block<n?run.options.block:n;
        auto found=refs.find(nb);if(found==refs.end())found=refs.emplace(nb,ref_qr<Bits>(A,m,n,nb,rank_bits)).first;const RefQR<Bits>& want=found->second;
        std::string l="complex qr bits="+std::to_string(Bits)+" "+std::to_string(m)+"x"+std::to_string(n)+" "+label+" ["+run.name+"]";
        if(expect_rank!=SIZE_MAX)require(want.rank==expect_rank,l+": reference rank "+std::to_string(want.rank)+", fixture expects "+std::to_string(expect_rank));
        bool aligned=k%2==0;Mat<CF<Bits>> a(A,aligned);ComplexQRFactor qr=la.factor_qr_complex(Bits,a.p,m,n,run.options);const QRInfo& info=qr.info();
        require(info.rank==want.rank&&info.reason==want.reason&&info.status==want.status,l+": rank "+std::to_string(info.rank)+" reason "+std::to_string(info.reason)+
            " status "+std::to_string(info.status)+", want "+std::to_string(want.rank)+" "+std::to_string(want.reason)+" "+std::to_string(want.status));
        require(qr.rows()==m&&qr.cols()==n&&qr.block()==nb&&qr.bits()==Bits&&qr.full_rank()==(want.rank==n),l+": shape");
        same<Bits>(static_cast<const CF<Bits>*>(qr.r()),want.R,n,l+" R");same<Bits>(static_cast<const CF<Bits>*>(qr.v()),want.V,n,l+" V");
        same<Bits>(static_cast<const CF<Bits>*>(qr.tau()),want.tau,1,l+" tau");same<Bits>(static_cast<const CF<Bits>*>(qr.t()),want.T,nb?nb:1,l+" T");
        ++factors_run;deficient+=want.rank<n;gpu_updates+=info.gpu_updates;host_updates+=info.host_updates;
        // Least squares (multiple right-hand sides; a status in B on one run) and Q / Q^H applied in place.
        std::size_t nrhs=k%3==0?1:k%3==1?3:5;auto B=rhs<Bits>(m,nrhs,rng);if(k==1&&m>3)B[nrhs+1].im.status=invalid;
        {Mat<CF<Bits>> b(B,!aligned),x(n*nrhs,aligned,czero<Bits>());qr.solve(b.p,nrhs,x.p);same<Bits>(x.p,ref_solve<Bits>(want,B,nrhs),nrhs,"solve nrhs="+std::to_string(nrhs)+" "+l);++solves_run;}
        for(bool adjoint:{true,false}){auto C=B;ref_apply<Bits>(want,C,nrhs,adjoint);Mat<CF<Bits>> b(B,aligned!=adjoint);qr.apply_q(b.p,nrhs,adjoint);
            same<Bits>(b.p,C,nrhs,std::string(adjoint?"apply Q^H":"apply Q")+" nrhs="+std::to_string(nrhs)+" "+l);++applies_run;}
    }
}
// A real matrix (zero imaginary parts): the complex sequence is the real QR's sequence (real parts bit-identical to
// Linalg::factor_qr, imaginary parts zero), including solve and Q application with a real right-hand side.
template<int Bits> void real_equivalence(Linalg& la,std::size_t m,std::size_t n,Kind kind,std::mt19937_64& rng,QROptions o){
    constexpr int N=Bits/32;auto A=matrix<Bits>(m,n,kind,rng);std::vector<F<Bits>> Ar(m*n);for(std::size_t q=0;q<m*n;++q){A[q].im=zero<N>();Ar[q]=A[q].re;}
    const std::size_t nb=o.block&&o.block<n?o.block:n,blocks=nb?(n+nb-1)/nb:0,nrhs=3;std::string l="real equivalence bits="+std::to_string(Bits)+" "+std::to_string(m)+"x"+std::to_string(n)+" block "+std::to_string(nb);
    QRFactor r=la.factor_qr(Bits,Ar.data(),m,n,o);ComplexQRFactor c=la.factor_qr_complex(Bits,A.data(),m,n,o);require(r.info().rank==c.info().rank,l+": rank");
    auto check=[&](const void* real,const void* cx,std::size_t count,const char* what){auto x=static_cast<const F<Bits>*>(real);auto z=static_cast<const CF<Bits>*>(cx);
        for(std::size_t q=0;q<count;++q)require(reference::equal<Bits>(x[q],z[q].re)&&reference::equal<Bits>(z[q].im,zero<N>()),l+": "+what+" entry "+std::to_string(q)+"\n real "+text<Bits>(x[q])+"\n cx   "+text<Bits>(z[q].re)+" / "+text<Bits>(z[q].im));};
    check(r.r(),c.r(),n*n,"R");check(r.v(),c.v(),m*n,"V");check(r.tau(),c.tau(),n,"tau");check(r.t(),c.t(),blocks*nb*nb,"T");
    std::vector<F<Bits>> Br(m*nrhs),Xr(n*nrhs);std::vector<CF<Bits>> Bc(m*nrhs),Xc(n*nrhs);for(std::size_t q=0;q<m*nrhs;++q){Br[q]=reference::random_number<Bits>(rng,8);Bc[q]={Br[q],zero<N>()};}
    r.solve(Br.data(),nrhs,Xr.data());c.solve(Bc.data(),nrhs,Xc.data());check(Xr.data(),Xc.data(),n*nrhs,"solve");
    r.apply_q(Br.data(),nrhs,false);c.apply_q(Bc.data(),nrhs,false);check(Br.data(),Bc.data(),m*nrhs,"apply Q");++real_matches;
}
template<int Bits> void special(Linalg& la,std::mt19937_64& rng){
    constexpr int N=Bits/32;
    // 1 x 1: purely imaginary (a reflector makes r real), real negative (tau = 0), complex.
    for(CF<Bits> z:{CF<Bits>{zero<N>(),power<Bits>(3)},CF<Bits>{power<Bits>(-2,-1),zero<N>()},CF<Bits>{power<Bits>(1),power<Bits>(5,-1)}})factor<Bits>(la,{z},1,1,"1x1",rng);
    // Zero column 3 (zero_column at 3), all-zero matrix (rank 0), zero first column.
    {auto A=matrix<Bits>(20,9,moderate,rng);for(std::size_t r=0;r<20;++r)A[r*9+3]=czero<Bits>();factor<Bits>(la,A,20,9,"zero column 3",rng,{},0,3);}
    factor<Bits>(la,std::vector<CF<Bits>>(12,czero<Bits>()),4,3,"zero matrix",rng,{},0,0);
    {auto A=matrix<Bits>(40,20,qsc,rng);for(std::size_t r=0;r<40;++r)A[r*20]=czero<Bits>();factor<Bits>(la,A,40,20,"zero column 0",rng,{},0,0);}
    // Upper triangular input: nothing below the diagonal; complex diagonal entries still need a reflector (real r_jj), real ones not.
    factor<Bits>(la,matrix<Bits>(12,12,triangular,rng),12,12,"upper triangular square",rng,{},0,12);
    {auto A=matrix<Bits>(30,18,triangular,rng);for(std::size_t j=0;j<18;j+=2){A[j*18+j].im=zero<N>();if(!A[j*18+j].re.sign)A[j*18+j].re=power<Bits>(1);}factor<Bits>(la,A,30,18,"upper triangular tall, half real diagonal",rng,{},0,18);}
    // Statuses: in column 4 (real part), in column 0 (imaginary part), and past a zero column (the zero column fails first).
    {auto A=matrix<Bits>(25,10,moderate,rng);A[7*10+4].re.status=invalid;factor<Bits>(la,A,25,10,"status at (7,4).re",rng,{},0,4);
     A=matrix<Bits>(25,10,moderate,rng);A[20*10+0].im.status=exponent_overflow;factor<Bits>(la,A,25,10,"status at (20,0).im",rng,{},0,0);
     A=matrix<Bits>(25,10,moderate,rng);A[3*10+8].im.status=invalid;for(std::size_t r=0;r<25;++r)A[r*10+6]=czero<Bits>();factor<Bits>(la,A,25,10,"zero column 6, status in 8",rng,{},0,6);
     A=matrix<Bits>(40,30,moderate,rng);A[5*30+20].re.status=invalid;for(std::size_t r=0;r<40;++r)A[r*30]=czero<Bits>();factor<Bits>(la,A,40,30,"zero column 0, status in 20",rng,{},0,0);}
    // Exactly dependent columns (column n-1 = column 1, column n/2 = i column 0): full rank at rank_bits = 0, found with rank_bits = bits/2.
    {auto A=matrix<Bits>(41,7,dependent,rng);factor<Bits>(la,A,41,7,"dependent, no rank test",rng);factor<Bits>(la,A,41,7,"dependent, rank_bits",rng,{},Bits/2,3);
     auto B=matrix<Bits>(90,40,dependent,rng);factor<Bits>(la,B,90,40,"dependent 90x40, rank_bits",rng,{},Bits/2,20);}
    // Exponent spreads beyond the exact windows (cluster path of exact_dot; several GPU bands or the host fallback).
    factor<Bits>(la,matrix<Bits>(300,24,extreme,rng),300,24,"extreme spread",rng);
    factor<Bits>(la,matrix<Bits>(70,40,spread,rng),70,40,"spread",rng);
    // An entry 2^-999999995 next to entries near 2^10: scaling v by 2^-exponent(Re v0) leaves the range (exponent_overflow in v).
    {auto A=matrix<Bits>(150,12,moderate,rng);for(std::size_t r=0;r<150;++r){A[r*12].re.exponent+=10;A[r*12].im.exponent+=10;}A[77*12].im.exponent=-999999995;factor<Bits>(la,A,150,12,"tiny entry in column 0",rng);}
    // Empty problems, a moved factor, and the documented refusals.
    {std::vector<CF<Bits>> none;ComplexQRFactor e=la.factor_qr_complex(Bits,none.data(),0,0);require(e.full_rank()&&e.info().rank==0,"0x0");e.solve(none.data(),3,none.data());
     auto A=matrix<Bits>(5,0,moderate,rng);ComplexQRFactor z=la.factor_qr_complex(Bits,A.data(),5,0);require(z.info().rank==0,"5x0");
     A=matrix<Bits>(6,4,moderate,rng);ComplexQRFactor f=la.factor_qr_complex(Bits,A.data(),6,4);ComplexQRFactor g=std::move(f);require(f.empty()&&!g.empty(),"move");
     std::vector<CF<Bits>> X;g.solve(none.data(),0,X.data());bool threw=false;try{la.factor_qr_complex(Bits,A.data(),3,4);}catch(const std::invalid_argument&){threw=true;}require(threw,"m < n must throw");
     QROptions p;p.pivot=true;ComplexQRFactor pe=la.factor_qr_complex(Bits,none.data(),0,0,p);require(pe.pivoted()&&pe.full_rank(),"pivoted 0x0");
     ComplexQRFactor pz=la.factor_qr_complex(Bits,A.data(),6,0,p);require(pz.info().rank==0,"pivoted 6x0");}
    // Real input: the real QR's bits.
    {QROptions d,g16,h7;g16.block=16;g16.host_macs=0;g16.solve_host_macs=0;h7.block=7;h7.gpu=false;
     real_equivalence<Bits>(la,33,7,graded,rng,d);real_equivalence<Bits>(la,70,40,qsc,rng,g16);real_equivalence<Bits>(la,70,40,moderate,rng,h7);real_equivalence<Bits>(la,150,60,wide,rng,d);}
}
// Pivoted factor against ref_qrp (permutation, rank, R, V, tau, T, recomputed norms), its basic least-squares solution and Q / Q^H
// applications; full-rank status-free inputs also against ref_qr of A P (the pivoted sequence is the unpivoted one of A P).
std::size_t pivoted_run=0,pivoted_deficient=0,recomputations=0,crossed_run=0;
std::vector<Run> pivot_runs(std::size_t n){QROptions def,gpu16,host7,single,gpu1;def.pivot=gpu16.pivot=host7.pivot=single.pivot=gpu1.pivot=true;gpu16.block=16;gpu16.host_macs=0;gpu16.solve_host_macs=0;
    host7.block=7;host7.gpu=false;single.block=0;single.solve_host_macs=0;gpu1.block=1;gpu1.host_macs=0;gpu1.solve_host_macs=0;
    std::vector<Run> rs={{"default",def},{"gpu block 16",gpu16},{"host block 7",host7}};if(n<=100)rs.push_back({"single block",single});if(n<=24)rs.push_back({"gpu block 1",gpu1});return rs;}
template<int Bits> void factor_pivoted(Linalg& la,const std::vector<CF<Bits>>& A,std::size_t m,std::size_t n,const std::string& label,std::mt19937_64& rng,
                                       int rank_bits=0,std::size_t expect_rank=SIZE_MAX,bool expect_recompute=false,std::vector<Run> rs={}){
    if(rs.empty())rs=pivot_runs(n);std::map<std::size_t,RefQR<Bits>> refs;bool crossed=false;
    for(std::size_t k=0;k<rs.size();++k){Run run=rs[k];run.options.rank_bits=rank_bits;const std::size_t nb=run.options.block&&run.options.block<n?run.options.block:n;
        auto found=refs.find(nb);if(found==refs.end())found=refs.emplace(nb,ref_qrp<Bits>(A,m,n,nb,rank_bits)).first;const RefQR<Bits>& want=found->second;
        std::string l="pivoted complex qr bits="+std::to_string(Bits)+" "+std::to_string(m)+"x"+std::to_string(n)+" "+label+" ["+run.name+"]";
        if(expect_rank!=SIZE_MAX)require(want.rank==expect_rank,l+": reference rank "+std::to_string(want.rank)+", fixture expects "+std::to_string(expect_rank));
        if(expect_recompute)require(want.recomputed>0,l+": fixture expects recomputed norms");
        bool aligned=k%2==0;Mat<CF<Bits>> a(A,aligned);ComplexQRFactor qr=la.factor_qr_complex(Bits,a.p,m,n,run.options);const QRInfo& info=qr.info();
        require(qr.pivoted()&&info.rank==want.rank&&info.reason==want.reason&&info.status==want.status&&info.norm_recomputations==want.recomputed,l+": rank "+std::to_string(info.rank)+
            " reason "+std::to_string(info.reason)+" status "+std::to_string(info.status)+" recomputed "+std::to_string(info.norm_recomputations)+", want "+std::to_string(want.rank)+" "+
            std::to_string(want.reason)+" "+std::to_string(want.status)+" "+std::to_string(want.recomputed));
        for(std::size_t j=0;j<n;++j)require(qr.permutation()[j]==want.perm[j],l+": permutation entry "+std::to_string(j)+" is "+std::to_string(qr.permutation()[j])+", want "+std::to_string(want.perm[j]));
        same<Bits>(static_cast<const CF<Bits>*>(qr.r()),want.R,n,l+" R");same<Bits>(static_cast<const CF<Bits>*>(qr.v()),want.V,n,l+" V");
        same<Bits>(static_cast<const CF<Bits>*>(qr.tau()),want.tau,1,l+" tau");same<Bits>(static_cast<const CF<Bits>*>(qr.t()),want.T,nb?nb:1,l+" T");
        if(!crossed&&want.rank==n&&!want.status&&n<=200){crossed=true;++crossed_run;std::vector<CF<Bits>> AP(m*n);for(std::size_t r=0;r<m;++r)for(std::size_t j=0;j<n;++j)AP[r*n+j]=A[r*n+want.perm[j]];
            auto u=ref_qr<Bits>(AP,m,n,nb,rank_bits);require(u.rank==n,l+": A P reference rank");
            same<Bits>(u.R.data(),want.R,n,l+" R of A P");same<Bits>(u.V.data(),want.V,n,l+" V of A P");same<Bits>(u.T.data(),want.T,nb?nb:1,l+" T of A P");same<Bits>(u.tau.data(),want.tau,1,l+" tau of A P");}
        ++pivoted_run;pivoted_deficient+=want.rank<n;recomputations+=want.recomputed;gpu_updates+=info.gpu_updates;host_updates+=info.host_updates;
        std::size_t nrhs=k%3==0?1:k%3==1?3:5;auto B=rhs<Bits>(m,nrhs,rng);if(k==1&&m>3)B[nrhs+1].im.status=invalid;
        {Mat<CF<Bits>> b(B,!aligned),x(n*nrhs,aligned,czero<Bits>());qr.solve(b.p,nrhs,x.p);same<Bits>(x.p,ref_solve_pivoted<Bits>(want,B,nrhs),nrhs,"pivoted solve nrhs="+std::to_string(nrhs)+" "+l);}
        for(bool adjoint:{true,false}){auto C=B;ref_apply<Bits>(want,C,nrhs,adjoint);Mat<CF<Bits>> b(B,aligned!=adjoint);qr.apply_q(b.p,nrhs,adjoint);
            same<Bits>(b.p,C,nrhs,std::string(adjoint?"pivoted Q^H":"pivoted Q")+" "+l);}
    }
}
template<int Bits> void pivoted(Linalg& la,std::mt19937_64& rng,bool thorough){
    constexpr int N=Bits/32;
    for(auto [m,n,kind]:std::vector<std::tuple<std::size_t,std::size_t,Kind>>{{1,1,moderate},{5,3,qsc},{33,7,graded},{60,30,qsc},{64,32,wide},{70,40,spread},{150,60,moderate},{40,20,realonly}})
        factor_pivoted<Bits>(la,matrix<Bits>(m,n,kind,rng),m,n,kind_name(kind),rng);
    // Rank deficiency: exactly dependent columns (column n-1 = column 1, column n/2 = i column 0; found with rank_bits), zero columns
    // (moved last), duplicate and i-multiple columns (ties go to the lower original index), a status (chosen first: rank 0, in either
    // part), and nearly parallel columns (a column plus a perturbation 2^-(bits/4+10) below it, or (1+i) times one: norms recomputed).
    factor_pivoted<Bits>(la,matrix<Bits>(41,7,dependent,rng),41,7,"dependent",rng,Bits/2,5);
    factor_pivoted<Bits>(la,matrix<Bits>(90,40,dependent,rng),90,40,"dependent 90x40",rng,Bits/2,38);
    {auto A=matrix<Bits>(30,12,moderate,rng);for(std::size_t r=0;r<30;++r){A[r*12+2]=czero<Bits>();A[r*12+7]=czero<Bits>();}factor_pivoted<Bits>(la,A,30,12,"zero columns 2, 7",rng,0,10);}
    {auto A=matrix<Bits>(25,9,moderate,rng);for(std::size_t r=0;r<25;++r){A[r*9+6]=A[r*9+2];const CF<Bits> z=A[r*9+2];A[r*9+8]={negate(z.im),z.re};}
     factor_pivoted<Bits>(la,A,25,9,"duplicate and i-multiple columns",rng,Bits/2,7);}
    {auto A=matrix<Bits>(25,10,moderate,rng);A[7*10+4].re.status=invalid;factor_pivoted<Bits>(la,A,25,10,"status at (7,4).re",rng,0,0);
     A=matrix<Bits>(40,24,qsc,rng);A[33*24+17].im.status=exponent_overflow;factor_pivoted<Bits>(la,A,40,24,"status at (33,17).im",rng,0,0);}
    {auto A=matrix<Bits>(80,20,moderate,rng);for(std::size_t r=0;r<80;++r){for(std::size_t j:{5,13}){CF<Bits> t{reference::random_number<Bits>(rng,4),reference::random_number<Bits>(rng,4)};
            t.re.exponent-=Bits/4+10;t.im.exponent-=Bits/4+10;A[r*20+j]=reference::complex<Bits>(Operation::complex_add,A[r*20+j-1],t);}
         const CF<Bits> one_i{power<Bits>(0),power<Bits>(0)};CF<Bits> t{reference::random_number<Bits>(rng,4),zero<N>()};t.re.exponent-=Bits/4+10;
         A[r*20+17]=reference::complex<Bits>(Operation::complex_add,reference::complex<Bits>(Operation::complex_mul,A[r*20+9],one_i),t);}
     factor_pivoted<Bits>(la,A,80,20,"nearly parallel columns",rng,0,SIZE_MAX,true);}
    // Rank deficiency found by the rank test after pivoting, in a block boundary case (rank 33 of 48 at block 16 / 32).
    {const std::size_t m=70,n=48;auto B=matrix<Bits>(m,33,moderate,rng),C=matrix<Bits>(33,15,moderate,rng);std::vector<CF<Bits>> A(m*n);
     for(std::size_t r=0;r<m;++r){for(std::size_t j=0;j<33;++j)A[r*n+j]=B[r*33+j];
         for(std::size_t j=0;j<15;++j){std::vector<CF<Bits>> u(B.begin()+r*33,B.begin()+r*33+33);A[r*n+33+j]=pdot<Bits>(u.data(),1,C.data()+j,15,33);}}
     QROptions b16,b32;b16.pivot=b32.pivot=true;b16.block=16;b16.host_macs=0;b16.solve_host_macs=0;b32=b16;b32.block=32;
     factor_pivoted<Bits>(la,A,m,n,"rank 33 of 48 (products of random factors)",rng,Bits/2,33,false,{{"gpu block 16",b16},{"gpu block 32",b32}});}
    factor_pivoted<Bits>(la,matrix<Bits>(300,150,qsc,rng),300,150,"QSC-like",rng);
    if(thorough)factor_pivoted<Bits>(la,matrix<Bits>(400,200,qsc,rng),400,200,"QSC-like",rng);
}
template<int Bits> void suite(Linalg& la,bool thorough){
    std::mt19937_64 rng(4343+Bits);special<Bits>(la,rng);pivoted<Bits>(la,rng,thorough);
    std::vector<std::tuple<std::size_t,std::size_t,Kind>> shapes={{1,1,moderate},{3,2,qsc},{33,7,graded},{128,64,wide},{257,129,qsc}};
    for(auto [m,n,kind]:shapes){if(m==257&&!thorough)continue;factor<Bits>(la,matrix<Bits>(m,n,kind,rng),m,n,kind_name(kind),rng);}
    if(!thorough)return;
    // Block boundaries for blocks 16 and 32 (square and tall).
    QROptions b16;b16.block=16;b16.host_macs=0;b16.solve_host_macs=0;QROptions b32=b16;b32.block=32;QROptions h32;h32.gpu=false;
    for(std::size_t n:{15,16,17,31,32,33,64,65}){std::vector<Run> rs={{"gpu block 16",b16},{"gpu block 32",b32},{"host block 32",h32}};
        factor<Bits>(la,matrix<Bits>(n+n/2+1,n,n%2?qsc:moderate,rng),n+n/2+1,n,"block boundary",rng,rs);
        if(n%16==0)factor<Bits>(la,matrix<Bits>(n,n,graded,rng),n,n,"square block boundary",rng,rs);}
}
// Every 32-bit width: small shapes with every product on the GPU (block 7 and 16), against the replay.
template<int Bits> void width(Linalg& la){std::mt19937_64 rng(77+Bits);QROptions g7,g16;g7.block=7;g7.host_macs=0;g7.solve_host_macs=0;g16=g7;g16.block=16;
    std::vector<Run> rs={{"gpu block 7",g7},{"gpu block 16",g16}};factor<Bits>(la,matrix<Bits>(45,23,qsc,rng),45,23,"all widths",rng,rs);
    factor<Bits>(la,matrix<Bits>(40,33,spread,rng),40,33,"all widths spread",rng,rs);
    QROptions p7=g7,p16=g16;p7.pivot=p16.pivot=true;std::vector<Run> ps={{"pivoted gpu block 7",p7},{"pivoted gpu block 16",p16}};
    factor_pivoted<Bits>(la,matrix<Bits>(45,23,qsc,rng),45,23,"all widths",rng,0,SIZE_MAX,false,ps);
    factor_pivoted<Bits>(la,matrix<Bits>(41,7,dependent,rng),41,7,"all widths dependent",rng,Bits/2,5,false,ps);}
template<int B=64> void all_widths(Linalg& la){width<B>(la);std::cout<<B<<' '<<std::flush;if constexpr(B<1024)all_widths<B+32>(la);}
int main(int argc,char** argv){try{
    bool quick=false,widths=false;for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--quick")quick=true;else if(a=="--all-widths")widths=true;else throw std::invalid_argument("usage: test_limbforge_qr_complex [--quick] [--all-widths]");}
    Linalg la;std::cout<<la.device_name()<<"\n";
    if(widths){std::cout<<"all widths: ";all_widths(la);std::cout<<"\n"<<factors_run<<" factorizations, "<<solves_run<<" solves and "<<applies_run<<" Q applications at every width bit-identical to the reference; "
                          <<pivoted_run<<" pivoted factorizations (with solves and Q applications) bit-identical to the pivoted reference\n";return 0;}
    suite<224>(la,!quick);std::cout<<"complex qr 224 bits: match the MPFR replay\n";
    suite<256>(la,!quick);std::cout<<"complex qr 256 bits: match the MPFR replay\n";
    suite<64>(la,false);suite<384>(la,false);std::cout<<"complex qr 64, 384 bits: match the MPFR replay\n";
    // QSC-like complex least squares: 400 x 200 (and 800 x 400 unless --quick), column scales 2^[-60, 60].
    for(int bits:{224,256}){std::mt19937_64 rng(400+bits);QROptions b16;b16.block=16;b16.host_macs=0;b16.solve_host_macs=0;std::vector<Run> rs={{"default",QROptions{}},{"gpu block 16",b16}};
        for(std::size_t n:{200,400}){if(n==400&&quick)continue;
            if(bits==224)factor<224>(la,matrix<224>(2*n,n,qsc,rng),2*n,n,"QSC-like",rng,rs);else factor<256>(la,matrix<256>(2*n,n,qsc,rng),2*n,n,"QSC-like",rng,rs);
            std::cout<<"QSC-like complex "<<2*n<<"x"<<n<<" at "<<bits<<" bits: match the MPFR replay\n";}}
    std::cout<<real_matches<<" real-input factorizations bit-identical to Linalg::factor_qr\n";
    std::cout<<pivoted_run<<" pivoted factorizations ("<<pivoted_deficient<<" rank-deficient, "<<recomputations<<" recomputed norms; "<<crossed_run<<" equal to the unpivoted replay of A P) bit-identical to the pivoted reference\n";
    std::cout<<factors_run<<" factorizations ("<<deficient<<" rank-deficient; "<<gpu_updates<<" GPU and "<<host_updates<<" host block updates), "<<solves_run<<" solves and "
             <<applies_run<<" Q applications bit-identical to the reference\n";return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
