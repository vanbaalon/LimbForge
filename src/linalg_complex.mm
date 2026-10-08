// Complex Householder QR and least squares (round 43, plan S5; docs/numerics.md, "Complex QR factorization and least squares").
// Reflectors follow LAPACK zgeqrf (H = I - tau v v^H, complex tau, real beta). Every complex dot product or update is two real
// exact dot products over the stacked [re, im] operands, rounded once per real component: on the host by exact_dot /
// exact_dot_add on the interleaved Number sequences, and in the trailing updates by the real compact-WY block update
// (Linalg::Impl::qr_block, through src/linalg_internal.hpp) applied to real embeddings: V and T as real 2x2-block matrices
// [[re, -im], [im, re]] and the work matrix "row split" (row 2r = Re of row r, row 2r+1 = Im). Each embedded output is one real
// exact dot of the complex output's real or imaginary part, so host and GPU placements give identical bits.
#import <Foundation/Foundation.h>
#include "limbforge/linalg.hpp"
#include "linalg_internal.hpp"
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace limbforge {
namespace detail {
// Zeroed, page-aligned storage of whole pages (the GPU uses it in place).
struct PageMemory {
    void* p=nullptr;std::size_t bytes=0;
    PageMemory()=default;
    explicit PageMemory(std::size_t n){if(!n)return;std::size_t page=std::size_t(getpagesize());bytes=(n+page-1)/page*page;if(posix_memalign(&p,page,bytes))throw std::bad_alloc();std::memset(p,0,bytes);}
    ~PageMemory(){std::free(p);}
    PageMemory(PageMemory&& o)noexcept:p(o.p),bytes(o.bytes){o.p=nullptr;o.bytes=0;}
    PageMemory& operator=(PageMemory&& o)noexcept{std::swap(p,o.p);std::swap(bytes,o.bytes);return *this;}
};
// Complex outputs (V, R, tau, T; Complex<N>) and the real embeddings used by the block updates and the solve: ev = V as a real
// 2m x 2n matrix of 2x2 blocks, et = T blocks as real 2nb x 2nb matrices, er = R as a real 2n x 2n matrix.
struct ComplexQRData {Linalg* owner=nullptr;int bits=0;std::size_t m=0,n=0,nb=0;QRInfo info;QROptions options;PageMemory v,r,t,tau,ev,et,er;};
}
namespace {
using Clock=std::chrono::steady_clock;
double since(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
using Hooks=detail::LinalgHooks;
template<int N=2,class F> auto words_of(int words,F&& f){if constexpr(N<32){if(words!=N)return words_of<N+1>(words,f);}return f(std::integral_constant<int,N>{});}
template<int N> using Cx=Complex<N>;
// A contiguous complex vector as the interleaved real sequence (re_0, im_0, re_1, ...).
template<int N> const Number<N>* flat(const Cx<N>* p){return reinterpret_cast<const Number<N>*>(p);}
template<int N> Number<N> scaled(Number<N> x,long e){if(!x.sign)return x;long r=long(x.exponent)-e;if(r>1000000000L||r< -1000000000L)return zero<N>(x.status|exponent_overflow);x.exponent=int(r);return x;}
template<int N> Number<N> power2(int e){Number<N> x=zero<N>();x.limb[N-1]=0x80000000u;x.sign=1;x.exponent=e;return x;}
template<int N> Cx<N> czero(){return {zero<N>(),zero<N>()};}
template<int N> Cx<N> conj(const Cx<N>& z){return {z.re,negate(z.im)};}
template<int N> Cx<N> times_i(const Cx<N>& z){return {negate(z.im),z.re};}   // i z
template<int N> Cx<N> i_conj(const Cx<N>& z){return {z.im,z.re};}           // i conj(z)
// Hermitian dot u^H w = sum conj(u_k) w_k over K contiguous entries: Re = (u . w), Im = (iu . w) on the interleaved sequences,
// with iu = i u supplied by the caller; one rounding per component.
template<int N> Cx<N> hdot(const Cx<N>* u,const Cx<N>* iu,const Cx<N>* w,std::size_t K){return {exact_dot<N>(flat(u),1,flat(w),1,2*K),exact_dot<N>(flat(iu),1,flat(w),1,2*K)};}
// Plain dot sum u_k w_k with wc = conj(w), wic = i conj(w) supplied: Re = (u . wc), Im = (u . wic).
template<int N> Cx<N> pdot(const Cx<N>* u,const Cx<N>* wc,const Cx<N>* wic,std::size_t K){return {exact_dot<N>(flat(u),1,flat(wc),1,2*K),exact_dot<N>(flat(u),1,flat(wic),1,2*K)};}
// The 2x2 real block [[re, -im], [im, re]] of z at (2r, 2c) of a real matrix with row stride ld.
template<int N> void embed(Number<N>* E,std::size_t ld,std::size_t r,std::size_t c,const Cx<N>& z){
    E[2*r*ld+2*c]=z.re;E[2*r*ld+2*c+1]=negate(z.im);E[(2*r+1)*ld+2*c]=z.im;E[(2*r+1)*ld+2*c+1]=z.re;}
// Split exact sums of the panel (as the real QR's panel): an exact window (positive and negative parts, 32-bit words) anchored
// at a bound on the term exponent sums; windows are added exactly and rounded once, which is exact_dot's value whatever the
// grouping. A window too wide for cap words is reported, and the caller uses exact_dot instead.
constexpr int cap=128;
struct Window {word status;long lo,hi;std::size_t terms;int words;bool wide;word* pos;word* neg;};
struct Range {long lo,hi;};
// Exponent range of the nonzero, status-free entries of K reals.
template<int N> Range range_of(const Number<N>* x,std::size_t K){Range g{LONG_MAX,LONG_MIN};for(std::size_t k=0;k<K;++k)if(x[k].sign&&!x[k].status){g.lo=std::min(g.lo,long(x[k].exponent));g.hi=std::max(g.hi,long(x[k].exponent));}return g;}
// w += x*y exactly at exponent-sum anchor lo (the window is wide enough by construction).
template<int N> void add_product(word* pos,word* neg,const Number<N>& x,const Number<N>& y,long lo){word t[64];detail::mantissa_mul<N>(x.limb,y.limb,t);
    detail::add_shifted(x.sign==y.sign?pos:neg,t,2*N,long(x.exponent)+y.exponent-lo);}
// Window of sum_k a[k] b[k] (k < K) anchored at its own lowest term: status OR, range of exponent sums, exact parts.
template<int N> void window_dot(Window& w,const Number<N>* a,const Number<N>* b,std::size_t K){
    w.status=0;w.lo=LONG_MAX;w.hi=LONG_MIN;w.terms=0;w.words=0;w.wide=false;
    for(std::size_t k=0;k<K;++k){w.status|=a[k].status|b[k].status;if(a[k].sign&&b[k].sign){long e=long(a[k].exponent)+b[k].exponent;w.lo=std::min(w.lo,e);w.hi=std::max(w.hi,e);++w.terms;}}
    if(w.status||!w.terms)return;
    int logc=0;while((std::size_t(1)<<logc)<w.terms)++logc;
    const long need=w.hi-w.lo+64*N+logc+1;if(need>32L*(cap-2)){w.wide=true;return;}
    w.words=int(need/32)+2;std::fill(w.pos,w.pos+w.words,0);std::fill(w.neg,w.neg+w.words,0);
    for(std::size_t k=0;k<K;++k)if(a[k].sign&&b[k].sign)add_product<N>(w.pos,w.neg,a[k],b[k],w.lo);
}
// RN(pos - neg) for exact nonnegative W-word parts at exponent-sum anchor lo (both are modified); zero(status) for a status.
template<int N> Number<N> round_parts(word* pos,word* neg,std::size_t W,long lo,word status,std::vector<word>* exact=nullptr){
    if(status)return zero<N>(status);int cmp=0;for(std::size_t i=W;i-->0;)if(pos[i]!=neg[i]){cmp=pos[i]>neg[i]?1:-1;break;}
    if(!cmp){if(exact)exact->clear();return zero<N>();}word* x=cmp>0?pos:neg;const word* y=cmp>0?neg:pos;dword borrow=0;
    for(std::size_t i=0;i<W;++i){dword d=dword(y[i])+borrow;borrow=dword(x[i])<d;x[i]=word(dword(x[i])-d);}
    if(exact)exact->assign(x,x+W);return detail::pack_words<N>(x,W,lo-2*(32*N-1),cmp,ok);}
// out = RN(exact total of windows [0, count)); false when a window or the total is too wide. With exact (a sum of squares,
// so nonnegative), also the total's magnitude words at exponent-sum anchor *lo_out.
template<int N> bool merge_windows(const Window* w,std::size_t count,Number<N>& out,std::vector<word>* exact=nullptr,long* lo_out=nullptr){
    word status=0;bool wide=false;long lo=LONG_MAX,hi=LONG_MIN;std::size_t terms=0;
    for(std::size_t i=0;i<count;++i){status|=w[i].status;wide|=w[i].wide;if(w[i].terms){lo=std::min(lo,w[i].lo);hi=std::max(hi,w[i].hi);terms+=w[i].terms;}}
    if(status){out=zero<N>(status);if(exact)exact->clear();return true;}
    if(wide)return false;
    if(!terms){out=zero<N>();if(exact)exact->clear();if(lo_out)*lo_out=0;return true;}
    int logc=0;while((std::size_t(1)<<logc)<terms)++logc;
    std::size_t W=std::size_t((hi-lo+64*N+logc+1)/32)+2;
    for(std::size_t i=0;i<count;++i)if(w[i].terms)W=std::max(W,std::size_t((w[i].lo-lo)/32)+std::size_t(w[i].words)+2);
    constexpr std::size_t top=256;if(W>top)return false;
    word pos[top],neg[top];std::fill(pos,pos+W,0);std::fill(neg,neg+W,0);
    for(std::size_t i=0;i<count;++i)if(w[i].terms){detail::add_shifted(pos,w[i].pos,w[i].words,w[i].lo-lo);detail::add_shifted(neg,w[i].neg,w[i].words,w[i].lo-lo);}
    out=round_parts<N>(pos,neg,W,lo,0,exact);if(lo_out)*lo_out=lo;return true;
}
// P(v^H v) for v = 2^-e (v0, x_1, ...), v0 = (v0re, Im alpha), without forming v: RN(2^-2e (v0re^2 + S - (Re alpha)^2)) for the
// exact S = |alpha|^2 + sum_{r>0} |x_r|^2 (magnitude words at exponent sum lo); requires every scaled entry in range.
template<int N> Number<N> reflector_norm(const std::vector<word>& S,long lo,const Number<N>& x0,const Number<N>& v0,long e){
    const long sv=2L*v0.exponent,sx=x0.sign?2L*x0.exponent:sv,low=std::min({lo,sv,sx});
    std::vector<word> acc(std::max(S.size()+std::size_t((lo-low)/32),std::size_t((std::max(sv,sx)-low)/32)+2*N)+3,0);word t[64];
    detail::accumulate(acc,S.data(),int(S.size()),lo-low,1);detail::mantissa_mul<N>(v0.limb,v0.limb,t);detail::accumulate(acc,t,2*N,sv-low,1);
    if(x0.sign){detail::mantissa_mul<N>(x0.limb,x0.limb,t);detail::accumulate(acc,t,2*N,sx-low,-1);}
    return detail::pack_words<N>(acc,low-2*(32*N-1)-2*e,1,ok);
}
}
template<int N> static void factor_n(Linalg& la,const Cx<N>* A,detail::ComplexQRData& f);
template<int N> static Timing apply_n(const detail::ComplexQRData& f,Cx<N>* B,std::size_t nrhs,bool adjoint);
template<int N> static Timing solve_n(const detail::ComplexQRData& f,const Cx<N>* B,std::size_t nrhs,Cx<N>* X);
// Resident-list scope of one call (restores Linalg's resident buffers on exit).
namespace { struct Scope {Linalg& la;std::size_t mark;explicit Scope(Linalg& l):la(l),mark(Hooks::enter(l)){} ~Scope(){Hooks::leave(la,mark);}}; }
template<int N> static void factor_n(Linalg& la,const Cx<N>* A,detail::ComplexQRData& f){@autoreleasepool{
    using T=Number<N>;using C=Cx<N>;auto start=Clock::now();QRInfo& info=f.info;info=QRInfo{};const std::size_t m=f.m,n=f.n,nb=f.nb;info.rank=n;
    const QROptions& fo=f.options;Scope scope(la);
    C* V=static_cast<C*>(f.v.p);C* Tc=static_cast<C*>(f.t.p);C* tau=static_cast<C*>(f.tau.p);T* EV=static_cast<T*>(f.ev.p);T* ET=static_cast<T*>(f.et.p);
    // Work matrix, row split (2m x n real: row 2r = Re of row r, row 2r+1 = Im), and the W, Y scratch (2nb x n) of the block updates.
    const bool gpu=fo.gpu&&n>nb;std::vector<T> host;T *Wk,*Wb,*Yb;
    if(gpu){Wk=static_cast<T*>(Hooks::scratch(la,"cqr_work",2*m*n*sizeof(T)));Wb=static_cast<T*>(Hooks::scratch(la,"cqr_w",2*nb*n*sizeof(T)));
        Yb=static_cast<T*>(Hooks::scratch(la,"cqr_y",2*nb*n*sizeof(T)));Hooks::resident(la,f.ev.p,f.ev.bytes);Hooks::resident(la,f.et.p,f.et.bytes);}
    else{host.resize(2*m*n+4*nb*n);Wk=host.data();Wb=Wk+2*m*n;Yb=Wb+2*nb*n;}
    Hooks::each(la,m,[&](std::size_t r){for(std::size_t j=0;j<n;++j){Wk[2*r*n+j]=A[r*n+j].re;Wk[(2*r+1)*n+j]=A[r*n+j].im;}});
    // Panel scratch, columns contiguous over rows [k0, m): the panel's columns (Pt), reflectors v (Vt) and i v (Vi); reflector
    // rows (Vr, nb per row) for the column updates; panel W rows (Wp[i][c] = P(v_i^H a_c), a_c as at the block start); Gram column.
    std::vector<C> Pt(nb*m),Vt(nb*m),Vi(nb*m),Vr(nb*m),Wp(nb*nb),G(nb),Gc(nb),Gic(nb),y(nb),yc(nb),yic(nb),tcol(nb),itcol(nb),wl(nb);
    const T two=power2<N>(1),one=power2<N>(0);long top=LONG_MIN;const unsigned lanes=Hooks::lanes(la);
    // Exact windows: the column norm per row block (merged into the exact S), notes on the entries below the diagonal per row
    // block, and per-thread windows of the Gram/W dots (two real dots, Re and Im, per complex dot), anchored at the product of
    // the two operands' exponent ranges (prange: panel columns at the block start; vrange: reflectors), which bounds every term.
    struct Tail {bool below;long emin,emax;};struct Acc {word status;bool used;word* pos;word* neg;};
    const std::size_t max_blocks=m/8+1,D=2*nb;std::vector<Window> parts(max_blocks);std::vector<word> arena(max_blocks*2*cap),sum;std::vector<Tail> tails(max_blocks);
    for(std::size_t i=0;i<parts.size();++i){parts[i].pos=arena.data()+i*2*cap;parts[i].neg=parts[i].pos+cap;}
    std::vector<Acc> accs(std::size_t(lanes)*D);std::vector<word> acc_arena(accs.size()*2*cap);
    for(std::size_t i=0;i<accs.size();++i){accs[i].pos=acc_arena.data()+i*2*cap;accs[i].neg=accs[i].pos+cap;}
    std::vector<Range> prange(nb),vrange(nb);std::vector<long> dlo(D);std::vector<int> dwords(D);
    auto factor_panel=[&](std::size_t k0,std::size_t k1)->std::size_t{auto t=Clock::now();++info.blocks;
        const std::size_t kb=k1-k0,mr=m-k0;C* Tb=Tc+(k0/nb)*nb*nb;
        Hooks::each(la,kb,[&](std::size_t c){C* x=Pt.data()+c*mr;for(std::size_t r=0;r<mr;++r)x[r]={Wk[2*(k0+r)*n+k0+c],Wk[(2*(k0+r)+1)*n+k0+c]};prange[c]=range_of<N>(flat(x),2*mr);
            std::fill(Vt.data()+c*mr,Vt.data()+(c+1)*mr,czero<N>());std::fill(Vi.data()+c*mr,Vi.data()+(c+1)*mr,czero<N>());});
        std::fill(Vr.begin(),Vr.begin()+mr*nb,czero<N>());
        // Row blocks of BR rows (about four per lane, claimed dynamically).
        const std::size_t BR=std::clamp<std::size_t>(mr/(4*lanes),8,256);
        for(std::size_t c=0;c<kb;++c){const std::size_t j=k0+c;C* x=Pt.data()+c*mr;const std::size_t q=std::min(j,info.rank)-k0;const bool reflect=info.rank==n;
            // Column j by the block's reflectors [k0, k0+q): y_i = P(conj(T_b[0..i][i]) . w[0..i]) (w_l = Wp[l][c]), then
            // x_r = D(x_r; V[r][k0..k0+q), y) per component. In the same pass every row block sums its part of |x[c..mr)|^2 exactly
            // and notes the nonzeros below row c and their exponents.
            for(std::size_t i=0;i<q;++i){for(std::size_t l=0;l<=i;++l){tcol[l]=Tb[l*nb+i];itcol[l]=times_i(tcol[l]);wl[l]=Wp[l*nb+c];}
                y[i]=hdot<N>(tcol.data(),itcol.data(),wl.data(),i+1);yc[i]=conj(y[i]);yic[i]=i_conj(y[i]);}
            const std::size_t nbk=(mr+BR-1)/BR;
            Hooks::each(la,nbk,[&](std::size_t k){const std::size_t r0=k*BR,r1=std::min(mr,r0+BR);
                if(q)for(std::size_t r=r0;r<r1;++r){const T* vr=flat(Vr.data()+r*nb);
                    x[r].re=exact_dot_add<N>(&x[r].re,true,vr,1,flat(yc.data()),1,2*q);x[r].im=exact_dot_add<N>(&x[r].im,true,vr,1,flat(yic.data()),1,2*q);}
                if(!reflect)return;
                const std::size_t a=std::max(r0,c);window_dot<N>(parts[k],flat(x+a),flat(x+a),r1>a?2*(r1-a):0);
                Tail tl{false,LONG_MAX,LONG_MIN};for(std::size_t r=std::max(r0,c+1);r<r1;++r)for(const T* z:{&x[r].re,&x[r].im})if(z->sign){tl.below=true;
                    tl.emin=std::min(tl.emin,long(z->exponent));tl.emax=std::max(tl.emax,long(z->exponent));}
                tails[k]=tl;});
            if(!reflect)continue;  // after a failing column: the remainder only
            // Reflector j from (alpha; x) = x[c..mr): s = P(x^H x); beta = -sgn(Re alpha) sqrt(s); v0 = (RN(Re alpha - beta), Im alpha);
            // v = 2^-e (v0, x_1, ...) with e = exponent(Re v0); tau from z = 1/tau = (P(v^H v)/2, RN(2^-e beta Im v0)). Nothing to
            // reflect (x below zero and Im alpha = 0): tau = 0, v = e_j, r_jj = alpha. s and P(v^H v) come from the exact S
            // (P(v^H v) = RN(2^-2e (v0re^2 + S - (Re alpha)^2))) when it fits and nothing leaves the exponent range, else exact_dot.
            const std::size_t len=mr-c;C* xc=x+c;T s;long slo=0;const bool exact=merge_windows<N>(parts.data(),nbk,s,&sum,&slo);
            if(!exact)s=exact_dot<N>(flat(xc),1,flat(xc),1,2*len);
            const C alpha=xc[0];bool below=false;long emin=LONG_MAX,emax=LONG_MIN;
            for(std::size_t k=0;k<nbk;++k){below|=tails[k].below;emin=std::min(emin,tails[k].emin);emax=std::max(emax,tails[k].emax);}
            const bool needs=below||alpha.im.sign;
            int reason=s.status?QRInfo::status_column:(!alpha.re.sign&&!alpha.im.sign&&!below)?QRInfo::zero_column:QRInfo::full_rank;T beta=alpha.re,v0=one;
            if(!reason&&needs){beta=sqrt(s);if(alpha.re.sign>=0)beta=negate(beta);v0=sub(alpha.re,beta);}
            if(!reason&&fo.rank_bits>0&&top!=LONG_MIN&&long(beta.exponent)+fo.rank_bits<top)reason=QRInfo::small_column;
            if(reason){info.rank=j;info.reason=reason;info.status=s.status;continue;}
            C* vc=Vt.data()+c*mr;C* ic=Vi.data()+c*mr;C tj=czero<N>();long e=0;bool scale_rows=false;vrange[c]={0,0};
            if(needs){e=v0.exponent;vc[c]={scaled(v0,e),scaled(alpha.im,e)};T sv;
                const long ai=alpha.im.sign?long(alpha.im.exponent)-e:0;
                if(exact&&(!below||(emin-e>=-1000000000L&&emax-e<=1000000000L))&&ai>=-1000000000L){sv=reflector_norm<N>(sum,slo,alpha.re,v0,e);scale_rows=true;
                    vrange[c]={std::min({0L,ai,below?emin-e:0L}),std::max(0L,below?emax-e:0L)};}
                else{for(std::size_t r=c+1;r<mr;++r)vc[r]={scaled(x[r].re,e),scaled(x[r].im,e)};sv=exact_dot<N>(flat(vc+c),1,flat(vc+c),1,2*len);vrange[c]=range_of<N>(flat(vc+c),2*len);}
                if(!vc[c].im.sign&&!vc[c].im.status)tj={div(two,sv),zero<N>()};  // z real: one real division (the real QR's tau)
                else{const T a=scaled(sv,1),bb=mul(scaled(beta,e),vc[c].im),d=dot2_add(a,a,bb,bb,zero<N>());tj={div(a,d),negate(div(bb,d))};}}
            else vc[c]={one,zero<N>()};
            xc[0]={beta,zero<N>()};tau[j]=tj;top=std::max(top,long(beta.exponent));ic[c]=times_i(vc[c]);Vr[c*nb+c]=vc[c];
            // Gram column G_u = P(v_u^H v_j) (u < c) and panel W row c: Wp[c][u] = P(v_j^H a_u) (u > c), rows [c, mr), as 2(kb-1) real
            // dots d (u = d/2, Im for odd d). Every thread adds the rows it claims into its own window per dot (after scaling those
            // rows of v when not done above, and setting i v and the reflector rows); the windows are added exactly and rounded once.
            int logl=0;while((std::size_t(1)<<logl)<=2*len)++logl;const std::size_t ndots=2*(kb-1);
            for(std::size_t d=0;d<ndots;++d){const std::size_t u=d/2;const Range &ra=u<c?vrange[u]:vrange[c],&rb=u<c?vrange[c]:prange[u+1];dwords[d]=-1;
                if(ra.lo>ra.hi||rb.lo>rb.hi){dwords[d]=0;continue;}
                const long lo=ra.lo+rb.lo,need=ra.hi+rb.hi-lo+64*N+logl+1;if(need<=32L*(cap-2)){dlo[d]=lo;dwords[d]=int(need/32)+2;}}
            for(auto& g:accs)g.used=false;
            Hooks::run(la,(len+BR-1)/BR,[&](std::size_t k,unsigned id){const std::size_t r0=c+k*BR,r1=std::min(mr,r0+BR);
                for(std::size_t r=std::max(r0,c+1);r<r1;++r){if(scale_rows)vc[r]={scaled(x[r].re,e),scaled(x[r].im,e)};ic[r]=times_i(vc[r]);Vr[r*nb+c]=vc[r];}
                Acc* ac=accs.data()+std::size_t(id)*D;
                for(std::size_t d=0;d<ndots;++d){if(dwords[d]<0)continue;const std::size_t u=d/2;const bool im=d%2;Acc& g=ac[d];
                    if(!g.used){g.used=true;g.status=0;std::fill(g.pos,g.pos+dwords[d],0);std::fill(g.neg,g.neg+dwords[d],0);}
                    const T* pa=flat(u<c?(im?Vi.data():Vt.data())+u*mr:(im?ic:vc));const T* pb=flat(u<c?vc:Pt.data()+(u+1)*mr);
                    for(std::size_t i=2*r0;i<2*r1;++i){const T &xa=pa[i],&xb=pb[i];const word st=xa.status|xb.status;g.status|=st;
                        if(!st&&xa.sign&&xb.sign)add_product<N>(g.pos,g.neg,xa,xb,dlo[d]);}}});
            for(std::size_t d=0;d<ndots;++d){const std::size_t u=d/2;const bool im=d%2;T& out=u<c?(im?G[u].im:G[u].re):(im?Wp[c*nb+u+1].im:Wp[c*nb+u+1].re);
                const T* pa=flat((u<c?(im?Vi.data():Vt.data())+u*mr:(im?ic:vc))+c);const T* pb=flat((u<c?vc:Pt.data()+(u+1)*mr)+c);
                if(dwords[d]<0){out=exact_dot<N>(pa,1,pb,1,2*len);continue;}
                const std::size_t W=std::size_t(dwords[d]);word status=0;word pos[cap+1],neg[cap+1];std::fill(pos,pos+W,0);std::fill(neg,neg+W,0);
                for(std::size_t id=0;id<lanes;++id){const Acc& g=accs[id*D+d];if(!g.used)continue;status|=g.status;dword cp=0,cn=0;
                    for(std::size_t i=0;i<W;++i){cp+=dword(pos[i])+g.pos[i];pos[i]=word(cp);cp>>=32;cn+=dword(neg[i])+g.neg[i];neg[i]=word(cn);cn>>=32;}}
                out=round_parts<N>(pos,neg,W,dlo[d],status);}
            // T_b column c: T_cc = tau_j, T_lc = -RN(tau_j * P(T_b[l][l..c) . G[l..c))) per component.
            for(std::size_t l=0;l<c;++l){Gc[l]=conj(G[l]);Gic[l]=i_conj(G[l]);}
            Tb[c*nb+c]=tj;for(std::size_t l=0;l<c;++l){const C p=pdot<N>(Tb+l*nb+l,Gc.data()+l,Gic.data()+l,c-l),z=cfma(tj,p,czero<N>());Tb[l*nb+c]={negate(z.re),negate(z.im)};}
        }
        // Write back the panel (R entries, remainders), V and its embedding, and the embedding of T_b.
        Hooks::each(la,mr,[&](std::size_t r){for(std::size_t c=0;c<kb;++c){const C& x=Pt[c*mr+r];const C& v=Vt[c*mr+r];
            Wk[2*(k0+r)*n+k0+c]=x.re;Wk[(2*(k0+r)+1)*n+k0+c]=x.im;V[(k0+r)*n+k0+c]=v;embed<N>(EV,2*n,k0+r,k0+c,v);}});
        T* Eb=ET+(k0/nb)*4*nb*nb;for(std::size_t l=0;l<kb;++l)for(std::size_t c=l;c<kb;++c)embed<N>(Eb,2*nb,l,c,Tb[l*nb+c]);
        info.panel_seconds+=since(t);return std::min(k1,info.rank);};
    // Block b applied (as Q^H) to columns [c0, c1): the real block update of the embeddings.
    auto update=[&](std::size_t b,std::size_t c0,std::size_t c1,bool& on_gpu){auto t=Clock::now();
        Timing tm=Hooks::qr_block(la,32*N,2*m,2*n,2*nb,EV,ET,b,Wk,n,c0,c1,true,Wb,Yb,fo.host_macs,gpu,on_gpu);tm.wall_seconds=since(t);return tm;};
    auto account=[&](Timing tm,bool on_gpu){info.timing.gpu_seconds+=tm.gpu_seconds;info.update_seconds+=tm.wall_seconds;++(on_gpu?info.gpu_updates:info.host_updates);};
    // Right-looking with one block of look-ahead, as the real QR: block b updates the next panel's columns, then the rest of
    // the trailing matrix on a second thread while this thread factors the next panel (disjoint columns, identical bits).
    std::size_t k0=0,k1=std::min(n,nb),p=n?factor_panel(0,k1):0;
    while(k1<n){const std::size_t b=k0/nb;bool on_gpu=false;
        if(p<k1){if(p>k0)account(update(b,k1,n,on_gpu),on_gpu);break;}
        const std::size_t k2=std::min(n,k1+nb);account(update(b,k1,k2,on_gpu),on_gpu);
        if(k2<n){Timing rest{0,0};bool rest_gpu=false,async=gpu;std::thread side;std::exception_ptr failed;
            if(async)side=std::thread([&]{@autoreleasepool{Hooks::side(true);try{rest=update(b,k2,n,rest_gpu);}catch(...){failed=std::current_exception();}Hooks::side(false);}});
            try{p=factor_panel(k1,k2);}catch(...){if(async)side.join();throw;}
            if(async){side.join();if(failed)std::rethrow_exception(failed);}else rest=update(b,k2,n,rest_gpu);
            account(rest,rest_gpu);}
        else p=factor_panel(k1,k2);
        k0=k1;k1=k2;}
    // R (upper; with rank p < n, rows i >= p keep the remainder in columns j >= p) and its embedding.
    C* R=static_cast<C*>(f.r.p);T* ER=static_cast<T*>(f.er.p);const std::size_t rank=info.rank;
    Hooks::each(la,n,[&](std::size_t i){for(std::size_t j=0;j<n;++j){C z=j>=std::min(i,rank)?C{Wk[2*i*n+j],Wk[(2*i+1)*n+j]}:czero<N>();R[i*n+j]=z;embed<N>(ER,2*n,i,j,z);}});
    info.timing.wall_seconds=since(start);
}}
// B (m x nrhs) <- Q^H B (adjoint: blocks in order) or Q B (reverse), over the blocks holding reflectors [0, rank): the real block
// updates of the embeddings on B row split.
template<int N> static Timing apply_n(const detail::ComplexQRData& f,Cx<N>* B,std::size_t nrhs,bool adjoint){@autoreleasepool{
    using T=Number<N>;Timing tm{0,0};auto start=Clock::now();const std::size_t m=f.m,n=f.n,nb=f.nb,blocks=nb?(f.info.rank+nb-1)/nb:0;
    if(!blocks||!nrhs){tm.wall_seconds=since(start);return tm;}
    Linalg& la=*f.owner;Scope scope(la);const bool gpu=f.options.gpu&&4.0*double(std::min(nb,n))*double(nrhs)*double(m)>=f.options.solve_host_macs;
    std::vector<T> host;T *W,*Wb,*Yb;
    if(gpu){W=static_cast<T*>(Hooks::scratch(la,"cqr_rhs",2*m*nrhs*sizeof(T)));Wb=static_cast<T*>(Hooks::scratch(la,"cqr_w",2*nb*nrhs*sizeof(T)));
        Yb=static_cast<T*>(Hooks::scratch(la,"cqr_y",2*nb*nrhs*sizeof(T)));Hooks::resident(la,f.ev.p,f.ev.bytes);Hooks::resident(la,f.et.p,f.et.bytes);}
    else{host.resize(2*m*nrhs+4*nb*nrhs);W=host.data();Wb=W+2*m*nrhs;Yb=Wb+2*nb*nrhs;}
    Hooks::each(la,m,[&](std::size_t r){for(std::size_t c=0;c<nrhs;++c){W[2*r*nrhs+c]=B[r*nrhs+c].re;W[(2*r+1)*nrhs+c]=B[r*nrhs+c].im;}});
    for(std::size_t q=0;q<blocks;++q){bool on_gpu;tm.gpu_seconds+=Hooks::qr_block(la,32*N,2*m,2*n,2*nb,f.ev.p,f.et.p,adjoint?q:blocks-1-q,W,nrhs,0,nrhs,adjoint,Wb,Yb,
        f.options.solve_host_macs,gpu,on_gpu).gpu_seconds;}
    Hooks::each(la,m,[&](std::size_t r){for(std::size_t c=0;c<nrhs;++c)B[r*nrhs+c]={W[2*r*nrhs+c],W[(2*r+1)*nrhs+c]};});
    tm.wall_seconds=since(start);return tm;
}}
// X = R^-1 (Q^H B)[0, n): Q^H B by blocks, then the real backward substitution of QRFactor::solve on the embedding of R (blocks
// 2 nb): the real diagonal makes each embedded diagonal block diag(r_ii, r_ii), so x_i = (RN(u_re / r_ii), RN(u_im / r_ii)).
template<int N> static Timing solve_n(const detail::ComplexQRData& f,const Cx<N>* B,std::size_t nrhs,Cx<N>* X){
    using T=Number<N>;auto start=Clock::now();Timing tm{0,0};const std::size_t m=f.m,n=f.n;if(!n||!nrhs)return tm;Linalg& la=*f.owner;
    if(f.info.rank<n){Scope scope(la);Hooks::each(la,n,[&](std::size_t i){for(std::size_t c=0;c<nrhs;++c)X[i*nrhs+c]={zero<N>(invalid),zero<N>(invalid)};});tm.wall_seconds=since(start);return tm;}
    std::vector<Cx<N>> C(B,B+m*nrhs);tm.gpu_seconds+=apply_n<N>(f,C.data(),nrhs,true).gpu_seconds;
    std::vector<T> Cs(2*n*nrhs),Xs(2*n*nrhs);
    for(std::size_t i=0;i<n;++i)for(std::size_t c=0;c<nrhs;++c){Cs[2*i*nrhs+c]=C[i*nrhs+c].re;Cs[(2*i+1)*nrhs+c]=C[i*nrhs+c].im;}
    FactorOptions fo=f.options;fo.block=2*f.nb;tm.gpu_seconds+=Hooks::solve_upper(la,32*N,f.er.p,2*n,Cs.data(),nrhs,Xs.data(),fo).gpu_seconds;
    for(std::size_t i=0;i<n;++i)for(std::size_t c=0;c<nrhs;++c)X[i*nrhs+c]={Xs[2*i*nrhs+c],Xs[(2*i+1)*nrhs+c]};
    tm.wall_seconds=since(start);return tm;
}
static void check_bits(int bits){if(bits<64||bits>1024||bits%32)throw std::invalid_argument("bits must be a multiple of 32 in [64,1024]");}
// The embeddings double both dimensions (32-bit indexing of the real products).
static void check_size(std::size_t n,std::size_t other){if(2*n>=46341||4*n*other>=(std::size_t(1)<<31))throw std::invalid_argument("matrix dimensions exceed 32-bit indexing");}
ComplexQRFactor Linalg::factor_qr_complex(int bits,const void* A,std::size_t m,std::size_t n,const QROptions& o){
    check_bits(bits);if(m<n)throw std::invalid_argument("factor_qr_complex: rows >= columns required");check_size(n,m);if(m*n&&!A)throw std::invalid_argument("null matrix");
    if(o.rank_bits<0)throw std::invalid_argument("factor_qr_complex: rank_bits >= 0");
    if(o.pivot)throw std::invalid_argument("factor_qr_complex: column pivoting is not supported");
    ComplexQRFactor q;q.d=std::make_unique<detail::ComplexQRData>();auto& f=*q.d;f.owner=this;f.bits=bits;f.m=m;f.n=n;f.nb=o.block?std::min(o.block,n):n;f.options=o;
    const std::size_t e=std::size_t(bits/8+12),blocks=f.nb?(n+f.nb-1)/f.nb:0;
    f.v=detail::PageMemory(2*m*n*e);f.r=detail::PageMemory(2*n*n*e);f.t=detail::PageMemory(2*blocks*f.nb*f.nb*e);f.tau=detail::PageMemory(2*n*e);
    f.ev=detail::PageMemory(4*m*n*e);f.et=detail::PageMemory(4*blocks*f.nb*f.nb*e);f.er=detail::PageMemory(4*n*n*e);
    words_of(bits/32,[&](auto w){constexpr int N=decltype(w)::value;factor_n<N>(*this,static_cast<const Cx<N>*>(A),f);});
    return q;
}
ComplexQRFactor::ComplexQRFactor()=default;ComplexQRFactor::~ComplexQRFactor()=default;
ComplexQRFactor::ComplexQRFactor(ComplexQRFactor&&)noexcept=default;ComplexQRFactor& ComplexQRFactor::operator=(ComplexQRFactor&&)noexcept=default;
static const detail::ComplexQRData& data(const std::unique_ptr<detail::ComplexQRData>& d){if(!d)throw std::logic_error("empty complex QR factor");return *d;}
bool ComplexQRFactor::empty()const{return !d;}
int ComplexQRFactor::bits()const{return data(d).bits;}
std::size_t ComplexQRFactor::rows()const{return data(d).m;}
std::size_t ComplexQRFactor::cols()const{return data(d).n;}
std::size_t ComplexQRFactor::block()const{return data(d).nb;}
const QRInfo& ComplexQRFactor::info()const{return data(d).info;}
bool ComplexQRFactor::full_rank()const{return data(d).info.rank==data(d).n;}
const void* ComplexQRFactor::r()const{return data(d).r.p;}
const void* ComplexQRFactor::v()const{return data(d).v.p;}
const void* ComplexQRFactor::tau()const{return data(d).tau.p;}
const void* ComplexQRFactor::t()const{return data(d).t.p;}
Timing ComplexQRFactor::solve(const void* B,std::size_t nrhs,void* X)const{const auto& f=data(d);check_size(f.n,nrhs);check_size(1,f.m*nrhs);if(f.n&&nrhs&&(!B||!X))throw std::invalid_argument("null matrix");
    return words_of(f.bits/32,[&](auto w){constexpr int N=decltype(w)::value;return solve_n<N>(f,static_cast<const Cx<N>*>(B),nrhs,static_cast<Cx<N>*>(X));});}
Timing ComplexQRFactor::apply_q(void* B,std::size_t nrhs,bool adjoint)const{const auto& f=data(d);check_size(f.n,nrhs);check_size(1,f.m*nrhs);if(f.m&&nrhs&&!B)throw std::invalid_argument("null matrix");
    return words_of(f.bits/32,[&](auto w){constexpr int N=decltype(w)::value;return apply_n<N>(f,static_cast<Cx<N>*>(B),nrhs,adjoint);});}
}
