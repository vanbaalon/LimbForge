#pragma once
// Dense real products with one rounding per output (plan D6 via L1c): C[i][j] = RN(sum_k L[i][k]*R[k][j]),
// or RN(C[i][j] - sum_k ...) as an update, computed exactly by an integer GEMM over residues (Metal TensorOps
// int8) and rounded once, ties to even. Contract and algorithm: docs/numerics.md, "Dense products". Host arrays
// only; element = Number<bits/32> (Float<bits>), matrices row-major. Statuses propagate: C[i][j] = zero(OR of the
// statuses of row i of L and column j of R, and of the old C[i][j] for an update) when that OR is nonzero. Empty K
// gives canonical zero (an update leaves C unchanged). Blocked Cholesky, triangular solves and Householder QR build on these updates.
#include "core.hpp"
#include "engine.hpp"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
namespace limbforge {
struct LinalgOptions {
    int band_bits=64;      // G: a band holds exponents [top-G, top] of one row/column; entries are exact (bits+G)-bit integers
    int max_bands=4;       // rows/columns needing more bands use the exact host fallback
    int max_spread=512;    // ... as do rows/columns whose nonzero exponents span more than this (accumulator bound)
    unsigned host_threads=0; // fallback/assembly threads; 0 = hardware concurrency
    bool force_fallback=false; // every output through the exact host path (testing)
};
struct LinalgPair { unsigned left_band,right_band,rows,cols,moduli; };
// Diagnostics of the last call. "Lines" are rows of the left operand and columns of the right operand.
struct LinalgReport {
    std::size_t m=0,n=0,k=0;
    std::vector<std::size_t> left_bands,right_bands; // [b] = lines with exactly b bands (index 0: zero or status lines)
    std::size_t left_fallback=0,right_fallback=0;    // lines routed to the exact host path
    std::size_t gpu_outputs=0,multi_band_outputs=0,fallback_outputs=0,trivial_outputs=0;
    std::vector<LinalgPair> pairs;                   // band pairs (sub-blocks of the extended product); SYRK lists b >= c
    std::size_t gemm_rows=0,gemm_cols=0,int8_gemms=0; // extended operand sizes (all bands); 3 int8 GEMMs per modulus
    bool zero_copy_output=false;                    // C was page aligned and written by the GPU in place
    double analysis_seconds=0,upload_seconds=0,gpu_seconds=0,assembly_seconds=0,fallback_seconds=0;
};
// Blocked Cholesky and triangular solves (docs/numerics.md, "Cholesky factorization"). The result depends on
// `block` (the documented rounding sequence) and on nothing else: which updates run on the GPU or the host
// changes only the speed. An update with fewer multiply-adds (outputs x K) than host_macs (factorization) or
// solve_host_macs (triangular solves) runs on the host.
struct FactorOptions {
    std::size_t block=32;          // column block nb; 0 selects a single block (every dot product exact over all k)
    double host_macs=5e4;          // factorization updates below this many multiply-adds use host exact dots (same results)
    double solve_host_macs=4e5;    // the same for triangular-solve updates (measured: the host wins for a few right-hand sides)
    bool gpu=true;                 // false: everything on the host (same results)
};
// Per-call outcome of cholesky. pivot = n on success; otherwise the first column p whose pivot
// s_pp = RN(a_pp - sum_k l_pk^2) is zero, negative or carries a status (then pivot_sign / pivot_status describe it).
// On failure L[i][j] is final for j < p, and every lower entry with j >= p is zero with status `invalid`.
struct CholeskyInfo {
    std::size_t pivot=0; int pivot_sign=0; word pivot_status=0;
    Timing timing{0,0};                       // gpu_seconds: summed command buffers; wall: whole call
    std::size_t blocks=0,gpu_updates=0,host_updates=0;
    double panel_seconds=0,update_seconds=0;  // host diagonal-block + panel work; trailing updates (GPU or host), wall
};
// Householder QR (docs/numerics.md, "QR factorization and least squares"): FactorOptions (block, GPU/host split; the
// result depends on block only) and an optional rank test: rank_bits > 0 also stops at the first column whose |r_jj|
// lies more than rank_bits binades below the largest earlier |r_kk| (exponent comparison, no rounding).
struct QROptions : FactorOptions { int rank_bits=0; };
// Per-factor outcome. rank = n for a full-rank factor; otherwise the first column p (in order) whose norm is zero
// (zero_column), carries a status (status_column; status = its OR) or fails rank_bits (small_column). Reflectors
// 0 .. p-1 are then final, and the factorization stops there (see QRFactor::r).
struct QRInfo {
    enum Reason {full_rank=0,zero_column=1,small_column=2,status_column=3};
    std::size_t rank=0; int reason=full_rank; word status=0;
    Timing timing{0,0};                       // gpu_seconds: summed command buffers; wall: whole call
    std::size_t blocks=0,gpu_updates=0,host_updates=0; // trailing block updates (3 products each) by placement
    double panel_seconds=0,update_seconds=0;  // host panels; trailing updates (GPU or host), wall
};
namespace detail { struct QRData; }
class Linalg;
// Owning QR factor of an m x n matrix (m >= n): Q = H_0 ... H_{n-1}, H_j = I - tau_j v_j v_j^T, and R. The factor copies
// everything it needs and never refers to A again: when A changes, factor it again (no caching by address). It uses the
// Linalg that made it (scratch buffers, GPU): that Linalg must outlive it, and calls follow the Linalg threading rule.
class QRFactor {
public:
    QRFactor(); ~QRFactor(); QRFactor(QRFactor&&) noexcept; QRFactor& operator=(QRFactor&&) noexcept;
    bool empty() const; int bits() const; std::size_t rows() const; std::size_t cols() const; std::size_t block() const;
    const QRInfo& info() const; bool full_rank() const;
    // Row-major Float<bits> arrays owned by the factor (valid while it lives): R (n x n, upper; with rank p < n, rows
    // i >= p hold the unreduced remainder in columns j >= p), V (m x n: column j is v_j, zero above row j, v_j[j] =
    // the scaled v0), tau (n) and T (blocks of nb x nb, the compact-WY upper triangles; block b at offset b*nb*nb).
    const void* r() const; const void* v() const; const void* tau() const; const void* t() const;
    // X (n x nrhs) = least-squares solution of min ||A x - b|| for each column b of B (m x nrhs): Q^T B, then R X = (Q^T B)[0:n).
    // Full rank only: a rank-deficient factor writes zero with status invalid to every entry of X (no minimum-norm claim).
    // X must not overlap B.
    Timing solve(const void* B,std::size_t nrhs,void* X) const;
    // B (m x nrhs) <- Q^T B (transpose) or Q B, in place.
    Timing apply_q(void* B,std::size_t nrhs,bool transpose) const;
private:
    friend class Linalg; std::unique_ptr<detail::QRData> d;
};
// One Linalg per host thread (scratch buffers and compiled libraries are reused). C must not overlap A or B;
// page-aligned A, B and C are used by the GPU in place. Calls return after the GPU has finished.
class Linalg {
public:
    explicit Linalg(LinalgOptions options={}); ~Linalg();
    Linalg(const Linalg&)=delete; Linalg& operator=(const Linalg&)=delete;
    std::string device_name() const;
    // C (cols x cols) = A^T A for A (rows x cols). lower_only writes C[i][j] for i >= j and leaves the
    // rest of C untouched; otherwise the full symmetric matrix is written. With subtract, C[i][j] = RN(C[i][j] - (A^T A)[i][j])
    // with one rounding (lower triangle only: subtract requires lower_only).
    Timing syrk(int bits,const void* A,std::size_t rows,std::size_t cols,void* C,bool lower_only=true,bool subtract=false);
    // C (m x n) = op(A) B with B (k x n); op(A) = A (m x k) or, with transpose_a, A^T for A (k x m).
    // With subtract, C = RN(C - op(A) B), one rounding per entry.
    Timing gemm(int bits,bool transpose_a,const void* A,const void* B,std::size_t m,std::size_t n,std::size_t k,void* C,bool subtract=false);
    // Lower L (n x n, row-major) with A = L L^T from the lower triangle of symmetric positive definite A; the strict
    // upper triangle of L is set to zero. L may equal A (in place); otherwise they must not overlap.
    CholeskyInfo cholesky(int bits,const void* A,std::size_t n,void* L,const FactorOptions& options={});
    // X (n x nrhs) solves L X = B, or L^T X = B with transpose, for lower L (strict upper triangle not read).
    // X may equal B; otherwise X must not overlap B or L.
    Timing trsm(int bits,bool transpose,const void* L,std::size_t n,const void* B,std::size_t nrhs,void* X,const FactorOptions& options={});
    // X solves (L L^T) X = B: trsm, then trsm with transpose.
    Timing cholesky_solve(int bits,const void* L,std::size_t n,const void* B,std::size_t nrhs,void* X,const FactorOptions& options={});
    // Householder QR of A (m x n, row-major, m >= n); A is only read. info().rank reports rank deficiency.
    QRFactor factor_qr(int bits,const void* A,std::size_t m,std::size_t n,const QROptions& options={});
    // QR of the augmented Levenberg-Marquardt matrix [J; diag(d)] ((m+n) x n) for J (m x n) and the caller's diagonal d
    // (n entries, e.g. sqrt(mu) D): the same factorization of the stacked matrix. Solve with B of m+n rows ([r; 0]).
    QRFactor factor_qr_augmented(int bits,const void* J,std::size_t m,std::size_t n,const void* d,const QROptions& options={});
    const LinalgReport& report() const;
    const LinalgOptions& options() const;
private:
    friend class QRFactor; struct Impl; std::unique_ptr<Impl> impl;
};
// ---- Exact dot product on the CPU: RN(sum_k a[k*sa]*b[k*sb]) with one rounding (the GPU fallback) ----
namespace detail {
// Rounds sign * integer(w) * 2^scale to N words exactly as core.hpp pack() does, for any word count.
template<int N> Number<N> pack_words(const word* w,std::size_t count,exponent_type scale,int sign,word status){
    std::size_t top=count;while(top&&!w[top-1])--top;if(!top)return zero<N>(status);
    long h=long(32*(top-1))+31-__builtin_clz(w[top-1]);
    exponent_type e=scale+h;if(e>1000000000||e< -1000000001)return zero<N>(status|exponent_overflow);
    // 32 bits of w from bit p (bits outside w are zero).
    auto chunk=[&](long p)->word{if(p<=-32)return 0;if(p<0)return w[0]<<(-p);std::size_t i=std::size_t(p/32);int s=int(p%32);
        word x=i<count?w[i]>>s:0;if(s&&i+1<count)x|=w[i+1]<<(32-s);return x;};
    Number<N> r=zero<N>(status);r.sign=sign;r.exponent=int(e);long shift=h-(32*N-1);
    for(int i=0;i<N;++i)r.limb[i]=chunk(shift+32L*i);
    if(shift<=0)return checked(r);
    long rb=shift-1;bool round=(w[rb/32]>>(rb%32))&1,below=(rb%32)&&(w[rb/32]&((word(1)<<(rb%32))-1));
    for(long i=0;i<rb/32&&!below;++i)below=w[i]!=0;
    if(round&&(below||(r.limb[0]&1)))increment(r);
    return checked(r);
}
template<int N> Number<N> pack_words(const std::vector<word>& w,exponent_type scale,int sign,word status){return pack_words<N>(w.data(),w.size(),scale,sign,status);}
// acc += x * 2^shift for a nonnegative accumulator wide enough by construction.
inline void add_shifted(word* acc,const word* x,int words,long shift){
    std::size_t w=std::size_t(shift/32);int s=int(shift%32);dword carry=0;
    for(int i=0;i<=words;++i){word v=i<words?x[i]<<s:0;if(s&&i>0)v|=x[i-1]>>(32-s);dword t=dword(acc[w+i])+v+carry;acc[w+i]=word(t);carry=t>>32;}
    for(std::size_t i=w+std::size_t(words)+1;carry;++i){dword t=dword(acc[i])+carry;acc[i]=word(t);carry=t>>32;}
}
// Two's complement accumulator: acc += sign * x * 2^shift (acc wide enough by construction).
inline void accumulate(std::vector<word>& acc,const word* x,int words,long shift,int sign){
    std::size_t w=std::size_t(shift/32);int s=int(shift%32);dword carry=0;bool borrow=false;
    for(std::size_t i=0;i+w<acc.size();++i){
        word v=0;if(i<std::size_t(words))v=x[i]<<s;if(s&&i>0&&i-1<std::size_t(words))v|=x[i-1]>>(32-s);
        if(i>std::size_t(words)&&!carry&&!borrow)break;
        if(sign>0){dword t=dword(acc[i+w])+v+carry;acc[i+w]=word(t);carry=t>>32;}
        else{dword t=dword(v)+borrow;borrow=dword(acc[i+w])<t;acc[i+w]=word(dword(acc[i+w])-t);}
    }
}
}
// RN(c - sum_k a[k*sa]*b[k*sb]) (subtract) or RN(c + sum ...) with one rounding; c == nullptr is a zero addend.
// The addend is one more exact term. Terms are sorted by exponent and grouped into clusters; a new cluster starts
// when every remaining term lies more than bits+log2(K+1)+4 bits below the current cluster's lowest bit. Each
// cluster is summed exactly. With S the first nonzero cluster sum and F the rest, |F| < 2^(lsb(S)-bits-2), so
// RN(S+F) = RN(S + sign(F)*eps) and sign(F) is the sign of the next nonzero cluster sum. Statuses: zero with the OR
// of the statuses of c and of every a[k], b[k]; without nonzero products the result is c itself.
template<int N> Number<N> exact_dot_add(const Number<N>* c,bool subtract,const Number<N>* a,std::ptrdiff_t sa,const Number<N>* b,std::ptrdiff_t sb,std::size_t K){
    word status=c?c->status:0;for(std::size_t k=0;k<K;++k)status|=a[std::ptrdiff_t(k)*sa].status|b[std::ptrdiff_t(k)*sb].status;
    if(status)return zero<N>(status);
    struct Term{exponent_type scale;std::size_t k;};Term small[65];std::vector<Term> big;Term* t=small;std::size_t count=0;
    if(K+1>65){big.resize(K+1);t=big.data();}
    for(std::size_t k=0;k<K;++k){const auto &x=a[std::ptrdiff_t(k)*sa],&y=b[std::ptrdiff_t(k)*sb];
        if(x.sign&&y.sign)t[count++]={exponent_type(x.exponent)+y.exponent-2*(32*N-1),k};}
    if(!count)return c?*c:zero<N>();
    if(c&&c->sign)t[count++]={exponent_type(c->exponent)-(32*N-1),K}; // the addend, an N-word term, is k = K
    // Exact term k (2N words, or the N-word addend) into p; returns its sign.
    auto term=[&](std::size_t k,word* p,int& words)->int{
        if(k==K){for(int u=0;u<N;++u)p[u]=c->limb[u];words=N;return c->sign;}
        const auto &x=a[std::ptrdiff_t(k)*sa],&y=b[std::ptrdiff_t(k)*sb];for(int u=0;u<2*N;++u)p[u]=0;words=2*N;
        for(int u=0;u<N;++u){dword cy=0;for(int v=0;v<N;++v){dword z=dword(x.limb[u])*y.limb[v]+p[u+v]+cy;p[u+v]=word(z);cy=z>>32;}p[u+N]=word(cy);}
        return subtract?-x.sign*y.sign:x.sign*y.sign;};
    // Common case: all terms fit one exact accumulator pair (positive and negative parts), one rounding of the difference.
    {exponent_type hi=t[0].scale,lo=t[0].scale;for(std::size_t i=1;i<count;++i){hi=std::max(hi,t[i].scale);lo=std::min(lo,t[i].scale);}
     int logc=0;while((std::size_t(1)<<logc)<count)++logc;constexpr std::size_t cap=256;const exponent_type need=hi-lo+64*N+logc+1;
     if(need<=32*exponent_type(cap-2)){const std::size_t W=std::size_t(need/32)+2;word pos[cap],neg[cap],p[64];std::fill(pos,pos+W,0);std::fill(neg,neg+W,0);
        for(std::size_t q=0;q<count;++q){int words;int sign=term(t[q].k,p,words);detail::add_shifted(sign>0?pos:neg,p,words,long(t[q].scale-lo));}
        int cmp=0;for(std::size_t i=W;i-->0;)if(pos[i]!=neg[i]){cmp=pos[i]>neg[i]?1:-1;break;}
        if(!cmp)return zero<N>();
        word* x=cmp>0?pos:neg;const word* y=cmp>0?neg:pos;dword borrow=0;
        for(std::size_t i=0;i<W;++i){dword v=dword(y[i])+borrow;borrow=dword(x[i])<v;x[i]=word(dword(x[i])-v);}
        return detail::pack_words<N>(x,W,lo,cmp,ok);}}
    std::sort(t,t+count,[](const Term& u,const Term& v){return u.scale>v.scale;});
    int logk=0;while((std::size_t(1)<<logk)<K+1)++logk;
    const exponent_type gap=32*N+logk+4,width=64*N; // a term is < 2^(scale+64N)
    struct Sum{int sign;exponent_type scale;std::vector<word> mag;};std::vector<Sum> sums;
    for(std::size_t i=0;i<count&&sums.size()<2;){
        std::size_t j=i+1;exponent_type low=t[i].scale;
        while(j<count&&t[j].scale+width-1+gap>=low){low=std::min(low,t[j].scale);++j;}
        std::vector<word> acc(std::size_t((t[i].scale+width-low)/32+3),0);word p[64];
        for(std::size_t q=i;q<j;++q){int words;int sign=term(t[q].k,p,words);detail::accumulate(acc,p,words,long(t[q].scale-low),sign);}
        int sign=1;if(acc.back()>>31){sign=-1;dword cy=1;for(auto& v:acc){dword z=dword(word(~v))+cy;v=word(z);cy=z>>32;}}
        if(std::any_of(acc.begin(),acc.end(),[](word v){return v!=0;}))sums.push_back({sign,low,std::move(acc)});
        i=j;
    }
    if(sums.empty())return zero<N>();
    const Sum& s=sums[0];if(sums.size()==1)return detail::pack_words<N>(s.mag,s.scale,s.sign,ok);
    // S*2^D -/+ 1: D puts eps below the rounding position even when S is narrower than the target.
    const int D=32*N+64;std::vector<word> ext(s.mag.size()+D/32+1,0);
    for(std::size_t i=0;i<s.mag.size();++i)ext[i+D/32]=s.mag[i];
    if(sums[1].sign==s.sign)ext[0]=1;else for(auto& v:ext){word old=v;v=old-1;if(old)break;}
    return detail::pack_words<N>(ext,s.scale-D,s.sign,ok);
}
// RN(sum_k a[k*sa]*b[k*sb]) with one rounding (the GPU fallback); empty K and exact cancellation give canonical zero.
template<int N> Number<N> exact_dot(const Number<N>* a,std::ptrdiff_t sa,const Number<N>* b,std::ptrdiff_t sb,std::size_t K){
    return exact_dot_add<N>(nullptr,false,a,sa,b,sb,K);}
}
