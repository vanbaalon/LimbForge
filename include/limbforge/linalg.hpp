#pragma once
// Dense real products with one rounding per output (plan D6 via L1c): C[i][j] = RN(sum_k L[i][k]*R[k][j]),
// computed exactly by an integer GEMM over residues (Metal TensorOps int8) and rounded once, ties to even.
// Contract and algorithm: docs/numerics.md, "Dense products". Host arrays only; element = Number<bits/32>
// (Float<bits>), matrices row-major. Statuses propagate: C[i][j] = zero(OR of the statuses of row i of L
// and column j of R) when that OR is nonzero. Empty K gives canonical zero.
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
// One Linalg per host thread (scratch buffers and compiled libraries are reused). C must not overlap A or B;
// page-aligned A, B and C are used by the GPU in place. Calls return after the GPU has finished.
class Linalg {
public:
    explicit Linalg(LinalgOptions options={}); ~Linalg();
    Linalg(const Linalg&)=delete; Linalg& operator=(const Linalg&)=delete;
    std::string device_name() const;
    // C (cols x cols) = A^T A for A (rows x cols). lower_only writes C[i][j] for i >= j and leaves the
    // rest of C untouched; otherwise the full symmetric matrix is written.
    Timing syrk(int bits,const void* A,std::size_t rows,std::size_t cols,void* C,bool lower_only=true);
    // C (m x n) = op(A) B with B (k x n); op(A) = A (m x k) or, with transpose_a, A^T for A (k x m).
    Timing gemm(int bits,bool transpose_a,const void* A,const void* B,std::size_t m,std::size_t n,std::size_t k,void* C);
    const LinalgReport& report() const;
    const LinalgOptions& options() const;
private:
    struct Impl; std::unique_ptr<Impl> impl;
};
// ---- Exact dot product on the CPU: RN(sum_k a[k*sa]*b[k*sb]) with one rounding (the GPU fallback) ----
namespace detail {
// Rounds sign * integer(w) * 2^scale to N words exactly as core.hpp pack() does, for any word count.
template<int N> Number<N> pack_words(const std::vector<word>& w,exponent_type scale,int sign,word status){
    int h=-1;for(std::size_t i=w.size();i-->0;)if(w[i]){h=int(32*i)+31-__builtin_clz(w[i]);break;}
    if(h<0)return zero<N>(status);
    exponent_type e=scale+h;if(e>1000000000||e< -1000000001)return zero<N>(status|exponent_overflow);
    auto bit=[&](long p)->word{return p<0||std::size_t(p/32)>=w.size()?0:(w[p/32]>>(p%32))&1;};
    auto chunk=[&](long p)->word{word x=0;for(int s=0;s<32;++s)x|=bit(p+s)<<s;return x;};
    Number<N> r=zero<N>(status);r.sign=sign;r.exponent=int(e);long shift=long(h)-(32*N-1);
    if(shift<=0){for(int i=0;i<N;++i)r.limb[i]=chunk(32L*i+shift);return checked(r);}
    for(int i=0;i<N;++i)r.limb[i]=chunk(shift+32L*i);
    bool round=bit(shift-1),below=false;
    for(long p=0;p<shift-1&&!below;)if(p%32==0&&p+32<=shift-1){below=w[p/32]!=0;p+=32;}else{below=bit(p);++p;}
    if(round&&(below||(r.limb[0]&1)))increment(r);
    return checked(r);
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
// Products are sorted by exponent and grouped into clusters; a new cluster starts when every remaining
// product lies more than bits+log2(K)+4 bits below the current cluster's lowest bit. Each cluster is summed
// exactly. With S the first nonzero cluster sum and F the rest, |F| < 2^(lsb(S)-bits-2), so
// RN(S+F) = RN(S + sign(F)*eps) and sign(F) is the sign of the next nonzero cluster sum.
template<int N> Number<N> exact_dot(const Number<N>* a,std::ptrdiff_t sa,const Number<N>* b,std::ptrdiff_t sb,std::size_t K){
    word status=0;for(std::size_t k=0;k<K;++k)status|=a[std::ptrdiff_t(k)*sa].status|b[std::ptrdiff_t(k)*sb].status;
    if(status)return zero<N>(status);
    struct Term{exponent_type scale;std::size_t k;};std::vector<Term> t;
    for(std::size_t k=0;k<K;++k){const auto &x=a[std::ptrdiff_t(k)*sa],&y=b[std::ptrdiff_t(k)*sb];
        if(x.sign&&y.sign)t.push_back({exponent_type(x.exponent)+y.exponent-2*(32*N-1),k});}
    std::sort(t.begin(),t.end(),[](const Term& u,const Term& v){return u.scale>v.scale;});
    int logk=0;while((std::size_t(1)<<logk)<K)++logk;
    const exponent_type gap=32*N+logk+4,width=64*N; // a product is < 2^(scale+64N)
    struct Sum{int sign;exponent_type scale;std::vector<word> mag;};std::vector<Sum> sums;
    for(std::size_t i=0;i<t.size();){
        std::size_t j=i+1;exponent_type low=t[i].scale;
        while(j<t.size()&&t[j].scale+width-1+gap>=low){low=std::min(low,t[j].scale);++j;}
        std::vector<word> acc(std::size_t((t[i].scale+width-low)/32+3),0);word p[64];
        for(std::size_t q=i;q<j;++q){const auto &x=a[std::ptrdiff_t(t[q].k)*sa],&y=b[std::ptrdiff_t(t[q].k)*sb];
            for(int u=0;u<2*N;++u)p[u]=0;
            for(int u=0;u<N;++u){dword c=0;for(int v=0;v<N;++v){dword z=dword(x.limb[u])*y.limb[v]+p[u+v]+c;p[u+v]=word(z);c=z>>32;}p[u+N]=word(c);}
            detail::accumulate(acc,p,2*N,long(t[q].scale-low),x.sign*y.sign);}
        int sign=1;if(acc.back()>>31){sign=-1;dword c=1;for(auto& v:acc){dword z=dword(word(~v))+c;v=word(z);c=z>>32;}}
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
}
