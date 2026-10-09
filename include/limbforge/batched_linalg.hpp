#pragma once
#include "engine.hpp"
#include "linalg.hpp"
#include <functional>
namespace limbforge {
// Row-major C_b = A_b B_b. Strides are in elements; zero input stride broadcasts one matrix.
// A: m*k, B: k*n, C: m*n. Output matrices must not overlap. Accumulation starts from C;
// otherwise from zero. k is visited in ascending order; composed mul+add, or explicit fma.
// This sequence differs from Linalg's single-round exact dot contract.
struct StridedGemm {
    std::size_t count=1,m=0,n=0,k=0,stride_a=0,stride_b=0,stride_c=0;
    bool accumulate=false,fused=false,negative=false;
};
// Same numerical sequence in both modes; compact bounds auxiliary power scratch to
// count*steps*(1+min(nmax+1,8)) elements, at the cost of additional dispatches.
enum class PowerStorage { full_table, compact };
struct PowerMoments {
    std::size_t count=0; unsigned steps=0,nmax=0,ncols=0; int n0=0;
    bool accumulate=false,fused=false;
};
// p_a = Ep_a * Horner(cp_a,y), q_a = Eq_a * Horner(cq_a,y); v <- v+p*(q^T*v).
// Coefficients in ascending degree: [coefficient_sets][4][terms], sets=1 or weight_groups.
// y: [steps][weight_groups], Ep/Eq: [steps][4][weight_groups]. Start: [4][lanes].
// Output: [4][lanes], or [steps+1][4][lanes] in application order for all_steps.
struct PolynomialRecurrence {
    std::size_t lanes=0,coefficient_sets=1;
    unsigned steps=0,terms=0,lanes_per_weight=1;
    bool all_steps=false,reverse=false,fused=false;
};
// All-device Cholesky sequence: ascending column k, rounded square/mul and sub updates,
// rounded sqrt/div. Different from Linalg's blocked exact updates. Status: 0 or failed pivot+1.
struct CholeskyTrials {std::size_t n=0,count=0;};
// Metadata for an external, page-aligned and page-sized allocation with inline significands.
// Offsets address little-endian 64-bit MPFR limbs; precision must already equal the GPU width.
// The caller keeps the allocation alive and immutable until submission completion.
struct InlineComplexRecord {
    std::uint64_t real_offset=0,imag_offset=0;
    std::int32_t real_exponent=0,imag_exponent=0,real_sign=0,imag_sign=0;
    std::uint32_t real_status=0,imag_status=0;
};
class BatchedLinalg {
    Engine* engine_; struct Impl; std::unique_ptr<Impl> impl_;
    void encode_gemm(CommandBatch&,int,bool,const StridedGemm&,detail::Operand,detail::Operand,detail::Operand);
    void encode_power(CommandBatch&,int,bool,const PowerMoments&,detail::Operand,detail::Operand,detail::Operand,detail::Operand,PowerStorage);
    void encode_normal(CommandBatch&,int,std::size_t,std::size_t,detail::Operand,detail::Operand,detail::Operand,detail::Operand,bool);
    void encode_trials(CommandBatch&,int,const CholeskyTrials&,detail::Operand,detail::Operand,detail::Operand,detail::Operand,detail::Operand);
    void encode_solve(CommandBatch&,int,std::size_t,std::size_t,std::size_t,detail::Operand,detail::Operand,detail::Operand,detail::Operand);
    void encode_inline(CommandBatch&,int,const void*,std::size_t,const InlineComplexRecord*,std::size_t,detail::Operand);
    void encode_augment(CommandBatch&,int,std::size_t,std::size_t,detail::Operand,detail::Operand,detail::Operand);
    void encode_extract(CommandBatch&,int,std::size_t,detail::Operand,detail::Operand,detail::Operand,const LinalgTicket&);
    void encode_polynomial(CommandBatch&,int,const PolynomialRecurrence&,detail::Operand,detail::Operand,detail::Operand,detail::Operand,detail::Operand,detail::Operand,detail::Operand);
public:
    explicit BatchedLinalg(Engine&); ~BatchedLinalg();
    BatchedLinalg(const BatchedLinalg&)=delete; BatchedLinalg& operator=(const BatchedLinalg&)=delete;
    // Synchronous host-array forms, including input staging and output readback. Same kernels/contracts.
    Timing gemm(int bits,bool complex,const StridedGemm&,const void* A,const void* B,void* C);
    Timing power_moments(int bits,bool complex,const PowerMoments&,const void* E,const void* y,const void* W,void* out);
    Timing power_moments(int bits,bool complex,const PowerMoments&,const void* E,const void* y,const void* W,void* out,PowerStorage);
    template<class T> void polynomial_recurrence(CommandBatch& b,const PolynomialRecurrence& s,const Buffer<T>& start,const Buffer<T>& cp,const Buffer<T>& cq,const Buffer<T>& y,const Buffer<T>& Ep,const Buffer<T>& Eq,Buffer<T>& out){
        static_assert(detail::Format<T>::complex,"polynomial recurrence uses complex buffers");
        encode_polynomial(b,detail::Format<T>::bits,s,detail::Access::operand(start),detail::Access::operand(cp),detail::Access::operand(cq),detail::Access::operand(y),detail::Access::operand(Ep),detail::Access::operand(Eq),detail::Access::operand(out));}
    template<class T> void gemm(CommandBatch& b,const StridedGemm& s,const Buffer<T>& A,const Buffer<T>& B,Buffer<T>& C){
        encode_gemm(b,detail::Format<T>::bits,detail::Format<T>::complex,s,detail::Access::operand(A),detail::Access::operand(B),detail::Access::operand(C));}
    // E,y: [count][steps], W: [count][steps][ncols], out: [count][nmax+1][ncols].
    // Powers start at powi(y,n0), then multiply by y for each ascending n; E is multiplied afterwards.
    template<class T> void power_moments(CommandBatch& b,const PowerMoments& s,const Buffer<T>& E,const Buffer<T>& y,const Buffer<T>& W,Buffer<T>& out){
        power_moments(b,s,E,y,W,out,PowerStorage::full_table);}
    template<class T> void power_moments(CommandBatch& b,const PowerMoments& s,const Buffer<T>& E,const Buffer<T>& y,const Buffer<T>& W,Buffer<T>& out,PowerStorage storage){
        encode_power(b,detail::Format<T>::bits,detail::Format<T>::complex,s,detail::Access::operand(E),detail::Access::operand(y),detail::Access::operand(W),detail::Access::operand(out),storage);}
    // General two-stage matrix product. Intermediate shape: [first.count][first.m][first.n].
    // first/second accumulation must be false; strides of the intermediate are supplied internally.
    template<class T> void product3(CommandBatch& b,StridedGemm first,StridedGemm second,const Buffer<T>& A,const Buffer<T>& B,const Buffer<T>& D,Buffer<T>& C,bool negative=false){
        if(first.count!=second.count||first.m!=second.m||first.n!=second.k||first.accumulate||second.accumulate)
            throw std::invalid_argument("product3: incompatible shapes or accumulation");
        if(first.m&&first.n>std::size_t(-1)/first.m)throw std::invalid_argument("product3: size overflow");
        first.stride_c=first.m*first.n;second.stride_a=first.stride_c;
        if(first.count&&first.stride_c>std::size_t(-1)/first.count)throw std::invalid_argument("product3: size overflow");
        auto tmp=engine_->make_buffer<T>(first.count*first.stride_c);second.negative=negative;gemm(b,first,A,B,tmp);gemm(b,second,tmp,D,C);
    }
    // A = J^T J (full symmetric), rhs = J^T g in the same dispatch. Sequential composed/fused dots.
    template<class T> void normal_equations(CommandBatch& b,const Buffer<T>& J,const Buffer<T>& g,std::size_t rows,std::size_t cols,Buffer<T>& A,Buffer<T>& rhs,bool fused=false){
        static_assert(!detail::Format<T>::complex,"normal equations use real buffers");
        encode_normal(b,detail::Format<T>::bits,rows,cols,detail::Access::operand(J),detail::Access::operand(g),detail::Access::operand(A),detail::Access::operand(rhs),fused);}
    // TensorOps exact-dot path: one augmented SYRK of [J g]. Same bits as Linalg::syrk/gemm.
    // Like resident Linalg products, A/rhs are final after wait() (host fallback may run then).
    // Do not feed them to a dependent device pass in the same batch. Use normal_equations for that.
    template<class T> LinalgTicket normal_equations_exact(Linalg& la,CommandBatch& b,const Buffer<T>& J,const Buffer<T>& g,std::size_t rows,std::size_t cols,Buffer<T>& A,Buffer<T>& rhs){
        static_assert(!detail::Format<T>::complex,"normal equations use real buffers");
        if(cols==std::size_t(-1)||((cols+1)&&rows>std::size_t(-1)/(cols+1))||(cols+1)>std::size_t(-1)/(cols+1))throw std::invalid_argument("normal equations: size overflow");
        auto a=detail::Access::operand(A),r=detail::Access::operand(rhs),j=detail::Access::operand(J),gg=detail::Access::operand(g);
        if(a.size<cols*cols||r.size<cols||j.size<rows*cols||gg.size<rows)throw std::invalid_argument("normal equations: undersized buffer");
        if((a.storage&&(a.storage==j.storage||a.storage==gg.storage||a.storage==r.storage))||(r.storage&&(r.storage==j.storage||r.storage==gg.storage)))throw std::invalid_argument("normal equations: output alias");
        auto augmented=engine_->make_buffer<T>(rows*(cols+1)),gram=engine_->make_buffer<T>((cols+1)*(cols+1));
        encode_augment(b,detail::Format<T>::bits,rows,cols,detail::Access::operand(J),detail::Access::operand(g),detail::Access::operand(augmented));
        auto ticket=la.syrk(b,augmented,rows,cols+1,gram,false);encode_extract(b,detail::Format<T>::bits,cols,detail::Access::operand(gram),detail::Access::operand(A),detail::Access::operand(rhs),ticket);return ticket;
    }
    // Factor A + mu[t]*diag(D), lower triangle of A; L: [count][n][n], status: [count].
    // A, D and mu stay on device; all trials encoded together, no host panels or intermediate readbacks.
    template<class T> void cholesky_trials(CommandBatch& b,const CholeskyTrials& s,const Buffer<T>& A,const Buffer<T>& D,const Buffer<T>& mu,Buffer<T>& L,Buffer<std::uint32_t>& status){
        static_assert(!detail::Format<T>::complex,"Cholesky uses real buffers");
        encode_trials(b,detail::Format<T>::bits,s,detail::Access::operand(A),detail::Access::operand(D),detail::Access::operand(mu),detail::Access::operand(L),detail::Access::operand(status));}
    // L/status from cholesky_trials; shared B [n][nrhs], X [count][n][nrhs]. X must not alias inputs.
    template<class T> void cholesky_solve(CommandBatch& b,const Buffer<T>& L,const Buffer<std::uint32_t>& status,std::size_t count,std::size_t n,const Buffer<T>& B,std::size_t nrhs,Buffer<T>& X){
        static_assert(!detail::Format<T>::complex,"Cholesky uses real buffers");
        encode_solve(b,detail::Format<T>::bits,count,n,nrhs,detail::Access::operand(L),detail::Access::operand(status),detail::Access::operand(B),detail::Access::operand(X));}
    template<class T> void import_inline_complex(CommandBatch& b,const void* allocation,std::size_t bytes,const InlineComplexRecord* records,std::size_t count,Buffer<T>& out){
        static_assert(detail::Format<T>::complex,"inline complex staging needs complex output");
        encode_inline(b,detail::Format<T>::bits,allocation,bytes,records,count,detail::Access::operand(out));}
};
// Wait on already-submitted batches concurrently with host work. Errors drain every submission before rethrowing.
std::future<std::vector<Timing>> wait_all_async(std::vector<Submission> submissions);
}
