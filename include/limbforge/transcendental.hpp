#pragma once
// Element-wise elementary functions (plan D7): real exp, expm1, log, log1p, sin, cos, atan2 and complex exp, log,
// correctly rounded (round to nearest, ties to even; each complex component rounded once from its exact value), and
// complex integer powers z^k by a documented sequence of fused products (not correctly rounded). Contract, error
// analysis and retry strategy: docs/numerics.md, "Transcendental functions".
//
// The GPU evaluates each element at W >= N+2 words (>= 64 guard bits) with a rigorous error bound and rounds only when the
// bound decides the result (Ziv); undecided elements are appended to a compacted retry list and re-evaluated on the
// host by the same algorithm at more words (ladder of widths up to 136 words). Host arrays: element = Float<bits>
// (Complex<bits/32> for complex functions); out may alias the input. Statuses propagate (input status -> zero payload
// with that status in every output component); domain errors give `invalid`, results outside the exponent range
// `exponent_overflow`.
//
// Resident passes (docs/execution.md, "Resident units"): run/powi on Buffer operands encode the same GPU pass into an
// Engine's CommandBatch (after an in-batch snapshot of the inputs); the undecided elements land on a resident compacted
// list, and Submission::wait() runs the host retries and patches `out` before it releases the buffers. `out` is therefore
// final exactly when the submission is waited (bit-identical to the host-array call). Until then it is provisional:
// writing it again in the same batch throws; operations encoded later in the same batch that read it see provisional
// values for the undecided elements (the ticket reports `retried` and `provisional_reads`).
#include "core.hpp"
#include "engine.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
namespace limbforge {
enum class Function { exp, expm1, log, log1p, sin, cos, atan2, complex_exp, complex_log, complex_powi };
inline bool function_is_complex(Function f){return f==Function::complex_exp||f==Function::complex_log||f==Function::complex_powi;}
struct TranscendentalReport {
    std::size_t count=0;          // elements of the last call
    std::size_t retried=0;        // elements the GPU (or the first host width) could not certify
    std::size_t resolved[4]={};   // retried elements resolved at host rung 1, 2, 3 (widths >= N+4, 2N+4, 4N+8 words)
    std::size_t unresolved=0;     // still undecided at 136 words: faithful result (never observed; docs/numerics.md)
    double gpu_seconds=0,wall_seconds=0,retry_seconds=0,compile_seconds=0; // compile: pipelines/tables built in this call
    // Resident pass only: a later operation of the same batch read `out` (it saw provisional values if retried > 0).
    // gpu_seconds and wall_seconds are 0 for a resident pass (the Submission's Timing covers the whole batch).
    bool provisional_reads=false;
};
namespace detail { struct TranscendentalPass; }
// Handle of a resident pass. Copies share state; the pass resolves whether or not a ticket is kept.
class TranscendentalTicket {
    std::shared_ptr<detail::TranscendentalPass> pass_;
    explicit TranscendentalTicket(std::shared_ptr<detail::TranscendentalPass> p):pass_(std::move(p)){}
    friend class Transcendentals;
public:
    TranscendentalTicket()=default;
    bool resolved() const;                       // the submission was waited and the retries ran (out is final)
    const TranscendentalReport& report() const;  // throws std::logic_error before resolved()
};
struct TranscendentalOptions { unsigned host_threads=0; unsigned threads_per_threadgroup=64; };
// One Transcendentals per host thread (buffers and compiled kernels are reused). Calls return after the GPU finished.
class Transcendentals {
public:
    explicit Transcendentals(TranscendentalOptions options={}); ~Transcendentals();
    // Compiles on the engine's Metal device (required for the resident passes below).
    explicit Transcendentals(Engine& engine,TranscendentalOptions options={});
    Transcendentals(const Transcendentals&)=delete; Transcendentals& operator=(const Transcendentals&)=delete;
    std::string device_name() const;
    // Real unary functions (exp, expm1, log, log1p, sin, cos) on Float<bits>[count], and atan2(a = y, b = x).
    // Complex exp/log on Complex<bits/32>[count] (b unused).
    Timing run(int bits,Function f,const void* a,void* out,std::size_t count,const void* b=nullptr);
    // z^k for Complex<bits/32> z[count]; k[i] (k_count == count) or k[0] for all (k_count == 1).
    Timing powi(int bits,const void* z,const std::int32_t* k,std::size_t k_count,void* out,std::size_t count);
    // Compiles the kernel of f at bits (first use otherwise compiles during the call); returns seconds spent.
    double prewarm(int bits,Function f);
    const TranscendentalReport& report() const; // of the last host-array call

    // ---- Resident passes (contract above; docs/execution.md, "Resident units") ----
    // T = Float<bits> for real functions, Complex<bits/32> for complex exp/log; a.size() == out.size(); out may alias a.
    // Buffers belong to the batch's engine; the unit's device must be the batch's. The batch, its submission and the
    // ticket may outlive this object. Memory: a snapshot of the inputs and a 4(count+1)-byte list per pass, released at wait.
    template<class T> TranscendentalTicket run(CommandBatch& batch,Function f,const Buffer<T>& a,Buffer<T>& out){
        static_assert(detail::Format<T>::bits,"transcendentals use arithmetic buffers");
        return encode(batch,detail::Format<T>::bits,detail::Format<T>::complex,f,detail::Access::operand(a),{},{},detail::Access::operand(out));}
    // atan2(y = a, x = b) on Float<bits> buffers of equal size.
    template<class T> TranscendentalTicket run(CommandBatch& batch,Function f,const Buffer<T>& a,const Buffer<T>& b,Buffer<T>& out){
        static_assert(detail::Format<T>::bits,"transcendentals use arithmetic buffers");
        return encode(batch,detail::Format<T>::bits,detail::Format<T>::complex,f,detail::Access::operand(a),detail::Access::operand(b),{},detail::Access::operand(out));}
    // z^k, k.size() == z.size() or 1 (no retries: out is final when the submission completes).
    template<int N> TranscendentalTicket powi(CommandBatch& batch,const Buffer<Complex<N>>& z,const Buffer<std::int32_t>& k,Buffer<Complex<N>>& out){
        return encode(batch,32*N,true,Function::complex_powi,detail::Access::operand(z),{},detail::Access::operand(k),detail::Access::operand(out));}
private:
    TranscendentalTicket encode(CommandBatch&,int bits,bool complex,Function f,const detail::Operand& a,const detail::Operand& b,const detail::Operand& k,const detail::Operand& out);
    struct Impl; std::unique_ptr<Impl> impl;
};
// The same functions on the CPU (identical results; host threads, 0 = hardware concurrency). The first evaluation uses
// the smallest ladder width >= N+2 words, then the retry ladder. For complex_powi, b is the int32 k array (k_count = count).
void transcendental_cpu(int bits,Function f,const void* a,void* out,std::size_t count,const void* b=nullptr,unsigned threads=0,TranscendentalReport* report=nullptr);
namespace detail {
// Test hooks: the constant tables for a working width W (layout transcendental::Tables<W> of src/transcendental_core.hpp)
// and the bits of 2/pi (MSB first). Built on first use, thread safe.
const void* transcendental_tables(int W);
const std::uint32_t* two_over_pi_bits();
}
}
