#pragma once
// Element-wise elementary functions (plan D7): real exp, expm1, log, log1p, sin, cos, atan2 and complex exp, log,
// correctly rounded (round to nearest, ties to even; each complex component rounded once from its exact value), and
// complex integer powers z^k by a documented sequence of fused products (not correctly rounded). Contract, error
// analysis and retry strategy: docs/numerics.md, "Transcendental functions".
//
// The GPU evaluates each element at W >= N+2 words (>= 64 guard bits) with a rigorous error bound and rounds only when the
// bound decides the result (Ziv); undecided elements are appended to a compacted list and re-evaluated on the GPU by up to
// three retry rungs at validated widths >= N+4, 2N+4, 4N+8 words, capped at 35 words (wider rung kernels miscompile;
// docs/gpu-codegen.md section 11); each rung compacts its still undecided elements for the next. Elements left after the
// last GPU rung (rungs above the cap run on the host: rung 3 from 512 bits, rungs 2-3 from 992 bits) are finished on the
// host by the host ladder (up to 136 words). A rung costs 1-5 ms of GPU latency, so host-array calls send up to
// TranscendentalOptions::gpu_retry_threshold undecided elements straight to the host ladder and only more to the GPU
// rungs. Host arrays: element = Float<bits> (Complex<bits/32> for complex functions); out may alias the input. Statuses
// propagate (input status -> zero payload with that status in every output component); domain errors give `invalid`,
// results outside the exponent range `exponent_overflow`.
//
// Resident passes (docs/execution.md, "Resident units"): run/powi on Buffer operands encode the first pass and the GPU
// rungs into an Engine's CommandBatch. When the GPU rungs decide every element (report().host_retried() == 0: random
// inputs essentially always; hard cases needing more than 35 words, from 384 bits, go to the host), `out` is final in
// the batch: operations encoded later in the same batch read final values. Elements left for the host are finished by
// Submission::wait() before it releases the buffers; until then they hold the RN value of the last GPU rung's
// approximation (provisional; the ticket reports `provisional_reads`). Because that host step may still patch `out`,
// writing `out` again in the same batch throws. After wait() `out` is bit-identical to the host-array call.
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
    std::size_t retried=0;        // elements the first evaluation (GPU pass, or the first host width) could not certify
    // GPU calls: resolved[r] = elements resolved by GPU rung r+1 (r < 3; rung_words[r] working words) and resolved[3] = by
    // the host final step (also every retry of a host-array call at or below gpu_retry_threshold; rung_words are then 0).
    // transcendental_cpu: resolved[0..2] = host rungs 1-3 (rung_words), resolved[3] = 0.
    std::size_t resolved[4]={};
    std::size_t unresolved=0;     // still undecided after the last host rung (136 words at most): faithful result (never observed)
    int first_words=0;            // working words of the first evaluation (GPU first pass, or the first host width)
    int rung_words[3]={};         // working words of retry rungs 1-3 (GPU calls: 0 = the rung runs on the host, in the final step)
    double gpu_seconds=0,wall_seconds=0,retry_seconds=0,compile_seconds=0; // retry: host retries (GPU calls: the host final step);
                                  // compile: pipelines/tables built in this call (first pass and GPU rungs)
    // Resident pass only: in_batch_reads = a later operation of the same batch read `out`; provisional_reads = it did so while
    // some elements were left for the host (host_retried() > 0), i.e. it saw provisional values for those elements.
    // gpu_seconds and wall_seconds are 0 for a resident pass (the Submission's Timing covers the whole batch).
    bool in_batch_reads=false,provisional_reads=false;
    // GPU calls: elements finished on the host after the GPU rungs (0: the GPU decided everything; resident `out` was final in the batch).
    std::size_t host_retried() const {return resolved[3]+unresolved;}
};
namespace detail { struct TranscendentalPass; }
// Handle of a resident pass. Copies share state; the pass resolves whether or not a ticket is kept.
class TranscendentalTicket {
    std::shared_ptr<detail::TranscendentalPass> pass_;
    explicit TranscendentalTicket(std::shared_ptr<detail::TranscendentalPass> p):pass_(std::move(p)){}
    friend class Transcendentals;
public:
    TranscendentalTicket()=default;
    bool resolved() const;                       // the submission was waited and the host final step ran (report available)
    const TranscendentalReport& report() const;  // throws std::logic_error before resolved()
};
// gpu_retry_threshold (host-array calls only): when the first pass leaves at most this many undecided elements, the host
// ladder resolves them right after it (a GPU rung costs 1-5 ms of single-thread latency, the host ~10-200 us per element
// and thread); above it the GPU rungs run in a second command buffer. 0: the rungs always run on the GPU, in the same
// command buffer as the first pass (as resident passes, which ignore the threshold). Results are identical either way.
struct TranscendentalOptions { unsigned host_threads=0; unsigned threads_per_threadgroup=64; std::size_t gpu_retry_threshold=512; };
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
    // ticket may outlive this object. Memory per pass, released at wait: two 4*count-byte retry lists plus a 64-byte header,
    // and a count-element buffer for operands of elements left to the host (its pages are touched only by such elements).
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
// the smallest host ladder width >= N+2 words, then the host retry ladder. For complex_powi, b is the int32 k array (k_count = count).
void transcendental_cpu(int bits,Function f,const void* a,void* out,std::size_t count,const void* b=nullptr,unsigned threads=0,TranscendentalReport* report=nullptr);
namespace detail {
// Test hooks: the constant tables for a working width W in [4, 136] (layout transcendental::Tables<W> of
// src/transcendental_core.hpp) and the bits of 2/pi (MSB first). Built on first use, thread safe.
const void* transcendental_tables(int W);
const std::uint32_t* two_over_pi_bits();
// Test hook (validation of the retry rungs): GPU levels below `level` (0 = first pass, 1-3 = rungs) treat every element as
// undecided, so rung `level` evaluates all elements; a level beyond the last GPU rung sends everything to the host final
// step. Applies to calls and passes encoded afterwards by any Transcendentals object; 0 restores normal operation.
void transcendental_force_level(int level);
}
}
