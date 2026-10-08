#pragma once
// Element-wise elementary functions (plan D7): real exp, expm1, log, log1p, sin, cos, atan2 and complex exp, log,
// correctly rounded (round to nearest, ties to even; each complex component rounded once from its exact value), and
// complex integer powers z^k by a documented sequence of fused products (not correctly rounded). Contract, error
// analysis and retry strategy: docs/numerics.md, "Transcendental functions".
//
// The GPU evaluates each element at W >= N+2 words (>= 64 guard bits) with a rigorous error bound and rounds only when the
// bound decides the result (Ziv); undecided elements are appended to a compacted retry list and re-evaluated on the
// host by the same algorithm at more words (ladder of widths up to 136 words). Host arrays only: element = Float<bits>
// (Complex<bits/32> for complex functions); out may alias the input. Statuses propagate (input status -> zero payload
// with that status in every output component); domain errors give `invalid`, results outside the exponent range
// `exponent_overflow`.
//
// Resident use (not implemented): a CommandBatch encoder would bind its Buffer<T> operands to the same kernels
// (`lf_unary`, `lf_atan2`, `lf_complex`, `lf_powi`) plus this unit's per-width table buffer, append undecided indices to a
// device counter/list, and resolve the list on the host after the Submission completes (the retry needs the inputs, so
// an in-place operation would keep a copy of undecided inputs or a staging input buffer).
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
};
struct TranscendentalOptions { unsigned host_threads=0; unsigned threads_per_threadgroup=64; };
// One Transcendentals per host thread (buffers and compiled kernels are reused). Calls return after the GPU finished.
class Transcendentals {
public:
    explicit Transcendentals(TranscendentalOptions options={}); ~Transcendentals();
    Transcendentals(const Transcendentals&)=delete; Transcendentals& operator=(const Transcendentals&)=delete;
    std::string device_name() const;
    // Real unary functions (exp, expm1, log, log1p, sin, cos) on Float<bits>[count], and atan2(a = y, b = x).
    // Complex exp/log on Complex<bits/32>[count] (b unused).
    Timing run(int bits,Function f,const void* a,void* out,std::size_t count,const void* b=nullptr);
    // z^k for Complex<bits/32> z[count]; k[i] (k_count == count) or k[0] for all (k_count == 1).
    Timing powi(int bits,const void* z,const std::int32_t* k,std::size_t k_count,void* out,std::size_t count);
    // Compiles the kernel of f at bits (first use otherwise compiles during the call); returns seconds spent.
    double prewarm(int bits,Function f);
    const TranscendentalReport& report() const;
private:
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
