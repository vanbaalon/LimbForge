#pragma once
// Polynomial values and Taylor jets (plan S1) and norms / status summaries (plan S4): host arrays, with their own Metal
// queue and runtime-compiled library (src/numerics.metal), and resident versions encoded into an Engine's CommandBatch. Contracts and exact rounding sequences:
// docs/numerics.md, "Polynomial values and jets" and "Norms and status summaries". Elements are Float<bits>
// (Number<bits/32>, bits/8 + 12 bytes) or Complex<bits/32>; bits is a multiple of 32 in [64, 1024].
#include "core.hpp"
#include "engine.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
namespace limbforge {
// Shape of a batched polynomial evaluation. Coefficients are in ascending powers, one set of `terms` values per set:
// coeffs[set*terms + k] multiplies x^k. Point i uses set i / points_per_set (adjacent points share a set);
// sets = ceil(points / points_per_set). terms = degree + 1; terms = 0 is the zero polynomial.
struct Polynomial {
    std::size_t points=0,terms=0,points_per_set=1;
    bool complex=false; // Complex<bits/32> coefficients, points and results (otherwise Float<bits>)
    bool fused=false;   // each Horner step one fma/cfma (one rounding per component) instead of mul then add
};
// Segmented reductions: `count` segments of `length` entries, segment-major (entry i of segment s at [s*length + i]);
// one result per segment. complex: entries are Complex<bits/32>, results are always real Float<bits>.
struct Segments { std::size_t count=1,length=0; bool complex=false; };
constexpr std::uint32_t no_index=0xffffffffu;
// Per-segment summary. status: OR of the statuses of all entries (both components of a complex entry; for
// scaled_residual also the scale and a per-entry division_by_zero / exponent_overflow) and of the returned value;
// failing: entries with a nonzero entry status; first_failing: lowest such index (no_index if none);
// index: lowest index attaining the maximum (norm_inf, norm_max, scaled_residual), first_failing when the segment
// has a failing entry, no_index for norm2, summarize_status and empty segments.
struct NormInfo { std::uint32_t status=0,failing=0,first_failing=no_index,index=no_index; };
namespace detail { template<> struct Format<NormInfo> {static constexpr int bits=0;static constexpr bool complex=false;}; } // Buffer<NormInfo>
// One Numerics per host thread (compiled libraries, pipelines and scratch buffers are reused). Calls return after
// the GPU has finished. Page-aligned inputs and outputs are used by the GPU in place; others are copied.
class Numerics {
public:
    Numerics(); ~Numerics();
    explicit Numerics(Engine& engine); // compiles on the engine's Metal device (required for the resident calls below)
    Numerics(const Numerics&)=delete; Numerics& operator=(const Numerics&)=delete;
    std::string device_name() const;
    // values[i] = p_set(points[i]) by Horner's rule (one value per point).
    Timing poly_eval(int bits,const Polynomial& shape,const void* coeffs,const void* points,void* values);
    // jets[i*(order+1) + j] = Taylor coefficient j of p_set at points[i]: p, p', p''/2 for order <= 2,
    // by simultaneous Horner updating the highest order first. order 0 is poly_eval.
    Timing poly_eval_jet(int bits,const Polynomial& shape,unsigned order,const void* coeffs,const void* points,void* jets);
    // values[s] = max_i |x_i| (complex: the modulus sqrt(re^2 + im^2), rounded once at the end of a documented sequence).
    Timing norm_inf(int bits,const Segments& segments,const void* x,void* values,NormInfo* info=nullptr);
    // values[s] = max_i max(|re_i|, |im_i|) (componentwise; exact). Equal to norm_inf for real input.
    Timing norm_max(int bits,const Segments& segments,const void* x,void* values,NormInfo* info=nullptr);
    // values[s] = sqrt(sum_i |x_i|^2) with exponent scaling and the fixed adjacent-pair tree of tree_sum (not correctly rounded).
    Timing norm2(int bits,const Segments& segments,const void* x,void* values,NormInfo* info=nullptr);
    // values[s] = max_i |r_i| / |scale_i| for r real or complex and a real scale (Float<bits>, same segment layout);
    // 0/0 counts as 0, nonzero/0 is a failing entry with division_by_zero.
    Timing scaled_residual(int bits,const Segments& segments,const void* r,const void* scale,void* values,NormInfo* info=nullptr);
    // Status summary only (info.index = no_index).
    Timing summarize_status(int bits,const Segments& segments,const void* x,NormInfo* info);

    // ---- Resident versions (docs/execution.md, "Resident units") ----
    // Encode the same pipelines into an Engine's CommandBatch on Buffer operands of that engine (ownership, barriers and
    // scratch lifetime as for engine operations); results are bit-identical to the host-array calls once the Submission is
    // waited. T = Float<bits> or Complex<bits/32>; shape.complex / segments.complex must match T. The unit's device must
    // be the batch's (construct with Numerics(Engine&); the system default device also matches). The batch and its
    // submission may outlive this object. Buffers may be larger than the shape requires.
    template<class T> void poly_eval(CommandBatch& batch,const Polynomial& shape,const Buffer<T>& coeffs,const Buffer<T>& points,Buffer<T>& values){
        poly_eval_jet(batch,shape,0,coeffs,points,values);}
    template<class T> void poly_eval_jet(CommandBatch& batch,const Polynomial& shape,unsigned order,const Buffer<T>& coeffs,const Buffer<T>& points,Buffer<T>& jets){
        static_assert(detail::Format<T>::bits,"polynomials use arithmetic buffers");
        encode_poly(batch,detail::Format<T>::bits,detail::Format<T>::complex,shape,order,detail::Access::operand(coeffs),detail::Access::operand(points),detail::Access::operand(jets));}
    // values: Float<bits> per segment; info (optional): NormInfo per segment.
    template<class T> using Real=Number<detail::Format<T>::bits/32>;
    template<class T> void norm_inf(CommandBatch& b,const Segments& s,const Buffer<T>& x,Buffer<Real<T>>& values){norm(b,1,s,x,Buffer<Real<T>>(),&values,nullptr);}
    template<class T> void norm_inf(CommandBatch& b,const Segments& s,const Buffer<T>& x,Buffer<Real<T>>& values,Buffer<NormInfo>& info){norm(b,1,s,x,Buffer<Real<T>>(),&values,&info);}
    template<class T> void norm_max(CommandBatch& b,const Segments& s,const Buffer<T>& x,Buffer<Real<T>>& values){norm(b,2,s,x,Buffer<Real<T>>(),&values,nullptr);}
    template<class T> void norm_max(CommandBatch& b,const Segments& s,const Buffer<T>& x,Buffer<Real<T>>& values,Buffer<NormInfo>& info){norm(b,2,s,x,Buffer<Real<T>>(),&values,&info);}
    template<class T> void norm2(CommandBatch& b,const Segments& s,const Buffer<T>& x,Buffer<Real<T>>& values){norm(b,4,s,x,Buffer<Real<T>>(),&values,nullptr);}
    template<class T> void norm2(CommandBatch& b,const Segments& s,const Buffer<T>& x,Buffer<Real<T>>& values,Buffer<NormInfo>& info){norm(b,4,s,x,Buffer<Real<T>>(),&values,&info);}
    template<class T> void scaled_residual(CommandBatch& b,const Segments& s,const Buffer<T>& r,const Buffer<Real<T>>& scale,Buffer<Real<T>>& values){norm(b,5,s,r,scale,&values,nullptr);}
    template<class T> void scaled_residual(CommandBatch& b,const Segments& s,const Buffer<T>& r,const Buffer<Real<T>>& scale,Buffer<Real<T>>& values,Buffer<NormInfo>& info){
        norm(b,5,s,r,scale,&values,&info);}
    template<class T> void summarize_status(CommandBatch& b,const Segments& s,const Buffer<T>& x,Buffer<NormInfo>& info){norm(b,0,s,x,Buffer<Real<T>>(),static_cast<Buffer<Real<T>>*>(nullptr),&info);}
private:
    template<class T> void norm(CommandBatch& b,int kind,const Segments& s,const Buffer<T>& x,const Buffer<Real<T>>& scale,Buffer<Real<T>>* values,Buffer<NormInfo>* info){
        static_assert(detail::Format<T>::bits,"norms use arithmetic buffers");detail::Operand v,i;if(values)v=detail::Access::operand(*values);if(info)i=detail::Access::operand(*info);
        encode_norm(b,detail::Format<T>::bits,detail::Format<T>::complex,kind,s,detail::Access::operand(x),detail::Access::operand(scale),v,i);}
    void encode_poly(CommandBatch&,int bits,bool complex,const Polynomial&,unsigned order,const detail::Operand& coeffs,const detail::Operand& points,const detail::Operand& out);
    void encode_norm(CommandBatch&,int bits,bool complex,int kind,const Segments&,const detail::Operand& x,const detail::Operand& scale,const detail::Operand& values,const detail::Operand& info);
    struct Impl; std::unique_ptr<Impl> impl;
};
}
