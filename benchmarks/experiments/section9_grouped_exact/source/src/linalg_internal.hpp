#pragma once
// Private hooks into Linalg's internals for src/linalg_complex.mm (complex QR, round 43). Implemented at the end of
// src/linalg.mm, where Linalg::Impl is complete; nothing here is public API.
#include "limbforge/linalg.hpp"
#include <functional>
#include <atomic>
namespace limbforge {
namespace detail {
struct GroupedSyrkPass {
    std::atomic<bool> done{false};bool provisional_reads=false;std::size_t gpu_outputs=0;
};
struct LinalgHooks {
    // Private fast path: freshly host-packed single-band A/C only. Null means no work encoded; use ordinary exact SYRK.
    static std::shared_ptr<GroupedSyrkPass> syrk_group(Linalg&,CommandBatch&,int bits,const Operand& A,const Operand& C,std::size_t trials,std::size_t K,std::size_t M);
    static void require_device(Linalg& la,const Engine& engine);
    // Starts a factorization or solve on the calling thread (its worker pool); returns the resident-list mark for leave().
    static std::size_t enter(Linalg& la);
    static void leave(Linalg& la,std::size_t mark);                       // drops the buffers made resident since enter()
    static void* scratch(Linalg& la,const char* role,std::size_t bytes);  // a pooled shared buffer, resident until leave()
    static void resident(Linalg& la,void* page_aligned,std::size_t bytes); // caller memory used by the GPU in place until leave()
    static unsigned lanes(Linalg& la);                                      // workers of the calling thread's pool
    static void each(Linalg& la,std::size_t n,const std::function<void(std::size_t)>& f);
    static void run(Linalg& la,std::size_t n,const std::function<void(std::size_t,unsigned)>& f); // f(item, worker < lanes())
    static void side(bool on);  // marks the calling thread as a factorization's second (trailing-update) thread
    // Block b of a real compact-WY factor (m x n reflectors V with stride n, blocks nb, T blocks of nb x nb) applied to columns
    // [c0, c1) of X (row stride ld): Linalg::Impl::qr_block with qt (Q^T) or not (Q). V and T must be resident. have_y: Yb already
    // holds Y (pivoted factorizations form it during the panel) and only X = D(X; V_b, Y) is applied.
    static Timing qr_block(Linalg& la,int bits,std::size_t m,std::size_t n,std::size_t nb,const void* V,const void* T,std::size_t b,void* X,std::size_t ld,
                           std::size_t c0,std::size_t c1,bool qt,void* Wb,void* Yb,double threshold,bool gpu,bool& on_gpu,bool have_y=false);
    // X (n x nrhs) solves R X = B for upper triangular R (n x n): the backward substitution of QRFactor::solve (blocks fo.block).
    static Timing solve_upper(Linalg& la,int bits,const void* R,std::size_t n,const void* B,std::size_t nrhs,void* X,const FactorOptions& fo);
};
}
}
