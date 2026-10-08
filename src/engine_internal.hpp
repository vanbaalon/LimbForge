#pragma once
// Private hook for library units (ObjC++ .mm files of this library only; not installed, not public API): the Metal objects
// behind Engine, CommandBatch and Buffer<T>, so that separate units (numerics.mm, transcendental.mm) encode their own
// pipelines into an Engine's CommandBatch with the engine's ownership rules (docs/execution.md, "Resident units").
// Pipelines must be compiled on Internal::device(engine) (the batch's device); dispatches use Internal::compute(batch).
#import <Metal/Metal.h>
#include "limbforge/engine.hpp"
#include <atomic>
#include <functional>
#include <memory>
namespace limbforge {
namespace detail {
struct BufferStorage {
    id<MTLBuffer> buffer;std::shared_ptr<int> owner;std::atomic<bool> busy{false};std::size_t bytes;
};
// Set when an operation encoded later in the same batch reads a provisional output (Internal::provisional).
using ProvisionalFlag=std::shared_ptr<std::atomic<bool>>;
struct Internal {
    static id<MTLDevice> device(const Engine&);
    static id<MTLCommandQueue> queue(const Engine&);
    static id<MTLDevice> device(const CommandBatch&);        // throws logic_error once submitted or moved
    // The batch's command buffer: identifies the batch while it, its submission or a completion step of it lives (units share
    // per-batch workspace between their calls in one batch, whose dispatches run in order).
    static id<MTLCommandBuffer> command(CommandBatch&);
    // The batch's compute encoder: created on first use (or after a copy), otherwise preceded by a buffer barrier, so each
    // dispatch sees every earlier write of the batch. Call once per dispatch.
    static id<MTLComputeCommandEncoder> compute(CommandBatch&);
    // In-batch copy (a blit pass; Metal orders it after earlier and before later passes of the batch). Retains both buffers.
    static void copy(CommandBatch&,const std::shared_ptr<BufferStorage>& from,const std::shared_ptr<BufferStorage>& to,std::size_t bytes);
    // Ownership as for engine operations (CommandBatch::Impl::retain): rejects empty, foreign and busy buffers; written
    // buffers must not hold provisional results of this batch; reading one sets its ProvisionalFlag.
    static void retain(CommandBatch&,const std::shared_ptr<BufferStorage>&,bool written=false);
    // Reject an input whose final value needs host completion. For passes without a repair path.
    static void require_final(CommandBatch&,const std::shared_ptr<BufferStorage>&);
    // Fresh shared storage of the batch's engine (>= 1 byte), retained until the submission is waited. The host may write it
    // before submission (the GPU has not seen it yet).
    static std::shared_ptr<BufferStorage> scratch(CommandBatch&,std::size_t bytes);
    // Keeps a Metal object (pipeline, unit-owned table) alive until the submission completes.
    static void keep(CommandBatch&,id object);
    // Marks a buffer written by this batch as provisional until completion: later writes in the batch are rejected and
    // later reads set the returned flag.
    static ProvisionalFlag provisional(CommandBatch&,const std::shared_ptr<BufferStorage>&);
    // Host step run by Submission::wait (or the submission's destructor) after the GPU completed without error and before
    // the batch's buffers are released; it may read and write their contents. An exception fails wait().
    static void on_completion(CommandBatch&,std::function<void()>);
};
}
}
