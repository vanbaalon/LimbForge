#pragma once
#include "core.hpp"
#include <cstddef>
#include <memory>
#include <string>
#include <stdexcept>
#define LIMBFORGE_RESIDENT_API 1
namespace limbforge {
enum class Operation { add, sub, mul, div, complex_add, complex_mul, complex_div, square };
inline bool operation_is_complex(Operation op){return int(op)>=4&&int(op)<=6;}
inline bool operation_is_unary(Operation op){return op==Operation::square;}
inline bool operation_is_valid(Operation op){return int(op)>=0&&int(op)<=7;}
struct Timing { double gpu_seconds, wall_seconds; };
struct EngineOptions { unsigned threads_per_threadgroup=0; }; // 0 selects the default policy.
struct PipelineInfo { unsigned simd_width,max_threads,threads_per_threadgroup; };
namespace detail {
struct BufferStorage;
void* mapped(const std::shared_ptr<BufferStorage>&);
void transfer(const std::shared_ptr<BufferStorage>&,void*,std::size_t,bool);
template<class T> struct Format;
template<int N> struct Format<Number<N>> {static_assert(N>=2&&N<=32,"unsupported precision");static constexpr int bits=32*N;static constexpr bool complex=false;};
template<int N> struct Format<Complex<N>> {static_assert(N>=2&&N<=32,"unsupported precision");static constexpr int bits=32*N;static constexpr bool complex=true;};
}
template<class T> class Buffer {
    std::shared_ptr<detail::BufferStorage> storage_;std::size_t count_=0;
    Buffer(std::shared_ptr<detail::BufferStorage> storage,std::size_t count):storage_(std::move(storage)),count_(count){}
    friend class Engine;friend class CommandBatch;
public:
    Buffer()=default;
    std::size_t size()const{return count_;}
    // Mapping/upload/download are rejected until the owning submission is waited.
    T* mapped(){return static_cast<T*>(detail::mapped(storage_));}
    const T* mapped()const{return static_cast<const T*>(detail::mapped(storage_));}
    void upload(const T* source,std::size_t count){if(count>count_)throw std::invalid_argument("upload exceeds buffer");detail::transfer(storage_,const_cast<T*>(source),count*sizeof(T),true);}
    void download(T* target,std::size_t count)const{if(count>count_)throw std::invalid_argument("download exceeds buffer");detail::transfer(storage_,target,count*sizeof(T),false);}
};
class Submission {
    struct Impl;std::shared_ptr<Impl> impl_;
    explicit Submission(std::shared_ptr<Impl> impl):impl_(std::move(impl)){}
    friend class CommandBatch;
public:
    Submission()=default;
    Submission(const Submission&)=delete;Submission& operator=(const Submission&)=delete;
    Submission(Submission&&)=default;Submission& operator=(Submission&&)=default;
    bool ready()const;
    Timing wait(); // Repeated waits return the same timing. Destruction waits safely.
};
class CommandBatch {
    struct Impl;std::unique_ptr<Impl> impl_;
    explicit CommandBatch(std::unique_ptr<Impl> impl);
    void encode(int,bool,Operation,const std::shared_ptr<detail::BufferStorage>&,
                const std::shared_ptr<detail::BufferStorage>&,const std::shared_ptr<detail::BufferStorage>&,std::size_t);
    friend class Engine;
public:
    ~CommandBatch();CommandBatch(CommandBatch&&)noexcept;CommandBatch& operator=(CommandBatch&&)noexcept;
    template<class T> void run(Operation op,const Buffer<T>& a,const Buffer<T>& b,Buffer<T>& out){
        if(operation_is_unary(op))throw std::invalid_argument("use the unary run overload");
        if(a.size()!=b.size()||a.size()!=out.size())throw std::invalid_argument("buffer counts must match");
        encode(detail::Format<T>::bits,detail::Format<T>::complex,op,a.storage_,b.storage_,out.storage_,a.size());
    }
    template<class T> void run(Operation op,const Buffer<T>& a,Buffer<T>& out){
        if(!operation_is_unary(op))throw std::invalid_argument("operation requires two inputs");
        if(a.size()!=out.size())throw std::invalid_argument("buffer counts must match");
        encode(detail::Format<T>::bits,detail::Format<T>::complex,op,a.storage_,a.storage_,out.storage_,a.size());
    }
    Submission submit(); // Single use; explicit barriers order dependent dispatches.
};
// One Engine per host thread. Buffers and specialized pipelines are reused.
class Engine {
public:
    explicit Engine(EngineOptions options={}); ~Engine();
    Engine(const Engine&)=delete; Engine& operator=(const Engine&)=delete;
    std::string device_name() const;
    PipelineInfo pipeline_info(int bits,Operation op);
    template<class T> Buffer<T> make_buffer(std::size_t count){
        static_assert(sizeof(T)==std::size_t(detail::Format<T>::bits/8+12)*(detail::Format<T>::complex?2:1),"buffer layout mismatch");
        if(count>std::size_t(-1)/sizeof(T))throw std::invalid_argument("buffer size overflow");
        return Buffer<T>(allocate(count*sizeof(T)),count);
    }
    CommandBatch batch();
    // Bits must be a multiple of 32 in [64,1024]. Binary struct layouts above.
    Timing run(int bits,Operation op,const void* a,const void* b,void* out,std::size_t count);
    Timing run_unary(int bits,Operation op,const void* a,void* out,std::size_t count);
    // Four seeds / instance. Adjacent states_per_weight instances share weights;
    // count must be divisible by states_per_weight. Layout is seeds[j*count+i],
    // weights[(step*4+j)*(count/states_per_weight)+i/states_per_weight].
    // Output: final value (fourth seed if steps=0).
    // Each GPU thread keeps its complete four-value ring through every step.
    Timing recurrence(int bits,const void* seeds,const void* weights,void* out,std::size_t count,unsigned steps,unsigned states_per_weight=1);
private:
    friend class CommandBatch;
    std::shared_ptr<detail::BufferStorage> allocate(std::size_t bytes);
    struct Impl; std::shared_ptr<Impl> impl;
};
}
