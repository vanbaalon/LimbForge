#pragma once
#include "core.hpp"
#include <cstddef>
#include <memory>
#include <string>
#include <stdexcept>
#define LIMBFORGE_RESIDENT_API 1
namespace limbforge {
// fma/fms: RN(a*b +/- c); complex_fma/complex_fms: each component one rounding of its exact value.
enum class Operation { add, sub, mul, div, complex_add, complex_mul, complex_div, square, sqrt, fma, fms, complex_fma, complex_fms };
inline bool operation_is_complex(Operation op){return (int(op)>=4&&int(op)<=6)||op==Operation::complex_fma||op==Operation::complex_fms;}
inline bool operation_is_unary(Operation op){return op==Operation::square||op==Operation::sqrt;}
inline bool operation_is_ternary(Operation op){return int(op)>=9&&int(op)<=12;}
inline bool operation_is_valid(Operation op){return int(op)>=0&&int(op)<=12;}
struct Timing { double gpu_seconds, wall_seconds; };
// 0 selects the default policy. cooperative_recurrence spreads each trajectory over 16/32 SIMD lanes
// for small counts (bit-identical; docs/experiments.md).
struct EngineOptions { unsigned threads_per_threadgroup=0; bool cooperative_reductions=true; bool cooperative_recurrence=true; };
struct PipelineInfo { unsigned simd_width,max_threads,threads_per_threadgroup; };
// Shape and options of Engine::vector_recurrence (docs/numerics.md, "Vector recurrences").
// Complex arrays are component-major: element j of lane i is at [j*lanes + i].
struct VectorRecurrence {
    std::size_t lanes=0; unsigned steps=0;
    unsigned lanes_per_weight=1;  // adjacent lanes sharing p/q (or M)
    bool affine=false;            // add r[k] (per lane) after each step
    bool matrix=false;            // v <- M[k] v instead of v <- v + p[k] (q[k].v)
    bool all_steps=false;         // write the state before step 0 and after every step
    bool reverse=false;           // apply coefficient steps k = steps-1 .. 0
    bool tangent=false;           // v is a tangent: v <- v + p(q.v) + dp(q.b) + p(dq.b), b = base state
    unsigned lanes_per_base=1,lanes_per_tangent=1; // lanes sharing a base trajectory / dp,dq
    bool fused=false;             // multiply-adds: cfma (one rounding per component) or cadd(c, cmul(a,b))
};
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
    void encode(int,bool,Operation,const std::shared_ptr<detail::BufferStorage>&,const std::shared_ptr<detail::BufferStorage>&,
                const std::shared_ptr<detail::BufferStorage>&,const std::shared_ptr<detail::BufferStorage>&,std::size_t);
    void encode_tree_sum(int,bool,const std::shared_ptr<detail::BufferStorage>&,
                         const std::shared_ptr<detail::BufferStorage>&,std::size_t);
    friend class Engine;
public:
    ~CommandBatch();CommandBatch(CommandBatch&&)noexcept;CommandBatch& operator=(CommandBatch&&)noexcept;
    template<class T> void run(Operation op,const Buffer<T>& a,const Buffer<T>& b,Buffer<T>& out){
        if(operation_is_unary(op)||operation_is_ternary(op))throw std::invalid_argument("operation requires a different number of inputs");
        if(a.size()!=b.size()||a.size()!=out.size())throw std::invalid_argument("buffer counts must match");
        encode(detail::Format<T>::bits,detail::Format<T>::complex,op,a.storage_,b.storage_,a.storage_,out.storage_,a.size());
    }
    // fma/fms/complex_fma/complex_fms: out = a*b +/- c. Any operand may alias out.
    template<class T> void run(Operation op,const Buffer<T>& a,const Buffer<T>& b,const Buffer<T>& c,Buffer<T>& out){
        if(!operation_is_ternary(op))throw std::invalid_argument("operation does not take three inputs");
        if(a.size()!=b.size()||a.size()!=c.size()||a.size()!=out.size())throw std::invalid_argument("buffer counts must match");
        encode(detail::Format<T>::bits,detail::Format<T>::complex,op,a.storage_,b.storage_,c.storage_,out.storage_,a.size());
    }
    template<class T> void run(Operation op,const Buffer<T>& a,Buffer<T>& out){
        if(!operation_is_unary(op))throw std::invalid_argument("operation requires two inputs");
        if(a.size()!=out.size())throw std::invalid_argument("buffer counts must match");
        encode(detail::Format<T>::bits,detail::Format<T>::complex,op,a.storage_,a.storage_,a.storage_,out.storage_,a.size());
    }
    // Adjacent pairs are rounded at each tree level; an odd tail is copied.
    // The output has one element. An empty input produces canonical zero.
    template<class T> void tree_sum(const Buffer<T>& input,Buffer<T>& out){
        if(out.size()!=1)throw std::invalid_argument("tree_sum output must contain one element");
        encode_tree_sum(detail::Format<T>::bits,detail::Format<T>::complex,input.storage_,out.storage_,input.size());
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
    Timing run_ternary(int bits,Operation op,const void* a,const void* b,const void* c,void* out,std::size_t count);
    // Four seeds / instance. Adjacent states_per_weight instances share weights;
    // count must be divisible by states_per_weight. Layout is seeds[j*count+i],
    // weights[(step*4+j)*(count/states_per_weight)+i/states_per_weight].
    // Output: final value (fourth seed if steps=0).
    // Each GPU thread keeps its complete four-value ring through every step.
    Timing recurrence(int bits,const void* seeds,const void* weights,void* out,std::size_t count,unsigned steps,unsigned states_per_weight=1);
    // Four-component complex vector recurrence. Layouts (G = lanes/lanes_per_weight):
    //   start [4][lanes]; p,q [steps][4][G]; M [steps][4][4][G] (passed as p, q unused);
    //   r [steps][4][lanes]; out [4][lanes], or [steps+1][4][lanes] with all_steps;
    //   tangent: base [steps+1][4][lanes/lanes_per_base] (an all_steps base run, same direction),
    //   dp,dq [steps][4][lanes/lanes_per_tangent]. Coefficient index k follows the direction;
    //   output and base indices follow the sequence position.
    Timing vector_recurrence(int bits,const VectorRecurrence& shape,const void* start,const void* p,const void* q,const void* r,void* out,
                             const void* base=nullptr,const void* dp=nullptr,const void* dq=nullptr);
private:
    friend class CommandBatch;
    std::shared_ptr<detail::BufferStorage> allocate(std::size_t bytes);
    struct Impl; std::shared_ptr<Impl> impl;
};
}
