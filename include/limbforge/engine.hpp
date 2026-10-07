#pragma once
#include "core.hpp"
#include <cstddef>
#include <memory>
#include <string>
namespace limbforge {
enum class Operation { add, sub, mul, div, complex_add, complex_mul, complex_div };
struct Timing { double gpu_seconds, wall_seconds; };
// One Engine per host thread. Buffers and specialized pipelines are reused.
class Engine {
public:
    Engine(); ~Engine();
    Engine(const Engine&)=delete; Engine& operator=(const Engine&)=delete;
    std::string device_name() const;
    // Bits must be a multiple of 32 in [64,1024]. Binary struct layouts above.
    Timing run(int bits,Operation op,const void* a,const void* b,void* out,std::size_t count);
    // Four seeds / instance. Adjacent states_per_weight instances share weights;
    // count must be divisible by states_per_weight. Layout is seeds[j*count+i],
    // weights[(step*4+j)*(count/states_per_weight)+i/states_per_weight].
    // Output: final value (fourth seed if steps=0).
    // Each GPU thread keeps its complete four-value ring through every step.
    Timing recurrence(int bits,const void* seeds,const void* weights,void* out,std::size_t count,unsigned steps,unsigned states_per_weight=1);
private:
    struct Impl; std::unique_ptr<Impl> impl;
};
}
