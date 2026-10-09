#pragma once
#include "limbforge/engine.hpp"
#include <memory>
namespace wolfnum::experimental {
// Fixed row-major 4x4 matrices; input stride 0 broadcasts. Strides count elements.
struct ExactGemm4Shape {std::size_t count=1,stride_a=16,stride_b=16,stride_c=16;bool accumulate=false,negative=false;};
struct ExactGemm4Report {std::size_t gpu_components=0,host_components=0;bool provisional_reads=false;};
struct ExactGemm4State;
class ExactGemm4Ticket {
    std::shared_ptr<ExactGemm4State> state_;friend class ExactGemm4;
    explicit ExactGemm4Ticket(std::shared_ptr<ExactGemm4State> s):state_(std::move(s)){}
public:
    bool resolved()const;
    ExactGemm4Report report()const;
};
// Each real component is RN(C_old + sum of exact signed products), then optional
// exact sign negation. C_old is omitted without accumulation. Complex components
// sum eight real products, with no intermediate rounding. Statuses are ORed per
// component across all its operands. Wide exponent spreads are repaired at wait.
// Outputs are provisional within their batch; wait before consuming them.
class ExactGemm4 {
    limbforge::Engine* engine_;struct Impl;std::unique_ptr<Impl> impl_;
    ExactGemm4Ticket encode(limbforge::CommandBatch&,int,bool,const ExactGemm4Shape&,limbforge::detail::Operand,limbforge::detail::Operand,limbforge::detail::Operand);
public:
    explicit ExactGemm4(limbforge::Engine&);
    ~ExactGemm4();
    ExactGemm4(const ExactGemm4&)=delete;ExactGemm4& operator=(const ExactGemm4&)=delete;
    template<class T> ExactGemm4Ticket gemm(limbforge::CommandBatch& b,const ExactGemm4Shape& s,const limbforge::Buffer<T>& a,const limbforge::Buffer<T>& bb,limbforge::Buffer<T>& c){
        static_assert(limbforge::detail::Format<T>::bits!=0,"arithmetic elements required");
        return encode(b,limbforge::detail::Format<T>::bits,limbforge::detail::Format<T>::complex,s,limbforge::detail::Access::operand(a),limbforge::detail::Access::operand(bb),limbforge::detail::Access::operand(c));}
    limbforge::Timing gemm(int bits,bool complex,const ExactGemm4Shape&,const void* a,const void* b,void* c,ExactGemm4Report* report=nullptr);
};
}
