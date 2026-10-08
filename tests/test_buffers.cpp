#include "reference.hpp"
#include <chrono>
#include <stdexcept>
using namespace limbforge;
void require(bool test,const char* message){if(!test)throw std::runtime_error(message);}
template<class Fn> void rejects(Fn f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}require(rejected,"invalid API use accepted");}
template<int Bits> void run(Engine& e){
    using F=Float<Bits>;constexpr std::size_t n=257;
    std::mt19937_64 rng(Bits);std::vector<F> a(n),b(n),out(n),expected(n);
    for(std::size_t i=0;i<n;++i){a[i]=reference::random_number<Bits>(rng,5);b[i]=reference::random_number<Bits>(rng,5);expected[i]=a[i];}
    auto x=e.make_buffer<F>(n),y=e.make_buffer<F>(n),z=e.make_buffer<F>(n);
    x.upload(a.data(),n);y.upload(b.data(),n);
    auto batch=e.batch();
    for(int step=0;step<16;++step){batch.run(Operation::mul,x,y,x);for(std::size_t i=0;i<n;++i)expected[i]=reference::real<Bits>(Operation::mul,expected[i],b[i]);}
    batch.run(Operation::sub,x,y,z);
    auto ticket=batch.submit();rejects([&]{x.mapped();});rejects([&]{auto next=e.batch();next.run(Operation::add,x,y,z);});
    auto t=ticket.wait();require(t.gpu_seconds>=0&&t.wall_seconds>0,"invalid timing");require(ticket.ready(),"completed ticket not ready");
    require(ticket.wait().wall_seconds==t.wall_seconds,"wait not idempotent");z.download(out.data(),n);
    for(std::size_t i=0;i<n;++i)require(reference::equal<Bits>(out[i],reference::real<Bits>(Operation::sub,expected[i],b[i])),"resident chain differs from MPFR");
    rejects([&]{batch.submit();});rejects([&]{batch.run(Operation::add,x,y,z);});
    rejects([&]{x.upload(a.data(),n+1);});rejects([&]{auto next=e.batch();next.run(Operation::complex_mul,x,y,z);});
    Engine other;auto foreign=other.make_buffer<F>(n);rejects([&]{auto next=e.batch();next.run(Operation::add,x,foreign,z);});
    auto short_buffer=e.make_buffer<F>(n-1);rejects([&]{auto next=e.batch();next.run(Operation::add,x,y,short_buffer);});
    x.upload(a.data(),n);auto unary=e.batch();unary.run(Operation::square,x,x);unary.submit().wait();x.download(out.data(),n);
    for(std::size_t i=0;i<n;++i)require(reference::equal<Bits>(out[i],reference::real<Bits>(Operation::square,a[i],b[i])),"resident square differs from MPFR");
    x.upload(a.data(),n);auto roots=e.batch();roots.run(Operation::sqrt,x,z);roots.submit().wait();z.download(out.data(),n);
    for(std::size_t i=0;i<n;++i)require(reference::equal<Bits>(out[i],reference::real<Bits>(Operation::sqrt,a[i],b[i])),"resident sqrt differs from MPFR");
    rejects([&]{auto bad=e.batch();bad.run(Operation::square,x,y,z);});
    rejects([&]{auto bad=e.batch();bad.run(Operation::mul,x,z);});
    // A ticket owns buffers and the command even after the batch and engine die.
    Submission pending;Buffer<F> surviving;
    {Engine temporary;auto lhs=temporary.make_buffer<F>(n),rhs=temporary.make_buffer<F>(n);surviving=temporary.make_buffer<F>(n);
        lhs.upload(a.data(),n);rhs.upload(b.data(),n);auto next=temporary.batch();next.run(Operation::add,lhs,rhs,surviving);pending=next.submit();}
    pending.wait();surviving.download(out.data(),n);
    for(std::size_t i=0;i<n;++i)require(reference::equal<Bits>(out[i],reference::real<Bits>(Operation::add,a[i],b[i])),"submission lifetime failure");
    {auto next=e.batch();next.run(Operation::add,x,y,z);auto abandoned=next.submit();} // RAII waits and releases mapped access.
    require(z.mapped()!=nullptr,"ticket destruction did not release buffers");
    auto empty=e.make_buffer<F>(0);auto next=e.batch();next.run(Operation::add,empty,empty,empty);next.submit().wait();
    std::cout<<Bits<<"-bit resident API passed\n";
}
// Pipelines compile on a background thread while the same Engine keeps working; afterwards the
// prewarmed shapes need no compilation, and request errors surface from the future.
void prewarm(Engine& e){
    using C=Complex<9>;VectorRecurrence shape;shape.lanes=8;shape.steps=1;shape.fused=true;
    Prewarm request;request.bits={288};request.operations={Operation::mul,Operation::complex_fma};request.vector_shapes={shape};request.recurrence=true;
    auto pending=e.prewarm_async(request);
    std::mt19937_64 rng(5);std::vector<Float<128>> a(64),b(64),out(64);for(auto& x:a)x=reference::random_number<128>(rng,5);for(auto& x:b)x=reference::random_number<128>(rng,5);
    for(int r=0;r<20;++r){e.run(128,Operation::mul,a.data(),b.data(),out.data(),64);
        for(std::size_t i=0;i<64;++i)require(reference::equal<128>(out[i],reference::real<128>(Operation::mul,a[i],b[i])),"engine use during prewarm");}
    pending.get();
    std::vector<C> x(4*8);auto t=std::chrono::steady_clock::now();e.vector_recurrence(288,shape,x.data(),x.data(),x.data(),nullptr,x.data());
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
    require(seconds<2.0,"prewarmed vector recurrence still compiled on first use");
    Prewarm bad;bad.bits={80};bool rejected=false;try{e.prewarm_async(bad).get();}catch(const std::invalid_argument&){rejected=true;}require(rejected,"invalid prewarm width accepted");
    std::cout<<"prewarm: background compilation, concurrent use and first call ("<<seconds<<" s) passed\n";
}
int main(){try{rejects([]{Engine bad({31});});rejects([]{Engine bad({2048});});Engine e;prewarm(e);
    auto info=e.pipeline_info(384,Operation::mul);require(info.threads_per_threadgroup<=info.max_threads&&info.threads_per_threadgroup%info.simd_width==0,"invalid pipeline group size");run<128>(e);run<384>(e);run<1024>(e);return 0;}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
