#include "section9_cases.hpp"
#include <type_traits>

// Compile every additive signature without evaluating empty operands.
void compile_signatures(Linalg& exact,BatchedLinalg& sequence,CommandBatch& batch,Buffer<F<64>>& a,Buffer<F<64>>& b,Buffer<F<64>>& c,Buffer<std::uint32_t>& status){
    exact.gemm_exact(batch,false,a,b,1,1,1,c);exact.syrk_exact(batch,a,1,1,c);
    exact.cholesky_blocked(a,1,c);exact.cholesky_solve_blocked(a,1,b,1,c);
    exact.syrk_exact(64,nullptr,0,0,nullptr);exact.cholesky_blocked(64,nullptr,0,nullptr);exact.cholesky_solve_blocked(64,nullptr,0,nullptr,0,nullptr);
    sequence.gemm_sequential(batch,{},a,b,c);sequence.normal_equations_sequential(batch,a,b,1,1,c,a);
    sequence.cholesky_trials_sequential(batch,{1,1},a,b,b,c,status);sequence.cholesky_solve_sequential(batch,a,status,1,1,b,1,c);
}
int main(){try{
    check_library_version();Engine engine;Linalg exact(engine);BatchedLinalg sequence(engine);
    auto one=from_decimal<64>("1"),large=one;large.exponent=100;
    std::vector<F<64>> a{large,one,negate(large)},b(3,one),out(1);
    auto precise=reference::dot<64>(a.data(),1,b.data(),1,3);auto rounded=zero<2>();
    for(int k=0;k<3;++k)rounded=ra<64>(rounded,rm<64>(a[k],b[k]));
    require(same<64>(precise,one)&&!rounded.sign,"cancellation oracle did not distinguish contracts");
    exact.gemm_exact(64,false,a.data(),b.data(),1,1,3,out.data());check<64>(out,{precise},"named exact host contract");
    StridedGemm shape{1,1,1,3,3,3,1};sequence.gemm_sequential(64,false,shape,a.data(),b.data(),out.data());check<64>(out,{rounded},"named composed sequential host contract");
    auto ab=put(engine,a),bb=put(engine,b),cb=engine.make_buffer<F<64>>(1);auto batch=engine.batch();auto ticket=exact.gemm_exact(batch,false,ab,bb,1,1,3,cb);batch.submit().wait();require(ticket.resolved(),"named exact resident ticket unresolved");check<64>(get(cb),{precise},"named exact resident contract");
    for(bool fused:{false,true}){shape.fused=fused;auto job=engine.batch();sequence.gemm_sequential(job,shape,ab,bb,cb);job.submit().wait();check<64>(get(cb),{rounded},"named sequential resident contract");}
    std::cout<<"Contract names: exact cancellation=1, composed/fused sequential=0; all additive signatures compile\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
