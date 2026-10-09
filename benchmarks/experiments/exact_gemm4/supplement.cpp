#define main original_exact4_test_main
#include "test.cpp"
#undef main
// Additional inputs for the unchanged kernel, retaining the independent MPFR oracle.
template<int B> void supplemental(Engine& e,ExactGemm4& unit){using T=Float<B>;using Z=Complex<B/32>;auto z=zero<B/32>(),one=from_decimal<B>("1"),high=one,tiny=one,half=one;
    high.exponent=1000000000;tiny.exponent=-10000;half.exponent=-B;ExactGemm4Shape s{4,16,16,16};
    std::vector<T> a(64,z),b(64,z),initial(64,one);
    for(unsigned t=0;t<4;++t)for(unsigned k=0;k<4;++k)b[t*16+k*4]=one;
    a[0]=high;a[1]=negate(high);b[0]=b[4]=high;a[2]=one; // Overflowing terms cancel and reveal 1.
    a[16]=high;a[17]=negate(high);b[16]=b[20]=high; // Bounded, exactly zero, no premature overflow.
    a[32]=one;a[33]=half;a[34]=tiny; // A tie plus a far positive epsilon.
    a[48]=one;a[49]=half;a[50]=negate(tiny); // The same tie minus a far epsilon.
    for(bool acc:{false,true}){s.accumulate=acc;auto got=initial;if(acc)got[1]=zero<B/32>(invalid|division_by_zero);
        auto expected=oracle<B>(s,a,b,got);ExactGemm4Report report;unit.gemm(B,false,s,a.data(),b.data(),got.data(),&report);compare<B>(got,expected,"overflow/tie rescue real");need(report.host_components>0&&report.gpu_components>0,"mixed supplemental paths");}
    std::vector<Z> ca(64,{z,z}),cb(64,{z,z}),old(64,{one,one});
    for(unsigned t=0;t<4;++t)for(unsigned k=0;k<4;++k)cb[t*16+k*4]={one,z};
    ca[0]={high,high};ca[1]={negate(high),negate(high)};cb[0]=cb[4]={high,negate(high)};ca[2]={one,one};
    ca[16]={high,high};ca[17]={negate(high),negate(high)};cb[16]=cb[20]={high,negate(high)};
    ca[32]={one,one};ca[33]={half,half};ca[34]={tiny,negate(tiny)};
    ca[48]={one,one};ca[49]={half,half};ca[50]={negate(tiny),tiny};
    for(bool acc:{false,true}){s.accumulate=acc;auto got=old;if(acc)got[1].re=zero<B/32>(invalid|division_by_zero);
        auto expected=oracle<B>(s,ca,cb,got);auto ab=upload(e,ca),bb=upload(e,cb),out=upload(e,got);auto batch=e.batch();auto ticket=unit.gemm(batch,s,ab,bb,out);batch.submit().wait();compare<B>(download(out),expected,"overflow/tie rescue complex");need(ticket.report().host_components>0,"complex supplemental fallback");}
    std::cout<<B<<" bits exact4 overflow cancellation, far-epsilon ties and old-component statuses PASS\n"<<std::flush;
}
int main(){try{Engine e;ExactGemm4 unit(e);for(int n=2;n<=32;++n)switch(n){
#define CASE(N) case N:supplemental<32*N>(e,unit);break;
CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9) CASE(10) CASE(11) CASE(12) CASE(13) CASE(14) CASE(15) CASE(16) CASE(17) CASE(18) CASE(19) CASE(20) CASE(21) CASE(22) CASE(23) CASE(24) CASE(25) CASE(26) CASE(27) CASE(28) CASE(29) CASE(30) CASE(31) CASE(32)
#undef CASE
    }std::cout<<"PASS exact4 supplementary31-width gate\n";return 0;}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
