#include "mx.hpp"
#include <iostream>
using namespace mx;
int main(int argc,char** argv){try{
  bool cpu=argc>1&&std::string(argv[1])=="--cpu-only";setenv("QSC_GPU",cpu?"0":"1",1);setenv("QSC_GPU_FOURIER","1",1);
  for(long precision:{352L,374L,448L}){
    set_prec(precision);Setup S;S.nPts=4;S.Nc=4;S.modeK=2;size_t points=8,modes=9;
    S.mph.resize(points*modes);C I=C::imag_unit();
    for(size_t x=0;x<S.mph.size();++x)S.mph[x]=C(double(int(x%11)-5)/8)+I*C(double(int(x%7)-3)/16);
    std::vector<ModeProjection> data(3);for(size_t j=0;j<data.size();++j){data[j].samples.resize(points*8);
      for(size_t x=0;x<data[j].samples.size();++x)data[j].samples[x]=C(double(int((x+j)%13)-6)/32)+I*C(double(int((2*x+j)%17)-8)/64);}
    C phase_seed,sample_seed;
    mpfr_set_str(mpc_realref(phase_seed.v),"0.123456789123456789123456789123456789123456789123456789123456789123456789123456789123456789123456789",10,MPFR_RNDN);
    mpfr_set_str(mpc_realref(sample_seed.v),"0.987654321987654321987654321987654321987654321987654321987654321987654321987654321987654321987654321",10,MPFR_RNDN);
    for(size_t x=0;x<S.mph.size();++x)S.mph[x]*=phase_seed;
    for(auto& item:data)for(auto& x:item.samples)x*=sample_seed;
    if(cpu){for(auto& d:data){auto out=mode_projection_cpu(S,d);if(out.size()!=modes||out[0].size()!=8)throw std::runtime_error("CPU projection shape");}
      std::vector<Mat> out(1,Mat(1,Vec(1,C(123))));if(lfx::fourier_batch(S,data,out,2)||!same(out[0][0][0],C(123)))throw std::runtime_error("global CPU switch/output preservation");
      std::cout<<precision<<" bits: CPU projection fixtures and disabled GPU guard pass\n";continue;}
    std::vector<Mat> out;if(!lfx::fourier_batch(S,data,out,2))throw std::runtime_error("Fourier GPU fallback");
    int B=lfx::bits();std::vector<C> negative(S.mph.size());for(size_t x=0;x<negative.size();++x)negative[x]=-S.mph[x];
    for(size_t j=0;j<data.size();++j)for(size_t q=0;q<modes;++q)for(size_t c=0;c<8;++c){
      std::vector<mpfr_ptr> x(2*points),y(2*points),xi(2*points);
      for(size_t r=0;r<points;++r){x[r]=mpc_realref(S.mph[r*modes+q].v);x[points+r]=mpc_imagref(negative[r*modes+q].v);
        xi[r]=mpc_imagref(S.mph[r*modes+q].v);xi[points+r]=mpc_realref(S.mph[r*modes+q].v);
        y[r]=mpc_realref(data[j].samples[r*8+c].v);y[points+r]=mpc_imagref(data[j].samples[r*8+c].v);}
      C truth;mpfr_set_prec(mpc_realref(truth.v),B);mpfr_set_prec(mpc_imagref(truth.v),B);
      mpfr_dot(mpc_realref(truth.v),x.data(),y.data(),2*points,MPFR_RNDN);mpfr_dot(mpc_imagref(truth.v),xi.data(),y.data(),2*points,MPFR_RNDN);
      C expected;mpc_set(expected.v,truth.v,RND);if(!same(out[j][q][c],expected))throw std::runtime_error("exact Fourier MPFR dot mismatch");
    }
    std::vector<Mat> preserved(1,Mat(1,Vec(1,C(123))));data[0].samples.pop_back();if(lfx::fourier_batch(S,data,preserved,2)||!same(preserved[0][0][0],C(123)))throw std::runtime_error("malformed sample/output preservation");
    std::cout<<precision<<" bits: Fourier MPFR exact-dot, packing, output and failure guards pass\n";
  }
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
