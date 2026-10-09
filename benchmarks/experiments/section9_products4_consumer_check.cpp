#include "mx.hpp"
#include <iostream>
using namespace mx;
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
// Independent MPFR primitives replay the public composed GEMM contract.
static C composed_product(const C& a,const C& b,long width){
  C result,p,q;
  for(C* x:{&result,&p,&q}){mpfr_set_prec(mpc_realref(x->v),width);mpfr_set_prec(mpc_imagref(x->v),width);}
  mpfr_mul(mpc_realref(p.v),mpc_realref(a.v),mpc_realref(b.v),MPFR_RNDN);
  mpfr_mul(mpc_realref(q.v),mpc_imagref(a.v),mpc_imagref(b.v),MPFR_RNDN);
  mpfr_sub(mpc_realref(result.v),mpc_realref(p.v),mpc_realref(q.v),MPFR_RNDN);
  mpfr_mul(mpc_realref(p.v),mpc_realref(a.v),mpc_imagref(b.v),MPFR_RNDN);
  mpfr_mul(mpc_realref(q.v),mpc_imagref(a.v),mpc_realref(b.v),MPFR_RNDN);
  mpfr_add(mpc_imagref(result.v),mpc_realref(p.v),mpc_realref(q.v),MPFR_RNDN);
  return result;
}
static Mat composed_matmul(const Mat& a,const Mat& b,long width){
  Mat result(4,Vec(4));
  for(auto& row:result)for(auto& x:row){mpfr_set_prec(mpc_realref(x.v),width);mpfr_set_prec(mpc_imagref(x.v),width);mpc_set_ui(x.v,0,RND);}
  for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int k=0;k<4;++k){C product=composed_product(a[i][k],b[k][j],width);
    mpfr_add(mpc_realref(result[i][j].v),mpc_realref(result[i][j].v),mpc_realref(product.v),MPFR_RNDN);
    mpfr_add(mpc_imagref(result[i][j].v),mpc_imagref(result[i][j].v),mpc_imagref(product.v),MPFR_RNDN);}
  return result;
}
static C number(int key,int scale){
  C x;mpfr_set_str(mpc_realref(x.v),"0.123456789123456789123456789123456789123456789123456789123456789123456789123456789123456789123456789",10,MPFR_RNDN);
  mpfr_set_str(mpc_imagref(x.v),"0.987654321987654321987654321987654321987654321987654321987654321987654321987654321987654321987654321",10,MPFR_RNDN);
  mpfr_mul_si(mpc_realref(x.v),mpc_realref(x.v),(key%11)-5,MPFR_RNDN);
  mpfr_mul_si(mpc_imagref(x.v),mpc_imagref(x.v),(key%7)-3,MPFR_RNDN);
  mpfr_mul_2si(mpc_realref(x.v),mpc_realref(x.v),scale,MPFR_RNDN);
  mpfr_mul_2si(mpc_imagref(x.v),mpc_imagref(x.v),scale,MPFR_RNDN);
  return x;
}
int main(int argc,char** argv){try{
  bool cpu=argc>1&&std::string(argv[1])=="--cpu-only";
  setenv("QSC_GPU",cpu?"0":"1",1);setenv("QSC_GPU_PRODUCTS4","1",1);
  for(long precision:{352L,374L,448L}){set_prec(precision);
    Rows base;base.Qi.assign(7,Mat(4,Vec(4)));
    std::vector<std::vector<Mat>> derivatives(5);for(size_t col:{0u,1u,3u,4u})derivatives[col].assign(7,Mat(4,Vec(4)));
    for(size_t j=0;j<7;++j)for(int a=0;a<4;++a)for(int b=0;b<4;++b){
      base.Qi[j][a][b]=number(int(j)*17+a*4+b,j==6?((a+b)%2?500:-500):0);
      for(size_t col:{0u,1u,3u,4u})derivatives[col][j][a][b]=number(int(col)*19+int(j)*3+a*4+b,j==6?((a+b)%2?-500:500):0);
    }
    // Exact cancellation in one inner dot, plus a skipped column and an odd point count.
    for(int a=0;a<4;++a){base.Qi[0][a][1]=base.Qi[0][a][0];derivatives[0][0][1][a]=-derivatives[0][0][0][a];}
    std::vector<std::vector<Mat>> out(1,std::vector<Mat>(1,Mat(1,Vec(1,C(123)))));
    if(cpu){require(!lfx::tangent_products4(base,derivatives,out,2),"global GPU guard");require(same(out[0][0][0][0],C(123)),"disabled output changed");
      for(size_t col:{0u,1u,3u,4u})for(size_t j=0;j<7;++j){auto result=tangent_inverse_product_cpu(base.Qi[j],derivatives[col][j]);require(result.size()==4&&result[0].size()==4,"CPU shape");}
      std::cout<<precision<<" bits: CPU fixtures and global guard pass\n";continue;}
    require(lfx::tangent_products4(base,derivatives,out,2),"GPU products4 fallback");
    require(out.size()==5&&out[2].empty(),"column mapping");
    int B=lfx::bits();
    for(size_t col:{0u,1u,3u,4u})for(size_t j=0;j<7;++j){auto intermediate=composed_matmul(base.Qi[j],derivatives[col][j],B);auto truth=composed_matmul(intermediate,base.Qi[j],B);
      for(int a=0;a<4;++a)for(int b=0;b<4;++b){mpc_neg(truth[a][b].v,truth[a][b].v,RND);C expected;mpc_set(expected.v,truth[a][b].v,RND);
        require(same(out[col][j][a][b],expected),"MPFR two-stage product mismatch");}}
    std::vector<std::vector<Mat>> preserved(1,std::vector<Mat>(1,Mat(1,Vec(1,C(123)))));
    derivatives[0][0][0].pop_back();require(!lfx::tangent_products4(base,derivatives,preserved,2),"malformed derivative accepted");
    require(same(preserved[0][0][0][0],C(123)),"failure changed output");
    derivatives[0][0][0].push_back(C(0));mpfr_set_nan(mpc_realref(derivatives[0][0][0][0].v));
    require(!lfx::tangent_products4(base,derivatives,preserved,2),"nonfinite conversion accepted");
    require(same(preserved[0][0][0][0],C(123)),"conversion failure changed output");
    setenv("QSC_GPU_PRODUCTS4","0",1);require(!lfx::tangent_products4(base,derivatives,preserved,2),"sub-switch ignored");setenv("QSC_GPU_PRODUCTS4","1",1);
    std::cout<<precision<<" bits: MPFR product3, rounded width, cancellation/wide data, mapping and failure guards pass\n";
  }
  set_prec(1056);Rows empty;std::vector<std::vector<Mat>> derivatives,out;
  require(!lfx::tangent_products4(empty,derivatives,out,1),">1024-bit fallback");
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
