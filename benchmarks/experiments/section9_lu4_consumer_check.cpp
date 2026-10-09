#include "mx.hpp"
#include <iostream>
using namespace mx;
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
// MPFR at each scalar rounding site of the public LU4 composed sequence.
static C product(const C& a,const C& b){C p,q,r;
  mpfr_mul(mpc_realref(p.v),mpc_realref(a.v),mpc_realref(b.v),MPFR_RNDN);
  mpfr_mul(mpc_realref(q.v),mpc_imagref(a.v),mpc_imagref(b.v),MPFR_RNDN);
  mpfr_sub(mpc_realref(r.v),mpc_realref(p.v),mpc_realref(q.v),MPFR_RNDN);
  mpfr_mul(mpc_realref(p.v),mpc_realref(a.v),mpc_imagref(b.v),MPFR_RNDN);
  mpfr_mul(mpc_realref(q.v),mpc_imagref(a.v),mpc_realref(b.v),MPFR_RNDN);
  mpfr_add(mpc_imagref(r.v),mpc_realref(p.v),mpc_realref(q.v),MPFR_RNDN);return r;
}
static C divide(const C& a,const C& b){C p,q,den,r;
  mpfr_mul(mpc_realref(p.v),mpc_realref(b.v),mpc_realref(b.v),MPFR_RNDN);
  mpfr_mul(mpc_realref(q.v),mpc_imagref(b.v),mpc_imagref(b.v),MPFR_RNDN);
  mpfr_add(mpc_realref(den.v),mpc_realref(p.v),mpc_realref(q.v),MPFR_RNDN);
  mpfr_mul(mpc_realref(p.v),mpc_realref(a.v),mpc_realref(b.v),MPFR_RNDN);
  mpfr_mul(mpc_realref(q.v),mpc_imagref(a.v),mpc_imagref(b.v),MPFR_RNDN);
  mpfr_add(mpc_realref(r.v),mpc_realref(p.v),mpc_realref(q.v),MPFR_RNDN);
  mpfr_div(mpc_realref(r.v),mpc_realref(r.v),mpc_realref(den.v),MPFR_RNDN);
  mpfr_mul(mpc_realref(p.v),mpc_imagref(a.v),mpc_realref(b.v),MPFR_RNDN);
  mpfr_mul(mpc_realref(q.v),mpc_realref(a.v),mpc_imagref(b.v),MPFR_RNDN);
  mpfr_sub(mpc_imagref(r.v),mpc_realref(p.v),mpc_realref(q.v),MPFR_RNDN);
  mpfr_div(mpc_imagref(r.v),mpc_imagref(r.v),mpc_realref(den.v),MPFR_RNDN);return r;
}
static C subtract_product(const C& c,const C& a,const C& b){C p=product(a,b),r;
  mpfr_sub(mpc_realref(r.v),mpc_realref(c.v),mpc_realref(p.v),MPFR_RNDN);
  mpfr_sub(mpc_imagref(r.v),mpc_imagref(c.v),mpc_imagref(p.v),MPFR_RNDN);return r;
}
static mpfr_srcptr magnitude(const C& x){return mpfr_cmpabs(mpc_realref(x.v),mpc_imagref(x.v))>=0?mpc_realref(x.v):mpc_imagref(x.v);}
static Mat inverse_reference(const Mat& input){
  Mat a=input,result(4,Vec(4));int permutation[4]={0,1,2,3};
  for(int k=0;k<4;++k){int pivot=k;
    for(int row=k+1;row<4;++row)if(mpfr_cmpabs(magnitude(a[row][k]),magnitude(a[pivot][k]))>0)pivot=row;
    require(!mpfr_zero_p(mpc_realref(a[pivot][k].v))||!mpfr_zero_p(mpc_imagref(a[pivot][k].v)),"singular reference fixture");
    if(pivot!=k){std::swap(a[pivot],a[k]);std::swap(permutation[pivot],permutation[k]);}
    for(int row=k+1;row<4;++row){C l=divide(a[row][k],a[k][k]);a[row][k]=l;
      for(int col=k+1;col<4;++col)a[row][col]=subtract_product(a[row][col],l,a[k][col]);}
  }
  for(int col=0;col<4;++col){Vec y(4);for(int i=0;i<4;++i)y[i]=C(permutation[i]==col?1:0);
    for(int i=1;i<4;++i)for(int t=0;t<i;++t)y[i]=subtract_product(y[i],a[i][t],y[t]);
    for(int i=3;i>=0;--i){for(int t=i+1;t<4;++t)y[i]=subtract_product(y[i],a[i][t],y[t]);y[i]=divide(y[i],a[i][i]);}
    for(int i=0;i<4;++i)result[i][col]=y[i];
  }
  return result;
}
static C number(int key){C x;
  mpfr_set_str(mpc_realref(x.v),"0.123456789123456789123456789123456789123456789123456789123456789123456789123456789123456789123456789",10,MPFR_RNDN);
  mpfr_set_str(mpc_imagref(x.v),"0.987654321987654321987654321987654321987654321987654321987654321987654321987654321987654321987654321",10,MPFR_RNDN);
  mpfr_mul_si(mpc_realref(x.v),mpc_realref(x.v),key%11-5,MPFR_RNDN);
  mpfr_mul_si(mpc_imagref(x.v),mpc_imagref(x.v),key%7-3,MPFR_RNDN);return x;
}
int main(int argc,char** argv){try{
  bool cpu=false,all=false;for(int i=1;i<argc;++i){std::string arg=argv[i];if(arg=="--cpu-only")cpu=true;else if(arg=="--all-widths")all=true;else throw std::invalid_argument("unknown option");}
  setenv("QSC_GPU",cpu?"0":"1",1);setenv("QSC_GPU_LU4","1",1);
  std::vector<long> widths={352,374,448};if(all){widths.clear();for(long b=64;b<=1024;b+=32)widths.push_back(b);widths.push_back(374);}
  for(long precision:widths){set_prec(precision);std::vector<Mat> input(17,Mat(4,Vec(4)));
    for(size_t job=0;job<input.size();++job)for(int i=0;i<4;++i)for(int j=0;j<4;++j){input[job][i][j]=number(int(job)*17+i*4+j);if(i==j)input[job][i][j]+=C(8);}
    input[1]=Mat(4,Vec(4));for(int i=0;i<4;++i)input[1][i][(i+1)%4]=C(1); // row swaps
    input[2][0][0]=C(1,1);input[2][1][0]=C(1,-1); // exact magnitude tie
    input[2][2][0]=C(0);input[2][3][0]=C(0);
    for(int i=0;i<4;++i)for(int j=0;j<4;++j){mpfr_mul_2si(mpc_realref(input[3][i][j].v),mpc_realref(input[3][i][j].v),i%2?300:-300,MPFR_RNDN);
      mpfr_mul_2si(mpc_imagref(input[3][i][j].v),mpc_imagref(input[3][i][j].v),i%2?300:-300,MPFR_RNDN);}
    std::vector<Mat> out(1,Mat(1,Vec(1,C(123))));
    if(cpu){require(!lfx::inverse4_batch(input,out,2),"global GPU guard");require(same(out[0][0][0],C(123)),"disabled output changed");
      for(auto& m:input){auto result=inverse(m);require(result.size()==4&&result[0].size()==4,"CPU fixture shape");}
      std::cout<<precision<<" bits: CPU inverse fixtures and global guard pass\n";continue;}
    require(lfx::inverse4_batch(input,out,2),"LU4 helper fallback");require(out.size()==input.size(),"inverse batch shape");
    const long B=lfx::bits();for(size_t job=0;job<input.size();++job){set_prec(B);Mat truth=inverse_reference(input[job]);set_prec(precision);
      for(int i=0;i<4;++i)for(int j=0;j<4;++j){C expected;mpc_set(expected.v,truth[i][j].v,RND);
        if(!same(expected,out[job][i][j]))throw std::runtime_error("LU4 MPFR mismatch: precision="+std::to_string(precision)+
          " width="+std::to_string(B)+" matrix="+std::to_string(job)+" row="+std::to_string(i)+" col="+std::to_string(j));}}
    std::vector<Mat> preserved(1,Mat(1,Vec(1,C(123))));auto malformed=input;malformed[0][0].pop_back();
    require(!lfx::inverse4_batch(malformed,preserved,2)&&same(preserved[0][0][0],C(123)),"malformed output preservation");
    auto nonfinite=input;mpfr_set_nan(mpc_realref(nonfinite[0][0][0].v));
    require(!lfx::inverse4_batch(nonfinite,preserved,2)&&same(preserved[0][0][0],C(123)),"nonfinite output preservation");
    auto singular=input;singular[2]=Mat(4,Vec(4));
    require(!lfx::inverse4_batch(singular,preserved,2)&&same(preserved[0][0][0],C(123)),"singular companion preservation");
    setenv("QSC_GPU_LU4","0",1);require(!lfx::inverse4_batch(input,preserved,2),"LU4 sub-switch");setenv("QSC_GPU_LU4","1",1);
    std::vector<Mat> empty;require(lfx::inverse4_batch(empty,out,2)&&out.empty(),"empty inverse batch");
    std::cout<<precision<<" bits: MPFR LU4 inverse, row swaps/ties/scales, rounded width and failure guards pass"<<std::endl;
  }
  set_prec(1056);std::vector<Mat> input,out;require(!lfx::inverse4_batch(input,out,1),">1024-bit fallback");
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
