// Round 23 (plan D6b / S5): normal equations (J^T J + damping) X = J^T R on QSC-like data. Linalg::cholesky (blocked,
// trailing updates through the exact residue GEMM with one rounding per entry) and Linalg::cholesky_solve, against a
// consumer-style MPFR right-looking Cholesky (serial and multithreaded) and MPFR substitution; end-to-end SYRK +
// Cholesky + solve. Modes: default timing table, --sweep (block size / host threshold), --accuracy (errors against a
// 2*bits MPFR solve).
#import <Foundation/Foundation.h>
#include "limbforge/linalg.hpp"
#include "benchmark_support.hpp"
#include <atomic>
#include <map>
// QSC-like Jacobian columns (as benchmarks/linalg.mm): column scale 2^[-60,60], entries within 2^-25..2^5 of it,
// 25% zeros, one entry 2^-150 below the column scale in every 17th column.
template<int Bits> std::vector<Float<Bits>> jacobian(std::size_t rows,std::size_t cols,std::mt19937_64& rng){
    std::vector<Float<Bits>> A(rows*cols);
    for(std::size_t c=0;c<cols;++c){long scale=long(rng()%121)-60;
        for(std::size_t r=0;r<rows;++r){auto x=reference::random_number<Bits>(rng,15);x.exponent+=int(scale-10);if(rng()%4==0)x=zero<Bits/32>();A[r*cols+c]=x;}
        if(c%17==5)A[(rng()%rows)*cols+c].exponent=int(scale-150);}
    return A;
}
// A = J^T J (lower, one rounding per entry) with Marquardt damping a_ii = RN(a_ii + 2^(e_ii - 10)); returns the SYRK wall time.
template<int Bits> double normal_matrix(Linalg& la,const std::vector<Float<Bits>>& J,std::size_t n,std::vector<Float<Bits>>& A){
    auto t=Clock::now();la.syrk(Bits,J.data(),2*n,n,A.data(),true);
    for(std::size_t i=0;i<n;++i){auto& a=A[i*n+i];if(a.sign){Float<Bits> d=zero<Bits/32>();d.limb[Bits/32-1]=0x80000000u;d.sign=1;d.exponent=a.exponent-10;a=add(a,d);}}
    return std::chrono::duration<double>(Clock::now()-t).count();
}
// Consumer-style MPFR right-looking Cholesky (lower, each operation rounded); workers == nullptr: serial.
double mpfr_cholesky(int bits,MPArray& C,std::size_t n,Workers* workers){
    auto start=Clock::now();
    for(std::size_t j=0;j<n;++j){mpfr_sqrt(C[j*n+j],C[j*n+j],MPFR_RNDN);
        auto column=[&](std::size_t a,std::size_t b){for(std::size_t i=a;i<b;++i)mpfr_div(C[i*n+j],C[i*n+j],C[j*n+j],MPFR_RNDN);};
        auto update=[&](std::size_t i){mpfr_t t;mpfr_init2(t,bits);for(std::size_t k=j+1;k<=i;++k){mpfr_mul(t,C[i*n+j],C[k*n+j],MPFR_RNDN);mpfr_sub(C[i*n+k],C[i*n+k],t,MPFR_RNDN);}mpfr_clear(t);};
        if(!workers||n-j<64){column(j+1,n);for(std::size_t i=j+1;i<n;++i)update(i);continue;}
        workers->run(n-j-1,[&](std::size_t a,std::size_t b){column(j+1+a,j+1+b);});
        std::atomic<std::size_t> next{j+1};workers->run(workers->size(),[&](std::size_t,std::size_t){for(std::size_t i;(i=next.fetch_add(1))<n;)update(i);});}
    return std::chrono::duration<double>(Clock::now()-start).count();
}
// MPFR forward and back substitution with rounded operations (X: n x nrhs, holds B on entry).
double mpfr_solve(int bits,MPArray& L,std::size_t n,MPArray& X,std::size_t nrhs){
    auto start=Clock::now();mpfr_t t;mpfr_init2(t,bits);
    for(std::size_t c=0;c<nrhs;++c){
        for(std::size_t i=0;i<n;++i){for(std::size_t k=0;k<i;++k){mpfr_mul(t,L[i*n+k],X[k*nrhs+c],MPFR_RNDN);mpfr_sub(X[i*nrhs+c],X[i*nrhs+c],t,MPFR_RNDN);}mpfr_div(X[i*nrhs+c],X[i*nrhs+c],L[i*n+i],MPFR_RNDN);}
        for(std::size_t i=n;i-->0;){for(std::size_t k=i+1;k<n;++k){mpfr_mul(t,L[k*n+i],X[k*nrhs+c],MPFR_RNDN);mpfr_sub(X[i*nrhs+c],X[i*nrhs+c],t,MPFR_RNDN);}mpfr_div(X[i*nrhs+c],X[i*nrhs+c],L[i*n+i],MPFR_RNDN);}}
    mpfr_clear(t);return std::chrono::duration<double>(Clock::now()-start).count();
}
struct Options {int repeats=5;std::size_t serial_max=400,parallel_max=1000,nrhs=4;unsigned threads=std::thread::hardware_concurrency();FactorOptions factor;};
double median(std::vector<double> x){return quantile(x,.5);}
template<int Bits> void bench(Linalg& la,Workers& workers,const Options& o,std::size_t n,std::mt19937_64& rng,bool header){
    if(header)std::cout<<"bits,n,nrhs,block,host_macs,repeats,syrk_wall_ms,chol_wall_ms,chol_wall_min_ms,chol_gpu_ms,chol_panel_ms,chol_update_ms,gpu_updates,host_updates,"
                         "solve_wall_ms,solve_gpu_ms,e2e_wall_ms,host_only_chol_ms,gpu_equals_host,mpfr_serial_chol_s,serial_measured,mpfr_parallel_chol_s,mpfr_threads,"
                         "mpfr_serial_solve_s,speedup_vs_serial,speedup_vs_parallel,e2e_mpfr_parallel_s\n";
    const std::size_t nrhs=o.nrhs;auto J=jacobian<Bits>(2*n,n,rng);std::vector<Float<Bits>> A(n*n,zero<Bits/32>()),L(n*n),B(n*nrhs),X(n*nrhs);
    for(auto& b:B)b=reference::random_number<Bits>(rng,20);
    std::vector<double> syrk,wall,gpu,panel,update,swall,sgpu,e2e;CholeskyInfo info;
    normal_matrix<Bits>(la,J,n,A);la.cholesky(Bits,A.data(),n,L.data(),o.factor);la.cholesky_solve(Bits,L.data(),n,B.data(),nrhs,X.data(),o.factor); // warm-up (pipelines)
    for(int r=0;r<o.repeats;++r){
        auto t=Clock::now();syrk.push_back(normal_matrix<Bits>(la,J,n,A));info=la.cholesky(Bits,A.data(),n,L.data(),o.factor);
        if(info.pivot!=n)throw std::runtime_error("benchmark matrix is not positive definite");
        Timing s=la.cholesky_solve(Bits,L.data(),n,B.data(),nrhs,X.data(),o.factor);e2e.push_back(std::chrono::duration<double>(Clock::now()-t).count());
        wall.push_back(info.timing.wall_seconds);gpu.push_back(info.timing.gpu_seconds);panel.push_back(info.panel_seconds);update.push_back(info.update_seconds);
        swall.push_back(s.wall_seconds);sgpu.push_back(s.gpu_seconds);}
    // The same factorization with every update on the host must be bit-identical (the GPU only changes speed).
    FactorOptions host=o.factor;host.gpu=false;std::vector<Float<Bits>> H(n*n);auto th=Clock::now();la.cholesky(Bits,A.data(),n,H.data(),host);double host_ms=std::chrono::duration<double>(Clock::now()-th).count()*1e3;
    bool same=std::equal(L.begin(),L.end(),H.begin(),[](const Float<Bits>& x,const Float<Bits>& y){return reference::equal<Bits>(x,y);});
    if(!same)throw std::runtime_error("GPU and host-only factorizations differ");
    // MPFR baselines from the same lower triangle.
    auto load=[&](MPArray& m){for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<=i;++j)to_mpfr<Bits>(m[i*n+j],A[i*n+j]);};
    double serial=0,parallel=0,ssolve=0;bool measured=n<=o.serial_max;
    {MPArray m(n*n,Bits);load(m);
     if(measured)serial=mpfr_cholesky(Bits,m,n,nullptr);
     else{std::size_t s=std::min<std::size_t>(o.serial_max,n);MPArray q(s*s,Bits);for(std::size_t i=0;i<s;++i)for(std::size_t j=0;j<=i;++j)mpfr_set(q[i*s+j],m[i*n+j],MPFR_RNDN);
         serial=mpfr_cholesky(Bits,q,s,nullptr)*std::pow(double(n)/double(s),3);} // n^3 extrapolation from the leading s x s block
     MPArray x(n*nrhs,Bits);for(std::size_t q=0;q<n*nrhs;++q)to_mpfr<Bits>(x[q],B[q]);
     if(n<=o.parallel_max){MPArray p(n*n,Bits);load(p);parallel=mpfr_cholesky(Bits,p,n,&workers);ssolve=mpfr_solve(Bits,p,n,x,nrhs);}}
    std::cout<<std::setprecision(5)<<Bits<<','<<n<<','<<nrhs<<','<<o.factor.block<<','<<o.factor.host_macs<<','<<o.repeats<<','<<median(syrk)*1e3<<','<<median(wall)*1e3<<','<<quantile(wall,0)*1e3<<','
        <<median(gpu)*1e3<<','<<median(panel)*1e3<<','<<median(update)*1e3<<','<<info.gpu_updates<<','<<info.host_updates<<','<<median(swall)*1e3<<','<<median(sgpu)*1e3<<','
        <<median(e2e)*1e3<<','<<host_ms<<','<<same<<','<<serial<<','<<measured<<','<<parallel<<','<<workers.size()<<','<<ssolve<<','<<serial/median(wall)<<','
        <<(parallel?parallel/median(wall):0)<<','<<(parallel?median(syrk)+parallel+ssolve:0)<<std::endl;
}
// Block-size and host-threshold sweep (wall time of the factorization, median of repeats).
template<int Bits> void sweep(Linalg& la,const Options& o,std::size_t n,std::mt19937_64& rng,bool header){
    if(header)std::cout<<"bits,n,block,host_macs,repeats,chol_wall_ms,chol_wall_min_ms,chol_gpu_ms,chol_panel_ms,chol_update_ms,gpu_updates,host_updates,solve_wall_ms\n";
    auto J=jacobian<Bits>(2*n,n,rng);std::vector<Float<Bits>> A(n*n,zero<Bits/32>()),L(n*n),B(n*o.nrhs),X(n*o.nrhs);normal_matrix<Bits>(la,J,n,A);
    for(auto& b:B)b=reference::random_number<Bits>(rng,20);
    for(std::size_t block:{16,24,32,40,48,64,96})for(double hm:{3e4,1e5,3e5}){FactorOptions f;f.block=block;f.host_macs=hm;CholeskyInfo info=la.cholesky(Bits,A.data(),n,L.data(),f);
        std::vector<double> wall,gpu,panel,update,solve;
        for(int r=0;r<o.repeats;++r){info=la.cholesky(Bits,A.data(),n,L.data(),f);wall.push_back(info.timing.wall_seconds);gpu.push_back(info.timing.gpu_seconds);
            panel.push_back(info.panel_seconds);update.push_back(info.update_seconds);solve.push_back(la.cholesky_solve(Bits,L.data(),n,B.data(),o.nrhs,X.data(),f).wall_seconds);}
        std::cout<<std::setprecision(5)<<Bits<<','<<n<<','<<block<<','<<hm<<','<<o.repeats<<','<<median(wall)*1e3<<','<<quantile(wall,0)*1e3<<','<<median(gpu)*1e3<<','
            <<median(panel)*1e3<<','<<median(update)*1e3<<','<<info.gpu_updates<<','<<info.host_updates<<','<<median(solve)*1e3<<std::endl;}
}
// Solve placement: every update on the host, every update on the GPU, and the default threshold (identical results).
template<int Bits> void solve_sweep(Linalg& la,const Options& o,std::size_t n,std::mt19937_64& rng,bool header){
    if(header)std::cout<<"bits,n,nrhs,block,repeats,host_only_ms,gpu_all_ms,default_ms,default_host_macs,identical\n";
    auto J=jacobian<Bits>(2*n,n,rng);std::vector<Float<Bits>> A(n*n,zero<Bits/32>()),L(n*n);normal_matrix<Bits>(la,J,n,A);la.cholesky(Bits,A.data(),n,L.data(),o.factor);
    for(std::size_t nrhs:{1,4,16,64}){std::vector<Float<Bits>> B(n*nrhs);for(auto& b:B)b=reference::random_number<Bits>(rng,20);
        FactorOptions host=o.factor,all=o.factor;host.gpu=false;all.host_macs=0;std::vector<double> t[3];std::vector<Float<Bits>> X[3];
        for(int r=0;r<o.repeats;++r)for(int v=0;v<3;++v){X[v].assign(n*nrhs,zero<Bits/32>());t[v].push_back(la.cholesky_solve(Bits,L.data(),n,B.data(),nrhs,X[v].data(),v==0?host:v==1?all:o.factor).wall_seconds);}
        bool same=true;for(int v=1;v<3;++v)same&=std::equal(X[0].begin(),X[0].end(),X[v].begin(),[](const Float<Bits>& x,const Float<Bits>& y){return reference::equal<Bits>(x,y);});
        std::cout<<std::setprecision(5)<<Bits<<','<<n<<','<<nrhs<<','<<o.factor.block<<','<<o.repeats<<','<<median(t[0])*1e3<<','<<median(t[1])*1e3<<','<<median(t[2])*1e3<<','<<o.factor.host_macs<<','<<same<<std::endl;
        if(!same)throw std::runtime_error("solve placements differ");}
}
// Errors against a 2*bits MPFR Cholesky solve of the same (exactly representable) A and B.
template<int Bits> void accuracy(Linalg& la,Workers& workers,const Options& o,std::size_t n,std::mt19937_64& rng,bool header){
    if(header)std::cout<<"bits,n,nrhs,block,method,log2_fwd_norm,log2_fwd_comp,log2_bwd_norm,log2_bwd_comp\n";
    const std::size_t nrhs=o.nrhs;const int P=2*Bits;auto J=jacobian<Bits>(2*n,n,rng);std::vector<Float<Bits>> A(n*n,zero<Bits/32>()),L(n*n),B(n*nrhs),X(n*nrhs);
    normal_matrix<Bits>(la,J,n,A);for(auto& b:B)b=reference::random_number<Bits>(rng,20);
    la.cholesky(Bits,A.data(),n,L.data(),o.factor);la.cholesky_solve(Bits,L.data(),n,B.data(),nrhs,X.data(),o.factor);
    // High-precision reference x* (2*bits).
    MPArray H(n*n,P),xs(n*nrhs,P);for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<=i;++j)to_mpfr<Bits>(H[i*n+j],A[i*n+j]);
    for(std::size_t q=0;q<n*nrhs;++q)to_mpfr<Bits>(xs[q],B[q]);mpfr_cholesky(P,H,n,&workers);mpfr_solve(P,H,n,xs,nrhs);
    // Consumer-style MPFR at bits.
    MPArray C(n*n,Bits),xc(n*nrhs,Bits);for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<=i;++j)to_mpfr<Bits>(C[i*n+j],A[i*n+j]);
    for(std::size_t q=0;q<n*nrhs;++q)to_mpfr<Bits>(xc[q],B[q]);mpfr_cholesky(Bits,C,n,&workers);mpfr_solve(Bits,C,n,xc,nrhs);
    auto report=[&](const char* method,auto get){ // get(q, mpfr_t out) at precision >= Bits
        const int W=4*Bits+64;mpfr_t d,m,num,den,t,r,an,xn,bn;mpfr_inits2(W,d,m,num,den,t,r,an,xn,bn,(mpfr_ptr)0);mpfr_t xq;mpfr_init2(xq,Bits);
        mpfr_set_zero(num,1);mpfr_set_zero(den,1);double comp=-1e9;
        for(std::size_t q=0;q<n*nrhs;++q){get(q,xq);mpfr_sub(d,xq,xs[q],MPFR_RNDN);mpfr_abs(d,d,MPFR_RNDN);if(mpfr_cmp(d,num)>0)mpfr_set(num,d,MPFR_RNDN);
            mpfr_abs(m,xs[q],MPFR_RNDN);if(mpfr_cmp(m,den)>0)mpfr_set(den,m,MPFR_RNDN);
            if(!mpfr_zero_p(xs[q])&&!mpfr_zero_p(d)){mpfr_div(t,d,m,MPFR_RNDN);comp=std::max(comp,double(mpfr_get_exp(t)));}}
        mpfr_div(t,num,den,MPFR_RNDN);double fwd=mpfr_zero_p(t)?-1e9:std::log2(mpfr_get_d(t,MPFR_RNDN));
        // Backward errors per column, maximum over columns: normwise ||B - A X||_inf / (||A||_inf ||X||_inf + ||B||_inf) and
        // componentwise (Oettli-Prager) max_i |r_i| / (|A| |x| + |b|)_i, which is the informative one for badly scaled A.
        double bwd=-1e9,bwc=-1e9;
        for(std::size_t c=0;c<nrhs;++c){mpfr_set_zero(r,1);mpfr_set_zero(an,1);mpfr_set_zero(xn,1);mpfr_set_zero(bn,1);
            for(std::size_t i=0;i<n;++i){get(i*nrhs+c,xq);mpfr_abs(t,xq,MPFR_RNDN);if(mpfr_cmp(t,xn)>0)mpfr_set(xn,t,MPFR_RNDN);}
            std::vector<mpfr_ptr> terms;MPArray prod(n+1,W);mpfr_t rowsum,s,cd;mpfr_init2(rowsum,W);mpfr_init2(s,W);mpfr_init2(cd,W);
            for(std::size_t i=0;i<n;++i){mpfr_set_zero(rowsum,1);mpfr_set_zero(cd,1);terms.clear();
                for(std::size_t j=0;j<n;++j){const auto& a=i>=j?A[i*n+j]:A[j*n+i];mpfr_t av;mpfr_init2(av,Bits);to_mpfr<Bits>(av,a);get(j*nrhs+c,xq);
                    mpfr_mul(prod[j],av,xq,MPFR_RNDN);terms.push_back(prod[j]);mpfr_abs(t,av,MPFR_RNDN);mpfr_add(rowsum,rowsum,t,MPFR_RNDN);mpfr_abs(t,prod[j],MPFR_RNDN);mpfr_add(cd,cd,t,MPFR_RNDN);mpfr_clear(av);}
                mpfr_t bv;mpfr_init2(bv,Bits);to_mpfr<Bits>(bv,B[i*nrhs+c]);mpfr_neg(prod[n],bv,MPFR_RNDN);terms.push_back(prod[n]);mpfr_abs(t,bv,MPFR_RNDN);mpfr_add(cd,cd,t,MPFR_RNDN);if(mpfr_cmp(t,bn)>0)mpfr_set(bn,t,MPFR_RNDN);mpfr_clear(bv);
                mpfr_sum(s,terms.data(),terms.size(),MPFR_RNDN);mpfr_abs(s,s,MPFR_RNDN);if(!mpfr_zero_p(s)&&!mpfr_zero_p(cd)){mpfr_div(t,s,cd,MPFR_RNDN);bwc=std::max(bwc,double(mpfr_get_exp(t)));}if(mpfr_cmp(s,r)>0)mpfr_set(r,s,MPFR_RNDN);if(mpfr_cmp(rowsum,an)>0)mpfr_set(an,rowsum,MPFR_RNDN);}
            mpfr_mul(t,an,xn,MPFR_RNDN);mpfr_add(t,t,bn,MPFR_RNDN);mpfr_div(t,r,t,MPFR_RNDN);if(!mpfr_zero_p(t))bwd=std::max(bwd,double(mpfr_get_exp(t)));mpfr_clear(rowsum);mpfr_clear(s);mpfr_clear(cd);}
        std::cout<<Bits<<','<<n<<','<<nrhs<<','<<o.factor.block<<','<<method<<','<<fwd<<','<<comp<<','<<bwd<<','<<bwc<<std::endl;
        mpfr_clears(d,m,num,den,t,r,an,xn,bn,(mpfr_ptr)0);mpfr_clear(xq);};
    report("limbforge_blocked",[&](std::size_t q,mpfr_t out){to_mpfr<Bits>(out,X[q]);});
    report("mpfr_sequential",[&](std::size_t q,mpfr_t out){mpfr_set(out,xc[q],MPFR_RNDN);});
}
int main(int argc,char** argv){try{
    Options o;std::vector<int> bits={224,256};std::vector<std::size_t> sizes={200,400,800,1000};std::string mode="time";
    for(int i=1;i<argc;++i){std::string a=argv[i];auto next=[&]{if(i+1>=argc)throw std::invalid_argument("missing value");return std::string(argv[++i]);};
        if(a=="--bits")bits={std::stoi(next())};else if(a=="--size")sizes={std::size_t(std::stoul(next()))};else if(a=="--repeats")o.repeats=std::stoi(next());
        else if(a=="--threads")o.threads=unsigned(std::stoul(next()));else if(a=="--serial-max")o.serial_max=std::stoul(next());else if(a=="--parallel-max")o.parallel_max=std::stoul(next());
        else if(a=="--nrhs")o.nrhs=std::stoul(next());else if(a=="--block")o.factor.block=std::stoul(next());else if(a=="--host-macs")o.factor.host_macs=std::stod(next());
        else if(a=="--sweep")mode="sweep";else if(a=="--accuracy")mode="accuracy";else if(a=="--solve-sweep")mode="solve";
        else throw std::invalid_argument("usage: cholesky_limbforge [--bits 224|256] [--size n] [--repeats r] [--threads t] [--serial-max n] [--parallel-max n] [--nrhs k] [--block nb] [--host-macs x] [--sweep|--solve-sweep|--accuracy]");}
    Linalg la;Workers workers(o.threads);std::mt19937_64 rng(20261023);bool header=true;
    std::cerr<<la.device_name()<<"; times: wall = host call (median), gpu = summed command buffers\n";
    for(int b:bits)for(auto n:sizes){
        auto go=[&](auto tag){constexpr int B=decltype(tag)::value;if(mode=="time")bench<B>(la,workers,o,n,rng,header);else if(mode=="sweep")sweep<B>(la,o,n,rng,header);else if(mode=="solve")solve_sweep<B>(la,o,n,rng,header);else accuracy<B>(la,workers,o,n,rng,header);};
        if(b==224)go(std::integral_constant<int,224>{});else if(b==256)go(std::integral_constant<int,256>{});else throw std::invalid_argument("bits: 224 or 256");header=false;}
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
