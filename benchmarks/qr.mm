// Round 31 (plan S5): Householder QR least squares on QSC-like J (m = 2n). Linalg::factor_qr (compact-WY blocks, trailing
// updates through the exact residue GEMM) + QRFactor::solve, against a consumer-style MPFR Householder (qsccpp mx.hpp QR:
// unblocked, every operation rounded; serial and multithreaded) and against the normal equations (SYRK + J^T b + Cholesky +
// solve, round 23). Modes: default timing table, --accuracy (errors against a high-precision MPFR least-squares solve, QR vs
// normal equations on well- and ill-conditioned J), --sweep (block size).
#import <Foundation/Foundation.h>
#include "limbforge/linalg.hpp"
#include "benchmark_support.hpp"
#include <atomic>
#include <cmath>
#include <map>
// QSC-like Jacobian columns (as benchmarks/cholesky.mm): column scale 2^[-60,60], entries within 2^-25..2^5 of it, 25% zeros,
// one entry 2^-150 below the column scale in every 17th column.
template<int Bits> std::vector<Float<Bits>> jacobian(std::size_t rows,std::size_t cols,std::mt19937_64& rng){
    std::vector<Float<Bits>> A(rows*cols);
    for(std::size_t c=0;c<cols;++c){long scale=long(rng()%121)-60;
        for(std::size_t r=0;r<rows;++r){auto x=reference::random_number<Bits>(rng,15);x.exponent+=int(scale-10);if(rng()%4==0)x=zero<Bits/32>();A[r*cols+c]=x;}
        if(c%17==5)A[(rng()%rows)*cols+c].exponent=int(scale-150);}
    return A;
}
// Ill-conditioned J = (G1 diag(2^(-kappa j/(n-1)))) G2 (one rounding per entry, Linalg::gemm) with random G1 (m x n), G2 (n x n),
// then the QSC column scales 2^[-60, 60] (exact): singular values spread over about kappa binades.
template<int Bits> std::vector<Float<Bits>> conditioned(Linalg& la,std::size_t m,std::size_t n,int kappa,std::mt19937_64& rng){
    std::vector<Float<Bits>> G1(m*n),G2(n*n),J(m*n);for(auto& x:G1)x=reference::random_number<Bits>(rng,1);for(auto& x:G2)x=reference::random_number<Bits>(rng,1);
    for(std::size_t r=0;r<m;++r)for(std::size_t c=0;c<n;++c)G1[r*n+c].exponent-=int(std::lround(double(kappa)*double(c)/double(std::max<std::size_t>(n-1,1))));
    la.gemm(Bits,false,G1.data(),G2.data(),m,n,n,J.data());
    for(std::size_t c=0;c<n;++c){int s=int(rng()%121)-60;for(std::size_t r=0;r<m;++r)if(J[r*n+c].sign)J[r*n+c].exponent+=s;}
    return J;
}
// Consumer-style MPFR Householder (mx.hpp QR): nrm2 = sum a_ik^2, alpha = sqrt, beta = -sgn(a_kk) alpha, v = x - beta e_k,
// two = 2 / v^T v, a_j -= v (two v^T a_j); every mpfr operation rounded. workers: trailing columns in parallel. On return A
// holds R (upper), V the reflectors, tw the scalars.
double mpfr_householder(int bits,MPArray& A,std::size_t m,std::size_t n,MPArray& V,MPArray& tw,Workers* workers){
    auto start=Clock::now();mpfr_t nrm,t,beta;mpfr_inits2(bits,nrm,t,beta,(mpfr_ptr)0);
    for(std::size_t k=0;k<n;++k){mpfr_set_zero(nrm,1);for(std::size_t i=k;i<m;++i){mpfr_sqr(t,A[i*n+k],MPFR_RNDN);mpfr_add(nrm,nrm,t,MPFR_RNDN);}
        mpfr_sqrt(beta,nrm,MPFR_RNDN);if(mpfr_zero_p(beta)){mpfr_set_zero(tw[k],1);continue;}if(mpfr_sgn(A[k*n+k])>=0)mpfr_neg(beta,beta,MPFR_RNDN);
        for(std::size_t i=k;i<m;++i)mpfr_set(V[i*n+k],A[i*n+k],MPFR_RNDN);mpfr_sub(V[k*n+k],V[k*n+k],beta,MPFR_RNDN);
        mpfr_set_zero(nrm,1);for(std::size_t i=k;i<m;++i){mpfr_sqr(t,V[i*n+k],MPFR_RNDN);mpfr_add(nrm,nrm,t,MPFR_RNDN);}mpfr_ui_div(tw[k],2,nrm,MPFR_RNDN);
        auto apply=[&](std::size_t a,std::size_t b){mpfr_t s,u;mpfr_init2(s,bits);mpfr_init2(u,bits);
            for(std::size_t j=k+1+a;j<k+1+b;++j){mpfr_set_zero(s,1);for(std::size_t i=k;i<m;++i){mpfr_mul(u,V[i*n+k],A[i*n+j],MPFR_RNDN);mpfr_add(s,s,u,MPFR_RNDN);}
                mpfr_mul(s,s,tw[k],MPFR_RNDN);for(std::size_t i=k;i<m;++i){mpfr_mul(u,V[i*n+k],s,MPFR_RNDN);mpfr_sub(A[i*n+j],A[i*n+j],u,MPFR_RNDN);}}
            mpfr_clear(s);mpfr_clear(u);};
        if(workers&&n-k>32)workers->run(n-k-1,apply);else apply(0,n-k-1);
        mpfr_set(A[k*n+k],beta,MPFR_RNDN);for(std::size_t i=k+1;i<m;++i)mpfr_set_zero(A[i*n+k],1);}
    mpfr_clears(nrm,t,beta,(mpfr_ptr)0);return std::chrono::duration<double>(Clock::now()-start).count();
}
// X (n x nrhs) from B (m x nrhs, overwritten by Q^T B): reflectors in order, then back substitution (rounded operations).
double mpfr_qr_solve(int bits,MPArray& A,MPArray& V,MPArray& tw,std::size_t m,std::size_t n,MPArray& B,std::size_t nrhs,MPArray& X){
    auto start=Clock::now();mpfr_t s,u;mpfr_init2(s,bits);mpfr_init2(u,bits);
    for(std::size_t c=0;c<nrhs;++c){
        for(std::size_t k=0;k<n;++k){if(mpfr_zero_p(tw[k]))continue;mpfr_set_zero(s,1);for(std::size_t i=k;i<m;++i){mpfr_mul(u,V[i*n+k],B[i*nrhs+c],MPFR_RNDN);mpfr_add(s,s,u,MPFR_RNDN);}
            mpfr_mul(s,s,tw[k],MPFR_RNDN);for(std::size_t i=k;i<m;++i){mpfr_mul(u,V[i*n+k],s,MPFR_RNDN);mpfr_sub(B[i*nrhs+c],B[i*nrhs+c],u,MPFR_RNDN);}}
        for(std::size_t i=n;i-->0;){mpfr_set(s,B[i*nrhs+c],MPFR_RNDN);for(std::size_t j=i+1;j<n;++j){mpfr_mul(u,A[i*n+j],X[j*nrhs+c],MPFR_RNDN);mpfr_sub(s,s,u,MPFR_RNDN);}
            mpfr_div(X[i*nrhs+c],s,A[i*n+i],MPFR_RNDN);}}
    mpfr_clear(s);mpfr_clear(u);return std::chrono::duration<double>(Clock::now()-start).count();
}
struct Options {int repeats=5;std::size_t serial_max=200,parallel_max=1000,nrhs=4;unsigned threads=std::thread::hardware_concurrency();QROptions factor;bool host_only=true;};
double median(std::vector<double> x){return quantile(x,.5);}
template<int Bits> bool identical(const void* a,const void* b,std::size_t count){auto x=static_cast<const Float<Bits>*>(a),y=static_cast<const Float<Bits>*>(b);
    for(std::size_t i=0;i<count;++i)if(!reference::equal<Bits>(x[i],y[i]))return false;return true;}
// Normal equations: A = J^T J (lower), c = J^T B, Cholesky, cholesky_solve; returns the wall time.
template<int Bits> double normal_equations(Linalg& la,const std::vector<Float<Bits>>& J,std::size_t m,std::size_t n,const std::vector<Float<Bits>>& B,std::size_t nrhs,std::vector<Float<Bits>>& X,bool& ok){
    auto t=Clock::now();std::vector<Float<Bits>> A(n*n,zero<Bits/32>()),C(n*nrhs);la.syrk(Bits,J.data(),m,n,A.data(),true);la.gemm(Bits,true,J.data(),B.data(),n,nrhs,m,C.data());
    CholeskyInfo info=la.cholesky(Bits,A.data(),n,A.data());ok=info.pivot==n;X.assign(n*nrhs,zero<Bits/32>());if(ok)la.cholesky_solve(Bits,A.data(),n,C.data(),nrhs,X.data());
    return std::chrono::duration<double>(Clock::now()-t).count();
}
std::map<std::pair<int,std::size_t>,double> serial_cache; // (bits, measured size) -> seconds
template<int Bits> void bench(Linalg& la,Workers& workers,const Options& o,std::size_t n,std::mt19937_64& rng,bool header){
    if(header)std::cout<<"bits,m,n,nrhs,block,host_macs,repeats,qr_wall_ms,qr_wall_min_ms,qr_gpu_ms,qr_panel_ms,qr_update_ms,gpu_updates,host_updates,solve_wall_ms,solve_gpu_ms,"
                         "qr_e2e_ms,host_only_qr_ms,gpu_equals_host,ne_e2e_ms,ne_e2e_min_ms,qr_over_ne,mpfr_serial_qr_s,serial_measured_n,mpfr_parallel_qr_s,mpfr_threads,"
                         "mpfr_solve_s,speedup_vs_serial,speedup_vs_parallel\n";
    const std::size_t m=2*n,nrhs=o.nrhs;auto J=jacobian<Bits>(m,n,rng);std::vector<Float<Bits>> B(m*nrhs),X(n*nrhs),XN;for(auto& b:B)b=reference::random_number<Bits>(rng,20);
    bool ok=true;{QRFactor q=la.factor_qr(Bits,J.data(),m,n,o.factor);q.solve(B.data(),nrhs,X.data());normal_equations<Bits>(la,J,m,n,B,nrhs,XN,ok);} // warm-up (pipelines)
    std::vector<double> wall,gpu,panel,update,swall,sgpu,e2e,ne;QRInfo info;QRFactor qr;
    for(int r=0;r<o.repeats;++r){
        auto t=Clock::now();qr=la.factor_qr(Bits,J.data(),m,n,o.factor);info=qr.info();if(!qr.full_rank())throw std::runtime_error("benchmark J is rank deficient");
        Timing s=qr.solve(B.data(),nrhs,X.data());e2e.push_back(std::chrono::duration<double>(Clock::now()-t).count());
        wall.push_back(info.timing.wall_seconds);gpu.push_back(info.timing.gpu_seconds);panel.push_back(info.panel_seconds);update.push_back(info.update_seconds);
        swall.push_back(s.wall_seconds);sgpu.push_back(s.gpu_seconds);
        ne.push_back(normal_equations<Bits>(la,J,m,n,B,nrhs,XN,ok));if(!ok)throw std::runtime_error("normal equations not positive definite");}
    // Every update on the host: bit-identical factor (the GPU only changes speed).
    double host_ms=0;bool same=true;
    if(o.host_only){QROptions h=o.factor;h.gpu=false;auto th=Clock::now();QRFactor hq=la.factor_qr(Bits,J.data(),m,n,h);host_ms=std::chrono::duration<double>(Clock::now()-th).count()*1e3;
        same=identical<Bits>(hq.r(),qr.r(),n*n)&&identical<Bits>(hq.v(),qr.v(),m*n)&&identical<Bits>(hq.tau(),qr.tau(),n);if(!same)throw std::runtime_error("GPU and host-only QR differ");}
    // MPFR baselines: serial measured at n <= serial_max (else the leading serial_max columns of a 2*serial_max-row J, scaled by n^3).
    auto load=[&](MPArray& a,std::size_t rows,std::size_t cols){for(std::size_t i=0;i<rows;++i)for(std::size_t j=0;j<cols;++j)to_mpfr<Bits>(a[i*cols+j],J[i*n+j]);};
    std::size_t s=std::min(n,o.serial_max);double serial=0,parallel=0,msolve=0;auto key=std::make_pair(Bits,s);
    if(n==s||!serial_cache.count(key)){MPArray a(2*s*s,Bits),v(2*s*s,Bits),tw(s,Bits);load(a,2*s,s);serial_cache[key]=mpfr_householder(Bits,a,2*s,s,v,tw,nullptr);}
    serial=serial_cache[key]*std::pow(double(n)/double(s),3);
    if(n<=o.parallel_max){MPArray a(m*n,Bits),v(m*n,Bits),tw(n,Bits),b(m*nrhs,Bits),x(n*nrhs,Bits);load(a,m,n);for(std::size_t q=0;q<m*nrhs;++q)to_mpfr<Bits>(b[q],B[q]);
        parallel=mpfr_householder(Bits,a,m,n,v,tw,&workers);msolve=mpfr_qr_solve(Bits,a,v,tw,m,n,b,nrhs,x);}
    std::cout<<std::setprecision(5)<<Bits<<','<<m<<','<<n<<','<<nrhs<<','<<o.factor.block<<','<<o.factor.host_macs<<','<<o.repeats<<','<<median(wall)*1e3<<','<<quantile(wall,0)*1e3<<','
        <<median(gpu)*1e3<<','<<median(panel)*1e3<<','<<median(update)*1e3<<','<<info.gpu_updates<<','<<info.host_updates<<','<<median(swall)*1e3<<','<<median(sgpu)*1e3<<','
        <<median(e2e)*1e3<<','<<host_ms<<','<<same<<','<<median(ne)*1e3<<','<<quantile(ne,0)*1e3<<','<<median(e2e)/median(ne)<<','<<serial<<','<<s<<','<<parallel<<','<<workers.size()<<','
        <<msolve<<','<<serial/median(wall)<<','<<(parallel?parallel/median(wall):0)<<std::endl;
}
// Block-size sweep (factor + solve wall, medians).
template<int Bits> void sweep(Linalg& la,const Options& o,std::size_t n,std::mt19937_64& rng,bool header){
    if(header)std::cout<<"bits,m,n,block,host_macs,repeats,qr_wall_ms,qr_wall_min_ms,qr_gpu_ms,qr_panel_ms,qr_update_ms,gpu_updates,host_updates,solve_wall_ms\n";
    const std::size_t m=2*n;auto J=jacobian<Bits>(m,n,rng);std::vector<Float<Bits>> B(m*o.nrhs),X(n*o.nrhs);for(auto& b:B)b=reference::random_number<Bits>(rng,20);
    for(std::size_t block:{8,16,24,32,48,64})for(double hm:{5e4,3e5}){QROptions f=o.factor;f.block=block;f.host_macs=hm;la.factor_qr(Bits,J.data(),m,n,f);
        std::vector<double> wall,gpu,panel,update,solve;QRInfo info;
        for(int r=0;r<o.repeats;++r){QRFactor q=la.factor_qr(Bits,J.data(),m,n,f);info=q.info();wall.push_back(info.timing.wall_seconds);gpu.push_back(info.timing.gpu_seconds);
            panel.push_back(info.panel_seconds);update.push_back(info.update_seconds);solve.push_back(q.solve(B.data(),o.nrhs,X.data()).wall_seconds);}
        std::cout<<std::setprecision(5)<<Bits<<','<<m<<','<<n<<','<<block<<','<<hm<<','<<o.repeats<<','<<median(wall)*1e3<<','<<quantile(wall,0)*1e3<<','<<median(gpu)*1e3<<','
            <<median(panel)*1e3<<','<<median(update)*1e3<<','<<info.gpu_updates<<','<<info.host_updates<<','<<median(solve)*1e3<<std::endl;}
}
// log2 of a nonnegative MPFR value (-1e9 for zero).
double lg(mpfr_srcptr x){if(mpfr_zero_p(x))return -1e9;long e;double d=mpfr_get_d_2exp(&e,x,MPFR_RNDN);return std::log2(std::fabs(d))+double(e);}
// Errors of least-squares solutions against x* from an MPFR Householder solve at P = 3*bits + 2*kappa bits (J and b exact).
// Each J is factored once (x*, LimbForge QR, consumer MPFR at bits) for both right-hand-side kinds: random b (large
// residual) and consistent b = RN(J x0) (residual at rounding level).
template<int Bits> void accuracy(Linalg& la,Workers& workers,const Options& o,std::size_t n,int kappa,std::mt19937_64& rng,bool header){
    if(header)std::cout<<"bits,m,n,kappa,rhs,nrhs,method,log2_fwd_norm,log2_fwd_comp,log2_ne_bwd_comp,log2_rel_residual,log2_opt_residual,log2_residual_excess\n";
    const std::size_t m=2*n,nrhs=o.nrhs;const int P=3*Bits+2*std::max(kappa,0),W=P+64;auto J=kappa<0?jacobian<Bits>(m,n,rng):conditioned<Bits>(la,m,n,kappa,rng);
    MPArray Jm(m*n,Bits),a(m*n,P),v(m*n,P),tw(n,P),c(m*n,Bits),cv(m*n,Bits),ct(n,Bits);
    for(std::size_t q=0;q<m*n;++q){to_mpfr<Bits>(Jm[q],J[q]);mpfr_set(a[q],Jm[q],MPFR_RNDN);mpfr_set(c[q],Jm[q],MPFR_RNDN);}
    mpfr_householder(P,a,m,n,v,tw,&workers);mpfr_householder(Bits,c,m,n,cv,ct,&workers);
    QRFactor qr=la.factor_qr(Bits,J.data(),m,n,o.factor);if(!qr.full_rank())throw std::runtime_error("accuracy J rank deficient");
    for(bool consistent:{false,true}){std::vector<Float<Bits>> B(m*nrhs);
        if(consistent){std::vector<Float<Bits>> x0(n*nrhs);for(auto& x:x0)x=reference::random_number<Bits>(rng,4);la.gemm(Bits,false,J.data(),x0.data(),m,nrhs,n,B.data());}
        else for(auto& b:B)b=reference::random_number<Bits>(rng,4);
        // b stays the right-hand side; bq and cb are overwritten by Q^T b in the MPFR solves.
        MPArray b(m*nrhs,P),bq(m*nrhs,P),xs(n*nrhs,P),cb(m*nrhs,Bits),cx(n*nrhs,Bits);
        for(std::size_t q=0;q<m*nrhs;++q){to_mpfr<Bits>(b[q],B[q]);mpfr_set(bq[q],b[q],MPFR_RNDN);mpfr_set(cb[q],b[q],MPFR_RNDN);}
        mpfr_qr_solve(P,a,v,tw,m,n,bq,nrhs,xs);mpfr_qr_solve(Bits,c,cv,ct,m,n,cb,nrhs,cx);
        auto report=[&](const char* method,auto get){ // get(q, out): solution entry q
            double fn=-1e9,fc=-1e9,bc=-1e9,rr=-1e9,ro=-1e9,rx=-1e9;
            for(std::size_t col=0;col<nrhs;++col){MPArray x(n,W),r(m,W),rs(m,W),scale(m,W),s(8,W);for(int k=0;k<8;++k)mpfr_set_zero(s[k],1);mpfr_ptr d=s[0],mx=s[1],mxs=s[2],t=s[3],u=s[4],bn=s[5],rn=s[6],rsn=s[7];
                for(std::size_t i=0;i<n;++i){get(i*nrhs+col,x[i]);mpfr_sub(d,x[i],xs[i*nrhs+col],MPFR_RNDN);mpfr_abs(d,d,MPFR_RNDN);if(mpfr_cmp(d,mx)>0)mpfr_set(mx,d,MPFR_RNDN);
                    mpfr_abs(t,xs[i*nrhs+col],MPFR_RNDN);if(mpfr_cmp(t,mxs)>0)mpfr_set(mxs,t,MPFR_RNDN);if(!mpfr_zero_p(t)&&!mpfr_zero_p(d)){mpfr_div(u,d,t,MPFR_RNDN);fc=std::max(fc,lg(u));}}
                if(!mpfr_zero_p(mxs)){mpfr_div(u,mx,mxs,MPFR_RNDN);fn=std::max(fn,lg(u));}
                // r = b - J x, r* = b - J x* and |b| + |J||x| per row (W bits; J x products exact).
                workers.run(m,[&](std::size_t i0,std::size_t i1){MPArray w(4,W);for(std::size_t i=i0;i<i1;++i){mpfr_set_zero(w[0],1);mpfr_set_zero(w[1],1);mpfr_set_zero(w[2],1);
                    for(std::size_t j=0;j<n;++j){mpfr_mul(w[3],Jm[i*n+j],x[j],MPFR_RNDN);mpfr_add(w[0],w[0],w[3],MPFR_RNDN);mpfr_abs(w[3],w[3],MPFR_RNDN);mpfr_add(w[1],w[1],w[3],MPFR_RNDN);
                        mpfr_mul(w[3],Jm[i*n+j],xs[j*nrhs+col],MPFR_RNDN);mpfr_add(w[2],w[2],w[3],MPFR_RNDN);}
                    mpfr_sub(r[i],b[i*nrhs+col],w[0],MPFR_RNDN);mpfr_sub(rs[i],b[i*nrhs+col],w[2],MPFR_RNDN);mpfr_abs(w[3],b[i*nrhs+col],MPFR_RNDN);mpfr_add(scale[i],w[1],w[3],MPFR_RNDN);}});
                for(std::size_t i=0;i<m;++i){mpfr_sqr(t,b[i*nrhs+col],MPFR_RNDN);mpfr_add(bn,bn,t,MPFR_RNDN);mpfr_sqr(t,r[i],MPFR_RNDN);mpfr_add(rn,rn,t,MPFR_RNDN);mpfr_sqr(t,rs[i],MPFR_RNDN);mpfr_add(rsn,rsn,t,MPFR_RNDN);}
                if(!mpfr_zero_p(bn)){mpfr_div(u,rn,bn,MPFR_RNDN);rr=std::max(rr,lg(u)/2);mpfr_div(u,rsn,bn,MPFR_RNDN);ro=std::max(ro,lg(u)/2);}
                // Residual excess ||J(x - x*)|| / ||r*|| = sqrt(||r||^2 - ||r*||^2) / ||r*|| (x* optimal).
                if(!mpfr_zero_p(rsn)){mpfr_sub(u,rn,rsn,MPFR_RNDN);mpfr_abs(u,u,MPFR_RNDN);mpfr_div(u,u,rsn,MPFR_RNDN);rx=std::max(rx,lg(u)/2);}
                // Normal-equation componentwise backward error: max_j |J^T r|_j / (|J|^T (|b| + |J||x|))_j.
                std::vector<double> e(n,-1e9);
                workers.run(n,[&](std::size_t j0,std::size_t j1){MPArray w(4,W);for(std::size_t j=j0;j<j1;++j){mpfr_set_zero(w[0],1);mpfr_set_zero(w[1],1);
                    for(std::size_t i=0;i<m;++i){mpfr_mul(w[2],Jm[i*n+j],r[i],MPFR_RNDN);mpfr_add(w[0],w[0],w[2],MPFR_RNDN);mpfr_abs(w[3],Jm[i*n+j],MPFR_RNDN);mpfr_mul(w[2],w[3],scale[i],MPFR_RNDN);mpfr_add(w[1],w[1],w[2],MPFR_RNDN);}
                    mpfr_abs(w[0],w[0],MPFR_RNDN);if(!mpfr_zero_p(w[0])&&!mpfr_zero_p(w[1])){mpfr_div(w[2],w[0],w[1],MPFR_RNDN);e[j]=lg(w[2]);}}});
                for(double x:e)bc=std::max(bc,x);}
            std::cout<<std::setprecision(5)<<Bits<<','<<m<<','<<n<<','<<kappa<<','<<(consistent?"consistent":"random")<<','<<nrhs<<','<<method<<','<<fn<<','<<fc<<','<<bc<<','<<rr<<','<<ro<<','<<rx<<std::endl;};
        std::vector<Float<Bits>> X(n*nrhs),XN;qr.solve(B.data(),nrhs,X.data());
        report("limbforge_qr",[&](std::size_t q,mpfr_ptr out){to_mpfr<Bits>(out,X[q]);});
        bool ok=true;normal_equations<Bits>(la,J,m,n,B,nrhs,XN,ok);
        if(ok)report("limbforge_normal_equations",[&](std::size_t q,mpfr_ptr out){to_mpfr<Bits>(out,XN[q]);});
        else std::cout<<Bits<<','<<m<<','<<n<<','<<kappa<<','<<(consistent?"consistent":"random")<<','<<nrhs<<",limbforge_normal_equations,cholesky_failed,,,,\n";
        report("mpfr_householder",[&](std::size_t q,mpfr_ptr out){mpfr_set(out,cx[q],MPFR_RNDN);});}
}
int main(int argc,char** argv){try{
    Options o;std::vector<int> bits={224,256};std::vector<std::size_t> sizes={200,400,800,1000};std::vector<int> kappas={-1,40,100};std::string mode="time";unsigned host_threads=0;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto next=[&]{if(i+1>=argc)throw std::invalid_argument("missing value");return std::string(argv[++i]);};
        if(a=="--bits")bits={std::stoi(next())};else if(a=="--size")sizes={std::size_t(std::stoul(next()))};else if(a=="--repeats")o.repeats=std::stoi(next());
        else if(a=="--threads")o.threads=unsigned(std::stoul(next()));else if(a=="--serial-max")o.serial_max=std::stoul(next());else if(a=="--parallel-max")o.parallel_max=std::stoul(next());
        else if(a=="--nrhs")o.nrhs=std::stoul(next());else if(a=="--block")o.factor.block=std::stoul(next());else if(a=="--host-macs")o.factor.host_macs=std::stod(next());
        else if(a=="--kappa")kappas={std::stoi(next())};else if(a=="--host-threads")host_threads=unsigned(std::stoul(next()));else if(a=="--no-host-only")o.host_only=false;
        else if(a=="--sweep")mode="sweep";else if(a=="--accuracy")mode="accuracy";
        else throw std::invalid_argument("usage: qr_limbforge [--bits 224|256] [--size n] [--repeats r] [--threads t] [--serial-max n] [--parallel-max n] [--nrhs k] [--block nb] [--host-macs x] [--no-host-only] [--sweep|--accuracy [--kappa k (-1: QSC-like)]]");}
    LinalgOptions lo;lo.host_threads=host_threads;Linalg la(lo);Workers workers(o.threads);std::mt19937_64 rng(20261031);bool header=true;
    std::cerr<<la.device_name()<<"; times: wall = host call (median), gpu = summed command buffers\n";
    for(int b:bits)for(auto n:sizes){
        auto go=[&](auto tag){constexpr int B=decltype(tag)::value;if(mode=="time")bench<B>(la,workers,o,n,rng,header);else if(mode=="sweep")sweep<B>(la,o,n,rng,header);
            else for(int k:kappas){accuracy<B>(la,workers,o,n,k,rng,header);header=false;}};
        if(b==224)go(std::integral_constant<int,224>{});else if(b==256)go(std::integral_constant<int,256>{});else throw std::invalid_argument("bits: 224 or 256");header=false;}
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
