// Round 43 (plan S5): complex Householder QR least squares on QSC-like complex J (m = 2n). Linalg::factor_qr_complex +
// ComplexQRFactor::solve, against the real QR (Linalg::factor_qr + solve) of the equivalent real 2m x 2n embedding
// [[Re J, -Im J], [Im J, Re J]] with right-hand sides [Re b; Im b], and against a consumer-style MPC Householder (qsccpp
// mx.hpp QR: unblocked, every operation rounded; serial and multithreaded). Modes: default timing table (interleaved repeats),
// --accuracy (errors against a high-precision MPC Householder solve: complex QR, real QR of the embedding, normal equations
// of the embedding, consumer MPC Householder).
#import <Foundation/Foundation.h>
#include "limbforge/linalg.hpp"
#include "benchmark_support.hpp"
#include <cmath>
#include <cstdlib>
#include <map>
template<int Bits> using CF=Complex<Bits/32>;
double load_average(){double l[1];return getloadavg(l,1)==1?l[0]:-1;}
// QSC-like complex Jacobian: column scale 2^[-60,60], both parts within 2^-25..2^5 of it, 25% zeros, one entry 2^-150 below
// the column scale in every 17th column (as benchmarks/qr.mm, per component).
template<int Bits> std::vector<CF<Bits>> jacobian(std::size_t rows,std::size_t cols,std::mt19937_64& rng){
    std::vector<CF<Bits>> A(rows*cols);
    for(std::size_t c=0;c<cols;++c){long scale=long(rng()%121)-60;
        for(std::size_t r=0;r<rows;++r){CF<Bits> z{reference::random_number<Bits>(rng,15),reference::random_number<Bits>(rng,15)};z.re.exponent+=int(scale-10);z.im.exponent+=int(scale-10);
            if(rng()%4==0)z={zero<Bits/32>(),zero<Bits/32>()};A[r*cols+c]=z;}
        if(c%17==5){auto& z=A[(rng()%rows)*cols+c];z.re.exponent=int(scale-150);z.im.exponent=int(scale-150);}}
    return A;
}
struct MCArray {
    std::unique_ptr<mpc_t[]> values;std::size_t size;
    MCArray(std::size_t n,int bits):values(new mpc_t[n]),size(n){for(std::size_t i=0;i<n;++i){mpc_init2(values[i],bits);mpc_set_ui(values[i],0,MPC_RNDNN);}}
    ~MCArray(){for(std::size_t i=0;i<size;++i)mpc_clear(values[i]);}
    mpc_ptr operator[](std::size_t i){return values[i];}
};
// Ill-conditioned complex J = G1 diag(2^(-kappa j/(n-1))) G2 (complex Gaussian-like G1, G2; products in MPC at 2 Bits + 64, one
// rounding per component), then the QSC column scales 2^[-60, 60] (exact): singular values spread over about kappa binades.
template<int Bits> std::vector<CF<Bits>> conditioned(std::size_t m,std::size_t n,int kappa,std::mt19937_64& rng,Workers& workers){
    std::vector<CF<Bits>> G1(m*n),G2(n*n),J(m*n);for(auto* g:{&G1,&G2})for(auto& z:*g)z={reference::random_number<Bits>(rng,1),reference::random_number<Bits>(rng,1)};
    for(std::size_t r=0;r<m;++r)for(std::size_t c=0;c<n;++c){int s=int(std::lround(double(kappa)*double(c)/double(std::max<std::size_t>(n-1,1))));G1[r*n+c].re.exponent-=s;G1[r*n+c].im.exponent-=s;}
    const int P=2*Bits+64;
    workers.run(m,[&](std::size_t r0,std::size_t r1){MCArray t(3,P);for(std::size_t r=r0;r<r1;++r)for(std::size_t c=0;c<n;++c){mpc_set_ui(t[0],0,MPC_RNDNN);
        for(std::size_t k=0;k<n;++k){to_mpc<Bits>(t[1],G1[r*n+k]);to_mpc<Bits>(t[2],G2[k*n+c]);mpc_mul(t[1],t[1],t[2],MPC_RNDNN);mpc_add(t[0],t[0],t[1],MPC_RNDNN);}
        J[r*n+c]=from_mpc<Bits>(t[0]);}});
    for(std::size_t c=0;c<n;++c){int s=int(rng()%121)-60;for(std::size_t r=0;r<m;++r){if(J[r*n+c].re.sign)J[r*n+c].re.exponent+=s;if(J[r*n+c].im.sign)J[r*n+c].im.exponent+=s;}}
    return J;
}
// Real embedding (2m x 2n) and stacked right-hand sides [Re B; Im B] (2m x nrhs); solutions come back as [Re x; Im x].
template<int Bits> std::vector<Float<Bits>> embedding(const std::vector<CF<Bits>>& J,std::size_t m,std::size_t n){std::vector<Float<Bits>> E(4*m*n);
    for(std::size_t r=0;r<m;++r)for(std::size_t c=0;c<n;++c){const auto& z=J[r*n+c];E[r*2*n+c]=z.re;E[r*2*n+n+c]=negate(z.im);E[(m+r)*2*n+c]=z.im;E[(m+r)*2*n+n+c]=z.re;}return E;}
template<int Bits> std::vector<Float<Bits>> stacked(const std::vector<CF<Bits>>& B,std::size_t m,std::size_t nrhs){std::vector<Float<Bits>> S(2*m*nrhs);
    for(std::size_t r=0;r<m;++r)for(std::size_t c=0;c<nrhs;++c){S[r*nrhs+c]=B[r*nrhs+c].re;S[(m+r)*nrhs+c]=B[r*nrhs+c].im;}return S;}
// Consumer-style MPC Householder (mx.hpp QR): nrm2 = sum |a_ik|^2, alpha = sqrt(nrm2), phase = a_kk / |a_kk| (1 for zero),
// ak = -phase alpha, v = x - ak e_k, two = 2 / sum |v_i|^2, a_j -= v (two v^H a_j) for j >= k; every MPC/MPFR operation rounded.
// workers: trailing columns in parallel. On return A holds R, V the reflectors, tw the scalars.
double mpc_householder(int bits,MCArray& A,std::size_t m,std::size_t n,MCArray& V,MCArray& tw,Workers* workers){
    auto start=Clock::now();mpfr_t nrm,t,alpha;mpfr_inits2(bits,nrm,t,alpha,(mpfr_ptr)0);mpc_t ak;mpc_init2(ak,bits);
    for(std::size_t k=0;k<n;++k){mpfr_set_zero(nrm,1);for(std::size_t i=k;i<m;++i){mpc_norm(t,A[i*n+k],MPFR_RNDN);mpfr_add(nrm,nrm,t,MPFR_RNDN);}
        mpfr_sqrt(alpha,nrm,MPFR_RNDN);if(mpfr_zero_p(alpha)){mpc_set_ui(tw[k],0,MPC_RNDNN);continue;}
        if(mpc_cmp_si(A[k*n+k],0)==0)mpc_set_ui(ak,1,MPC_RNDNN);else{mpc_norm(t,A[k*n+k],MPFR_RNDN);mpfr_sqrt(t,t,MPFR_RNDN);mpc_div_fr(ak,A[k*n+k],t,MPC_RNDNN);}
        mpc_mul_fr(ak,ak,alpha,MPC_RNDNN);mpc_neg(ak,ak,MPC_RNDNN);
        for(std::size_t i=k;i<m;++i)mpc_set(V[i*n+k],A[i*n+k],MPC_RNDNN);mpc_sub(V[k*n+k],V[k*n+k],ak,MPC_RNDNN);
        mpfr_set_zero(nrm,1);for(std::size_t i=k;i<m;++i){mpc_norm(t,V[i*n+k],MPFR_RNDN);mpfr_add(nrm,nrm,t,MPFR_RNDN);}mpfr_ui_div(t,2,nrm,MPFR_RNDN);mpc_set_fr(tw[k],t,MPC_RNDNN);
        auto apply=[&](std::size_t a,std::size_t b){mpc_t s,u,c;mpc_init2(s,bits);mpc_init2(u,bits);mpc_init2(c,bits);
            for(std::size_t j=k+1+a;j<k+1+b;++j){mpc_set_ui(s,0,MPC_RNDNN);for(std::size_t i=k;i<m;++i){mpc_conj(c,V[i*n+k],MPC_RNDNN);mpc_mul(u,c,A[i*n+j],MPC_RNDNN);mpc_add(s,s,u,MPC_RNDNN);}
                mpc_mul(s,s,tw[k],MPC_RNDNN);for(std::size_t i=k;i<m;++i){mpc_mul(u,V[i*n+k],s,MPC_RNDNN);mpc_sub(A[i*n+j],A[i*n+j],u,MPC_RNDNN);}}
            mpc_clear(s);mpc_clear(u);mpc_clear(c);};
        if(workers&&n-k>32)workers->run(n-k-1,apply);else apply(0,n-k-1);
        mpc_set(A[k*n+k],ak,MPC_RNDNN);for(std::size_t i=k+1;i<m;++i)mpc_set_ui(A[i*n+k],0,MPC_RNDNN);}
    mpfr_clears(nrm,t,alpha,(mpfr_ptr)0);mpc_clear(ak);return std::chrono::duration<double>(Clock::now()-start).count();
}
// X (n x nrhs) from B (m x nrhs, overwritten by Q^H B): reflectors in order, then back substitution (rounded operations).
double mpc_qr_solve(int bits,MCArray& A,MCArray& V,MCArray& tw,std::size_t m,std::size_t n,MCArray& B,std::size_t nrhs,MCArray& X){
    auto start=Clock::now();mpc_t s,u,c;mpc_init2(s,bits);mpc_init2(u,bits);mpc_init2(c,bits);
    for(std::size_t col=0;col<nrhs;++col){
        for(std::size_t k=0;k<n;++k){if(mpc_cmp_si(tw[k],0)==0)continue;mpc_set_ui(s,0,MPC_RNDNN);
            for(std::size_t i=k;i<m;++i){mpc_conj(c,V[i*n+k],MPC_RNDNN);mpc_mul(u,c,B[i*nrhs+col],MPC_RNDNN);mpc_add(s,s,u,MPC_RNDNN);}
            mpc_mul(s,s,tw[k],MPC_RNDNN);for(std::size_t i=k;i<m;++i){mpc_mul(u,V[i*n+k],s,MPC_RNDNN);mpc_sub(B[i*nrhs+col],B[i*nrhs+col],u,MPC_RNDNN);}}
        for(std::size_t i=n;i-->0;){mpc_set(s,B[i*nrhs+col],MPC_RNDNN);for(std::size_t j=i+1;j<n;++j){mpc_mul(u,A[i*n+j],X[j*nrhs+col],MPC_RNDNN);mpc_sub(s,s,u,MPC_RNDNN);}
            mpc_div(X[i*nrhs+col],s,A[i*n+i],MPC_RNDNN);}}
    mpc_clear(s);mpc_clear(u);mpc_clear(c);return std::chrono::duration<double>(Clock::now()-start).count();
}
struct Options {int repeats=5;std::size_t serial_max=200,parallel_max=400,nrhs=4;unsigned threads=std::thread::hardware_concurrency();QROptions factor;bool host_only=true,embed=true;};
double median(std::vector<double> x){return quantile(x,.5);}
template<int Bits> bool identical(const void* a,const void* b,std::size_t count){auto x=static_cast<const CF<Bits>*>(a),y=static_cast<const CF<Bits>*>(b);
    for(std::size_t i=0;i<count;++i)if(!reference::equal_complex<Bits>(x[i],y[i]))return false;return true;}
std::map<std::pair<int,std::size_t>,double> serial_cache; // (bits, measured size) -> seconds
template<int Bits> void bench(Linalg& la,Workers& workers,const Options& o,std::size_t n,std::mt19937_64& rng,bool header){
    if(header)std::cout<<"bits,m,n,nrhs,block,repeats,load_before,load_after,cqr_e2e_ms,cqr_e2e_min_ms,cqr_factor_ms,cqr_panel_ms,cqr_update_ms,cqr_gpu_ms,cqr_solve_ms,gpu_updates,host_updates,"
                         "host_only_equal,embed_e2e_ms,embed_e2e_min_ms,embed_factor_ms,embed_over_complex,mpc_serial_s,serial_measured_n,mpc_parallel_s,mpc_threads,mpc_solve_s,speedup_vs_serial,speedup_vs_parallel\n";
    const std::size_t m=2*n,nrhs=o.nrhs;auto J=jacobian<Bits>(m,n,rng);std::vector<CF<Bits>> B(m*nrhs),X(n*nrhs);
    for(auto& b:B)b={reference::random_number<Bits>(rng,20),reference::random_number<Bits>(rng,20)};
    auto E=embedding<Bits>(J,m,n);auto S=stacked<Bits>(B,m,nrhs);std::vector<Float<Bits>> XE(2*n*nrhs);
    {ComplexQRFactor q=la.factor_qr_complex(Bits,J.data(),m,n,o.factor);q.solve(B.data(),nrhs,X.data());if(o.embed){QRFactor e=la.factor_qr(Bits,E.data(),2*m,2*n,o.factor);e.solve(S.data(),nrhs,XE.data());}} // warm-up
    const double l0=load_average();std::vector<double> e2e,factor,panel,update,gpu,solve,ee2e,efactor;QRInfo info;ComplexQRFactor qr;
    for(int r=0;r<o.repeats;++r){
        auto complex_run=[&]{auto t=Clock::now();qr=la.factor_qr_complex(Bits,J.data(),m,n,o.factor);info=qr.info();if(!qr.full_rank())throw std::runtime_error("benchmark J is rank deficient");
            Timing s=qr.solve(B.data(),nrhs,X.data());e2e.push_back(std::chrono::duration<double>(Clock::now()-t).count());factor.push_back(info.timing.wall_seconds);
            panel.push_back(info.panel_seconds);update.push_back(info.update_seconds);gpu.push_back(info.timing.gpu_seconds);solve.push_back(s.wall_seconds);};
        auto embed_run=[&]{if(!o.embed)return;auto t=Clock::now();QRFactor e=la.factor_qr(Bits,E.data(),2*m,2*n,o.factor);if(!e.full_rank())throw std::runtime_error("embedding rank deficient");
            e.solve(S.data(),nrhs,XE.data());ee2e.push_back(std::chrono::duration<double>(Clock::now()-t).count());efactor.push_back(e.info().timing.wall_seconds);};
        if(r%2){embed_run();complex_run();}else{complex_run();embed_run();}}
    const double l1=load_average();
    // Every update on the host: bit-identical factor and solution (the GPU only changes speed).
    bool same=true;
    if(o.host_only){QROptions h=o.factor;h.gpu=false;ComplexQRFactor hq=la.factor_qr_complex(Bits,J.data(),m,n,h);std::vector<CF<Bits>> HX(n*nrhs);hq.solve(B.data(),nrhs,HX.data());
        same=identical<Bits>(hq.r(),qr.r(),n*n)&&identical<Bits>(hq.v(),qr.v(),m*n)&&identical<Bits>(hq.tau(),qr.tau(),n)&&identical<Bits>(HX.data(),X.data(),n*nrhs);
        if(!same)throw std::runtime_error("GPU and host-only complex QR differ");}
    // MPC baselines: serial measured at n <= serial_max (else the leading serial_max columns of a 2*serial_max-row J, scaled by n^3).
    auto load=[&](MCArray& a,std::size_t rows,std::size_t cols){for(std::size_t i=0;i<rows;++i)for(std::size_t j=0;j<cols;++j)to_mpc<Bits>(a[i*cols+j],J[i*n+j]);};
    std::size_t s=std::min(n,o.serial_max);double serial=0,parallel=0,msolve=0;auto key=std::make_pair(Bits,s);
    if(s&&(n==s||!serial_cache.count(key))){MCArray a(2*s*s,Bits),v(2*s*s,Bits),tw(s,Bits);load(a,2*s,s);serial_cache[key]=mpc_householder(Bits,a,2*s,s,v,tw,nullptr);}
    if(s)serial=serial_cache[key]*std::pow(double(n)/double(s),3);
    if(n<=o.parallel_max){MCArray a(m*n,Bits),v(m*n,Bits),tw(n,Bits),b(m*nrhs,Bits),x(n*nrhs,Bits);load(a,m,n);for(std::size_t q=0;q<m*nrhs;++q)to_mpc<Bits>(b[q],B[q]);
        parallel=mpc_householder(Bits,a,m,n,v,tw,&workers);msolve=mpc_qr_solve(Bits,a,v,tw,m,n,b,nrhs,x);}
    const double e=median(e2e);
    std::cout<<std::setprecision(5)<<Bits<<','<<m<<','<<n<<','<<nrhs<<','<<o.factor.block<<','<<o.repeats<<','<<l0<<','<<l1<<','<<e*1e3<<','<<quantile(e2e,0)*1e3<<','
        <<median(factor)*1e3<<','<<median(panel)*1e3<<','<<median(update)*1e3<<','<<median(gpu)*1e3<<','<<median(solve)*1e3<<','<<info.gpu_updates<<','<<info.host_updates<<','<<same<<','
        <<(o.embed?median(ee2e)*1e3:0)<<','<<(o.embed?quantile(ee2e,0)*1e3:0)<<','<<(o.embed?median(efactor)*1e3:0)<<','<<(o.embed?median(ee2e)/e:0)<<','
        <<serial<<','<<s<<','<<parallel<<','<<workers.size()<<','<<msolve<<','<<(serial?serial/e:0)<<','<<(parallel?(parallel+msolve)/e:0)<<std::endl;
}
// log2 of a nonnegative MPFR value (-1e9 for zero).
double lg(mpfr_srcptr x){if(mpfr_zero_p(x))return -1e9;long e;double d=mpfr_get_d_2exp(&e,x,MPFR_RNDN);return std::log2(std::fabs(d))+double(e);}
// Errors of least-squares solutions against x* from an MPC Householder solve at P = 3*bits + 2*kappa bits (J and b exact):
// forward normwise max_i |x_i - x*_i| / max_i |x*_i| (complex moduli) and the residual excess ||J (x - x*)|| / ||b - J x*||.
// Right-hand sides: random b (residual of the size of b) and consistent b = RN(J x0) (residual at the rounding level).
template<int Bits> void accuracy(Linalg& la,Workers& workers,const Options& o,std::size_t n,int kappa,std::mt19937_64& rng,bool header){
    if(header)std::cout<<"bits,m,n,kappa,rhs,nrhs,method,log2_fwd_norm,log2_rel_residual,log2_opt_residual,log2_residual_excess\n";
    const std::size_t m=2*n,nrhs=o.nrhs;const int P=3*Bits+2*std::max(kappa,0),W=P+64;auto J=kappa<0?jacobian<Bits>(m,n,rng):conditioned<Bits>(m,n,kappa,rng,workers);
    MCArray Jm(m*n,Bits),a(m*n,P),v(m*n,P),tw(n,P),c(m*n,Bits),cv(m*n,Bits),ct(n,Bits);
    for(std::size_t q=0;q<m*n;++q){to_mpc<Bits>(Jm[q],J[q]);mpc_set(a[q],Jm[q],MPC_RNDNN);mpc_set(c[q],Jm[q],MPC_RNDNN);}
    mpc_householder(P,a,m,n,v,tw,&workers);mpc_householder(Bits,c,m,n,cv,ct,&workers);
    ComplexQRFactor qr=la.factor_qr_complex(Bits,J.data(),m,n,o.factor);if(!qr.full_rank())throw std::runtime_error("accuracy J rank deficient");
    auto E=embedding<Bits>(J,m,n);QRFactor qe=la.factor_qr(Bits,E.data(),2*m,2*n,o.factor);
    for(bool consistent:{false,true}){std::vector<CF<Bits>> B(m*nrhs);
        if(consistent){std::vector<CF<Bits>> x0(n*nrhs);for(auto& z:x0)z={reference::random_number<Bits>(rng,4),reference::random_number<Bits>(rng,4)};
            workers.run(m,[&](std::size_t r0,std::size_t r1){MCArray t(3,2*Bits+64);for(std::size_t r=r0;r<r1;++r)for(std::size_t col=0;col<nrhs;++col){mpc_set_ui(t[0],0,MPC_RNDNN);
                for(std::size_t k=0;k<n;++k){to_mpc<Bits>(t[1],J[r*n+k]);to_mpc<Bits>(t[2],x0[k*nrhs+col]);mpc_mul(t[1],t[1],t[2],MPC_RNDNN);mpc_add(t[0],t[0],t[1],MPC_RNDNN);}
                B[r*nrhs+col]=from_mpc<Bits>(t[0]);}});}
        else for(auto& z:B)z={reference::random_number<Bits>(rng,4),reference::random_number<Bits>(rng,4)};
        MCArray b(m*nrhs,P),bq(m*nrhs,P),xs(n*nrhs,P),cb(m*nrhs,Bits),cx(n*nrhs,Bits);
        for(std::size_t q=0;q<m*nrhs;++q){to_mpc<Bits>(b[q],B[q]);mpc_set(bq[q],b[q],MPC_RNDNN);mpc_set(cb[q],b[q],MPC_RNDNN);}
        mpc_qr_solve(P,a,v,tw,m,n,bq,nrhs,xs);mpc_qr_solve(Bits,c,cv,ct,m,n,cb,nrhs,cx);
        auto report=[&](const char* method,auto get){ // get(q, out): solution entry q
            double fn=-1e9,rr=-1e9,ro=-1e9,rx=-1e9;
            for(std::size_t col=0;col<nrhs;++col){MCArray x(n,W);MPArray s(6,W);for(int k=0;k<6;++k)mpfr_set_zero(s[k],1);mpfr_ptr mx=s[0],mxs=s[1],t=s[2],bn=s[3],rn=s[4],rsn=s[5];
                {mpc_t d;mpc_init2(d,W);for(std::size_t i=0;i<n;++i){get(i*nrhs+col,x[i]);mpc_sub(d,x[i],xs[i*nrhs+col],MPC_RNDNN);mpc_abs(t,d,MPFR_RNDN);if(mpfr_cmp(t,mx)>0)mpfr_set(mx,t,MPFR_RNDN);
                    mpc_abs(t,xs[i*nrhs+col],MPFR_RNDN);if(mpfr_cmp(t,mxs)>0)mpfr_set(mxs,t,MPFR_RNDN);}mpc_clear(d);}
                if(!mpfr_zero_p(mxs)){mpfr_div(t,mx,mxs,MPFR_RNDN);fn=std::max(fn,lg(t));}
                // r = b - J x, r* = b - J x* (W bits; products exact to W).
                std::vector<double> parts(3*m);MPArray rr2(m,W),rs2(m,W);
                workers.run(m,[&](std::size_t i0,std::size_t i1){MCArray w(3,W);for(std::size_t i=i0;i<i1;++i){mpc_set(w[0],b[i*nrhs+col],MPC_RNDNN);mpc_set(w[1],b[i*nrhs+col],MPC_RNDNN);
                    for(std::size_t j=0;j<n;++j){mpc_mul(w[2],Jm[i*n+j],x[j],MPC_RNDNN);mpc_sub(w[0],w[0],w[2],MPC_RNDNN);mpc_mul(w[2],Jm[i*n+j],xs[j*nrhs+col],MPC_RNDNN);mpc_sub(w[1],w[1],w[2],MPC_RNDNN);}
                    mpc_norm(rr2[i],w[0],MPFR_RNDN);mpc_norm(rs2[i],w[1],MPFR_RNDN);}});
                for(std::size_t i=0;i<m;++i){mpc_norm(t,b[i*nrhs+col],MPFR_RNDN);mpfr_add(bn,bn,t,MPFR_RNDN);mpfr_add(rn,rn,rr2[i],MPFR_RNDN);mpfr_add(rsn,rsn,rs2[i],MPFR_RNDN);}
                if(!mpfr_zero_p(bn)){mpfr_div(t,rn,bn,MPFR_RNDN);rr=std::max(rr,lg(t)/2);mpfr_div(t,rsn,bn,MPFR_RNDN);ro=std::max(ro,lg(t)/2);}
                // Residual excess ||J(x - x*)|| / ||r*|| = sqrt(||r||^2 - ||r*||^2) / ||r*|| (x* optimal).
                if(!mpfr_zero_p(rsn)){mpfr_sub(t,rn,rsn,MPFR_RNDN);mpfr_abs(t,t,MPFR_RNDN);mpfr_div(t,t,rsn,MPFR_RNDN);rx=std::max(rx,lg(t)/2);}}
            std::cout<<std::setprecision(5)<<Bits<<','<<m<<','<<n<<','<<kappa<<','<<(consistent?"consistent":"random")<<','<<nrhs<<','<<method<<','<<fn<<','<<rr<<','<<ro<<','<<rx<<std::endl;};
        std::vector<CF<Bits>> X(n*nrhs);qr.solve(B.data(),nrhs,X.data());
        report("limbforge_complex_qr",[&](std::size_t q,mpc_ptr out){to_mpc<Bits>(out,X[q]);});
        auto S=stacked<Bits>(B,m,nrhs);std::vector<Float<Bits>> XE(2*n*nrhs);qe.solve(S.data(),nrhs,XE.data());
        report("limbforge_real_qr_embedding",[&](std::size_t q,mpc_ptr out){std::size_t i=q/nrhs,col=q%nrhs;to_mpfr<Bits>(mpc_realref(out),XE[i*nrhs+col]);to_mpfr<Bits>(mpc_imagref(out),XE[(n+i)*nrhs+col]);});
        // Normal equations of the embedding (the complex normal equations J^H J x = J^H b in real form): SYRK, gemm, Cholesky, solve.
        {std::vector<Float<Bits>> Ae(4*n*n,zero<Bits/32>()),Ce(2*n*nrhs),XN(2*n*nrhs);la.syrk(Bits,E.data(),2*m,2*n,Ae.data(),true);la.gemm(Bits,true,E.data(),S.data(),2*n,nrhs,2*m,Ce.data());
         CholeskyInfo ci=la.cholesky(Bits,Ae.data(),2*n,Ae.data());
         if(ci.pivot==2*n){la.cholesky_solve(Bits,Ae.data(),2*n,Ce.data(),nrhs,XN.data());
             report("limbforge_normal_equations_embedding",[&](std::size_t q,mpc_ptr out){std::size_t i=q/nrhs,col=q%nrhs;to_mpfr<Bits>(mpc_realref(out),XN[i*nrhs+col]);to_mpfr<Bits>(mpc_imagref(out),XN[(n+i)*nrhs+col]);});}
         else std::cout<<Bits<<','<<m<<','<<n<<','<<kappa<<','<<(consistent?"consistent":"random")<<','<<nrhs<<",limbforge_normal_equations_embedding,cholesky_failed,,,\n";}
        report("mpc_householder",[&](std::size_t q,mpc_ptr out){mpc_set(out,cx[q],MPC_RNDNN);});}
}
int main(int argc,char** argv){try{
    Options o;std::vector<int> bits={224,256};std::vector<std::size_t> sizes={200,400,1000};std::vector<int> kappas={-1,40,100};std::string mode="time";unsigned host_threads=0;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto next=[&]{if(i+1>=argc)throw std::invalid_argument("missing value");return std::string(argv[++i]);};
        if(a=="--bits")bits={std::stoi(next())};else if(a=="--size")sizes={std::size_t(std::stoul(next()))};else if(a=="--repeats")o.repeats=std::stoi(next());
        else if(a=="--threads")o.threads=unsigned(std::stoul(next()));else if(a=="--serial-max")o.serial_max=std::stoul(next());else if(a=="--parallel-max")o.parallel_max=std::stoul(next());
        else if(a=="--nrhs")o.nrhs=std::stoul(next());else if(a=="--block")o.factor.block=std::stoul(next());else if(a=="--host-macs")o.factor.host_macs=std::stod(next());
        else if(a=="--kappa")kappas={std::stoi(next())};else if(a=="--host-threads")host_threads=unsigned(std::stoul(next()));else if(a=="--no-host-only")o.host_only=false;
        else if(a=="--no-embedding")o.embed=false;else if(a=="--accuracy")mode="accuracy";
        else throw std::invalid_argument("usage: qr_complex_limbforge [--bits 224|256] [--size n] [--repeats r] [--threads t] [--serial-max n] [--parallel-max n] [--nrhs k] [--block nb] [--host-macs x] [--no-host-only] [--no-embedding] [--accuracy [--kappa k (-1: QSC-like)]]");}
    LinalgOptions lo;lo.host_threads=host_threads;Linalg la(lo);Workers workers(o.threads);std::mt19937_64 rng(20261043);bool header=true;
    std::cerr<<la.device_name()<<"; times: wall = host call (median of interleaved repeats), gpu = summed command buffers\n";
    for(int b:bits)for(auto n:sizes){
        auto go=[&](auto tag){constexpr int B=decltype(tag)::value;if(mode=="time")bench<B>(la,workers,o,n,rng,header);else for(int k:kappas){accuracy<B>(la,workers,o,n,k,rng,header);header=false;}};
        if(b==224)go(std::integral_constant<int,224>{});else if(b==256)go(std::integral_constant<int,256>{});else throw std::invalid_argument("bits: 224 or 256");header=false;}
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
