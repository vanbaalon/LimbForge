// Damping trials and multi-RHS solves. Every GPU sample is checked against its own MPFR contract.
#include "limbforge/batched_linalg.hpp"
#include "benchmark_support.hpp"
#include <cstdlib>
#include <atomic>
#include <random>
struct Options {int bits=352;std::size_t n=200,count=8,nrhs=3;unsigned workers=18,repeats=5;bool resident=false,check_only=false,warm=false;};
template<int B> void run(const Options& o){
    using F=Float<B>;auto n=o.n,c=o.count,r=o.nrhs,n2=n*n;
    Engine e;BatchedLinalg batched(e);LinalgOptions exact_options;exact_options.host_threads=o.workers;Linalg exact(e,exact_options);Workers pool(o.workers);std::mt19937_64 rng(123+B);
    std::vector<F> J((n+n/6)*n),A(n2),D(n),mu(c),rhs(n*r),expected(c*n2),solution(c*n*r);
    for(auto& v:J)v=reference::random_number<B>(rng,2);
    exact.syrk(B,J.data(),n+n/6,n,A.data(),false);
    for(std::size_t i=0;i<n;++i)D[i]=A[i*n+i];
    auto damping=from_decimal<B>("0.001");for(std::size_t t=0;t<c;++t){mu[t]=damping;mu[t].exponent-=int(t);}
    for(auto& v:rhs)v=reference::random_number<B>(rng,2);
    MPArray input(n2,B),diagonal(n,B),shifts(c,B),right(n*r,B),L(c*n2,B),X(c*n*r,B);
    for(std::size_t i=0;i<n2;++i)to_mpfr<B>(input[i],A[i]);
    for(std::size_t i=0;i<n;++i)to_mpfr<B>(diagonal[i],D[i]);
    for(std::size_t t=0;t<c;++t)to_mpfr<B>(shifts[t],mu[t]);
    for(std::size_t i=0;i<n*r;++i)to_mpfr<B>(right[i],rhs[i]);
    std::atomic<bool> reference_failure{false};
    std::vector<std::unique_ptr<MPArray>> scratch;for(std::size_t t=0;t<c;++t)scratch.emplace_back(new MPArray(34,B));
    auto cpu=[&]{reference_failure=false;pool.run(c,[&](std::size_t first,std::size_t last){
        for(auto t=first;t<last;++t){auto& tmp=*scratch[t];auto off=t*n2,xoff=t*n*r;
            for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j){auto q=i*n+j;if(i>=j)mpfr_set(L[off+q],input[q],MPFR_RNDN);else mpfr_set_zero(L[off+q],1);}
            for(std::size_t i=0;i<n;++i){mpfr_mul(tmp[0],shifts[t],diagonal[i],MPFR_RNDN);mpfr_add(L[off+i*n+i],L[off+i*n+i],tmp[0],MPFR_RNDN);}
            for(std::size_t k=0;k<n;++k){if(mpfr_sgn(L[off+k*n+k])<=0){reference_failure=true;return;}
                mpfr_sqrt(L[off+k*n+k],L[off+k*n+k],MPFR_RNDN);
                for(std::size_t i=k+1;i<n;++i)mpfr_div(L[off+i*n+k],L[off+i*n+k],L[off+k*n+k],MPFR_RNDN);
                for(std::size_t i=k+1;i<n;++i)for(std::size_t j=k+1;j<=i;++j){mpfr_mul(tmp[0],L[off+i*n+k],L[off+j*n+k],MPFR_RNDN);mpfr_sub(L[off+i*n+j],L[off+i*n+j],tmp[0],MPFR_RNDN);}}
            for(std::size_t i=0;i<n*r;++i)mpfr_set(X[xoff+i],right[i],MPFR_RNDN);
            for(std::size_t col=0;col<r;++col){for(std::size_t i=0;i<n;++i){for(std::size_t j=0;j<i;++j){mpfr_mul(tmp[0],L[off+i*n+j],X[xoff+j*r+col],MPFR_RNDN);mpfr_sub(X[xoff+i*r+col],X[xoff+i*r+col],tmp[0],MPFR_RNDN);}mpfr_div(X[xoff+i*r+col],X[xoff+i*r+col],L[off+i*n+i],MPFR_RNDN);}
                for(std::size_t i=n;i-->0;){for(std::size_t j=i+1;j<n;++j){mpfr_mul(tmp[0],L[off+j*n+i],X[xoff+j*r+col],MPFR_RNDN);mpfr_sub(X[xoff+i*r+col],X[xoff+i*r+col],tmp[0],MPFR_RNDN);}mpfr_div(X[xoff+i*r+col],X[xoff+i*r+col],L[off+i*n+i],MPFR_RNDN);}}
        }});if(reference_failure)throw std::runtime_error("benchmark fixture is not SPD");};
    cpu();for(std::size_t i=0;i<expected.size();++i)expected[i]=from_mpfr<B>(L[i]);for(std::size_t i=0;i<solution.size();++i)solution[i]=from_mpfr<B>(X[i]);
    // Independent left-looking replay of Linalg's block-32 exact-dot contract.
    std::vector<F> exact_expected(c*n2,zero<B/32>()),exact_solution(c*n*r);
    pool.run(c,[&](std::size_t first,std::size_t last){for(auto t=first;t<last;++t){auto off=t*n2,xoff=t*n*r;
        auto value=[&](std::size_t i,std::size_t j){F v=A[i*n+j];if(i==j)v=reference::real<B>(Operation::add,v,reference::real<B>(Operation::mul,mu[t],D[i]));auto kb=j/32*32;
            for(std::size_t block=0;block<kb;block+=32)v=reference::dot_sub<B>(v,exact_expected.data()+off+i*n+block,1,exact_expected.data()+off+j*n+block,1,32);
            return reference::dot_sub<B>(v,exact_expected.data()+off+i*n+kb,1,exact_expected.data()+off+j*n+kb,1,j-kb);};
        for(std::size_t j=0;j<n;++j){auto v=value(j,j);if(v.status||v.sign<=0){reference_failure=true;return;}
            exact_expected[off+j*n+j]=reference::real<B>(Operation::sqrt,v,v);
            for(std::size_t i=j+1;i<n;++i)exact_expected[off+i*n+j]=reference::real<B>(Operation::div,value(i,j),exact_expected[off+j*n+j]);}
        std::copy(rhs.begin(),rhs.end(),exact_solution.begin()+xoff);
        auto ls=exact_expected.data()+off,xx=exact_solution.data()+xoff;
        for(std::size_t i=0;i<n;++i)for(std::size_t col=0;col<r;++col){F v=xx[i*r+col];auto kb=i/32*32;
            for(std::size_t block=0;block<kb;block+=32)v=reference::dot_sub<B>(v,ls+i*n+block,1,xx+block*r+col,r,32);
            v=reference::dot_sub<B>(v,ls+i*n+kb,1,xx+kb*r+col,r,i-kb);xx[i*r+col]=reference::real<B>(Operation::div,v,ls[i*n+i]);}
        for(std::size_t i=n;i-->0;)for(std::size_t col=0;col<r;++col){F v=xx[i*r+col];auto kb=i/32*32,end=std::min(n,kb+32);
            for(std::size_t block=(n-1)/32*32;block>=end&&block>kb;block-=32)v=reference::dot_sub<B>(v,ls+block*n+i,n,xx+block*r+col,r,std::min(n,block+32)-block);
            if(i+1<end)v=reference::dot_sub<B>(v,ls+(i+1)*n+i,n,xx+(i+1)*r+col,r,end-i-1);xx[i*r+col]=reference::real<B>(Operation::div,v,ls[i*n+i]);}
    }});if(reference_failure)throw std::runtime_error("exact reference fixture is not SPD");
    // Fast timed exact-contract CPU path; independent from the Float/dot_sub replay above.
    // mpfr_dot includes the previous entry and negated factors, rounding each block once.
    auto cpu_exact=[&]{reference_failure=false;pool.run(c,[&](std::size_t first,std::size_t last){for(auto t=first;t<last;++t){
        auto& tmp=*scratch[t];auto off=t*n2,xoff=t*n*r;mpfr_set_ui(tmp[0],1,MPFR_RNDN);
        auto subtract=[&](mpfr_ptr value,std::size_t ia,std::size_t ib,std::size_t da,std::size_t db,std::size_t terms,bool solve=false){
            mpfr_ptr a[33],b[33];a[0]=value;b[0]=tmp[0];
            for(std::size_t j=0;j<terms;++j){mpfr_neg(tmp[j+1],L[ia+j*da],MPFR_RNDN);a[j+1]=tmp[j+1];b[j+1]=solve?X[ib+j*db]:L[ib+j*db];}
            mpfr_dot(value,a,b,terms+1,MPFR_RNDN);
        };
        for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j){auto q=i*n+j;if(i>=j)mpfr_set(L[off+q],input[q],MPFR_RNDN);else mpfr_set_zero(L[off+q],1);}
        for(std::size_t i=0;i<n;++i){mpfr_mul(tmp[33],shifts[t],diagonal[i],MPFR_RNDN);mpfr_add(L[off+i*n+i],L[off+i*n+i],tmp[33],MPFR_RNDN);}
        auto entry=[&](std::size_t i,std::size_t j){mpfr_set(tmp[33],L[off+i*n+j],MPFR_RNDN);auto kb=j/32*32;
            for(std::size_t block=0;block<kb;block+=32)subtract(tmp[33],off+i*n+block,off+j*n+block,1,1,32);
            subtract(tmp[33],off+i*n+kb,off+j*n+kb,1,1,j-kb);
        };
        for(std::size_t j=0;j<n;++j){entry(j,j);if(mpfr_sgn(tmp[33])<=0){reference_failure=true;return;}
            mpfr_sqrt(L[off+j*n+j],tmp[33],MPFR_RNDN);
            for(std::size_t i=j+1;i<n;++i){entry(i,j);mpfr_div(L[off+i*n+j],tmp[33],L[off+j*n+j],MPFR_RNDN);}}
        for(std::size_t i=0;i<n*r;++i)mpfr_set(X[xoff+i],right[i],MPFR_RNDN);
        for(std::size_t i=0;i<n;++i)for(std::size_t col=0;col<r;++col){auto value=X[xoff+i*r+col];auto kb=i/32*32;
            for(std::size_t block=0;block<kb;block+=32)subtract(value,off+i*n+block,xoff+block*r+col,1,r,32,true);
            subtract(value,off+i*n+kb,xoff+kb*r+col,1,r,i-kb,true);mpfr_div(value,value,L[off+i*n+i],MPFR_RNDN);}
        for(std::size_t i=n;i-->0;)for(std::size_t col=0;col<r;++col){auto value=X[xoff+i*r+col];auto kb=i/32*32,end=std::min(n,kb+32);
            for(std::size_t block=(n-1)/32*32;block>=end&&block>kb;block-=32)subtract(value,off+block*n+i,xoff+block*r+col,n,r,std::min(n,block+32)-block,true);
            if(i+1<end)subtract(value,off+(i+1)*n+i,xoff+(i+1)*r+col,n,r,end-i-1,true);mpfr_div(value,value,L[off+i*n+i],MPFR_RNDN);}
    }});if(reference_failure)throw std::runtime_error("exact timed CPU fixture is not SPD");};
    auto verify_cpu=[&](bool exact_contract){const auto& factor=exact_contract?exact_expected:expected;const auto& solve=exact_contract?exact_solution:solution;
        for(std::size_t i=0;i<factor.size();++i)if(!reference::equal<B>(from_mpfr<B>(L[i]),factor[i]))throw std::runtime_error("CPU factor reference mismatch");
        for(std::size_t i=0;i<solve.size();++i)if(!reference::equal<B>(from_mpfr<B>(X[i]),solve[i]))throw std::runtime_error("CPU solve reference mismatch");};
    cpu_exact();verify_cpu(true);
    auto put=[&](const std::vector<F>& values){auto b=e.make_buffer<F>(values.size());b.upload(values.data(),values.size());return b;};
    auto a=put(A),d=put(D),u=put(mu),b=put(rhs),l=e.make_buffer<F>(c*n2),x=e.make_buffer<F>(c*n*r);auto info=e.make_buffer<std::uint32_t>(c);
    std::vector<F> got(c*n2),sx(c*n*r);std::vector<std::uint32_t> statuses(c);
    unsigned active=0;std::vector<F> shifted(n2),single_factor(n2);std::vector<double> cpu_times[2],wall[3],device[3],factor_wall[3],solve_wall[3];
    const char* paths[]={"sequential_scalar","exact_linalg_loop","exact_batched_panels"};double last_factor=0,last_solve=0,loop_panel=0,loop_update=0;std::size_t loop_gpu=0,loop_host=0;BlockedTrialsInfo last_exact;
    auto gpu=[&]{auto begin=Clock::now();double device_seconds=0;
        if(active==1){loop_panel=loop_update=0;loop_gpu=loop_host=0;auto t0=Clock::now();for(std::size_t t=0;t<c;++t){shifted=A;for(std::size_t i=0;i<n;++i)shifted[i*n+i]=add(shifted[i*n+i],mul(mu[t],D[i]));
                FactorOptions fo;auto factor=exact.cholesky(B,shifted.data(),n,single_factor.data(),fo);if(factor.pivot!=n)throw std::runtime_error("exact loop factor failed");
                device_seconds+=factor.timing.gpu_seconds;loop_panel+=factor.panel_seconds;loop_update+=factor.update_seconds;loop_gpu+=factor.gpu_updates;loop_host+=factor.host_updates;std::copy(single_factor.begin(),single_factor.end(),got.begin()+t*n2);}
            last_factor=std::chrono::duration<double>(Clock::now()-t0).count();t0=Clock::now();
            for(std::size_t t=0;t<c;++t)device_seconds+=exact.cholesky_solve(B,got.data()+t*n2,n,rhs.data(),r,sx.data()+t*n*r).gpu_seconds;
            last_solve=std::chrono::duration<double>(Clock::now()-t0).count();return Timing{device_seconds,std::chrono::duration<double>(Clock::now()-begin).count()};}
        auto t0=Clock::now();if(!o.resident){a.upload(A.data(),A.size());d.upload(D.data(),D.size());u.upload(mu.data(),mu.size());}
        if(active==2){auto factor=batched.cholesky_trials_blocked(exact,{n,c},a,d,u,l,info);device_seconds+=factor.timing.gpu_seconds;last_exact=factor;}
        else{auto encoded=e.batch();batched.cholesky_trials(encoded,{n,c},a,d,u,l,info);device_seconds+=encoded.submit().wait().gpu_seconds;}
        if(!o.resident){l.download(got.data(),got.size());info.download(statuses.data(),statuses.size());}
        last_factor=std::chrono::duration<double>(Clock::now()-t0).count();t0=Clock::now();if(!o.resident)b.upload(rhs.data(),rhs.size());
        if(active==2){auto factors=l.mapped(),result=x.mapped(),right=b.mapped();auto state=info.mapped();
            for(std::size_t t=0;t<c;++t){if(state[t])throw std::runtime_error("batched exact factor failed");device_seconds+=exact.cholesky_solve(B,factors+t*n2,n,right,r,result+t*n*r).gpu_seconds;}}
        else{auto encoded=e.batch();batched.cholesky_solve(encoded,l,info,c,n,b,r,x);device_seconds+=encoded.submit().wait().gpu_seconds;}
        if(!o.resident)x.download(sx.data(),sx.size());last_solve=std::chrono::duration<double>(Clock::now()-t0).count();
        return Timing{device_seconds,std::chrono::duration<double>(Clock::now()-begin).count()};};
    auto verify=[&]{if(active!=1&&o.resident){l.download(got.data(),got.size());x.download(sx.data(),sx.size());info.download(statuses.data(),statuses.size());}
        if(active!=1)for(auto s:statuses)if(s)throw std::runtime_error("GPU trial failed");const auto& factors=active?exact_expected:expected;const auto& solves=active?exact_solution:solution;
        for(std::size_t i=0;i<got.size();++i)if(!reference::equal<B>(got[i],factors[i]))throw std::runtime_error("GPU factor reference mismatch at "+std::to_string(i));
        for(std::size_t i=0;i<sx.size();++i)if(!reference::equal<B>(sx[i],solves[i]))throw std::runtime_error("GPU solve reference mismatch at "+std::to_string(i));};
    for(unsigned p=0;p<3;++p){active=p;gpu();verify();}
    if(o.check_only){std::cout<<B<<" bits: "<<c<<" trials, n="<<n<<", "<<r<<" RHS: sequential and both exact factor/solve paths match independent MPFR; no timings recorded\n";return;}
    for(unsigned rep=0;rep<o.repeats;++rep){for(unsigned j=0;j<2;++j){auto mode=(rep+j)%2;auto start=Clock::now();if(mode)cpu_exact();else cpu();
            cpu_times[mode].push_back(std::chrono::duration<double>(Clock::now()-start).count());verify_cpu(mode);std::cerr<<"cpu_sample,"<<rep<<','<<j<<','<<(mode?"exact_block32":"sequential")<<','<<std::setprecision(12)<<cpu_times[mode].back()<<'\n';}
        for(unsigned j=0;j<3;++j){unsigned p=(rep+j)%3;active=p;if(o.warm){gpu();verify();}auto timing=gpu();wall[p].push_back(timing.wall_seconds);device[p].push_back(timing.gpu_seconds);
            factor_wall[p].push_back(last_factor);solve_wall[p].push_back(last_solve);verify();std::cerr<<"sample,"<<B<<','<<n<<','<<rep<<','<<j<<','<<paths[p]<<','<<std::setprecision(12)<<cpu_times[p!=0].back()<<','<<timing.wall_seconds<<','<<timing.gpu_seconds<<','<<last_factor<<','<<last_solve<<','<<(o.warm?"verified-warm-call":"cpu-interleaved")<<'\n';
            if(p==1)std::cerr<<"factor_profile,"<<rep<<','<<paths[p]<<','<<loop_panel<<','<<loop_update<<','<<loop_gpu<<','<<loop_host<<'\n';
            if(p==2){std::cerr<<"factor_profile,"<<rep<<','<<paths[p]<<','<<last_exact.panel_seconds<<','<<last_exact.update_seconds<<','<<last_exact.gpu_updates<<','<<last_exact.host_updates<<','<<last_exact.submissions<<','<<last_exact.gpu_outputs<<','<<last_exact.fallback_outputs<<','<<last_exact.grouped_updates<<'\n';
                auto bw=batched.workspaces();auto lw=exact.workspaces();std::cerr<<"workspace_profile,"<<rep<<','<<bw.idle_bytes<<','<<bw.busy_bytes<<','<<lw.scratch_bytes<<','<<lw.idle_bytes<<','<<lw.busy_bytes<<'\n';}}}
    std::cerr<<"Device: "<<e.device_name()<<"; split factor/solve; A=J^T J K=n+n/6; mu=0.001*2^-trial; every timed CPU/GPU factor/solve checked; MPFR concurrency <= "<<std::min<std::size_t>(c,o.workers)<<"; exact-batched uses shared resident SYRK and the production blocked solve; numeric-array bridge conversion excluded\n";
    std::cout<<"bits,n,count,nrhs,cpu_workers,repeats,resident,path,cpu_contract,cpu_wall,gpu_wall,gpu_device,factor_wall,solve_wall,wall_p25,wall_p75,wall_min,wall_max,clock,cpu_active_workers,profile\n";
    for(unsigned p=0;p<3;++p)std::cout<<B<<','<<n<<','<<c<<','<<r<<','<<o.workers<<','<<o.repeats<<','<<(p!=1&&o.resident)<<','<<paths[p]<<','<<(p?"exact_block32":"sequential")<<','<<std::setprecision(9)<<quantile(cpu_times[p!=0],.5)<<','<<quantile(wall[p],.5)<<','<<quantile(device[p],.5)<<','<<quantile(factor_wall[p],.5)<<','<<quantile(solve_wall[p],.5)<<','<<quantile(wall[p],.25)<<','<<quantile(wall[p],.75)<<','<<quantile(wall[p],0)<<','<<quantile(wall[p],1)<<','<<(o.warm?"verified-warm-call":"cpu-interleaved")<<','<<std::min<std::size_t>(c,o.workers)<<','<<e.device_name()<<" / exact-trials-candidate / " LIMBFORGE_VERSION_STRING " / "<<(p?"exact-block32":"sequential")<<" / split-factor-solve / schedule="<<(std::getenv("WOLFNUM_EXACT_TRIAL_SCHEDULE")?std::getenv("WOLFNUM_EXACT_TRIAL_SCHEDULE"):"serialized")<<"\n";
}
template<int B=64> void width(const Options& o){if(o.bits==B)run<B>(o);else if constexpr(B<1024)width<B+32>(o);}
int main(int argc,char** argv){try{Options o;for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--warm"){o.warm=true;continue;}if(a=="--resident"){o.resident=true;continue;}if(a=="--check-only"){o.check_only=true;continue;}if(a=="--help"){std::cout<<"section9_exact_trials_limbforge --bits B --n N --count C --nrhs R --workers W --repeats R [--resident] [--check-only] [--warm]\n";return 0;}
        if(++i==argc)throw std::invalid_argument("missing value");std::string s=argv[i];std::size_t used;auto v=std::stoull(s,&used);if(used!=s.size()||s[0]=='-'||!v||v>4096)throw std::invalid_argument("invalid value");
        if(a=="--bits")o.bits=int(v);else if(a=="--n")o.n=v;else if(a=="--count")o.count=v;else if(a=="--nrhs")o.nrhs=v;else if(a=="--workers")o.workers=unsigned(v);else if(a=="--repeats")o.repeats=unsigned(v);else throw std::invalid_argument("unknown option");}
    if(o.bits<64||o.bits>1024||o.bits%32||o.workers>256||o.count>128||o.nrhs>128||o.repeats>1000)throw std::invalid_argument("invalid options");width(o);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
