// Exact, composed and fused normal equations, checked against independent MPFR sequences.
#include "limbforge/batched_linalg.hpp"
#include "benchmark_support.hpp"
#include <random>
#include <cmath>
struct Options{int bits=352;std::size_t rows=1100,cols=944;unsigned workers=18,repeats=5;bool resident=false,warm=false,check_only=false;};
template<int B> void run(const Options& o){
    using F=Float<B>;auto k=o.rows,n=o.cols,n2=n*n,total=n2+n,tri=n*(n+1)/2;
    Engine engine;BatchedLinalg batch_unit(engine);Linalg exact(engine);Workers pool(o.workers);std::mt19937_64 rng(B+91);
    std::vector<F> J(k*n),g(k),expected[3],got(total);for(auto* v:{&J,&g})for(auto& x:*v)x=reference::random_number<B>(rng,4);
    MPArray jm(J.size(),B),gm(g.size(),B),truth(total,2*B+64);
    for(std::size_t i=0;i<J.size();++i)to_mpfr<B>(jm[i],J[i]);for(std::size_t i=0;i<g.size();++i)to_mpfr<B>(gm[i],g[i]);
    for(auto& v:expected)v.resize(total);
    std::vector<std::vector<mpfr_ptr>> columns(n,std::vector<mpfr_ptr>(k));std::vector<mpfr_ptr> residual(k);
    for(std::size_t row=0;row<k;++row){residual[row]=gm[row];for(std::size_t col=0;col<n;++col)columns[col][row]=jm[row*n+col];}
    struct Scratch{reference::MP sum,product,full;Scratch():sum(B),product(B),full(2*B+64){}};
    std::vector<std::unique_ptr<Scratch>> scratch;for(unsigned i=0;i<o.workers;++i)scratch.emplace_back(new Scratch);
    // Exponents are bounded by the fixture's span=4; 2B+64 covers every aligned
    // product and ceil(log2(rows)) carry bits (rows<=4096). This is an exact sum.
    auto cpu=[&](unsigned mode,bool high_truth=false){pool.run(o.workers,[&](std::size_t first,std::size_t last){for(auto worker=first;worker<last;++worker){auto& t=*scratch[worker];
        for(auto index=(tri+n)*worker/o.workers;index<(tri+n)*(worker+1)/o.workers;++index){
            std::size_t i,j;bool rhs=index>=tri;
            if(rhs){i=index-tri;j=n;}else{i=std::size_t((std::sqrt(double(8*index+1))-1)/2);j=index-i*(i+1)/2;}
            mpfr_set_zero(t.sum.x,1);mpfr_set_zero(t.full.x,1);
            if(mode==2&&!high_truth)mpfr_dot(t.sum.x,columns[i].data(),rhs?residual.data():columns[j].data(),k,MPFR_RNDN);
            else for(std::size_t row=0;row<k;++row){auto a=jm[row*n+i],b=rhs?gm[row]:jm[row*n+j];
                if(mode==2)mpfr_fma(t.full.x,a,b,t.full.x,MPFR_RNDN);
                else if(mode==1)mpfr_fma(t.sum.x,a,b,t.sum.x,MPFR_RNDN);
                else{mpfr_mul(t.product.x,a,b,MPFR_RNDN);mpfr_add(t.sum.x,t.sum.x,t.product.x,MPFR_RNDN);}}
            auto offset=rhs?n2+i:i*n+j;auto value=from_mpfr<B>(mode==2&&high_truth?t.full.x:t.sum.x);expected[mode][offset]=value;
            if(mode==2&&high_truth)mpfr_set(truth[offset],t.full.x,MPFR_RNDN);
            if(!rhs&&i!=j){expected[mode][j*n+i]=value;if(mode==2&&high_truth)mpfr_set(truth[j*n+i],t.full.x,MPFR_RNDN);}
        }} });};
    for(unsigned mode=0;mode<3;++mode)cpu(mode,mode==2);
    const auto exact_oracle=expected[2];cpu(2);for(std::size_t i=0;i<total;++i)if(!reference::equal<B>(expected[2][i],exact_oracle[i]))throw std::runtime_error("MPFR dot differs from high-precision exact oracle");
    std::size_t differs[3]={};double error_units[3]={};reference::MP scale(2*B+64),diff(2*B+64),maximum(2*B+64),value(2*B+64);
    mpfr_set_zero(scale.x,1);for(std::size_t i=0;i<total;++i){mpfr_abs(value.x,truth[i],MPFR_RNDN);if(mpfr_cmp(value.x,scale.x)>0)mpfr_set(scale.x,value.x,MPFR_RNDN);}
    for(unsigned mode=0;mode<3;++mode){mpfr_set_zero(maximum.x,1);for(std::size_t i=0;i<total;++i){if(!reference::equal<B>(expected[mode][i],expected[2][i]))++differs[mode];
        to_mpfr<B>(value.x,expected[mode][i]);mpfr_sub(diff.x,value.x,truth[i],MPFR_RNDN);mpfr_abs(diff.x,diff.x,MPFR_RNDN);if(mpfr_cmp(diff.x,maximum.x)>0)mpfr_set(maximum.x,diff.x,MPFR_RNDN);}
        if(!mpfr_zero_p(scale.x)){mpfr_div(maximum.x,maximum.x,scale.x,MPFR_RNDN);mpfr_mul_2ui(maximum.x,maximum.x,B,MPFR_RNDN);}error_units[mode]=mpfr_get_d(maximum.x,MPFR_RNDN);}
    auto put=[&](const std::vector<F>& v){auto b=engine.make_buffer<F>(v.size());b.upload(v.data(),v.size());return b;};
    auto jb=put(J),gb=put(g),ab=engine.make_buffer<F>(n2),rb=engine.make_buffer<F>(n);unsigned active=0;
    auto download=[&]{ab.download(got.data(),n2);rb.download(got.data()+n2,n);};
    auto gpu=[&]{auto start=Clock::now();if(!o.resident){jb.upload(J.data(),J.size());gb.upload(g.data(),g.size());}
        auto batch=engine.batch();
        if(active<2)batch_unit.normal_equations(batch,jb,gb,k,n,ab,rb,active==1);
        else if(active==2)batch_unit.normal_equations_exact(exact,batch,jb,gb,k,n,ab,rb);
        else{exact.syrk(batch,jb,k,n,ab,false);exact.gemm(batch,true,jb,gb,n,1,k,rb);}
        auto timing=batch.submit().wait();if(!o.resident)download();timing.wall_seconds=std::chrono::duration<double>(Clock::now()-start).count();return timing;};
    auto verify=[&]{if(o.resident)download();auto mode=std::min(active,2u);for(std::size_t i=0;i<total;++i)if(!reference::equal<B>(got[i],expected[mode][i]))throw std::runtime_error("normal path "+std::to_string(active)+" mismatch at "+std::to_string(i));};
    const char* names[]={"sequential_composed","sequential_fused","exact_augmented","exact_syrk_plus_gemm"};
    for(active=0;active<4;++active){gpu();verify();}
    if(o.check_only){std::cout<<B<<" bits, "<<k<<"x"<<n<<": all four normal-equation paths match MPFR; no timings recorded\n";return;}
    std::vector<double> wall[4],device[4],cpu_times[3];
    for(unsigned rep=0;rep<o.repeats;++rep){for(unsigned mode=0;mode<3;++mode){auto start=Clock::now();cpu(mode);cpu_times[mode].push_back(std::chrono::duration<double>(Clock::now()-start).count());
            if(mode==2)for(std::size_t i=0;i<total;++i)if(!reference::equal<B>(expected[2][i],exact_oracle[i]))throw std::runtime_error("timed MPFR dot mismatch");}
        for(unsigned j=0;j<4;++j){active=(rep+j)%4;if(o.warm){gpu();verify();}auto timing=gpu();wall[active].push_back(timing.wall_seconds);device[active].push_back(timing.gpu_seconds);verify();
            std::cerr<<"sample,"<<B<<','<<k<<','<<n<<','<<rep<<','<<j<<','<<names[active]<<','<<std::setprecision(12)<<timing.wall_seconds<<','<<timing.gpu_seconds<<'\n';}}
    std::cout<<"bits,rows,cols,path,resident,workers,repeats,clock,cpu_reference_wall,gpu_wall,gpu_device,wall_p25,wall_p75,wall_min,wall_max,different_from_exact,error_units_2neg_bits,profile\n";
    for(unsigned p=0;p<4;++p){auto mode=std::min(p,2u);std::cout<<B<<','<<k<<','<<n<<','<<names[p]<<','<<o.resident<<','<<o.workers<<','<<o.repeats<<','<<(o.warm?"warm":"cpu_interleaved")<<','<<std::setprecision(12)<<quantile(cpu_times[mode],.5)<<','<<quantile(wall[p],.5)<<','<<quantile(device[p],.5)<<','<<quantile(wall[p],.25)<<','<<quantile(wall[p],.75)<<','<<quantile(wall[p],0)<<','<<quantile(wall[p],1)<<','<<differs[mode]<<','<<error_units[mode]<<','<<engine.device_name()<<" / " LIMBFORGE_VERSION_STRING " / normal-equations\n";}
}
template<int B=64> void width(const Options& o){if(o.bits==B)run<B>(o);else if constexpr(B<1024)width<B+32>(o);}
int main(int argc,char** argv){try{Options o;for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--resident"){o.resident=true;continue;}if(a=="--warm"){o.warm=true;continue;}if(a=="--check-only"){o.check_only=true;continue;}if(a=="--help"){std::cout<<"section9_normal_limbforge --bits B --rows K --cols N --workers W --repeats R [--resident] [--warm] [--check-only]\n";return 0;}if(++i==argc)throw std::invalid_argument("missing value");std::string s=argv[i];std::size_t used;auto v=std::stoull(s,&used);if(used!=s.size()||s[0]=='-'||!v||v>4096)throw std::invalid_argument("invalid value");if(a=="--bits")o.bits=int(v);else if(a=="--rows")o.rows=v;else if(a=="--cols")o.cols=v;else if(a=="--workers")o.workers=unsigned(v);else if(a=="--repeats")o.repeats=unsigned(v);else throw std::invalid_argument("unknown option");}if(o.bits<64||o.bits>1024||o.bits%32||o.workers>256||o.repeats>1000)throw std::invalid_argument("invalid options");width(o);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
