// Compare distinct complex GEMM contracts. No library arithmetic or defaults change.
#include "limbforge/batched_linalg.hpp"
#include "benchmark_support.hpp"
#include <random>
#include <array>
#include <cmath>

struct Options {
    int bits=352;
    std::size_t count=1,m=66,n=3800,k=130;
    unsigned workers=18,repeats=5;
    int path=-1; // -1: all; 0: composed; 1: fused; 2: exact embedding; 3: Gauss.
    bool resident=false,warm=false,check_only=false,cpu_only=false;
};
std::size_t checked(std::size_t a,std::size_t b) {
    if(b&&a>std::size_t(-1)/b)throw std::invalid_argument("shape overflow");
    return a*b;
}
template<int B> void run(const Options& o) {
    using F=Float<B>;using C=Complex<B/32>;
    const auto ac=checked(checked(o.count,o.m),o.k);
    const auto bc=checked(checked(o.count,o.k),o.n);
    const auto oc=checked(checked(o.count,o.m),o.n);
    std::vector<unsigned> paths;if(o.path<0)paths={0,1,2,3};else paths={unsigned(o.path)};
    Workers pool(o.workers);std::mt19937_64 rng(B+137);
    std::vector<C> A(ac),BB(bc),got(oc);
    MPArray am(2*ac,B),bm(2*bc,B),as(ac,B),bs(bc,B),neg_ai(ac,B),truth(2*oc,2*B+64);
    std::array<std::unique_ptr<MPArray>,4> results;
    std::array<std::vector<C>,4> expected;
    for(unsigned p=0;p<4;++p){results[p]=std::make_unique<MPArray>(2*oc,B);expected[p].resize(oc);}
    auto fill=[&](std::vector<C>& v,MPArray& mp){for(std::size_t j=0;j<v.size();++j){
        v[j]={reference::random_number<B>(rng,4),reference::random_number<B>(rng,4)};
        to_mpfr<B>(mp[2*j],v[j].re);to_mpfr<B>(mp[2*j+1],v[j].im);}};
    fill(A,am);fill(BB,bm);
    for(std::size_t j=0;j<ac;++j)mpfr_neg(neg_ai[j],am[2*j+1],MPFR_RNDN);
    // Reused pointer tables keep pointer gathering/allocation outside timed exact dots.
    std::vector<std::vector<mpfr_ptr>> left(2*o.count*o.m,std::vector<mpfr_ptr>(2*o.k));
    std::vector<std::vector<mpfr_ptr>> right(o.count*o.n,std::vector<mpfr_ptr>(2*o.k));
    for(std::size_t batch=0;batch<o.count;++batch){
        for(std::size_t row=0;row<o.m;++row)for(std::size_t k=0;k<o.k;++k){auto i=(batch*o.m+row)*o.k+k;
            left[(batch*o.m+row)*2][k]=am[2*i];left[(batch*o.m+row)*2][o.k+k]=neg_ai[i];
            left[(batch*o.m+row)*2+1][k]=am[2*i+1];left[(batch*o.m+row)*2+1][o.k+k]=am[2*i];}
        for(std::size_t col=0;col<o.n;++col)for(std::size_t k=0;k<o.k;++k){auto i=(batch*o.k+k)*o.n+col;
            right[batch*o.n+col][k]=bm[2*i];right[batch*o.n+col][o.k+k]=bm[2*i+1];}}
    std::vector<std::unique_ptr<MPArray>> scratch;
    for(unsigned w=0;w<o.workers;++w)scratch.emplace_back(new MPArray(12,2*B+64));
    // Most temporaries round at B. Slots 10/11 retain an independent exact oracle.
    for(auto& s:scratch)for(unsigned j=0;j<10;++j)mpfr_set_prec((*s)[j],B);
    auto cpu=[&](unsigned mode,bool oracle=false){
        if(mode==3)pool.run(o.workers,[&](std::size_t first,std::size_t last){for(auto w=first;w<last;++w){
            for(auto i=ac*w/o.workers;i<ac*(w+1)/o.workers;++i)mpfr_add(as[i],am[2*i],am[2*i+1],MPFR_RNDN);
            for(auto i=bc*w/o.workers;i<bc*(w+1)/o.workers;++i)mpfr_add(bs[i],bm[2*i],bm[2*i+1],MPFR_RNDN);}});
        pool.run(o.workers,[&](std::size_t first,std::size_t last){for(auto w=first;w<last;++w){auto& s=*scratch[w];
            for(auto index=oc*w/o.workers;index<oc*(w+1)/o.workers;++index){
                auto batch=index/(o.m*o.n),row=(index/o.n)%o.m,col=index%o.n;
                auto rr=(*results[mode])[2*index],ri=(*results[mode])[2*index+1];
                if(mode==2&&!oracle){auto& l=left[(batch*o.m+row)*2];auto& li=left[(batch*o.m+row)*2+1];auto& r=right[batch*o.n+col];
                    mpfr_dot(rr,l.data(),r.data(),2*o.k,MPFR_RNDN);mpfr_dot(ri,li.data(),r.data(),2*o.k,MPFR_RNDN);continue;}
                for(unsigned j=0;j<3;++j)mpfr_set_zero(s[j],1);
                mpfr_set_zero(s[10],1);mpfr_set_zero(s[11],1);
                for(std::size_t k=0;k<o.k;++k){auto a=(batch*o.m+row)*o.k+k,b=(batch*o.k+k)*o.n+col;
                    auto ar=am[2*a],ai=am[2*a+1],br=bm[2*b],bi=bm[2*b+1];
                    if(oracle){mpfr_fma(s[10],ar,br,s[10],MPFR_RNDN);
                        mpfr_mul(s[8],ai,bi,MPFR_RNDN);mpfr_sub(s[10],s[10],s[8],MPFR_RNDN);
                        mpfr_fma(s[11],ar,bi,s[11],MPFR_RNDN);mpfr_fma(s[11],ai,br,s[11],MPFR_RNDN);}
                    else if(mode==0){mpfr_mul(s[3],ar,br,MPFR_RNDN);mpfr_mul(s[4],ai,bi,MPFR_RNDN);
                        mpfr_sub(s[3],s[3],s[4],MPFR_RNDN);mpfr_add(s[0],s[0],s[3],MPFR_RNDN);
                        mpfr_mul(s[3],ar,bi,MPFR_RNDN);mpfr_mul(s[4],ai,br,MPFR_RNDN);
                        mpfr_add(s[3],s[3],s[4],MPFR_RNDN);mpfr_add(s[1],s[1],s[3],MPFR_RNDN);}
                    else if(mode==1){mpfr_neg(s[3],ai,MPFR_RNDN);
                        mpfr_ptr x[]={ar,s[3],s[0]},y[]={br,bi,s[5]};mpfr_set_ui(s[5],1,MPFR_RNDN);
                        mpfr_dot(s[0],x,y,3,MPFR_RNDN);
                        mpfr_ptr xi[]={ar,ai,s[1]},yi[]={bi,br,s[5]};mpfr_dot(s[1],xi,yi,3,MPFR_RNDN);}
                    else{mpfr_mul(s[3],ar,br,MPFR_RNDN);mpfr_add(s[0],s[0],s[3],MPFR_RNDN);
                        mpfr_mul(s[3],ai,bi,MPFR_RNDN);mpfr_add(s[1],s[1],s[3],MPFR_RNDN);
                        mpfr_mul(s[3],as[a],bs[b],MPFR_RNDN);mpfr_add(s[2],s[2],s[3],MPFR_RNDN);}}
                if(oracle){mpfr_set(truth[2*index],s[10],MPFR_RNDN);mpfr_set(truth[2*index+1],s[11],MPFR_RNDN);
                    mpfr_set(rr,s[10],MPFR_RNDN);mpfr_set(ri,s[11],MPFR_RNDN);}
                else if(mode==3){mpfr_sub(rr,s[0],s[1],MPFR_RNDN);mpfr_sub(ri,s[2],s[0],MPFR_RNDN);mpfr_sub(ri,ri,s[1],MPFR_RNDN);}
                else{mpfr_set(rr,s[0],MPFR_RNDN);mpfr_set(ri,s[1],MPFR_RNDN);}
            }}});
    };
    for(auto& s:scratch)mpfr_set_prec((*s)[8],2*B+64);
    for(unsigned p=0;p<4;++p){cpu(p,p==2);for(std::size_t i=0;i<oc;++i)expected[p][i]={from_mpfr<B>((*results[p])[2*i]),from_mpfr<B>((*results[p])[2*i+1])};}
    auto check_cpu=[&](unsigned p){for(std::size_t i=0;i<oc;++i){C value{from_mpfr<B>((*results[p])[2*i]),from_mpfr<B>((*results[p])[2*i+1])};
        if(!reference::equal_complex<B>(value,expected[p][i]))throw std::runtime_error("CPU reference mismatch: path "+std::to_string(p)+" index "+std::to_string(i));}};
    cpu(2);check_cpu(2);
    if(o.cpu_only){std::cout<<B<<" bits: four CPU contracts prepared; MPFR exact dots match independent high-precision sums; no GPU or timings\n";return;}
    std::size_t differs[4]={};double error[4]={};reference::MP scale(2*B+64),diff(2*B+64),maximum(2*B+64),value(2*B+64);
    mpfr_set_zero(scale.x,1);for(std::size_t i=0;i<2*oc;++i){mpfr_abs(value.x,truth[i],MPFR_RNDN);if(mpfr_cmp(value.x,scale.x)>0)mpfr_set(scale.x,value.x,MPFR_RNDN);}
    for(unsigned p=0;p<4;++p){mpfr_set_zero(maximum.x,1);for(std::size_t i=0;i<oc;++i){if(!reference::equal_complex<B>(expected[p][i],expected[2][i]))++differs[p];
        for(unsigned c=0;c<2;++c){to_mpfr<B>(value.x,c?expected[p][i].im:expected[p][i].re);mpfr_sub(diff.x,value.x,truth[2*i+c],MPFR_RNDN);mpfr_abs(diff.x,diff.x,MPFR_RNDN);if(mpfr_cmp(diff.x,maximum.x)>0)mpfr_set(maximum.x,diff.x,MPFR_RNDN);}}
        if(!mpfr_zero_p(scale.x)){mpfr_div(maximum.x,maximum.x,scale.x,MPFR_RNDN);mpfr_mul_2ui(maximum.x,maximum.x,B,MPFR_RNDN);}error[p]=mpfr_get_d(maximum.x,MPFR_RNDN);}
    Engine engine;BatchedLinalg seq(engine);Linalg exact(engine);
    bool use_complex=o.path<2,use_exact=o.path<0||o.path==2,use_gauss=o.path<0||o.path==3;
    std::vector<F> ar(use_gauss?ac:0),ai(ar.size()),br(use_gauss?bc:0),bi(br.size());
    std::vector<F> embedded_a(use_exact?4*ac:0),embedded_b(use_exact?2*bc:0),embedded_c(use_exact?2*oc:0),real(use_gauss?oc:0),imag(real.size());
    auto pack=[&](unsigned p){if(p==2){for(std::size_t batch=0;batch<o.count;++batch){
        for(std::size_t row=0;row<o.m;++row)for(std::size_t k=0;k<o.k;++k){auto a=(batch*o.m+row)*o.k+k,base=(batch*2*o.m+2*row)*2*o.k;
            embedded_a[base+k]=A[a].re;embedded_a[base+o.k+k]=A[a].im;embedded_a[base+o.k+k].sign=-embedded_a[base+o.k+k].sign;
            embedded_a[base+2*o.k+k]=A[a].im;embedded_a[base+3*o.k+k]=A[a].re;}
        for(std::size_t k=0;k<o.k;++k)for(std::size_t col=0;col<o.n;++col){auto b=(batch*o.k+k)*o.n+col,base=batch*2*o.k*o.n;
            embedded_b[base+k*o.n+col]=BB[b].re;embedded_b[base+(o.k+k)*o.n+col]=BB[b].im;}}}
        else if(p==3){for(std::size_t i=0;i<ac;++i){ar[i]=A[i].re;ai[i]=A[i].im;}for(std::size_t i=0;i<bc;++i){br[i]=BB[i].re;bi[i]=BB[i].im;}}};
    if(use_exact)pack(2);if(use_gauss)pack(3);
    Buffer<C> ca,cb,cc;if(use_complex){ca=engine.make_buffer<C>(ac);cb=engine.make_buffer<C>(bc);cc=engine.make_buffer<C>(oc);}
    Buffer<F> rar,rai,rbr,rbi,ras,rbs,rp,rq,rs,rr,ri;
    if(use_gauss){rar=engine.make_buffer<F>(ac);rai=engine.make_buffer<F>(ac);rbr=engine.make_buffer<F>(bc);rbi=engine.make_buffer<F>(bc);
        ras=engine.make_buffer<F>(ac);rbs=engine.make_buffer<F>(bc);rp=engine.make_buffer<F>(oc);rq=engine.make_buffer<F>(oc);rs=engine.make_buffer<F>(oc);rr=engine.make_buffer<F>(oc);ri=engine.make_buffer<F>(oc);}
    std::vector<Buffer<F>> ea,eb,ec;if(use_exact)for(std::size_t batch=0;batch<o.count;++batch){ea.push_back(engine.make_buffer<F>(4*o.m*o.k));eb.push_back(engine.make_buffer<F>(2*o.k*o.n));ec.push_back(engine.make_buffer<F>(2*o.m*o.n));}
    auto upload=[&](unsigned p){if(p<2){ca.upload(A.data(),ac);cb.upload(BB.data(),bc);}
        else if(p==2){for(std::size_t b=0;b<o.count;++b){ea[b].upload(embedded_a.data()+b*4*o.m*o.k,4*o.m*o.k);eb[b].upload(embedded_b.data()+b*2*o.k*o.n,2*o.k*o.n);}}
        else{rar.upload(ar.data(),ac);rai.upload(ai.data(),ac);rbr.upload(br.data(),bc);rbi.upload(bi.data(),bc);}};
    if(use_complex)upload(0);if(use_exact)upload(2);if(use_gauss)upload(3);
    auto download=[&](unsigned p){if(p<2)cc.download(got.data(),oc);
        else if(p==2){for(std::size_t b=0;b<o.count;++b)ec[b].download(embedded_c.data()+b*2*o.m*o.n,2*o.m*o.n);
            for(std::size_t b=0;b<o.count;++b)for(std::size_t row=0;row<o.m;++row)for(std::size_t col=0;col<o.n;++col){auto i=(b*o.m+row)*o.n+col,base=(b*2*o.m+row*2)*o.n+col;got[i]={embedded_c[base],embedded_c[base+o.n]};}}
        else{rr.download(real.data(),oc);ri.download(imag.data(),oc);for(std::size_t i=0;i<oc;++i)got[i]={real[i],imag[i]};}};
    auto gpu=[&](unsigned p){auto start=Clock::now();if(!o.resident){pack(p);upload(p);}auto batch=engine.batch();
        StridedGemm shape{o.count,o.m,o.n,o.k,o.m*o.k,o.k*o.n,o.m*o.n};
        if(p<2){shape.fused=p==1;seq.gemm(batch,shape,ca,cb,cc);}
        else if(p==2){for(std::size_t b=0;b<o.count;++b)exact.gemm(batch,false,ea[b],eb[b],2*o.m,o.n,2*o.k,ec[b]);}
        else{batch.run(Operation::add,rar,rai,ras);batch.run(Operation::add,rbr,rbi,rbs);
            seq.gemm(batch,shape,rar,rbr,rp);seq.gemm(batch,shape,rai,rbi,rq);seq.gemm(batch,shape,ras,rbs,rs);
            batch.run(Operation::sub,rp,rq,rr);batch.run(Operation::sub,rs,rp,ri);batch.run(Operation::sub,ri,rq,ri);}
        auto timing=batch.submit().wait();if(!o.resident)download(p);timing.wall_seconds=std::chrono::duration<double>(Clock::now()-start).count();return timing;};
    auto verify=[&](unsigned p){if(o.resident)download(p);for(std::size_t i=0;i<oc;++i)if(!reference::equal_complex<B>(got[i],expected[p][i]))
        throw std::runtime_error("GPU mismatch: path "+std::to_string(p)+" index "+std::to_string(i));};
    const char* names[]={"complex_composed","complex_fused","exact_real_embedding","gauss_three_real_composed"};
    for(auto p:paths){gpu(p);verify(p);}
    if(o.check_only){std::cout<<B<<" bits, "<<o.count<<"x("<<o.m<<"x"<<o.k<<"x"<<o.n<<"): "<<paths.size()<<" selected GPU contracts match MPFR; no timings\n";return;}
    std::vector<double> cpu_times[4],wall[4],device[4];
    for(unsigned rep=0;rep<o.repeats;++rep){for(unsigned j=0;j<paths.size();++j){auto p=paths[(rep+j)%paths.size()];auto start=Clock::now();cpu(p);auto seconds=std::chrono::duration<double>(Clock::now()-start).count();cpu_times[p].push_back(seconds);check_cpu(p);
            std::cerr<<"cpu_sample,"<<rep<<','<<j<<','<<names[p]<<','<<std::setprecision(12)<<seconds<<'\n';}
        for(unsigned j=0;j<paths.size();++j){auto p=paths[(rep+j)%paths.size()];if(o.warm){gpu(p);verify(p);}auto t=gpu(p);wall[p].push_back(t.wall_seconds);device[p].push_back(t.gpu_seconds);verify(p);
            std::cerr<<"gpu_sample,"<<rep<<','<<j<<','<<names[p]<<','<<std::setprecision(12)<<t.wall_seconds<<','<<t.gpu_seconds<<'\n';}}
    std::cout<<"bits,count,m,n,k,path,resident,workers,repeats,clock,cpu_wall,gpu_wall,gpu_device,wall_p25,wall_p75,wall_min,wall_max,different_complex_entries,error_units_2neg_bits,profile\n";
    for(auto p:paths)std::cout<<B<<','<<o.count<<','<<o.m<<','<<o.n<<','<<o.k<<','<<names[p]<<','<<o.resident<<','<<o.workers<<','<<o.repeats<<','<<(o.warm?"verified_warm_call":"cpu_interleaved")<<','<<std::setprecision(12)<<quantile(cpu_times[p],.5)<<','<<quantile(wall[p],.5)<<','<<quantile(device[p],.5)<<','<<quantile(wall[p],.25)<<','<<quantile(wall[p],.75)<<','<<quantile(wall[p],0)<<','<<quantile(wall[p],1)<<','<<differs[p]<<','<<error[p]<<','<<engine.device_name()<<" / " LIMBFORGE_VERSION_STRING " / complex-gemm-contracts\n";
}
template<int B=64> void width(const Options& o){if(o.bits==B)run<B>(o);else if constexpr(B<1024)width<B+32>(o);}
int main(int argc,char** argv){try{Options o;for(int i=1;i<argc;++i){std::string a=argv[i];
    if(a=="--resident"){o.resident=true;continue;}if(a=="--warm"){o.warm=true;continue;}if(a=="--check-only"){o.check_only=true;continue;}if(a=="--cpu-only"){o.cpu_only=true;continue;}
    if(a=="--help"){std::cout<<"section9_gemm_limbforge --bits B --count C --m M --n N --k K --workers W --repeats R [--path composed|fused|exact|gauss] [--resident] [--warm] [--check-only] [--cpu-only]\n";return 0;}
    if(++i==argc)throw std::invalid_argument("missing value");std::string s=argv[i];
    if(a=="--path"){if(s=="composed")o.path=0;else if(s=="fused")o.path=1;else if(s=="exact")o.path=2;else if(s=="gauss")o.path=3;else throw std::invalid_argument("unknown path");continue;}
    std::size_t used;auto v=std::stoull(s,&used);
    if(used!=s.size()||s[0]=='-'||!v||v>100000)throw std::invalid_argument("invalid positive integer");
    if(a=="--bits")o.bits=int(v);else if(a=="--count")o.count=v;else if(a=="--m")o.m=v;else if(a=="--n")o.n=v;else if(a=="--k")o.k=v;else if(a=="--workers")o.workers=unsigned(v);else if(a=="--repeats")o.repeats=unsigned(v);else throw std::invalid_argument("unknown option");}
    if(o.bits<64||o.bits>1024||o.bits%32||o.workers>256||o.repeats>1000||o.k>4096||checked(checked(o.count,o.m),o.n)>25000000)throw std::invalid_argument("invalid options");
    checked(checked(o.count,o.m),o.k);checked(checked(o.count,o.k),o.n);width(o);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
