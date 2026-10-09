// Shared-source recurrence calibration. CPU Horner work is shared too.
#include "limbforge/batched_linalg.hpp"
#include "benchmark_support.hpp"
#include <array>
#include <random>
#include <limits>
#include <cmath>

struct Options {
    int bits=352;std::size_t lanes=513;unsigned steps=3,terms=3,share=2,workers=18,repeats=5;
    bool resident=false,fused=false,reverse=false,all=false,per_group=false,check_only=false,warm=false,cpu_check=false;
};
std::size_t extent(std::size_t a,std::size_t b){if(b&&a>std::size_t(-1)/b)throw std::invalid_argument("shape overflow");return a*b;}
// Four separately rounded products, then RN subtraction/addition.
void product(mpfr_ptr r,mpfr_ptr i,mpfr_srcptr ar,mpfr_srcptr ai,mpfr_srcptr br,mpfr_srcptr bi,MPArray& t){
    mpfr_mul(t[4],ar,br,MPFR_RNDN);mpfr_mul(t[5],ai,bi,MPFR_RNDN);
    mpfr_mul(t[6],ar,bi,MPFR_RNDN);mpfr_mul(t[7],ai,br,MPFR_RNDN);
    mpfr_sub(r,t[4],t[5],MPFR_RNDN);mpfr_add(i,t[6],t[7],MPFR_RNDN);
}
// mpfr_dot rounds the exact two products plus addend once, including cancellation.
// Temporary destinations permit aliasing of a caller's output with any input.
void mac(mpfr_ptr r,mpfr_ptr i,mpfr_srcptr ar,mpfr_srcptr ai,mpfr_srcptr br,mpfr_srcptr bi,mpfr_srcptr cr,mpfr_srcptr ci,MPArray& t,bool fused){
    if(fused){mpfr_neg(t[2],ai,MPFR_RNDN);mpfr_set_ui(t[3],1,MPFR_RNDN);
        mpfr_ptr xr[3]={const_cast<mpfr_ptr>(ar),t[2],const_cast<mpfr_ptr>(cr)},yr[3]={const_cast<mpfr_ptr>(br),const_cast<mpfr_ptr>(bi),t[3]};
        mpfr_ptr xi[3]={const_cast<mpfr_ptr>(ar),const_cast<mpfr_ptr>(ai),const_cast<mpfr_ptr>(ci)},yi[3]={const_cast<mpfr_ptr>(bi),const_cast<mpfr_ptr>(br),t[3]};
        mpfr_dot(t[0],xr,yr,3,MPFR_RNDN);mpfr_dot(t[1],xi,yi,3,MPFR_RNDN);
    }else{product(t[0],t[1],ar,ai,br,bi,t);mpfr_add(t[0],t[0],cr,MPFR_RNDN);mpfr_add(t[1],t[1],ci,MPFR_RNDN);}
    mpfr_set(r,t[0],MPFR_RNDN);mpfr_set(i,t[1],MPFR_RNDN);
}
template<int B> void run(const Options& o){
    using C=Complex<B/32>;
    // Audit this benchmark's mpfr_dot helper against the separate exact-product/mpfr_sum
    // reference, with output/input aliasing, zeros and widely separated exponents.
    if(o.cpu_check){MPArray t(20,B);std::mt19937_64 random(B+103);
        for(int trial=0;trial<64;++trial){auto make=[&]{return C{reference::random_number<B>(random,trial%2?1000:2),reference::random_number<B>(random,trial%2?1000:2)};};auto a=make(),b=make(),c=make();
            if(trial%8==0)a={zero<B/32>(),zero<B/32>()};
            if(trial%8==1){a.im=a.re;b.im=b.re;}
            if(trial%8==2){c=reference::complex<B>(Operation::complex_mul,a,b);c={negate(c.re),negate(c.im)};}
            for(bool fused:{false,true}){to_mpfr<B>(t[8],a.re);to_mpfr<B>(t[9],a.im);to_mpfr<B>(t[10],b.re);to_mpfr<B>(t[11],b.im);to_mpfr<B>(t[12],c.re);to_mpfr<B>(t[13],c.im);
                auto expected=fused?reference::complex_fused<B>(a,b,c):reference::complex<B>(Operation::complex_add,reference::complex<B>(Operation::complex_mul,a,b),c);
                mac(t[8],t[9],t[8],t[9],t[10],t[11],t[12],t[13],t,fused);C actual{from_mpfr<B>(t[8]),from_mpfr<B>(t[9])};
                if(!reference::equal_complex<B>(actual,expected))throw std::runtime_error("CPU helper reference mismatch");
            }
        }std::cout<<"PASS "<<B<<"-bit CPU helper reference, no GPU or timing\n";return;
    }
    const auto groups=o.lanes/o.share+(o.lanes%o.share!=0),sets=o.per_group?groups:1;
    const auto coeff=extent(extent(sets,4),o.terms),ys=extent(groups,o.steps),sources=extent(ys,4),vn=extent(o.lanes,4),on=extent(vn,o.all?std::size_t(o.steps)+1:1);
    if(ys>UINT32_MAX)throw std::invalid_argument("source grid exceeds 32-bit dispatch limit");
    MPArray st(extent(vn,2),B),cp(extent(coeff,2),B),cq(extent(coeff,2),B),y(extent(ys,2),B),ep(extent(sources,2),B),eq(extent(sources,2),B),p(extent(sources,2),B),q(extent(sources,2),B),result(extent(on,2),B);
    std::vector<C> start(vn),c_p(coeff),c_q(coeff),Y(ys),Ep(sources),Eq(sources),output(on),want(on);
    std::mt19937_64 rng(B+91);auto fill=[&](std::vector<C>& v,MPArray& m){for(std::size_t i=0;i<v.size();++i){v[i]={reference::random_number<B>(rng,2),reference::random_number<B>(rng,2)};to_mpfr<B>(m[2*i],v[i].re);to_mpfr<B>(m[2*i+1],v[i].im);}};
    fill(start,st);fill(c_p,cp);fill(c_q,cq);fill(Y,y);fill(Ep,ep);fill(Eq,eq);
    Workers pool(o.workers);std::vector<std::unique_ptr<MPArray>> scratch;
    for(unsigned w=0;w<o.workers;++w)scratch.emplace_back(std::make_unique<MPArray>(20,B));
    auto cpu=[&]{
        // Independent Horner sources once per group, not once per trajectory.
        pool.run(o.workers,[&](std::size_t first,std::size_t last){for(auto w=first;w<last;++w){auto& t=*scratch[w];
            for(auto x=sources*w/o.workers;x<sources*(w+1)/o.workers;++x){auto k=x/(4*groups),a=(x/groups)%4,g=x%groups,set=o.per_group?g:0,c=(set*4+a)*o.terms,yy=k*groups+g;
                for(auto pair:{std::array<MPArray*,3>{&cp,&ep,&p},std::array<MPArray*,3>{&cq,&eq,&q}}){
                    if(o.terms){mpfr_set(t[8],(*pair[0])[2*(c+o.terms-1)],MPFR_RNDN);mpfr_set(t[9],(*pair[0])[2*(c+o.terms-1)+1],MPFR_RNDN);
                        for(unsigned n=o.terms-1;n-->0;)mac(t[8],t[9],t[8],t[9],y[2*yy],y[2*yy+1],(*pair[0])[2*(c+n)],(*pair[0])[2*(c+n)+1],t,o.fused);
                    }else{mpfr_set_zero(t[8],1);mpfr_set_zero(t[9],1);}
                    product((*pair[2])[2*x],(*pair[2])[2*x+1],(*pair[1])[2*x],(*pair[1])[2*x+1],t[8],t[9],t);
                }
            }
        }});
        pool.run(o.workers,[&](std::size_t first,std::size_t last){for(auto w=first;w<last;++w){auto& t=*scratch[w];
            for(auto lane=o.lanes*w/o.workers;lane<o.lanes*(w+1)/o.workers;++lane){auto g=lane/o.share;
                auto save=[&](std::size_t step){for(unsigned a=0;a<4;++a){auto x=step*vn+a*o.lanes+lane;mpfr_set(result[2*x],t[8+2*a],MPFR_RNDN);mpfr_set(result[2*x+1],t[9+2*a],MPFR_RNDN);}};
                for(unsigned a=0;a<4;++a){mpfr_set(t[8+2*a],st[2*(a*o.lanes+lane)],MPFR_RNDN);mpfr_set(t[9+2*a],st[2*(a*o.lanes+lane)+1],MPFR_RNDN);}if(o.all)save(0);
                for(unsigned step=0;step<o.steps;++step){auto k=o.reverse?o.steps-1-step:step;mpfr_set_zero(t[16],1);mpfr_set_zero(t[17],1);
                    for(unsigned a=0;a<4;++a){auto x=(std::size_t(k)*4+a)*groups+g;mac(t[16],t[17],q[2*x],q[2*x+1],t[8+2*a],t[9+2*a],t[16],t[17],t,o.fused);}
                    for(unsigned a=0;a<4;++a){auto x=(std::size_t(k)*4+a)*groups+g;mac(t[8+2*a],t[9+2*a],p[2*x],p[2*x+1],t[16],t[17],t[8+2*a],t[9+2*a],t,o.fused);}if(o.all)save(step+1);
                }if(!o.all)save(0);
            }
        }});
    };
    cpu();for(std::size_t x=0;x<on;++x)want[x]={from_mpfr<B>(result[2*x]),from_mpfr<B>(result[2*x+1])};
    Engine engine;BatchedLinalg la(engine);auto put=[&](const std::vector<C>& v){auto b=engine.make_buffer<C>(v.size());b.upload(v.data(),v.size());return b;};
    auto sb=put(start),pb=put(c_p),qb=put(c_q),yb=put(Y),eb=put(Ep),fb=put(Eq),out=engine.make_buffer<C>(on);
    PolynomialRecurrence shape{o.lanes,sets,o.steps,o.terms,o.share,o.all,o.reverse,o.fused};
    auto gpu=[&](unsigned path){auto begin=Clock::now();if(!o.resident){sb.upload(start.data(),vn);pb.upload(c_p.data(),coeff);qb.upload(c_q.data(),coeff);yb.upload(Y.data(),ys);eb.upload(Ep.data(),sources);fb.upload(Eq.data(),sources);}
        auto batch=engine.batch();la.polynomial_recurrence(batch,shape,sb,pb,qb,yb,eb,fb,out,path?PolynomialEvaluation::shared_sources:PolynomialEvaluation::per_lane);auto time=batch.submit().wait();
        if(!o.resident)out.download(output.data(),on);time.wall_seconds=std::chrono::duration<double>(Clock::now()-begin).count();return time;};
    auto verify=[&]{if(o.resident)out.download(output.data(),on);for(std::size_t x=0;x<on;++x)if(!reference::equal_complex<B>(output[x],want[x]))throw std::runtime_error("recurrence mismatch at "+std::to_string(x));};
    for(unsigned path=0;path<2;++path){gpu(path);verify();} // compile and pages outside timing
    if(o.check_only){std::cout<<"PASS "<<B<<" bits; both paths, "<<on<<" independently checked outputs; no timings\n";return;}
    std::array<std::vector<double>,2> wall,device;std::vector<double> cpu_time;
    std::cerr<<"Device: "<<engine.device_name()<<"; optimized shared-source MPFR CPU; "<<o.workers<<" workers requested; active source/lane workers "<<std::min<std::size_t>(o.workers,sources)<<'/'<<std::min<std::size_t>(o.workers,o.lanes)<<"; every GPU sample checked\n";
    for(unsigned r=0;r<o.repeats;++r){auto begin=Clock::now();cpu();auto c=std::chrono::duration<double>(Clock::now()-begin).count();cpu_time.push_back(c);
        for(std::size_t x=0;x<on;++x)if(!reference::equal_complex<B>(C{from_mpfr<B>(result[2*x]),from_mpfr<B>(result[2*x+1])},want[x]))throw std::runtime_error("CPU replay changed");
        for(unsigned j=0;j<2;++j){auto path=(r+j)%2;if(o.warm){gpu(path);verify();}auto t=gpu(path);verify();wall[path].push_back(t.wall_seconds);device[path].push_back(t.gpu_seconds);
            std::cerr<<"sample="<<r<<" order="<<j<<" path="<<path<<" cpu="<<std::setprecision(17)<<c<<" wall="<<t.wall_seconds<<" device="<<t.gpu_seconds<<'\n';}
    }
    std::cout<<"bits,lanes,groups,steps,terms,share,coefficient_sets,fused,reverse,all_steps,resident,clock_mode,workers,repeats,path,cpu_median,wall_median,wall_p25,wall_p75,wall_min,wall_max,device_median,scratch_bytes\n";
    for(unsigned path=0;path<2;++path){auto tables=extent(sources,4);if(tables>std::size_t(-1)-o.lanes)throw std::invalid_argument("scratch size overflow");auto bytes=path?extent(tables+o.lanes,sizeof(C)):0;
        std::cout<<B<<','<<o.lanes<<','<<groups<<','<<o.steps<<','<<o.terms<<','<<o.share<<','<<sets<<','<<o.fused<<','<<o.reverse<<','<<o.all<<','<<o.resident<<','<<(o.warm?"verified-warm-call":"cpu-interleaved")<<','<<o.workers<<','<<o.repeats<<','<<(path?"shared_sources":"per_lane")<<','<<std::setprecision(17)<<quantile(cpu_time,.5)<<','<<quantile(wall[path],.5)<<','<<quantile(wall[path],.25)<<','<<quantile(wall[path],.75)<<','<<quantile(wall[path],0)<<','<<quantile(wall[path],1)<<','<<quantile(device[path],.5)<<','<<bytes<<'\n';}
}
template<int B=64> void width(const Options& o){if(o.bits==B)run<B>(o);else if constexpr(B<1024)width<B+32>(o);}
int main(int argc,char** argv){try{Options o;for(int i=1;i<argc;++i){std::string arg=argv[i];
    if(arg=="--help"){std::cout<<"section9_sources_limbforge --bits B --lanes L --steps K --terms T --share S --workers W --repeats R [--resident] [--fused] [--reverse] [--all-steps] [--per-group] [--warm] [--check-only] [--cpu-reference-check]\n";return 0;}
    if(arg=="--resident")o.resident=true;else if(arg=="--fused")o.fused=true;else if(arg=="--reverse")o.reverse=true;else if(arg=="--all-steps")o.all=true;else if(arg=="--per-group")o.per_group=true;else if(arg=="--warm")o.warm=true;else if(arg=="--check-only")o.check_only=true;else if(arg=="--cpu-reference-check")o.cpu_check=true;
    else{if(i+1==argc)throw std::invalid_argument("missing option value");std::string v=argv[++i];std::size_t used=0;if(v.empty()||v[0]=='-')throw std::invalid_argument("invalid integer");auto n=std::stoull(v,&used);if(used!=v.size()||n>UINT32_MAX)throw std::invalid_argument("invalid integer");
        if(arg=="--bits")o.bits=int(n);else if(arg=="--lanes")o.lanes=n;else if(arg=="--steps")o.steps=n;else if(arg=="--terms")o.terms=n;else if(arg=="--share")o.share=n;else if(arg=="--workers")o.workers=n;else if(arg=="--repeats")o.repeats=n;else throw std::invalid_argument("unknown option");}}
    if(o.bits<64||o.bits>1024||o.bits%32||!o.lanes||!o.share||!o.workers||o.workers>256||!o.repeats||o.repeats>1000)throw std::invalid_argument("invalid calibration options");
    width(o);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
