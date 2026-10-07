#include "reference.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iomanip>
#include <mutex>
#include <thread>
using namespace limbforge;
using Clock=std::chrono::steady_clock;
class Workers {
    std::vector<std::thread> threads;std::mutex mutex;std::condition_variable ready,done;
    std::function<void(std::size_t,std::size_t)> task;std::size_t count=0,generation=0,pending=0;bool stopping=false;
public:
    explicit Workers(unsigned n){for(unsigned id=0;id<n;++id)threads.emplace_back([this,id,n]{
        std::size_t seen=0;std::unique_lock<std::mutex> lock(mutex);
        for(;;){ready.wait(lock,[&]{return stopping||generation!=seen;});if(stopping)return;
            seen=generation;auto job=task;auto total=count;lock.unlock();job(total*id/n,total*(id+1)/n);lock.lock();
            if(!--pending)done.notify_one();}
    });}
    ~Workers(){{std::lock_guard<std::mutex> lock(mutex);stopping=true;}ready.notify_all();for(auto& t:threads)t.join();}
    void run(std::size_t n,std::function<void(std::size_t,std::size_t)> job){std::unique_lock<std::mutex> lock(mutex);
        count=n;task=std::move(job);pending=threads.size();++generation;ready.notify_all();done.wait(lock,[&]{return !pending;});}
    unsigned size()const{return unsigned(threads.size());}
};
struct MPArray {
    std::unique_ptr<mpfr_t[]> values;std::size_t size;
    MPArray(std::size_t n,int bits):values(new mpfr_t[n]),size(n){for(std::size_t i=0;i<n;++i)mpfr_init2(values[i],bits);}
    ~MPArray(){for(std::size_t i=0;i<size;++i)mpfr_clear(values[i]);}
    mpfr_ptr operator[](std::size_t i){return values[i];}
};
struct Samples {std::vector<double> cpu,parallel,device,wall;};
double quantile(std::vector<double> x,double q){std::sort(x.begin(),x.end());return x[std::size_t(q*(x.size()-1))];}
const char* name(Operation op){static const char* names[]={"add","sub","mul","div","complex_add","complex_mul","complex_div"};return names[int(op)];}
void report(int bits,const char* operation,std::size_t count,unsigned steps,Workers& workers,const Samples& s){
    std::cout<<bits<<','<<operation<<','<<count<<','<<steps<<','<<s.wall.size()<<','<<workers.size()<<','
        <<quantile(s.cpu,.5)<<','<<quantile(s.parallel,.5)<<','<<quantile(s.device,.5)<<','
        <<quantile(s.wall,.5)<<','<<quantile(s.wall,0)<<','<<quantile(s.wall,.9)<<std::endl;
}
template<int Bits,bool IsComplex> void arithmetic(Engine& engine,Workers& workers,std::size_t count,int repeats){
    using T=std::conditional_t<IsComplex,Complex<Bits/32>,Float<Bits>>;
    std::mt19937_64 rng(20261007+Bits);std::vector<T>a(count),b(count),out(count);
    MPArray ar(count,Bits),ai(count,Bits),br(count,Bits),bi(count,Bits),rr(count,Bits),ri(count,Bits);
    for(std::size_t i=0;i<count;++i){
        if constexpr(IsComplex){a[i]={reference::random_number<Bits>(rng,5),reference::random_number<Bits>(rng,5)};
            b[i]={reference::random_number<Bits>(rng,5),reference::random_number<Bits>(rng,5)};
            to_mpfr<Bits>(ar[i],a[i].re);to_mpfr<Bits>(ai[i],a[i].im);to_mpfr<Bits>(br[i],b[i].re);to_mpfr<Bits>(bi[i],b[i].im);
        }else{a[i]=reference::random_number<Bits>(rng,5);b[i]=reference::random_number<Bits>(rng,5);to_mpfr<Bits>(ar[i],a[i]);to_mpfr<Bits>(br[i],b[i]);}
    }
    for(int which=IsComplex?4:0;which<(IsComplex?7:4);++which){auto op=Operation(which);Samples samples;
        auto cpu=[&](std::size_t lo,std::size_t hi){reference::MP t0(Bits),t1(Bits),den(Bits);
            for(std::size_t i=lo;i<hi;++i){
                if constexpr(!IsComplex){switch(op){case Operation::add:mpfr_add(rr[i],ar[i],br[i],MPFR_RNDN);break;
                    case Operation::sub:mpfr_sub(rr[i],ar[i],br[i],MPFR_RNDN);break;
                    case Operation::mul:mpfr_mul(rr[i],ar[i],br[i],MPFR_RNDN);break;default:mpfr_div(rr[i],ar[i],br[i],MPFR_RNDN);}}
                else if(op==Operation::complex_add){mpfr_add(rr[i],ar[i],br[i],MPFR_RNDN);mpfr_add(ri[i],ai[i],bi[i],MPFR_RNDN);}
                else if(op==Operation::complex_mul){
                    mpfr_mul(t0.x,ar[i],br[i],MPFR_RNDN);mpfr_mul(t1.x,ai[i],bi[i],MPFR_RNDN);mpfr_sub(rr[i],t0.x,t1.x,MPFR_RNDN);
                    mpfr_mul(t0.x,ar[i],bi[i],MPFR_RNDN);mpfr_mul(t1.x,ai[i],br[i],MPFR_RNDN);mpfr_add(ri[i],t0.x,t1.x,MPFR_RNDN);
                }else{
                    mpfr_mul(t0.x,br[i],br[i],MPFR_RNDN);mpfr_mul(t1.x,bi[i],bi[i],MPFR_RNDN);mpfr_add(den.x,t0.x,t1.x,MPFR_RNDN);
                    mpfr_mul(t0.x,ar[i],br[i],MPFR_RNDN);mpfr_mul(t1.x,ai[i],bi[i],MPFR_RNDN);mpfr_add(rr[i],t0.x,t1.x,MPFR_RNDN);mpfr_div(rr[i],rr[i],den.x,MPFR_RNDN);
                    mpfr_mul(t0.x,ai[i],br[i],MPFR_RNDN);mpfr_mul(t1.x,ar[i],bi[i],MPFR_RNDN);mpfr_sub(ri[i],t0.x,t1.x,MPFR_RNDN);mpfr_div(ri[i],ri[i],den.x,MPFR_RNDN);
                }
            }
        };
        for(int warm=0;warm<2;++warm){cpu(0,count);workers.run(count,cpu);engine.run(Bits,op,a.data(),b.data(),out.data(),count);}
        for(int repeat=0;repeat<repeats;++repeat){
            auto serial=[&]{auto start=Clock::now();cpu(0,count);samples.cpu.push_back(std::chrono::duration<double>(Clock::now()-start).count());};
            auto parallel=[&]{auto start=Clock::now();workers.run(count,cpu);samples.parallel.push_back(std::chrono::duration<double>(Clock::now()-start).count());};
            if(repeat%2){parallel();serial();}else{serial();parallel();}
            auto t=engine.run(Bits,op,a.data(),b.data(),out.data(),count);samples.device.push_back(t.gpu_seconds);samples.wall.push_back(t.wall_seconds);
        }
        for(std::size_t i=0;i<count;++i){bool equal;
            if constexpr(IsComplex)equal=reference::equal_complex<Bits>(out[i],T{from_mpfr<Bits>(rr[i]),from_mpfr<Bits>(ri[i])});
            else equal=reference::equal<Bits>(out[i],from_mpfr<Bits>(rr[i]));
            if(!equal)throw std::runtime_error("benchmark MPFR mismatch: "+std::string(name(op)));
        }
        report(Bits,name(op),count,1,workers,samples);
#ifdef LIMBFORGE_RESIDENT_API
        auto da=engine.make_buffer<T>(count),db=engine.make_buffer<T>(count),dc=engine.make_buffer<T>(count);
        da.upload(a.data(),count);db.upload(b.data(),count);Samples resident;resident.cpu=samples.cpu;resident.parallel=samples.parallel;
        for(int repeat=-2;repeat<repeats;++repeat){auto batch=engine.batch();batch.run(op,da,db,dc);auto t=batch.submit().wait();
            if(repeat>=0){resident.device.push_back(t.gpu_seconds);resident.wall.push_back(t.wall_seconds);}}
        dc.download(out.data(),count);
        for(std::size_t i=0;i<count;++i){bool equal;
            if constexpr(IsComplex)equal=reference::equal_complex<Bits>(out[i],T{from_mpfr<Bits>(rr[i]),from_mpfr<Bits>(ri[i])});
            else equal=reference::equal<Bits>(out[i],from_mpfr<Bits>(rr[i]));
            if(!equal)throw std::runtime_error("resident benchmark MPFR mismatch");}
        auto label=std::string(name(op))+"_resident";report(Bits,label.c_str(),count,1,workers,resident);
#endif
    }
}
template<int Bits> void chain(Engine& engine,Workers& workers,std::size_t count,int repeats){
    constexpr unsigned steps=16;using F=Float<Bits>;std::mt19937_64 rng(20261007+Bits);
    std::vector<F>a(count),b(count),out(count);MPArray ar(count,Bits),br(count,Bits),rr(count,Bits);
    for(std::size_t i=0;i<count;++i){a[i]=reference::random_number<Bits>(rng,0);b[i]=reference::random_number<Bits>(rng,0);to_mpfr<Bits>(ar[i],a[i]);to_mpfr<Bits>(br[i],b[i]);}
    auto cpu=[&](std::size_t lo,std::size_t hi){for(unsigned s=0;s<steps;++s)for(auto i=lo;i<hi;++i)mpfr_mul(rr[i],rr[i],br[i],MPFR_RNDN);};
    auto reset=[&]{for(std::size_t i=0;i<count;++i)mpfr_set(rr[i],ar[i],MPFR_RNDN);};Samples samples;
    for(int repeat=-2;repeat<repeats;++repeat){reset();auto start=Clock::now();cpu(0,count);double serial=std::chrono::duration<double>(Clock::now()-start).count();
        reset();start=Clock::now();workers.run(count,cpu);double parallel=std::chrono::duration<double>(Clock::now()-start).count();
        out=a;double device=0;start=Clock::now();for(unsigned s=0;s<steps;++s)device+=engine.run(Bits,Operation::mul,out.data(),b.data(),out.data(),count).gpu_seconds;
        double wall=std::chrono::duration<double>(Clock::now()-start).count();
        if(repeat>=0){samples.cpu.push_back(serial);samples.parallel.push_back(parallel);samples.device.push_back(device);samples.wall.push_back(wall);}
    }
    for(std::size_t i=0;i<count;++i)if(!reference::equal<Bits>(out[i],from_mpfr<Bits>(rr[i])))throw std::runtime_error("chain MPFR mismatch");
    report(Bits,"mul_chain_host",count,steps,workers,samples);
#ifdef LIMBFORGE_RESIDENT_API
    auto x=engine.make_buffer<F>(count),y=engine.make_buffer<F>(count);Samples resident;
    resident.cpu=samples.cpu;resident.parallel=samples.parallel;
    for(int repeat=-2;repeat<repeats;++repeat){auto start=Clock::now();x.upload(a.data(),count);y.upload(b.data(),count);
        auto batch=engine.batch();for(unsigned s=0;s<steps;++s)batch.run(Operation::mul,x,y,x);
        auto t=batch.submit().wait();x.download(out.data(),count);double wall=std::chrono::duration<double>(Clock::now()-start).count();
        if(repeat>=0){resident.device.push_back(t.gpu_seconds);resident.wall.push_back(wall);}}
    for(std::size_t i=0;i<count;++i)if(!reference::equal<Bits>(out[i],from_mpfr<Bits>(rr[i])))throw std::runtime_error("resident chain MPFR mismatch");
    report(Bits,"mul_chain_resident",count,steps,workers,resident);
#endif
}
template<int Bits> void suite(Engine& e,Workers& w,const std::vector<std::size_t>& counts,int repeats){for(auto n:counts){arithmetic<Bits,false>(e,w,n,repeats);arithmetic<Bits,true>(e,w,n,repeats);chain<Bits>(e,w,n,repeats);}}
int main(int argc,char** argv){try{
    std::vector<std::size_t> counts={256,4096,65536};int repeats=9;unsigned workers=std::max(1u,std::thread::hardware_concurrency());
    for(int i=1;i<argc;++i){std::string arg=argv[i];if(arg=="--quick")counts={4096};else if(arg=="--repeats"&&i+1<argc)repeats=std::stoi(argv[++i]);
        else if(arg=="--workers"&&i+1<argc)workers=unsigned(std::stoul(argv[++i]));else if(arg=="--count"&&i+1<argc)counts={std::stoull(argv[++i])};
        else throw std::invalid_argument("usage: benchmark_limbforge [--quick|--count N] [--repeats N] [--workers N]");}
    if(repeats<3||repeats>100||!workers||workers>128)throw std::invalid_argument("invalid repeats or workers");
    for(auto n:counts)if(!n||n>1000000)throw std::invalid_argument("count must be in 1..1000000");
    Engine e;Workers w(workers);std::cerr<<"Device: "<<e.device_name()<<"; MPFR "<<mpfr_get_version()<<"; workers="<<workers<<"; repeats="<<repeats<<"; seed=20261007\n";
    std::cout<<std::setprecision(10)<<"bits,operation,count,steps,samples,cpu_workers,cpu_serial_s,cpu_parallel_s,gpu_s,wall_median_s,wall_min_s,wall_p90_s\n";
    suite<256>(e,w,counts,repeats);suite<384>(e,w,counts,repeats);suite<1024>(e,w,counts,repeats);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
