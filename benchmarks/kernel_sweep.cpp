// Warm-clock device-time sweep over all 31 precisions. Each case first keeps the GPU busy with the same
// operation, then times command buffers holding R dependent dispatches; device time / R removes submission
// latency and clock ramp-up. A cold column times one dispatch after an idle pause. Outputs are MPFR-checked.
#include "benchmark_support.hpp"
#include <thread>
struct Config {std::size_t count=65536,lanes=4096;int dispatches=32,samples=7,cold=3,only_bits=0;unsigned steps=64;double warm=0.2;std::string only_op;};
Config cfg;
double median(std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];}
double minimum(const std::vector<double>& x){return *std::min_element(x.begin(),x.end());}
// Nominal 32x32-bit limb products per value: N^2 per real multiply or quotient.
double products(Operation op,int n){switch(op){case Operation::mul:case Operation::div:case Operation::square:return double(n)*n;
    case Operation::complex_mul:return 4.*n*n;case Operation::complex_div:return 8.*n*n;default:return 0;}}
bool selected(const std::string& op){return cfg.only_op.empty()||cfg.only_op==op;}
void row(int bits,const std::string& op,std::size_t count,int dispatches,const std::vector<double>& warm,const std::vector<double>& cold,
         double bytes,double limb_products,const PipelineInfo& p){
    double t=median(warm);
    std::cout<<bits<<','<<op<<','<<count<<','<<dispatches<<','<<warm.size()<<','<<t<<','<<minimum(warm)<<','<<(cold.empty()?0:median(cold))<<','
        <<t/count*1e9<<','<<bytes*count/t/1e9<<','<<limb_products*count/t<<','<<p.simd_width<<','<<p.max_threads<<','<<p.threads_per_threadgroup<<std::endl;
}
template<int Bits,bool IsComplex> void arithmetic(Engine& engine){
    using T=std::conditional_t<IsComplex,Complex<Bits/32>,Float<Bits>>;const std::size_t n=cfg.count;
    std::mt19937_64 rng(20261007+Bits);std::vector<T>a(n),b(n),out(n),expected(n);
    for(std::size_t i=0;i<n;++i){if constexpr(IsComplex){a[i]={reference::random_number<Bits>(rng,5),reference::random_number<Bits>(rng,5)};b[i]={reference::random_number<Bits>(rng,5),reference::random_number<Bits>(rng,5)};}
        else {a[i]=reference::random_number<Bits>(rng,5);b[i]=reference::random_number<Bits>(rng,5);}}
    auto x=engine.make_buffer<T>(n),y=engine.make_buffer<T>(n),z=engine.make_buffer<T>(n);y.upload(b.data(),n);
    std::vector<Operation> ops=IsComplex?std::vector<Operation>{Operation::complex_add,Operation::complex_mul,Operation::complex_div}
        :std::vector<Operation>{Operation::add,Operation::sub,Operation::mul,Operation::div,Operation::square,Operation::sqrt};
    for(auto op:ops){if(!selected(name(op)))continue;bool unary=operation_is_unary(op);
        if constexpr(!IsComplex)if(op==Operation::sqrt)for(auto& v:a)v.sign=v.sign?1:0;
        x.upload(a.data(),n);
        auto submit=[&](int r){auto batch=engine.batch();for(int k=0;k<r;++k){if(unary)batch.run(op,x,z);else batch.run(op,x,y,z);}return batch.submit().wait().gpu_seconds;};
        submit(1);std::vector<double> cold,warm;
        for(int s=0;s<cfg.cold;++s){std::this_thread::sleep_for(std::chrono::milliseconds(100));cold.push_back(submit(1));}
        for(auto start=Clock::now();std::chrono::duration<double>(Clock::now()-start).count()<cfg.warm;)submit(cfg.dispatches);
        for(int s=0;s<cfg.samples;++s)warm.push_back(submit(cfg.dispatches)/cfg.dispatches);
        z.download(out.data(),n);
        for(std::size_t i=0;i<n;++i){bool equal;
            if constexpr(IsComplex)equal=reference::equal_complex<Bits>(out[i],reference::complex<Bits>(op,a[i],b[i]));
            else equal=reference::equal<Bits>(out[i],reference::real<Bits>(op,a[i],b[i]));
            if(!equal)throw std::runtime_error("sweep MPFR mismatch bits="+std::to_string(Bits)+" op="+name(op)+" index="+std::to_string(i));}
        row(Bits,name(op),n,cfg.dispatches,warm,cold,double(sizeof(T))*(unary?2:3),products(op,Bits/32),engine.pipeline_info(Bits,op));
    }
}
// One recurrence dispatch runs `steps` dependent steps (4 complex multiply-adds each); report time per step.
template<int Bits> void recurrence(Engine& engine){
    if(!selected("recurrence_step"))return;
    using C=Complex<Bits/32>;const std::size_t n=cfg.lanes;const unsigned steps=cfg.steps;std::mt19937_64 rng(Bits);
    std::vector<C> seed(4*n),weight(4*steps),out(n);
    for(auto& v:seed)v={reference::random_number<Bits>(rng,5),reference::random_number<Bits>(rng,5)};
    for(auto& v:weight){v={reference::random_number<Bits>(rng,3),reference::random_number<Bits>(rng,3)};v.re.exponent-=5;v.im.exponent-=5;}
    // All lanes share one weight sequence, so the weight stream stays cached.
    auto submit=[&]{return engine.recurrence(Bits,seed.data(),weight.data(),out.data(),n,steps,unsigned(n)).gpu_seconds;};
    submit();std::vector<double> cold,warm;
    for(int s=0;s<cfg.cold;++s){std::this_thread::sleep_for(std::chrono::milliseconds(100));cold.push_back(submit()/steps);}
    for(auto start=Clock::now();std::chrono::duration<double>(Clock::now()-start).count()<cfg.warm;)submit();
    for(int s=0;s<cfg.samples;++s)warm.push_back(submit()/steps);
    for(std::size_t i=0;i<std::min<std::size_t>(n,16);++i){C ring[4];for(int j=0;j<4;++j)ring[j]=seed[j*n+i];int head=0;
        for(unsigned t=0;t<steps;++t){C sum={zero<Bits/32>(),zero<Bits/32>()};
            for(int j=0;j<4;++j)sum=reference::complex<Bits>(Operation::complex_add,sum,reference::complex<Bits>(Operation::complex_mul,weight[t*4+j],ring[(head+j)%4]));
            ring[head]=sum;head=(head+1)%4;}
        if(!reference::equal_complex<Bits>(out[i],ring[(head+3)%4]))throw std::runtime_error("sweep recurrence MPFR mismatch bits="+std::to_string(Bits));}
    row(Bits,"recurrence_step",n,1,warm,cold,0,16.*(Bits/32)*(Bits/32),PipelineInfo{0,0,0});
}
template<int Bits> void all(Engine& engine){
    if(!cfg.only_bits||cfg.only_bits==Bits){arithmetic<Bits,false>(engine);arithmetic<Bits,true>(engine);recurrence<Bits>(engine);}
    if constexpr(Bits<1024)all<Bits+32>(engine);
}
int main(int argc,char** argv){try{
    for(int i=1;i<argc;++i){std::string arg=argv[i];auto next=[&]{if(i+1>=argc)throw std::invalid_argument("missing value for "+arg);return std::string(argv[++i]);};
        if(arg=="--count")cfg.count=std::stoull(next());else if(arg=="--lanes")cfg.lanes=std::stoull(next());else if(arg=="--steps")cfg.steps=unsigned(std::stoul(next()));
        else if(arg=="--dispatches")cfg.dispatches=std::stoi(next());else if(arg=="--samples")cfg.samples=std::stoi(next());else if(arg=="--cold")cfg.cold=std::stoi(next());
        else if(arg=="--warm")cfg.warm=std::stod(next());else if(arg=="--bits")cfg.only_bits=std::stoi(next());else if(arg=="--operation")cfg.only_op=next();
        else throw std::invalid_argument("usage: kernel_sweep [--bits B] [--operation OP|recurrence_step] [--count N] [--dispatches R] [--samples S] [--cold C] [--warm SECONDS] [--lanes L] [--steps K]");}
    if(!cfg.count||cfg.count>1000000||!cfg.lanes||cfg.lanes>1000000||cfg.dispatches<1||cfg.samples<1||cfg.cold<0||!cfg.steps)throw std::invalid_argument("invalid sweep configuration");
    if(cfg.only_bits&&(cfg.only_bits<64||cfg.only_bits>1024||cfg.only_bits%32))throw std::invalid_argument("bits must be a multiple of 32 in [64,1024]");
    Engine engine;std::cerr<<engine.device_name()<<"; warm sweep: "<<cfg.dispatches<<" dispatches per command buffer, every output MPFR-checked\n";
    std::cout<<std::setprecision(6)<<"bits,operation,count,dispatches_per_buffer,samples,warm_device_s,warm_device_min_s,cold_device_s,"
        "ns_per_value,effective_GBps,nominal_limb_products_per_s,simd_width,max_threads,threads_per_threadgroup\n";
    all<64>(engine);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
