// Unit-level clock probe: the exact kernel/source is unchanged. No installed API.
#define main original_exact4_test_main
#include "test.cpp"
#undef main
#include <chrono>
#include <thread>
#include <iomanip>
using Clock=std::chrono::steady_clock;
struct ClockOptions {int bits=352;unsigned count=1024,repeats=9;bool check_only=false;};
template<int B> void clocks(const ClockOptions& o){using T=Complex<B/32>;Engine e;ExactGemm4 unit(e);ExactGemm4Shape shape{o.count,16,16,16};
    std::mt19937_64 rng(0x6578616374+B);std::vector<T> a(16*o.count),b(a.size()),zeroes(a.size());
    for(auto* v:{&a,&b})for(auto& x:*v)x=random_value<B,T>(rng,4);auto expected=oracle<B>(shape,a,b,zeroes);
    auto ab=upload(e,a),bb=upload(e,b);std::vector<Buffer<T>> outputs;for(unsigned i=0;i<32;++i)outputs.push_back(e.make_buffer<T>(a.size()));
    auto batch_call=[&](unsigned calls){auto start=Clock::now();auto batch=e.batch();std::vector<ExactGemm4Ticket> tickets;tickets.reserve(calls);
        for(unsigned i=0;i<calls;++i)tickets.push_back(unit.gemm(batch,shape,ab,bb,outputs[i]));auto timing=batch.submit().wait();
        for(auto& t:tickets){auto r=t.report();need(!r.host_components&&!r.provisional_reads,"clock probe unexpectedly repaired/read provisional");need(r.gpu_components==32*o.count,"clock component count");}
        timing.wall_seconds=std::chrono::duration<double>(Clock::now()-start).count();return timing;};
    auto check=[&](unsigned calls){for(unsigned i=0;i<calls;++i)compare<B>(download(outputs[i]),expected,"clock independent multi-call output");};
    batch_call(1);check(1);batch_call(32);check(32);
    if(o.check_only){std::cout<<B<<" bits: 1/32-call resident batches match independent MPFR, all outputs/reports checked; no timings\n";return;}
    std::cout<<"phase,repetition,bits,count,calls,gpu_seconds_per_call,wall_seconds_per_call\n";
    for(unsigned rep=0;rep<o.repeats;++rep){std::this_thread::sleep_for(std::chrono::milliseconds(100));auto t=batch_call(1);check(1);
        std::cout<<"idle_delay_100ms,"<<rep<<','<<B<<','<<o.count<<",1,"<<std::setprecision(12)<<t.gpu_seconds<<','<<t.wall_seconds<<'\n'<<std::flush;
        double device_warmup=0;do{auto warm=batch_call(32);device_warmup+=warm.gpu_seconds;check(32);}while(device_warmup<.2);
        auto warm=batch_call(32);check(32);std::cout<<"batch32_after_200ms_device_work,"<<rep<<','<<B<<','<<o.count<<",32,"<<warm.gpu_seconds/32<<','<<warm.wall_seconds/32<<'\n'<<std::flush;
    }
    std::cerr<<"Every warm/timed output matches MPFR. Resident unit scope: ordered snapshots/dispatches included in device; encoding, scratch allocation, wait and report included in wall; result transfers and verification excluded. No controlled GPU-frequency or continuously-warm claim.\n";
}
int main(int argc,char** argv){try{ClockOptions o;for(int i=1;i<argc;++i){std::string arg=argv[i];if(arg=="--check-only"){o.check_only=true;continue;}if(++i==argc)throw std::invalid_argument("missing clock argument");std::size_t used=0;std::string value=argv[i];auto n=std::stoul(value,&used);if(used!=value.size()||value[0]=='-'||!n||n>2048)throw std::invalid_argument("invalid clock argument");if(arg=="--bits")o.bits=n;else if(arg=="--count")o.count=n;else if(arg=="--repeats")o.repeats=n;else throw std::invalid_argument("unknown clock argument");}
    if(o.bits<64||o.bits>1024||o.bits%32||o.repeats>20)throw std::invalid_argument("unsupported clock precision/repetitions");switch(o.bits/32){
#define CASE(N) case N:clocks<32*N>(o);break;
CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9) CASE(10) CASE(11) CASE(12) CASE(13) CASE(14) CASE(15) CASE(16) CASE(17) CASE(18) CASE(19) CASE(20) CASE(21) CASE(22) CASE(23) CASE(24) CASE(25) CASE(26) CASE(27) CASE(28) CASE(29) CASE(30) CASE(31) CASE(32)
#undef CASE
    }return 0;}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
