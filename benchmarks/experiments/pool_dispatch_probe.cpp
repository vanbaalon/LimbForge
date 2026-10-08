// Round 40 diagnostic (not a library file): dispatch cost of host worker pools, 2000 dispatches of `items` items of ~`work`
// microseconds each. OldPool waits for every worker (rounds 23-31), Pool returns when every item is done (round 40, as in
// src/linalg.mm), SpinPool also spins up to 20 or 100 us before blocking (rejected: no consistent gain on the loaded host).
// Build: clang++ -std=c++20 -O2 pool_dispatch_probe.cpp -o pool_dispatch_probe; run: pool_dispatch_probe [threads items work_us]
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
class OldPool {
    std::vector<std::thread> workers;std::mutex m;std::condition_variable wake,idle;std::function<void(unsigned)> job;std::size_t generation=0;unsigned active=0;bool stop=false;
public:
    explicit OldPool(unsigned n){for(unsigned i=1;i<n;++i)workers.emplace_back([this,i]{std::size_t seen=0;std::unique_lock<std::mutex> l(m);
        for(;;){wake.wait(l,[&]{return stop||generation!=seen;});if(stop)return;seen=generation;auto f=job;l.unlock();f(i);l.lock();if(!--active)idle.notify_all();}});}
    ~OldPool(){{std::lock_guard<std::mutex> l(m);stop=true;}wake.notify_all();for(auto& w:workers)w.join();}
    void run(const std::function<void(unsigned)>& f){
        {std::lock_guard<std::mutex> l(m);job=f;active=unsigned(workers.size());++generation;}wake.notify_all();f(0);
        std::unique_lock<std::mutex> l(m);idle.wait(l,[&]{return !active;});}
    void items(std::size_t n,const std::function<void(std::size_t)>& f){std::atomic<std::size_t> next{0};run([&](unsigned){for(std::size_t i;(i=next.fetch_add(1))<n;)f(i);});}
};
class Pool {
    struct Batch {std::function<void(std::size_t,unsigned)> f;std::size_t n=0;std::atomic<std::size_t> next{0},done{0};};
    std::vector<std::thread> workers;std::mutex m;std::condition_variable wake,finished;std::shared_ptr<Batch> batch;std::size_t generation=0;bool stop=false;
    void work(Batch& b,unsigned id){for(std::size_t i;(i=b.next.fetch_add(1))<b.n;){b.f(i,id);if(b.done.fetch_add(1)+1==b.n){std::lock_guard<std::mutex> l(m);finished.notify_all();}}}
public:
    explicit Pool(unsigned n){for(unsigned i=1;i<n;++i)workers.emplace_back([this,i]{std::size_t seen=0;std::unique_lock<std::mutex> l(m);
        for(;;){wake.wait(l,[&]{return stop||generation!=seen;});if(stop)return;seen=generation;auto b=batch;l.unlock();work(*b,i);l.lock();}});}
    ~Pool(){{std::lock_guard<std::mutex> l(m);stop=true;}wake.notify_all();for(auto& w:workers)w.join();}
    void run(std::size_t n,std::function<void(std::size_t,unsigned)> f){
        if(!n)return;if(workers.empty()||n==1){for(std::size_t i=0;i<n;++i)f(i,0);return;}
        auto b=std::make_shared<Batch>();b->f=std::move(f);b->n=n;
        {std::lock_guard<std::mutex> l(m);batch=b;++generation;}
        if(n-1<workers.size())for(std::size_t i=0;i+1<n;++i)wake.notify_one();else wake.notify_all();
        work(*b,0);
        if(b->done.load()<n){std::unique_lock<std::mutex> l(m);finished.wait(l,[&]{return b->done.load()==n;});}}
};
class SpinPool {
    struct Batch {std::function<void(std::size_t,unsigned)> f;std::size_t n=0;std::atomic<std::size_t> next{0},done{0};};
    std::vector<std::thread> workers;std::mutex m;std::condition_variable wake,finished;std::shared_ptr<Batch> batch;std::atomic<std::size_t> generation{0};bool stop=false;double us;
    void work(Batch& b,unsigned id){for(std::size_t i;(i=b.next.fetch_add(1))<b.n;){b.f(i,id);if(b.done.fetch_add(1)+1==b.n){std::lock_guard<std::mutex> l(m);finished.notify_all();}}}
    bool spin_until(const std::function<bool()>& c){auto t=std::chrono::steady_clock::now();while(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count()<us){if(c())return true;}return c();}
public:
    explicit SpinPool(unsigned n,double spin_us):us(spin_us){for(unsigned i=1;i<n;++i)workers.emplace_back([this,i]{std::size_t seen=0;
        for(;;){spin_until([&]{return generation.load()!=seen;});std::unique_lock<std::mutex> l(m);wake.wait(l,[&]{return stop||generation.load()!=seen;});if(stop)return;seen=generation.load();auto b=batch;l.unlock();work(*b,i);}});}
    ~SpinPool(){{std::lock_guard<std::mutex> l(m);stop=true;generation.fetch_add(1);}wake.notify_all();for(auto& w:workers)w.join();}
    void run(std::size_t n,std::function<void(std::size_t,unsigned)> f){
        auto b=std::make_shared<Batch>();b->f=std::move(f);b->n=n;
        {std::lock_guard<std::mutex> l(m);batch=b;generation.fetch_add(1);}wake.notify_all();
        work(*b,0);
        if(!spin_until([&]{return b->done.load()==n;})){std::unique_lock<std::mutex> l(m);finished.wait(l,[&]{return b->done.load()==n;});}}
};
volatile double sink;
void spin(double us){auto t=std::chrono::steady_clock::now();double x=0;while(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count()<us)x+=1;sink=x;}
int main(int argc,char** argv){unsigned threads=argc>1?std::stoul(argv[1]):18;std::size_t items=argc>2?std::stoul(argv[2]):72;double work=argc>3?std::stod(argv[3]):3;int reps=2000;
    for(int round=0;round<2;++round){
        {OldPool p(threads);auto t=std::chrono::steady_clock::now();for(int r=0;r<reps;++r)p.items(items,[&](std::size_t){spin(work);});
         double s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();std::printf("old pool: %d dispatches of %zu x %.1f us: %.0f ms (ideal %.0f ms)\n",reps,items,work,s*1e3,reps*items*work/threads/1e3);}
        {Pool p(threads);auto t=std::chrono::steady_clock::now();for(int r=0;r<reps;++r)p.run(items,[&](std::size_t,unsigned){spin(work);});
         double s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();std::printf("new pool: %d dispatches of %zu x %.1f us: %.0f ms (ideal %.0f ms)\n",reps,items,work,s*1e3,reps*items*work/threads/1e3);}
        for(double us:{20.0,100.0}){SpinPool p(threads,us);auto t=std::chrono::steady_clock::now();for(int r=0;r<reps;++r)p.run(items,[&](std::size_t,unsigned){spin(work);});
         double s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();std::printf("spin %3.0f us: %d dispatches of %zu x %.1f us: %.0f ms (ideal %.0f ms)\n",us,reps,items,work,s*1e3,reps*items*work/threads/1e3);}}
}
