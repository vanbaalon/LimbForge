#pragma once
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
const char* name(Operation op){static const char* names[]={"add","sub","mul","div","complex_add","complex_mul","complex_div","square","sqrt"};return names[int(op)];}
void report(int bits,const char* operation,std::size_t count,unsigned steps,Workers& workers,const Samples& s){
    std::cout<<bits<<','<<operation<<','<<count<<','<<steps<<','<<s.wall.size()<<','<<workers.size()<<','
        <<quantile(s.cpu,.5)<<','<<quantile(s.parallel,.5)<<','<<quantile(s.device,.5)<<','
        <<quantile(s.wall,.5)<<','<<quantile(s.wall,0)<<','<<quantile(s.wall,.9)<<std::endl;
}
