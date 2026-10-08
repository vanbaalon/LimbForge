#pragma once
#include <cmath>
#include <map>
#include <string>
#include <tuple>
#include <stdexcept>
namespace limbforge {
// A measured lookup, not online autotuning. Match the entire shape and arithmetic contract.
struct DispatchKey {
    std::string operation;int bits=0;std::size_t count=0,m=0,n=0,k=0;unsigned cpu_workers=1;bool fused=false;
    // Profile identity includes machine/library/build settings; host and resident timings differ.
    std::string profile;bool resident=false;
    auto tuple()const{return std::tie(operation,bits,count,m,n,k,cpu_workers,fused,profile,resident);}
    bool operator<(const DispatchKey& b)const{return tuple()<b.tuple();}
};
struct DispatchSample {double cpu_wall=0,gpu_wall=0;};
enum class BackendRecommendation {unknown,cpu,gpu};
class BreakEvenTable {
    std::map<DispatchKey,DispatchSample> samples_;
public:
    void record(const DispatchKey& key,DispatchSample sample){
        if(key.operation.empty()||key.bits<64||key.bits>1024||key.bits%32||!key.cpu_workers||
           !std::isfinite(sample.cpu_wall)||!std::isfinite(sample.gpu_wall)||sample.cpu_wall<=0||sample.gpu_wall<=0)
            throw std::invalid_argument("break-even table: invalid key or timing");
        samples_[key]=sample;
    }
    // Near ties remain unknown. No extrapolation to unmeasured widths or shapes; caller chooses a default.
    BackendRecommendation recommend(const DispatchKey& key,double margin=0.05)const{
        if(!std::isfinite(margin)||margin<0||margin>=1)throw std::invalid_argument("invalid recommendation margin");
        auto it=samples_.find(key);if(it==samples_.end())return BackendRecommendation::unknown;auto s=it->second;
        if(s.gpu_wall<s.cpu_wall*(1-margin))return BackendRecommendation::gpu;
        if(s.cpu_wall<s.gpu_wall*(1-margin))return BackendRecommendation::cpu;return BackendRecommendation::unknown;
    }
};
}
