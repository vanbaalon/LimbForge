#import <Foundation/Foundation.h>
#include "limbforge/batched_linalg.hpp"
#include "engine_internal.hpp"
#include "linalg_internal.hpp"
#include <chrono>
#include <cmath>
#include <cstdlib>
namespace limbforge {
namespace {
using Clock=std::chrono::steady_clock;using O=detail::Operand;using S=std::shared_ptr<detail::BufferStorage>;
using I=detail::Internal;using H=detail::LinalgHooks;
double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
std::size_t checked_size(std::size_t a,std::size_t b){if(b&&a>std::size_t(-1)/b)throw std::invalid_argument("blocked trials: size overflow");return a*b;}
struct Claim {
    std::vector<S> held;
    Claim(Engine& e,std::initializer_list<std::pair<O,std::size_t>> operands){
        // Validate engine ownership before claiming, including buffers not used by a GPU update.
        auto b=e.batch();for(auto& item:operands){const auto& o=item.first;if(o.size<item.second)throw std::invalid_argument("blocked trials: undersized buffer");
            if(!item.second&&!o.storage)continue;I::retain(b,o.storage);if(std::find(held.begin(),held.end(),o.storage)==held.end())held.push_back(o.storage);}
        std::size_t done=0;try{for(auto& s:held){bool idle=false;if(!s->busy.compare_exchange_strong(idle,true))throw std::logic_error("blocked trials: buffer belongs to an unwaited submission");++done;}}
        catch(...){for(std::size_t i=0;i<done;++i)held[i]->busy.store(false,std::memory_order_release);throw;}
    }
    ~Claim(){for(auto& s:held)s->busy.store(false,std::memory_order_release);}
};
template<int N> Number<N>* pointer(O o){return o.storage?static_cast<Number<N>*>(o.storage->buffer.contents):nullptr;}
// Source and trailing matrices are pooled by BatchedLinalg, one slot per simultaneously encoded trial.
using Scratch=std::function<std::pair<O,O>(CommandBatch&,std::size_t,std::size_t)>;
template<int N> BlockedTrialsInfo factor(Engine& engine,Linalg& la,const CholeskyTrials& s,O A,O diagonal,O mu,O output,O status,const FactorOptions& o,const Scratch& scratch,bool lookahead,bool grouped){@autoreleasepool{
    using T=Number<N>;auto start=Clock::now();BlockedTrialsInfo result;if(!s.count)return result;
    const auto n=s.n,count=s.count,n2=n*n,nb=o.block?std::min(o.block,n):n;
    Claim claim(engine,{{A,n2},{diagonal,n},{mu,count},{output,count*n2},{status,count}});
    auto a=pointer<N>(A),d=pointer<N>(diagonal),shifts=pointer<N>(mu),L=pointer<N>(output);auto states=static_cast<std::uint32_t*>(status.storage->buffer.contents);
    H::each(la,count*n,[&](std::size_t q){auto t=q/n,i=q%n;auto row=L+t*n2+i*n;
        for(std::size_t j=0;j<n;++j)row[j]=j>i?zero<N>():a[i*n+j];row[i]=add(row[i],mul(shifts[t],d[i]));});
    std::fill(states,states+count,0);std::vector<std::size_t> end(count),first(count);std::vector<std::size_t> active;
    auto D=[](const T& c,const T* x,const T* y,std::size_t k){return exact_dot_add<N>(&c,true,x,1,y,1,k);};
    auto row_panel=[&](std::size_t t,std::size_t i,std::size_t k0,std::size_t ke){auto w=L+t*n2;auto row=w+i*n;
        for(std::size_t j=k0;j<ke;++j)row[j]=div(D(row[j],row+k0,w+j*n+k0,j-k0),w[j*n+j]);};
    auto panel=[&](std::size_t k0,std::size_t k1,const std::vector<std::size_t>& active){
        ++result.panels;auto begin=Clock::now();
        // Independent diagonal blocks across trials, followed by all outside rows on the same persistent worker pool.
        H::each(la,active.size(),[&](std::size_t q){auto t=active[q];auto w=L+t*n2;std::size_t p=k1;
            for(std::size_t i=k0;i<k1;++i){row_panel(t,i,k0,i);auto value=D(w[i*n+i],w+i*n+k0,w+i*n+k0,i-k0);
                if(value.status||value.sign<=0){p=i;states[t]=std::uint32_t(i+1);break;}w[i*n+i]=sqrt(value);}
            end[t]=p;first[t]=p<k1?p+1:k1;});
        H::each(la,active.size()*(n-k0),[&](std::size_t q){auto t=active[q/(n-k0)],i=k0+q%(n-k0);if(i>=first[t])row_panel(t,i,k0,end[t]);});
        H::each(la,active.size()*(n-k0),[&](std::size_t q){auto t=active[q/(n-k0)],i=k0+q%(n-k0);if(states[t]&&i>=end[t]){
            auto row=L+t*n2+i*n;for(std::size_t j=end[t];j<=i;++j)row[j]=zero<N>(invalid);}});
        result.panel_seconds+=elapsed(begin);
    };
    bool prepared=false;
    for(std::size_t k0=0;k0<n;k0+=nb){const auto k1=std::min(n,k0+nb),K=k1-k0;active.clear();for(std::size_t t=0;t<count;++t)if(!states[t])active.push_back(t);
        if(active.empty())break;
        if(!prepared)panel(k0,k1,active);prepared=false;if(k1==n)break;
        active.erase(std::remove_if(active.begin(),active.end(),[&](std::size_t t){return states[t]!=0;}),active.end());if(active.empty())break;
        auto begin=Clock::now();const auto r=n-k1;const bool gpu=o.gpu&&double(active.size())*double(r)*double(r+1)/2*double(K)>=o.host_macs;
        if(!gpu){H::each(la,active.size()*r,[&](std::size_t q){auto t=active[q/r],i=k1+q%r;auto w=L+t*n2;
                for(std::size_t j=k1;j<=i;++j)w[i*n+j]=D(w[i*n+j],w+i*n+k0,w+j*n+k0,K);});result.host_updates+=active.size();}
        else{
            auto batch=engine.batch();std::vector<Buffer<T>> packed,updated;std::vector<T*> p,c;std::vector<LinalgTicket> tickets;
            Buffer<T> grouped_a,grouped_c;std::shared_ptr<detail::GroupedSyrkPass> grouped_pass;
            if(grouped){auto space=scratch(batch,checked_size(active.size(),K*r),checked_size(active.size(),r*r));
                grouped_a=detail::Access::buffer<T>(space.first);grouped_c=detail::Access::buffer<T>(space.second);
                auto pa=grouped_a.mapped(),pc=grouped_c.mapped();for(std::size_t q=0;q<active.size();++q){p.push_back(pa+q*K*r);c.push_back(pc+q*r*r);}}
            else for(std::size_t q=0;q<active.size();++q){auto space=scratch(batch,K*r,r*r);packed.push_back(detail::Access::buffer<T>(space.first));updated.push_back(detail::Access::buffer<T>(space.second));
                p.push_back(packed.back().mapped());c.push_back(updated.back().mapped());}
            H::each(la,active.size()*r,[&](std::size_t q){auto b=q/r,i=q%r,t=active[b];auto w=L+t*n2;
                for(std::size_t k=0;k<K;++k)p[b][k*r+i]=w[(k1+i)*n+k0+k];
                for(std::size_t j=0;j<r;++j)c[b][i*r+j]=j<=i?w[(k1+i)*n+k1+j]:zero<N>();});
            if(grouped)grouped_pass=H::syrk_group(la,batch,32*N,detail::Access::operand(grouped_a),detail::Access::operand(grouped_c),active.size(),K,r);
            if(grouped&&!grouped_pass){for(std::size_t q=0;q<active.size();++q){auto space=scratch(batch,K*r,r*r);
                packed.push_back(detail::Access::buffer<T>(space.first));updated.push_back(detail::Access::buffer<T>(space.second));
                auto pa=packed.back().mapped(),pc=updated.back().mapped();std::copy(p[q],p[q]+K*r,pa);std::copy(c[q],c[q]+r*r,pc);p[q]=pa;c[q]=pc;}}
            if(!grouped_pass)for(std::size_t q=0;q<active.size();++q)tickets.push_back(la.syrk(batch,packed[q],K,r,updated[q],true,true));
            auto submission=batch.submit();std::size_t ready_columns=0;
            if(lookahead){
                const auto next=std::min(n,k1+nb);ready_columns=next-k1;
                // The GPU reads packed current-panel columns and writes separate scratch.
                // Recompute just the next-panel Schur strip on the original host matrix;
                // then factor that panel while the full trailing SYRK is in flight.
                // Both paths round once from the same old entry and exact K products.
                try{
                    H::each(la,active.size()*r,[&](std::size_t q){auto t=active[q/r],i=k1+q%r;auto w=L+t*n2;
                        for(std::size_t j=k1;j<next&&j<=i;++j)w[i*n+j]=D(w[i*n+j],w+i*n+k0,w+j*n+k0,K);});
                    panel(k1,next,active);prepared=true;
                }catch(...){try{submission.wait();}catch(...){}throw;}
            }
            auto timing=submission.wait();result.timing.gpu_seconds+=timing.gpu_seconds;++result.submissions;result.gpu_updates+=active.size();
            // wait resolves every repair before scatter or any dependent read of GPU scratch.
            if(grouped_pass){if(!grouped_pass->done.load(std::memory_order_acquire)||grouped_pass->provisional_reads)throw std::logic_error("blocked trials: unresolved grouped update");result.gpu_outputs+=grouped_pass->gpu_outputs;result.grouped_updates+=active.size();}
            for(auto& ticket:tickets){if(!ticket.resolved()||ticket.provisional_reads())throw std::logic_error("blocked trials: unresolved trailing update");
                result.gpu_outputs+=ticket.report().gpu_outputs;result.fallback_outputs+=ticket.report().fallback_outputs;}
            H::each(la,active.size()*r,[&](std::size_t q){auto b=q/r,i=q%r,t=active[b];auto w=L+t*n2;
                // Preserve the completed look-ahead columns and a newly failed trial's
                // invalid tail. A successful trial still needs the remaining Schur matrix.
                if(!prepared||!states[t])for(std::size_t j=prepared?ready_columns:0;j<=i;++j)w[(k1+i)*n+k1+j]=c[b][i*r+j];});
        }
        result.update_seconds+=elapsed(begin);
    }
    result.timing.wall_seconds=elapsed(start);return result;
}}
template<int N=2> BlockedTrialsInfo width(int bits,Engine& e,Linalg& la,const CholeskyTrials& s,O a,O d,O mu,O l,O status,const FactorOptions& o,const Scratch& scratch,bool lookahead,bool grouped){
    if(bits==32*N)return factor<N>(e,la,s,a,d,mu,l,status,o,scratch,lookahead,grouped);if constexpr(N<32)return width<N+1>(bits,e,la,s,a,d,mu,l,status,o,scratch,lookahead,grouped);throw std::invalid_argument("blocked trials: invalid bits");}
}
BlockedTrialsInfo BatchedLinalg::factor_trials(Linalg& la,int bits,const CholeskyTrials& s,O A,O D,O mu,O L,O status,const FactorOptions& o){
    if(bits<64||bits>1024||bits%32)throw std::invalid_argument("blocked trials: bits must be a multiple of 32 in [64,1024]");
    if(s.count&&!s.n)throw std::invalid_argument("blocked trials: n must be positive");if(s.n>=std::size_t(UINT32_MAX))throw std::invalid_argument("blocked trials: pivot status exceeds 32-bit indexing");
    auto n2=checked_size(s.n,s.n);checked_size(s.count,n2);checked_size(s.count,s.n);
    if(n2>=(std::size_t(1)<<31))throw std::invalid_argument("blocked trials: matrix exceeds 32-bit indexing");
    if(!std::isfinite(o.host_macs)||o.host_macs<0)throw std::invalid_argument("blocked trials: host_macs must be finite and nonnegative");
    if((L.storage&&(L.storage==A.storage||L.storage==D.storage||L.storage==mu.storage||L.storage==status.storage))||
       (status.storage&&(status.storage==A.storage||status.storage==D.storage||status.storage==mu.storage)))throw std::invalid_argument("blocked trials: output aliases input");
    H::require_device(la,*engine_);
    // Private A/B switch for this unreleased scheduling experiment; no public
    // request layout, numerical sequence or production default is changed.
    const char* mode=std::getenv("WOLFNUM_EXACT_TRIAL_SCHEDULE");
    const std::string schedule=mode?mode:"serialized";
    if(schedule!="serialized"&&schedule!="lookahead"&&schedule!="grouped")throw std::invalid_argument("blocked trials: invalid experimental schedule");
    return width(bits,*engine_,la,s,A,D,mu,L,status,o,[&](CommandBatch& b,std::size_t first,std::size_t second){return workspace(b,bits,false,first,second);},schedule!="serialized",schedule=="grouped");
}
}
