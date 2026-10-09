#include "section9_cases.hpp"
#include <type_traits>
template<int B> C<B> mac(C<B> a,C<B> b,C<B> c,bool fused){return fused?reference::complex_fused<B>(a,b,c):ca<B>(c,cm<B>(a,b));}
template<int B,class T> T zvalue(){if constexpr(std::is_same_v<T,F<B>>)return zero<B/32>();else return {zero<B/32>(),zero<B/32>()};}
template<int B,class T> T randomvalue(std::mt19937_64& rng,int span=8){if constexpr(std::is_same_v<T,F<B>>)return reference::random_number<B>(rng,span);else return {reference::random_number<B>(rng,span),reference::random_number<B>(rng,span)};}
template<int B,class T> T errorvalue(){if constexpr(std::is_same_v<T,F<B>>)return zero<B/32>(invalid);else return {zero<B/32>(),zero<B/32>(invalid)};}

template<int B> void vector_cpu_sources(Engine& e,bool dense=true){
    const int lanes=dense?513:5,groups=(lanes+1)/2,maxsteps=3;std::mt19937_64 rng(B+31);
    for(bool shared:{false,true})for(unsigned terms:{0u,3u}){
        PolynomialRecurrence s;s.lanes=lanes;s.lanes_per_weight=2;s.coefficient_sets=shared?1:groups;s.terms=terms;
        std::vector<C<B>> start(4*lanes),cp(s.coefficient_sets*4*terms),cq(cp.size()),Y(maxsteps*groups),Ep(maxsteps*4*groups),Eq(Ep.size());
        for(auto* v:{&start,&cp,&cq,&Y,&Ep,&Eq})for(auto& z:*v)z=randomvalue<B,C<B>>(rng);
        if(terms&&!shared)cp[terms-1]=errorvalue<B,C<B>>();
        for(unsigned steps:{0u,3u})for(bool fused:{false,true})for(bool reverse:{false,true})for(bool all:{false,true}){
            s.steps=steps;s.fused=fused;s.reverse=reverse;s.all_steps=all;std::vector<C<B>> want((all?steps+1:1)*4*lanes,zvalue<B,C<B>>());
            if(all)std::copy(start.begin(),start.end(),want.begin());
            auto horner=[&](const auto& c,int set,int a,C<B> yy){if(!terms)return zvalue<B,C<B>>();auto z=c[(set*4+a)*terms+terms-1];for(unsigned n=terms-1;n-->0;)z=mac<B>(z,yy,c[(set*4+a)*terms+n],fused);return z;};
            for(int lane=0;lane<lanes;++lane){int g=lane/2,set=shared?0:g;C<B> v[4];for(int a=0;a<4;++a)v[a]=start[a*lanes+lane];
                for(unsigned step=0;step<steps;++step){unsigned k=reverse?steps-1-step:step;auto dot=zvalue<B,C<B>>();
                    for(int a=0;a<4;++a)dot=mac<B>(cm<B>(Eq[(k*4+a)*groups+g],horner(cq,set,a,Y[k*groups+g])),v[a],dot,fused);
                    for(int a=0;a<4;++a){v[a]=mac<B>(cm<B>(Ep[(k*4+a)*groups+g],horner(cp,set,a,Y[k*groups+g])),dot,v[a],fused);if(all)want[((step+1)*4+a)*lanes+lane]=v[a];}}
                if(!all)for(int a=0;a<4;++a)want[a*lanes+lane]=v[a];}
            const auto label=std::string("baseline vector shared=")+std::to_string(shared)+" terms="+std::to_string(terms)+" steps="+std::to_string(steps)+" fused="+std::to_string(fused)+" reverse="+std::to_string(reverse)+" all_steps="+std::to_string(all);
                std::vector<C<B>> wp(steps*4*groups),wq(wp.size());
                for(unsigned k=0;k<steps;++k)for(int a=0;a<4;++a)for(int g=0;g<groups;++g){int set=shared?0:g;auto idx=(k*4+a)*groups+g;
                    wp[idx]=cm<B>(Ep[idx],horner(cp,set,a,Y[k*groups+g]));wq[idx]=cm<B>(Eq[idx],horner(cq,set,a,Y[k*groups+g]));}
                const int padded=groups*2;std::vector<C<B>> vs(4*padded,zvalue<B,C<B>>());
                for(int a=0;a<4;++a)std::copy_n(start.data()+a*lanes,lanes,vs.data()+a*padded);
                auto vstart=put(e,vs),vp=put(e,wp),vq=put(e,wq),vo=e.make_buffer<C<B>>((all?steps+1:1)*4*padded);
                VectorRecurrence shape;shape.lanes=padded;shape.steps=steps;shape.lanes_per_weight=2;shape.fused=fused;shape.reverse=reverse;shape.all_steps=all;
                {auto vector=e.batch();vector.vector_recurrence(shape,vstart,vp,vq,Buffer<C<B>>(),vo);vector.submit().wait();}
                auto raw=get(vo);std::vector<C<B>> trimmed(want.size());
                for(unsigned k=0;k<(all?steps+1:1);++k)for(int a=0;a<4;++a)std::copy_n(raw.data()+(k*4+a)*padded,lanes,trimmed.data()+(k*4+a)*lanes);
                check<B>(trimmed,want,label.c_str());
        }
    }
}
int main(){try{Engine e;vector_cpu_sources<128>(e);std::cout<<"Baseline 128-bit vector CPU-source references passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
