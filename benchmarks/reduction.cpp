#include "benchmark_support.hpp"
template<int Bits,bool IsComplex>void compare(Engine& cooperative,Engine& global,std::size_t count){
    using T=std::conditional_t<IsComplex,Complex<Bits/32>,Float<Bits>>;
    std::mt19937_64 rng(20261007+Bits);std::vector<T> input(count);
    for(auto& x:input){if constexpr(IsComplex)x={reference::random_number<Bits>(rng,5),reference::random_number<Bits>(rng,5)};else x=reference::random_number<Bits>(rng,5);}
    auto expected=reference::tree_sum<Bits>(input);
    Buffer<T> x[]={global.make_buffer<T>(count),cooperative.make_buffer<T>(count)},out[]={global.make_buffer<T>(1),cooperative.make_buffer<T>(1)};
    Engine* engines[]={&global,&cooperative};std::vector<double> gpu[2],wall[2];
    for(auto& buffer:x)buffer.upload(input.data(),count);
    for(int repeat=-2;repeat<9;++repeat)for(int k=0;k<2;++k){int method=(repeat+2+k)%2;
        auto b=engines[method]->batch();b.tree_sum(x[method],out[method]);auto t=b.submit().wait();T actual;out[method].download(&actual,1);
        bool equal;if constexpr(IsComplex)equal=reference::equal_complex<Bits>(actual,expected);else equal=reference::equal<Bits>(actual,expected);
        if(!equal)throw std::runtime_error("interleaved reduction MPFR mismatch");
        if(repeat>=0){gpu[method].push_back(t.gpu_seconds);wall[method].push_back(t.wall_seconds);}
    }
    for(int method=0;method<2;++method)std::cout<<Bits<<','<<count<<','<<(IsComplex?"complex":"real")<<','<<(method?"cooperative":"global")<<",9,"<<quantile(gpu[method],.5)<<','<<quantile(wall[method],.5)<<std::endl;
}
template<int Bits,bool IsComplex>void cases(Engine& e,Workers& workers,std::size_t count){
    using T=std::conditional_t<IsComplex,Complex<Bits/32>,Float<Bits>>;
    std::mt19937_64 rng(20261007+Bits);std::vector<T> input(count);
    MPArray ar(count,Bits),ai(count,Bits),sr0(count/2+count%2,Bits),sr1(count/2+count%2,Bits),si0(count/2+count%2,Bits),si1(count/2+count%2,Bits);
    for(std::size_t i=0;i<count;++i){
        if constexpr(IsComplex){input[i]={reference::random_number<Bits>(rng,5),reference::random_number<Bits>(rng,5)};to_mpfr<Bits>(ar[i],input[i].re);to_mpfr<Bits>(ai[i],input[i].im);}
        else {input[i]=reference::random_number<Bits>(rng,5);to_mpfr<Bits>(ar[i],input[i]);}
    }
    T expected=reference::tree_sum<Bits>(input);
    auto equal=[&](const T& x){if constexpr(IsComplex)return reference::equal_complex<Bits>(x,expected);else return reference::equal<Bits>(x,expected);};
    auto cpu=[&](bool parallel){
        MPArray* source_r=&ar,*source_i=&ai,*dest_r=nullptr,*dest_i=nullptr;std::size_t n=count;unsigned level=0;
        while(n>1){std::size_t next=n/2+n%2;dest_r=level%2?&sr1:&sr0;dest_i=level%2?&si1:&si0;
            auto pairs=[&](std::size_t lo,std::size_t hi){for(auto i=lo;i<hi;++i){std::size_t j=2*i;
                if(j+1<n){mpfr_add((*dest_r)[i],(*source_r)[j],(*source_r)[j+1],MPFR_RNDN);if constexpr(IsComplex)mpfr_add((*dest_i)[i],(*source_i)[j],(*source_i)[j+1],MPFR_RNDN);}
                else {mpfr_set((*dest_r)[i],(*source_r)[j],MPFR_RNDN);if constexpr(IsComplex)mpfr_set((*dest_i)[i],(*source_i)[j],MPFR_RNDN);}
            }};
            if(parallel&&next>=workers.size()*16)workers.run(next,pairs);else pairs(0,next);
            source_r=dest_r;source_i=dest_i;n=next;++level;
        }
        return std::make_pair(source_r,source_i);
    };
    auto x=e.make_buffer<T>(count),out=e.make_buffer<T>(1);x.upload(input.data(),count);
    std::vector<double> times[4],device[2];unsigned levels=0;for(auto n=count;n>1;n=n/2+n%2)++levels;
    for(int repeat=-2;repeat<9;++repeat)for(int k=0;k<4;++k){int method=(repeat+2+k)%4;auto start=Clock::now();double gpu=0;T result;std::pair<MPArray*,MPArray*> cpu_result;
        if(method<2)cpu_result=cpu(method==1);
        else {if(method==3)x.upload(input.data(),count);auto b=e.batch();b.tree_sum(x,out);auto t=b.submit().wait();gpu=t.gpu_seconds;
            if(method==3)out.download(&result,1);}
        double wall=std::chrono::duration<double>(Clock::now()-start).count();
        if(method<2){if constexpr(IsComplex)result={from_mpfr<Bits>((*cpu_result.first)[0]),from_mpfr<Bits>((*cpu_result.second)[0])};else result=from_mpfr<Bits>((*cpu_result.first)[0]);}
        if(method==2)out.download(&result,1);
        if(!equal(result))throw std::runtime_error("reduction benchmark MPFR mismatch bits="+std::to_string(Bits));
        if(repeat>=0){times[method].push_back(wall);if(method>=2)device[method-2].push_back(gpu);}
    }
    for(int method=0;method<2;++method){Samples samples{times[0],times[1],device[method],times[method+2]};
        std::string label=IsComplex?"complex_tree_sum":"tree_sum";if(!method)label+="_resident";
        report(Bits,label.c_str(),count,levels,workers,samples);}
}
int main(int argc,char** argv){try{bool cooperative=true,comparison=false;if(argc==2&&std::string(argv[1])=="--global-tree")cooperative=false;else if(argc==2&&std::string(argv[1])=="--compare")comparison=true;else if(argc!=1)throw std::invalid_argument("usage: reduction_limbforge [--global-tree|--compare]");Engine e({0,cooperative});
    if(comparison){Engine global({0,false});std::cerr<<e.device_name()<<"; interleaved cooperative/global resident reductions; every dispatch checked against MPFR\n";
        std::cout<<std::setprecision(10)<<"bits,count,format,method,samples,gpu_median_s,wall_median_s\n";
        for(auto n:{257u,4096u,65537u}){compare<256,false>(e,global,n);compare<256,true>(e,global,n);compare<384,false>(e,global,n);compare<384,true>(e,global,n);compare<1024,false>(e,global,n);compare<1024,true>(e,global,n);}return 0;}
    Workers workers(std::max(1u,std::thread::hardware_concurrency()));
    std::cerr<<e.device_name()<<"; cooperative="<<cooperative<<"; "<<workers.size()<<" persistent MPFR workers; fixed pair tree; serial tail below workers*16; 2 warmups, 9 samples, four methods interleaved, every sample checked\n";
    std::cout<<std::setprecision(10)<<"bits,operation,count,steps,samples,cpu_workers,cpu_serial_s,cpu_parallel_s,gpu_s,wall_median_s,wall_min_s,wall_p90_s\n";
    for(auto n:{257u,4096u,65537u}){cases<256,false>(e,workers,n);cases<256,true>(e,workers,n);cases<384,false>(e,workers,n);cases<384,true>(e,workers,n);cases<1024,false>(e,workers,n);cases<1024,true>(e,workers,n);}return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
