#pragma once
#include "limbforge/batched_linalg.hpp"
#include "limbforge/mpc_staging.hpp"
#include "reference.hpp"
#include "limbforge/dispatch_policy.hpp"
#include <iostream>
#include <random>
#include <cstdlib>
#include <unistd.h>
using namespace limbforge;
void require(bool b,const char* message){if(!b)throw std::runtime_error(message);}
template<class T> Buffer<T> put(Engine& e,const std::vector<T>& x){auto b=e.make_buffer<T>(x.size());b.upload(x.data(),x.size());return b;}
template<class T> std::vector<T> get(const Buffer<T>& b){std::vector<T> x(b.size());b.download(x.data(),x.size());return x;}
template<int B> using F=Float<B>;
template<int B> using C=Complex<B/32>;
template<int B> F<B> rm(F<B> a,F<B> b){return reference::real<B>(Operation::mul,a,b);}
template<int B> F<B> ra(F<B> a,F<B> b){return reference::real<B>(Operation::add,a,b);}
template<int B> C<B> cm(C<B> a,C<B> b){return reference::complex<B>(Operation::complex_mul,a,b);}
template<int B> C<B> ca(C<B> a,C<B> b){return reference::complex<B>(Operation::complex_add,a,b);}
template<int B> bool same(F<B> a,F<B> b){return reference::equal<B>(a,b);}
template<int B> bool same(C<B> a,C<B> b){return reference::equal_complex<B>(a,b);}
template<int B> void dump(F<B> v){std::cerr<<"sign="<<v.sign<<" exponent="<<v.exponent<<" status="<<v.status<<" limbs=";for(int i=B/32;i-->0;)std::cerr<<std::hex<<v.limb[i]<<' ';std::cerr<<std::dec;}
template<int B> void dump(C<B> v){std::cerr<<"re{";dump<B>(v.re);std::cerr<<"} im{";dump<B>(v.im);std::cerr<<'}';}
template<int B,class T> void check(const std::vector<T>& a,const std::vector<T>& b,const char* label){require(a.size()==b.size(),"size");for(std::size_t i=0;i<a.size();++i)if(!same<B>(a[i],b[i])){std::cerr<<label<<" index "<<i<<"\nGPU: ";dump<B>(a[i]);std::cerr<<"\nMPFR: ";dump<B>(b[i]);std::cerr<<'\n';throw std::runtime_error(std::string(label)+" index "+std::to_string(i));}}
template<int B> void products(Engine& e,BatchedLinalg& la){
    std::mt19937_64 rng(B);auto random=[&]{return C<B>{reference::random_number<B>(rng,4),reference::random_number<B>(rng,4)};};
    StridedGemm s{2,9,5,17,9*17,0,9*5+3};std::vector<C<B>> A(2*s.stride_a),M(17*5),initial(2*s.stride_c),expected=initial;
    for(auto& z:A)z=random();for(auto& z:M)z=random();for(auto& z:initial)z=random();expected=initial;
    auto ab=put(e,A),mb=put(e,M),out=put(e,initial);s.accumulate=true;s.negative=true;
    for(std::size_t b=0;b<s.count;++b)for(std::size_t i=0;i<s.m;++i)for(std::size_t j=0;j<s.n;++j){auto z=expected[b*s.stride_c+i*s.n+j];
        for(std::size_t k=0;k<s.k;++k)z=ca<B>(z,cm<B>(A[b*s.stride_a+i*s.k+k],M[k*s.n+j]));expected[b*s.stride_c+i*s.n+j]={negate(z.re),negate(z.im)};}
    auto batch=e.batch();la.gemm(batch,s,ab,mb,out);batch.submit().wait();check<B>(get(out),expected,"complex strided GEMM");
    auto host=initial;la.gemm(B,true,s,A.data(),M.data(),host.data());check<B>(host,expected,"host GEMM");
    // Triple 4x4 product with a shared right factor, negated (inverse derivative use case).
    StridedGemm four{2,4,4,4,16,16,16};std::vector<C<B>> a(32),b(32),d(32),want(32),tmp(32);for(auto* v:{&a,&b,&d})for(auto& z:*v)z=random();
    for(int t=0;t<2;++t)for(int i=0;i<4;++i)for(int j=0;j<4;++j){auto z=C<B>{zero<B/32>(),zero<B/32>()};for(int k=0;k<4;++k)z=ca<B>(z,cm<B>(a[t*16+i*4+k],b[t*16+k*4+j]));tmp[t*16+i*4+j]=z;}
    for(int t=0;t<2;++t)for(int i=0;i<4;++i)for(int j=0;j<4;++j){auto z=C<B>{zero<B/32>(),zero<B/32>()};for(int k=0;k<4;++k)z=ca<B>(z,cm<B>(tmp[t*16+i*4+k],d[t*16+k*4+j]));want[t*16+i*4+j]={negate(z.re),negate(z.im)};}
    auto aa=put(e,a),bb=put(e,b),dd=put(e,d),cc=e.make_buffer<C<B>>(32);auto triple=e.batch();la.product3(triple,four,four,aa,bb,dd,cc,true);triple.submit().wait();check<B>(get(cc),want,"triple product");
    PowerMoments p{2,7,3,4,-2};std::vector<C<B>> E(14),Y(14),W(56),result(32),ref(32);for(auto* v:{&E,&Y,&W})for(auto& z:*v)z=random();
    for(int t=0;t<2;++t)for(int n=0;n<4;++n)for(int j=0;j<4;++j){auto z=C<B>{zero<B/32>(),zero<B/32>()};for(int k=0;k<7;++k){
        auto one=C<B>{from_decimal<B>("1"),zero<B/32>()};auto pw=reference::complex<B>(Operation::complex_div,one,cm<B>(Y[t*7+k],Y[t*7+k]));
        for(int r=0;r<n;++r)pw=cm<B>(pw,Y[t*7+k]);z=ca<B>(z,cm<B>(cm<B>(E[t*7+k],pw),W[(t*7+k)*4+j]));}ref[(t*4+n)*4+j]=z;}
    auto eb=put(e,E),yb=put(e,Y),wb=put(e,W),rb=e.make_buffer<C<B>>(32);auto powers=e.batch();la.power_moments(powers,p,eb,yb,wb,rb);powers.submit().wait();check<B>(get(rb),ref,"power moments");
    auto compact=e.batch();la.power_moments(compact,p,eb,yb,wb,rb,PowerStorage::compact);compact.submit().wait();check<B>(get(rb),ref,"compact resident power moments");
    la.power_moments(B,true,p,E.data(),Y.data(),W.data(),result.data(),PowerStorage::compact);check<B>(result,ref,"compact host power moments");
    PowerMoments empty_power{2,0,12,4,0,true,true};std::vector<C<B>> empty_result(2*13*4,random()),empty_expected=empty_result;
    la.power_moments(B,true,empty_power,nullptr,nullptr,nullptr,empty_result.data(),PowerStorage::compact);check<B>(empty_result,empty_expected,"compact empty reduction accumulation");
    empty_power.accumulate=false;for(auto& z:empty_expected)z={zero<B/32>(),zero<B/32>()};
    la.power_moments(B,true,empty_power,nullptr,nullptr,nullptr,empty_result.data(),PowerStorage::compact);check<B>(empty_result,empty_expected,"compact empty reduction overwrite");
    bool invalid_mode=false;try{auto bad=e.batch();la.power_moments(bad,p,eb,yb,wb,rb,static_cast<PowerStorage>(99));}catch(const std::invalid_argument&){invalid_mode=true;}require(invalid_mode,"invalid resident power storage accepted");
    invalid_mode=false;try{la.power_moments(B,true,p,E.data(),Y.data(),W.data(),result.data(),static_cast<PowerStorage>(99));}catch(const std::invalid_argument&){invalid_mode=true;}require(invalid_mode,"invalid host power storage accepted");
    bool rejected=false;try{auto bad=e.batch();s.stride_c=1;la.gemm(bad,s,ab,mb,out);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"overlapping outputs accepted");
    std::cout<<B<<" bits: complex GEMM tile tails/broadcast/strides, triple product, negative power moments\n";
}
template<int B> void normal(Engine& e,BatchedLinalg& la){
    std::mt19937_64 rng(B+1);constexpr int m=9,n=4,t=3;std::vector<F<B>> J(m*n),g(m),A(n*n),rhs(n),D(n,from_decimal<B>("1")),mu{from_decimal<B>("0.25"),from_decimal<B>("1"),from_decimal<B>("-10000")};
    for(auto& x:J)x=reference::random_number<B>(rng,2);for(auto& x:g)x=reference::random_number<B>(rng,2);
    for(int i=0;i<n;++i){for(int j=0;j<n;++j){auto s=zero<B/32>();for(int k=0;k<m;++k)s=ra<B>(s,rm<B>(J[k*n+i],J[k*n+j]));A[i*n+j]=s;}
        auto s=zero<B/32>();for(int k=0;k<m;++k)s=ra<B>(s,rm<B>(J[k*n+i],g[k]));rhs[i]=s;}
    auto jb=put(e,J),gb=put(e,g),ab=e.make_buffer<F<B>>(n*n),rb=e.make_buffer<F<B>>(n),db=put(e,D),mb=put(e,mu),lb=e.make_buffer<F<B>>(t*n*n),xb=e.make_buffer<F<B>>(t*n);auto st=e.make_buffer<std::uint32_t>(t);
    auto batch=e.batch();la.normal_equations(batch,jb,gb,m,n,ab,rb);la.cholesky_trials(batch,{n,t},ab,db,mb,lb,st);la.cholesky_solve(batch,lb,st,t,n,rb,1,xb);
    std::vector<Submission> jobs;jobs.push_back(batch.submit());auto done=wait_all_async(std::move(jobs));done.get();check<B>(get(ab),A,"normal matrix");check<B>(get(rb),rhs,"normal rhs");
    auto l=get(lb),x=get(xb);auto status=get(st);require(status[0]==0&&status[1]==0&&status[2]==1,"damping trial statuses");
    for(int trial=0;trial<2;++trial){auto R=A;for(int i=0;i<n;++i){R[i*n+i]=ra<B>(R[i*n+i],rm<B>(mu[trial],D[i]));for(int j=i+1;j<n;++j)R[i*n+j]=zero<B/32>();}
        for(int k=0;k<n;++k){R[k*n+k]=reference::real<B>(Operation::sqrt,R[k*n+k],zero<B/32>());for(int i=k+1;i<n;++i)R[i*n+k]=reference::real<B>(Operation::div,R[i*n+k],R[k*n+k]);
            for(int i=k+1;i<n;++i)for(int j=k+1;j<=i;++j)R[i*n+j]=reference::real<B>(Operation::sub,R[i*n+j],rm<B>(R[i*n+k],R[j*n+k]));}
        for(int q=0;q<n*n;++q)require(same<B>(l[trial*n*n+q],R[q]),"resident Cholesky sequence");auto y=rhs;
        for(int i=0;i<n;++i){for(int j=0;j<i;++j)y[i]=reference::real<B>(Operation::sub,y[i],rm<B>(R[i*n+j],y[j]));y[i]=reference::real<B>(Operation::div,y[i],R[i*n+i]);}
        for(int i=n;i-->0;){for(int j=i+1;j<n;++j)y[i]=reference::real<B>(Operation::sub,y[i],rm<B>(R[j*n+i],y[j]));y[i]=reference::real<B>(Operation::div,y[i],R[i*n+i]);}
        for(int i=0;i<n;++i)require(same<B>(x[trial*n+i],y[i]),"resident solves");}
    for(int i=2*n;i<3*n;++i)require(x[i].status==invalid,"failed trial solve");
    // Exact augmented SYRK, including a forced host-fallback repair at wait.
    LinalgOptions opt;opt.force_fallback=true;Linalg exact(e,opt);auto exact_batch=e.batch();auto ticket=la.normal_equations_exact(exact,exact_batch,jb,gb,m,n,ab,rb);bool blocked=false;try{la.cholesky_trials(exact_batch,{n,t},ab,db,mb,lb,st);}catch(const std::logic_error&){blocked=true;}require(blocked,"provisional input accepted");exact_batch.submit().wait();require(ticket.resolved(),"exact normal ticket");
    auto ea=get(ab),er=get(rb);for(int i=0;i<n;++i){require(same<B>(er[i],reference::dot<B>(J.data()+i,n,g.data(),1,m)),"exact normal rhs");for(int j=0;j<n;++j)require(same<B>(ea[i*n+j],reference::dot<B>(J.data()+i,n,J.data()+j,n,m)),"exact normal matrix");}
    Linalg native(e);auto no_fallback=e.batch();auto native_ticket=la.normal_equations_exact(native,no_fallback,jb,gb,m,n,ab,rb);no_fallback.submit().wait();require(native_ticket.report().fallback_outputs==0,"expected native exact-dot path");check<B>(get(ab),ea,"GPU exact normal matrix");check<B>(get(rb),er,"GPU exact normal rhs");
    std::cout<<B<<" bits: resident normal equations, three damping trials, solves, async wait, exact fallback repair\n";
}
template<int B> void bridge(Engine& e,BatchedLinalg& la){
    reference::MP source(B),back(B);mpfr_set_str(source.x,"1.234567891",10,MPFR_RNDN);F<B> f;from_mpfr(B,source.x,&f);require(same<B>(f,from_mpfr<B>(source.x)),"runtime bridge");to_mpfr(B,back.x,&f);require(mpfr_equal_p(source.x,back.x),"runtime bridge back");std::vector<unsigned char> bytes_unaligned(bridge_element_bytes(B)+1);from_mpfr(B,source.x,bytes_unaligned.data()+1);to_mpfr(B,back.x,bytes_unaligned.data()+1);require(mpfr_equal_p(source.x,back.x),"unaligned runtime bridge");
    mpfr_t array[2];for(auto& x:array)mpfr_init2(x,B);mpfr_set(array[0],source.x,MPFR_RNDN);mpfr_neg(array[1],source.x,MPFR_RNDN);std::vector<unsigned char> packed(2*bridge_element_bytes(B)+1);
    from_mpfr_array(B,array,packed.data()+1,2,1);mpfr_set_zero(array[0],1);to_mpfr_array(B,array,packed.data()+1,2,1);require(mpfr_equal_p(array[0],source.x),"runtime array bridge");for(auto& x:array)mpfr_clear(x);
#ifdef LIMBFORGE_HAS_MPC
    std::size_t bytes=std::size_t(getpagesize());void* storage=nullptr;if(posix_memalign(&storage,bytes,bytes))throw std::bad_alloc();std::unique_ptr<void,decltype(&std::free)> owned(storage,&std::free);
    std::memset(storage,0,bytes);mpc_t v[2];mpc_srcptr pointers[2]={v[0],v[1]};
    for(int i=0;i<2;++i){mpfr_custom_init_set(mpc_realref(v[i]),MPFR_ZERO_KIND,0,B,static_cast<char*>(storage)+i*256);
        mpfr_custom_init_set(mpc_imagref(v[i]),MPFR_ZERO_KIND,0,B,static_cast<char*>(storage)+i*256+128);
        mpfr_set_si(mpc_realref(v[i]),i?0:7,MPFR_RNDN);mpfr_set_si(mpc_imagref(v[i]),-3-i,MPFR_RNDN);}
    auto records=describe_inline_mpc(B,storage,bytes,pointers,2);auto out=e.make_buffer<C<B>>(2);auto batch=e.batch();la.import_inline_complex(batch,storage,bytes,records.data(),2,out);batch.submit().wait();auto got=get(out);
    for(int i=0;i<2;++i){C<B> runtime;from_mpc(B,v[i],&runtime);require(same<B>(got[i],runtime),"zero-copy MPC import");}
#endif
    std::cout<<B<<" bits: runtime bridge and inline MPC staging\n";
}
template<int B> void polynomial(Engine& e,BatchedLinalg& la){
    PolynomialRecurrence s;s.lanes=5;s.steps=3;s.terms=3;s.lanes_per_weight=2;s.coefficient_sets=3;s.all_steps=true;s.reverse=true;
    constexpr int groups=3;std::mt19937_64 rng(B+7);
    auto random=[&]{return C<B>{reference::random_number<B>(rng,2),reference::random_number<B>(rng,2)};};
    std::vector<C<B>> start(20),cp(36),cq(36),Y(9),Ep(36),Eq(36),want(80);
    for(auto* v:{&start,&cp,&cq,&Y,&Ep,&Eq})for(auto& z:*v)z=random();
    auto st=put(e,start),p=put(e,cp),q=put(e,cq),y=put(e,Y),ep=put(e,Ep),eq=put(e,Eq),out=e.make_buffer<C<B>>(want.size());
    for(bool fused:{false,true}){
        s.fused=fused;auto mac=[&](C<B> a,C<B> b,C<B> c){return fused?reference::complex_fused<B>(a,b,c):ca<B>(c,cm<B>(a,b));};
        auto horner=[&](const std::vector<C<B>>& c,int set,int a,C<B> yy){auto z=c[(set*4+a)*3+2];for(int n=2;n-->0;)z=mac(z,yy,c[(set*4+a)*3+n]);return z;};
        std::copy(start.begin(),start.end(),want.begin());
        for(int lane=0;lane<5;++lane){int g=lane/2;C<B> v[4];for(int a=0;a<4;++a)v[a]=start[a*5+lane];
            for(int step=0;step<3;++step){int k=2-step;auto dot=C<B>{zero<B/32>(),zero<B/32>()};
                for(int a=0;a<4;++a)dot=mac(cm<B>(Eq[(k*4+a)*groups+g],horner(cq,g,a,Y[k*groups+g])),v[a],dot);
                for(int a=0;a<4;++a){v[a]=mac(cm<B>(Ep[(k*4+a)*groups+g],horner(cp,g,a,Y[k*groups+g])),dot,v[a]);want[((step+1)*4+a)*5+lane]=v[a];}}
        }
        auto b=e.batch();la.polynomial_recurrence(b,s,st,p,q,y,ep,eq,out);b.submit().wait();check<B>(get(out),want,"polynomial recurrence");
    }
    std::cout<<B<<" bits: on-device Horner recurrence, reverse/all_steps, sharing tails, composed/fused\n";
}
template<int B> void real_products(Engine& e,BatchedLinalg& la){
    StridedGemm s{3,4,4,4,16,0,16};std::vector<F<B>> A(48),M(16),want(48);
    for(int i=0;i<48;++i)A[i]=from_decimal<B>(std::to_string(i-17));for(int i=0;i<16;++i)M[i]=from_decimal<B>(std::to_string(i-8));
    for(int b=0;b<3;++b)for(int i=0;i<4;++i)for(int j=0;j<4;++j){auto z=zero<B/32>();for(int k=0;k<4;++k)z=ra<B>(z,rm<B>(A[b*16+i*4+k],M[k*4+j]));want[b*16+i*4+j]=z;}
    auto a=put(e,A),m=put(e,M),out=e.make_buffer<F<B>>(48);auto batch=e.batch();la.gemm(batch,s,a,m,out);batch.submit().wait();check<B>(get(out),want,"real odd-count 4x4 GEMM");
    s.k=0;Buffer<F<B>> unused;auto empty=e.batch();la.gemm(empty,s,unused,unused,out);empty.submit().wait();check<B>(get(out),std::vector<F<B>>(48,zero<B/32>()),"empty dot");
    std::cout<<B<<" bits: real 4x4 GEMM odd batch/broadcast and empty dots\n";
}
void policy(){BreakEvenTable t;DispatchKey k{"gemm_complex",352,2,4,4,4,18,false};require(t.recommend(k)==BackendRecommendation::unknown,"unmeasured policy");t.record(k,{2,1});require(t.recommend(k)==BackendRecommendation::gpu,"policy lookup");k.bits=384;require(t.recommend(k)==BackendRecommendation::unknown,"policy extrapolation");}

// Lifetime checks are separate from arithmetic sweeps: gram repair must keep per-call storage.
void workspace_lifetimes(){
    constexpr int bits=352;using T=F<bits>;Engine e;BatchedLinalg unit(e);
    auto one=from_decimal<bits>("1"),two=from_decimal<bits>("2"),three=from_decimal<bits>("3");
    auto a=put(e,std::vector<T>{one,zero<bits/32>(),zero<bits/32>(),two});
    auto id=put(e,std::vector<T>{one,zero<bits/32>(),zero<bits/32>(),one});auto out=e.make_buffer<T>(4);
    StridedGemm shape{1,2,2,2,4,4,4};
    auto first=e.batch();unit.product3(first,shape,shape,a,id,id,out);
    auto busy=unit.workspaces();require(busy.busy_workspaces==1&&busy.busy_bytes>=4*sizeof(T),"workspace not held by unsubmitted batch");
    auto detached=unit.release_workspaces();require(detached.busy_workspaces==1&&unit.workspaces().busy_bytes==busy.busy_bytes,"release freed busy scratch");
    auto submitted=first.submit();require(unit.workspaces().busy_workspaces==1,"submission released scratch before wait");submitted.wait();
    check<bits>(get(out),get(a),"released product workspace");require(unit.workspaces().busy_workspaces==0&&unit.workspaces().idle_bytes==0,"detached workspace retained after wait");
    for(int rep=0;rep<3;++rep){auto b=e.batch();unit.product3(b,shape,shape,a,id,id,out);b.submit().wait();}
    auto warmed=unit.workspaces();require(warmed.idle_bytes==4*sizeof(T)&&warmed.busy_workspaces==0,"product workspace capacity grew on reuse");
    {auto abandoned=e.batch();unit.product3(abandoned,shape,shape,a,id,id,out);unit.release_workspaces();require(unit.workspaces().busy_workspaces==1,"abandoned batch lost scratch");}
    require(unit.workspaces().busy_bytes==0,"unsubmitted destruction retained scratch");
    StridedGemm empty=shape;empty.count=0;Buffer<T> unused;
    auto empty_batch=e.batch();auto empty_done=empty_batch.submit();empty_done.wait();
    unit.product3(empty_batch,empty,empty,unused,unused,unused,unused);
    require(unit.workspaces().idle_bytes==0&&unit.workspaces().busy_bytes==0,"empty product allocated workspace");
    Engine foreign;bool rejected=false;try{auto b=foreign.batch();unit.product3(b,shape,shape,a,id,id,out);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"foreign Engine accepted by workspace allocation");unit.release_workspaces();
    LinalgOptions options;options.force_fallback=true;Linalg exact(e,options);
    auto j1=put(e,std::vector<T>{one,two}),g1=put(e,std::vector<T>{three,one});
    auto j2=put(e,std::vector<T>{two,three}),g2=put(e,std::vector<T>{one,two});
    auto A1=e.make_buffer<T>(1),A2=e.make_buffer<T>(1),r1=e.make_buffer<T>(1),r2=e.make_buffer<T>(1);
    auto together=e.batch();auto t1=unit.normal_equations_exact(exact,together,j1,g1,2,1,A1,r1);
    auto t2=unit.normal_equations_exact(exact,together,j2,g2,2,1,A2,r2);
    require(unit.workspaces().busy_workspaces==2,"exact normal calls reused a live gram");unit.release_workspaces();together.submit().wait();
    require(t1.resolved()&&t2.resolved(),"exact normal tickets unresolved");
    require(same<bits>(get(A1)[0],from_decimal<bits>("5"))&&same<bits>(get(r1)[0],from_decimal<bits>("5")),"first exact repair used overwritten gram");
    require(same<bits>(get(A2)[0],from_decimal<bits>("13"))&&same<bits>(get(r2)[0],from_decimal<bits>("8")),"second exact repair");
    require(unit.workspaces().busy_bytes==0,"released exact scratch retained");
    // A destroyed algebra unit must not invalidate completion callbacks.
    auto survives=e.batch();{BatchedLinalg temporary(e);temporary.normal_equations_exact(exact,survives,j1,g1,2,1,A1,r1);}survives.submit().wait();
    require(same<bits>(get(A1)[0],from_decimal<bits>("5")),"unit destruction invalidated scratch");
    unit.release_workspaces();std::cout<<"Batched workspace reuse, detachment, ownership and exact repair passed\n";
}
