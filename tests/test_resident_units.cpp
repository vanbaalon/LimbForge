// Resident Numerics (S1/S4) and Transcendentals (D7) passes in an Engine's CommandBatch (docs/execution.md, "Resident
// units"): bitwise equal to the host-array APIs (themselves MPFR/MPC-validated by test_numerics / test_transcendental) at
// 64/224/256/384/1024 bits, chains with engine operations in one batch, ownership errors, provisional outputs and lifetimes.
// Usage: test_limbforge_resident_units [--quick]
#include "reference.hpp"
#include "limbforge/numerics.hpp"
#include "limbforge/transcendental.hpp"
#include <cstring>
#include <functional>
#include <stdexcept>
#include <thread>
#include <type_traits>
using namespace limbforge;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
template<class E,class Fn> void rejects(Fn f,const std::string& what){bool ok=false;try{f();}catch(const E&){ok=true;}require(ok,"not rejected: "+what);}
template<class T> bool same(const T& a,const T& b){return std::memcmp(&a,&b,sizeof(T))==0;}
template<class T> void same_all(const std::vector<T>& a,const std::vector<T>& b,const std::string& what){
    require(a.size()==b.size(),what+": size");for(std::size_t i=0;i<a.size();++i)require(same(a[i],b[i]),what+": element "+std::to_string(i));}
template<class T> Buffer<T> put(Engine& e,const std::vector<T>& v){auto b=e.make_buffer<T>(v.size());b.upload(v.data(),v.size());return b;}
template<class T> std::vector<T> get(const Buffer<T>& b){std::vector<T> v(b.size());b.download(v.data(),v.size());return v;}
template<class T> constexpr int bits_of = detail::Format<T>::bits;
template<int Bits> Float<Bits> pow2(int e,int sign=1){auto x=zero<Bits/32>();x.limb[Bits/32-1]=0x80000000u;x.sign=sign;x.exponent=e;return x;}
// Random value (complex: both parts) with exponents in [-span, span]; occasionally a zero or a status entry.
template<class T> T value(std::mt19937_64& g,int span,bool specials=true){
    constexpr int B=bits_of<T>;auto one=[&]{auto x=reference::random_number<B>(g,span);
        if(specials){unsigned r=unsigned(g()%64);if(r==0)x=zero<B/32>();else if(r==1)x=zero<B/32>(invalid);}return x;};
    if constexpr(detail::Format<T>::complex)return T{one(),one()};else return one();}

// ---------------- Numerics ----------------
template<class T> void poly_resident(Engine& e,Numerics& nm,std::mt19937_64& g){
    constexpr int B=bits_of<T>;constexpr bool Cx=detail::Format<T>::complex;
    for(bool fused:{false,true})for(std::size_t terms:{0ul,5ul,33ul}){
        const std::size_t pps=3,P=37,sets=(P+pps-1)/pps;Polynomial shape{P,terms,pps,Cx,fused};
        std::vector<T> c(sets*terms),x(P);for(auto& v:c)v=value<T>(g,8);for(auto& v:x)v=value<T>(g,1);
        auto C=put(e,c),X=put(e,x);Buffer<T> out[3];auto batch=e.batch();
        for(unsigned order=0;order<3;++order){out[order]=e.make_buffer<T>(P*(order+1)+3); // larger than needed
            if(order)nm.poly_eval_jet(batch,shape,order,C,X,out[order]);else nm.poly_eval(batch,shape,C,X,out[order]);}
        batch.submit().wait();
        for(unsigned order=0;order<3;++order){std::vector<T> host(P*(order+1));nm.poly_eval_jet(B,shape,order,c.data(),x.data(),host.data());
            auto r=get(out[order]);r.resize(host.size());same_all(r,host,"poly_eval_jet bits="+std::to_string(B)+(Cx?" complex":" real")+(fused?" fused":"")+" terms="+std::to_string(terms)+" order="+std::to_string(order));}
    }
}
enum class Op { inf, max, two, ratio, status };
// All five reductions of one data set in one batch (and twice more without info / after an engine op), against the host calls.
template<class T> void norms_resident(Engine& e,Numerics& nm,std::size_t S,std::size_t L,std::mt19937_64& g){
    constexpr int B=bits_of<T>;constexpr bool Cx=detail::Format<T>::complex;using R=Float<B>;Segments seg{S,L,Cx};
    std::vector<T> x(S*L);std::vector<R> scale(S*L);for(auto& v:x)v=value<T>(g,40,L<1000);for(auto& v:scale)v=value<R>(g,40,false);
    if(S>1&&L)for(std::size_t i=0;i<L;i+=3)scale[L+i]=zero<B/32>();                              // zero scales in segment 1
    if(S>2&&L>2){T top=value<T>(g,0,false);x[2*L+L/3]=top;x[2*L+L-1]=top;}                      // ties at the maximum
    auto X=put(e,x);auto Sc=put(e,scale);std::string label=" bits="+std::to_string(B)+(Cx?" complex":" real")+" S="+std::to_string(S)+" L="+std::to_string(L);
    std::vector<Buffer<R>> vals;std::vector<Buffer<NormInfo>> infos;for(int i=0;i<5;++i){vals.push_back(e.make_buffer<R>(S));infos.push_back(e.make_buffer<NormInfo>(S));}
    auto plain=e.make_buffer<R>(S);
    auto batch=e.batch();
    nm.norm_inf(batch,seg,X,vals[0],infos[0]);nm.norm_max(batch,seg,X,vals[1],infos[1]);nm.norm2(batch,seg,X,vals[2],infos[2]);
    nm.scaled_residual(batch,seg,X,Sc,vals[3],infos[3]);nm.summarize_status(batch,seg,X,infos[4]);nm.norm2(batch,seg,X,plain);
    batch.submit().wait();
    for(int k=0;k<5;++k){std::vector<R> hv(S);std::vector<NormInfo> hi(S);const char* name[]={"norm_inf","norm_max","norm2","scaled_residual","summarize_status"};
        switch(Op(k)){case Op::inf:nm.norm_inf(B,seg,x.data(),hv.data(),hi.data());break;case Op::max:nm.norm_max(B,seg,x.data(),hv.data(),hi.data());break;
            case Op::two:nm.norm2(B,seg,x.data(),hv.data(),hi.data());break;case Op::ratio:nm.scaled_residual(B,seg,x.data(),scale.data(),hv.data(),hi.data());break;
            default:nm.summarize_status(B,seg,x.data(),hi.data());}
        if(Op(k)!=Op::status)same_all(get(vals[k]),hv,std::string(name[k])+" values"+label);
        same_all(get(infos[k]),hi,std::string(name[k])+" info"+label);
        if(Op(k)==Op::two)same_all(get(plain),hv,"norm2 without info"+label);}
}
template<class T> void numerics_width(Engine& e,Numerics& nm,bool quick){
    std::mt19937_64 g(bits_of<T>*3+detail::Format<T>::complex);poly_resident<T>(e,nm,g);
    for(std::size_t L:{0ul,1ul,33ul,129ul})norms_resident<T>(e,nm,4,L,g);
    if(!quick||bits_of<T> ==256)norms_resident<T>(e,nm,2,20000,g); // three tree passes (157 -> 2 -> 1 block roots)
    std::cout<<bits_of<T><<" bits "<<(detail::Format<T>::complex?"complex":"real")<<": resident poly_eval(_jet) and norms equal the host-array calls"<<std::endl;
}

// ---------------- Transcendentals ----------------
template<int B> std::vector<Float<B>> real_inputs(Function f,std::mt19937_64& g,std::size_t n){
    constexpr int N=B/32;std::vector<Float<B>> v(n);int lo=-40,hi=4;
    if(f==Function::log)lo=-100,hi=100;if(f==Function::sin||f==Function::cos)lo=-30,hi=20;if(f==Function::log1p)lo=-60,hi=-1;
    for(auto& x:v){x=reference::random_number<B>(g,0);x.exponent=lo+int(g()%unsigned(hi-lo+1));if(f==Function::log&&g()%16)x.sign=1;
        if(g()%8==0)for(int i=0;i<N-1;++i)x.limb[i]=0;} // short significands
    // Hard cases (retries at the second/third rung): exp(+-2^-32N), expm1(2^(1-32N)), log1p(2^-32N), cos(2^-16N) and neighbours.
    for(int e:{-32*N,1-32*N,-16*N,-32*N-1})for(int s:{1,-1})for(word j:{0u,1u,2u}){auto x=pow2<B>(e,s);x.limb[0]=j;v.push_back(x);}
    v.push_back(zero<N>());v.push_back(zero<N>(invalid));return v;
}
template<int B> void transcendental_width(Engine& e,Transcendentals& tr,bool all,std::size_t& retried){
    using F=Float<B>;using C=Complex<B/32>;std::mt19937_64 g(B*11);const std::size_t n=300;
    std::vector<Function> real={Function::exp,Function::log1p,Function::cos,Function::atan2};
    if(all)for(auto f:{Function::expm1,Function::log,Function::sin})real.push_back(f);
    std::vector<Function> cx={Function::complex_exp};if(all){cx.push_back(Function::complex_log);cx.push_back(Function::complex_powi);}
    // One batch for every function: in place for odd positions, separate outputs otherwise.
    auto batch=e.batch();std::vector<TranscendentalTicket> tickets;std::vector<std::function<void()>> checks;std::size_t slot=0;
    for(auto f:real){
        auto a=real_inputs<B>(f,g,n);std::vector<F> b(a.size());for(auto& y:b)y=value<F>(g,3);bool in_place=slot++%2;
        auto A=put(e,a),Bb=put(e,b);auto O=in_place?A:e.make_buffer<F>(a.size());
        tickets.push_back(f==Function::atan2?tr.run(batch,f,A,Bb,O):tr.run(batch,f,A,O));
        checks.push_back([&tr,f,a,b,O,B_=B,&retried,t=tickets.back()]{std::vector<F> host(a.size());tr.run(B_,f,a.data(),host.data(),a.size(),b.data());
            const TranscendentalReport& r=t.report();require(r.retried==tr.report().retried&&r.unresolved==tr.report().unresolved&&r.count==a.size(),"retry report differs");
            retried+=r.retried;same_all(get(O),host,"resident transcendental "+std::to_string(int(f))+" bits="+std::to_string(B_));});
    }
    for(auto f:cx){
        std::vector<C> z(n);for(auto& w:z){w=value<C>(g,3);if(f==Function::complex_exp){w.re.exponent=std::min(w.re.exponent,4);}}
        z.push_back({pow2<B>(0),pow2<B>(-16*(B/32))});z.push_back({pow2<B>(-32*(B/32)),zero<B/32>()});bool in_place=slot++%2;
        auto Z=put(e,z);auto O=in_place?Z:e.make_buffer<C>(z.size());std::vector<std::int32_t> k(z.size());for(auto& q:k)q=int(g()%41)-20;auto K=put(e,k);
        tickets.push_back(f==Function::complex_powi?tr.powi(batch,Z,K,O):tr.run(batch,f,Z,O));
        checks.push_back([&tr,f,z,k,O,B_=B,&retried,t=tickets.back()]{std::vector<C> host(z.size());
            if(f==Function::complex_powi)tr.powi(B_,z.data(),k.data(),k.size(),host.data(),z.size());else tr.run(B_,f,z.data(),host.data(),z.size());
            require(t.report().retried==tr.report().retried,"retry report differs (complex)");retried+=t.report().retried;
            same_all(get(O),host,"resident transcendental "+std::to_string(int(f))+" bits="+std::to_string(B_));});
    }
    for(auto& t:tickets)require(!t.resolved(),"ticket resolved before submission");
    auto submission=batch.submit();for(auto& t:tickets)rejects<std::logic_error>([&]{t.report();},"report before wait");
    submission.wait();for(auto& t:tickets)require(t.resolved()&&!t.report().provisional_reads,"ticket state after wait");
    for(auto& c:checks)c();
    std::cout<<B<<" bits: resident transcendentals ("<<real.size()+cx.size()<<" functions, in place and separate) equal the host-array calls"<<std::endl;
}

// ---------------- chains, ownership, lifetime ----------------
// vector recurrence -> norms of the result; complex_mul -> complex exp -> poly_eval; tree passes after an engine op.
void chains(Engine& e,Numerics& nm,Transcendentals& tr){
    {using C=Complex<8>;using R=Float<256>;std::mt19937_64 g(41);VectorRecurrence s;s.lanes=64;s.steps=8;
        std::vector<C> start(4*s.lanes),p(s.steps*4*s.lanes),q(p.size()),out(4*s.lanes);for(auto* v:{&start,&p,&q})for(auto& w:*v)w=value<C>(g,1,false);
        e.vector_recurrence(256,s,start.data(),p.data(),q.data(),nullptr,out.data());Segments seg{4,s.lanes,true};
        std::vector<R> h2(4),hi(4);std::vector<NormInfo> i2(4),ii(4);nm.norm2(256,seg,out.data(),h2.data(),i2.data());nm.norm_inf(256,seg,out.data(),hi.data(),ii.data());
        auto S=put(e,start),P=put(e,p),Q=put(e,q);auto O=e.make_buffer<C>(out.size());auto V2=e.make_buffer<R>(4),Vi=e.make_buffer<R>(4);auto I2=e.make_buffer<NormInfo>(4),Ii=e.make_buffer<NormInfo>(4);
        auto batch=e.batch();batch.vector_recurrence(s,S,P,Q,Buffer<C>(),O);nm.norm2(batch,seg,O,V2,I2);nm.norm_inf(batch,seg,O,Vi,Ii);batch.submit().wait();
        same_all(get(O),out,"chain vector recurrence");same_all(get(V2),h2,"chain norm2");same_all(get(I2),i2,"chain norm2 info");same_all(get(Vi),hi,"chain norm_inf");same_all(get(Ii),ii,"chain norm_inf info");}
    for(int width:{224,384}){auto chain=[&](auto tag){using C=std::decay_t<decltype(tag)>;constexpr int B=bits_of<C>;std::mt19937_64 g(B);const std::size_t n=200;
        std::vector<C> a(n),b(n),prod(n),ex(n),coef(4*7),vals(n);for(auto& v:a)v=value<C>(g,1,false);for(auto& v:b)v=value<C>(g,1,false);for(auto& v:coef)v=value<C>(g,2,false);
        Polynomial shape{n,7,50,true,true};
        e.run(B,Operation::complex_mul,a.data(),b.data(),prod.data(),n);tr.run(B,Function::complex_exp,prod.data(),ex.data(),n);nm.poly_eval(B,shape,coef.data(),ex.data(),vals.data());
        auto A=put(e,a),Bb=put(e,b),Co=put(e,coef);auto P=e.make_buffer<C>(n),E=e.make_buffer<C>(n),V=e.make_buffer<C>(n);
        auto batch=e.batch();batch.run(Operation::complex_mul,A,Bb,P);auto t=tr.run(batch,Function::complex_exp,P,E);nm.poly_eval(batch,shape,Co,E,V);
        auto sub=batch.submit();rejects<std::logic_error>([&]{E.mapped();},"mapping a pass output before wait");sub.wait();
        require(t.report().provisional_reads&&t.report().retried==0,"chain ticket (random inputs are decided on the GPU)");
        same_all(get(P),prod,"chain complex_mul");same_all(get(E),ex,"chain complex exp");same_all(get(V),vals,"chain poly_eval after complex exp");};
        if(width==224)chain(Complex<7>{});else chain(Complex<12>{});}
    // In place with retries: exp(x) -> x, then an engine product reads the provisional x; x is final (and equal to the host
    // call) after wait, the ticket reports the retries and the provisional read. Writing x again in the batch is rejected.
    {using F=Float<256>;std::mt19937_64 g(5);auto x=real_inputs<256>(Function::exp,g,64);std::vector<F> host(x.size()),y(x.size());for(auto& v:y)v=value<F>(g,2,false);
        tr.run(256,Function::exp,x.data(),host.data(),x.size());auto host_retried=tr.report().retried;require(host_retried>0,"hard exp inputs did not retry");
        auto X=put(e,x),Y=put(e,y);auto Z=e.make_buffer<F>(x.size());auto batch=e.batch();auto t=tr.run(batch,Function::exp,X,X);
        rejects<std::logic_error>([&]{batch.run(Operation::add,Y,Y,X);},"engine write to a provisional output");
        rejects<std::logic_error>([&]{tr.run(batch,Function::cos,Y,X);},"second pass into a provisional output");
        rejects<std::logic_error>([&]{nm.poly_eval(batch,Polynomial{x.size(),1,1,false,false},Y,Y,X);},"unit write to a provisional output");
        batch.run(Operation::mul,X,Y,Z);auto sub=batch.submit();require(!t.resolved(),"resolved before wait");
        rejects<std::logic_error>([&]{X.download(host.data(),1);},"download before wait");sub.wait();
        require(t.resolved()&&t.report().retried==host_retried&&t.report().provisional_reads,"in-place ticket report");same_all(get(X),host,"in-place exp with retries");}
    // Tree passes inside one batch after an engine op that writes their input; numerics outputs feeding numerics and engine inputs.
    {using F=Float<384>;std::mt19937_64 g(9);const std::size_t L=20000;std::vector<F> x(L),y(L),xy(L),c={value<F>(g,2,false),value<F>(g,2,false)},pv(L),h(1),hm(1),hw(1);
        for(auto& v:x)v=value<F>(g,30,false);for(auto& v:y)v=value<F>(g,30,false);Polynomial shape{L,2,L,false,false};
        e.run(384,Operation::mul,x.data(),y.data(),xy.data(),L);nm.norm2(384,Segments{1,L,false},xy.data(),h.data());
        nm.poly_eval(384,shape,c.data(),xy.data(),pv.data());nm.norm_max(384,Segments{1,L,false},pv.data(),hm.data());e.run(384,Operation::mul,h.data(),hm.data(),hw.data(),1);
        auto X=put(e,x),Y=put(e,y),Cs=put(e,c);auto V=e.make_buffer<F>(1),P=e.make_buffer<F>(L),M=e.make_buffer<F>(1),W=e.make_buffer<F>(1);
        auto batch=e.batch();batch.run(Operation::mul,X,Y,X);nm.norm2(batch,Segments{1,L,false},X,V);nm.poly_eval(batch,shape,Cs,X,P);
        nm.norm_max(batch,Segments{1,L,false},P,M);batch.run(Operation::mul,V,M,W);batch.submit().wait();
        same_all(get(X),xy,"engine mul before norm2");same_all(get(V),h,"norm2 after engine mul (three tree passes)");same_all(get(P),pv,"poly after engine mul");
        same_all(get(M),hm,"norm_max of a poly_eval output");same_all(get(W),hw,"engine mul of two norms");}
    std::cout<<"chains: vector recurrence -> norms, complex_mul -> complex exp -> poly_eval (224/384), in-place exp with retries, engine op -> tree passes"<<std::endl;
}
void ownership(Engine& e,Numerics& nm,Transcendentals& tr){
    using F=Float<256>;using C=Complex<8>;std::mt19937_64 g(3);std::vector<F> x(64);for(auto& v:x)v=value<F>(g,3,false);
    auto X=put(e,x),V=e.make_buffer<F>(1),Z=e.make_buffer<F>(64);auto I=e.make_buffer<NormInfo>(1);Engine other;auto foreign=put(other,x);
    {auto b=e.batch();rejects<std::invalid_argument>([&]{nm.norm_inf(b,Segments{1,64,false},foreign,V);},"foreign norm input");
     rejects<std::invalid_argument>([&]{tr.run(b,Function::exp,foreign,Z);},"foreign transcendental input");
     rejects<std::invalid_argument>([&]{tr.run(b,Function::exp,X,foreign);},"foreign transcendental output");
     rejects<std::invalid_argument>([&]{nm.norm_inf(b,Segments{1,64,true},X,V);},"segments.complex mismatch");
     rejects<std::invalid_argument>([&]{nm.norm_inf(b,Segments{2,64,false},X,V);},"input too small");
     rejects<std::invalid_argument>([&]{nm.norm_inf(b,Segments{1,64,false},X,V,I);nm.norm_inf(b,Segments{2,32,false},X,V,I);},"values too small");
     rejects<std::invalid_argument>([&]{nm.poly_eval(b,Polynomial{64,1,1,true,false},X,X,Z);},"shape.complex mismatch");
     rejects<std::invalid_argument>([&]{nm.poly_eval_jet(b,Polynomial{64,1,1,false,false},3,X,X,Z);},"jet order 3");
     rejects<std::invalid_argument>([&]{tr.run(b,Function::complex_exp,X,Z);},"real buffer for a complex function");
     rejects<std::invalid_argument>([&]{tr.run(b,Function::atan2,X,Z);},"atan2 without x");
     rejects<std::invalid_argument>([&]{auto W=e.make_buffer<F>(63);tr.run(b,Function::exp,X,W);},"transcendental size mismatch");}
    // Unwaited submissions own their buffers.
    {auto b=e.batch();b.run(Operation::square,X,X);auto pending=b.submit();auto next=e.batch();
     rejects<std::logic_error>([&]{nm.norm2(next,Segments{1,64,false},X,V);},"norm of a buffer owned by an unwaited submission");
     rejects<std::logic_error>([&]{tr.run(next,Function::log,X,Z);},"transcendental of a buffer owned by an unwaited submission");
     rejects<std::logic_error>([&]{nm.summarize_status(next,Segments{1,64,false},Z,I);auto t=tr.run(next,Function::sin,Z,X);},"output owned by an unwaited submission");
     pending.wait();}
    // Submitted batches accept nothing; a discarded batch leaves its ticket unresolved.
    {auto b=e.batch();nm.norm2(b,Segments{1,64,false},X,V);b.submit().wait();
     rejects<std::logic_error>([&]{nm.norm2(b,Segments{1,64,false},X,V);},"encode into a submitted batch");rejects<std::logic_error>([&]{tr.run(b,Function::exp,X,Z);},"pass into a submitted batch");}
    TranscendentalTicket orphan;{auto b=e.batch();orphan=tr.run(b,Function::exp,X,Z);}require(!orphan.resolved(),"discarded batch resolved its pass");
    rejects<std::logic_error>([&]{orphan.report();},"report of a discarded pass");require(Z.mapped()!=nullptr,"discarded batch kept its buffers");
    {auto b=e.batch();auto t=tr.run(b,Function::exp,Z,Z);} // zero-sized and discarded passes are harmless
    {auto empty=e.make_buffer<C>(0);auto b=e.batch();auto K=e.make_buffer<std::int32_t>(0);auto t=tr.powi(b,empty,K,empty);auto u=tr.run(b,Function::complex_log,empty,empty);b.submit().wait();
     require(t.resolved()&&u.resolved()&&u.report().count==0,"empty passes resolve");}
    // A submission destroyed without wait() still runs the retries before releasing the buffers.
    {auto y=real_inputs<256>(Function::expm1,g,16);std::vector<F> host(y.size());tr.run(256,Function::expm1,y.data(),host.data(),y.size());auto Y=put(e,y);
     TranscendentalTicket t;{auto b=e.batch();t=tr.run(b,Function::expm1,Y,Y);auto dropped=b.submit();}
     require(t.resolved()&&t.report().retried==tr.report().retried,"retries at submission destruction");same_all(get(Y),host,"expm1 resolved by the submission destructor");}
    std::cout<<"ownership: foreign/busy/submitted/provisional buffers and shape errors rejected; discarded and dropped submissions handled"<<std::endl;
}
// The batch, submission and ticket outlive the units and the engine; units on the system default device share the batch's.
void lifetime(){
    using F=Float<224>;using C=Complex<7>;std::mt19937_64 g(17);const std::size_t n=96;
    std::vector<F> x(n);for(auto& v:x)v=value<F>(g,3,false);auto hard=real_inputs<224>(Function::log1p,g,n);std::vector<C> c(5),z(n);for(auto& v:c)v=value<C>(g,2,false);for(auto& v:z)v=value<C>(g,1,false);
    std::vector<F> h_norm(1),h_log(hard.size());std::vector<C> h_poly(n);TranscendentalReport h_rep;
    {Numerics nm;Transcendentals tr;nm.norm_inf(224,Segments{1,n,false},x.data(),h_norm.data());tr.run(224,Function::log1p,hard.data(),h_log.data(),hard.size());h_rep=tr.report();
     nm.poly_eval(224,Polynomial{n,5,n,true,false},c.data(),z.data(),h_poly.data());}
    Submission pending;TranscendentalTicket ticket;Buffer<F> V,L;Buffer<C> P;
    {Engine engine;auto X=put(engine,x),H=put(engine,hard);auto Cc=put(engine,c),Z=put(engine,z);V=engine.make_buffer<F>(1);L=engine.make_buffer<F>(hard.size());P=engine.make_buffer<C>(n);
     auto batch=engine.batch();
     {Numerics nm(engine);Transcendentals tr(engine);nm.norm_inf(batch,Segments{1,n,false},X,V);ticket=tr.run(batch,Function::log1p,H,L);}
     {Numerics plain;plain.poly_eval(batch,Polynomial{n,5,n,true,false},Cc,Z,P);} // default device = the engine's (single-GPU Macs)
     pending=batch.submit();}
    pending.wait();require(ticket.resolved()&&ticket.report().retried==h_rep.retried&&h_rep.retried>0,"ticket after unit/engine destruction");
    same_all(get(V),h_norm,"norm after unit destruction");same_all(get(L),h_log,"log1p after unit destruction");same_all(get(P),h_poly,"poly after unit destruction");
    std::cout<<"lifetime: batch, submission and ticket outlive units and engine"<<std::endl;
}
// Two threads, each with its own Engine, share one Numerics/Transcendentals and compile a new width concurrently.
void concurrent(){
    using F=Float<288>;std::mt19937_64 g(23);const std::size_t n=500;std::vector<F> x(n);for(auto& v:x)v=value<F>(g,5,false);
    Engine first;Numerics nm(first);Transcendentals tr(first);std::vector<std::string> errors(2);std::vector<std::vector<F>> norms(2),sines(2);
    auto work=[&](int id){try{Engine e;auto X=put(e,x);auto V=e.make_buffer<F>(1);auto S=e.make_buffer<F>(n);auto b=e.batch();nm.norm2(b,Segments{1,n,false},X,V);
        auto t=tr.run(b,Function::sin,X,S);b.submit().wait();norms[id]=get(V);sines[id]=get(S);}catch(const std::exception& ex){errors[id]=ex.what();}};
    std::thread a(work,0),b(work,1);a.join();b.join();for(auto& s:errors)require(s.empty(),s);
    std::vector<F> h(1),hs(n);nm.norm2(288,Segments{1,n,false},x.data(),h.data());tr.run(288,Function::sin,x.data(),hs.data(),n);
    for(int i=0;i<2;++i){same_all(norms[i],h,"concurrent norm2");same_all(sines[i],hs,"concurrent sin");}
    std::cout<<"concurrent: two threads share the units' pipeline caches (288 bits compiled concurrently)"<<std::endl;
}
int main(int argc,char** argv){try{
    bool quick=argc>1&&std::string(argv[1])=="--quick";Engine e;Numerics nm(e);Transcendentals tr(e);
    numerics_width<Float<64>>(e,nm,quick);numerics_width<Complex<2>>(e,nm,quick);numerics_width<Float<224>>(e,nm,quick);numerics_width<Complex<7>>(e,nm,quick);
    numerics_width<Float<256>>(e,nm,quick);numerics_width<Complex<8>>(e,nm,quick);numerics_width<Float<384>>(e,nm,quick);numerics_width<Complex<12>>(e,nm,quick);
    numerics_width<Float<1024>>(e,nm,quick);numerics_width<Complex<32>>(e,nm,quick);
    std::size_t retried=0;transcendental_width<64>(e,tr,false,retried);transcendental_width<224>(e,tr,false,retried);transcendental_width<256>(e,tr,true,retried);
    transcendental_width<384>(e,tr,true,retried);if(!quick)transcendental_width<1024>(e,tr,false,retried);
    require(retried>0,"no resident retries exercised");std::cout<<"resident retries exercised: "<<retried<<std::endl;
    chains(e,nm,tr);ownership(e,nm,tr);lifetime();concurrent();
    std::cout<<"All resident unit checks passed."<<std::endl;return 0;}
catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}}
