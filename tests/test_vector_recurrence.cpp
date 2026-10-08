// Vector recurrence (plan D4) against an independent MPFR implementation of the documented contract.
#include "reference.hpp"
#include <stdexcept>
using namespace limbforge;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
template<int Bits> struct Ref {
    using C=Complex<Bits/32>;
    static C zero_c(){return {zero<Bits/32>(),zero<Bits/32>()};}
    static inline bool fused=false;
    static C fma(const C& a,const C& b,const C& c){return fused?reference::complex_fused<Bits>(a,b,c):reference::complex<Bits>(Operation::complex_add,c,reference::complex<Bits>(Operation::complex_mul,a,b));}
    static C add(const C& a,const C& b){return reference::complex<Bits>(Operation::complex_add,a,b);}
    static C dot(const C* a,std::size_t stride,const C* v){C s=fma(a[0],v[0],zero_c());for(int j=1;j<4;++j)s=fma(a[j*stride],v[j],s);return s;}
};
template<int Bits> Complex<Bits/32> random_c(std::mt19937_64& rng,int span,int shift=0){
    Complex<Bits/32> z={reference::random_number<Bits>(rng,span),reference::random_number<Bits>(rng,span)};z.re.exponent+=shift;z.im.exponent+=shift;return z;}
// Runs one configuration on the GPU and with the MPFR reference; returns the GPU output for chaining.
template<int Bits> std::vector<Complex<Bits/32>> check(Engine& e,const VectorRecurrence& s,const std::vector<Complex<Bits/32>>& start,const std::vector<Complex<Bits/32>>& p,
        const std::vector<Complex<Bits/32>>& q,const std::vector<Complex<Bits/32>>& r,const std::vector<Complex<Bits/32>>& base,const std::vector<Complex<Bits/32>>& dp,
        const std::vector<Complex<Bits/32>>& dq,const std::string& label){
    using C=Complex<Bits/32>;using R=Ref<Bits>;R::fused=s.fused;const std::size_t L=s.lanes,G=L/s.lanes_per_weight,B=L/s.lanes_per_base,T=L/s.lanes_per_tangent,steps=s.steps;
    std::vector<C> out((s.all_steps?steps+1:1)*4*L),expected(out.size());
    e.vector_recurrence(Bits,s,start.data(),p.data(),q.data(),r.data(),out.data(),base.data(),dp.data(),dq.data());
    for(std::size_t lane=0;lane<L;++lane){C v[4];for(int j=0;j<4;++j)v[j]=start[j*L+lane];std::size_t g=lane/s.lanes_per_weight;
        auto store=[&](std::size_t t){for(int j=0;j<4;++j)expected[(s.all_steps?t*4*L:0)+j*L+lane]=v[j];};
        if(s.all_steps)store(0);
        for(std::size_t t=0;t<steps;++t){std::size_t k=s.reverse?steps-1-t:t;
            if(s.matrix){C n[4];for(int i=0;i<4;++i)n[i]=R::dot(&p[(k*16+i*4)*G+g],G,v);for(int i=0;i<4;++i)v[i]=n[i];}
            else{C sq=R::dot(&q[k*4*G+g],G,v);
                if(s.tangent){C b[4];for(int j=0;j<4;++j)b[j]=base[(t*4+j)*B+lane/s.lanes_per_base];std::size_t h=lane/s.lanes_per_tangent;
                    C sb=R::dot(&q[k*4*G+g],G,b),tq=R::dot(&dq[k*4*T+h],T,b);
                    for(int i=0;i<4;++i)v[i]=R::add(R::fma(p[(k*4+i)*G+g],sq,v[i]),R::fma(p[(k*4+i)*G+g],tq,R::fma(dp[(k*4+i)*T+h],sb,R::zero_c())));}
                else for(int i=0;i<4;++i)v[i]=R::fma(p[(k*4+i)*G+g],sq,v[i]);}
            if(s.affine)for(int i=0;i<4;++i)v[i]=R::add(v[i],r[(k*4+i)*L+lane]);
            if(s.all_steps)store(t+1);}
        if(!s.all_steps)store(0);}
    for(std::size_t i=0;i<out.size();++i)require(reference::equal_complex<Bits>(out[i],expected[i]),"GPU vector recurrence vs MPFR "+label+(s.fused?" fused":" composed")+" bits="+std::to_string(Bits)+" index="+std::to_string(i));
    return out;
}
template<int Bits> void run(Engine& e,unsigned steps,bool fused){
    using C=Complex<Bits/32>;std::mt19937_64 rng(31+Bits);const std::size_t L=24;
    auto fill=[&](std::size_t n,int span,int shift){std::vector<C> x(n);for(auto& z:x)z=random_c<Bits>(rng,span,shift);return x;};
    VectorRecurrence s;s.lanes=L;s.steps=steps;s.fused=fused;
    // |p q^T| ~ 1: entries of magnitude ~1/2 with random signs and small exponent spread.
    auto start=fill(4*L,3,0),p=fill(steps*4*L,1,-1),q=fill(steps*4*L,1,-1),r=fill(steps*4*L,3,-4),none=std::vector<C>(1);
    // Cancellation: for lanes 0..3 the first q is chosen so that q.v nearly vanishes at step 0.
    for(std::size_t lane=0;lane<4;++lane){C acc=Ref<Bits>::zero_c();for(int j=0;j<3;++j)acc=Ref<Bits>::fma(q[j*L+lane],start[j*L+lane],acc);
        auto inv=reference::complex<Bits>(Operation::complex_div,acc,start[3*L+lane]);q[3*L+lane]={negate(inv.re),negate(inv.im)};}
    check<Bits>(e,s,start,p,q,none,none,none,none,"rank-one");
    s.lanes_per_weight=4;s.reverse=true;s.affine=true;check<Bits>(e,s,start,p,q,r,none,none,none,"shared-weights reverse affine");
    s.lanes_per_weight=1;s.reverse=false;s.affine=false;s.all_steps=true;auto chain=check<Bits>(e,s,start,p,q,none,none,none,none,"all-steps base");
    // Tangent: lanes 4b..4b+3 share base trajectory b and dp/dq per pair of lanes.
    VectorRecurrence t=s;t.all_steps=false;t.tangent=true;t.lanes_per_base=4;t.lanes_per_tangent=2;
    std::vector<C> base((steps+1)*4*(L/4));for(std::size_t k=0;k<=steps;++k)for(int j=0;j<4;++j)for(std::size_t b=0;b<L/4;++b)base[(k*4+j)*(L/4)+b]=chain[(k*4+j)*L+b];
    auto dv=fill(4*L,3,-2),dp=fill(steps*4*(L/2),1,-3),dq=fill(steps*4*(L/2),1,-3);
    check<Bits>(e,t,dv,p,q,none,base,dp,dq,"tangent");
    VectorRecurrence m;m.lanes=L;m.steps=steps/2+1;m.matrix=true;m.fused=fused;m.lanes_per_weight=2;auto M=fill(m.steps*16*(L/2),1,-2);
    check<Bits>(e,m,start,M,none,none,none,none,none,"matrix");
    VectorRecurrence z=s;z.steps=0;z.all_steps=false;check<Bits>(e,z,start,none,none,none,none,none,none,"zero steps");
    std::cout<<Bits<<" bits: vector recurrence ("<<steps<<" steps, "<<(fused?"fused":"composed")<<") matches MPFR in all modes"<<std::endl;
}
// Hard cfma cases through the fused kernel (its exact complex fma, cfma_rolled, exists only there). One rank-one
// step with q = (1, 0, 0, 0) gives s = v0 = b exactly, so v_i' = cfma(p_i, b, v_i): three chosen triples per lane
// (as in test_fused: exact and near cancellation, a tiny product exposed, far-apart products and addends, ties,
// zeros) plus cfma(p_0, b, b). Run at every width; the MPFR chain of check() is the reference.
template<int N> Number<N> half_ulp(const Number<N>& x,int sign){Number<N> h=zero<N>();h.limb[N-1]=0x80000000u;h.sign=sign;h.exponent=x.exponent-32*N;return h;}
template<int Bits> void hard(Engine& e){
    using C=Complex<Bits/32>;constexpr int N=Bits/32;std::mt19937_64 rng(1009+Bits);const std::size_t L=192;
    VectorRecurrence s;s.lanes=L;s.steps=1;s.fused=true;
    C one={from_decimal<Bits>("1"),zero<N>()},nil={zero<N>(),zero<N>()};
    std::vector<C> start(4*L),p(4*L),q(4*L,nil),none(1);
    for(std::size_t lane=0;lane<L;++lane){
        C b={reference::random_number<Bits>(rng,30),reference::random_number<Bits>(rng,30)};start[lane]=b;q[lane]=one;p[lane]=random_c<Bits>(rng,30);
        for(std::size_t i=1;i<4;++i){
            C a={reference::random_number<Bits>(rng,30),reference::random_number<Bits>(rng,30)},c={reference::random_number<Bits>(rng,60),reference::random_number<Bits>(rng,60)};
            switch((3*lane+i)%12){
            case 0:a={b.im,b.re};break;                                      // re: b.im*b.re - b.re*b.im = 0 exactly
            case 1:a={b.im,b.re};a.re.limb[0]^=1;break;                      // near-total cancellation
            case 2:a.re=from_decimal<Bits>("1");c.re=negate(b.re);a.im.exponent=b.im.exponent-3*Bits;break; // tiny product exposed
            case 3:{C m=cmul(a,b);c={negate(m.re),negate(m.im)};break;}       // addend cancels the rounded product
            case 4:a.im.exponent-=int(rng()%(4*Bits));break;                 // products far apart
            case 5:c.re.exponent-=int(rng()%(4*Bits));c.im.exponent+=int(rng()%(4*Bits));break; // addend far below / above
            case 6:a.im=zero<N>();break;case 7:c=nil;break;
            case 8:a=one;c.re=half_ulp(b.re,-1);break;                       // tie
            case 9:a=one;c.re=half_ulp(b.re,1);c.re.limb[0]|=1;break;        // just past a tie
            case 10:a.re.exponent=b.im.exponent+5*Bits;c.re.exponent=a.re.exponent+b.re.exponent+int(rng()%(2*Bits))-Bits;break; // spread over three windows
            default:break;}
            p[i*L+lane]=a;start[i*L+lane]=c;}
    }
    check<Bits>(e,s,start,p,q,none,none,none,none,"hard cfma cases");
}
template<int Bits> void hard_all(Engine& e){hard<Bits>(e);if constexpr(Bits<1024)hard_all<Bits+32>(e);}
// Resident: base (all steps) then tangent reading that chain in one batch; in-place rank-one run.
template<int Bits> void resident(Engine& e,bool fused){
    using C=Complex<Bits/32>;std::mt19937_64 rng(77+Bits);const std::size_t L=32,B=L/4;const unsigned steps=40;
    auto fill=[&](std::size_t n,int span,int shift){std::vector<C> x(n);for(auto& z:x)z=random_c<Bits>(rng,span,shift);return x;};
    auto bstart=fill(4*B,3,0),p=fill(steps*4*B,1,-1),q=fill(steps*4*B,1,-1),dv=fill(4*L,3,-2),dp=fill(steps*4*(L/2),1,-3),dq=fill(steps*4*(L/2),1,-3);
    VectorRecurrence base;base.lanes=B;base.steps=steps;base.all_steps=true;base.fused=fused;
    VectorRecurrence tangent;tangent.lanes=L;tangent.steps=steps;tangent.lanes_per_weight=4;tangent.lanes_per_base=4;tangent.lanes_per_tangent=2;tangent.tangent=true;tangent.fused=fused;
    // Host-array path (validated against MPFR above) as the expected result.
    std::vector<C> chain((steps+1)*4*B),expected(4*L),got(4*L);
    e.vector_recurrence(Bits,base,bstart.data(),p.data(),q.data(),nullptr,chain.data());
    e.vector_recurrence(Bits,tangent,dv.data(),p.data(),q.data(),nullptr,expected.data(),chain.data(),dp.data(),dq.data());
    auto up=[&](const std::vector<C>& x){auto b=e.make_buffer<C>(x.size());b.upload(x.data(),x.size());return b;};
    auto bs=up(bstart),bp=up(p),bq=up(q),bdv=up(dv),bdp=up(dp),bdq=up(dq);auto bchain=e.make_buffer<C>(chain.size());Buffer<C> none;
    auto batch=e.batch();batch.vector_recurrence(base,bs,bp,bq,none,bchain);batch.vector_recurrence(tangent,bdv,bp,bq,none,bdv,bchain,bdp,bdq);batch.submit().wait();
    bdv.download(got.data(),got.size());
    for(std::size_t i=0;i<got.size();++i)require(reference::equal_complex<Bits>(got[i],expected[i]),"resident tangent differs from host path bits="+std::to_string(Bits)+" index="+std::to_string(i));
    bool rejected=false;try{auto small=e.make_buffer<C>(4);auto b2=e.batch();b2.vector_recurrence(base,bs,bp,bq,none,small);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"undersized resident output accepted");
    std::cout<<Bits<<" bits: resident base + tangent batch matches the host path ("<<(fused?"fused":"composed")<<")"<<std::endl;
}
int main(){try{Engine e;resident<256>(e,false);resident<224>(e,false);resident<256>(e,true);for(bool f:{false,true}){run<224>(e,150,f);run<256>(e,150,f);run<384>(e,60,f);run<64>(e,40,f);run<512>(e,12,f);run<768>(e,8,f);run<1024>(e,6,f);}
    hard_all<64>(e);std::cout<<"hard fused cases match MPFR at all 31 precisions"<<std::endl;
    bool rejected=false;try{VectorRecurrence s;s.lanes=6;s.lanes_per_weight=4;std::vector<Complex<2>> x(24);e.vector_recurrence(64,s,x.data(),x.data(),x.data(),nullptr,x.data());}
    catch(const std::invalid_argument&){rejected=true;}require(rejected,"indivisible lanes_per_weight accepted");
    std::cout<<"All vector recurrence checks passed."<<std::endl;return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
