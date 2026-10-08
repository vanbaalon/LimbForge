// Polynomial values and jets (plan S1) and norms / status summaries (plan S4) on the GPU against independent MPFR
// replays of the documented rounding sequences (docs/numerics.md), bitwise; plus analytic jet identities.
#include "reference.hpp"
#include "limbforge/numerics.hpp"
#include <stdexcept>
#include <type_traits>
using namespace limbforge;
using reference::MP;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
template<int N> Number<N> re_of(const Number<N>& x){return x;}
template<int N> Number<N> im_of(const Number<N>&){return zero<N>();}
template<int N> Number<N> re_of(const Complex<N>& z){return z.re;}
template<int N> Number<N> im_of(const Complex<N>& z){return z.im;}
template<int Bits> Float<Bits> power_of_two(int exponent,int sign=1){auto x=zero<Bits/32>();x.limb[Bits/32-1]=0x80000000u;x.sign=sign;x.exponent=exponent;return x;}
template<int Bits> Float<Bits> integer(long v){reference::ExponentRange range;MP m(Bits);mpfr_set_si(m.x,v,MPFR_RNDN);return from_mpfr<Bits>(m.x);}
template<int Bits> Float<Bits> with_exponent(Float<Bits> x,int e){if(x.sign)x.exponent=e;return x;}
// MPFR array of n values at `bits` precision.
struct MPArray { std::vector<__mpfr_struct> v; MPArray(std::size_t n,int bits):v(n){for(auto& x:v)mpfr_init2(&x,bits);} ~MPArray(){for(auto& x:v)mpfr_clear(&x);} mpfr_ptr operator[](std::size_t i){return &v[i];} };
template<class T> bool same(const T& a,const T& b){
    if constexpr(detail::Format<T>::complex)return reference::equal_complex<detail::Format<T>::bits>(a,b);else return reference::equal<detail::Format<T>::bits>(a,b);}

// ---------------- S1: polynomial values and jets ----------------
// Reference: r_j = 0; for k = terms-1 .. 0: r_2 = mac(r_2,x,r_1), r_1 = mac(r_1,x,r_0), r_0 = mac(r_0,x,c_k),
// mac = mpfr_mul then mpfr_add (complex: the composed complex product, then componentwise add) or mpfr_fma / exact cfma.
template<int Bits,class T> T mac(const T& a,const T& x,const T& b,bool fused){
    using O=Operation;
    if constexpr(detail::Format<T>::complex){if(fused)return reference::complex_fused<Bits>(a,x,b);return reference::complex<Bits>(O::complex_add,reference::complex<Bits>(O::complex_mul,a,x),b);}
    else{if(fused)return reference::fused<Bits>(a,x,b);return reference::real<Bits>(O::add,reference::real<Bits>(O::mul,a,x),b);}
}
template<class T> T zero_like(){if constexpr(detail::Format<T>::complex)return T{zero<detail::Format<T>::bits/32>(),zero<detail::Format<T>::bits/32>()};else return zero<detail::Format<T>::bits/32>();}
template<int Bits,class T> void reference_jets(const T* c,std::size_t terms,const T& x,bool fused,T* r){
    r[0]=r[1]=r[2]=zero_like<T>();
    for(std::size_t k=terms;k-->0;){r[2]=mac<Bits>(r[2],x,r[1],fused);r[1]=mac<Bits>(r[1],x,r[0],fused);r[0]=mac<Bits>(r[0],x,c[k],fused);}
}
template<int Bits,bool Cx> void poly_case(Numerics& nm,std::size_t terms,bool fused){
    using T=std::conditional_t<Cx,Complex<Bits/32>,Float<Bits>>;std::mt19937_64 rng(Bits*131+terms*7+Cx*2+fused);
    auto rnd=[&](int span)->T{if constexpr(Cx)return T{reference::random_number<Bits>(rng,span),reference::random_number<Bits>(rng,span)};else return reference::random_number<Bits>(rng,span);};
    auto real_t=[&](Float<Bits> v)->T{if constexpr(Cx)return T{v,zero<Bits/32>()};else return v;};
    const std::size_t pps=4,sets=6,P=pps*sets-1; // the last set is shared by 3 points only
    std::vector<T> c(sets*terms),x(P);for(auto& v:c)v=rnd(20);for(auto& v:x)v=rnd(2);
    for(std::size_t k=0;k<terms;k+=3)c[1*terms+k]=zero_like<T>();                                // set 1: zero coefficients
    if(terms)c[2*terms+terms/2]=real_t(zero<Bits/32>(invalid));                                // set 2: coefficient with a status
    {std::size_t n=terms?std::min<std::size_t>(terms-1,30):0;long b=1;                             // set 3: (x-1)^n, points near 1 (cancellation)
     // coefficient k of (x-1)^n is C(n,k) (-1)^(n-k); b walks C(n,k)
     for(std::size_t k=0;k<terms;++k){c[3*terms+k]=k<=n?real_t(integer<Bits>(((n-k)%2?-b:b))):zero_like<T>();if(k<n)b=b*long(n-k)/long(k+1);}
     for(std::size_t i=3*pps;i<4*pps;++i){T d=rnd(0);if constexpr(Cx){d.re=with_exponent<Bits>(d.re,-12);d.im=with_exponent<Bits>(d.im,-14);x[i]=T{reference::real<Bits>(Operation::add,d.re,integer<Bits>(1)),d.im};}
        else x[i]=reference::real<Bits>(Operation::add,with_exponent<Bits>(d,-12),integer<Bits>(1));}}
    for(std::size_t k=0;k<terms;++k){T v=rnd(0);if constexpr(Cx){v.re.exponent=0;v.im.exponent=0;if(k%2)v={negate(v.re),negate(v.im)};}else{v.exponent=0;if(k%2)v=negate(v);}c[4*terms+k]=v;} // set 4: alternating, equal size
    x[1]=zero_like<T>();if constexpr(Cx)x[5].im.status=invalid;else x[5].status=invalid;           // a zero point; a point with a status
    Polynomial shape{P,terms,pps,Cx,fused};
    std::vector<T> j2(3*P),j1(2*P),v0(P);
    nm.poly_eval_jet(Bits,shape,2,c.data(),x.data(),j2.data());nm.poly_eval_jet(Bits,shape,1,c.data(),x.data(),j1.data());nm.poly_eval(Bits,shape,c.data(),x.data(),v0.data());
    for(std::size_t i=0;i<P;++i){T r[3];reference_jets<Bits>(&c[(i/pps)*terms],terms,x[i],fused,r);
        std::string label=" bits="+std::to_string(Bits)+(Cx?" complex":" real")+(fused?" fused":" composed")+" terms="+std::to_string(terms)+" point="+std::to_string(i);
        for(int j=0;j<3;++j)require(same(j2[3*i+j],r[j]),"poly_eval_jet order 2, jet "+std::to_string(j)+label);
        for(int j=0;j<2;++j)require(same(j1[2*i+j],r[j]),"poly_eval_jet order 1, jet "+std::to_string(j)+label);
        require(same(v0[i],r[0]),"poly_eval"+label);}
    // Sanity on the fixtures themselves: the status point and (for terms > 0) the status set propagate.
    if(terms){require(re_of(v0[5]).status||im_of(v0[5]).status,"status point did not propagate");require(re_of(v0[2*pps]).status==invalid,"status coefficient did not propagate");}
    else for(std::size_t i=0;i<P;++i)require(same(v0[i],zero_like<T>()),"empty polynomial is not zero");
}
// Analytic identity: p(x) = (x-a)^k has jets ((x-a)^k, k (x-a)^(k-1), C(k,2) (x-a)^(k-2)). Points far from a keep the
// Horner condition number below 2^14, so each jet agrees with the exact value to 2^(24-bits) relatively.
template<int Bits,bool Cx> void poly_identity(Numerics& nm,bool fused){
    using T=std::conditional_t<Cx,Complex<Bits/32>,Float<Bits>>;std::mt19937_64 rng(Bits*17+Cx+2*fused);reference::ExponentRange range;
    const long a=3;const int hp=4*Bits;
    for(unsigned k:{1u,2u,5u,12u}){
        std::vector<T> c(k+1),x(16);long b=1;
        for(unsigned j=0;j<=k;++j){long v=b;for(unsigned t=j;t<k;++t)v*=-a;Float<Bits> f=integer<Bits>(v);if constexpr(Cx)c[j]={f,zero<Bits/32>()};else c[j]=f;b=b*long(k-j)/long(j+1);}
        for(auto& p:x){Float<Bits> u=reference::random_number<Bits>(rng,0);u.exponent=3;u.sign=1;
            if constexpr(Cx){Float<Bits> w=reference::random_number<Bits>(rng,0);w.exponent=0;p={u,w};}else p=u;}
        std::vector<T> jets(3*x.size());nm.poly_eval_jet(Bits,Polynomial{x.size(),k+1,x.size(),Cx,fused},2,c.data(),x.data(),jets.data());
        MPArray m(8,hp);// 0,1: d = x - a; 2,3: d^(k-j) power; 4,5: scratch; 6,7: error
        for(std::size_t i=0;i<x.size();++i){
            to_mpfr<Bits>(m[0],re_of(x[i]));mpfr_sub_si(m[0],m[0],a,MPFR_RNDN);if constexpr(Cx)to_mpfr<Bits>(m[1],im_of(x[i]));else mpfr_set_zero(m[1],1);
            for(int j=0;j<3;++j){if(unsigned(j)>k)continue;
                mpfr_set_ui(m[2],1,MPFR_RNDN);mpfr_set_zero(m[3],1);
                for(unsigned t=0;t<k-unsigned(j);++t){mpfr_mul(m[4],m[2],m[0],MPFR_RNDN);mpfr_fms(m[4],m[3],m[1],m[4],MPFR_RNDN);mpfr_neg(m[4],m[4],MPFR_RNDN); // re*dre - im*dim
                    mpfr_mul(m[5],m[2],m[1],MPFR_RNDN);mpfr_fma(m[5],m[3],m[0],m[5],MPFR_RNDN);mpfr_set(m[2],m[4],MPFR_RNDN);mpfr_set(m[3],m[5],MPFR_RNDN);}
                long scale=j==0?1:j==1?long(k):long(k)*long(k-1)/2;mpfr_mul_si(m[2],m[2],scale,MPFR_RNDN);mpfr_mul_si(m[3],m[3],scale,MPFR_RNDN);
                to_mpfr<Bits>(m[4],re_of(jets[3*i+j]));to_mpfr<Bits>(m[5],im_of(jets[3*i+j]));
                mpfr_sub(m[4],m[4],m[2],MPFR_RNDN);mpfr_sub(m[5],m[5],m[3],MPFR_RNDN);mpfr_hypot(m[6],m[4],m[5],MPFR_RNDN);mpfr_hypot(m[7],m[2],m[3],MPFR_RNDN);
                mpfr_mul_2si(m[7],m[7],24-Bits,MPFR_RNDN);
                require(mpfr_cmp(m[6],m[7])<=0,"jet identity of (x-3)^"+std::to_string(k)+" jet "+std::to_string(j)+" bits="+std::to_string(Bits)+(Cx?" complex":" real")+(fused?" fused":" composed"));}
        }
    }
}

// ---------------- S4: norms and status summaries ----------------
enum class Op { inf, max, two, ratio, status };
const char* op_name(Op op){static const char* n[]={"norm_inf","norm_max","norm2","scaled_residual","summarize_status"};return n[int(op)];}
// Squared scaled component: RN((|c| 2^-e)^2), or 0 when c is zero or more than Bits+32 bits below 2^e.
template<int Bits> void scaled_square(mpfr_ptr out,const Float<Bits>& c,long e){
    if(!c.sign||long(c.exponent)-e< -(Bits+32)){mpfr_set_zero(out,1);return;}
    MP y(Bits);to_mpfr<Bits>(y.x,c);mpfr_abs(y.x,y.x,MPFR_RNDN);mpfr_mul_2si(y.x,y.x,-e,MPFR_RNDN);mpfr_sqr(out,y.x,MPFR_RNDN);
}
// Modulus key s * 4^e, s = RN(RN(a'^2) + RN(b'^2)) (exact power-of-4 scaling; MPFR's extended exponent range).
template<int Bits> void modulus_key(mpfr_ptr key,const Float<Bits>& a,const Float<Bits>& b){
    if(!a.sign&&!b.sign){mpfr_set_zero(key,1);return;}
    long e=!b.sign||(a.sign&&a.exponent>=b.exponent)?a.exponent:b.exponent;MP t(Bits),u(Bits);
    scaled_square<Bits>(t.x,a,e);scaled_square<Bits>(u.x,b,e);mpfr_add(key,t.x,u.x,MPFR_RNDN);mpfr_mul_2si(key,key,2*e,MPFR_RNDN);
}
template<int Bits,class T> std::pair<Float<Bits>,NormInfo> reference_norm(Op op,const T* x,const Float<Bits>* scale,std::size_t L){
    constexpr bool Cx=detail::Format<T>::complex;reference::ExponentRange range;NormInfo info;Float<Bits> value=zero<Bits/32>();
    if(!L)return {value,info};
    std::vector<Float<Bits>> ratios(op==Op::ratio?L:0);
    for(std::size_t i=0;i<L;++i){Float<Bits> a=re_of(x[i]),b=im_of(x[i]);word st=a.status|b.status;
        if(op==Op::ratio){st|=scale[i].status;
            if(!st){bool rz=!a.sign&&!b.sign;
                if(!scale[i].sign){if(!rz)st|=division_by_zero;ratios[i]=zero<Bits/32>();}
                else if(rz)ratios[i]=zero<Bits/32>();
                else{MP h(Bits),s(Bits),q(Bits);if(Cx){modulus_key<Bits>(h.x,a,b);mpfr_sqrt(h.x,h.x,MPFR_RNDN);}else{to_mpfr<Bits>(h.x,a);mpfr_abs(h.x,h.x,MPFR_RNDN);}
                    to_mpfr<Bits>(s.x,scale[i]);mpfr_abs(s.x,s.x,MPFR_RNDN);mpfr_div(q.x,h.x,s.x,MPFR_RNDN);ratios[i]=from_mpfr<Bits>(q.x);st|=ratios[i].status;}}}
        if(st){info.status|=st;if(!info.failing++)info.first_failing=std::uint32_t(i);}}
    if(op==Op::status)return {value,info};
    if(info.status){info.index=info.first_failing;return {zero<Bits/32>(info.status),info};}
    if(op==Op::two){
        long E=LONG_MIN;for(std::size_t i=0;i<L;++i)for(auto c:{re_of(x[i]),im_of(x[i])})if(c.sign&&c.exponent>E)E=c.exponent;
        if(E==LONG_MIN)return {value,info};
        MPArray t(L,Bits);MP u(Bits);for(std::size_t i=0;i<L;++i){scaled_square<Bits>(t[i],re_of(x[i]),E);if(Cx){scaled_square<Bits>(u.x,im_of(x[i]),E);mpfr_add(t[i],t[i],u.x,MPFR_RNDN);}}
        for(std::size_t n=L;n>1;n=(n+1)/2)for(std::size_t i=0;i<(n+1)/2;++i){if(2*i+1<n)mpfr_add(t[i],t[2*i],t[2*i+1],MPFR_RNDN);else mpfr_set(t[i],t[2*i],MPFR_RNDN);}
        mpfr_sqrt(t[0],t[0],MPFR_RNDN);mpfr_mul_2si(t[0],t[0],E,MPFR_RNDN);value=from_mpfr<Bits>(t[0]);info.status|=value.status;return {value,info};
    }
    MP best(Bits),key(Bits);std::size_t arg=0;
    for(std::size_t i=0;i<L;++i){Float<Bits> a=re_of(x[i]),b=im_of(x[i]);
        if(op==Op::ratio)to_mpfr<Bits>(key.x,ratios[i]);
        else if(op==Op::inf&&Cx)modulus_key<Bits>(key.x,a,b);
        else{to_mpfr<Bits>(key.x,a);mpfr_abs(key.x,key.x,MPFR_RNDN);if(op==Op::max&&Cx){MP y(Bits);to_mpfr<Bits>(y.x,b);mpfr_abs(y.x,y.x,MPFR_RNDN);if(mpfr_cmp(y.x,key.x)>0)mpfr_set(key.x,y.x,MPFR_RNDN);}}
        if(!i||mpfr_cmp(key.x,best.x)>0){mpfr_set(best.x,key.x,MPFR_RNDN);arg=i;}}
    if(op==Op::inf&&Cx)mpfr_sqrt(best.x,best.x,MPFR_RNDN);
    value=from_mpfr<Bits>(best.x);info.index=std::uint32_t(arg);info.status|=value.status;return {value,info};
}
template<int Bits,bool Cx> struct NormData {
    using T=std::conditional_t<Cx,Complex<Bits/32>,Float<Bits>>;
    std::vector<T> x;std::vector<Float<Bits>> scale;std::size_t S=0,L=0;
};
// Segment kinds: 0 random; 1 zeros; 2 ties at the maximum; 3 huge, tiny and ordinary exponents (a tiny scale: ratio overflow);
// 4 statuses; 5 all huge (norm2 / complex modulus overflow, huge scales); 6 a zero scale under a nonzero residual.
template<int Bits,bool Cx> NormData<Bits,Cx> norm_data(std::size_t L,std::size_t S,std::mt19937_64& rng){
    using T=typename NormData<Bits,Cx>::T;NormData<Bits,Cx> d;d.S=S;d.L=L;d.x.resize(S*L);d.scale.resize(S*L);
    auto rnd=[&](int span)->T{if constexpr(Cx)return T{reference::random_number<Bits>(rng,span),reference::random_number<Bits>(rng,span)};else return reference::random_number<Bits>(rng,span);};
    auto expo=[&](T v,int e,int f)->T{if constexpr(Cx){v.re.exponent=e;v.im.exponent=f;}else v.exponent=e;return v;};
    for(std::size_t s=0;s<S;++s)for(std::size_t i=0;i<L;++i){T& v=d.x[s*L+i];Float<Bits>& sc=d.scale[s*L+i];std::size_t kind=s%7;
        v=rnd(50);sc=reference::random_number<Bits>(rng,50);
        if(kind==1){v=zero_like<T>();if(i%3==0)sc=zero<Bits/32>();}
        if(kind==2)sc=power_of_two<Bits>(0);
        if(kind==3){if(i%7==3)v=expo(v,-1000000000+int(i%4),-999999999);if(i==0||i==L-1||i==L/2)v=expo(v,999999997,999999990+int(i%5));if(i==0)sc=with_exponent<Bits>(sc,-999999990);}
        if(kind==5){if constexpr(Cx){v.re.limb[Bits/32-1]|=0xf0000000u;v.im.limb[Bits/32-1]|=0xf0000000u;}else v.limb[Bits/32-1]|=0xf0000000u;
            v=expo(v,1000000000,1000000000);sc=with_exponent<Bits>(sc,999999980);}
        if(kind==6&&i==L/2)sc=zero<Bits/32>();}
    for(std::size_t s=2;s<S;s+=7){T top=expo(rnd(0),60,59);std::size_t p[3]={L/3,L/2,L-1};T alt=top;
        if constexpr(Cx)alt=T{top.im,top.re};
        d.x[s*L+p[0]]=top;d.x[s*L+p[1]]=alt;
        if constexpr(Cx)d.x[s*L+p[2]]=T{negate(top.re),negate(top.im)};else d.x[s*L+p[2]]=negate(top);}
    for(std::size_t s=4;s<S;s+=7){T& u=d.x[s*L+L/2];if constexpr(Cx)u.im.status=invalid;else u.status=invalid;d.x[s*L+L-1]=zero_like<T>();
        if constexpr(Cx)d.x[s*L+L-1].re.status=exponent_overflow;else d.x[s*L+L-1].status=exponent_overflow;d.scale[s*L]=zero<Bits/32>(division_by_zero);}
    return d;
}
template<int Bits,bool Cx> void norm_check(Numerics& nm,const NormData<Bits,Cx>& d,const std::string& what){
    const std::size_t S=d.S,L=d.L;Segments seg{S,L,Cx};
    for(Op op:{Op::inf,Op::max,Op::two,Op::ratio,Op::status}){
        std::vector<Float<Bits>> v(S,power_of_two<Bits>(5,-1)); /* overwritten by every call */std::vector<NormInfo> info(S);
        switch(op){case Op::inf:nm.norm_inf(Bits,seg,d.x.data(),v.data(),info.data());break;case Op::max:nm.norm_max(Bits,seg,d.x.data(),v.data(),info.data());break;
            case Op::two:nm.norm2(Bits,seg,d.x.data(),v.data(),info.data());break;case Op::ratio:nm.scaled_residual(Bits,seg,d.x.data(),d.scale.data(),v.data(),info.data());break;
            default:nm.summarize_status(Bits,seg,d.x.data(),info.data());}
        for(std::size_t s=0;s<S;++s){auto r=reference_norm<Bits>(op,&d.x[s*L],&d.scale[s*L],L);
            std::string label=std::string(" ")+op_name(op)+" bits="+std::to_string(Bits)+(Cx?" complex":" real")+" "+what+" L="+std::to_string(L)+" segment="+std::to_string(s);
            const NormInfo& g=info[s];
            require(g.status==r.second.status&&g.failing==r.second.failing&&g.first_failing==r.second.first_failing&&g.index==r.second.index,
                    "info"+label+" got {"+std::to_string(g.status)+","+std::to_string(g.failing)+","+std::to_string(g.first_failing)+","+std::to_string(g.index)+"} expected {"+
                    std::to_string(r.second.status)+","+std::to_string(r.second.failing)+","+std::to_string(r.second.first_failing)+","+std::to_string(r.second.index)+"}");
            if(op!=Op::status)require(reference::equal<Bits>(v[s],r.first),"value"+label);}
    }
}
template<int Bits,bool Cx> void norms(Numerics& nm,bool quick){
    std::mt19937_64 rng(Bits*7+Cx);std::vector<std::size_t> lengths={1,2,3,31,32,33,127,128,129,1000};if(quick)lengths={1,33,129};
    for(std::size_t L:lengths)norm_check<Bits,Cx>(nm,norm_data<Bits,Cx>(L,7,rng),"kinds");
    {auto d=norm_data<Bits,Cx>(7,quick?300:3000,rng);norm_check<Bits,Cx>(nm,d,"many segments");}
    if(!quick){auto d=norm_data<Bits,Cx>(70000,3,rng);using T=typename NormData<Bits,Cx>::T;T top=d.x[2*70000+70000/3]; // three passes; ties far apart
        d.x[2*70000+10]=top;d.x[2*70000+60000]=top;norm_check<Bits,Cx>(nm,d,"multi-pass");}
    // Empty segments and no segments.
    {std::vector<Float<Bits>> v(3,reference::random_number<Bits>(rng,3));std::vector<NormInfo> info(3,NormInfo{7,7,7,7});
     nm.norm2(Bits,Segments{3,0,Cx},nullptr,v.data(),info.data());
     for(int s=0;s<3;++s)require(reference::equal<Bits>(v[s],zero<Bits/32>())&&info[s].status==0&&info[s].failing==0&&info[s].first_failing==no_index&&info[s].index==no_index,"empty segment");
     nm.norm_inf(Bits,Segments{0,5,Cx},nullptr,nullptr,nullptr);}
    std::cout<<Bits<<" bits "<<(Cx?"complex":"real")<<": norms and status summaries match MPFR"<<std::endl;
}
template<int Bits> void all(Numerics& nm,bool quick){
    for(bool fused:{false,true}){
        for(std::size_t terms:{0ul,1ul,2ul,24ul,32ul,62ul,101ul}){poly_case<Bits,false>(nm,terms,fused);poly_case<Bits,true>(nm,terms,fused);}
        poly_identity<Bits,false>(nm,fused);poly_identity<Bits,true>(nm,fused);}
    std::cout<<Bits<<" bits: polynomial values and jets match MPFR (degrees 0, 1, 23, 31, 61, 100; empty), identities hold"<<std::endl;
    norms<Bits,false>(nm,quick);norms<Bits,true>(nm,quick);
}
int main(int argc,char** argv){try{Numerics nm;bool quick=argc>1&&std::string(argv[1])=="--quick";
    all<224>(nm,quick);all<256>(nm,quick);all<384>(nm,quick);all<64>(nm,quick);all<1024>(nm,quick);
    bool rejected=false;try{Float<64> z{};nm.poly_eval_jet(64,Polynomial{1,1,1,false,false},3,&z,&z,&z);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"jet order 3 accepted");
    std::cout<<"All polynomial and norm checks passed."<<std::endl;return 0;}
catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}}
