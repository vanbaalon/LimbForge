#include "section9_cases.hpp"
#include <type_traits>

// All numerical oracles replay MPFR primitives, never the CPU arithmetic core.
template<int B> F<B> mac(F<B> a,F<B> b,F<B> c,bool fused){return fused?reference::fused<B>(a,b,c):ra<B>(c,rm<B>(a,b));}
template<int B> C<B> mac(C<B> a,C<B> b,C<B> c,bool fused){return fused?reference::complex_fused<B>(a,b,c):ca<B>(c,cm<B>(a,b));}
template<int B> F<B> times(F<B> a,F<B> b){return rm<B>(a,b);}
template<int B> C<B> times(C<B> a,C<B> b){return cm<B>(a,b);}
template<int B> F<B> negative(F<B> a){return negate(a);}
template<int B> C<B> negative(C<B> a){return {negate(a.re),negate(a.im)};}
template<int B,class T> T zvalue(){if constexpr(std::is_same_v<T,F<B>>)return zero<B/32>();else return {zero<B/32>(),zero<B/32>()};}
template<int B,class T> T onevalue(){if constexpr(std::is_same_v<T,F<B>>)return from_decimal<B>("1");else return {from_decimal<B>("1"),zero<B/32>()};}
template<int B,class T> T randomvalue(std::mt19937_64& rng){if constexpr(std::is_same_v<T,F<B>>)return reference::random_number<B>(rng,8);else return {reference::random_number<B>(rng,8),reference::random_number<B>(rng,8)};}
template<int B,class T> T errorvalue(){if constexpr(std::is_same_v<T,F<B>>)return zero<B/32>(invalid);else return {zero<B/32>(),zero<B/32>(invalid)};}
template<int B,class T> T power_reference(T x,int exponent){
    T r=onevalue<B,T>();long long e=exponent;bool inv=e<0;unsigned long long k=inv?-e:e;
    while(k){if(k&1)r=times<B>(r,x);k>>=1;if(k)x=times<B>(x,x);}
    if(!inv)return r;
    if constexpr(std::is_same_v<T,F<B>>)return reference::real<B>(Operation::div,onevalue<B,T>(),r);
    else return reference::complex<B>(Operation::complex_div,onevalue<B,T>(),r);
}
template<int B,class T> std::vector<T> gemm_reference(const StridedGemm& s,const std::vector<T>& a,const std::vector<T>& b,std::vector<T> out){
    for(std::size_t t=0;t<s.count;++t)for(std::size_t i=0;i<s.m;++i)for(std::size_t j=0;j<s.n;++j){
        auto z=s.accumulate?out[t*s.stride_c+i*s.n+j]:zvalue<B,T>();
        for(std::size_t k=0;k<s.k;++k)z=mac<B>(a[t*s.stride_a+i*s.k+k],b[t*s.stride_b+k*s.n+j],z,s.fused);
        out[t*s.stride_c+i*s.n+j]=s.negative?negative<B>(z):z;
    }return out;
}
template<int B,class T> void gemm_variants(Engine& e,BatchedLinalg& la,bool dense){
    std::mt19937_64 rng(B+sizeof(T));
    for(bool small:{false,true})for(bool fused:{false,true})for(bool accumulate:{false,true}){
        StridedGemm s;s.count=dense?(small?4097:257):3;s.m=small?4:9;s.n=small?4:5;s.k=small?4:17;
        s.stride_a=accumulate?0:s.m*s.k+2;s.stride_b=accumulate?s.k*s.n+3:0;s.stride_c=s.m*s.n+3;
        s.fused=fused;s.accumulate=accumulate;s.negative=accumulate;
        const auto label=std::string(std::is_same_v<T,F<B>>?"real":"complex")+" GEMM "+(small?"4x4":"tiled")+" fused="+std::to_string(fused)+" accumulate="+std::to_string(accumulate);
        std::vector<T> a(s.stride_a?(s.count-1)*s.stride_a+s.m*s.k:s.m*s.k),b(s.stride_b?(s.count-1)*s.stride_b+s.k*s.n:s.k*s.n),initial(s.count*s.stride_c);
        for(auto* v:{&a,&b,&initial})for(auto& z:*v)z=randomvalue<B,T>(rng);
        // The first row contains exact cancelling pairs; later rows retain full-width random limbs.
        a[1]=negative<B>(a[0]);for(std::size_t j=0;j<s.n;++j)b[s.n+j]=b[j];
        a[s.k+2]=errorvalue<B,T>();
        auto expected=gemm_reference<B>(s,a,b,initial);auto ab=put(e,a),bb=put(e,b),out=put(e,initial);
        for(int repeat=0;repeat<(dense?3:1);++repeat){out.upload(initial.data(),initial.size());auto batch=e.batch();la.gemm(batch,s,ab,bb,out);batch.submit().wait();check<B>(get(out),expected,label.c_str());}
    }
    StridedGemm s{3,4,4,4,16,0,16};std::vector<T> a(48),b(16),d(16),initial(48);
    for(auto* v:{&a,&b,&d})for(auto& z:*v)z=randomvalue<B,T>(rng);
    auto ab=put(e,a),bb=put(e,b),db=put(e,d),out=put(e,initial);
    for(bool fused:{false,true}){s.fused=fused;auto second=s;second.stride_a=16;second.negative=true;
        auto tmp=gemm_reference<B>(s,a,b,initial),expected=gemm_reference<B>(second,tmp,d,initial);
        auto batch=e.batch();la.product3(batch,s,second,ab,bb,db,out,true);batch.submit().wait();check<B>(get(out),expected,"product3 variant");}
    // Empty dots must preserve accumulation and padding, even with negation/fusion selected.
    for(auto& z:initial)z=randomvalue<B,T>(rng);
    s.k=0;s.accumulate=true;s.negative=true;s.fused=true;Buffer<T> unused;auto expected=gemm_reference<B>(s,initial,initial,initial);
    out.upload(initial.data(),initial.size());auto batch=e.batch();la.gemm(batch,s,unused,unused,out);batch.submit().wait();check<B>(get(out),expected,"empty accumulated dot");
}
template<int B,class T> void power_variants(Engine& e,BatchedLinalg& la){
    std::mt19937_64 rng(B+19);PowerMoments s{2,5,9,3,0};
    std::vector<T> E(10),Y(10),W(30),initial(60);for(auto* v:{&E,&Y,&W,&initial})for(auto& z:*v)z=randomvalue<B,T>(rng);
    Y[0]=zvalue<B,T>();E[9]=errorvalue<B,T>();auto eb=put(e,E),yb=put(e,Y),wb=put(e,W),out=put(e,initial);
    for(int n0:{-3,0,3})for(bool fused:{false,true})for(bool accumulate:{false,true}){
        s.n0=n0;s.fused=fused;s.accumulate=accumulate;auto expected=initial;std::vector<T> table(100);
        for(int t=0;t<2;++t)for(int k=0;k<5;++k){auto pw=power_reference<B>(Y[t*5+k],n0);
            for(int n=0;n<10;++n){table[(t*10+n)*5+k]=times<B>(E[t*5+k],pw);if(n+1<10)pw=times<B>(pw,Y[t*5+k]);}}
        for(int t=0;t<2;++t)for(int n=0;n<10;++n)for(int j=0;j<3;++j){auto z=accumulate?initial[(t*10+n)*3+j]:zvalue<B,T>();
            for(int k=0;k<5;++k)z=mac<B>(table[(t*10+n)*5+k],W[(t*5+k)*3+j],z,fused);expected[(t*10+n)*3+j]=z;}
        const auto label=std::string(std::is_same_v<T,F<B>>?"real":"complex")+" powers n0="+std::to_string(n0)+" fused="+std::to_string(fused)+" accumulate="+std::to_string(accumulate);
        out.upload(initial.data(),initial.size());auto batch=e.batch();la.power_moments(batch,s,eb,yb,wb,out);batch.submit().wait();check<B>(get(out),expected,label.c_str());
    }
}
template<int B> void fused_normal(Engine& e,BatchedLinalg& la){
    std::mt19937_64 rng(B+23);constexpr int m=17,n=5;std::vector<F<B>> J(m*n),g(m),A(n*n),rhs(n);
    for(auto* v:{&J,&g})for(auto& x:*v)x=reference::random_number<B>(rng,8);
    J[2*n+3]=zero<B/32>(invalid);g[7]=zero<B/32>(division_by_zero);
    for(int i=0;i<n;++i){for(int j=0;j<n;++j){auto z=zero<B/32>();for(int k=0;k<m;++k)z=mac<B>(J[k*n+i],J[k*n+j],z,true);A[i*n+j]=z;}
        auto z=zero<B/32>();for(int k=0;k<m;++k)z=mac<B>(J[k*n+i],g[k],z,true);rhs[i]=z;}
    auto jb=put(e,J),gb=put(e,g),ab=e.make_buffer<F<B>>(A.size()),rb=e.make_buffer<F<B>>(rhs.size());auto batch=e.batch();la.normal_equations(batch,jb,gb,m,n,ab,rb,true);batch.submit().wait();check<B>(get(ab),A,"fused normal/status");check<B>(get(rb),rhs,"fused rhs/status");
}
template<int B> void trial_edges(Engine& e,BatchedLinalg& la){
    constexpr int n=3,count=4,nrhs=2;auto z=zero<B/32>();auto one=from_decimal<B>("1");
    std::vector<F<B>> A{one,z,z,z,from_decimal<B>("4"),z,z,z,from_decimal<B>("-1")},D{z,z,one},mu{from_decimal<B>("2"),z,one,zero<B/32>(invalid)},rhs(n*nrhs,one);
    auto a=put(e,A),d=put(e,D),m=put(e,mu),b=put(e,rhs),l=e.make_buffer<F<B>>(count*n*n),x=e.make_buffer<F<B>>(count*n*nrhs);auto st=e.make_buffer<std::uint32_t>(count);
    auto batch=e.batch();la.cholesky_trials(batch,{n,count},a,d,m,l,st);la.cholesky_solve(batch,l,st,count,n,b,nrhs,x);batch.submit().wait();
    require(get(st)==std::vector<std::uint32_t>({0,3,3,1}),"late/zero/status pivots");auto got=get(l),sol=get(x);
    for(int t=0;t<count;++t)for(int i=0;i<n;++i)for(int j=0;j<n;++j){if(i<j)require(same<B>(got[(t*n+i)*n+j],z),"upper Cholesky zero");if(t&&i>=j&&j>=(t==3?0:2))require(same<B>(got[(t*n+i)*n+j],zero<B/32>(invalid)),"failed columns invalid");}
    for(int i=0;i<n;++i)for(int j=0;j<nrhs;++j)require(same<B>(sol[i*nrhs+j],from_decimal<B>(i==1?"0.25":"1")),"multiple RHS solve");
    for(int i=n*nrhs;i<count*n*nrhs;++i)require(same<B>(sol[i],zero<B/32>(invalid)),"failed solve payload");
}
template<int B> void polynomial_variants(Engine& e,BatchedLinalg& la){
    constexpr int lanes=5,groups=3,maxsteps=3;std::mt19937_64 rng(B+31);
    for(bool shared:{false,true})for(unsigned terms:{0u,3u}){
        PolynomialRecurrence s;s.lanes=lanes;s.lanes_per_weight=2;s.coefficient_sets=shared?1:groups;s.terms=terms;
        std::vector<C<B>> start(4*lanes),cp(s.coefficient_sets*4*terms),cq(cp.size()),Y(maxsteps*groups),Ep(maxsteps*4*groups),Eq(Ep.size());
        for(auto* v:{&start,&cp,&cq,&Y,&Ep,&Eq})for(auto& z:*v)z=randomvalue<B,C<B>>(rng);
        if(terms&&!shared)cp[terms-1]=errorvalue<B,C<B>>();
        auto st=put(e,start),p=put(e,cp),q=put(e,cq),y=put(e,Y),ep=put(e,Ep),eq=put(e,Eq);
        for(unsigned steps:{0u,3u})for(bool fused:{false,true})for(bool reverse:{false,true})for(bool all:{false,true}){
            s.steps=steps;s.fused=fused;s.reverse=reverse;s.all_steps=all;std::vector<C<B>> want((all?steps+1:1)*4*lanes,zvalue<B,C<B>>());
            if(all)std::copy(start.begin(),start.end(),want.begin());
            auto horner=[&](const auto& c,int set,int a,C<B> yy){if(!terms)return zvalue<B,C<B>>();auto z=c[(set*4+a)*terms+terms-1];for(unsigned n=terms-1;n-->0;)z=mac<B>(z,yy,c[(set*4+a)*terms+n],fused);return z;};
            for(int lane=0;lane<lanes;++lane){int g=lane/2,set=shared?0:g;C<B> v[4];for(int a=0;a<4;++a)v[a]=start[a*lanes+lane];
                for(unsigned step=0;step<steps;++step){unsigned k=reverse?steps-1-step:step;auto dot=zvalue<B,C<B>>();
                    for(int a=0;a<4;++a)dot=mac<B>(cm<B>(Eq[(k*4+a)*groups+g],horner(cq,set,a,Y[k*groups+g])),v[a],dot,fused);
                    for(int a=0;a<4;++a){v[a]=mac<B>(cm<B>(Ep[(k*4+a)*groups+g],horner(cp,set,a,Y[k*groups+g])),dot,v[a],fused);if(all)want[((step+1)*4+a)*lanes+lane]=v[a];}}
                if(!all)for(int a=0;a<4;++a)want[a*lanes+lane]=v[a];}
            const auto label=std::string("polynomial shared=")+std::to_string(shared)+" terms="+std::to_string(terms)+" steps="+std::to_string(steps)+" fused="+std::to_string(fused)+" reverse="+std::to_string(reverse)+" all_steps="+std::to_string(all);
            auto out=put(e,std::vector<C<B>>(want.size(),errorvalue<B,C<B>>()));auto batch=e.batch();la.polynomial_recurrence(batch,s,st,p,q,y,ep,eq,out);batch.submit().wait();check<B>(get(out),want,label.c_str());
        }
    }
}
template<int B> void inline_edges(Engine& e,BatchedLinalg& la){
#ifdef LIMBFORGE_HAS_MPC
    reference::ExponentRange range;std::size_t bytes=getpagesize();void* storage=nullptr;if(posix_memalign(&storage,bytes,bytes))throw std::bad_alloc();std::unique_ptr<void,decltype(&std::free)> owned(storage,&std::free);
    std::memset(storage,0,bytes);constexpr int count=6;mpc_t v[count];mpc_srcptr pointers[count];std::mt19937_64 rng(B+37);
    for(int i=0;i<count;++i){pointers[i]=v[i];mpfr_custom_init_set(mpc_realref(v[i]),MPFR_ZERO_KIND,0,B,static_cast<char*>(storage)+i*256);mpfr_custom_init_set(mpc_imagref(v[i]),MPFR_ZERO_KIND,0,B,static_cast<char*>(storage)+i*256+128);mpfr_set_si(mpc_realref(v[i]),7+i,MPFR_RNDN);mpfr_set_si(mpc_imagref(v[i]),-3-i,MPFR_RNDN);}
    for(int i=0;i<count;++i){to_mpfr<B>(mpc_realref(v[i]),reference::random_number<B>(rng,4));to_mpfr<B>(mpc_imagref(v[i]),reference::random_number<B>(rng,4));}
    mpfr_set_nan(mpc_realref(v[0]));mpfr_set_inf(mpc_imagref(v[1]),-1);mpfr_set_zero(mpc_realref(v[2]),-1);
    require(mpfr_set_exp(mpc_realref(v[3]),1000000001L)==0,"set upper valid exponent");require(mpfr_set_exp(mpc_imagref(v[3]),-999999999L)==0,"set lower valid exponent");
    require(mpfr_set_exp(mpc_realref(v[4]),1000000002L)==0,"set upper overflow exponent");require(mpfr_set_exp(mpc_imagref(v[5]),-1000000000L)==0,"set lower overflow exponent");
    auto records=describe_inline_mpc(B,storage,bytes,pointers,count);std::vector<C<B>> want(count);for(int i=0;i<count;++i)want[i]=from_mpc<B>(v[i]);
    auto out=e.make_buffer<C<B>>(count);auto batch=e.batch();la.import_inline_complex(batch,storage,bytes,records.data(),count,out);batch.submit().wait();check<B>(get(out),want,"inline NaN/Inf/zero/exponent boundaries");
#else
    (void)e;(void)la;std::cout<<"inline MPC edge cases unavailable (built without MPC)\n";
#endif
}
template<int B> void audit(Engine& e,BatchedLinalg& la,bool dense){
    std::cout<<"Auditing "<<B<<" bits"<<(dense?" (dense repeats)":"")<<std::endl;
    products<B>(e,la);normal<B>(e,la);real_products<B>(e,la);bridge<B>(e,la);polynomial<B>(e,la);
    gemm_variants<B,F<B>>(e,la,dense);gemm_variants<B,C<B>>(e,la,dense);power_variants<B,F<B>>(e,la);power_variants<B,C<B>>(e,la);
    fused_normal<B>(e,la);trial_edges<B>(e,la);polynomial_variants<B>(e,la);inline_edges<B>(e,la);
    std::cout<<B<<" bits: expanded reference cases passed\n";
}
void dispatch(int bits,Engine& e,BatchedLinalg& la,bool dense){switch(bits){
#define LF_WIDTH(B) case B:audit<B>(e,la,dense);break;
    LF_WIDTH(64) LF_WIDTH(96) LF_WIDTH(128) LF_WIDTH(160) LF_WIDTH(192) LF_WIDTH(224) LF_WIDTH(256) LF_WIDTH(288)
    LF_WIDTH(320) LF_WIDTH(352) LF_WIDTH(384) LF_WIDTH(416) LF_WIDTH(448) LF_WIDTH(480) LF_WIDTH(512) LF_WIDTH(544)
    LF_WIDTH(576) LF_WIDTH(608) LF_WIDTH(640) LF_WIDTH(672) LF_WIDTH(704) LF_WIDTH(736) LF_WIDTH(768) LF_WIDTH(800)
    LF_WIDTH(832) LF_WIDTH(864) LF_WIDTH(896) LF_WIDTH(928) LF_WIDTH(960) LF_WIDTH(992) LF_WIDTH(1024)
#undef LF_WIDTH
    default:throw std::invalid_argument("bits must be a multiple of 32 in [64,1024]");}}
int main(int argc,char** argv){try{
    int bits=352;bool all=false,dense=false,bits_given=false;
    for(int i=1;i<argc;++i){std::string arg=argv[i];if(arg=="--all-widths")all=true;else if(arg=="--dense")dense=true;else if(arg=="--bits"&&i+1<argc){bits_given=true;std::string value=argv[++i];std::size_t used;bits=std::stoi(value,&used);if(used!=value.size()||bits<64||bits>1024||bits%32)throw std::invalid_argument("invalid --bits");}
        else if(arg=="--help"){std::cout<<"Usage: test_limbforge_section9_audit [--bits B | --all-widths] [--dense]\nDefault: focused 352-bit cases. --dense repeats larger GEMM batches three times.\nUse MTL_SHADER_VALIDATION=1 for the validation pass.\n";return 0;}else throw std::invalid_argument("unknown/incomplete option: "+arg);}
    if(all&&bits_given)throw std::invalid_argument("choose --bits or --all-widths, not both");
    Engine e;BatchedLinalg la(e);if(all)for(int b=64;b<=1024;b+=32)dispatch(b,e,la,dense);else dispatch(bits,e,la,dense);policy();
    std::cout<<"Section 9 reference run passed; "<<(all?"all 31 widths":"selected width")<<(dense?", dense repeats":"; dense stress not run")<<".\n";return 0;
}catch(const std::exception& e){std::cerr<<"Section 9 audit: "<<e.what()<<'\n';return 1;}}
