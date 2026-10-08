// Batched 4x4 complex LU, solve, inverse and determinant (plan D5) against an MPFR implementation of
// the documented sequence (pivot predicate: exact magnitude comparison).
#include "reference.hpp"
#include <stdexcept>
using namespace limbforge;
void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
template<int Bits> struct Ref {
    using C=Complex<Bits/32>;using F=Float<Bits>;bool fused;
    static C mul(const C& a,const C& b){return reference::complex<Bits>(Operation::complex_mul,a,b);}
    static C div(const C& a,const C& b){return reference::complex<Bits>(Operation::complex_div,a,b);}
    C minus(const C& c,const C& a,const C& b)const{ // c - a*b
        if(fused){C r=reference::complex_fused<Bits>(a,b,c,true);return {negate(r.re),negate(r.im)};}
        C q=mul(a,b);return {reference::real<Bits>(Operation::sub,c.re,q.re),reference::real<Bits>(Operation::sub,c.im,q.im)};}
    static F mag(const C& z){return magnitude_compare(z.re,z.im)>=0?z.re:z.im;}
    // Returns status; fills x ([4][R]) and det.
    unsigned solve(C a[16],const C* b,unsigned R,bool inverse,C* x,C& det)const{
        unsigned perm[4]={0,1,2,3},st=0,swaps=0;
        for(unsigned k=0;k<4&&!st;++k){unsigned piv=k;for(unsigned r=k+1;r<4;++r)if(magnitude_compare(mag(a[r*4+k]),mag(a[piv*4+k]))>0)piv=r;
            if(!a[piv*4+k].re.sign&&!a[piv*4+k].im.sign){st=k+1;break;}
            if(piv!=k){for(unsigned c=0;c<4;++c)std::swap(a[k*4+c],a[piv*4+c]);std::swap(perm[k],perm[piv]);++swaps;}
            for(unsigned r=k+1;r<4;++r){C l=div(a[r*4+k],a[k*4+k]);a[r*4+k]=l;for(unsigned c=k+1;c<4;++c)a[r*4+c]=minus(a[r*4+c],l,a[k*4+c]);}}
        C z={zero<Bits/32>(),zero<Bits/32>()};det=z;
        if(!st){det=a[0];for(unsigned k=1;k<4;++k)det=mul(det,a[k*5]);if(swaps&1)det={negate(det.re),negate(det.im)};}
        for(unsigned j=0;j<R;++j){C y[4];
            for(unsigned i=0;i<4;++i){if(inverse){y[i]=z;if(perm[i]==j){y[i].re.limb[Bits/32-1]=0x80000000u;y[i].re.sign=1;}}else y[i]=b[perm[i]*R+j];}
            if(st){for(unsigned i=0;i<4;++i)x[i*R+j]={zero<Bits/32>(division_by_zero),zero<Bits/32>(division_by_zero)};continue;}
            for(unsigned i=1;i<4;++i)for(unsigned t=0;t<i;++t)y[i]=minus(y[i],a[i*4+t],y[t]);
            for(int i=3;i>=0;--i){for(unsigned t=unsigned(i)+1;t<4;++t)y[i]=minus(y[i],a[i*4+t],y[t]);y[i]=div(y[i],a[i*5]);}
            for(unsigned i=0;i<4;++i)x[i*R+j]=y[i];}
        return st;}
};
template<int Bits> void run(Engine& e,bool fused){
    using C=Complex<Bits/32>;std::mt19937_64 rng(Bits*7+fused);const std::size_t M=96;
    auto rc=[&](int span){return C{reference::random_number<Bits>(rng,span),reference::random_number<Bits>(rng,span)};};
    std::vector<C> A(M*16),B(M*4*3);for(auto& z:A)z=rc(4);for(auto& z:B)z=rc(4);
    C z0={zero<Bits/32>(),zero<Bits/32>()},one=z0;one.re.limb[Bits/32-1]=0x80000000u;one.re.sign=1;
    for(int r=0;r<4;++r)A[16+r*4]=z0;                                   // matrix 1: zero first column -> status 1
    for(int c=0;c<4;++c)A[32+8+c]=A[32+c];                              // matrix 2: duplicate rows -> zero last pivot
    for(std::size_t m=3;m<9;++m){int p[4]={0,1,2,3};std::shuffle(p,p+4,rng);for(int i=0;i<16;++i)A[m*16+i]=z0;for(int r=0;r<4;++r)A[m*16+r*4+p[r]]=one;} // permutations
    Ref<Bits> ref{fused};
    for(int mode=0;mode<2;++mode){Batched4 s;s.count=M;s.fused=fused;s.determinant=true;if(mode)s.inverse=true;else s.rhs=3;unsigned R=mode?4:3;
        std::vector<C> X(M*4*R),det(M);std::vector<std::uint32_t> status(M);
        e.lu4(Bits,s,A.data(),B.data(),X.data(),det.data(),status.data());
        for(std::size_t m=0;m<M;++m){C a[16];for(int i=0;i<16;++i)a[i]=A[m*16+i];C x[16],d;unsigned st=ref.solve(a,&B[m*12],R,mode==1,x,d);
            std::string label=" bits="+std::to_string(Bits)+(fused?" fused":" composed")+(mode?" inverse":" solve")+" matrix="+std::to_string(m);
            require(status[m]==st,"status"+label);require(reference::equal_complex<Bits>(det[m],d),"determinant"+label);
            for(unsigned i=0;i<4*R;++i)require(reference::equal_complex<Bits>(X[m*4*R+i],x[i]),"solution"+label+" entry="+std::to_string(i));}
        require(status[1]==1&&status[2]==4,"expected singular statuses"+std::string(fused?" fused":" composed"));
    }
    std::cout<<Bits<<" bits ("<<(fused?"fused":"composed")<<"): batched 4x4 solve, inverse and determinant match MPFR"<<std::endl;
}
int main(){try{Engine e;for(bool f:{false,true}){run<224>(e,f);run<256>(e,f);run<384>(e,f);run<64>(e,f);run<1024>(e,f);}std::cout<<"All batched 4x4 checks passed."<<std::endl;return 0;}
catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}}
