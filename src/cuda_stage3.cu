#include "cuda_stage2.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call) do{cudaError_t _cuda_err=(call);if(_cuda_err!=cudaSuccess){std::ostringstream os;os<<"CUDA error "<<cudaGetErrorName(_cuda_err)<<": "<<cudaGetErrorString(_cuda_err)<<" at "<<__FILE__<<":"<<__LINE__;throw std::runtime_error(os.str());}}while(0)

namespace{
constexpr int Q=19;
constexpr float CS2=1.0f/3.0f;
constexpr float EOS_A=2.0f/49.0f;
constexpr float EOS_B=2.0f/21.0f;
constexpr float EOS_R=1.0f;
constexpr float EOS_OMEGA=0.344f;
constexpr float TCR=0.072922f;

__constant__ int C_E[Q*3];
__constant__ float C_W[Q];
__constant__ int C_OPP[Q];
__constant__ float C_M[Q*Q];
__constant__ float C_MINV[Q*Q];
__constant__ float C_S[Q];

struct Params{
    int nx,ny,nz_geo,n_buffer,nz,n_fluid,top_bc_k;
    float niu,G_int,beta_sc,Tr,G_ads,rho_liq,rho_gas,rho_l_eq,rho_g_eq,rho_dry;
};

__device__ __forceinline__ int wrap_xy(int a,int n){if(a<0)a+=n;else if(a>=n)a-=n;return a;}
__device__ __forceinline__ size_t idx3d(int i,int j,int k,int ny,int nz){return (static_cast<size_t>(i)*ny+j)*nz+k;}

__device__ __forceinline__ float pressure_pr(float rho,float Tr){
    const float T=Tr*TCR;
    const float kappa=0.37464f+1.54226f*EOS_OMEGA-0.26992f*EOS_OMEGA*EOS_OMEGA;
    const float t=1.0f+kappa*(1.0f-sqrtf(T/TCR));
    const float epsT=t*t;
    float r=rho;
    if(r<1.0e-8f)r=1.0e-8f;
    const float upper=0.95f/EOS_B;
    if(r>upper)r=upper;
    float d1=1.0f-EOS_B*r;
    float d2=1.0f+2.0f*EOS_B*r-EOS_B*EOS_B*r*r;
    if(d1<1.0e-6f)d1=1.0e-6f;
    if(d2<1.0e-6f)d2=1.0e-6f;
    return r*EOS_R*T/d1-EOS_A*r*r*epsT/d2;
}

__device__ __forceinline__ float psi_from_eos(float rho,float p,float G){
    float v=2.0f*(p-rho*CS2)/G;
    if(v<0.0f)v=0.0f;
    if(v>1.0e6f)v=1.0e6f;
    return sqrtf(v);
}

__device__ __forceinline__ float feq(int q,float rho,float ux,float uy,float uz){
    const int ex=C_E[q*3],ey=C_E[q*3+1],ez=C_E[q*3+2];
    const float eu=ex*ux+ey*uy+ez*uz;
    const float uu=ux*ux+uy*uy+uz*uz;
    return C_W[q]*rho*(1.0f+3.0f*eu+4.5f*eu*eu-1.5f*uu);
}

__global__ void k_init(Params p,const float* rho0,float* rho,float* psi,float* pressure,
                       float* vx,float* vy,float* vz,float* fx,float* fy,float* fz,float* f,float* F){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=p.n_fluid)return;
    float r=rho0[id];
    float rc=r<1.0e-6f?1.0e-6f:r;
    float pr=pressure_pr(rc,p.Tr), ps=psi_from_eos(rc,pr,p.G_int);
    rho[id]=r;pressure[id]=pr;psi[id]=ps;
    vx[id]=vy[id]=vz[id]=0.0f;
    fx[id]=fy[id]=fz[id]=0.0f;
    for(int q=0;q<Q;++q){float v=feq(q,rc,0,0,0);f[id*Q+q]=v;F[id*Q+q]=v;}
}

__global__ void k_top(Params p,const int3* xyz,float* rho,float* psi,float* pressure,
                      float* vx,float* vy,float* vz,const float* forcez,float* f,float* F,double* bc){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=p.n_fluid)return;
    int3 c=xyz[id];
    if(c.z!=p.top_bc_k)return;
    float rb=p.rho_dry;if(rb<1.0e-6f)rb=1.0e-6f;
    float pb=pressure_pr(rb,p.Tr),psib=psi_from_eos(rb,pb,p.G_int);
    float* fi=f+id*Q;float* Fi=F+id*Q;
    float f0=fi[0],f1=fi[1],f2=fi[2],f3=fi[3],f4=fi[4],f5=fi[5],f7=fi[7],f8=fi[8],f9=fi[9],f10=fi[10],f11=fi[11],f14=fi[14],f15=fi[15],f18=fi[18];
    float outgoing=Fi[5]+Fi[11]+Fi[14]+Fi[15]+Fi[18];
    float sum0=f0+f1+f2+f3+f4+f7+f8+f9+f10;
    float sump=f5+f11+f14+f15+f18;
    float uz=(sum0+2.0f*sump+0.5f*forcez[id])/rb-1.0f;
    float Nx=(f1-f2)+(f7-f8)+(f9-f10);
    float Ny=(f3-f4)+(f7-f8)-(f9-f10);
    float f13=f14-(1.0f/6.0f)*rb*uz-0.5f*Nx;
    float f12=f11-(1.0f/6.0f)*rb*uz+0.5f*Nx;
    float f17=f18-(1.0f/6.0f)*rb*uz-0.5f*Ny;
    float f16=f15-(1.0f/6.0f)*rb*uz+0.5f*Ny;
    float f6=rb-sum0-sump-(f13+f12+f17+f16);
    float incoming=f6+f12+f13+f16+f17;
    bc[c.x*p.ny+c.y]+=double(incoming-outgoing);
    fi[6]=Fi[6]=f6;fi[12]=Fi[12]=f12;fi[13]=Fi[13]=f13;fi[16]=Fi[16]=f16;fi[17]=Fi[17]=f17;
    rho[id]=rb;vx[id]=0;vy[id]=0;vz[id]=uz;pressure[id]=pb;psi[id]=psib;
}

__global__ void k_macro(Params p,const float* f,const float* fx,const float* fy,const float* fz,
                        float* rho,float* vx,float* vy,float* vz){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=p.n_fluid)return;
    float r=0,mx=0,my=0,mz=0;
    for(int q=0;q<Q;++q){float fq=f[id*Q+q];r+=fq;mx+=fq*C_E[q*3];my+=fq*C_E[q*3+1];mz+=fq*C_E[q*3+2];}
    rho[id]=r;
    if(r>1.0e-12f){vx[id]=mx/r+0.5f*fx[id]/r;vy[id]=my/r+0.5f*fy[id]/r;vz[id]=mz/r+0.5f*fz[id]/r;}
    else vx[id]=vy[id]=vz[id]=0;
}

__global__ void k_pressure_psi(Params p,const float* rho,float* pressure,float* psi){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=p.n_fluid)return;
    float pr=pressure_pr(rho[id],p.Tr);pressure[id]=pr;psi[id]=psi_from_eos(rho[id],pr,p.G_int);
}

__global__ void k_force(Params p,const int3* xyz,const int32_t* grid,const int32_t* ff,const int32_t* ads,
                        const float* psi,float* fx,float* fy,float* fz){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=p.n_fluid)return;
    float pc=psi[id];int3 c=xyz[id];
    float s1x=0,s1y=0,s1z=0,s2x=0,s2y=0,s2z=0,asx=0,asy=0,asz=0;
    for(int q=1;q<Q;++q){
        int dx=C_E[q*3],dy=C_E[q*3+1],dz=C_E[q*3+2];
        int nb=ff[id*Q+q];
        float pn=pc;
        if(nb>=0)pn=psi[nb];
        else if(nb==-1){
            int gi=wrap_xy(c.x+dx,p.nx),gj=wrap_xy(c.y+dy,p.ny),gk=c.z;
            int ghost=grid[idx3d(gi,gj,gk,p.ny,p.nz)];
            if(ghost>=0)pn=psi[ghost];
        }
        float coef=3.0f*C_W[q],p2=pn*pn;
        s1x+=coef*pn*dx;s1y+=coef*pn*dy;s1z+=coef*pn*dz;
        s2x+=coef*p2*dx;s2y+=coef*p2*dy;s2z+=coef*p2*dz;
        if(ads[id*Q+q]==1){asx+=coef*dx;asy+=coef*dy;asz+=coef*dz;}
    }
    fx[id]=-p.beta_sc*p.G_int*pc*s1x-0.5f*(1.0f-p.beta_sc)*p.G_int*s2x-p.G_ads*pc*asx;
    fy[id]=-p.beta_sc*p.G_int*pc*s1y-0.5f*(1.0f-p.beta_sc)*p.G_int*s2y-p.G_ads*pc*asy;
    fz[id]=-p.beta_sc*p.G_int*pc*s1z-0.5f*(1.0f-p.beta_sc)*p.G_int*s2z-p.G_ads*pc*asz;
}


__global__ void k_collision(Params p,const float* rho,const float* fx,const float* fy,const float* fz,const float* f,float* F){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=p.n_fluid)return;
    const float r=rho[id];
    float mx=0.0f,my=0.0f,mz=0.0f;
    #pragma unroll
    for(int q=0;q<Q;++q){
        float fq=f[id*Q+q];
        mx+=fq*C_E[q*3];my+=fq*C_E[q*3+1];mz+=fq*C_E[q*3+2];
    }
    float ux=0.0f,uy=0.0f,uz=0.0f,dux=0.0f,duy=0.0f,duz=0.0f;
    if(r>1.0e-12f){
        ux=mx/r;uy=my/r;uz=mz/r;
        dux=fx[id]/r;duy=fy[id]/r;duz=fz[id]/r;
    }
    float feq_u[Q],feq_ud[Q],m[Q],meq[Q],mstar[Q];
    #pragma unroll
    for(int q=0;q<Q;++q){
        feq_u[q]=feq(q,r,ux,uy,uz);
        feq_ud[q]=feq(q,r,ux+dux,uy+duy,uz+duz);
    }
    #pragma unroll
    for(int i=0;i<Q;++i){
        float a=0.0f,b=0.0f;
        #pragma unroll
        for(int j=0;j<Q;++j){
            a+=C_M[i*Q+j]*f[id*Q+j];
            b+=C_M[i*Q+j]*feq_u[j];
        }
        m[i]=a;meq[i]=b;
        mstar[i]=a-C_S[i]*(a-b);
    }
    #pragma unroll
    for(int i=0;i<Q;++i){
        float fs=0.0f;
        #pragma unroll
        for(int j=0;j<Q;++j) fs+=C_MINV[i*Q+j]*mstar[j];
        F[id*Q+i]=fs+(feq_ud[i]-feq_u[i]);
    }
}

__global__ void k_stream(Params p,const int32_t* pull,const float* F,float* f){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=p.n_fluid)return;
    #pragma unroll
    for(int q=0;q<Q;++q){
        int src=pull[id*Q+q];
        f[id*Q+q]=(src>=0)?F[src*Q+q]:F[id*Q+C_OPP[q]];
    }
}

static void build_mrt_matrices(float niu,float* mout,float* minv,float* sout){
    const float base[Q][Q]={
{1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f},
{-30.0f,-11.0f,-11.0f,-11.0f,-11.0f,-11.0f,-11.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f},
{12.0f,-4.0f,-4.0f,-4.0f,-4.0f,-4.0f,-4.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f},
{0.0f,1.0f,-1.0f,0.0f,0.0f,0.0f,0.0f,1.0f,-1.0f,1.0f,-1.0f,1.0f,-1.0f,1.0f,-1.0f,0.0f,0.0f,0.0f,0.0f},
{0.0f,-4.0f,4.0f,0.0f,0.0f,0.0f,0.0f,1.0f,-1.0f,1.0f,-1.0f,1.0f,-1.0f,1.0f,-1.0f,0.0f,0.0f,0.0f,0.0f},
{0.0f,0.0f,0.0f,1.0f,-1.0f,0.0f,0.0f,1.0f,1.0f,-1.0f,-1.0f,0.0f,0.0f,0.0f,0.0f,1.0f,-1.0f,1.0f,-1.0f},
{0.0f,0.0f,0.0f,-4.0f,4.0f,0.0f,0.0f,1.0f,1.0f,-1.0f,-1.0f,0.0f,0.0f,0.0f,0.0f,1.0f,-1.0f,1.0f,-1.0f},
{0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,-1.0f,0.0f,0.0f,0.0f,0.0f,1.0f,1.0f,-1.0f,-1.0f,1.0f,1.0f,-1.0f,-1.0f},
{0.0f,0.0f,0.0f,0.0f,0.0f,-4.0f,4.0f,0.0f,0.0f,0.0f,0.0f,1.0f,1.0f,-1.0f,-1.0f,1.0f,1.0f,-1.0f,-1.0f},
{0.0f,2.0f,2.0f,-1.0f,-1.0f,-1.0f,-1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,-2.0f,-2.0f,-2.0f,-2.0f},
{0.0f,-4.0f,-4.0f,2.0f,2.0f,2.0f,2.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,-2.0f,-2.0f,-2.0f,-2.0f},
{0.0f,0.0f,0.0f,1.0f,1.0f,-1.0f,-1.0f,1.0f,1.0f,1.0f,1.0f,-1.0f,-1.0f,-1.0f,-1.0f,0.0f,0.0f,0.0f,0.0f},
{0.0f,0.0f,0.0f,-2.0f,-2.0f,2.0f,2.0f,1.0f,1.0f,1.0f,1.0f,-1.0f,-1.0f,-1.0f,-1.0f,0.0f,0.0f,0.0f,0.0f},
{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,-1.0f,-1.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,-1.0f,-1.0f,1.0f},
{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,-1.0f,-1.0f,1.0f,0.0f,0.0f,0.0f,0.0f},
{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,-1.0f,1.0f,-1.0f,-1.0f,1.0f,-1.0f,1.0f,0.0f,0.0f,0.0f,0.0f},
{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,-1.0f,-1.0f,1.0f,1.0f,0.0f,0.0f,0.0f,0.0f,1.0f,-1.0f,1.0f,-1.0f},
{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,1.0f,-1.0f,-1.0f,-1.0f,-1.0f,1.0f,1.0f}};
    for(int i=0;i<Q;++i)for(int j=0;j<Q;++j)mout[i*Q+j]=base[i][j];
    const int swaps[3][2]={{8,10},{12,14},{16,18}};
    for(auto &sp:swaps)for(int i=0;i<Q;++i)std::swap(mout[i*Q+sp[0]],mout[i*Q+sp[1]]);
    double aug[Q][2*Q]{};
    for(int i=0;i<Q;++i)for(int j=0;j<Q;++j){aug[i][j]=mout[i*Q+j];aug[i][Q+j]=(i==j)?1.0:0.0;}
    for(int col=0;col<Q;++col){
        int piv=col;for(int r=col+1;r<Q;++r)if(std::abs(aug[r][col])>std::abs(aug[piv][col]))piv=r;
        if(piv!=col)for(int j=0;j<2*Q;++j)std::swap(aug[col][j],aug[piv][j]);
        double d=aug[col][col];for(int j=0;j<2*Q;++j)aug[col][j]/=d;
        for(int r=0;r<Q;++r)if(r!=col){double a=aug[r][col];for(int j=0;j<2*Q;++j)aug[r][j]-=a*aug[col][j];}
    }
    for(int i=0;i<Q;++i)for(int j=0;j<Q;++j)minv[i*Q+j]=static_cast<float>(aug[i][Q+j]);
    float sv=1.0f/(3.0f*niu+0.5f);
    const float st[Q]={0.0f,1.19f,1.4f,0.0f,1.2f,0.0f,1.2f,0.0f,1.2f,sv,1.4f,sv,1.4f,sv,sv,sv,1.98f,1.98f,1.98f};
    for(int i=0;i<Q;++i)sout[i]=st[i];
}

float host_pressure(float rho,float Tr){
    float T=Tr*TCR,k=0.37464f+1.54226f*EOS_OMEGA-0.26992f*EOS_OMEGA*EOS_OMEGA;
    float t=1.0f+k*(1.0f-std::sqrt(T/TCR)),eps=t*t,r=rho;
    if(r<1e-8f)r=1e-8f;if(r>0.95f/EOS_B)r=0.95f/EOS_B;
    float d1=std::max(1e-6f,1.0f-EOS_B*r),d2=std::max(1e-6f,1.0f+2.0f*EOS_B*r-EOS_B*EOS_B*r*r);
    return r*EOS_R*T/d1-EOS_A*r*r*eps/d2;
}
float host_psi(float rho,float p,float G){float v=2.0f*(p-rho*CS2)/G;v=std::max(0.0f,std::min(1.0e6f,v));return std::sqrt(v);}

struct DM{
    int3* xyz=nullptr;int32_t *grid=nullptr,*pull=nullptr,*solidnb=nullptr,*ff=nullptr,*ads=nullptr;
    float *rho0=nullptr,*rho=nullptr,*psi=nullptr,*pressure=nullptr,*vx=nullptr,*vy=nullptr,*vz=nullptr,*fx=nullptr,*fy=nullptr,*fz=nullptr,*f=nullptr,*F=nullptr;
    double* bc=nullptr;
    ~DM(){cudaFree(xyz);cudaFree(grid);cudaFree(pull);cudaFree(solidnb);cudaFree(ff);cudaFree(ads);cudaFree(rho0);cudaFree(rho);cudaFree(psi);cudaFree(pressure);cudaFree(vx);cudaFree(vy);cudaFree(vz);cudaFree(fx);cudaFree(fy);cudaFree(fz);cudaFree(f);cudaFree(F);cudaFree(bc);}
};
template<class T> void ac(T*& d,const std::vector<T>& h){CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d),h.size()*sizeof(T)));CUDA_CHECK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T> void ao(T*& d,size_t n){CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d),n*sizeof(T)));}
}

void run_cuda_stage2(
    int nx,int ny,int nz_geo,int n_buffer,
    float niu,float G_int,float beta_sc,float Tr,float G_ads,
    float rho_liq,float rho_gas,float rho_l_eq,float rho_g_eq,float rho_dry,
    const std::vector<std::array<int32_t,3>>& fluid_xyz,
    const std::vector<int32_t>& grid_to_idx,
    const std::vector<int32_t>& pull_nb,
    const std::vector<int32_t>& is_solid_nb,
    const std::vector<int32_t>& ff_nb,
    const std::vector<int32_t>& ads_solid_nb,
    const std::vector<float>& rho_init)
{
    int dev=-1;CUDA_CHECK(cudaGetDevice(&dev));cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    std::cout<<"[CUDA] device = "<<prop.name<<" | compute capability "<<prop.major<<"."<<prop.minor<<"\n";
    const int e[Q*3]={0,0,0,1,0,0,-1,0,0,0,1,0,0,-1,0,0,0,1,0,0,-1,1,1,0,-1,-1,0,1,-1,0,-1,1,0,1,0,1,-1,0,-1,1,0,-1,-1,0,1,0,1,1,0,-1,-1,0,1,-1,0,-1,1};
    const float w[Q]={1.0f/3.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f};
    const int opp[Q]={0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17};
    float hm[Q*Q],hmi[Q*Q],hs[Q];build_mrt_matrices(niu,hm,hmi,hs);
    CUDA_CHECK(cudaMemcpyToSymbol(C_E,e,sizeof(e)));CUDA_CHECK(cudaMemcpyToSymbol(C_W,w,sizeof(w)));
    CUDA_CHECK(cudaMemcpyToSymbol(C_OPP,opp,sizeof(opp)));CUDA_CHECK(cudaMemcpyToSymbol(C_M,hm,sizeof(hm)));
    CUDA_CHECK(cudaMemcpyToSymbol(C_MINV,hmi,sizeof(hmi)));CUDA_CHECK(cudaMemcpyToSymbol(C_S,hs,sizeof(hs)));

    Params p{};p.nx=nx;p.ny=ny;p.nz_geo=nz_geo;p.n_buffer=n_buffer;p.nz=nz_geo+n_buffer;p.n_fluid=(int)fluid_xyz.size();p.top_bc_k=p.nz-1;
    p.niu=niu;p.G_int=G_int;p.beta_sc=beta_sc;p.Tr=Tr;p.G_ads=G_ads;p.rho_liq=rho_liq;p.rho_gas=rho_gas;p.rho_l_eq=rho_l_eq;p.rho_g_eq=rho_g_eq;p.rho_dry=rho_dry;

    std::vector<int3> xyz(fluid_xyz.size());for(size_t i=0;i<xyz.size();++i)xyz[i]=make_int3(fluid_xyz[i][0],fluid_xyz[i][1],fluid_xyz[i][2]);
    DM d;ac(d.xyz,xyz);ac(d.grid,grid_to_idx);ac(d.pull,pull_nb);ac(d.solidnb,is_solid_nb);ac(d.ff,ff_nb);ac(d.ads,ads_solid_nb);ac(d.rho0,rho_init);
    size_t n=fluid_xyz.size(),nq=n*Q;ao(d.rho,n);ao(d.psi,n);ao(d.pressure,n);ao(d.vx,n);ao(d.vy,n);ao(d.vz,n);ao(d.fx,n);ao(d.fy,n);ao(d.fz,n);ao(d.f,nq);ao(d.F,nq);ao(d.bc,(size_t)nx*ny);CUDA_CHECK(cudaMemset(d.bc,0,(size_t)nx*ny*sizeof(double)));

    constexpr int BS=256;int GS=(p.n_fluid+BS-1)/BS;
    k_init<<<GS,BS>>>(p,d.rho0,d.rho,d.psi,d.pressure,d.vx,d.vy,d.vz,d.fx,d.fy,d.fz,d.f,d.F);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    k_top<<<GS,BS>>>(p,d.xyz,d.rho,d.psi,d.pressure,d.vx,d.vy,d.vz,d.fz,d.f,d.F,d.bc);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    k_macro<<<GS,BS>>>(p,d.f,d.fx,d.fy,d.fz,d.rho,d.vx,d.vy,d.vz);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    k_pressure_psi<<<GS,BS>>>(p,d.rho,d.pressure,d.psi);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(d.bc,0,(size_t)nx*ny*sizeof(double)));

    std::vector<float> rho(n),psi(n),pressure(n);CUDA_CHECK(cudaMemcpy(rho.data(),d.rho,n*sizeof(float),cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(psi.data(),d.psi,n*sizeof(float),cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(pressure.data(),d.pressure,n*sizeof(float),cudaMemcpyDeviceToHost));
    double maxP=0,maxPsi=0,msoil=0,mbuffer=0,veq=0;size_t nliq=0;float rmin=std::numeric_limits<float>::max(),rmax=-std::numeric_limits<float>::max();float rth=0.5f*(rho_liq+rho_gas);
    for(size_t i=0;i<n;++i){
        float hp=host_pressure(rho[i],Tr),hs=host_psi(rho[i],hp,G_int);maxP=std::max(maxP,std::abs(double(pressure[i])-hp));maxPsi=std::max(maxPsi,std::abs(double(psi[i])-hs));rmin=std::min(rmin,rho[i]);rmax=std::max(rmax,rho[i]);
        if(fluid_xyz[i][2]<nz_geo){msoil+=rho[i];if(rho[i]>rth)++nliq;double a=(double(rho[i])-rho_g_eq)/(double(rho_l_eq)-rho_g_eq);a=std::max(0.0,std::min(1.0,a));veq+=a;}else mbuffer+=rho[i];
    }
    size_t nreal=n-(size_t)nx*ny*n_buffer;double satth=double(nliq)/std::max<size_t>(1,nreal),sateq=veq/std::max<size_t>(1,nreal);
    std::cout<<std::fixed<<std::setprecision(6)<<"[CUDA Init] Msoil="<<msoil<<" Mbuffer="<<mbuffer<<" Mdomain="<<(msoil+mbuffer)<<" | Sat_th="<<satth<<" Sat_eq="<<sateq<<" | rho=["<<rmin<<","<<rmax<<"]\n";
    std::cout<<std::scientific<<std::setprecision(6)<<"[CUDA Check] max|P_gpu-P_cpu|="<<maxP<<" max|psi_gpu-psi_cpu|="<<maxPsi<<"\n";
    if(maxP>2e-6||maxPsi>2e-6)throw std::runtime_error("CUDA initialization/EOS validation exceeded tolerance.");

    k_force<<<GS,BS>>>(p,d.xyz,d.grid,d.ff,d.ads,d.psi,d.fx,d.fy,d.fz);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> fx(n),fy(n),fz(n);CUDA_CHECK(cudaMemcpy(fx.data(),d.fx,n*sizeof(float),cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(fy.data(),d.fy,n*sizeof(float),cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(fz.data(),d.fz,n*sizeof(float),cudaMemcpyDeviceToHost));
    double maxF=0;float minx=std::numeric_limits<float>::max(),maxx=-minx,miny=minx,maxy=-minx,minz=minx,maxz=-minx;
    for(size_t i=0;i<n;++i){maxF=std::max(maxF,std::sqrt(double(fx[i])*fx[i]+double(fy[i])*fy[i]+double(fz[i])*fz[i]));minx=std::min(minx,fx[i]);maxx=std::max(maxx,fx[i]);miny=std::min(miny,fy[i]);maxy=std::max(maxy,fy[i]);minz=std::min(minz,fz[i]);maxz=std::max(maxz,fz[i]);}
    std::cout<<"[CUDA Force] Fx=["<<minx<<","<<maxx<<"] Fy=["<<miny<<","<<maxy<<"] Fz=["<<minz<<","<<maxz<<"] max|F|="<<maxF<<"\n";
    std::cout<<"[CUDA PASS] initialization, PR-EOS/psi, fluid-fluid force and adsorption force kernels executed.\n";

    const double msoil0=msoil,mbuffer0=mbuffer,mdomain0=msoil+mbuffer,liquid_mass0=rho_l_eq*double(nreal);
    const int validation_steps=100;
    std::cout<<"[CUDA Stage-3] Running "<<validation_steps<<" full MRT+EDM/streaming steps...\n";
    for(int step=1;step<=validation_steps;++step){
        k_macro<<<GS,BS>>>(p,d.f,d.fx,d.fy,d.fz,d.rho,d.vx,d.vy,d.vz);
        k_pressure_psi<<<GS,BS>>>(p,d.rho,d.pressure,d.psi);
        k_force<<<GS,BS>>>(p,d.xyz,d.grid,d.ff,d.ads,d.psi,d.fx,d.fy,d.fz);
        k_collision<<<GS,BS>>>(p,d.rho,d.fx,d.fy,d.fz,d.f,d.F);
        k_stream<<<GS,BS>>>(p,d.pull,d.F,d.f);
        k_top<<<GS,BS>>>(p,d.xyz,d.rho,d.psi,d.pressure,d.vx,d.vy,d.vz,d.fz,d.f,d.F,d.bc);
        k_macro<<<GS,BS>>>(p,d.f,d.fx,d.fy,d.fz,d.rho,d.vx,d.vy,d.vz);
        k_pressure_psi<<<GS,BS>>>(p,d.rho,d.pressure,d.psi);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(rho.data(),d.rho,n*sizeof(float),cudaMemcpyDeviceToHost));
    std::vector<double> bcxy((size_t)nx*ny);CUDA_CHECK(cudaMemcpy(bcxy.data(),d.bc,bcxy.size()*sizeof(double),cudaMemcpyDeviceToHost));
    double ms=0,mb=0,ve=0,bcchg=0;size_t nl=0;float rrmin=std::numeric_limits<float>::max(),rrmax=-std::numeric_limits<float>::max();
    for(size_t i=0;i<n;++i){
        rrmin=std::min(rrmin,rho[i]);rrmax=std::max(rrmax,rho[i]);
        if(fluid_xyz[i][2]<nz_geo){
            ms+=rho[i];if(rho[i]>rth)++nl;
            double a=(double(rho[i])-rho_g_eq)/(double(rho_l_eq)-rho_g_eq);a=std::max(0.0,std::min(1.0,a));ve+=a;
        }else mb+=rho[i];
    }
    for(double x:bcxy)bcchg+=x;
    double sat=ve/std::max<size_t>(1,nreal),lm=rho_l_eq*ve,area=double(nx)*ny;
    double er=(liquid_mass0-lm)/(validation_steps*area);
    double js=(msoil0-ms)/(validation_steps*area);
    double topout=-bcchg,jtop=topout/(validation_steps*area);
    double dl=mdomain0-(ms+mb);
    double bcrel=std::abs(dl-topout)/std::max({std::abs(dl),std::abs(topout),1e-30});
    std::cout<<std::fixed<<std::setprecision(6)
             <<"[Stage-3 100] Sat_eq="<<sat
             <<" | ERliq="<<std::scientific<<std::setprecision(6)<<er
             <<" | Jsoil="<<js<<" | Jtop="<<jtop
             <<" | bc_rel="<<std::scientific<<std::setprecision(3)<<bcrel
             <<" | rho=["<<std::fixed<<std::setprecision(5)<<rrmin<<","<<rrmax<<"]\n";
    std::cout<<"[Reference 100] Sat_eq=0.976922 | ERliq=9.194102e-02 | Jsoil=1.473787e-02 | Jtop=-4.470767e-03 | bc_rel=3.182e-05 | rho=[0.37000,7.58782]\n";
}
