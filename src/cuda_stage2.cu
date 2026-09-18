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

#define CUDA_CHECK(call) do{cudaError_t e=(call);if(e!=cudaSuccess){std::ostringstream os;os<<"CUDA error "<<cudaGetErrorName(e)<<": "<<cudaGetErrorString(e)<<" at "<<__FILE__<<":"<<__LINE__;throw std::runtime_error(os.str());}}while(0)

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
template<class T> void ac(T*& d,const std::vector<T>& h){CUDA_CHECK(cudaMalloc(&d,h.size()*sizeof(T)));CUDA_CHECK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T> void ao(T*& d,size_t n){CUDA_CHECK(cudaMalloc(&d,n*sizeof(T)));}
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
    CUDA_CHECK(cudaMemcpyToSymbol(C_E,e,sizeof(e)));CUDA_CHECK(cudaMemcpyToSymbol(C_W,w,sizeof(w)));

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
}
