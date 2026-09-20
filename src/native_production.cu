#include "native_production.hpp"
#include "trial_guard.hpp"
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
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <numeric>

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
        else if(nb==-1 && ads[id*Q+q]==1){
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


namespace fs = std::filesystem;

namespace{
struct HostState{
    double mass_soil=0.0,mass_buffer=0.0,mass_domain=0.0;
    double sat_threshold=0.0,sat_equiv=0.0,liquid_volume=0.0,liquid_mass=0.0;
    double rho_min=0.0,rho_max=0.0,top_outflow_cum=0.0;
};

struct CheckpointHeader{
    char magic[8];
    uint32_t version;
    int32_t nx,ny,nz_geo,n_buffer;
    uint64_t n_fluid;
    int32_t step,prev_record_step;
    double wall_time_sec;
    HostState s0,prev;
    uint32_t signature_len;
};

static fs::path p8(const std::string& s){
#ifdef _WIN32
    return fs::u8path(s);
#else
    return fs::path(s);
#endif
}

template<class T>
static void write_vec(std::ofstream& out,const std::vector<T>& v){
    uint64_t n=(uint64_t)v.size();
    out.write(reinterpret_cast<const char*>(&n),sizeof(n));
    if(n)out.write(reinterpret_cast<const char*>(v.data()),(std::streamsize)(n*sizeof(T)));
}
template<class T>
static void read_vec(std::ifstream& in,std::vector<T>& v,size_t expected){
    uint64_t n=0;in.read(reinterpret_cast<char*>(&n),sizeof(n));
    if(n!=expected)throw std::runtime_error("Checkpoint vector size mismatch.");
    v.resize((size_t)n);
    if(n)in.read(reinterpret_cast<char*>(v.data()),(std::streamsize)(n*sizeof(T)));
    if(!in)throw std::runtime_error("Checkpoint is truncated.");
}

static void prune_ckpt(const fs::path& dir,const std::string& prefix,int keep){
    std::vector<fs::path> v;
    if(!fs::exists(dir))return;
    for(auto& e:fs::directory_iterator(dir)){
        if(!e.is_regular_file())continue;
        auto n=e.path().filename().string();
        if(n.rfind(prefix,0)==0 && e.path().extension()==".bin")v.push_back(e.path());
    }
    std::sort(v.begin(),v.end());
    while((int)v.size()>keep){std::error_code ec;fs::remove(v.front(),ec);v.erase(v.begin());}
}

static std::vector<fs::path> checkpoint_candidates(const fs::path& dir,const std::string& prefix){
    std::vector<fs::path> v;
    if(!fs::exists(dir))return v;
    for(auto& e:fs::directory_iterator(dir)){
        if(!e.is_regular_file())continue;
        auto n=e.path().filename().string();
        if(n.rfind(prefix,0)==0 && e.path().extension()==".bin")v.push_back(e.path());
    }
    std::sort(v.rbegin(),v.rend());
    return v;
}

static void truncate_stats_to_step(const fs::path& p,int step){
    if(!fs::exists(p))return;
    std::ifstream in(p,std::ios::binary);
    std::string line;std::vector<std::string> keep;
    if(std::getline(in,line))keep.push_back(line);
    while(std::getline(in,line)){
        std::stringstream ss(line);std::string cell;int col=0,it=-1;
        while(std::getline(ss,cell,',')){if(col++==3){try{it=std::stoi(cell);}catch(...){it=-1;}break;}}
        if(it>=0 && it<=step)keep.push_back(line);
    }
    std::ofstream out(p,std::ios::binary|std::ios::trunc);
    for(auto&s:keep)out<<s<<"\n";
}

static void write_vtk_legacy(
    const fs::path& path,int nx,int ny,int nz,
    const std::vector<uint8_t>& solid,const std::vector<uint8_t>& buffer,
    const std::vector<int32_t>& grid,const std::vector<float>& rho,
    const std::vector<float>& psi,const std::vector<float>& pressure,
    const std::vector<float>& vx,const std::vector<float>& vy,const std::vector<float>& vz)
{
    // Portable legacy ASCII VTK. It is intentionally simple and requires no external library.
    std::ofstream out(path);
    if(!out)throw std::runtime_error("Cannot write VTK: "+path.string());
    const uint64_t n=(uint64_t)nx*ny*nz;
    out<<"# vtk DataFile Version 3.0\nSoilEvaporationLBM native\nASCII\n";
    out<<"DATASET STRUCTURED_POINTS\nDIMENSIONS "<<nx+1<<" "<<ny+1<<" "<<nz+1<<"\n";
    out<<"ORIGIN 0 0 0\nSPACING 1 1 1\nCELL_DATA "<<n<<"\n";
    auto dense_index=[=](int i,int j,int k){return ((size_t)i*ny+j)*nz+k;};
    out<<"SCALARS Solid unsigned_char 1\nLOOKUP_TABLE default\n";
    for(int k=0;k<nz;++k)for(int j=0;j<ny;++j)for(int i=0;i<nx;++i)out<<(int)solid[dense_index(i,j,k)]<<"\n";
    out<<"SCALARS Wall unsigned_char 1\nLOOKUP_TABLE default\n";
    static const int ve[18][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},{1,1,0},{-1,-1,0},{1,-1,0},{-1,1,0},{1,0,1},{-1,0,-1},{1,0,-1},{-1,0,1},{0,1,1},{0,-1,-1},{0,1,-1},{0,-1,1}};
    for(int k=0;k<nz;++k)for(int j=0;j<ny;++j)for(int i=0;i<nx;++i){
        uint8_t wall=0;
        if(solid[dense_index(i,j,k)]){
            for(const auto& d:ve){
                int ii=i+d[0],jj=j+d[1],kk=k+d[2];
                if(ii>=0&&ii<nx&&jj>=0&&jj<ny&&kk>=0&&kk<nz&&!solid[dense_index(ii,jj,kk)]){wall=1;break;}
            }
        }
        out<<(int)wall<<"\n";
    }
    out<<"SCALARS BufferMask unsigned_char 1\nLOOKUP_TABLE default\n";
    for(int k=0;k<nz;++k)for(int j=0;j<ny;++j)for(int i=0;i<nx;++i)out<<(int)buffer[dense_index(i,j,k)]<<"\n";
    auto write_scalar=[&](const char* name,const std::vector<float>& a){
        out<<"SCALARS "<<name<<" float 1\nLOOKUP_TABLE default\n";
        out<<std::setprecision(8);
        for(int k=0;k<nz;++k)for(int j=0;j<ny;++j)for(int i=0;i<nx;++i){
            int id=grid[dense_index(i,j,k)];out<<(id>=0?a[(size_t)id]:0.0f)<<"\n";
        }
    };
    write_scalar("rho",rho);write_scalar("psi",psi);write_scalar("pressure",pressure);
    out<<"SCALARS vel_mag float 1\nLOOKUP_TABLE default\n"<<std::setprecision(8);
    for(int k=0;k<nz;++k)for(int j=0;j<ny;++j)for(int i=0;i<nx;++i){
        int id=grid[dense_index(i,j,k)];
        if(id>=0)out<<std::sqrt(vx[(size_t)id]*vx[(size_t)id]+vy[(size_t)id]*vy[(size_t)id]+vz[(size_t)id]*vz[(size_t)id])<<"\n";
        else out<<"0\n";
    }
    out<<"VECTORS velocity float\n"<<std::setprecision(8);
    for(int k=0;k<nz;++k)for(int j=0;j<ny;++j)for(int i=0;i<nx;++i){
        int id=grid[dense_index(i,j,k)];
        if(id>=0)out<<vx[(size_t)id]<<" "<<vy[(size_t)id]<<" "<<vz[(size_t)id]<<"\n";
        else out<<"0 0 0\n";
    }
}

static void write_done_json(const fs::path& p,const NativeRunOptions& opt,const NativeRunResult& r){
    std::ofstream out(p);
    out<<"{\n";
    out<<"  \"status\":\"DONE\",\n";
    out<<"  \"schema\":\"soil_isothermal_evaporation_native_v1\",\n";
    out<<"  \"signature\":\""<<opt.signature<<"\",\n";
    out<<"  \"case_name\":\""<<opt.case_name<<"\",\n";
    out<<"  \"total_steps\":"<<opt.total_steps<<",\n";
    out<<"  \"final_saturation_equiv\":"<<std::setprecision(17)<<r.final_saturation_equiv<<",\n";
    out<<"  \"final_ER_liquid_equiv_lu\":"<<r.final_ER_liquid_equiv_lu<<",\n";
    out<<"  \"final_J_soil_lu\":"<<r.final_J_soil_lu<<",\n";
    out<<"  \"final_J_out_mass_lu\":"<<r.final_J_out_mass_lu<<",\n";
    out<<"  \"final_J_top_direct_lu\":"<<r.final_J_top_direct_lu<<",\n";
    out<<"  \"final_mass_partition_error_rel\":"<<r.final_mass_partition_error_rel<<",\n";
    out<<"  \"final_bc_mass_balance_error_rel\":"<<r.final_bc_mass_balance_error_rel<<",\n";
    out<<"  \"rho_min_final\":"<<r.rho_min_final<<",\n";
    out<<"  \"rho_max_final\":"<<r.rho_max_final<<",\n";
    out<<"  \"wall_time_sec\":"<<r.wall_time_sec<<",\n";
    out<<"  \"stats_csv\":\""<<r.stats_csv<<"\"\n";
    out<<"}\n";
}
}

NativeRunResult run_native_production(
    int nx,int ny,int nz_geo,int n_buffer,
    float niu,float G_int,float beta_sc,float Tr,float G_ads,
    float rho_liq,float rho_gas,float rho_l_eq,float rho_g_eq,float rho_dry,
    const std::vector<std::array<int32_t,3>>& fluid_xyz,
    const std::vector<int32_t>& grid_to_idx,
    const std::vector<int32_t>& pull_nb,
    const std::vector<int32_t>& ff_nb,
    const std::vector<int32_t>& ads_solid_nb,
    const std::vector<float>& rho_init,
    const std::vector<uint8_t>& solid_dense,
    const std::vector<uint8_t>& buffer_dense,
    const NativeRunOptions& opt)
{
    if(opt.total_steps<0||opt.record_interval<=0||opt.print_interval<=0)throw std::runtime_error("Invalid RUN intervals.");
    if(opt.print_interval%opt.record_interval!=0)throw std::runtime_error("print_interval must be a multiple of record_interval.");
    if(opt.checkpoint_interval<0 || (opt.checkpoint_interval>0 && opt.checkpoint_interval%opt.record_interval!=0))throw std::runtime_error("checkpoint_interval must be 0 or a multiple of record_interval.");
    if(opt.keep_checkpoints<2)throw std::runtime_error("keep_checkpoints must be >=2.");
    if(rho_l_eq<=rho_g_eq)throw std::runtime_error("rho_l_eq must be greater than rho_g_eq.");

    int dev=-1;CUDA_CHECK(cudaGetDevice(&dev));cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    if(opt.verbose) std::cout<<"[CUDA] device="<<prop.name<<" | compute capability "<<prop.major<<"."<<prop.minor
             <<" | VRAM="<<(prop.totalGlobalMem/(1024ull*1024ull))<<" MB\n";
    const bool supported_cc=(prop.major==7&&prop.minor==5)||(prop.major==8&&(prop.minor==0||prop.minor==6||prop.minor==9))||(prop.major==9&&prop.minor==0)||(prop.major==12&&prop.minor==0);
    if(!supported_cc){
        if(opt.verbose) std::cout<<"[CUDA][Warn] This release embeds native code for sm_75/sm_80/sm_86/sm_89/sm_90/sm_120. Unsupported GPUs may fail to launch.\n";
    }

    const int e[Q*3]={0,0,0,1,0,0,-1,0,0,0,1,0,0,-1,0,0,0,1,0,0,-1,1,1,0,-1,-1,0,1,-1,0,-1,1,0,1,0,1,-1,0,-1,1,0,-1,-1,0,1,0,1,1,0,-1,-1,0,1,-1,0,-1,1};
    const float w[Q]={1.0f/3.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/18.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f,1.0f/36.0f};
    const int opp[Q]={0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17};
    float hm[Q*Q],hmi[Q*Q],hs[Q];build_mrt_matrices(niu,hm,hmi,hs);
    CUDA_CHECK(cudaMemcpyToSymbol(C_E,e,sizeof(e)));CUDA_CHECK(cudaMemcpyToSymbol(C_W,w,sizeof(w)));
    CUDA_CHECK(cudaMemcpyToSymbol(C_OPP,opp,sizeof(opp)));CUDA_CHECK(cudaMemcpyToSymbol(C_M,hm,sizeof(hm)));
    CUDA_CHECK(cudaMemcpyToSymbol(C_MINV,hmi,sizeof(hmi)));CUDA_CHECK(cudaMemcpyToSymbol(C_S,hs,sizeof(hs)));

    Params p{};p.nx=nx;p.ny=ny;p.nz_geo=nz_geo;p.n_buffer=n_buffer;p.nz=nz_geo+n_buffer;p.n_fluid=(int)fluid_xyz.size();p.top_bc_k=p.nz-1;
    p.niu=niu;p.G_int=G_int;p.beta_sc=beta_sc;p.Tr=Tr;p.G_ads=G_ads;p.rho_liq=rho_liq;p.rho_gas=rho_gas;p.rho_l_eq=rho_l_eq;p.rho_g_eq=rho_g_eq;p.rho_dry=rho_dry;
    const size_t n=fluid_xyz.size(),nq=n*Q;
    const size_t nreal=n-(size_t)nx*ny*n_buffer;
    const float rho_threshold=0.5f*(rho_liq+rho_gas);
    const double area=double(nx)*ny;

    std::vector<int3> xyz(n);for(size_t i=0;i<n;++i)xyz[i]=make_int3(fluid_xyz[i][0],fluid_xyz[i][1],fluid_xyz[i][2]);
    std::vector<int32_t> dummy_solid(nq,0);
    DM d;ac(d.xyz,xyz);ac(d.grid,grid_to_idx);ac(d.pull,pull_nb);ac(d.solidnb,dummy_solid);ac(d.ff,ff_nb);ac(d.ads,ads_solid_nb);ac(d.rho0,rho_init);
    ao(d.rho,n);ao(d.psi,n);ao(d.pressure,n);ao(d.vx,n);ao(d.vy,n);ao(d.vz,n);ao(d.fx,n);ao(d.fy,n);ao(d.fz,n);ao(d.f,nq);ao(d.F,nq);ao(d.bc,(size_t)nx*ny);
    constexpr int BS=256;const int GS=(p.n_fluid+BS-1)/BS;

    fs::path outdir=p8(opt.output_dir);fs::create_directories(outdir);
    fs::path stats_tmp=outdir/"stats_in_progress.csv",stats_final=outdir/"stats.csv";
    const std::string ckprefix=opt.output_prefix+"checkpoint_";

    std::vector<float> hrho(n),hpsi(n),hp(n),hvx(n),hvy(n),hvz(n),hfx(n),hfy(n),hfz(n),hf(nq),hF(nq);
    std::vector<double> hbc((size_t)nx*ny);

    auto collect=[&]()->HostState{
        CUDA_CHECK(cudaMemcpy(hrho.data(),d.rho,n*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hbc.data(),d.bc,hbc.size()*sizeof(double),cudaMemcpyDeviceToHost));
        HostState s;double veq=0.0,bcchg=0.0;size_t nliq=0;
        float rmin=std::numeric_limits<float>::max(),rmax=-std::numeric_limits<float>::max();
        for(size_t i=0;i<n;++i){
            double r=hrho[i];rmin=std::min(rmin,hrho[i]);rmax=std::max(rmax,hrho[i]);
            if(fluid_xyz[i][2]<nz_geo){
                s.mass_soil+=r;if(hrho[i]>rho_threshold)++nliq;
                double a=(r-rho_g_eq)/(double(rho_l_eq)-rho_g_eq);a=std::max(0.0,std::min(1.0,a));veq+=a;
            }else s.mass_buffer+=r;
        }
        for(double x:hbc)bcchg+=x;
        s.mass_domain=s.mass_soil+s.mass_buffer;
        s.sat_threshold=double(nliq)/std::max<size_t>(1,nreal);
        s.liquid_volume=veq;s.liquid_mass=double(rho_l_eq)*veq;s.sat_equiv=veq/std::max<size_t>(1,nreal);
        s.rho_min=rmin;s.rho_max=rmax;s.top_outflow_cum=-bcchg;
        return s;
    };

    auto copy_all_from_device=[&](){
        CUDA_CHECK(cudaMemcpy(hrho.data(),d.rho,n*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hpsi.data(),d.psi,n*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hp.data(),d.pressure,n*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hvx.data(),d.vx,n*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hvy.data(),d.vy,n*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hvz.data(),d.vz,n*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hfx.data(),d.fx,n*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hfy.data(),d.fy,n*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hfz.data(),d.fz,n*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hf.data(),d.f,nq*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hF.data(),d.F,nq*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hbc.data(),d.bc,hbc.size()*sizeof(double),cudaMemcpyDeviceToHost));
    };
    auto copy_all_to_device=[&](){
        CUDA_CHECK(cudaMemcpy(d.rho,hrho.data(),n*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.psi,hpsi.data(),n*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.pressure,hp.data(),n*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.vx,hvx.data(),n*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.vy,hvy.data(),n*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.vz,hvz.data(),n*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.fx,hfx.data(),n*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.fy,hfy.data(),n*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.fz,hfz.data(),n*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.f,hf.data(),nq*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.F,hF.data(),nq*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.bc,hbc.data(),hbc.size()*sizeof(double),cudaMemcpyHostToDevice));
    };

    HostState s0{},prev{},last{};int start_step=0,prev_record_step=0;double wall_base=0.0;bool resumed=false;
    auto ckpts=checkpoint_candidates(outdir,ckprefix);
    if(!opt.force){
        for(const auto& cp:ckpts){
            try{
                std::ifstream in(cp,std::ios::binary);CheckpointHeader h{};
                in.read(reinterpret_cast<char*>(&h),sizeof(h));
                if(!in || std::memcmp(h.magic,"SELBM01",7)!=0 || h.version!=1)continue;
                std::string sig(h.signature_len,'\0');if(h.signature_len)in.read(sig.data(),h.signature_len);
                if(sig!=opt.signature||h.nx!=nx||h.ny!=ny||h.nz_geo!=nz_geo||h.n_buffer!=n_buffer||h.n_fluid!=n)continue;
                read_vec(in,hf,nq);read_vec(in,hF,nq);read_vec(in,hrho,n);read_vec(in,hvx,n);read_vec(in,hvy,n);read_vec(in,hvz,n);
                read_vec(in,hpsi,n);read_vec(in,hp,n);read_vec(in,hfx,n);read_vec(in,hfy,n);read_vec(in,hfz,n);read_vec(in,hbc,(size_t)nx*ny);
                copy_all_to_device();s0=h.s0;prev=h.prev;last=prev;start_step=h.step;prev_record_step=h.prev_record_step;wall_base=h.wall_time_sec;resumed=true;
                truncate_stats_to_step(stats_tmp,start_step);
                if(opt.verbose) std::cout<<"[Resume] checkpoint="<<cp.filename().string()<<" step="<<start_step<<"\n";
                break;
            }catch(const std::exception& ex){if(opt.verbose) std::cout<<"[Resume][Warn] "<<cp.filename().string()<<": "<<ex.what()<<"\n";}
        }
    }

    auto save_checkpoint=[&](int step,const HostState& prev_state,int prev_step,double wall){
        copy_all_from_device();
        CheckpointHeader h{};std::memcpy(h.magic,"SELBM01",7);h.version=1;h.nx=nx;h.ny=ny;h.nz_geo=nz_geo;h.n_buffer=n_buffer;h.n_fluid=n;
        h.step=step;h.prev_record_step=prev_step;h.wall_time_sec=wall;h.s0=s0;h.prev=prev_state;h.signature_len=(uint32_t)opt.signature.size();
        std::ostringstream nm;nm<<ckprefix<<std::setw(6)<<std::setfill('0')<<step<<".bin";fs::path cp=outdir/nm.str();
        fs::path tmp=cp;tmp+=".tmp";std::ofstream out(tmp,std::ios::binary|std::ios::trunc);
        out.write(reinterpret_cast<const char*>(&h),sizeof(h));out.write(opt.signature.data(),(std::streamsize)opt.signature.size());
        write_vec(out,hf);write_vec(out,hF);write_vec(out,hrho);write_vec(out,hvx);write_vec(out,hvy);write_vec(out,hvz);
        write_vec(out,hpsi);write_vec(out,hp);write_vec(out,hfx);write_vec(out,hfy);write_vec(out,hfz);write_vec(out,hbc);out.close();
        if(!out)throw std::runtime_error("Checkpoint write failed.");
        std::error_code ec;fs::remove(cp,ec);fs::rename(tmp,cp);prune_ckpt(outdir,ckprefix,opt.keep_checkpoints);
        if(opt.verbose) std::cout<<"[Checkpoint] "<<cp.filename().string()<<"\n";
    };

    const char* csv_header="case_name,G_ADS,rho_dry,iter,wall_time_sec,rho_l_eq,rho_g_eq,Tr,G_int,beta_sc,real_pore_nodes,real_porosity,mass_soil,mass_buffer,mass_domain,saturation_threshold,saturation_equiv,liquid_volume_equiv,liquid_mass_equiv,soil_mass_loss_interval,domain_mass_loss_interval,buffer_mass_change_interval,liquid_mass_loss_interval,top_outflow_mass_interval,J_soil_lu,J_out_mass_lu,J_top_direct_lu,ER_liquid_equiv_lu,soil_mass_loss_cumulative,domain_mass_loss_cumulative,liquid_mass_loss_cumulative,top_outflow_mass_cumulative,mass_partition_error,mass_partition_error_rel,bc_mass_balance_error_interval,bc_mass_balance_error_rel,rho_min,rho_max\n";
    auto append_row=[&](int it,double wall,const HostState& s,const HostState& pr,int prstep)->std::array<double,10>{
        int dn=it-prstep;if(dn<=0)dn=1;
        double soil_loss=pr.mass_soil-s.mass_soil,domain_loss=pr.mass_domain-s.mass_domain,buffer_change=s.mass_buffer-pr.mass_buffer;
        double liquid_loss=pr.liquid_mass-s.liquid_mass,top_interval=s.top_outflow_cum-pr.top_outflow_cum;
        double jsoil=soil_loss/(dn*area),jout=domain_loss/(dn*area),jtop=top_interval/(dn*area),er=liquid_loss/(dn*area);
        double soilcum=s0.mass_soil-s.mass_soil,domcum=s0.mass_domain-s.mass_domain,liqcum=s0.liquid_mass-s.liquid_mass;
        double perr=s.mass_domain-s.mass_soil-s.mass_buffer,prel=std::abs(perr)/std::max(std::abs(s.mass_domain),1e-30);
        double bcerr=domain_loss-top_interval,bcrel=std::abs(bcerr)/std::max({std::abs(domain_loss),std::abs(top_interval),1e-30});
        std::ofstream out(stats_tmp,std::ios::app);
        out<<opt.case_name<<","<<std::setprecision(17)<<G_ads<<","<<rho_dry<<","<<it<<","<<wall<<","<<rho_l_eq<<","<<rho_g_eq<<","<<Tr<<","<<G_int<<","<<beta_sc<<","
           <<nreal<<","<<(double(nreal)/(double(nx)*ny*nz_geo))<<","<<s.mass_soil<<","<<s.mass_buffer<<","<<s.mass_domain<<","<<s.sat_threshold<<","<<s.sat_equiv<<","<<s.liquid_volume<<","<<s.liquid_mass<<","
           <<soil_loss<<","<<domain_loss<<","<<buffer_change<<","<<liquid_loss<<","<<top_interval<<","<<jsoil<<","<<jout<<","<<jtop<<","<<er<<","<<soilcum<<","<<domcum<<","<<liqcum<<","<<s.top_outflow_cum<<","
           <<perr<<","<<prel<<","<<bcerr<<","<<bcrel<<","<<s.rho_min<<","<<s.rho_max<<"\n";
        return {jsoil,jout,jtop,er,prel,bcrel,soilcum,domcum,liqcum,top_interval};
    };

    if(!resumed){
        CUDA_CHECK(cudaMemset(d.bc,0,(size_t)nx*ny*sizeof(double)));
        k_init<<<GS,BS>>>(p,d.rho0,d.rho,d.psi,d.pressure,d.vx,d.vy,d.vz,d.fx,d.fy,d.fz,d.f,d.F);
        k_top<<<GS,BS>>>(p,d.xyz,d.rho,d.psi,d.pressure,d.vx,d.vy,d.vz,d.fz,d.f,d.F,d.bc);
        k_macro<<<GS,BS>>>(p,d.f,d.fx,d.fy,d.fz,d.rho,d.vx,d.vy,d.vz);
        k_pressure_psi<<<GS,BS>>>(p,d.rho,d.pressure,d.psi);CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(d.bc,0,(size_t)nx*ny*sizeof(double)));
        s0=collect();prev=s0;last=s0;start_step=0;prev_record_step=0;wall_base=0.0;
        std::ofstream out(stats_tmp,std::ios::trunc);out<<csv_header;out.close();
        append_row(0,0.0,s0,s0,-1); // interval values are ignored for iter 0 below by replacing file line.
        // Rewrite the iter=0 row to force all interval/flux terms to exactly zero.
        std::ifstream rin(stats_tmp);std::string head,row;std::getline(rin,head);std::getline(rin,row);rin.close();
        std::ofstream rout(stats_tmp,std::ios::trunc);rout<<head<<"\n";
        rout<<opt.case_name<<","<<std::setprecision(17)<<G_ads<<","<<rho_dry<<",0,0,"<<rho_l_eq<<","<<rho_g_eq<<","<<Tr<<","<<G_int<<","<<beta_sc<<","
            <<nreal<<","<<(double(nreal)/(double(nx)*ny*nz_geo))<<","<<s0.mass_soil<<","<<s0.mass_buffer<<","<<s0.mass_domain<<","<<s0.sat_threshold<<","<<s0.sat_equiv<<","<<s0.liquid_volume<<","<<s0.liquid_mass
            <<",0,0,0,0,0,0,0,0,0,0,0,0,0,"<<(s0.mass_domain-s0.mass_soil-s0.mass_buffer)<<","<<std::abs(s0.mass_domain-s0.mass_soil-s0.mass_buffer)/std::max(std::abs(s0.mass_domain),1e-30)
            <<",0,0,"<<s0.rho_min<<","<<s0.rho_max<<"\n";rout.close();
        if(opt.verbose) std::cout<<std::fixed<<std::setprecision(6)<<"[Init] pores="<<nreal<<" phi="<<(double(nreal)/(double(nx)*ny*nz_geo))
                 <<" | Msoil="<<s0.mass_soil<<" Mbuffer="<<s0.mass_buffer<<" Mdomain="<<s0.mass_domain
                 <<" | Sat_th="<<s0.sat_threshold<<" Sat_eq="<<s0.sat_equiv<<" | rho=["<<s0.rho_min<<","<<s0.rho_max<<"]\n";
        if(opt.save_vtk){copy_all_from_device();write_vtk_legacy(outdir/(opt.output_prefix+"0.vtk"),nx,ny,p.nz,solid_dense,buffer_dense,grid_to_idx,hrho,hpsi,hp,hvx,hvy,hvz);}
        save_checkpoint(0,prev,0,0.0);
    }

    auto t0=std::chrono::steady_clock::now(),lastprint=t0;
    std::array<double,10> last_metrics{};
    for(int it=start_step+1;it<=opt.total_steps;++it){
        if(it==start_step+1 || it%1000==0) trial_guard::heartbeat_or_throw();
        k_macro<<<GS,BS>>>(p,d.f,d.fx,d.fy,d.fz,d.rho,d.vx,d.vy,d.vz);
        k_pressure_psi<<<GS,BS>>>(p,d.rho,d.pressure,d.psi);
        k_force<<<GS,BS>>>(p,d.xyz,d.grid,d.ff,d.ads,d.psi,d.fx,d.fy,d.fz);
        k_collision<<<GS,BS>>>(p,d.rho,d.fx,d.fy,d.fz,d.f,d.F);
        k_stream<<<GS,BS>>>(p,d.pull,d.F,d.f);
        k_top<<<GS,BS>>>(p,d.xyz,d.rho,d.psi,d.pressure,d.vx,d.vy,d.vz,d.fz,d.f,d.F,d.bc);
        k_macro<<<GS,BS>>>(p,d.f,d.fx,d.fy,d.fz,d.rho,d.vx,d.vy,d.vz);
        k_pressure_psi<<<GS,BS>>>(p,d.rho,d.pressure,d.psi);
        if(it%opt.record_interval==0 || it==opt.total_steps){
            CUDA_CHECK(cudaDeviceSynchronize());HostState s=collect();
            auto now=std::chrono::steady_clock::now();double wall=wall_base+std::chrono::duration<double>(now-t0).count();
            last_metrics=append_row(it,wall,s,prev,prev_record_step);last=s;prev=s;prev_record_step=it;
            if(!std::isfinite(s.mass_domain)||!std::isfinite(s.sat_equiv)||!std::isfinite(s.rho_min)||!std::isfinite(s.rho_max))throw std::runtime_error("NaN/Inf detected.");
        }
        if(opt.verbose && (it%opt.print_interval==0 || it==opt.total_steps)){
            auto now=std::chrono::steady_clock::now();double dt=std::chrono::duration<double>(now-lastprint).count();lastprint=now;
            std::cout<<std::fixed<<std::setprecision(6)<<"iter="<<std::setw(7)<<it<<" | Sat_eq="<<last.sat_equiv
                     <<" | ERliq="<<std::scientific<<std::setprecision(6)<<last_metrics[3]
                     <<" | Jsoil="<<last_metrics[0]<<" | Jtop="<<last_metrics[2]
                     <<" | bc_rel="<<std::scientific<<std::setprecision(3)<<last_metrics[5]
                     <<" | rho=["<<std::fixed<<std::setprecision(5)<<last.rho_min<<","<<last.rho_max<<"] | dt="<<std::setprecision(2)<<dt<<"s\n";
        }
        if(opt.save_vtk && opt.vtk_interval>0 && it%opt.vtk_interval==0){
            CUDA_CHECK(cudaDeviceSynchronize());copy_all_from_device();
            write_vtk_legacy(outdir/(opt.output_prefix+std::to_string(it)+".vtk"),nx,ny,p.nz,solid_dense,buffer_dense,grid_to_idx,hrho,hpsi,hp,hvx,hvy,hvz);
        }
        if((opt.checkpoint_interval>0 && it%opt.checkpoint_interval==0)||it==opt.total_steps){
            CUDA_CHECK(cudaDeviceSynchronize());auto now=std::chrono::steady_clock::now();double wall=wall_base+std::chrono::duration<double>(now-t0).count();
            save_checkpoint(it,prev,prev_record_step,wall);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto tend=std::chrono::steady_clock::now();double wall=wall_base+std::chrono::duration<double>(tend-t0).count();
    if(opt.total_steps==0){last=s0;last_metrics.fill(0.0);}
    if(fs::exists(stats_final)){std::error_code ec;fs::remove(stats_final,ec);}
    fs::rename(stats_tmp,stats_final);

    NativeRunResult rr;rr.final_step=opt.total_steps;rr.resumed=resumed;rr.wall_time_sec=wall;
    rr.final_saturation_threshold=last.sat_threshold;rr.final_saturation_equiv=last.sat_equiv;rr.final_liquid_mass_equiv=last.liquid_mass;
    rr.final_J_soil_lu=last_metrics[0];rr.final_J_out_mass_lu=last_metrics[1];rr.final_J_top_direct_lu=last_metrics[2];rr.final_ER_liquid_equiv_lu=last_metrics[3];
    rr.final_mass_partition_error_rel=last_metrics[4];rr.final_bc_mass_balance_error_rel=last_metrics[5];rr.rho_min_final=last.rho_min;rr.rho_max_final=last.rho_max;
    rr.stats_csv=stats_final.u8string();
    write_done_json(outdir/"DONE.json",opt,rr);
    if(opt.verbose) std::cout<<"[DONE] stats="<<stats_final.u8string()<<"\n";
    return rr;
}
