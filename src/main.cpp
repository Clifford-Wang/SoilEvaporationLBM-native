#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <thread>
#include <cstdio>
#include "native_production.hpp"
#include "trial_guard.hpp"

namespace fs = std::filesystem;

static std::string trim(std::string s){
    auto not_space=[](unsigned char c){return !std::isspace(c);};
    s.erase(s.begin(),std::find_if(s.begin(),s.end(),not_space));
    s.erase(std::find_if(s.rbegin(),s.rend(),not_space).base(),s.end());
    if(s.size()>=2 && ((s.front()=='"'&&s.back()=='"')||(s.front()=='\''&&s.back()=='\''))) s=s.substr(1,s.size()-2);
    return s;
}

struct Ini {
    std::map<std::string,std::map<std::string,std::string>> sec;
    static Ini load(const fs::path& p){
        std::ifstream f(p);
        if(!f) throw std::runtime_error("Cannot open config: "+p.string());
        Ini out; std::string line,section;
        while(std::getline(f,line)){
            if(!line.empty() && line.back()=='\r') line.pop_back();
            auto s=trim(line);
            if(s.empty()||s[0]=='#'||s[0]==';') continue;
            if(s.front()=='['&&s.back()==']'){section=trim(s.substr(1,s.size()-2));continue;}
            auto pos=s.find('='); if(pos==std::string::npos) continue;
            out.sec[section][trim(s.substr(0,pos))]=trim(s.substr(pos+1));
        }
        return out;
    }
    std::string get(const std::string& s,const std::string& k,const std::string& d="") const{
        auto a=sec.find(s); if(a==sec.end()) return d;
        auto b=a->second.find(k); return b==a->second.end()?d:b->second;
    }
    int geti(const std::string&s,const std::string&k,int d=0)const{auto v=get(s,k,"");return v.empty()?d:std::stoi(v);}
    double getd(const std::string&s,const std::string&k,double d=0.0)const{auto v=get(s,k,"");return v.empty()?d:std::stod(v);}
};

static fs::path utf8_path(const std::string& s){
#ifdef _WIN32
    return fs::u8path(s);
#else
    return fs::path(s);
#endif
}

static std::string path_utf8(const fs::path& p){
#ifdef _WIN32
    return p.u8string();
#else
    return p.string();
#endif
}

static fs::path resolve_path(const fs::path& cfg,const std::string& raw){
    fs::path p=utf8_path(trim(raw));
    if(p.is_absolute()) return p;
    return fs::weakly_canonical(cfg.parent_path()/p);
}

static std::set<std::string> parse_names(const std::string& raw){
    std::set<std::string> out; std::stringstream ss(raw); std::string x;
    while(std::getline(ss,x,',')){
        x=trim(x); if(x.empty()) continue;
        bool digits=!x.empty()&&std::all_of(x.begin(),x.end(),[](unsigned char c){return std::isdigit(c);});
        if(digits) x="soil_sample_"+x;
        out.insert(x);
    }
    return out;
}

static std::string extract_case_name(const fs::path& p){
    std::string base=path_utf8(p.filename());
    base=std::regex_replace(base,std::regex("\\s+")," ");
    std::smatch m;
    std::regex re("(.+?)_connected_Z_THROUGH.*\\.txt$",std::regex::icase);
    if(std::regex_match(base,m,re)) return trim(m[1].str());
    return path_utf8(p.stem());
}

static std::regex glob_to_regex(std::string pat){
    // Match Python pathlib Path.glob semantics needed by the reference runner.
    // In particular, "**/" must also match zero directory levels, so files
    // directly under geometry_root are included.
    std::string r="^";
    for(size_t i=0;i<pat.size();){
        if(i+3<=pat.size() && pat.compare(i,3,"**/")==0){
            r+="(?:.*/)?";
            i+=3;
            continue;
        }
        if(i+2<=pat.size() && pat.compare(i,2,"**")==0){
            r+=".*";
            i+=2;
            continue;
        }
        char ch=pat[i++];
        if(ch=='*') r+="[^/]*";
        else if(ch=='?') r+="[^/]";
        else{
            if(std::string(".^$|()[]{}+\\").find(ch)!=std::string::npos) r+='\\';
            r+=ch;
        }
    }
    r+="$";
    return std::regex(r,std::regex::icase);
}

static std::vector<fs::path> find_cases(const fs::path& root,const std::string& pat,const std::set<std::string>& only,const std::set<std::string>& exclude){
    if(!fs::is_directory(root)) throw std::runtime_error("Geometry root does not exist: "+path_utf8(root));
    std::vector<fs::path> out; auto re=glob_to_regex(pat);
    for(const auto& e:fs::recursive_directory_iterator(root)){
        if(!e.is_regular_file()) continue;
        auto rel=fs::relative(e.path(),root).generic_u8string();
        if(!std::regex_match(rel,re)) continue;
        auto name=extract_case_name(e.path());
        if(!only.empty()&&!only.count(name)) continue;
        if(exclude.count(name)) continue;
        out.push_back(e.path());
    }
    std::sort(out.begin(),out.end());
    return out;
}

static std::vector<long long> load_ints(const fs::path& p,size_t expected){
    std::ifstream f(p); if(!f) throw std::runtime_error("Cannot open geometry: "+path_utf8(p));
    std::vector<long long> v; v.reserve(expected); long long x;
    while(f>>x) v.push_back(x);
    if(v.size()!=expected) throw std::runtime_error("Geometry element count mismatch: expected "+std::to_string(expected)+", got "+std::to_string(v.size()));
    return v;
}

static std::vector<double> load_doubles(const fs::path& p,size_t expected){
    std::ifstream f(p); if(!f) throw std::runtime_error("Cannot open phase: "+path_utf8(p));
    std::vector<double> v; v.reserve(expected); double x;
    while(f>>x) v.push_back(x);
    if(v.size()!=expected) throw std::runtime_error("Phase element count mismatch: expected "+std::to_string(expected)+", got "+std::to_string(v.size()));
    return v;
}

static inline size_t idx3(int i,int j,int k,int nx,int ny,int nz){
    (void)nx;
    return (static_cast<size_t>(i)*ny + j)*nz + k;
}
static inline size_t flatF(int i,int j,int k,int nx,int ny){
    return static_cast<size_t>(i)+static_cast<size_t>(nx)*(static_cast<size_t>(j)+static_cast<size_t>(ny)*k);
}
static inline int wrap(int a,int n){if(a<0)a+=n;else if(a>=n)a-=n;return a;}

static uint64_t fnv1a_u64(uint64_t h,uint64_t x){
    for(int b=0;b<8;++b){h^=(x&0xffu);h*=1099511628211ull;x>>=8;}
    return h;
}
static uint64_t hash_i32(uint64_t h,int32_t x){return fnv1a_u64(h,static_cast<uint32_t>(x));}
static uint64_t hash_u8(uint64_t h,uint8_t x){h^=x;h*=1099511628211ull;return h;}


static std::vector<double> parse_float_list(const std::string& raw){
    std::vector<double> out;std::stringstream ss(raw);std::string x;
    while(std::getline(ss,x,',')){x=trim(x);if(!x.empty())out.push_back(std::stod(x));}
    if(out.empty())throw std::runtime_error("G_ads list is empty.");
    for(double v:out)if(!std::isfinite(v))throw std::runtime_error("G_ads contains non-finite value.");
    return out;
}
static std::string float_tag(double v,const std::string& prefix){
    std::ostringstream os;os<<std::fixed<<std::setprecision(6)<<std::abs(v);std::string s=os.str();
    while(!s.empty()&&s.back()=='0')s.pop_back();if(!s.empty()&&s.back()=='.')s.pop_back();
    std::replace(s.begin(),s.end(),'.','p');return prefix+"_"+(v>=0?"p":"m")+s;
}
static uint64_t fnv_file(const fs::path& p){
    std::ifstream in(p,std::ios::binary);if(!in)throw std::runtime_error("Cannot hash file: "+path_utf8(p));
    uint64_t h=1469598103934665603ull;char b[1<<16];
    while(in){in.read(b,sizeof(b));auto n=in.gcount();for(std::streamsize i=0;i<n;++i){h^=(uint8_t)b[i];h*=1099511628211ull;}}
    return h;
}
static std::string hex64(uint64_t x){std::ostringstream o;o<<std::hex<<std::setw(16)<<std::setfill('0')<<x;return o.str();}
static std::string job_signature(uint64_t gh,uint64_t ph,const std::string& case_name,double gads,double rho_dry,
                                 int nx,int ny,int nz,int nb,int pv,double niu,double gi,double beta,double tr,
                                 double rl,double rg,double rle,double rge,const Ini& ini){
    std::ostringstream s;s<<std::setprecision(17)
      <<"native_v1|"<<hex64(gh)<<"|"<<hex64(ph)<<"|"<<case_name<<"|"<<gads<<"|"<<rho_dry<<"|"
      <<nx<<"|"<<ny<<"|"<<nz<<"|"<<nb<<"|"<<pv<<"|"<<niu<<"|"<<gi<<"|"<<beta<<"|"<<tr<<"|"<<rl<<"|"<<rg<<"|"<<rle<<"|"<<rge<<"|"
      <<ini.geti("RUN","total_steps")<<"|"<<ini.geti("RUN","record_interval")<<"|"<<ini.geti("RUN","vtk_interval")<<"|"<<ini.geti("RUN","checkpoint_interval");
    uint64_t h=1469598103934665603ull;auto z=s.str();for(unsigned char c:z){h^=c;h*=1099511628211ull;}return hex64(h);
}
static bool file_text_contains(const fs::path& p,const std::string& needle){
    if(!fs::exists(p))return false;std::ifstream in(p,std::ios::binary);std::string s((std::istreambuf_iterator<char>(in)),{});return s.find(needle)!=std::string::npos;
}
static std::string time_tag(){
    auto t=std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm,&t);
#else
    localtime_r(&t,&tm);
#endif
    std::ostringstream o;o<<std::put_time(&tm,"%Y%m%d-%H%M%S");return o.str();
}
static void archive_dir_if_needed(const fs::path& dir,const std::string& sig,bool force,bool verbose){
    if(!fs::exists(dir))return;
    fs::path sf=dir/"run_signature.txt";
    bool same=false;if(fs::exists(sf)){std::ifstream in(sf);std::string old;std::getline(in,old);same=(trim(old)==sig);}
    if(!force&&same)return;
    fs::path dst=dir;dst+=std::string("__archive_")+(force?"force_":"config_")+time_tag();
    int k=1;while(fs::exists(dst)){dst=dir;dst+=std::string("__archive_")+time_tag()+"_"+std::to_string(k++);}
    fs::rename(dir,dst);
    if(verbose) std::cout<<"[Archive] "<<path_utf8(dst)<<"\n";
}
static void write_signature(const fs::path& dir,const std::string& sig){fs::create_directories(dir);std::ofstream(dir/"run_signature.txt")<<sig<<"\n";}

int main(int argc,char** argv){
    int pause_when_finished=0;
    try{
        trial_guard::initialize_or_throw();
        fs::path cfg=(argc>1?utf8_path(argv[1]):utf8_path("config.txt"));cfg=fs::absolute(cfg);
        auto ini=Ini::load(cfg);pause_when_finished=ini.geti("CONTROL","pause_when_finished",1);
        const int nx=ini.geti("GRID","nx"),ny=ini.geti("GRID","ny"),nz_geo=ini.geti("GRID","nz_geo"),n_buffer=ini.geti("GRID","n_buffer"),pore_value=ini.geti("GRID","pore_value",0);
        if(nx<=0||ny<=0||nz_geo<=0||n_buffer<0)throw std::runtime_error("Invalid GRID settings.");
        const size_t expected=(size_t)nx*ny*nz_geo;const int nz=nz_geo+n_buffer;
        const double niu=ini.getd("PHYSICS","niu",0.20),G_int=ini.getd("PHYSICS","G_int",-1.0),beta=ini.getd("PHYSICS","beta_sc",1.16),Tr=ini.getd("PHYSICS","Tr",0.86);
        const double rho_liq=ini.getd("PHYSICS","rho_liq",6.498946),rho_gas=ini.getd("PHYSICS","rho_gas",0.379679),rho_l_eq=ini.getd("PHYSICS","rho_l_eq",rho_liq),rho_g_eq=ini.getd("PHYSICS","rho_g_eq",rho_gas),rho_dry=ini.getd("PHYSICS","rho_dry",0.37);
        auto gads_values=parse_float_list(ini.get("PHYSICS","G_ads","-0.20"));
        if(rho_dry<=0||rho_l_eq<=rho_g_eq)throw std::runtime_error("Invalid density settings.");

        const int total_steps=ini.geti("RUN","total_steps",200000),record_interval=ini.geti("RUN","record_interval",1000),print_interval=ini.geti("RUN","print_interval",1000);
        const int vtk_interval=ini.geti("RUN","vtk_interval",10000),checkpoint_interval=ini.geti("RUN","checkpoint_interval",100000),keep_ck=ini.geti("RUN","keep_checkpoints",2),save_vtk=ini.geti("RUN","save_vtk",1);
        const double memfrac=ini.getd("RUN","device_memory_fraction",0.86),wait_seconds=ini.getd("RUN","wait_seconds",3.0);
        const int force=ini.geti("CONTROL","force",0),dry_run=ini.geti("CONTROL","dry_run",0);
        std::string console_mode=ini.get("CONTROL","console_mode","minimal");
        std::transform(console_mode.begin(),console_mode.end(),console_mode.begin(),[](unsigned char ch){return (char)std::tolower(ch);});
        if(console_mode!="full"&&console_mode!="minimal")throw std::runtime_error("CONTROL.console_mode must be full or minimal.");
        const bool verbose=(console_mode=="full");

        auto geo_root=resolve_path(cfg,ini.get("PATHS","geometry_root"));
        auto phase_file=resolve_path(cfg,ini.get("PATHS","phase_file"));
        auto output_root=resolve_path(cfg,ini.get("PATHS","output_root","result"));
        auto pattern=ini.get("PATHS","geometry_pattern","**/*_connected_Z_THROUGH*.txt");
        auto only=parse_names(ini.get("SAMPLES","only")),exclude=parse_names(ini.get("SAMPLES","exclude"));
        auto cases=find_cases(geo_root,pattern,only,exclude);if(cases.empty())throw std::runtime_error("No geometry matched current config.");
        auto phase=load_doubles(phase_file,expected);uint64_t phase_hash=fnv_file(phase_file);
        fs::create_directories(output_root);

        if(verbose){
            std::cout<<"####################################################################################################\n";
            std::cout<<"SoilEvaporationLBM Native CUDA | no Python/Taichi runtime\n";
            std::cout<<"Config        = "<<path_utf8(cfg)<<"\nGeometry root = "<<path_utf8(geo_root)<<"\nPhase file    = "<<path_utf8(phase_file)<<"\nOutput root   = "<<path_utf8(output_root)<<"\n";
            std::cout<<"Samples       = "<<cases.size()<<"\nG_ads list    = [";for(size_t i=0;i<gads_values.size();++i){if(i)std::cout<<",";std::cout<<gads_values[i];}std::cout<<"]\n";
            std::cout<<"Total jobs    = "<<cases.size()*gads_values.size()<<"\n";
            std::cout<<"####################################################################################################\n";
        }
        if(dry_run){
            int k=0;for(auto& gp:cases)for(double ga:gads_values){auto cn=extract_case_name(gp);std::cout<<++k<<". "<<cn<<" | Gads="<<ga<<" | "<<path_utf8(output_root/utf8_path(cn)/utf8_path(float_tag(ga,"Gads")))<<"\n";}
            if(pause_when_finished){std::cout<<"Press Enter to exit...";std::cin.get();}return 0;
        }

        struct Summary{std::string case_name,status,output;double gads=0,sat=0,er=0,js=0,jt=0,bcr=0,rmin=0,rmax=0,wall=0;};
        std::vector<Summary> summary;int success=0,skipped=0,failed=0;int jobno=0,totaljobs=(int)(cases.size()*gads_values.size());

        static const int ev[19][3]={{0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},{1,1,0},{-1,-1,0},{1,-1,0},{-1,1,0},{1,0,1},{-1,0,-1},{1,0,-1},{-1,0,1},{0,1,1},{0,-1,-1},{0,1,-1},{0,-1,1}};

        for(const auto& geo_file:cases){
            const std::string case_name=extract_case_name(geo_file);auto gf=load_ints(geo_file,expected);uint64_t geo_hash=fnv_file(geo_file);
            std::vector<uint8_t> solid((size_t)nx*ny*nz,1),buffer((size_t)nx*ny*nz,0);size_t nreal=0,nliq=0;
            for(int k=0;k<nz_geo;++k)for(int j=0;j<ny;++j)for(int i=0;i<nx;++i){size_t ff=flatF(i,j,k,nx,ny);bool pore=(gf[ff]==pore_value);solid[idx3(i,j,k,nx,ny,nz)]=pore?0:1;if(pore){++nreal;if(phase[ff]>0.5)++nliq;}}
            if(n_buffer>0)for(int i=0;i<nx;++i)for(int j=0;j<ny;++j)for(int k=nz_geo;k<nz;++k){solid[idx3(i,j,k,nx,ny,nz)]=0;buffer[idx3(i,j,k,nx,ny,nz)]=1;}
            size_t nfluid=nreal+(size_t)nx*ny*n_buffer;
            std::vector<std::array<int32_t,3>> xyz;xyz.reserve(nfluid);std::vector<int32_t> grid((size_t)nx*ny*nz,-1);
            for(int i=0;i<nx;++i)for(int j=0;j<ny;++j)for(int k=0;k<nz;++k)if(!solid[idx3(i,j,k,nx,ny,nz)]){int id=(int)xyz.size();xyz.push_back({i,j,k});grid[idx3(i,j,k,nx,ny,nz)]=id;}
            if(xyz.size()!=nfluid)throw std::runtime_error("Sparse mapping size mismatch.");
            std::vector<int32_t> pull(nfluid*19),ffnb(nfluid*19),ads(nfluid*19);
            for(size_t id=0;id<nfluid;++id){int i=xyz[id][0],j=xyz[id][1],k=xyz[id][2];for(int q=0;q<19;++q){size_t o=id*19+q;
                int si=wrap(i-ev[q][0],nx),sj=wrap(j-ev[q][1],ny),sk=k-ev[q][2];
                if(sk<0||sk>=nz)pull[o]=-2;else if(!solid[idx3(si,sj,sk,nx,ny,nz)])pull[o]=grid[idx3(si,sj,sk,nx,ny,nz)];else pull[o]=-1;
                int ni=wrap(i+ev[q][0],nx),nj=wrap(j+ev[q][1],ny),nk=k+ev[q][2];ads[o]=0;
                if(nk<0||nk>=nz)ffnb[o]=-1;else if(!solid[idx3(ni,nj,nk,nx,ny,nz)])ffnb[o]=grid[idx3(ni,nj,nk,nx,ny,nz)];else{ffnb[o]=-1;ads[o]=1;}
            }}
            std::vector<float> rho_init(nfluid,(float)rho_dry);for(size_t id=0;id<nfluid;++id){int i=xyz[id][0],j=xyz[id][1],k=xyz[id][2];if(k<nz_geo)rho_init[id]=(phase[flatF(i,j,k,nx,ny)]>0.5)?(float)rho_liq:(float)rho_gas;}
            if(verbose) std::cout<<"[Geometry] "<<case_name<<" pores="<<nreal<<" phi="<<std::fixed<<std::setprecision(6)<<(double)nreal/expected<<" liquid_init="<<nliq<<" n_fluid="<<nfluid<<"\n";

            for(double gads:gads_values){
                ++jobno;fs::path outdir=output_root/utf8_path(case_name)/utf8_path(float_tag(gads,"Gads"));
                std::string sig=job_signature(geo_hash,phase_hash,case_name,gads,rho_dry,nx,ny,nz_geo,n_buffer,pore_value,niu,G_int,beta,Tr,rho_liq,rho_gas,rho_l_eq,rho_g_eq,ini);
                bool done_same=fs::exists(outdir/"DONE.json")&&file_text_contains(outdir/"DONE.json","\"signature\":\""+sig+"\"")&&file_text_contains(outdir/"DONE.json","\"total_steps\":"+std::to_string(total_steps));
                if(verbose) std::cout<<"\n====================================================================================================\n[Batch "<<jobno<<"/"<<totaljobs<<"] "<<case_name<<" | Gads="<<std::showpos<<std::fixed<<std::setprecision(6)<<gads<<std::noshowpos<<"\n====================================================================================================\n";
                if(done_same&&!force){
                    ++skipped;summary.push_back({case_name,"SKIPPED",path_utf8(outdir),gads});
                    if(verbose) std::cout<<"[Batch] completed configuration matches; skipped.\n";
                    else std::cout<<"SUCCESS: "<<case_name<<" | Gads="<<gads<<" | already completed\n";
                    continue;
                }
                try{
                    archive_dir_if_needed(outdir,sig,force!=0,verbose);fs::create_directories(outdir);write_signature(outdir,sig);
                    NativeRunOptions opt;opt.total_steps=total_steps;opt.record_interval=record_interval;opt.print_interval=print_interval;opt.vtk_interval=vtk_interval;opt.checkpoint_interval=checkpoint_interval;opt.keep_checkpoints=keep_ck;opt.save_vtk=save_vtk;opt.device_memory_fraction=memfrac;opt.force=force;opt.verbose=verbose;
                    opt.case_name=case_name;opt.output_dir=path_utf8(outdir);opt.output_prefix=case_name+"_"+float_tag(gads,"Gads")+"_"+float_tag(rho_dry,"RhoDry")+"_";opt.signature=sig;
                    auto rr=run_native_production(nx,ny,nz_geo,n_buffer,(float)niu,(float)G_int,(float)beta,(float)Tr,(float)gads,(float)rho_liq,(float)rho_gas,(float)rho_l_eq,(float)rho_g_eq,(float)rho_dry,xyz,grid,pull,ffnb,ads,rho_init,solid,buffer,opt);
                    ++success;summary.push_back({case_name,"DONE",path_utf8(outdir),gads,rr.final_saturation_equiv,rr.final_ER_liquid_equiv_lu,rr.final_J_soil_lu,rr.final_J_top_direct_lu,rr.final_bc_mass_balance_error_rel,rr.rho_min_final,rr.rho_max_final,rr.wall_time_sec});
                    if(!verbose) std::cout<<"SUCCESS: "<<case_name<<" | Gads="<<gads<<"\n";
                }catch(const std::exception& ex){
                    ++failed;summary.push_back({case_name,std::string("FAILED: ")+ex.what(),path_utf8(outdir),gads});
                    if(verbose) std::cerr<<"[Batch][FAILED] "<<ex.what()<<"\n";
                    else std::cerr<<"FAILED: "<<case_name<<" | Gads="<<gads<<" | "<<ex.what()<<"\n";
                }
                if(jobno<totaljobs&&wait_seconds>0)std::this_thread::sleep_for(std::chrono::duration<double>(wait_seconds));
            }
        }

        fs::path sumfile=output_root/"batch_summary.csv";std::ofstream so(sumfile);so<<"case_name,G_ADS,status,final_saturation_equiv,final_ER_liquid_equiv_lu,final_J_soil_lu,final_J_top_direct_lu,final_bc_mass_balance_error_rel,rho_min_final,rho_max_final,wall_time_sec,output_dir\n";
        for(auto&r:summary)so<<r.case_name<<","<<std::setprecision(17)<<r.gads<<","<<r.status<<","<<r.sat<<","<<r.er<<","<<r.js<<","<<r.jt<<","<<r.bcr<<","<<r.rmin<<","<<r.rmax<<","<<r.wall<<","<<r.output<<"\n";
        if(verbose){
            std::cout<<"\n####################################################################################################\n[Batch] ALL DONE\n[Batch] success="<<success<<"\n[Batch] skipped="<<skipped<<"\n[Batch] failed="<<failed<<"\n[Batch] summary="<<path_utf8(sumfile)<<"\n####################################################################################################\n";
        }else{
            if(failed==0) std::cout<<"SUCCESS: all jobs completed | success="<<success<<" | skipped="<<skipped<<"\n";
            else std::cerr<<"FAILED: batch completed with "<<failed<<" failed job(s)\n";
        }
        if(pause_when_finished){if(verbose)std::cout<<"Press Enter to exit...";std::cin.get();}
        return failed?1:0;
    }catch(const trial_guard::TrialViolation& e){
        std::cerr<<e.what()<<"\n";
        if(pause_when_finished)std::cin.get();
        return 2;
    }catch(const std::exception& e){
        std::cerr<<"FAILED: "<<e.what()<<"\n";
        if(pause_when_finished)std::cin.get();
        return 1;
    }
}
