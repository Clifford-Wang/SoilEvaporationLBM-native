#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
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
    std::string r="^";
    for(size_t i=0;i<pat.size();++i){
        char c=pat[i];
        if(c=='*'){
            if(i+1<pat.size()&&pat[i+1]=='*'){r+=".*";++i;}
            else r+="[^/]*";
        }else if(c=='?') r+=".";
        else{
            if(std::string(".^$|()[]{}+\\").find(c)!=std::string::npos) r+='\\';
            r+=c;
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

int main(int argc,char** argv){
    try{
        fs::path cfg=(argc>1?fs::path(argv[1]):fs::path("config.txt"));
        cfg=fs::absolute(cfg);
        auto ini=Ini::load(cfg);

        const int nx=ini.geti("GRID","nx"),ny=ini.geti("GRID","ny"),nz_geo=ini.geti("GRID","nz_geo");
        const int n_buffer=ini.geti("GRID","n_buffer"),pore_value=ini.geti("GRID","pore_value",0);
        if(nx<=0||ny<=0||nz_geo<=0||n_buffer<0) throw std::runtime_error("Invalid GRID settings.");
        const int nz=nz_geo+n_buffer;
        const size_t expected=static_cast<size_t>(nx)*ny*nz_geo;

        auto geo_root=resolve_path(cfg,ini.get("PATHS","geometry_root"));
        auto phase_file=resolve_path(cfg,ini.get("PATHS","phase_file"));
        auto pattern=ini.get("PATHS","geometry_pattern","**/*_connected_Z_THROUGH*.txt");
        auto only=parse_names(ini.get("SAMPLES","only"));
        auto exclude=parse_names(ini.get("SAMPLES","exclude"));
        auto cases=find_cases(geo_root,pattern,only,exclude);
        if(cases.empty()) throw std::runtime_error("No geometry matched current config.");
        if(cases.size()!=1){
            std::ostringstream os;os<<"Stage-1 validation expects exactly one selected geometry; matched "<<cases.size()<<":\n";
            for(auto&p:cases)os<<"  "<<path_utf8(p)<<"\n";
            throw std::runtime_error(os.str());
        }
        auto geo_file=cases.front();
        auto case_name=extract_case_name(geo_file);

        std::cout<<"================================================================================\n";
        std::cout<<"SoilEvaporationLBM native Stage-1 geometry/mapping validator\n";
        std::cout<<"================================================================================\n";
        std::cout<<"Config    : "<<path_utf8(cfg)<<"\n";
        std::cout<<"Case      : "<<case_name<<"\n";
        std::cout<<"Geometry  : "<<path_utf8(geo_file)<<"\n";
        std::cout<<"Phase     : "<<path_utf8(phase_file)<<"\n";
        std::cout<<"Grid      : "<<nx<<" x "<<ny<<" x "<<nz_geo<<" + buffer "<<n_buffer<<"\n";

        auto gf=load_ints(geo_file,expected);
        auto pf=load_doubles(phase_file,expected);
        std::map<long long,size_t> counts;
        size_t n_real_pore=0,n_liq=0;
        std::vector<uint8_t> solid(static_cast<size_t>(nx)*ny*nz,1);
        std::vector<uint8_t> buffer(static_cast<size_t>(nx)*ny*nz,0);

        for(int k=0;k<nz_geo;++k)for(int j=0;j<ny;++j)for(int i=0;i<nx;++i){
            size_t ff=flatF(i,j,k,nx,ny); counts[gf[ff]]++;
            bool pore=(gf[ff]==pore_value);
            solid[idx3(i,j,k,nx,ny,nz)]=pore?0:1;
            if(pore){++n_real_pore;if(pf[ff]>0.5)++n_liq;}
        }
        if(n_buffer>0){
            for(int i=0;i<nx;++i)for(int j=0;j<ny;++j)for(int k=nz_geo;k<nz;++k){
                solid[idx3(i,j,k,nx,ny,nz)]=0; buffer[idx3(i,j,k,nx,ny,nz)]=1;
            }
        }
        size_t n_gas=n_real_pore-n_liq;
        size_t n_buffer_nodes=static_cast<size_t>(nx)*ny*n_buffer;
        size_t n_fluid=n_real_pore+n_buffer_nodes;

        std::cout<<"[Geometry] value counts = {";
        bool first=true;for(auto&kv:counts){if(!first)std::cout<<", ";first=false;std::cout<<kv.first<<": "<<kv.second;}std::cout<<"}\n";
        std::cout<<"[Geometry] pore_value   = "<<pore_value<<"\n";
        std::cout<<"[Geometry] real pore nodes = "<<n_real_pore<<"\n";
        std::cout<<std::fixed<<std::setprecision(9)<<"[Geometry] real porosity   = "<<(double)n_real_pore/(double)expected<<"\n";
        std::cout<<"[Phase] initial liquid pore nodes = "<<n_liq<<"\n";
        std::cout<<"[Phase] initial gas pore nodes    = "<<n_gas<<"\n";
        std::cout<<"[Sparse] buffer nodes = "<<n_buffer_nodes<<"\n";
        std::cout<<"[Sparse] n_fluid      = "<<n_fluid<<"\n";

        std::vector<std::array<int32_t,3>> fluid_xyz; fluid_xyz.reserve(n_fluid);
        std::vector<int32_t> grid_to_idx(static_cast<size_t>(nx)*ny*nz,-1);
        for(int i=0;i<nx;++i)for(int j=0;j<ny;++j)for(int k=0;k<nz;++k){
            if(solid[idx3(i,j,k,nx,ny,nz)]==0){
                int32_t id=static_cast<int32_t>(fluid_xyz.size());
                fluid_xyz.push_back({i,j,k}); grid_to_idx[idx3(i,j,k,nx,ny,nz)]=id;
            }
        }
        if(fluid_xyz.size()!=n_fluid) throw std::runtime_error("n_fluid mismatch after mapping.");

        static const int e[19][3]={
            {0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
            {1,1,0},{-1,-1,0},{1,-1,0},{-1,1,0},{1,0,1},{-1,0,-1},{1,0,-1},{-1,0,1},
            {0,1,1},{0,-1,-1},{0,1,-1},{0,-1,1}
        };
        uint64_t h_pull=1469598103934665603ull,h_ff=h_pull,h_ads=h_pull,h_solid=h_pull,h_xyz=h_pull,h_grid=h_pull;
        uint64_t pull_fluid=0,pull_solid=0,pull_oob=0,ff_fluid=0,ff_solid=0,ff_oob=0;
        for(auto&xyz:fluid_xyz){h_xyz=hash_i32(h_xyz,xyz[0]);h_xyz=hash_i32(h_xyz,xyz[1]);h_xyz=hash_i32(h_xyz,xyz[2]);}
        for(int32_t x:grid_to_idx)h_grid=hash_i32(h_grid,x);

        for(size_t id=0;id<fluid_xyz.size();++id){
            int i=fluid_xyz[id][0],j=fluid_xyz[id][1],k=fluid_xyz[id][2];
            for(int q=0;q<19;++q){
                int src_i=wrap(i-e[q][0],nx),src_j=wrap(j-e[q][1],ny),src_k=k-e[q][2];
                int32_t pull; uint8_t is_solid=0;
                if(src_k<0||src_k>=nz){pull=-2;++pull_oob;}
                else if(solid[idx3(src_i,src_j,src_k,nx,ny,nz)]==0){pull=grid_to_idx[idx3(src_i,src_j,src_k,nx,ny,nz)];++pull_fluid;}
                else{pull=-1;is_solid=1;++pull_solid;}
                h_pull=hash_i32(h_pull,pull);h_solid=hash_u8(h_solid,is_solid);

                int ni=wrap(i+e[q][0],nx),nj=wrap(j+e[q][1],ny),nk=k+e[q][2];
                int32_t ff; uint8_t ads=0;
                if(nk<0||nk>=nz){ff=-1;++ff_oob;}
                else if(solid[idx3(ni,nj,nk,nx,ny,nz)]==0){ff=grid_to_idx[idx3(ni,nj,nk,nx,ny,nz)];++ff_fluid;}
                else{ff=-1;ads=1;++ff_solid;}
                h_ff=hash_i32(h_ff,ff);h_ads=hash_u8(h_ads,ads);
            }
        }

        auto hex=[](uint64_t x){std::ostringstream o;o<<std::hex<<std::setw(16)<<std::setfill('0')<<x;return o.str();};
        std::cout<<"[Hash] fluid_xyz      = "<<hex(h_xyz)<<"\n";
        std::cout<<"[Hash] grid_to_idx     = "<<hex(h_grid)<<"\n";
        std::cout<<"[Hash] pull_nb_list    = "<<hex(h_pull)<<"\n";
        std::cout<<"[Hash] is_solid_nb     = "<<hex(h_solid)<<"\n";
        std::cout<<"[Hash] ff_nb_list      = "<<hex(h_ff)<<"\n";
        std::cout<<"[Hash] ads_solid_nb    = "<<hex(h_ads)<<"\n";
        std::cout<<"[Neighbor] pull fluid/solid/oob = "<<pull_fluid<<" / "<<pull_solid<<" / "<<pull_oob<<"\n";
        std::cout<<"[Neighbor] ff   fluid/solid/oob = "<<ff_fluid<<" / "<<ff_solid<<" / "<<ff_oob<<"\n";
        std::cout<<"[Sample] first fluid_xyz = ("<<fluid_xyz.front()[0]<<","<<fluid_xyz.front()[1]<<","<<fluid_xyz.front()[2]<<")\n";
        std::cout<<"[Sample] last  fluid_xyz = ("<<fluid_xyz.back()[0]<<","<<fluid_xyz.back()[1]<<","<<fluid_xyz.back()[2]<<")\n";
        std::cout<<"================================================================================\n";
        std::cout<<"[PASS] Native Stage-1 preprocessing completed. No Python/Taichi used.\n";
        std::cout<<"================================================================================\n";
        return 0;
    }catch(const std::exception& e){
        std::cerr<<"\n[ERROR] "<<e.what()<<"\n";
        return 1;
    }
}
