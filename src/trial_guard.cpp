#include "trial_guard.hpp"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <wincrypt.h>
#include <wincred.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstring>

#ifndef SOIL_LBM_BUILD_UNIX
#define SOIL_LBM_BUILD_UNIX 0
#endif

namespace fs=std::filesystem;

namespace trial_guard {
namespace {

constexpr uint64_t K_MAGIC=0x314C4152544D424Cull; // "LBMTRAL1" encoded
constexpr uint32_t K_VERSION=2;
constexpr uint64_t K_TRIAL_SECONDS=5ull*24ull*60ull*60ull;
constexpr uint64_t K_ROLLBACK_TOLERANCE=300ull;
constexpr uint64_t K_BUILD_CLOCK_TOLERANCE=24ull*60ull*60ull;
constexpr uint32_t K_FLAG_EXPIRED=1u;
constexpr uint32_t K_FLAG_TAMPER=2u;

#pragma pack(push,1)
struct TrialState{
    uint64_t magic=K_MAGIC;
    uint32_t version=K_VERSION;
    uint32_t flags=0;
    uint64_t first_run=0;
    uint64_t last_seen=0;
    uint64_t active_seconds=0;
    uint64_t machine_hash=0;
    uint64_t generation=0;
    uint64_t checksum=0;
};
#pragma pack(pop)

struct RawStore{
    bool present=false;
    bool readable=true;
    std::vector<unsigned char> blob;
};

TrialState g_state{};
bool g_initialized=false;
uint64_t g_active_at_init=0;
std::chrono::steady_clock::time_point g_session_start{};
std::chrono::steady_clock::time_point g_last_persist{};

uint64_t now_unix(){
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

uint64_t fnv64(const void* data,size_t n,uint64_t h=1469598103934665603ull){
    const auto* p=static_cast<const unsigned char*>(data);
    for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ull;}
    return h;
}

std::wstring read_machine_guid(){
    HKEY key=nullptr;
    if(RegOpenKeyExW(HKEY_LOCAL_MACHINE,L"SOFTWARE\\Microsoft\\Cryptography",0,KEY_READ|KEY_WOW64_64KEY,&key)!=ERROR_SUCCESS)
        return L"";
    wchar_t buf[256]{};
    DWORD type=0,cb=sizeof(buf);
    LONG rc=RegQueryValueExW(key,L"MachineGuid",nullptr,&type,reinterpret_cast<LPBYTE>(buf),&cb);
    RegCloseKey(key);
    if(rc!=ERROR_SUCCESS||(type!=REG_SZ&&type!=REG_EXPAND_SZ))return L"";
    return std::wstring(buf);
}

std::wstring computer_name(){
    wchar_t buf[256]{};
    DWORD n=static_cast<DWORD>(std::size(buf));
    if(!GetComputerNameW(buf,&n))return L"";
    return std::wstring(buf,n);
}

uint32_t system_volume_serial(){
    wchar_t win[MAX_PATH]{};
    if(!GetWindowsDirectoryW(win,MAX_PATH))return 0;
    wchar_t root[]=L"C:\\";
    root[0]=win[0];
    DWORD serial=0;
    GetVolumeInformationW(root,nullptr,0,&serial,nullptr,nullptr,nullptr,0);
    return serial;
}

std::wstring machine_material(){
    std::wstring s=L"SoilEvaporationLBM|";
    s+=read_machine_guid();
    s+=L"|";
    s+=computer_name();
    s+=L"|";
    s+=std::to_wstring(system_volume_serial());
    s+=L"|7E0E99D8-5DBB-4A94-8BD5-1C6A4B87D2B1";
    return s;
}

uint64_t machine_hash(){
    auto s=machine_material();
    return fnv64(s.data(),s.size()*sizeof(wchar_t));
}

uint64_t state_checksum(TrialState s){
    s.checksum=0;
    uint64_t h=0xD4A7B93C61E25F08ull;
    h=fnv64(&s,sizeof(s),h);
    const uint64_t k1=0x8F2C61D9A73B405Eull,k2=0x35E9C7A14D826BF0ull;
    h=fnv64(&k1,sizeof(k1),h);
    h=fnv64(&k2,sizeof(k2),h);
    return h;
}

bool valid_plain_state(const TrialState& s){
    return s.magic==K_MAGIC&&s.version==K_VERSION&&s.first_run>0&&
           s.machine_hash==machine_hash()&&s.checksum==state_checksum(s);
}

std::vector<unsigned char> entropy_bytes(){
    auto s=machine_material();
    const auto* p=reinterpret_cast<const unsigned char*>(s.data());
    return std::vector<unsigned char>(p,p+s.size()*sizeof(wchar_t));
}

bool protect_state(const TrialState& state,std::vector<unsigned char>& out){
    TrialState s=state;
    s.checksum=state_checksum(s);
    auto entropy=entropy_bytes();
    DATA_BLOB in{static_cast<DWORD>(sizeof(s)),reinterpret_cast<BYTE*>(&s)};
    DATA_BLOB ent{static_cast<DWORD>(entropy.size()),entropy.data()};
    DATA_BLOB enc{};
    if(!CryptProtectData(&in,L"SoilEvaporationLBM runtime state",&ent,nullptr,nullptr,
                         CRYPTPROTECT_LOCAL_MACHINE|CRYPTPROTECT_UI_FORBIDDEN,&enc))return false;
    out.assign(enc.pbData,enc.pbData+enc.cbData);
    LocalFree(enc.pbData);
    return true;
}

bool unprotect_state(const std::vector<unsigned char>& blob,TrialState& state){
    if(blob.empty())return false;
    auto entropy=entropy_bytes();
    DATA_BLOB in{static_cast<DWORD>(blob.size()),const_cast<BYTE*>(blob.data())};
    DATA_BLOB ent{static_cast<DWORD>(entropy.size()),entropy.data()};
    DATA_BLOB dec{};
    LPWSTR desc=nullptr;
    if(!CryptUnprotectData(&in,&desc,&ent,nullptr,nullptr,CRYPTPROTECT_UI_FORBIDDEN,&dec))return false;
    bool ok=dec.cbData==sizeof(TrialState);
    if(ok)std::memcpy(&state,dec.pbData,sizeof(TrialState));
    if(desc)LocalFree(desc);
    LocalFree(dec.pbData);
    return ok&&valid_plain_state(state);
}

fs::path env_path(const wchar_t* name){
    DWORD n=GetEnvironmentVariableW(name,nullptr,0);
    if(!n)return {};
    std::wstring s(n,L'\0');
    DWORD got=GetEnvironmentVariableW(name,s.data(),n);
    if(!got)return {};
    s.resize(got);
    return fs::path(s);
}

fs::path local_state_path(){
    auto p=env_path(L"LOCALAPPDATA");
    return p.empty()?p:p/L"SoilEvaporationLBM"/L".runtime.dat";
}

fs::path program_state_path(){
    auto p=env_path(L"PROGRAMDATA");
    return p.empty()?p:p/L"SoilEvaporationLBM"/L".runtime.dat";
}

bool write_file_store(const fs::path& p,const std::vector<unsigned char>& blob){
    if(p.empty())return false;
    std::error_code ec;
    fs::create_directories(p.parent_path(),ec);
    if(ec)return false;
    auto tmp=p;
    tmp+=L".tmp";
    {
        std::ofstream out(tmp,std::ios::binary|std::ios::trunc);
        if(!out)return false;
        out.write(reinterpret_cast<const char*>(blob.data()),static_cast<std::streamsize>(blob.size()));
        out.flush();
        if(!out)return false;
    }
    if(!MoveFileExW(tmp.c_str(),p.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)){
        DeleteFileW(tmp.c_str());
        return false;
    }
    SetFileAttributesW(p.c_str(),FILE_ATTRIBUTE_HIDDEN|FILE_ATTRIBUTE_SYSTEM);
    return true;
}

RawStore read_file_store(const fs::path& p){
    RawStore r;
    if(p.empty())return r;
    std::error_code ec;
    if(!fs::exists(p,ec)||ec)return r;
    r.present=true;
    std::ifstream in(p,std::ios::binary);
    if(!in){r.readable=false;return r;}
    r.blob.assign(std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>());
    if(r.blob.empty())r.readable=false;
    return r;
}

bool write_registry_store(const std::vector<unsigned char>& blob){
    HKEY key=nullptr;
    DWORD disp=0;
    if(RegCreateKeyExW(HKEY_CURRENT_USER,L"Software\\SoilEvaporationLBM\\Runtime",0,nullptr,0,
                       KEY_SET_VALUE,nullptr,&key,&disp)!=ERROR_SUCCESS)return false;
    LONG rc=RegSetValueExW(key,L"State",0,REG_BINARY,blob.data(),static_cast<DWORD>(blob.size()));
    RegCloseKey(key);
    return rc==ERROR_SUCCESS;
}

RawStore read_registry_store(){
    RawStore r;
    HKEY key=nullptr;
    if(RegOpenKeyExW(HKEY_CURRENT_USER,L"Software\\SoilEvaporationLBM\\Runtime",0,KEY_QUERY_VALUE,&key)!=ERROR_SUCCESS)
        return r;
    DWORD type=0,cb=0;
    LONG rc=RegQueryValueExW(key,L"State",nullptr,&type,nullptr,&cb);
    if(rc==ERROR_FILE_NOT_FOUND){RegCloseKey(key);return r;}
    r.present=true;
    if(rc!=ERROR_SUCCESS||type!=REG_BINARY||cb==0){r.readable=false;RegCloseKey(key);return r;}
    r.blob.resize(cb);
    rc=RegQueryValueExW(key,L"State",nullptr,&type,r.blob.data(),&cb);
    RegCloseKey(key);
    if(rc!=ERROR_SUCCESS){r.readable=false;r.blob.clear();}
    return r;
}

constexpr wchar_t K_CRED_TARGET[]=L"SoilEvaporationLBM/RuntimeState/v2";

bool write_credential_store(const std::vector<unsigned char>& blob){
    CREDENTIALW c{};
    c.Type=CRED_TYPE_GENERIC;
    c.TargetName=const_cast<LPWSTR>(K_CRED_TARGET);
    c.CredentialBlobSize=static_cast<DWORD>(blob.size());
    c.CredentialBlob=const_cast<LPBYTE>(blob.data());
    c.Persist=CRED_PERSIST_LOCAL_MACHINE;
    c.UserName=const_cast<LPWSTR>(L"SoilEvaporationLBM");
    return CredWriteW(&c,0)!=FALSE;
}

RawStore read_credential_store(){
    RawStore r;
    PCREDENTIALW c=nullptr;
    if(!CredReadW(K_CRED_TARGET,CRED_TYPE_GENERIC,0,&c)){
        if(GetLastError()==ERROR_NOT_FOUND)return r;
        return r;
    }
    r.present=true;
    if(!c||!c->CredentialBlob||c->CredentialBlobSize==0){r.readable=false;}
    else r.blob.assign(c->CredentialBlob,c->CredentialBlob+c->CredentialBlobSize);
    if(c)CredFree(c);
    return r;
}

int persist_state(TrialState& s){
    s.machine_hash=machine_hash();
    s.checksum=state_checksum(s);
    std::vector<unsigned char> blob;
    if(!protect_state(s,blob))return 0;
    int ok=0;
    ok+=write_registry_store(blob)?1:0;
    ok+=write_file_store(local_state_path(),blob)?1:0;
    ok+=write_file_store(program_state_path(),blob)?1:0;
    ok+=write_credential_store(blob)?1:0;
    return ok;
}

std::string expired_message(){
    return std::string(
        "\nEvaluation period expired.\n\n"
        "The 5-day evaluation license has expired.\n"
        "For continued use, please contact:\n\n"
        "clifford220810@gmail.com\n");
}

std::string tamper_message(){
    return std::string(
        "\nLicense validation failed.\n\n"
        "The local evaluation state or system clock could not be validated.\n"
        "For assistance, please contact:\n\n"
        "clifford220810@gmail.com\n");
}

[[noreturn]] void lock_and_throw(uint32_t flag,const std::string& message){
    uint64_t now=now_unix();
    if(g_state.first_run==0)g_state.first_run=now;
    g_state.last_seen=std::max(g_state.last_seen,now);
    g_state.machine_hash=machine_hash();
    g_state.flags|=flag|K_FLAG_EXPIRED;
    ++g_state.generation;
    persist_state(g_state);
    throw TrialViolation(message);
}

uint64_t current_active_seconds(){
    if(!g_initialized)return g_state.active_seconds;
    auto elapsed=std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now()-g_session_start).count();
    if(elapsed<0)elapsed=0;
    return g_active_at_init+static_cast<uint64_t>(elapsed);
}

void validate_current_or_throw(bool force_persist){
    const uint64_t now=now_unix();
    g_state.active_seconds=current_active_seconds();

    if((g_state.flags&(K_FLAG_EXPIRED|K_FLAG_TAMPER))!=0)
        lock_and_throw(g_state.flags,expired_message());

    if(now+K_ROLLBACK_TOLERANCE<g_state.last_seen)
        lock_and_throw(K_FLAG_TAMPER,tamper_message());

    if(now<g_state.first_run)
        lock_and_throw(K_FLAG_TAMPER,tamper_message());

    const uint64_t wall_elapsed=now-g_state.first_run;
    if(wall_elapsed>=K_TRIAL_SECONDS||g_state.active_seconds>=K_TRIAL_SECONDS)
        lock_and_throw(K_FLAG_EXPIRED,expired_message());

    g_state.last_seen=std::max(g_state.last_seen,now);

    const auto steady_now=std::chrono::steady_clock::now();
    if(force_persist||steady_now-g_last_persist>=std::chrono::minutes(5)){
        ++g_state.generation;
        if(persist_state(g_state)<2)
            throw TrialViolation("\nLicense state could not be securely stored.\nPlease contact:\n\nclifford220810@gmail.com\n");
        g_last_persist=steady_now;
    }
}

} // namespace

void initialize_or_throw(){
    if(g_initialized){validate_current_or_throw(true);return;}

    const uint64_t now=now_unix();
    const uint64_t build=static_cast<uint64_t>(SOIL_LBM_BUILD_UNIX);
    if(build>0&&now+K_BUILD_CLOCK_TOLERANCE<build){
        g_state.first_run=now;
        g_state.last_seen=now;
        g_state.machine_hash=machine_hash();
        lock_and_throw(K_FLAG_TAMPER,tamper_message());
    }

    std::vector<RawStore> stores;
    stores.push_back(read_registry_store());
    stores.push_back(read_file_store(local_state_path()));
    stores.push_back(read_file_store(program_state_path()));
    stores.push_back(read_credential_store());

    std::vector<TrialState> valid;
    bool invalid_present=false;
    for(const auto& r:stores){
        if(!r.present)continue;
        if(!r.readable){invalid_present=true;continue;}
        TrialState s{};
        if(unprotect_state(r.blob,s))valid.push_back(s);
        else invalid_present=true;
    }

    if(invalid_present){
        if(!valid.empty())g_state=valid.front();
        else{
            g_state.first_run=now;
            g_state.last_seen=now;
            g_state.machine_hash=machine_hash();
        }
        lock_and_throw(K_FLAG_TAMPER,tamper_message());
    }

    if(valid.empty()){
        g_state=TrialState{};
        g_state.first_run=now;
        g_state.last_seen=now;
        g_state.machine_hash=machine_hash();
        g_state.generation=1;
        if(persist_state(g_state)<2)
            throw TrialViolation("\nLicense state could not be securely initialized.\nPlease contact:\n\nclifford220810@gmail.com\n");
    }else{
        g_state=valid.front();
        uint64_t min_first=g_state.first_run,max_first=g_state.first_run;
        uint64_t max_last=g_state.last_seen,max_active=g_state.active_seconds,max_gen=g_state.generation;
        uint32_t flags=g_state.flags;
        for(const auto& s:valid){
            min_first=std::min(min_first,s.first_run);
            max_first=std::max(max_first,s.first_run);
            max_last=std::max(max_last,s.last_seen);
            max_active=std::max(max_active,s.active_seconds);
            max_gen=std::max(max_gen,s.generation);
            flags|=s.flags;
        }
        g_state.first_run=min_first;
        g_state.last_seen=max_last;
        g_state.active_seconds=max_active;
        g_state.generation=max_gen;
        g_state.flags=flags;
        if(max_first-min_first>K_ROLLBACK_TOLERANCE)
            lock_and_throw(K_FLAG_TAMPER,tamper_message());
    }

    g_initialized=true;
    g_active_at_init=g_state.active_seconds;
    g_session_start=std::chrono::steady_clock::now();
    g_last_persist=g_session_start;
    validate_current_or_throw(true);
}

void heartbeat_or_throw(){
    if(!g_initialized)initialize_or_throw();
    validate_current_or_throw(false);
}

const char* contact_email(){
    return "clifford220810@gmail.com";
}

} // namespace trial_guard

#else

namespace trial_guard {
void initialize_or_throw(){throw TrialViolation("This licensed build supports Windows only.");}
void heartbeat_or_throw(){throw TrialViolation("This licensed build supports Windows only.");}
const char* contact_email(){return "clifford220810@gmail.com";}
}

#endif
