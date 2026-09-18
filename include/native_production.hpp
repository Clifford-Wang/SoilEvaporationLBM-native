#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct NativeRunOptions{
    int total_steps=200000;
    int record_interval=1000;
    int print_interval=1000;
    int vtk_interval=10000;
    int checkpoint_interval=100000;
    int keep_checkpoints=2;
    int save_vtk=1;
    double device_memory_fraction=0.86;
    int force=0;
    bool verbose=true;
    std::string case_name;
    std::string output_dir;
    std::string output_prefix;
    std::string signature;
};

struct NativeRunResult{
    int final_step=0;
    bool resumed=false;
    double wall_time_sec=0.0;
    double final_saturation_threshold=0.0;
    double final_saturation_equiv=0.0;
    double final_liquid_mass_equiv=0.0;
    double final_ER_liquid_equiv_lu=0.0;
    double final_J_soil_lu=0.0;
    double final_J_out_mass_lu=0.0;
    double final_J_top_direct_lu=0.0;
    double final_mass_partition_error_rel=0.0;
    double final_bc_mass_balance_error_rel=0.0;
    double rho_min_final=0.0;
    double rho_max_final=0.0;
    std::string stats_csv;
};

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
    const NativeRunOptions& opt
);
