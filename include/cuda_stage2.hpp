#pragma once
#include <array>
#include <cstdint>
#include <vector>

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
    const std::vector<float>& rho_init
);
