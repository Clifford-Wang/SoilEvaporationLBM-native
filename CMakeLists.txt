cmake_minimum_required(VERSION 3.24)
project(SoilEvaporationLBM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

string(TIMESTAMP SOIL_LBM_BUILD_UNIX "%s" UTC)

add_executable(SoilEvaporationLBM
  src/main.cpp
  src/native_production.cu
  src/trial_guard.cpp
)

target_include_directories(SoilEvaporationLBM PRIVATE include)
target_compile_definitions(SoilEvaporationLBM PRIVATE SOIL_LBM_BUILD_UNIX=${SOIL_LBM_BUILD_UNIX})
target_link_libraries(SoilEvaporationLBM PRIVATE CUDA::cudart_static advapi32 crypt32)
set_property(TARGET SoilEvaporationLBM PROPERTY CUDA_RUNTIME_LIBRARY Static)
# One binary for common NVIDIA workstation/laptop generations:
# Turing RTX20 (75), Ampere A100/RTX30 (80/86), Ada RTX40 (89), Hopper (90), Blackwell RTX50 (120).
set_property(TARGET SoilEvaporationLBM PROPERTY CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real;120-real")
set_property(TARGET SoilEvaporationLBM PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

if(MSVC)
  target_compile_options(SoilEvaporationLBM PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/O2 /GL /EHsc /utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use-local-env>
  )
  target_link_options(SoilEvaporationLBM PRIVATE
    $<$<CONFIG:Release>:/LTCG /OPT:REF /OPT:ICF>
  )
endif()
