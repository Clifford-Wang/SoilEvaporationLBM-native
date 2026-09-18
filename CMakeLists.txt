cmake_minimum_required(VERSION 3.24)
project(SoilEvaporationLBMNativeStage2 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(SoilEvaporationLBM_stage2
  src/main.cpp
  src/cuda_stage2.cu
)

target_include_directories(SoilEvaporationLBM_stage2 PRIVATE include)
target_link_libraries(SoilEvaporationLBM_stage2 PRIVATE CUDA::cudart_static)
set_property(TARGET SoilEvaporationLBM_stage2 PROPERTY CUDA_RUNTIME_LIBRARY Static)
set_property(TARGET SoilEvaporationLBM_stage2 PROPERTY CUDA_ARCHITECTURES 89-real)
set_property(TARGET SoilEvaporationLBM_stage2 PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

if(MSVC)
  target_compile_options(SoilEvaporationLBM_stage2 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/O2 /GL /EHsc /utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use-local-env>
  )
  target_link_options(SoilEvaporationLBM_stage2 PRIVATE
    $<$<CONFIG:Release>:/LTCG /OPT:REF /OPT:ICF>
  )
endif()
