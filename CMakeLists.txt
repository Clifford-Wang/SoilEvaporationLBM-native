cmake_minimum_required(VERSION 3.20)
project(SoilEvaporationLBMNativeStage1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(SoilEvaporationLBM_stage1 src/main.cpp)

if(MSVC)
  set_property(TARGET SoilEvaporationLBM_stage1 PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_compile_options(SoilEvaporationLBM_stage1 PRIVATE
    $<$<CONFIG:Release>:/O2 /GL /EHsc /utf-8>
  )
  target_link_options(SoilEvaporationLBM_stage1 PRIVATE
    $<$<CONFIG:Release>:/LTCG /OPT:REF /OPT:ICF>
  )
else()
  target_compile_options(SoilEvaporationLBM_stage1 PRIVATE -O2)
endif()
