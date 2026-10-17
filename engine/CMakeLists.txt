cmake_minimum_required(VERSION 3.24)
project(mcengine VERSION 0.1.0 LANGUAGES CXX)

option(MC_ENABLE_CUDA "Build the CUDA backend when a CUDA compiler is available" ON)
option(MC_BUILD_TESTS "Build GoogleTest unit tests" ON)
option(MC_BUILD_BENCH "Build the CPU vs GPU benchmark" ON)
option(MC_BUILD_PYTHON "Build the pybind11 module" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------- CUDA ------
set(MC_HAS_CUDA OFF)
if(MC_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # Turing (T4), Ampere (A100 / RTX 30xx), Ada (RTX 40xx / L4)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(MC_HAS_CUDA ON)
  else()
    message(WARNING "No CUDA compiler found: building the CPU engine only")
  endif()
endif()

# ---------------------------------------------------------------- core ------
add_library(mc_core STATIC src/cpu_engine.cpp)
target_include_directories(mc_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
find_package(Threads REQUIRED)
target_link_libraries(mc_core PUBLIC Threads::Threads)

if(MC_HAS_CUDA)
  target_sources(mc_core PRIVATE src/gpu_engine.cu src/kernels.cu)
  target_link_libraries(mc_core PUBLIC CUDA::cudart)
  target_compile_definitions(mc_core PUBLIC MC_WITH_CUDA=1)
  target_compile_options(mc_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>)
else()
  target_sources(mc_core PRIVATE src/gpu_stub.cpp)
endif()

if(MSVC)
  target_compile_options(mc_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2 /W4 /permissive->)
else()
  target_compile_options(mc_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>)
endif()

message(STATUS "mcengine: CUDA backend = ${MC_HAS_CUDA}")

include(FetchContent)

# ---------------------------------------------------------------- bench -----
if(MC_BUILD_BENCH)
  add_executable(mc_bench bench/bench_main.cpp)
  target_link_libraries(mc_bench PRIVATE mc_core)
endif()

# ---------------------------------------------------------------- tests -----
if(MC_BUILD_TESTS)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(mc_tests tests/test_cpu_engine.cpp tests/test_gpu_engine.cpp)
  target_link_libraries(mc_tests PRIVATE mc_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(mc_tests)
endif()

# ---------------------------------------------------------------- python ----
if(MC_BUILD_PYTHON)
  find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
  set(PYBIND11_FINDPYTHON ON)
  FetchContent_Declare(pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(pybind11)

  pybind11_add_module(mcengine src/bindings.cpp)
  target_link_libraries(mcengine PRIVATE mc_core)
  # Drop the compiled module straight into the Python package (no per-config subdir).
  set(OSB_PACKAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../backtester/osb)
  set_target_properties(mcengine PROPERTIES LIBRARY_OUTPUT_DIRECTORY "$<1:${OSB_PACKAGE_DIR}>")
endif()
