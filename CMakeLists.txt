cmake_minimum_required(VERSION 3.27)

project(mnnvl_sharp_allreduce LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit 13.1 REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

execute_process(
  COMMAND "${CMAKE_CUDA_COMPILER}" --list-gpu-code
  OUTPUT_VARIABLE MNNVL_NVCC_GPU_CODE
  ERROR_VARIABLE MNNVL_NVCC_GPU_CODE_ERR
  RESULT_VARIABLE MNNVL_NVCC_GPU_CODE_RC
)
if(NOT MNNVL_NVCC_GPU_CODE_RC EQUAL 0)
  message(FATAL_ERROR "Could not query nvcc GPU code support: ${MNNVL_NVCC_GPU_CODE_ERR}")
endif()

execute_process(
  COMMAND "${CMAKE_CUDA_COMPILER}" --help
  OUTPUT_VARIABLE MNNVL_NVCC_HELP
  ERROR_VARIABLE MNNVL_NVCC_HELP_ERR
  RESULT_VARIABLE MNNVL_NVCC_HELP_RC
)
if(NOT MNNVL_NVCC_HELP_RC EQUAL 0)
  message(FATAL_ERROR "Could not query nvcc architecture help: ${MNNVL_NVCC_HELP_ERR}")
endif()

set(MNNVL_BLACKWELL_ARCH "")
foreach(candidate 100a 100f 110a 110f 103a 103f 120a 120f 121a 121f)
  if(MNNVL_NVCC_HELP MATCHES "sm_${candidate}")
    set(MNNVL_BLACKWELL_ARCH "${candidate}")
    break()
  endif()
endforeach()
if(MNNVL_BLACKWELL_ARCH STREQUAL "")
  message(FATAL_ERROR "CUDA compiler does not expose a feature-specific Blackwell target such as sm_100a required for multimem FP8")
endif()

set(MNNVL_TMA_ASYNC_CHECK "${CMAKE_BINARY_DIR}/mnnvl_check_tma_async.cu")
file(WRITE "${MNNVL_TMA_ASYNC_CHECK}" [[
#include <cuda/ptx>
#include <cuda_fp16.h>
#include <stdint.h>
__device__ void check_tma(__half* dst, const __half* src, uint32_t bytes) {
  const uint64_t g = static_cast<uint64_t>(__cvta_generic_to_global(dst));
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(src));
  asm volatile("multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [%0], [%1], %2;"
               : : "l"(g), "r"(s), "r"(bytes) : "memory");
  cuda::ptx::cp_async_bulk_commit_group();
  cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
  cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
}
]])
execute_process(
  COMMAND "${CMAKE_CUDA_COMPILER}" -std=c++20 "-arch=sm_${MNNVL_BLACKWELL_ARCH}"
          -c "${MNNVL_TMA_ASYNC_CHECK}" -o "${CMAKE_BINARY_DIR}/mnnvl_check_tma_async.o"
  OUTPUT_VARIABLE MNNVL_TMA_ASYNC_CHECK_OUT
  ERROR_VARIABLE MNNVL_TMA_ASYNC_CHECK_ERR
  RESULT_VARIABLE MNNVL_TMA_ASYNC_CHECK_RC
)
if(MNNVL_TMA_ASYNC_CHECK_RC EQUAL 0)
  set(MNNVL_SHARP_HAS_TMA_ASYNC 1)
else()
  set(MNNVL_SHARP_HAS_TMA_ASYNC 0)
  message(WARNING "CUDA compiler does not accept multimem.cp.reduce.async.bulk; tma_async backend will report unsupported: ${MNNVL_TMA_ASYNC_CHECK_ERR}")
endif()

add_executable(mnnvl_sharp_allreduce
  src/config.cc
  src/e4m3_ref.cc
  src/fabric_memory.cc
  src/main.cc
  src/rack_config.cc
  src/runtime.cc
  src/sharp_kernels.cu
)

target_include_directories(mnnvl_sharp_allreduce PRIVATE src)
set_target_properties(mnnvl_sharp_allreduce PROPERTIES
  CUDA_ARCHITECTURES "${MNNVL_BLACKWELL_ARCH}"
  CUDA_SEPARABLE_COMPILATION OFF
)

target_compile_definitions(mnnvl_sharp_allreduce PRIVATE
  MNNVL_SHARP_HAS_E4M3=1
  MNNVL_SHARP_HAS_TMA_ASYNC=${MNNVL_SHARP_HAS_TMA_ASYNC}
)

target_compile_options(mnnvl_sharp_allreduce PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_link_libraries(mnnvl_sharp_allreduce PRIVATE
  CUDA::cudart
  CUDA::cuda_driver
  MPI::MPI_CXX
  yaml-cpp
  Threads::Threads
)
