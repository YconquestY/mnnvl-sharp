#include "sharp_kernels.h"

#include <cuda/ptx>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "runtime.h"

#ifndef MNNVL_SHARP_HAS_TMA_ASYNC
#define MNNVL_SHARP_HAS_TMA_ASYNC 0
#endif

namespace mnnvl {
namespace {

__device__ uint8_t deterministic_e4m3_byte(int rank, std::size_t element_index) {
  const uint32_t mix = static_cast<uint32_t>((element_index * 17u) ^ (element_index >> 7) ^
                                            (static_cast<std::size_t>(rank + 1) * 29u));
  const uint8_t sign = ((mix >> 4) & 1u) ? 0x80u : 0x00u;
  const uint8_t exp = static_cast<uint8_t>(3u + (mix % 3u));
  const uint8_t mant = static_cast<uint8_t>((mix / 3u) & 0x7u);
  return static_cast<uint8_t>(sign | (exp << 3) | mant);
}

__device__ uint32_t deterministic_e4m3_word(int rank, std::size_t word_index) {
  uint32_t word = 0;
  for (int lane = 0; lane < 4; ++lane) {
    word |= static_cast<uint32_t>(deterministic_e4m3_byte(rank, word_index * 4 + lane)) << (lane * 8);
  }
  return word;
}

__device__ uint16_t deterministic_f16_bits(int rank, std::size_t element_index) {
  constexpr uint16_t kValues[] = {
      0xb800u,  // -0.5
      0xb400u,  // -0.25
      0xb000u,  // -0.125
      0x0000u,  //  0.0
      0x3000u,  //  0.125
      0x3400u,  //  0.25
      0x3800u,  //  0.5
      0x0000u,  //  0.0
  };
  const uint32_t mix = static_cast<uint32_t>((element_index * 13u) ^ (element_index >> 5) ^
                                            (static_cast<std::size_t>(rank + 1) * 7u));
  return kValues[mix & 7u];
}

__device__ uint32_t multimem_ld_reduce_e4m3x4_acc_f16(const uint32_t* addr) {
  uint32_t reduced = 0;
  const uint64_t global_addr = static_cast<uint64_t>(__cvta_generic_to_global(addr));
  // CUDA 13.1 CCCL does not wrap the FP8 SHARP form with acc::f16.e4m3x4.
  asm volatile("multimem.ld_reduce.weak.global.add.acc::f16.e4m3x4 %0, [%1];"
               : "=r"(reduced)
               : "l"(global_addr)
               : "memory");
  return reduced;
}

#if MNNVL_SHARP_HAS_TMA_ASYNC
__device__ void multimem_cp_reduce_async_bulk_add_noftz_f16(__half* dst, const __half* src, uint32_t bytes) {
  const uint64_t global_addr = static_cast<uint64_t>(__cvta_generic_to_global(dst));
  const uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
  // CUDA 13.1 CCCL wraps cp.reduce.async.bulk, but not the PTX 9.1 multimem form.
  asm volatile("multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [%0], [%1], %2;"
               :
               : "l"(global_addr), "r"(shared_addr), "r"(bytes)
               : "memory");
}
#endif

__global__ void init_e4m3_kernel(uint32_t* dst_words, std::size_t word_count, int world_rank) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < word_count; i += stride) {
    dst_words[i] = deterministic_e4m3_word(world_rank, i);
  }
}

__global__ void init_f16_kernel(uint16_t* dst, std::size_t element_count, int world_rank) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < element_count; i += stride) {
    dst[i] = deterministic_f16_bits(world_rank, i);
  }
}

__global__ void alias_fence_kernel() {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    cuda::ptx::fence_proxy_alias();
#endif
  }
}

__global__ void async_proxy_fence_kernel() {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cuda::ptx::fence_proxy_async(cuda::ptx::space_global);
#endif
  }
}

__global__ void sharp_allreduce_e4m3_kernel(uint32_t* mc_alias_words, std::size_t word_count) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < word_count; i += stride) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const uint32_t reduced = multimem_ld_reduce_e4m3x4_acc_f16(mc_alias_words + i);
    // The reduced e4m3x4 payload is already packed in one 32-bit register.
    cuda::ptx::multimem_st(cuda::ptx::sem_weak, mc_alias_words + i, reduced);
#else
    (void)mc_alias_words;
#endif
  }
}

#if MNNVL_SHARP_HAS_TMA_ASYNC
__global__ void sharp_allreduce_f16_tma_async_kernel(const uint16_t* input,
                                                     uint16_t* mc_output,
                                                     std::size_t element_count) {
  constexpr std::size_t kTileBytes = 4096;
  extern __shared__ __align__(16) unsigned char shared_tile[];

  const std::size_t total_bytes = element_count * sizeof(uint16_t);
  const std::size_t tile_count = (total_bytes + kTileBytes - 1) / kTileBytes;
  if (threadIdx.x == 0) {
    cuda::ptx::fence_proxy_async(cuda::ptx::space_global);
  }
  __syncthreads();

  for (std::size_t tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
    const std::size_t byte_offset = tile * kTileBytes;
    const std::size_t remaining = total_bytes - byte_offset;
    const std::size_t tile_bytes = remaining < kTileBytes ? remaining : kTileBytes;
    const std::size_t vector_count = tile_bytes / sizeof(uint4);
    const auto* src4 = reinterpret_cast<const uint4*>(
        reinterpret_cast<const unsigned char*>(input) + byte_offset);
    auto* shared4 = reinterpret_cast<uint4*>(shared_tile);
    for (std::size_t i = threadIdx.x; i < vector_count; i += blockDim.x) {
      shared4[i] = src4[i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
      auto* dst = reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(mc_output) + byte_offset);
      auto* src = reinterpret_cast<const __half*>(shared_tile);
      multimem_cp_reduce_async_bulk_add_noftz_f16(dst, src, static_cast<uint32_t>(tile_bytes));
      cuda::ptx::cp_async_bulk_commit_group();
      cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
      cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
    }
    __syncthreads();
  }
}
#endif

__global__ void canary_kernel(uint32_t* self_slot_words, uint32_t rank_tag) {
  const int i = threadIdx.x;
  if (blockIdx.x == 0 && i < kCanaryWords) {
    self_slot_words[i] = 0xC0010000u | ((rank_tag & 0xffu) << 8) | static_cast<uint32_t>(i);
  }
}

__device__ uint64_t mix64(uint64_t value) {
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdu;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53u;
  value ^= value >> 33;
  return value;
}

__global__ void hash_kernel(const uint32_t* words, std::size_t word_count, uint64_t* out_hash) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < word_count; i += stride) {
    const uint64_t value = (static_cast<uint64_t>(words[i]) << 32) ^ static_cast<uint64_t>(i);
    atomicXor(reinterpret_cast<unsigned long long*>(out_hash),
              static_cast<unsigned long long>(mix64(value)));
  }
}

int BlocksFor(std::size_t word_count) {
  constexpr int kThreads = 256;
  constexpr int kMaxBlocks = 4096;
  const std::size_t blocks = (word_count + kThreads - 1) / kThreads;
  return static_cast<int>(blocks < 1 ? 1 : (blocks > kMaxBlocks ? kMaxBlocks : blocks));
}

void CheckLaunch(const char* kernel_name) {
  const cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(kernel_name) + " launch failed: " + cudaGetErrorString(err));
  }
}

}  // namespace

void LaunchInitE4M3(uint32_t* dst_words, std::size_t word_count, int world_rank, cudaStream_t stream) {
  init_e4m3_kernel<<<BlocksFor(word_count), 256, 0, stream>>>(dst_words, word_count, world_rank);
  CheckLaunch("init_e4m3_kernel");
}

void LaunchInitF16(uint16_t* dst, std::size_t element_count, int world_rank, cudaStream_t stream) {
  init_f16_kernel<<<BlocksFor((element_count + 1) / 2), 256, 0, stream>>>(dst, element_count, world_rank);
  CheckLaunch("init_f16_kernel");
}

void LaunchAliasFence(cudaStream_t stream) {
  alias_fence_kernel<<<1, 1, 0, stream>>>();
  CheckLaunch("alias_fence_kernel");
}

void LaunchAsyncProxyFence(cudaStream_t stream) {
  async_proxy_fence_kernel<<<1, 1, 0, stream>>>();
  CheckLaunch("async_proxy_fence_kernel");
}

void LaunchSharpAllReduceE4M3(uint32_t* mc_alias_words, std::size_t word_count, cudaStream_t stream) {
  sharp_allreduce_e4m3_kernel<<<BlocksFor(word_count), 256, 0, stream>>>(mc_alias_words, word_count);
  CheckLaunch("sharp_allreduce_e4m3_kernel");
}

void LaunchSharpAllReduceF16TmaAsync(const uint16_t* input,
                                     uint16_t* mc_output,
                                     std::size_t element_count,
                                     cudaStream_t stream) {
#if MNNVL_SHARP_HAS_TMA_ASYNC
  if ((element_count % 8) != 0) {
    throw std::runtime_error("TMA async F16 element_count must be a multiple of 8");
  }
  constexpr std::size_t kTileBytes = 4096;
  constexpr int kThreads = 256;
  constexpr int kMaxBlocks = 4096;
  const std::size_t total_bytes = element_count * sizeof(uint16_t);
  const std::size_t tile_count = (total_bytes + kTileBytes - 1) / kTileBytes;
  const int blocks = static_cast<int>(tile_count < 1 ? 1 : (tile_count > kMaxBlocks ? kMaxBlocks : tile_count));
  sharp_allreduce_f16_tma_async_kernel<<<blocks, kThreads, kTileBytes, stream>>>(input, mc_output, element_count);
  CheckLaunch("sharp_allreduce_f16_tma_async_kernel");
#else
  (void)input;
  (void)mc_output;
  (void)element_count;
  (void)stream;
  throw std::runtime_error("TMA async SHARP backend was not compiled into this binary");
#endif
}

void LaunchCanary(uint32_t* self_slot_words, uint32_t rank_tag, cudaStream_t stream) {
  canary_kernel<<<1, kCanaryWords, 0, stream>>>(self_slot_words, rank_tag);
  CheckLaunch("canary_kernel");
}

void LaunchHash(const uint32_t* words, std::size_t word_count, uint64_t* out_hash, cudaStream_t stream) {
  MNNVL_CHECK_CUDA(cudaMemsetAsync(out_hash, 0, sizeof(uint64_t), stream));
  hash_kernel<<<BlocksFor(word_count), 256, 0, stream>>>(words, word_count, out_hash);
  CheckLaunch("hash_kernel");
}

}  // namespace mnnvl
