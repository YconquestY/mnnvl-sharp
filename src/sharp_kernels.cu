#include "sharp_kernels.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "runtime.h"

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

__device__ uint64_t as_global_address(const void* ptr) {
  uint64_t addr = 0;
  asm("cvta.to.global.u64 %0, %1;" : "=l"(addr) : "l"(ptr));
  return addr;
}

__global__ void init_e4m3_kernel(uint32_t* dst_words, std::size_t word_count, int world_rank) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < word_count; i += stride) {
    dst_words[i] = deterministic_e4m3_word(world_rank, i);
  }
}

__global__ void alias_fence_kernel() {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    asm volatile("fence.proxy.alias;" : : : "memory");
#endif
  }
}

__global__ void sharp_allreduce_e4m3_kernel(uint32_t* mc_alias_words, std::size_t word_count) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < word_count; i += stride) {
    uint32_t reduced = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const uint64_t addr = as_global_address(mc_alias_words + i);
    asm volatile("multimem.ld_reduce.weak.global.add.acc::f16.e4m3x4 %0, [%1];"
                 : "=r"(reduced)
                 : "l"(addr)
                 : "memory");
    asm volatile("multimem.st.weak.global.e4m3x4 [%0], %1;"
                 :
                 : "l"(addr), "r"(reduced)
                 : "memory");
#else
    (void)mc_alias_words;
    reduced = 0;
#endif
  }
}

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

void LaunchAliasFence(cudaStream_t stream) {
  alias_fence_kernel<<<1, 1, 0, stream>>>();
  CheckLaunch("alias_fence_kernel");
}

void LaunchSharpAllReduceE4M3(uint32_t* mc_alias_words, std::size_t word_count, cudaStream_t stream) {
  sharp_allreduce_e4m3_kernel<<<BlocksFor(word_count), 256, 0, stream>>>(mc_alias_words, word_count);
  CheckLaunch("sharp_allreduce_e4m3_kernel");
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
