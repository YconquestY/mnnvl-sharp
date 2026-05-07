#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace mnnvl {

constexpr int kCanaryWords = 8;

void LaunchInitE4M3(uint32_t* dst_words, std::size_t word_count, int world_rank, cudaStream_t stream);
void LaunchAliasFence(cudaStream_t stream);
void LaunchSharpAllReduceE4M3(uint32_t* mc_alias_words, std::size_t word_count, cudaStream_t stream);
void LaunchCanary(uint32_t* self_slot_words, uint32_t rank_tag, cudaStream_t stream);
void LaunchHash(const uint32_t* words, std::size_t word_count, uint64_t* out_hash, cudaStream_t stream);

}  // namespace mnnvl
