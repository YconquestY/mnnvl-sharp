#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mnnvl {

struct PrecisionMetrics {
  std::size_t exact_mismatches = 0;
  std::size_t requantized_mismatches = 0;
  int max_requantized_ulp_distance = 0;
  float max_abs_error = 0.0f;
  double mean_abs_error = 0.0;
};

uint8_t DeterministicE4M3Byte(int rank, std::size_t element_index);
uint32_t DeterministicE4M3Word(int rank, std::size_t word_index);
uint16_t DeterministicF16Bits(int rank, std::size_t element_index);

float DecodeE4M3(uint8_t bits);
uint8_t EncodeE4M3(float value);
float DecodeF16(uint16_t bits);
uint16_t EncodeF16(float value);
float RoundToFp16(float value);

std::vector<uint32_t> CpuAllReduceE4M3Semantic(std::size_t word_count, int ranks);
std::vector<float> CpuAllReduceE4M3F32(std::size_t element_count, int ranks);
std::vector<uint16_t> CpuAllReduceF16Semantic(std::size_t element_count, int ranks);
std::vector<float> CpuAllReduceF16F32(std::size_t element_count, int ranks);

PrecisionMetrics CompareE4M3(const uint32_t* result_words, std::size_t word_count, int ranks);
PrecisionMetrics CompareF16(const uint16_t* result, std::size_t element_count, int ranks);

}  // namespace mnnvl
