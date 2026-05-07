#include "e4m3_ref.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace mnnvl {
namespace {

int RoundTiesToEven(double value) {
  const double floor_value = std::floor(value);
  const double frac = value - floor_value;
  int rounded = static_cast<int>(floor_value);
  if (frac > 0.5) {
    ++rounded;
  } else if (frac == 0.5 && (rounded & 1) != 0) {
    ++rounded;
  }
  return rounded;
}

uint16_t FloatToHalfBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  int exp = static_cast<int>((bits >> 23) & 0xffu);
  uint32_t mant = bits & 0x7fffffu;

  if (exp == 0xff) {
    if (mant == 0) {
      return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | 0x7e00u);
  }

  exp = exp - 127 + 15;
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    mant |= 0x800000u;
    const int shift = 14 - exp;
    uint32_t half_mant = mant >> shift;
    const uint32_t remainder = mant & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (half_mant & 1u))) {
      ++half_mant;
    }
    return static_cast<uint16_t>(sign | half_mant);
  }

  uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
  const uint32_t remainder = mant & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) {
    ++half;
  }
  return static_cast<uint16_t>(half);
}

float HalfBitsToFloat(uint16_t half) {
  const uint32_t sign = (static_cast<uint32_t>(half & 0x8000u)) << 16;
  uint32_t exp = (half >> 10) & 0x1fu;
  uint32_t mant = half & 0x03ffu;
  uint32_t bits = 0;

  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 1;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x03ffu;
      bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }

  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint8_t SemanticByteForElement(std::size_t element_index, int ranks) {
  float acc = 0.0f;
  for (int rank = 0; rank < ranks; ++rank) {
    const float value = DecodeE4M3(DeterministicE4M3Byte(rank, element_index));
    acc = RoundToFp16(acc + value);
  }
  return EncodeE4M3(acc);
}

int OrderedFiniteE4M3Code(uint8_t bits) {
  const int magnitude = bits & 0x7f;
  if (bits & 0x80) {
    return 0x80 - magnitude;
  }
  return 0x80 + magnitude;
}

}  // namespace

uint8_t DeterministicE4M3Byte(int rank, std::size_t element_index) {
  const uint32_t mix = static_cast<uint32_t>((element_index * 17u) ^ (element_index >> 7) ^
                                            (static_cast<std::size_t>(rank + 1) * 29u));
  const uint8_t sign = ((mix >> 4) & 1u) ? 0x80u : 0x00u;
  const uint8_t exp = static_cast<uint8_t>(3u + (mix % 3u));       // actual exponents -4..-2
  const uint8_t mant = static_cast<uint8_t>((mix / 3u) & 0x7u);
  return static_cast<uint8_t>(sign | (exp << 3) | mant);
}

uint32_t DeterministicE4M3Word(int rank, std::size_t word_index) {
  uint32_t word = 0;
  for (int lane = 0; lane < 4; ++lane) {
    word |= static_cast<uint32_t>(DeterministicE4M3Byte(rank, word_index * 4 + lane)) << (lane * 8);
  }
  return word;
}

float DecodeE4M3(uint8_t bits) {
  const int sign = (bits & 0x80u) ? -1 : 1;
  const int exp = (bits >> 3) & 0x0f;
  const int mant = bits & 0x07;
  if ((bits & 0x7fu) == 0) {
    return sign < 0 ? -0.0f : 0.0f;
  }
  if (exp == 0x0f && mant == 0x07) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (exp == 0) {
    return static_cast<float>(sign * std::ldexp(static_cast<double>(mant), -9));
  }
  const double significand = 1.0 + static_cast<double>(mant) / 8.0;
  return static_cast<float>(sign * std::ldexp(significand, exp - 7));
}

uint8_t EncodeE4M3(float value) {
  if (std::isnan(value)) {
    return 0x7fu;
  }
  const bool negative = std::signbit(value);
  double abs_value = std::fabs(static_cast<double>(value));
  const uint8_t sign = negative ? 0x80u : 0x00u;
  if (abs_value == 0.0) {
    return sign;
  }

  const double max_finite = DecodeE4M3(0x7eu);
  if (abs_value >= max_finite) {
    return static_cast<uint8_t>(sign | 0x7eu);
  }

  constexpr double min_normal = 0x1p-6;
  if (abs_value < min_normal) {
    int mant = RoundTiesToEven(abs_value / 0x1p-9);
    if (mant <= 0) {
      return sign;
    }
    if (mant >= 8) {
      return static_cast<uint8_t>(sign | 0x08u);
    }
    return static_cast<uint8_t>(sign | mant);
  }

  int exp = static_cast<int>(std::floor(std::log2(abs_value)));
  double scaled = std::ldexp(abs_value, -exp) - 1.0;
  int mant = RoundTiesToEven(scaled * 8.0);
  if (mant == 8) {
    mant = 0;
    ++exp;
  }
  int biased = exp + 7;
  if (biased >= 15) {
    return static_cast<uint8_t>(sign | 0x7eu);
  }
  if (biased <= 0) {
    int sub = RoundTiesToEven(abs_value / 0x1p-9);
    return static_cast<uint8_t>(sign | std::max(0, std::min(7, sub)));
  }
  return static_cast<uint8_t>(sign | (static_cast<uint8_t>(biased) << 3) |
                              static_cast<uint8_t>(mant));
}

float RoundToFp16(float value) {
  return HalfBitsToFloat(FloatToHalfBits(value));
}

std::vector<uint32_t> CpuAllReduceE4M3Semantic(std::size_t word_count, int ranks) {
  std::vector<uint32_t> out(word_count);
  for (std::size_t word = 0; word < word_count; ++word) {
    uint32_t packed = 0;
    for (int lane = 0; lane < 4; ++lane) {
      packed |= static_cast<uint32_t>(SemanticByteForElement(word * 4 + lane, ranks)) << (lane * 8);
    }
    out[word] = packed;
  }
  return out;
}

std::vector<float> CpuAllReduceE4M3F32(std::size_t element_count, int ranks) {
  std::vector<float> out(element_count);
  for (std::size_t element = 0; element < element_count; ++element) {
    float acc = 0.0f;
    for (int rank = 0; rank < ranks; ++rank) {
      acc += DecodeE4M3(DeterministicE4M3Byte(rank, element));
    }
    out[element] = acc;
  }
  return out;
}

PrecisionMetrics CompareE4M3(const uint32_t* result_words, std::size_t word_count, int ranks) {
  PrecisionMetrics metrics;
  const std::vector<uint32_t> semantic = CpuAllReduceE4M3Semantic(word_count, ranks);
  const std::vector<float> f32 = CpuAllReduceE4M3F32(word_count * 4, ranks);
  double abs_sum = 0.0;

  for (std::size_t word = 0; word < word_count; ++word) {
    for (int lane = 0; lane < 4; ++lane) {
      const uint8_t got_bits = static_cast<uint8_t>((result_words[word] >> (lane * 8)) & 0xffu);
      const uint8_t expected_bits = static_cast<uint8_t>((semantic[word] >> (lane * 8)) & 0xffu);
      if (got_bits != expected_bits) {
        ++metrics.exact_mismatches;
      }
      const std::size_t element = word * 4 + lane;
      const float got = DecodeE4M3(got_bits);
      const float err = std::fabs(got - f32[element]);
      metrics.max_abs_error = std::max(metrics.max_abs_error, err);
      abs_sum += err;
      const uint8_t requantized_bits = EncodeE4M3(f32[element]);
      if (got_bits != requantized_bits) {
        ++metrics.requantized_mismatches;
      }
      metrics.max_requantized_ulp_distance =
          std::max(metrics.max_requantized_ulp_distance,
                   std::abs(OrderedFiniteE4M3Code(got_bits) -
                            OrderedFiniteE4M3Code(requantized_bits)));
    }
  }
  metrics.mean_abs_error = word_count == 0 ? 0.0 : abs_sum / static_cast<double>(word_count * 4);
  return metrics;
}

}  // namespace mnnvl
