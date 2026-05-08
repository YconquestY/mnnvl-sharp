#pragma once

#include <cstddef>
#include <optional>
#include <string>

namespace mnnvl {

enum class SharpBackend {
  kLegacy,
  kTmaAsync,
};

enum class DataType {
  kAuto,
  kE4M3,
  kF16,
};

struct Config {
  std::string rack_config_path;
  std::optional<int> rank_count;
  std::string rank_selection = "balanced";
  SharpBackend sharp_backend = SharpBackend::kLegacy;
  std::size_t bytes = 1073741824ull;
  int warmup = 3;
  int iters = 20;
  DataType types = DataType::kAuto;
  std::size_t reference_check_bytes = 16777216ull;
  bool init_only = false;
  bool json = false;
};

Config ParseConfig(int argc, char** argv);
std::string Usage(const char* argv0);
const char* ToString(SharpBackend backend);
const char* ToString(DataType type);
DataType ResolveDataType(const Config& cfg);

}  // namespace mnnvl
