#include "config.h"

#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

namespace mnnvl {
namespace {

std::size_t ParseSize(const std::string& value, const std::string& flag) {
  std::size_t idx = 0;
  unsigned long long parsed = 0;
  try {
    parsed = std::stoull(value, &idx, 0);
  } catch (const std::exception&) {
    throw std::invalid_argument("invalid integer for " + flag + ": " + value);
  }
  if (idx != value.size()) {
    throw std::invalid_argument("invalid integer for " + flag + ": " + value);
  }
  return static_cast<std::size_t>(parsed);
}

int ParseInt(const std::string& value, const std::string& flag) {
  std::size_t idx = 0;
  int parsed = 0;
  try {
    parsed = std::stoi(value, &idx, 0);
  } catch (const std::exception&) {
    throw std::invalid_argument("invalid integer for " + flag + ": " + value);
  }
  if (idx != value.size()) {
    throw std::invalid_argument("invalid integer for " + flag + ": " + value);
  }
  return parsed;
}

std::string NeedValue(int& i, int argc, char** argv, const std::string& flag) {
  if (i + 1 >= argc) {
    throw std::invalid_argument("missing value after " + flag);
  }
  ++i;
  return argv[i];
}

SharpBackend ParseBackend(const std::string& value) {
  if (value == "legacy") {
    return SharpBackend::kLegacy;
  }
  if (value == "tma_async") {
    return SharpBackend::kTmaAsync;
  }
  throw std::invalid_argument("--sharp-backend must be legacy or tma_async");
}

DataType ParseDataType(const std::string& value) {
  if (value == "auto") {
    return DataType::kAuto;
  }
  if (value == "e4m3") {
    return DataType::kE4M3;
  }
  if (value == "f16") {
    return DataType::kF16;
  }
  throw std::invalid_argument("--types must be auto, e4m3, or f16");
}

}  // namespace

const char* ToString(SharpBackend backend) {
  switch (backend) {
    case SharpBackend::kLegacy:
      return "legacy";
    case SharpBackend::kTmaAsync:
      return "tma_async";
  }
  return "unknown";
}

const char* ToString(DataType type) {
  switch (type) {
    case DataType::kAuto:
      return "auto";
    case DataType::kE4M3:
      return "e4m3";
    case DataType::kF16:
      return "f16";
  }
  return "unknown";
}

DataType ResolveDataType(const Config& cfg) {
  if (cfg.types != DataType::kAuto) {
    return cfg.types;
  }
  return cfg.sharp_backend == SharpBackend::kLegacy ? DataType::kE4M3 : DataType::kF16;
}

std::string Usage(const char* argv0) {
  std::ostringstream os;
  os << "usage: " << argv0
     << " --rack-config rack.yaml [--rank-count N]"
     << " [--rank-selection balanced|prefix] [--sharp-backend legacy|tma_async]"
     << " [--bytes 1073741824] [--warmup 3] [--iters 20] [--types auto|e4m3|f16]"
     << " [--reference-check-bytes 16777216] [--init-only] [--json]\n";
  return os.str();
}

Config ParseConfig(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--rack-config") {
      cfg.rack_config_path = NeedValue(i, argc, argv, arg);
    } else if (arg == "--rank-count") {
      cfg.rank_count = ParseInt(NeedValue(i, argc, argv, arg), arg);
    } else if (arg == "--rank-selection") {
      cfg.rank_selection = NeedValue(i, argc, argv, arg);
    } else if (arg == "--sharp-backend") {
      cfg.sharp_backend = ParseBackend(NeedValue(i, argc, argv, arg));
    } else if (arg == "--bytes") {
      cfg.bytes = ParseSize(NeedValue(i, argc, argv, arg), arg);
    } else if (arg == "--warmup") {
      cfg.warmup = ParseInt(NeedValue(i, argc, argv, arg), arg);
    } else if (arg == "--iters") {
      cfg.iters = ParseInt(NeedValue(i, argc, argv, arg), arg);
    } else if (arg == "--types") {
      cfg.types = ParseDataType(NeedValue(i, argc, argv, arg));
    } else if (arg == "--reference-check-bytes") {
      cfg.reference_check_bytes = ParseSize(NeedValue(i, argc, argv, arg), arg);
    } else if (arg == "--init-only") {
      cfg.init_only = true;
    } else if (arg == "--json") {
      cfg.json = true;
    } else if (arg == "--help" || arg == "-h") {
      throw std::invalid_argument(Usage(argv[0]));
    } else {
      throw std::invalid_argument("unknown argument: " + arg + "\n" + Usage(argv[0]));
    }
  }

  if (cfg.rack_config_path.empty()) {
    throw std::invalid_argument("--rack-config is required\n" + Usage(argv[0]));
  }
  if (cfg.bytes == 0 || (cfg.bytes % 4) != 0) {
    throw std::invalid_argument("--bytes must be nonzero and a multiple of 4");
  }
  if (cfg.rank_count && *cfg.rank_count <= 0) {
    throw std::invalid_argument("--rank-count must be >= 1");
  }
  if (cfg.rank_selection != "balanced" && cfg.rank_selection != "prefix") {
    throw std::invalid_argument("--rank-selection must be balanced or prefix");
  }
  if (cfg.reference_check_bytes == 0 || (cfg.reference_check_bytes % 4) != 0) {
    throw std::invalid_argument("--reference-check-bytes must be nonzero and a multiple of 4");
  }
  if (cfg.reference_check_bytes > cfg.bytes) {
    throw std::invalid_argument("--reference-check-bytes cannot exceed --bytes");
  }
  if (cfg.warmup < 0) {
    throw std::invalid_argument("--warmup must be >= 0");
  }
  if (cfg.iters < 0) {
    throw std::invalid_argument("--iters must be >= 0");
  }
  const DataType resolved_type = ResolveDataType(cfg);
  if (cfg.sharp_backend == SharpBackend::kLegacy && resolved_type != DataType::kE4M3) {
    throw std::invalid_argument("--types f16 is not supported with --sharp-backend legacy");
  }
  if (cfg.sharp_backend == SharpBackend::kTmaAsync && resolved_type != DataType::kF16) {
    throw std::invalid_argument("--types e4m3 is not supported with --sharp-backend tma_async");
  }
  if (cfg.sharp_backend == SharpBackend::kTmaAsync && (cfg.bytes % 16) != 0) {
    throw std::invalid_argument("--bytes must be a multiple of 16 for --sharp-backend tma_async");
  }
  return cfg;
}

}  // namespace mnnvl
