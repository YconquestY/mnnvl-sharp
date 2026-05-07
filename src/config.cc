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

}  // namespace

std::string Usage(const char* argv0) {
  std::ostringstream os;
  os << "usage: " << argv0
     << " --rack-config rack.yaml [--bytes 1073741824] [--warmup 3]"
     << " [--iters 20] [--types auto|e4m3]"
     << " [--reference-check-bytes 16777216] [--json]\n";
  return os.str();
}

Config ParseConfig(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--rack-config") {
      cfg.rack_config_path = NeedValue(i, argc, argv, arg);
    } else if (arg == "--bytes") {
      cfg.bytes = ParseSize(NeedValue(i, argc, argv, arg), arg);
    } else if (arg == "--warmup") {
      cfg.warmup = ParseInt(NeedValue(i, argc, argv, arg), arg);
    } else if (arg == "--iters") {
      cfg.iters = ParseInt(NeedValue(i, argc, argv, arg), arg);
    } else if (arg == "--types") {
      cfg.types = NeedValue(i, argc, argv, arg);
    } else if (arg == "--reference-check-bytes") {
      cfg.reference_check_bytes = ParseSize(NeedValue(i, argc, argv, arg), arg);
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
  if (cfg.types != "auto" && cfg.types != "e4m3") {
    throw std::invalid_argument("--types must be auto or e4m3");
  }
  return cfg;
}

}  // namespace mnnvl
