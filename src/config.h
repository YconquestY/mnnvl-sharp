#pragma once

#include <cstddef>
#include <string>

namespace mnnvl {

struct Config {
  std::string rack_config_path;
  std::size_t bytes = 1073741824ull;
  int warmup = 3;
  int iters = 20;
  std::string types = "auto";
  std::size_t reference_check_bytes = 16777216ull;
  bool json = false;
};

Config ParseConfig(int argc, char** argv);
std::string Usage(const char* argv0);

}  // namespace mnnvl
