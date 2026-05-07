#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mnnvl {

struct RackEntry {
  std::string host_label;
  std::string control_hostname;
  int ssh_port = 4399;
  std::vector<int> devices;
};

class RackConfig {
 public:
  static RackConfig Load(const std::string& path);

  const std::vector<RackEntry>& entries() const { return entries_; }
  const RackEntry* FindByHostLabel(const std::string& label) const;
  std::string MpiHostList() const;

 private:
  std::vector<RackEntry> entries_;
};

std::string NormalizeHostLabel(const std::string& host);

}  // namespace mnnvl
