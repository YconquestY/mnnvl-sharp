#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mnnvl {

struct RackEntry {
  std::string host_label;
  std::string control_hostname;
  int ssh_port = 4399;
  std::vector<int> devices;
};

struct RankTarget {
  int rank_index_in_plan = -1;
  int rack_index = -1;
  std::string host_label;
  std::string control_hostname;
  int ssh_port = 4399;
  int gpu_ordinal = -1;
};

class RankPlan {
 public:
  explicit RankPlan(std::vector<RankTarget> targets = {}) : targets_(std::move(targets)) {}

  const std::vector<RankTarget>& targets() const { return targets_; }
  int size() const { return static_cast<int>(targets_.size()); }
  const RankTarget& target(int rank) const;
  int SlotCountForHost(const std::string& host_label) const;
  std::map<std::string, int> SlotCountsByHost() const;
  std::map<std::string, std::vector<int>> DevicesByHost() const;

 private:
  std::vector<RankTarget> targets_;
};

class RackConfig {
 public:
  static RackConfig Load(const std::string& path);

  const std::vector<RackEntry>& entries() const { return entries_; }
  const RackEntry* FindByHostLabel(const std::string& label) const;
  std::size_t AvailableRankCount() const;

 private:
  std::vector<RackEntry> entries_;
};

std::string NormalizeHostLabel(const std::string& host);
RankPlan BuildRankPlan(const RackConfig& rack, int rank_count, const std::string& selection);

}  // namespace mnnvl
