#include "rack_config.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <yaml-cpp/yaml.h>

namespace mnnvl {
namespace {

template <typename T>
T RequiredScalar(const YAML::Node& node, const std::string& name, const std::string& host_label) {
  if (!node[name] || !node[name].IsScalar()) {
    throw std::runtime_error("rack entry " + host_label + " requires scalar field '" + name + "'");
  }
  return node[name].as<T>();
}

YAML::Node EntryBody(const YAML::Node& item, std::string* host_label) {
  if (!item.IsMap() || item.size() != 1) {
    throw std::runtime_error("each rack item must be a single-key map");
  }
  const auto kv = *item.begin();
  if (!kv.first.IsScalar()) {
    throw std::runtime_error("rack item key must be a scalar host label");
  }
  *host_label = kv.first.as<std::string>();
  YAML::Node body = kv.second;
  if (!body.IsSequence() && !body.IsMap()) {
    throw std::runtime_error("rack entry " + *host_label + " must be a map or sequence of maps");
  }
  if (body.IsMap()) {
    return body;
  }
  YAML::Node merged(YAML::NodeType::Map);
  for (const auto& part : body) {
    if (!part.IsMap()) {
      throw std::runtime_error("rack entry " + *host_label + " sequence items must be maps");
    }
    for (const auto& field : part) {
      merged[field.first.as<std::string>()] = field.second;
    }
  }
  return merged;
}

}  // namespace

std::string NormalizeHostLabel(const std::string& host) {
  const auto pos = host.find('.');
  if (pos == std::string::npos) {
    return host;
  }
  return host.substr(0, pos);
}

const RankTarget& RankPlan::target(int rank) const {
  if (rank < 0 || rank >= static_cast<int>(targets_.size())) {
    throw std::runtime_error("rank index is outside the selected rank plan");
  }
  return targets_[static_cast<std::size_t>(rank)];
}

int RankPlan::SlotCountForHost(const std::string& host_label) const {
  const std::string normalized = NormalizeHostLabel(host_label);
  int count = 0;
  for (const auto& target : targets_) {
    if (NormalizeHostLabel(target.host_label) == normalized) {
      ++count;
    }
  }
  return count;
}

std::map<std::string, int> RankPlan::SlotCountsByHost() const {
  std::map<std::string, int> counts;
  for (const auto& target : targets_) {
    ++counts[NormalizeHostLabel(target.host_label)];
  }
  return counts;
}

std::map<std::string, std::vector<int>> RankPlan::DevicesByHost() const {
  std::map<std::string, std::vector<int>> devices;
  for (const auto& target : targets_) {
    devices[NormalizeHostLabel(target.host_label)].push_back(target.gpu_ordinal);
  }
  return devices;
}

RackConfig RackConfig::Load(const std::string& path) {
  YAML::Node root = YAML::LoadFile(path);
  if (!root["rack"] || !root["rack"].IsSequence()) {
    throw std::runtime_error("rack config must contain a top-level sequence named 'rack'");
  }

  RackConfig cfg;
  std::set<std::string> labels;
  for (const auto& item : root["rack"]) {
    RackEntry entry;
    YAML::Node body = EntryBody(item, &entry.host_label);
    if (entry.host_label.empty()) {
      throw std::runtime_error("rack entry host label cannot be empty");
    }
    if (!labels.insert(NormalizeHostLabel(entry.host_label)).second) {
      throw std::runtime_error("duplicate rack host label: " + entry.host_label);
    }
    entry.control_hostname = RequiredScalar<std::string>(body, "hostname", entry.host_label);
    if (body["port"]) {
      if (!body["port"].IsScalar()) {
        throw std::runtime_error("rack entry " + entry.host_label + " field 'port' must be scalar");
      }
      entry.ssh_port = body["port"].as<int>();
    }
    if (entry.ssh_port <= 0 || entry.ssh_port > 65535) {
      throw std::runtime_error("rack entry " + entry.host_label + " has invalid SSH port");
    }
    if (!body["device"] || !body["device"].IsSequence()) {
      throw std::runtime_error("rack entry " + entry.host_label + " requires 'device' as a YAML sequence");
    }
    std::set<int> devices;
    for (const auto& dev : body["device"]) {
      if (!dev.IsScalar()) {
        throw std::runtime_error("rack entry " + entry.host_label + " device entries must be scalar integers");
      }
      const int ordinal = dev.as<int>();
      if (ordinal < 0) {
        throw std::runtime_error("rack entry " + entry.host_label + " lists a negative CUDA device ordinal");
      }
      if (!devices.insert(ordinal).second) {
        throw std::runtime_error("rack entry " + entry.host_label + " lists duplicate CUDA device ordinals");
      }
      entry.devices.push_back(ordinal);
    }
    if (entry.devices.empty() || entry.devices.size() > 4) {
      throw std::runtime_error("rack entry " + entry.host_label + " must list 1 to 4 CUDA device ordinals");
    }
    cfg.entries_.push_back(entry);
  }

  if (cfg.entries_.empty() || cfg.entries_.size() > 18) {
    throw std::runtime_error("rack config must contain 1 to 18 rack entries");
  }
  if (cfg.AvailableRankCount() > 72) {
    throw std::runtime_error("rack config describes more than 72 CUDA devices");
  }
  return cfg;
}

const RackEntry* RackConfig::FindByHostLabel(const std::string& label) const {
  const std::string normalized = NormalizeHostLabel(label);
  for (const auto& entry : entries_) {
    if (NormalizeHostLabel(entry.host_label) == normalized) {
      return &entry;
    }
  }
  return nullptr;
}

std::size_t RackConfig::AvailableRankCount() const {
  return std::accumulate(entries_.begin(), entries_.end(), std::size_t{0},
                         [](std::size_t sum, const RackEntry& entry) {
                           return sum + entry.devices.size();
                         });
}

RankPlan BuildRankPlan(const RackConfig& rack, int rank_count, const std::string& selection) {
  if (rank_count <= 0) {
    throw std::runtime_error("rank_count must be positive");
  }
  const std::size_t available = rack.AvailableRankCount();
  if (static_cast<std::size_t>(rank_count) > available) {
    throw std::runtime_error("rank_count " + std::to_string(rank_count) +
                             " exceeds rack.yaml device count " + std::to_string(available));
  }
  if (selection != "balanced" && selection != "prefix") {
    throw std::runtime_error("rank selection must be balanced or prefix");
  }

  std::vector<RankTarget> targets;
  targets.reserve(static_cast<std::size_t>(rank_count));
  auto append = [&](std::size_t rack_index, int gpu_ordinal) {
    const RackEntry& entry = rack.entries()[rack_index];
    targets.push_back(RankTarget{
        .rank_index_in_plan = static_cast<int>(targets.size()),
        .rack_index = static_cast<int>(rack_index),
        .host_label = entry.host_label,
        .control_hostname = entry.control_hostname,
        .ssh_port = entry.ssh_port,
        .gpu_ordinal = gpu_ordinal,
    });
  };

  if (selection == "prefix") {
    for (std::size_t rack_index = 0; rack_index < rack.entries().size(); ++rack_index) {
      for (int gpu : rack.entries()[rack_index].devices) {
        append(rack_index, gpu);
        if (static_cast<int>(targets.size()) == rank_count) {
          return RankPlan(std::move(targets));
        }
      }
    }
  } else {
    for (std::size_t device_slot = 0; device_slot < 4; ++device_slot) {
      for (std::size_t rack_index = 0; rack_index < rack.entries().size(); ++rack_index) {
        const auto& devices = rack.entries()[rack_index].devices;
        if (device_slot >= devices.size()) {
          continue;
        }
        append(rack_index, devices[device_slot]);
        if (static_cast<int>(targets.size()) == rank_count) {
          return RankPlan(std::move(targets));
        }
      }
    }
  }

  throw std::runtime_error("failed to build a complete rank plan");
}

}  // namespace mnnvl
