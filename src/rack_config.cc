#include "rack_config.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>

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
    for (const auto& dev : body["device"]) {
      if (!dev.IsScalar()) {
        throw std::runtime_error("rack entry " + entry.host_label + " device entries must be scalar integers");
      }
      entry.devices.push_back(dev.as<int>());
    }
    if (entry.devices.size() != 2) {
      throw std::runtime_error("rack entry " + entry.host_label + " must list exactly two CUDA device ordinals");
    }
    if (entry.devices[0] == entry.devices[1]) {
      throw std::runtime_error("rack entry " + entry.host_label + " lists duplicate CUDA device ordinals");
    }
    cfg.entries_.push_back(entry);
  }

  if (cfg.entries_.size() != 2) {
    throw std::runtime_error("this v1 sample requires exactly two rack entries");
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

std::string RackConfig::MpiHostList() const {
  std::ostringstream os;
  for (std::size_t i = 0; i < entries_.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << entries_[i].host_label << ":2";
  }
  return os.str();
}

}  // namespace mnnvl
