#include "runtime.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

#include <sched.h>

namespace mnnvl {
namespace {

std::string Hostname() {
  std::array<char, 256> buf{};
  if (gethostname(buf.data(), buf.size() - 1) != 0) {
    throw std::runtime_error("gethostname failed: " + std::string(std::strerror(errno)));
  }
  return std::string(buf.data());
}

void CopyString(char* dst, std::size_t n, const std::string& src) {
  if (n == 0) {
    return;
  }
  std::snprintf(dst, n, "%s", src.c_str());
}

std::vector<int> ParseCpuList(const std::string& text) {
  std::vector<int> cpus;
  std::size_t pos = 0;
  while (pos < text.size()) {
    while (pos < text.size() && (text[pos] == ',' || text[pos] == '\n' || text[pos] == ' ')) {
      ++pos;
    }
    if (pos >= text.size()) {
      break;
    }
    std::size_t end = pos;
    while (end < text.size() && text[end] != ',' && text[end] != '\n') {
      ++end;
    }
    const std::string token = text.substr(pos, end - pos);
    const auto dash = token.find('-');
    if (dash == std::string::npos) {
      cpus.push_back(std::stoi(token));
    } else {
      const int first = std::stoi(token.substr(0, dash));
      const int last = std::stoi(token.substr(dash + 1));
      if (last < first) {
        throw std::runtime_error("invalid CPU range in cpulist: " + token);
      }
      for (int cpu = first; cpu <= last; ++cpu) {
        cpus.push_back(cpu);
      }
    }
    pos = end + 1;
  }
  return cpus;
}

int DeviceAttribute(CUdevice dev, CUdevice_attribute attr) {
  int value = 0;
  MNNVL_CHECK_CU(cuDeviceGetAttribute(&value, attr, dev));
  return value;
}

RankMappingRecord ToRecord(const RankInfo& rank) {
  RankMappingRecord record;
  record.world_rank = rank.world_rank;
  record.local_rank = rank.local_rank;
  record.rack_index = rank.rack_index;
  record.gpu_ordinal = rank.gpu_ordinal;
  record.host_numa_id = rank.host_numa_id;
  CopyString(record.host_label, sizeof(record.host_label), rank.host_label);
  CopyString(record.os_hostname, sizeof(record.os_hostname), rank.os_hostname);
  return record;
}

}  // namespace

MpiComm::~MpiComm() {
  if (comm_ != MPI_COMM_NULL && comm_ != MPI_COMM_WORLD && comm_ != MPI_COMM_SELF) {
    MPI_Comm_free(&comm_);
  }
}

MpiComm::MpiComm(MpiComm&& other) noexcept : comm_(other.comm_) {
  other.comm_ = MPI_COMM_NULL;
}

MpiComm& MpiComm::operator=(MpiComm&& other) noexcept {
  if (this != &other) {
    if (comm_ != MPI_COMM_NULL && comm_ != MPI_COMM_WORLD && comm_ != MPI_COMM_SELF) {
      MPI_Comm_free(&comm_);
    }
    comm_ = other.comm_;
    other.comm_ = MPI_COMM_NULL;
  }
  return *this;
}

std::string CudaErrorString(CUresult result) {
  const char* name = nullptr;
  const char* text = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &text);
  std::ostringstream os;
  os << (name ? name : "CUDA_ERROR_UNKNOWN");
  if (text != nullptr) {
    os << ": " << text;
  }
  return os.str();
}

void CheckCu(CUresult result, const char* expr) {
  if (result != CUDA_SUCCESS) {
    throw std::runtime_error(std::string(expr) + " failed: " + CudaErrorString(result));
  }
}

void CheckCudaRuntime(cudaError_t result, const char* expr) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(expr) + " failed: " + cudaGetErrorString(result));
  }
}

RankInfo DiscoverRankInfo(MPI_Comm world, const RackConfig& rack, const RankPlan& plan) {
  RankInfo rank;
  rank.rack_count = static_cast<int>(rack.entries().size());
  MPI_Comm_rank(world, &rank.world_rank);
  MPI_Comm_size(world, &rank.world_size);
  if (rank.world_size != plan.size()) {
    throw std::runtime_error("MPI world_size " + std::to_string(rank.world_size) +
                             " does not match selected rank_count " + std::to_string(plan.size()));
  }

  MPI_Comm local_raw = MPI_COMM_NULL;
  MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, rank.world_rank, MPI_INFO_NULL, &local_raw);
  MpiComm local(local_raw);
  MPI_Comm_rank(local.get(), &rank.local_rank);
  MPI_Comm_size(local.get(), &rank.local_size);

  rank.os_hostname = Hostname();
  const char* env_label = std::getenv("MNNVL_HOST_LABEL");
  const std::string match_label = env_label != nullptr ? std::string(env_label) : rank.os_hostname;
  const RackEntry* entry = rack.FindByHostLabel(match_label);
  if (entry == nullptr) {
    throw std::runtime_error("hostname '" + rank.os_hostname + "' did not match rack.yaml; set MNNVL_HOST_LABEL to the rack key if the container hostname differs");
  }

  const RankTarget& target = plan.target(rank.world_rank);
  if (NormalizeHostLabel(entry->host_label) != NormalizeHostLabel(target.host_label)) {
    throw std::runtime_error("rank " + std::to_string(rank.world_rank) + " is running on host '" +
                             entry->host_label + "' but the selected rank plan expects '" +
                             target.host_label + "'");
  }

  for (std::size_t i = 0; i < rack.entries().size(); ++i) {
    if (&rack.entries()[i] == entry) {
      rank.rack_index = static_cast<int>(i);
      break;
    }
  }
  rank.host_label = entry->host_label;
  rank.control_hostname = entry->control_hostname;
  rank.ssh_port = entry->ssh_port;

  const int expected_local_size = plan.SlotCountForHost(entry->host_label);
  if (rank.local_size != expected_local_size) {
    throw std::runtime_error("host " + entry->host_label + " has MPI local_size " +
                             std::to_string(rank.local_size) + " but selected rank plan expects " +
                             std::to_string(expected_local_size));
  }
  rank.gpu_ordinal = target.gpu_ordinal;
  return rank;
}

CapabilityInfo InitializeCudaAndQuery(RankInfo* rank) {
  if (rank == nullptr) {
    throw std::runtime_error("InitializeCudaAndQuery received null RankInfo");
  }
  MNNVL_CHECK_CU(cuInit(0));
  MNNVL_CHECK_CUDA(cudaSetDevice(rank->gpu_ordinal));
  MNNVL_CHECK_CU(cuDeviceGet(&rank->cu_device, rank->gpu_ordinal));
  CUcontext ctx = nullptr;
  MNNVL_CHECK_CU(cuCtxGetCurrent(&ctx));
  if (ctx == nullptr) {
    throw std::runtime_error("cudaSetDevice did not create or retain a current CUDA context");
  }

  CapabilityInfo caps;
  caps.vmm_supported = DeviceAttribute(rank->cu_device, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED);
  caps.fabric_handle_supported = DeviceAttribute(rank->cu_device, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED);
  caps.multicast_supported = DeviceAttribute(rank->cu_device, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED);
  rank->host_numa_id = DeviceAttribute(rank->cu_device, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID);
#if MNNVL_SHARP_HAS_E4M3
  caps.sharp_e4m3_supported = caps.multicast_supported != 0;
#else
  caps.sharp_e4m3_supported = false;
#endif
  caps.nvfp4_supported = false;
  caps.mxfp4_supported = false;
  return caps;
}

std::vector<RankMappingRecord> GatherRankMappings(MPI_Comm world, const RankInfo& rank) {
  RankMappingRecord local = ToRecord(rank);
  int world_size = 0;
  MPI_Comm_size(world, &world_size);
  std::vector<RankMappingRecord> records(static_cast<std::size_t>(world_size));
  MPI_Allgather(&local, sizeof(local), MPI_BYTE, records.data(), sizeof(local), MPI_BYTE, world);
  std::sort(records.begin(), records.end(), [](const auto& a, const auto& b) {
    return a.world_rank < b.world_rank;
  });
  return records;
}

void ValidateRankMappings(const std::vector<RankMappingRecord>& records,
                          const RackConfig& rack,
                          const RankPlan& plan) {
  if (records.size() != static_cast<std::size_t>(plan.size())) {
    throw std::runtime_error("rank mapping record count does not match selected rank plan");
  }
  std::set<int> seen_ranks;
  std::map<std::string, std::vector<RankMappingRecord>> by_host;
  for (const auto& record : records) {
    if (!seen_ranks.insert(record.world_rank).second) {
      throw std::runtime_error("duplicate world rank in mapping records");
    }
    by_host[NormalizeHostLabel(record.host_label)].push_back(record);
    const RankTarget& target = plan.target(record.world_rank);
    if (NormalizeHostLabel(record.host_label) != NormalizeHostLabel(target.host_label)) {
      throw std::runtime_error("rank " + std::to_string(record.world_rank) +
                               " host does not match selected rank plan");
    }
    if (record.rack_index != target.rack_index) {
      throw std::runtime_error("rank " + std::to_string(record.world_rank) +
                               " rack index does not match selected rank plan");
    }
    if (record.gpu_ordinal != target.gpu_ordinal) {
      throw std::runtime_error("rank " + std::to_string(record.world_rank) +
                               " GPU ordinal does not match selected rank plan");
    }
    if (record.host_numa_id < 0) {
      throw std::runtime_error("rank " + std::to_string(record.world_rank) +
                               " did not resolve a valid HOST_NUMA_ID");
    }
  }
  for (const auto& entry : rack.entries()) {
    const std::string host = NormalizeHostLabel(entry.host_label);
    const int expected_slots = plan.SlotCountForHost(entry.host_label);
    auto it = by_host.find(host);
    if (expected_slots == 0) {
      if (it != by_host.end()) {
        throw std::runtime_error("rack entry " + entry.host_label +
                                 " has MPI ranks but is not in the selected plan");
      }
      continue;
    }
    if (it == by_host.end() || it->second.size() != static_cast<std::size_t>(expected_slots)) {
      throw std::runtime_error("rack entry " + entry.host_label + " has " +
                               std::to_string(it == by_host.end() ? 0 : it->second.size()) +
                               " MPI ranks but selected rank plan expects " +
                               std::to_string(expected_slots));
    }
    std::set<int> gpus;
    for (const auto& record : it->second) {
      gpus.insert(record.gpu_ordinal);
    }
    std::set<int> expected_gpus;
    for (const auto& target : plan.targets()) {
      if (NormalizeHostLabel(target.host_label) == host) {
        expected_gpus.insert(target.gpu_ordinal);
      }
    }
    if (gpus != expected_gpus) {
      throw std::runtime_error("rack entry " + entry.host_label +
                               " MPI ranks did not map to selected GPUs");
    }
  }
}

void ValidateCapabilities(MPI_Comm world, const CapabilityInfo& caps) {
  const int local_ok = caps.vmm_supported && caps.fabric_handle_supported && caps.multicast_supported &&
                       caps.sharp_e4m3_supported;
  int all_ok = 0;
  MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_LAND, world);
  if (!all_ok) {
    throw std::runtime_error("one or more selected GPUs lack VMM, fabric handle, multicast, or compiled E4M3 SHARP support");
  }
}

void PinThreadToNuma(int numa_id) {
  if (numa_id < 0) {
    throw std::runtime_error("cannot pin CPU affinity because HOST_NUMA_ID is unavailable");
  }
  const std::string path = "/sys/devices/system/node/node" + std::to_string(numa_id) + "/cpulist";
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("could not read " + path);
  }
  std::stringstream buffer;
  buffer << input.rdbuf();
  const std::vector<int> cpus = ParseCpuList(buffer.str());
  if (cpus.empty()) {
    throw std::runtime_error("NUMA cpulist is empty for node " + std::to_string(numa_id));
  }

  cpu_set_t set;
  CPU_ZERO(&set);
  for (int cpu : cpus) {
    CPU_SET(cpu, &set);
  }
  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    throw std::runtime_error("sched_setaffinity failed for NUMA node " + std::to_string(numa_id) + ": " +
                             std::strerror(errno));
  }
}

void Barrier(MPI_Comm comm) {
  MPI_Barrier(comm);
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream os;
  for (char c : value) {
    switch (c) {
      case '\\':
        os << "\\\\";
        break;
      case '"':
        os << "\\\"";
        break;
      case '\n':
        os << "\\n";
        break;
      case '\r':
        os << "\\r";
        break;
      case '\t':
        os << "\\t";
        break;
      default:
        os << c;
        break;
    }
  }
  return os.str();
}

std::string HostMappingJson(const std::vector<RankMappingRecord>& records) {
  std::ostringstream os;
  os << "[";
  for (std::size_t i = 0; i < records.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    const auto& r = records[i];
    os << "{\"rank\":" << r.world_rank
       << ",\"local_rank\":" << r.local_rank
       << ",\"rack_index\":" << r.rack_index
       << ",\"host_label\":\"" << JsonEscape(r.host_label) << "\""
       << ",\"os_hostname\":\"" << JsonEscape(r.os_hostname) << "\""
       << ",\"gpu\":" << r.gpu_ordinal
       << ",\"host_numa_id\":" << r.host_numa_id << "}";
  }
  os << "]";
  return os.str();
}

}  // namespace mnnvl
