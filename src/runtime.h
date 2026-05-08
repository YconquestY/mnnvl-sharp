#pragma once

#include <mpi.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>
#include <vector>

#include "config.h"
#include "rack_config.h"

namespace mnnvl {

struct RankInfo {
  int world_rank = -1;
  int world_size = 0;
  int local_rank = -1;
  int local_size = 0;
  int rack_index = -1;
  int rack_count = 0;
  std::string host_label;
  std::string control_hostname;
  int ssh_port = 0;
  std::string os_hostname;
  int gpu_ordinal = -1;
  int host_numa_id = -1;
  CUdevice cu_device = -1;
};

struct CapabilityInfo {
  int vmm_supported = 0;
  int fabric_handle_supported = 0;
  int multicast_supported = 0;
  bool sharp_e4m3_supported = false;
  bool sharp_tma_async_supported = false;
  bool nvfp4_supported = false;
  bool mxfp4_supported = false;
};

struct RankMappingRecord {
  int world_rank = -1;
  int local_rank = -1;
  int rack_index = -1;
  int gpu_ordinal = -1;
  int host_numa_id = -1;
  char host_label[128] = {};
  char os_hostname[128] = {};
};

class MpiComm {
 public:
  explicit MpiComm(MPI_Comm comm = MPI_COMM_NULL) : comm_(comm) {}
  ~MpiComm();
  MpiComm(const MpiComm&) = delete;
  MpiComm& operator=(const MpiComm&) = delete;
  MpiComm(MpiComm&& other) noexcept;
  MpiComm& operator=(MpiComm&& other) noexcept;
  MPI_Comm get() const { return comm_; }

 private:
  MPI_Comm comm_;
};

std::string CudaErrorString(CUresult result);
void CheckCu(CUresult result, const char* expr);
void CheckCudaRuntime(cudaError_t result, const char* expr);

#define MNNVL_CHECK_CU(expr) ::mnnvl::CheckCu((expr), #expr)
#define MNNVL_CHECK_CUDA(expr) ::mnnvl::CheckCudaRuntime((expr), #expr)

RankInfo DiscoverRankInfo(MPI_Comm world, const RackConfig& rack, const RankPlan& plan);
CapabilityInfo InitializeCudaAndQuery(RankInfo* rank);
std::vector<RankMappingRecord> GatherRankMappings(MPI_Comm world, const RankInfo& rank);
void ValidateRankMappings(const std::vector<RankMappingRecord>& records,
                          const RackConfig& rack,
                          const RankPlan& plan);
void ValidateCapabilities(MPI_Comm world, const CapabilityInfo& caps, SharpBackend backend);
void PinThreadToNuma(int numa_id);
void Barrier(MPI_Comm comm);

std::string JsonEscape(const std::string& value);
std::string HostMappingJson(const std::vector<RankMappingRecord>& records);

}  // namespace mnnvl
