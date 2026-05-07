#pragma once

#include <mpi.h>

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "runtime.h"

namespace mnnvl {

struct FabricAllocation {
  std::size_t payload_bytes = 0;
  std::size_t alloc_bytes = 0;
  std::size_t vmm_granularity = 0;
  std::size_t multicast_granularity = 0;

  CUmemGenericAllocationHandle local_handle = 0;
  std::vector<CUmemGenericAllocationHandle> imported_handles;

  CUdeviceptr uc_base = 0;
  std::vector<CUdeviceptr> uc_slots;

  double peer_init_local_ms = 0.0;
  std::vector<double> peer_init_all_ms;

  CUmemGenericAllocationHandle multicast_handle = 0;
  CUdeviceptr mc_base = 0;
  bool multicast_bound = false;
  bool owns_multicast_handle = false;
};

std::size_t AlignUp(std::size_t value, std::size_t alignment);
FabricAllocation CreateFabricAllocation(MPI_Comm world, const RankInfo& rank, std::size_t payload_bytes);
void DestroyFabricAllocation(const RankInfo& rank, FabricAllocation* allocation);

}  // namespace mnnvl
