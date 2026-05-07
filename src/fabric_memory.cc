#include "fabric_memory.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace mnnvl {
namespace {

static_assert(CU_IPC_HANDLE_SIZE == 64, "unexpected CUDA fabric handle size");

CUmemAllocationProp DeviceFabricAllocationProp(const RankInfo& rank) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = rank.gpu_ordinal;
  return prop;
}

CUmulticastObjectProp MulticastProp(std::size_t size, int num_devices) {
  CUmulticastObjectProp prop{};
  prop.numDevices = static_cast<unsigned int>(num_devices);
  prop.size = size;
  prop.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  prop.flags = 0;
  return prop;
}

CUmemAccessDesc DeviceAccessDesc(const RankInfo& rank) {
  CUmemAccessDesc desc{};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = rank.gpu_ordinal;
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  return desc;
}

void ExportFabric(CUmemGenericAllocationHandle handle, CUmemFabricHandle* fabric) {
  std::memset(fabric, 0, sizeof(*fabric));
  MNNVL_CHECK_CU(cuMemExportToShareableHandle(fabric, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
}

CUmemGenericAllocationHandle ImportFabric(const CUmemFabricHandle& fabric) {
  CUmemGenericAllocationHandle handle = 0;
  CUmemFabricHandle copy = fabric;
  MNNVL_CHECK_CU(cuMemImportFromShareableHandle(&handle, &copy, CU_MEM_HANDLE_TYPE_FABRIC));
  return handle;
}

void MapAllocation(CUdeviceptr ptr,
                   std::size_t size,
                   CUmemGenericAllocationHandle handle,
                   const CUmemAccessDesc& access) {
  MNNVL_CHECK_CU(cuMemMap(ptr, size, 0, handle, 0));
  MNNVL_CHECK_CU(cuMemSetAccess(ptr, size, &access, 1));
}

}  // namespace

std::size_t AlignUp(std::size_t value, std::size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  if (value > std::numeric_limits<std::size_t>::max() - alignment + 1) {
    throw std::overflow_error("size overflow while aligning allocation");
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

FabricAllocation CreateFabricAllocation(MPI_Comm world, const RankInfo& rank, std::size_t payload_bytes) {
  FabricAllocation allocation;
  allocation.payload_bytes = payload_bytes;
  int world_size = 0;
  MPI_Comm_size(world, &world_size);
  if (world_size <= 0) {
    throw std::runtime_error("fabric allocation requires a positive MPI world size");
  }

  const CUmemAllocationProp alloc_prop = DeviceFabricAllocationProp(rank);
  MNNVL_CHECK_CU(cuMemGetAllocationGranularity(&allocation.vmm_granularity, &alloc_prop,
                                               CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  CUmulticastObjectProp mc_probe = MulticastProp(payload_bytes, world_size);
  MNNVL_CHECK_CU(cuMulticastGetGranularity(&allocation.multicast_granularity, &mc_probe,
                                           CU_MULTICAST_GRANULARITY_MINIMUM));

  const std::size_t granularity = std::max(allocation.vmm_granularity, allocation.multicast_granularity);
  allocation.alloc_bytes = AlignUp(payload_bytes, granularity);

  MNNVL_CHECK_CU(cuMemCreate(&allocation.local_handle, allocation.alloc_bytes, &alloc_prop, 0));

  MPI_Barrier(world);
  const double peer_init_t0 = MPI_Wtime();

  CUmemFabricHandle local_fabric{};
  ExportFabric(allocation.local_handle, &local_fabric);

  std::vector<CUmemFabricHandle> fabric_handles(static_cast<std::size_t>(world_size));
  MPI_Allgather(local_fabric.data, CU_IPC_HANDLE_SIZE, MPI_BYTE,
                fabric_handles.data(), CU_IPC_HANDLE_SIZE, MPI_BYTE, world);

  allocation.imported_handles.resize(static_cast<std::size_t>(world_size), 0);
  for (int r = 0; r < world_size; ++r) {
    if (r == rank.world_rank) {
      allocation.imported_handles[r] = allocation.local_handle;
    } else {
      allocation.imported_handles[r] = ImportFabric(fabric_handles[r]);
    }
  }

  const CUmemAccessDesc access = DeviceAccessDesc(rank);
  const std::size_t uc_size = allocation.alloc_bytes * static_cast<std::size_t>(world_size);
  MNNVL_CHECK_CU(cuMemAddressReserve(&allocation.uc_base, uc_size, granularity, 0, 0));
  allocation.uc_slots.resize(static_cast<std::size_t>(world_size), 0);
  for (int r = 0; r < world_size; ++r) {
    const CUdeviceptr slot = allocation.uc_base + static_cast<CUdeviceptr>(r * allocation.alloc_bytes);
    MapAllocation(slot, allocation.alloc_bytes, allocation.imported_handles[r], access);
    allocation.uc_slots[static_cast<std::size_t>(r)] = slot;
  }
  allocation.peer_init_local_ms = (MPI_Wtime() - peer_init_t0) * 1000.0;
  allocation.peer_init_all_ms.resize(static_cast<std::size_t>(world_size), 0.0);
  MPI_Allgather(&allocation.peer_init_local_ms, 1, MPI_DOUBLE,
                allocation.peer_init_all_ms.data(), 1, MPI_DOUBLE, world);

  CUmulticastObjectProp mc_prop = MulticastProp(allocation.alloc_bytes, world_size);
  CUmemFabricHandle mc_fabric{};
  if (rank.world_rank == 0) {
    MNNVL_CHECK_CU(cuMulticastCreate(&allocation.multicast_handle, &mc_prop));
    allocation.owns_multicast_handle = true;
    ExportFabric(allocation.multicast_handle, &mc_fabric);
  }

  MPI_Bcast(mc_fabric.data, CU_IPC_HANDLE_SIZE, MPI_BYTE, 0, world);
  if (rank.world_rank != 0) {
    allocation.multicast_handle = ImportFabric(mc_fabric);
    allocation.owns_multicast_handle = false;
  }

  MNNVL_CHECK_CU(cuMulticastAddDevice(allocation.multicast_handle, rank.cu_device));
  MPI_Barrier(world);

  MNNVL_CHECK_CU(cuMulticastBindMem_v2(allocation.multicast_handle, rank.cu_device, 0,
                                       allocation.local_handle, 0, allocation.alloc_bytes, 0));
  allocation.multicast_bound = true;
  MPI_Barrier(world);

  MNNVL_CHECK_CU(cuMemAddressReserve(&allocation.mc_base, allocation.alloc_bytes, granularity, 0, 0));
  MapAllocation(allocation.mc_base, allocation.alloc_bytes, allocation.multicast_handle, access);
  MPI_Barrier(world);
  return allocation;
}

void DestroyFabricAllocation(const RankInfo& rank, FabricAllocation* allocation) {
  if (allocation == nullptr) {
    return;
  }

  if (allocation->mc_base != 0) {
    cuMemUnmap(allocation->mc_base, allocation->alloc_bytes);
    cuMemAddressFree(allocation->mc_base, allocation->alloc_bytes);
    allocation->mc_base = 0;
  }
  if (allocation->multicast_bound) {
    cuMulticastUnbind(allocation->multicast_handle, rank.cu_device, 0, allocation->alloc_bytes);
    allocation->multicast_bound = false;
  }
  if (allocation->uc_base != 0) {
    for (std::size_t r = 0; r < allocation->imported_handles.size(); ++r) {
      if (allocation->uc_slots[r] != 0) {
        cuMemUnmap(allocation->uc_slots[r], allocation->alloc_bytes);
      }
    }
    const std::size_t uc_size = allocation->alloc_bytes * allocation->imported_handles.size();
    cuMemAddressFree(allocation->uc_base, uc_size);
    allocation->uc_base = 0;
  }

  for (std::size_t r = 0; r < allocation->imported_handles.size(); ++r) {
    if (static_cast<int>(r) != rank.world_rank && allocation->imported_handles[r] != 0) {
      cuMemRelease(allocation->imported_handles[r]);
      allocation->imported_handles[r] = 0;
    }
  }
  if (allocation->multicast_handle != 0) {
    cuMemRelease(allocation->multicast_handle);
    allocation->multicast_handle = 0;
  }
  if (allocation->local_handle != 0) {
    cuMemRelease(allocation->local_handle);
    allocation->local_handle = 0;
  }
}

}  // namespace mnnvl
