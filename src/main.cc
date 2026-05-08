#include <mpi.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.h"
#include "e4m3_ref.h"
#include "fabric_memory.h"
#include "rack_config.h"
#include "runtime.h"
#include "sharp_kernels.h"

namespace mnnvl {
namespace {

struct TimingStats {
  double min = 0.0;
  double median = 0.0;
  double p95 = 0.0;
  double max = 0.0;
};

TimingStats Summarize(std::vector<double> values) {
  TimingStats stats;
  if (values.empty()) {
    return stats;
  }
  std::sort(values.begin(), values.end());
  stats.min = values.front();
  stats.max = values.back();
  stats.median = values[values.size() / 2];
  const std::size_t p95_index = std::min(values.size() - 1,
                                         static_cast<std::size_t>(std::ceil(values.size() * 0.95)) - 1);
  stats.p95 = values[p95_index];
  return stats;
}

CUdeviceptr InputPtr(const FabricAllocation& allocation, int world_rank) {
  return allocation.uc_slots[world_rank] + static_cast<CUdeviceptr>(allocation.input_offset);
}

CUdeviceptr OutputPtr(const FabricAllocation& allocation, int world_rank) {
  return allocation.uc_slots[world_rank] + static_cast<CUdeviceptr>(allocation.output_offset);
}

void InitializeLocalBuffer(const Config& cfg,
                           const RankInfo& rank,
                           const FabricAllocation& allocation,
                           cudaStream_t stream) {
  if (cfg.sharp_backend == SharpBackend::kLegacy) {
    auto* ptr = reinterpret_cast<uint32_t*>(OutputPtr(allocation, rank.world_rank));
    LaunchInitE4M3(ptr, allocation.payload_bytes / sizeof(uint32_t), rank.world_rank, stream);
    LaunchAliasFence(stream);
  } else {
    auto* input = reinterpret_cast<uint16_t*>(InputPtr(allocation, rank.world_rank));
    auto* output = reinterpret_cast<void*>(OutputPtr(allocation, rank.world_rank));
    LaunchInitF16(input, allocation.payload_bytes / sizeof(uint16_t), rank.world_rank, stream);
    MNNVL_CHECK_CUDA(cudaMemsetAsync(output, 0, allocation.payload_bytes, stream));
    LaunchAsyncProxyFence(stream);
    LaunchAliasFence(stream);
  }
  MNNVL_CHECK_CUDA(cudaStreamSynchronize(stream));
}

void RunAllReduceIteration(MPI_Comm world,
                           const Config& cfg,
                           const RankInfo& rank,
                           const FabricAllocation& allocation,
                           cudaStream_t stream,
                           cudaEvent_t start,
                           cudaEvent_t stop,
                           bool measure,
                           double* device_ms,
                           double* wall_ms) {
  InitializeLocalBuffer(cfg, rank, allocation, stream);

  const double t0 = MPI_Wtime();
  MPI_Barrier(world);
  LaunchAliasFence(stream);
  MNNVL_CHECK_CUDA(cudaStreamSynchronize(stream));
  MPI_Barrier(world);

  float local_device_ms = 0.0f;
  if (cfg.sharp_backend == SharpBackend::kLegacy) {
    if (rank.world_rank == 0) {
      auto* mc_words = reinterpret_cast<uint32_t*>(allocation.mc_base);
      if (measure) {
        MNNVL_CHECK_CUDA(cudaEventRecord(start, stream));
      }
      LaunchSharpAllReduceE4M3(mc_words, allocation.payload_bytes / sizeof(uint32_t), stream);
      if (measure) {
        MNNVL_CHECK_CUDA(cudaEventRecord(stop, stream));
        MNNVL_CHECK_CUDA(cudaEventSynchronize(stop));
        MNNVL_CHECK_CUDA(cudaEventElapsedTime(&local_device_ms, start, stop));
      } else {
        MNNVL_CHECK_CUDA(cudaStreamSynchronize(stream));
      }
    }
  } else {
    auto* input = reinterpret_cast<const uint16_t*>(InputPtr(allocation, rank.world_rank));
    auto* mc_output = reinterpret_cast<uint16_t*>(allocation.mc_base);
    if (measure) {
      MNNVL_CHECK_CUDA(cudaEventRecord(start, stream));
    }
    LaunchSharpAllReduceF16TmaAsync(input, mc_output, allocation.payload_bytes / sizeof(uint16_t), stream);
    if (measure) {
      MNNVL_CHECK_CUDA(cudaEventRecord(stop, stream));
      MNNVL_CHECK_CUDA(cudaEventSynchronize(stop));
      MNNVL_CHECK_CUDA(cudaEventElapsedTime(&local_device_ms, start, stop));
    } else {
      MNNVL_CHECK_CUDA(cudaStreamSynchronize(stream));
    }
  }

  double max_device_ms = 0.0;
  if (cfg.sharp_backend == SharpBackend::kTmaAsync && measure) {
    const double local_device_double = static_cast<double>(local_device_ms);
    MPI_Reduce(&local_device_double, &max_device_ms, 1, MPI_DOUBLE, MPI_MAX, 0, world);
  }

  MPI_Barrier(world);
  LaunchAliasFence(stream);
  MNNVL_CHECK_CUDA(cudaStreamSynchronize(stream));
  const double local_wall_ms = (MPI_Wtime() - t0) * 1000.0;
  double max_wall_ms = 0.0;
  MPI_Reduce(&local_wall_ms, &max_wall_ms, 1, MPI_DOUBLE, MPI_MAX, 0, world);

  if (rank.world_rank == 0) {
    if (device_ms != nullptr) {
      *device_ms = cfg.sharp_backend == SharpBackend::kTmaAsync ? max_device_ms
                                                                : static_cast<double>(local_device_ms);
    }
    if (wall_ms != nullptr) {
      *wall_ms = max_wall_ms;
    }
  }
}

void RunCanarySmokeTest(MPI_Comm world,
                        const RankInfo& rank,
                        const FabricAllocation& allocation,
                        cudaStream_t stream) {
  auto* self_slot = reinterpret_cast<uint32_t*>(InputPtr(allocation, rank.world_rank));
  LaunchCanary(self_slot, static_cast<uint32_t>(rank.world_rank), stream);
  LaunchAliasFence(stream);
  MNNVL_CHECK_CUDA(cudaStreamSynchronize(stream));
  MPI_Barrier(world);

  std::array<uint32_t, kCanaryWords> host{};
  for (int r = 0; r < rank.world_size; ++r) {
    MNNVL_CHECK_CU(cuMemcpyDtoH(host.data(), InputPtr(allocation, r),
                                sizeof(uint32_t) * host.size()));
    for (int i = 0; i < kCanaryWords; ++i) {
      const uint32_t expected = 0xC0010000u | ((static_cast<uint32_t>(r) & 0xffu) << 8) |
                                static_cast<uint32_t>(i);
      if (host[static_cast<std::size_t>(i)] != expected) {
        std::ostringstream os;
        os << "canary mismatch on rank " << rank.world_rank << " while reading slot " << r
           << " word " << i << ": got 0x" << std::hex << host[static_cast<std::size_t>(i)]
           << " expected 0x" << expected;
        throw std::runtime_error(os.str());
      }
    }
  }
  MPI_Barrier(world);
}

std::vector<uint64_t> GatherHashes(MPI_Comm world,
                                   const RankInfo& rank,
                                   const FabricAllocation& allocation,
                                   cudaStream_t stream) {
  uint64_t* d_hash = nullptr;
  MNNVL_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_hash), sizeof(uint64_t)));
  uint64_t local_hash = 0;
  LaunchHash(reinterpret_cast<const uint32_t*>(OutputPtr(allocation, rank.world_rank)),
             allocation.payload_bytes / sizeof(uint32_t), d_hash, stream);
  MNNVL_CHECK_CUDA(cudaMemcpyAsync(&local_hash, d_hash, sizeof(local_hash), cudaMemcpyDeviceToHost, stream));
  MNNVL_CHECK_CUDA(cudaStreamSynchronize(stream));
  MNNVL_CHECK_CUDA(cudaFree(d_hash));

  std::vector<uint64_t> hashes(static_cast<std::size_t>(rank.world_size));
  MPI_Allgather(&local_hash, 1, MPI_UINT64_T, hashes.data(), 1, MPI_UINT64_T, world);
  return hashes;
}

bool HashesMatch(const std::vector<uint64_t>& hashes) {
  if (hashes.empty()) {
    return true;
  }
  return std::all_of(hashes.begin(), hashes.end(), [&](uint64_t value) { return value == hashes[0]; });
}

void CopyReferenceRegion(const RankInfo& rank,
                         const FabricAllocation& allocation,
                         std::size_t bytes,
                         std::vector<uint32_t>* out) {
  if (rank.world_rank != 0) {
    return;
  }
  out->resize(bytes / sizeof(uint32_t));
  MNNVL_CHECK_CU(cuMemcpyDtoH(out->data(), OutputPtr(allocation, rank.world_rank), bytes));
}

void PrintHumanReport(const Config& cfg,
                      const RankInfo& rank,
                      const CapabilityInfo& caps,
                      const std::vector<RankMappingRecord>& records,
                      const FabricAllocation& allocation,
                      const std::vector<uint64_t>* hashes,
                      const PrecisionMetrics* metrics,
                      const TimingStats& peer_init,
                      const TimingStats& device,
                      const TimingStats& wall) {
  std::cout << "mnnvl_sharp_allreduce\n";
  std::cout << "  selected_rank_count: " << records.size() << "\n";
  std::cout << "  rank_selection: " << cfg.rank_selection << "\n";
  std::cout << "  sharp_backend: " << ToString(cfg.sharp_backend) << "\n";
  std::cout << "  datatype: " << ToString(ResolveDataType(cfg)) << "\n";
  std::cout << "  init_only: " << (cfg.init_only ? "true" : "false") << "\n";
  std::cout << "  payload_bytes: " << cfg.bytes << "\n";
  std::cout << "  allocation_bytes: " << allocation.alloc_bytes << "\n";
  std::cout << "  input_offset: " << allocation.input_offset << "\n";
  std::cout << "  output_offset: " << allocation.output_offset << "\n";
  std::cout << "  multicast_bytes: " << allocation.multicast_bytes << "\n";
  std::cout << "  vmm_granularity: " << allocation.vmm_granularity << "\n";
  std::cout << "  multicast_granularity: " << allocation.multicast_granularity << "\n";
  std::cout << "  datatype_support: e4m3_legacy="
            << (caps.sharp_e4m3_supported ? "supported" : "unsupported")
            << " f16_tma_async="
            << (caps.sharp_tma_async_supported ? "supported" : "unsupported")
            << " nvfp4=unsupported mxfp4=unsupported\n";
  std::cout << "  gpu_mapping:\n";
  for (const auto& r : records) {
    std::cout << "    rank " << r.world_rank << " host=" << r.host_label
              << " local_rank=" << r.local_rank << " gpu=" << r.gpu_ordinal
              << " host_numa_id=" << r.host_numa_id << "\n";
  }
  std::cout << "  capabilities: vmm=" << caps.vmm_supported
            << " fabric_handle=" << caps.fabric_handle_supported
            << " multicast=" << caps.multicast_supported << "\n";
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "  peer_init_ms: min=" << peer_init.min << " median=" << peer_init.median
            << " p95=" << peer_init.p95 << " max=" << peer_init.max << "\n";
  std::cout << "  smoke_test: passed\n";
  if (cfg.init_only) {
    std::cout << "  sharp_allreduce: skipped\n";
    return;
  }
  std::cout << "  replica_hashes:";
  for (uint64_t h : *hashes) {
    std::cout << " 0x" << std::hex << h << std::dec;
  }
  std::cout << "\n";
  std::cout << "  hashes_match: " << (HashesMatch(*hashes) ? "true" : "false") << "\n";
  std::cout << "  semantic_exact_mismatches: " << metrics->exact_mismatches << "\n";
  std::cout << "  f32_requantized_mismatches: " << metrics->requantized_mismatches << "\n";
  std::cout << "  f32_max_requantized_ulp_distance: "
            << metrics->max_requantized_ulp_distance << "\n";
  std::cout << "  f32_max_abs_error: " << metrics->max_abs_error << "\n";
  std::cout << "  f32_mean_abs_error: " << metrics->mean_abs_error << "\n";
  std::cout << "  sharp_kernel_ms: min=" << device.min << " median=" << device.median
            << " p95=" << device.p95 << " max=" << device.max << "\n";
  std::cout << "  collective_wall_ms: min=" << wall.min << " median=" << wall.median
            << " p95=" << wall.p95 << " max=" << wall.max << "\n";
  if (rank.world_rank == 0 && device.median > 0.0) {
    const double logical_gib = (static_cast<double>(records.size()) * static_cast<double>(cfg.bytes)) /
                               (1024.0 * 1024.0 * 1024.0);
    const double bandwidth = logical_gib / (device.median / 1000.0);
    std::cout << "  effective_logical_bandwidth_gib_s_median: " << bandwidth << "\n";
  }
}

void PrintJsonReport(const Config& cfg,
                     const CapabilityInfo& caps,
                     const std::vector<RankMappingRecord>& records,
                     const FabricAllocation& allocation,
                     const std::vector<uint64_t>* hashes,
                     const PrecisionMetrics* metrics,
                     const TimingStats& peer_init,
                     const TimingStats& device,
                     const TimingStats& wall) {
  std::ostringstream os;
  os << "{";
  os << "\"selected_rank_count\":" << records.size();
  os << ",\"rank_selection\":\"" << JsonEscape(cfg.rank_selection) << "\"";
  os << ",\"sharp_backend\":\"" << ToString(cfg.sharp_backend) << "\"";
  os << ",\"datatype\":\"" << ToString(ResolveDataType(cfg)) << "\"";
  os << ",\"init_only\":" << (cfg.init_only ? "true" : "false");
  os << ",\"payload_bytes\":" << cfg.bytes;
  os << ",\"allocation_bytes\":" << allocation.alloc_bytes;
  os << ",\"input_offset\":" << allocation.input_offset;
  os << ",\"output_offset\":" << allocation.output_offset;
  os << ",\"multicast_bytes\":" << allocation.multicast_bytes;
  os << ",\"datatype_support\":{\"e4m3_legacy\":\""
     << (caps.sharp_e4m3_supported ? "supported" : "unsupported")
     << "\",\"f16_tma_async\":\""
     << (caps.sharp_tma_async_supported ? "supported" : "unsupported")
     << "\",\"nvfp4\":\"unsupported\",\"mxfp4\":\"unsupported\"}";
  os << ",\"host_mapping\":" << HostMappingJson(records);
  os << ",\"peer_init_ms\":" << peer_init.max;
  os << ",\"peer_init_min_ms\":" << peer_init.min;
  os << ",\"peer_init_median_ms\":" << peer_init.median;
  os << ",\"peer_init_p95_ms\":" << peer_init.p95;
  os << ",\"peer_init_max_ms\":" << peer_init.max;
  os << ",\"peer_init_local_ms_by_rank\":[";
  for (std::size_t i = 0; i < allocation.peer_init_all_ms.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << allocation.peer_init_all_ms[i];
  }
  os << "]";
  if (cfg.init_only) {
    os << ",\"hashes_match\":null,\"hashes\":[]";
    os << ",\"semantic_exact_mismatches\":null";
    os << ",\"f32_requantized_mismatches\":null";
    os << ",\"f32_max_requantized_ulp_distance\":null";
    os << ",\"f32_max_abs_error\":null";
    os << ",\"f32_mean_abs_error\":null";
  } else {
    os << ",\"hashes_match\":" << (HashesMatch(*hashes) ? "true" : "false");
    os << ",\"hashes\":[";
    for (std::size_t i = 0; i < hashes->size(); ++i) {
      if (i != 0) {
        os << ",";
      }
      os << "\"" << std::hex << (*hashes)[i] << std::dec << "\"";
    }
    os << "]";
    os << ",\"semantic_exact_mismatches\":" << metrics->exact_mismatches;
    os << ",\"f32_requantized_mismatches\":" << metrics->requantized_mismatches;
    os << ",\"f32_max_requantized_ulp_distance\":" << metrics->max_requantized_ulp_distance;
    os << ",\"f32_max_abs_error\":" << metrics->max_abs_error;
    os << ",\"f32_mean_abs_error\":" << metrics->mean_abs_error;
  }
  os << ",\"sharp_kernel_ms\":{\"min\":" << device.min << ",\"median\":" << device.median
     << ",\"p95\":" << device.p95 << ",\"max\":" << device.max << "}";
  os << ",\"collective_wall_ms\":{\"min\":" << wall.min << ",\"median\":" << wall.median
     << ",\"p95\":" << wall.p95 << ",\"max\":" << wall.max << "}";
  os << "}";
  std::cout << os.str() << "\n";
}

void MaybeDumpMismatches(const std::vector<uint32_t>& result_words, int ranks, const PrecisionMetrics& metrics) {
  if (metrics.exact_mismatches == 0 || std::getenv("MNNVL_DEBUG_MISMATCH") == nullptr) {
    return;
  }
  const std::size_t words = std::min<std::size_t>(result_words.size(), 16);
  const std::vector<uint32_t> expected = CpuAllReduceE4M3Semantic(words, ranks);
  std::cerr << "debug_first_words:";
  for (std::size_t i = 0; i < words; ++i) {
    std::cerr << " [" << i << "] got=0x" << std::hex << result_words[i]
              << " expected=0x" << expected[i] << std::dec;
  }
  std::cerr << "\n";
}

void MaybeDumpF16Mismatches(const std::vector<uint32_t>& result_words,
                            std::size_t element_count,
                            int ranks,
                            const PrecisionMetrics& metrics) {
  if (metrics.exact_mismatches == 0 || std::getenv("MNNVL_DEBUG_MISMATCH") == nullptr) {
    return;
  }
  const auto* result = reinterpret_cast<const uint16_t*>(result_words.data());
  const std::size_t elements = std::min<std::size_t>(element_count, 16);
  const std::vector<uint16_t> expected = CpuAllReduceF16Semantic(elements, ranks);
  std::cerr << "debug_first_f16:";
  for (std::size_t i = 0; i < elements; ++i) {
    std::cerr << " [" << i << "] got=0x" << std::hex << result[i]
              << " expected=0x" << expected[i] << std::dec;
  }
  std::cerr << "\n";
}

int Run(int argc, char** argv) {
  Config cfg = ParseConfig(argc, argv);
  RackConfig rack = RackConfig::Load(cfg.rack_config_path);
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const int selected_rank_count = cfg.rank_count.value_or(world_size);
  RankPlan plan = BuildRankPlan(rack, selected_rank_count, cfg.rank_selection);
  RankInfo rank = DiscoverRankInfo(MPI_COMM_WORLD, rack, plan);
  CapabilityInfo caps = InitializeCudaAndQuery(&rank);
  PinThreadToNuma(rank.host_numa_id);
  const std::vector<RankMappingRecord> records = GatherRankMappings(MPI_COMM_WORLD, rank);
  ValidateRankMappings(records, rack, plan);
  ValidateCapabilities(MPI_COMM_WORLD, caps, cfg.sharp_backend);

  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  MNNVL_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  MNNVL_CHECK_CUDA(cudaEventCreate(&start));
  MNNVL_CHECK_CUDA(cudaEventCreate(&stop));

  FabricAllocation allocation = CreateFabricAllocation(MPI_COMM_WORLD, rank, cfg.bytes, cfg.sharp_backend);
  RunCanarySmokeTest(MPI_COMM_WORLD, rank, allocation, stream);
  const TimingStats peer_init = Summarize(allocation.peer_init_all_ms);

  if (cfg.init_only) {
    if (rank.world_rank == 0) {
      const TimingStats empty{};
      if (cfg.json) {
        PrintJsonReport(cfg, caps, records, allocation, nullptr, nullptr, peer_init, empty, empty);
      } else {
        PrintHumanReport(cfg, rank, caps, records, allocation, nullptr, nullptr, peer_init, empty, empty);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    DestroyFabricAllocation(rank, &allocation);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    return 0;
  }

  for (int i = 0; i < cfg.warmup; ++i) {
    RunAllReduceIteration(MPI_COMM_WORLD, cfg, rank, allocation, stream, start, stop, false, nullptr, nullptr);
  }

  std::vector<double> device_times;
  std::vector<double> wall_times;
  for (int i = 0; i < cfg.iters; ++i) {
    double device_ms = 0.0;
    double wall_ms = 0.0;
    RunAllReduceIteration(MPI_COMM_WORLD, cfg, rank, allocation, stream, start, stop, true, &device_ms, &wall_ms);
    if (rank.world_rank == 0) {
      device_times.push_back(device_ms);
      wall_times.push_back(wall_ms);
    }
  }
  if (cfg.iters == 0) {
    RunAllReduceIteration(MPI_COMM_WORLD, cfg, rank, allocation, stream, start, stop, false, nullptr, nullptr);
  }

  const std::vector<uint64_t> hashes = GatherHashes(MPI_COMM_WORLD, rank, allocation, stream);
  std::vector<uint32_t> checked_words;
  CopyReferenceRegion(rank, allocation, cfg.reference_check_bytes, &checked_words);

  int final_status = 0;
  if (rank.world_rank == 0) {
    PrecisionMetrics metrics;
    if (ResolveDataType(cfg) == DataType::kE4M3) {
      metrics = CompareE4M3(checked_words.data(), checked_words.size(), rank.world_size);
      MaybeDumpMismatches(checked_words, rank.world_size, metrics);
    } else {
      metrics = CompareF16(reinterpret_cast<const uint16_t*>(checked_words.data()),
                           cfg.reference_check_bytes / sizeof(uint16_t),
                           rank.world_size);
      MaybeDumpF16Mismatches(checked_words, cfg.reference_check_bytes / sizeof(uint16_t),
                             rank.world_size, metrics);
    }
    const TimingStats device = Summarize(device_times);
    const TimingStats wall = Summarize(wall_times);
    if (cfg.json) {
      PrintJsonReport(cfg, caps, records, allocation, &hashes, &metrics, peer_init, device, wall);
    } else {
      PrintHumanReport(cfg, rank, caps, records, allocation, &hashes, &metrics, peer_init, device, wall);
    }
    if (!HashesMatch(hashes) || metrics.exact_mismatches != 0) {
      final_status = 2;
    }
  }
  MPI_Bcast(&final_status, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  DestroyFabricAllocation(rank, &allocation);
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cudaStreamDestroy(stream);
  return final_status;
}

}  // namespace
}  // namespace mnnvl

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int rc = 0;
  try {
    rc = mnnvl::Run(argc, argv);
  } catch (const std::exception& ex) {
    std::cerr << "rank " << rank << " error: " << ex.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
    rc = 1;
  }
  MPI_Finalize();
  return rc;
}
