# Concrete Implementation Plan for `mnnvl_sharp_allreduce`

## Summary
- Implement a standalone CUDA 13.1 + MPI sample named `mnnvl_sharp_allreduce`.
- The sample will run one MPI rank per selected GPU, with the selected ranks drawn from a rack YAML file that can describe up to 18 compute trays and up to 4 GPUs per tray.
- The selected CUDA containers, SSH/control-plane addresses, SSH ports, and local CUDA device indices will come from the rack YAML file.
- The implementation will use CUDA Driver API virtual memory management, IMEX fabric handles, multicast objects, and selectable inline PTX `multimem` SHARP backends.
- The sample will execute `FP8 E4M3` through the legacy multimem backend and F16 through the TMA async backend. It will detect and report `NVFP4` and `MXFP4` as unsupported under the chosen CUDA 13.1 programming paths.
- The sample will produce correctness, precision, SHARP timing, and peer-access initialization timing output in a single run.
- The benchmark tooling will sweep rank counts `4, 8, 16, 32, 64, 72` from the same rack YAML and generate an initialization-latency graph.
- The implementation assumes the rack remains on the default 72-GPU NVLink partition and that the selected trays are members of the same IMEX domain.

## Deliverables
- `CMakeLists.txt`
- `README.md`
- `scripts/preflight_imex.sh`
- `scripts/run_mnnvl_sharp.sh`
- `scripts/run_init_sweep.py`
- `scripts/plot_init_latency.py`
- `src/main.cc`
- `src/config.h`
- `src/config.cc`
- `src/rack_config.h`
- `src/rack_config.cc`
- `src/runtime.h`
- `src/runtime.cc`
- `src/fabric_memory.h`
- `src/fabric_memory.cc`
- `src/sharp_kernels.cu`
- `src/e4m3_ref.h`
- `src/e4m3_ref.cc`

## Repository Structure
- `src/main.cc`
  - Own process startup, MPI setup, top-level orchestration, and final reporting.
- `src/config.*`
  - Parse CLI arguments and validate invariant combinations.
- `src/rack_config.*`
  - Parse and validate rack YAML, normalize host labels, container SSH addresses, SSH ports, and device lists.
- `src/runtime.*`
  - Own topology discovery, rack-entry/rank mapping, NUMA validation, CPU affinity, CUDA device selection, capability checks, MPI handle exchange, and result aggregation.
- `src/fabric_memory.*`
  - Own all CUDA Driver API VMM work: local allocation, fabric-handle export/import, symmetric unicast mapping, multicast object creation/import, binding, access setup, and teardown.
- `src/sharp_kernels.cu`
  - Own deterministic buffer initialization, alias fences, legacy SHARP `multimem` kernel, TMA async SHARP kernel, byte hash kernel, and device-side helpers.
- `src/e4m3_ref.*`
  - Own host-side `E4M3` encode/decode, F16 helpers, CPU semantic all-reduce references, and precision metrics.
- `scripts/preflight_imex.sh`
  - Validate host prerequisites before attempting the run.
- `scripts/run_mnnvl_sharp.sh`
  - Provide a reproducible rank-count-aware launch wrapper around `mpirun`.
- `scripts/run_init_sweep.py`
  - Sweep rank counts from one rack YAML and collect JSON results.
- `scripts/plot_init_latency.py`
  - Convert sweep CSV output into a PNG graph of initialization latency versus rank count.

## Build Plan
- Use CMake with a single executable target `mnnvl_sharp_allreduce`.
- Require:
  - CMake `>= 3.27`
  - CUDA Toolkit `13.1`
  - MPI with C++ bindings available through `find_package(MPI REQUIRED)`
  - `yaml-cpp` available through `find_package(yaml-cpp REQUIRED)`
  - Linux only
- Helper-script requirements:
  - Python 3
  - PyYAML for extracting MPI host order from `rack.yaml`
  - matplotlib for plotting sweep output
- Link against:
  - `CUDA::cudart`
  - `CUDA::cuda_driver`
  - `MPI::MPI_CXX`
  - `yaml-cpp`
  - `pthread`
- Configure the CUDA target as follows:
  - Build the `.cu` translation unit with relocatable device code disabled.
  - Detect supported GPU code via `nvcc --list-gpu-code` during configure.
  - Fail configuration if the compiler does not expose the Blackwell target needed for `multimem` `E4M3` support.
  - Export a compile definition such as `MNNVL_SHARP_HAS_E4M3=1` only when that check passes.
  - Detect PTX ISA support for `multimem.cp.reduce.async.bulk` and export `MNNVL_SHARP_HAS_TMA_ASYNC=1` only when CUDA 13.1/PTX 9.1 support is present.
- Default build type: `RelWithDebInfo`.

## CLI and Runtime Contract
- CLI:
  - `--rack-config rack.yaml`
  - `--rank-count <N>` with default `MPI_COMM_WORLD` size
  - `--rank-selection balanced|prefix` with default `balanced`
  - `--sharp-backend legacy|tma_async` with default `legacy`
  - `--bytes 1073741824`
  - `--warmup 3`
  - `--iters 20`
  - `--types auto|e4m3|f16`
  - `--reference-check-bytes <N>` with default `16777216`
  - `--init-only` to stop after peer-access initialization, smoke test, and JSON reporting
  - `--json` to emit one machine-readable summary line
- Rack YAML schema:
  ```yaml
  rack:
    - GB200-Rack5-01:
        - hostname: 10.135.1.31
        - port: 4399
        - device: [0, 1, 2, 3]
    - GB200-Rack5-02:
        - hostname: 10.135.1.32
        - device:
            - 0
            - 1
            - 2
            - 3
    # ...
    - GB200-Rack5-18:
        - hostname: 10.135.1.48
        - device: [0, 1, 2, 3]
  ```
- Rack YAML compatibility:
  - Canonical form is the sequence-of-single-key-maps form shown above.
  - `port` is optional and defaults to `4399`.
  - `device` must be a YAML sequence with 1 to 4 unique CUDA device indices and may be written inline as `[0, 2]` or as a block list.
  - The parser will reject scalar or comma-separated string device values such as `"0,2"`.
- Rack YAML semantics:
  - The key under each `rack` item is the compute-tray host label and should match `hostname` output on that tray.
  - `hostname` is the SSH/control-plane address used to reach the CUDA container.
  - `port` is the SSH port for the CUDA container.
  - `device` is an ordered list of local CUDA device indices to use inside that container.
  - YAML order defines rack-entry order, tray order, and rank-selection input order.
  - The CUDA container is assumed to be privileged and started with host networking and IPC, such as `--net=host --ipc=host`, so CUDA, NIC, memory, IMEX, and NVLink resources match the host machine.
  - `--rack-config` should be passed as an absolute path that is readable at the same path inside every selected container; `scripts/run_mnnvl_sharp.sh` will resolve it with `realpath` and check readability over SSH before launch.
- Flag semantics:
  - `--rank-count` is the number of ranks selected from the full YAML inventory and must match `MPI_COMM_WORLD` size when provided.
  - `--rank-selection balanced` selects device slot 0 across all trays, then slot 1 across all trays, and so on until `N` ranks are selected.
  - `--rank-selection prefix` selects trays in YAML order and devices in each tray's listed order until `N` ranks are selected.
  - `--sharp-backend legacy` uses the existing rank-0 `multimem.ld_reduce` plus `multimem.st` kernel.
  - `--sharp-backend tma_async` uses the PTX 9.1 `multimem.cp.reduce.async.bulk` instruction and requires `MNNVL_SHARP_HAS_TMA_ASYNC=1`.
  - `--types auto` selects `e4m3` for `legacy` and `f16` for `tma_async`.
  - `--types e4m3 --sharp-backend tma_async` is rejected because PTX 9.1 `multimem.cp.reduce.async.bulk` does not expose an `.e4m3` element type.
  - `--types f16 --sharp-backend legacy` is rejected for v1 because the legacy implementation only ships the packed `e4m3x4` kernel.
  - `--reference-check-bytes` is the number of leading payload bytes compared against CPU references for exactness and precision metrics.
  - Set `--reference-check-bytes` equal to `--bytes` for a full-buffer CPU reference.
- Invariants:
  - `1 <= rack entry count <= 18`
  - `1 <= device count per rack entry <= 4`
  - total available ranks is `sum(entry.devices.size())` and must be at most `72`
  - `1 <= world_size <= total available ranks`
  - if `--rank-count` is provided, `world_size == rank_count`
  - `bytes` is a multiple of `4` because the implementation uses packed `e4m3x4`
- Rank selection and mapping:
  - Parse the full YAML into an ordered rank inventory of `(rack entry, device)`.
  - Build the selected rank plan from `--rank-count` and `--rank-selection`.
  - Compute per-container slot counts from the selected rank plan.
  - Determine local rank with `MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, ...)`.
  - Gather container OS hostnames across `MPI_COMM_WORLD`.
  - Match the local OS hostname to a rack YAML key after normalizing short hostname vs FQDN.
  - If the OS hostname cannot be matched, allow `MNNVL_HOST_LABEL` to override the host label for that process.
  - Use `selected_rank_plan[world_rank]` as the authoritative `(container, GPU)` target for the rank.
  - Validate that the current container host label matches `selected_rank_plan[world_rank].host_label`.
  - Validate that local MPI process count for each container equals the selected slot count for that container.
- CPU affinity:
  - Query `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID` for the selected GPU.
  - Read `/sys/devices/system/node/node<id>/cpulist`.
  - Parse the cpulist and call `sched_setaffinity` so each rank is pinned to its GPU-local NUMA node.

## Preflight and Environment Checks
- Scope model for this sample:
  - `NVLink domain` is the physical rack-scale NVLink fabric.
  - `NVLink partition` is a fabric control-plane partition managed by FM/NVLSM/NMX on the switch side.
  - `IMEX domain` is the software service membership defined by `/etc/nvidia-imex/nodes_config.cfg`.
  - `IMEX channel` is the user/job isolation device node. In the single-user rack-wide setup, `channel0` on each selected node is sufficient.
- No extra partition configuration is required if the rack is still using the default partition:
  - expected state is one healthy `Default Partition` covering all `72` GPUs in the rack
  - if user partitions have been created, all selected trays must belong to the same NVLink partition or the sample must fail preflight
- `scripts/preflight_imex.sh` will support:
  - local-only mode: validate the current node
  - rack mode: `./scripts/preflight_imex.sh rack.yaml`
- The script will run the following checks and fail fast if any check fails:
  - local host checks:
    - `systemctl is-active nvidia-imex`
    - `nvidia-imex-ctl -N`
    - `nvidia-smi topo -m`
    - `nvidia-smi -L`
    - `ls /dev/nvidia-caps-imex-channels`
    - `cat /etc/nvidia-imex/nodes_config.cfg`
  - rack-mode checks when `rack.yaml` is provided:
    - parse each rack entry and use its `hostname` and `port` fields as the SSH target
    - `ssh -p <port> <control-host> hostname`
    - `ssh -p <port> <control-host> systemctl is-active nvidia-imex`
    - `ssh -p <port> <control-host> ls /dev/nvidia-caps-imex-channels`
    - `ssh -p <port> <control-host> cat /etc/nvidia-imex/nodes_config.cfg`
    - `ssh -p <port> <control-host> nvidia-imex-ctl -N`
- The script will print:
  - hostname
  - GPU inventory
  - rack YAML host label, control-plane address, and SSH port
  - IMEX channel inventory
  - normalized `nodes_config.cfg` membership
  - IMEX domain health
  - a note that selected GPU-to-NUMA mapping will be revalidated inside the binary
- The script will enforce the following policy:
  - every selected node must expose `channel0`
  - every selected container's `hostname` output must match the corresponding rack YAML key, unless the user documents that `MNNVL_HOST_LABEL` will be used at launch
  - every selected container must have identical normalized `nodes_config.cfg` contents
  - `nvidia-imex-ctl -N` in every selected container must show all nodes from `nodes_config.cfg` in `READY` state, or explicitly mark unreachable nodes as out of scope if the deployment is intentionally per-job
  - if `nodes_config.cfg` contains more than the selected trays, that is acceptable for this sample as long as the selected trays are in the same healthy IMEX domain
- External admin-side NVLink partition verification:
  - this cannot be verified from the CUDA process on the compute tray
  - required command on the leader NVSwitch or NMX endpoint:
    - `nv show sdn partition`
  - accepted result for the default rack-wide case:
    - one healthy `Default Partition` with `72` GPUs
  - if the rack is partitioned:
    - the selected trays must both map to the same user partition
    - the plan assumes an operator confirms this before the run
- The program itself will validate:
  - `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`
  - `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED`
  - `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`
  - every selected GPU resolves to a valid `HOST_NUMA_ID`
  - requested allocation size is compatible with both VMM and multicast granularity
  - imported-memory operations fail fast with a clear error if the runtime IMEX membership or permissions are inconsistent with the preflight assumptions

## Core Data Structures
- `Config`
  - parsed CLI values
  - rack config path
  - optional rank count
  - rank-selection policy
  - SHARP backend selection
  - reference-check byte count
  - byte counts
  - iteration counts
  - init-only mode
  - output mode
- `RackEntry`
  - `host_label`
  - `control_hostname`
  - `ssh_port`
  - ordered local CUDA device list
- `RankTarget`
  - `rank_index_in_plan`
  - `rack_index`
  - `host_label`
  - `control_hostname`
  - `ssh_port`
  - `gpu_ordinal`
- `RankPlan`
  - ordered selected rank targets
  - per-host selected device lists
  - per-host MPI slot counts
  - helper to emit MPI host aliases as `<host_label>:<selected_count>`
  - helper to emit an Open MPI rankfile so `world_rank == RankTarget.rank_index_in_plan`
- `RackConfig`
  - ordered list of `RackEntry`
  - helper lookup by host label
  - helper to count available rank targets
  - helper to emit an SSH config with per-host `HostName` and `Port`
- `RankInfo`
  - `world_rank`
  - `world_size`
  - `local_rank`
  - `local_size`
  - `rack_index`
  - `rack_count`
  - `host_label`
  - `control_hostname`
  - `ssh_port`
  - `hostname`
  - `gpu_ordinal`
  - `host_numa_id`
- `CapabilityInfo`
  - `vmm_supported`
  - `fabric_handle_supported`
  - `multicast_supported`
  - `sharp_e4m3_supported`
  - `sharp_tma_async_supported`
  - `nvfp4_supported`
  - `mxfp4_supported`
- `FabricAllocation`
  - local allocation handle
  - rounded allocation size
  - backend-specific allocation layout
  - exported opaque fabric handle bytes
  - local unicast base VA
  - vector-backed slot pointers `uc_slots[world_size]`
  - multicast object handle
  - multicast alias pointer `mc_alias`
  - for `legacy`: one payload region containing packed `e4m3x4`
  - for `tma_async`: an input region and a zero-initialized F16 output region used as the multicast reduction destination
- `RunStats`
  - peer-access initialization local latency
  - peer-access initialization min, median, p95, max across ranks
  - peer-access initialization global max latency
  - warmup latencies
  - measured latencies
  - end-to-end wall times
  - bytewise mismatch count
  - max absolute error
  - hash values per rank

## Memory and Handle Exchange Design
- Local allocation per rank:
  - Call `cudaSetDevice(gpu_ordinal)` first so runtime and driver APIs share the same primary context.
  - Create a local physical allocation with `cuMemCreate` using a device-local allocation property and fabric-shareable handle type.
  - For `legacy`, allocate one `payload_bytes` region for packed `e4m3x4` input/output.
  - For `tma_async`, allocate an input region plus an F16 output region, each `payload_bytes` unless the implementation later exposes separate sizing.
  - Zero the TMA output region on every rank before each TMA reduction because `multimem.cp.reduce.async.bulk` adds into the current destination contents.
  - After zeroing through the generic proxy, issue a generic-to-async proxy fence before the TMA async reduction kernel observes the output region.
  - Query allocation granularity via `cuMemGetAllocationGranularity`.
  - Query multicast granularity via `cuMulticastGetGranularity`.
  - Round the backend-specific total allocation size up to `alloc_bytes = align_up(layout_bytes, max(vmm_granularity, mc_granularity))`.
- Export/import:
  - Run `MPI_Barrier(world)` immediately before peer-access initialization.
  - Start a local wall-clock timer with `MPI_Wtime()`.
  - Export the local physical allocation to an opaque fabric-shareable handle.
  - Exchange fixed-size serialized handle blobs with `MPI_Allgather`.
  - Import all selected-rank handles on every rank.
- Symmetric unicast mapping:
  - Reserve one contiguous VA range of size `world_size * alloc_bytes`.
  - Map the imported physical handle for global rank `r` at `base + r * alloc_bytes`.
  - Set read-write access on the local device for every slot.
  - Stop the local timer after all imported rank slots are mapped and access permissions are set.
  - Use `MPI_Allgather` or `MPI_Allreduce` to report per-rank peer-init latencies and the global max latency.
  - Publish `uc_slots[world_size]` in global-rank order.
- Multicast object:
  - Rank 0 creates a single multicast object sized to the backend's reduction destination region and configured for `world_size` devices.
  - Rank 0 exports the multicast shareable handle and broadcasts it via MPI.
  - All ranks import the multicast object handle.
  - Each rank adds its local GPU to the multicast object.
  - For `legacy`, each rank binds its packed `E4M3` payload region.
  - For `tma_async`, each rank binds only its F16 output region as the multimem destination.
- Multicast alias mapping:
  - Reserve a second VA range of size `alloc_bytes`.
  - Map the imported multicast object into this range.
  - Set read-write access on the local GPU.
  - This `mc_alias` VA aliases the same local physical memory visible through `uc_slots[self]`.
- Teardown order:
  - synchronize stream
  - unmap `mc_alias`
  - unmap all unicast slots
  - release imported multicast handle
  - release local physical allocation
  - free reserved VA ranges

## Execution Sequence
1. Parse CLI, parse `rack.yaml`, and initialize MPI.
2. Build the rank plan from the YAML inventory, `--rank-count`, and `--rank-selection`.
3. Discover `world_rank`, `local_rank`, hostname set, matched rack entry, and selected GPU ordinal from the rank plan.
4. Select CUDA device, obtain current context, query capability attributes, query `HOST_NUMA_ID`, and pin CPU affinity.
5. Validate:
   - `world_size == selected rank count`
   - selected rank count is at most total YAML devices
   - per-container MPI local size matches selected per-container slot count
   - every MPI host maps to exactly one rack entry
   - all GPUs support VMM, fabric handles, and multicast
6. Determine datatype support:
   - `FP8 E4M3` is enabled only when both compile-time and runtime checks pass.
   - TMA async backend is enabled only when `MNNVL_SHARP_HAS_TMA_ASYNC=1` and runtime multicast support is present.
   - `NVFP4` and `MXFP4` are always reported unsupported in this implementation.
7. Allocate local physical fabric memory without mapping peer slots.
8. Run timed peer-access initialization:
   - global barrier
   - export local fabric handle
   - all-gather all selected handles
   - import handles
   - reserve/map `world_size` peer slots
   - set access permissions
   - stop local timer and report global max latency
9. Create/import the multicast object and map the `mc_alias`.
10. Run the symmetric-memory smoke test:
   - each rank writes a canary to `uc_slots[self]`
   - all ranks read the canaries from all selected rank slots
   - fail immediately if any readback mismatches
11. If `--init-only` is set, print JSON/human summary and skip SHARP all-reduce, CPU reference, and timing loops.
12. Initialize the local buffer for the selected backend:
    - `legacy`: packed `E4M3` input/output region
    - `tma_async`: deterministic F16 input region and zeroed F16 output region
13. Run warmup iterations:
    - reinitialize local buffer
    - global barrier
    - alias fence
    - `legacy`: SHARP kernel on rank 0
    - `tma_async`: contributor kernel on every rank
    - global barrier
14. Run measured iterations:
    - same flow as warmup
    - capture CUDA event elapsed time on rank 0
    - capture wall-clock elapsed time around the collective on all ranks
15. Copy the final buffer to host and gather per-rank hashes.
16. Rank 0 reconstructs host references, compares results, and prints the report.
17. Tear down mappings, handles, and MPI state.

## Kernel Plan
- `init_e4m3_kernel(uint32_t* dst_words, size_t word_count, int world_rank)`
  - One thread writes one packed `e4m3x4` word.
  - Generate four deterministic scalar values from `(world_rank, word_index, lane)`.
  - Restrict the generated range to small finite values so the 4-way sum remains representable in `E4M3`.
  - Encode the four values into one packed 32-bit word.
- `alias_fence_kernel()`
  - Single-block helper that issues `fence.proxy.alias`.
  - Launch:
    - after unicast initialization and before the legacy rank-0 SHARP read
    - after the SHARP store and before unicast-side validation reads
- `sharp_allreduce_e4m3_legacy_kernel(uint32_t* mc_alias_words, size_t word_count)`
  - Only rank 0 launches this kernel.
  - One thread processes one packed `e4m3x4` word.
  - Inline PTX flow:
    - `multimem.ld_reduce.add.acc::f16.e4m3x4` from `mc_alias_words[i]`
    - `multimem.st.e4m3x4` back to `mc_alias_words[i]`
  - Use a grid-stride loop for the full 1 GiB region.
- `sharp_allreduce_f16_tma_async_kernel(half* local_input, half* mc_output, size_t element_count)`
  - Every rank launches this kernel so every rank contributes its local data.
  - Each CTA processes one or more contiguous 16-byte-aligned tiles.
  - Issue `fence.proxy.async` before the first async reduction if the zeroing was performed by ordinary global-memory writes.
  - Load a tile from the local F16 input region into shared memory.
  - Issue `multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [mc_output + offset], [shared_tile], tile_bytes`.
  - Use the PTX bulk async-group commit and wait sequence so the kernel does not exit until all issued reductions are complete.
  - Rely on completion observation to make the async-proxy writes visible to later generic-proxy validation reads.
  - Require source shared-memory address, destination multimem address, and tile byte count to be 16-byte aligned.
  - Use grid-stride tiling for the full payload.
- `hash_kernel(const uint32_t* words, size_t word_count, uint64_t* out_hash)`
  - Produce one 64-bit hash per rank for fast identity comparison across replicas.
- `canary_kernel(uint32_t* self_slot, uint32_t rank_tag)`
  - Write a small known header into the local slot for the smoke test.

## Datatype Handling
- Reported support matrix:
  - `FP8 E4M3`: executable with `--sharp-backend legacy`
  - `F16`: executable with `--sharp-backend tma_async`
  - `NVFP4`: reported unsupported
  - `MXFP4`: reported unsupported
- Reasoning encoded into the program:
  - runtime support for SHARP requires multicast capability on the device
  - executable datatype support requires a compiled `multimem` kernel path
  - legacy backend ships an `E4M3` kernel because CUDA 13.1 exposes packed `e4m3x4` support through `multimem.ld_reduce`
  - TMA async backend ships an F16 kernel because `multimem.cp.reduce.async.bulk` supports `.f16/.bf16/.f32/...` and does not expose an `.e4m3` type
  - CUDA 13.1 does not expose a direct `NVFP4` or `MXFP4` SHARP reduction path through either chosen backend
- Output example:
  - `datatype_support: {"e4m3_legacy":"supported","f16_tma_async":"supported","nvfp4":"unsupported","mxfp4":"unsupported"}`

## Host Reference and Precision Comparison
- `src/e4m3_ref.*` will implement:
  - `uint8_t encode_e4m3(float x)`
  - `float decode_e4m3(uint8_t bits)`
  - `float round_to_fp16(float x)`
  - `void cpu_allreduce_e4m3_semantic(...)`
  - `void cpu_allreduce_e4m3_f32(...)`
  - `void cpu_allreduce_f16_semantic(...)`
  - `void cpu_allreduce_f16_f32(...)`
- Semantic reference:
  - regenerate all selected-rank input buffers using the exact same deterministic formula as the selected backend initializer
  - for `legacy`, decode each `E4M3` element to float, accumulate rank contributions in `FP16` rounding after each add, and re-encode the final result to `E4M3`
  - for `tma_async`, accumulate F16 input values with the selected F16 semantic model and compare against the F16 output region
- Precision reference:
  - for `legacy`, decode each `E4M3` element to float and accumulate in `float32`
  - for `tma_async`, decode each F16 element to float and accumulate in `float32`
  - keep the final float32 result for error reporting
- Comparison policy:
  - against semantic reference: exact byte-for-byte match is required
  - against float32 reference: report max absolute decoded error, mean absolute decoded error, and mismatch count after backend-specific requantization
- Scaling plan for the CPU reference:
  - default to comparing the first `16 MiB` of payload for precision metrics to keep host cost bounded
  - always compare the full-rank hashes for whole-buffer identity
  - allow full-buffer CPU reference by setting `--reference-check-bytes` equal to `--bytes`

## Timing Plan
- Peer-access initialization timing:
  - local physical allocation is completed before the timed region
  - all ranks enter `MPI_Barrier(world)`
  - all ranks start `MPI_Wtime()`
  - timed work includes fabric-handle export, handle all-gather, handle import, VA reservation, peer mapping, and access-permission setup
  - each rank stops its local timer after its peer mappings are usable
  - rank 0 reports min, median, p95, max, and global max over local peer-init timings
  - the sweep graph uses the global max peer-init latency as the y value
- Device timing:
  - `legacy`: rank 0 records CUDA events immediately before and after the rank-0 SHARP kernel
  - `tma_async`: every rank records CUDA events around its contributor kernel and rank 0 reports the max elapsed time across ranks
  - report min, median, p95, and max over measured iterations
- End-to-end timing:
  - all ranks record wall clock time around:
    - pre-kernel barrier
    - alias fence
    - backend SHARP kernel launch
    - post-kernel barrier
  - reduce the maximum wall time across ranks per iteration
  - report min, median, and p95 of the global max latency
- Bandwidth reporting:
  - logical all-reduce bytes = `world_size * bytes`
  - effective logical bandwidth = `logical_bytes / device_time`
  - print the metric with a note that it is a logical collective bandwidth, not raw link bandwidth

## Logging and Output
- Human-readable summary:
  - rack entry and GPU selection
  - selected SHARP backend
  - detected NUMA IDs
  - capability matrix
  - datatype support matrix
  - peer-access initialization timing
  - smoke-test result
  - correctness result
  - precision metrics
  - timing statistics
- JSON mode:
  - one compact JSON object printed by rank 0
  - include:
    - hostname mapping
    - GPU mapping
    - datatype support
    - selected rank count
    - selected SHARP backend
    - peer initialization timing
    - exact-match result
    - precision metrics
    - timing stats

## Scripts
- `scripts/preflight_imex.sh`
  - usage:
    - `./scripts/preflight_imex.sh`
    - `./scripts/preflight_imex.sh rack.yaml`
  - behavior:
    - local mode validates IMEX service, channel devices, local `nodes_config.cfg`, and local topology
    - rack mode parses `rack.yaml`, SSHes to each container using its `hostname` and optional `port`, compares `nodes_config.cfg` across the selected containers, and summarizes `nvidia-imex-ctl -N` state for each container
    - rack mode accepts 1 to 18 rack entries and 1 to 4 devices per entry
    - prints a reminder that NVLink partition verification is an external switch-side admin check, not a compute-node check
- `scripts/run_mnnvl_sharp.sh`
  - usage:
    - `./scripts/run_mnnvl_sharp.sh rack.yaml [--rank-count N] [mnnvl_sharp_allreduce args...]`
  - behavior:
    - build the project if needed
    - derive selected rank plan from `rack.yaml`, `--rank-count`, and `--rank-selection`
    - derive the MPI host list from the selected rank plan by using each selected host label with its selected slot count
    - generate a temporary SSH config mapping each host label to its `hostname` and `port`
    - generate a temporary Open MPI rankfile mapping each global rank to its selected host label and per-host local slot index
    - resolve `rack.yaml` to an absolute path and verify the same path is readable inside every selected container
    - verify the executable path is present and executable inside every selected container
    - run `mpirun -np <rank-count> --host <label0>:<slots0>,<label1>:<slots1> --rankfile <tmp-rankfile> --mca plm_rsh_args "-F <tmp-ssh-config>" ./build/mnnvl_sharp_allreduce --rack-config <abs-rack-yaml> --rank-count <rank-count>`
    - pass through additional CLI arguments after the rack config path
    - print a warning if the preflight summary indicates that the selected containers are in a larger rack-wide IMEX domain than the job itself
- `scripts/run_init_sweep.py`
  - usage:
    - `scripts/run_init_sweep.py rack.yaml --rank-counts 4,8,16,32,64,72 --repeats 3 --out-dir results/init_sweep`
  - behavior:
    - parse the full rack YAML and compute total available ranks
    - filter default rank counts to values not exceeding total available ranks
    - include total available ranks if it is at least 4 and not already in the rank-count list
    - for each rank count and repeat, invoke `scripts/run_mnnvl_sharp.sh rack.yaml --rank-count <N> --init-only --json`
    - parse rank-0 JSON and append one CSV row per run
    - write `init_latency.csv` with columns `rank_count,repeat,peer_init_ms,peer_init_min_ms,peer_init_median_ms,peer_init_p95_ms,peer_init_max_ms`
    - call `scripts/plot_init_latency.py` after all runs complete
- `scripts/plot_init_latency.py`
  - usage:
    - `scripts/plot_init_latency.py results/init_sweep/init_latency.csv --output results/init_sweep/init_latency.png`
  - behavior:
    - group CSV rows by rank count
    - plot median global-max peer-init latency as the y value
    - use min/max across repeats as error bars
    - label x axis `rank count`
    - label y axis `peer access initialization latency (ms)`

## Acceptance Criteria
- The binary launches successfully for any selected rank count from 1 through the YAML's available rank count, up to 72.
- The binary accepts `--rack-config rack.yaml` and rejects malformed rack YAML with actionable errors.
- The launcher can reach each privileged CUDA container using the YAML `hostname` and `port`, with `port` defaulting to `4399`.
- Preflight confirms that the selected trays are in the same IMEX domain and have access to `channel0`.
- Operator-side fabric check confirms either:
  - the rack is on the default 72-GPU NVLink partition, or
  - all selected trays are in the same user partition.
- Each selected GPU resolves to a valid host NUMA ID and rank pinning succeeds.
- Each rank allocates and exposes a 1 GiB fabric-shareable GPU allocation.
- Every rank can read canaries from all selected symmetric unicast slots.
- The program reports peer-access initialization latency, including global max `peer_init_ms`.
- The program reports `E4M3` legacy support, F16 TMA async support, and `NVFP4`/`MXFP4` unsupported.
- `--sharp-backend legacy` executes the rank-0 `multimem.ld_reduce` path.
- `--sharp-backend tma_async` executes `multimem.cp.reduce.async.bulk` when PTX 9.1 support is available and fails with an actionable error otherwise.
- The SHARP kernel executes without access faults or mapping errors.
- All selected-rank replicas produce identical final hashes.
- Rank 0 reports exact byte match against the CPU semantic reference for the validated region.
- Rank 0 reports numeric error metrics against the float32 reference.
- Timing statistics are printed when at least 1 measured iteration completes successfully.
- The sweep script emits a CSV and PNG graph for rank counts `4, 8, 16, 32, 64, 72` when those counts are available.

## Implementation Order
1. Extend CLI parsing for `--rank-count`, `--rank-selection`, `--sharp-backend`, and `--init-only`.
2. Extend rack YAML parsing to accept 1 to 18 entries and 1 to 4 devices per entry.
3. Implement rank-plan construction, per-container slot derivation, MPI topology discovery, GPU selection, and NUMA validation.
4. Convert fixed-size rank assumptions in runtime and fabric memory code to `world_size` vectors.
5. Add timed peer-access initialization around fabric-handle export, all-gather, import, mapping, and access setup.
6. Update multicast object creation/import, binding, and alias mapping for `world_size` devices and backend-specific destination regions.
7. Split SHARP kernels into legacy E4M3 and TMA async F16 paths.
8. Update canary, hashing, and host-reference paths for variable rank count and backend-specific output representation.
9. Add JSON fields and human-readable output for peer-init timing and selected backend.
10. Add rank-count-aware launcher behavior.
11. Add sweep and plotting scripts plus README usage documentation.

## Risks and Defaults
- Default: treat rack YAML order as tray order and rank-selection input order. The scripts and README must state this explicitly.
- Default: `balanced` rank selection spreads small rank counts across trays before taking additional GPUs from the same tray.
- Default: `--sharp-backend legacy` remains the compatibility path.
- Default: assume the NVL72 rack remains on the default 72-GPU NVLink partition unless an operator states otherwise.
- Default: accept a rack-wide IMEX domain even when the job only uses a subset of trays, as long as the selected trays are healthy members of that same domain.
- Default: `E4M3` is executable on the legacy backend and F16 is executable on the TMA async backend. FP4 support remains a reported capability decision, not a runtime path.
- Risk: CUDA 13.1 Blackwell toolchain naming may differ across environments. The configure step must fail loudly instead of guessing.
- Risk: exact SHARP semantic behavior may differ from the planned `FP16` accumulation model. If the first hardware run disagrees, the semantic reference becomes the first item to recalibrate using observed results and PTX documentation.
- Risk: `multimem.cp.reduce.async.bulk` does not directly support `.e4m3`; the TMA async backend validates F16 output and is not byte-equivalent to the legacy packed-E4M3 backend.
- Risk: full 1 GiB CPU reference is expensive. The default sampled precision comparison keeps runtime practical while still validating end-to-end correctness through hashes.
- Risk: `channel0` availability alone does not prove IMEX membership consistency. The preflight must compare `nodes_config.cfg` and `nvidia-imex-ctl -N` state across the selected containers.
- Risk: if the rack has been split into user partitions, compute-node-only checks will not prove partition compatibility. The runbook must require one switch-side `nv show sdn partition` confirmation before debugging CUDA import failures.
