# Concrete Implementation Plan for `mnnvl_sharp_allreduce`

## Summary
- Implement a standalone CUDA 13.1 + MPI sample named `mnnvl_sharp_allreduce`.
- The sample will run on 2 GB200 NVL72 compute trays with 2 MPI ranks per tray and 1 GPU per rank.
- The selected CUDA containers, SSH/control-plane addresses, SSH ports, and local CUDA device indices will come from a rack YAML file.
- The implementation will use CUDA Driver API virtual memory management, IMEX fabric handles, multicast objects, and inline PTX `multimem` instructions for the SHARP path.
- The sample will only execute the SHARP all-reduce for `FP8 E4M3`. It will detect and report `NVFP4` and `MXFP4` as unsupported under the chosen CUDA 13.1 programming path.
- The sample will produce correctness, precision, and timing output in a single run.
- The implementation assumes the rack remains on the default 72-GPU NVLink partition and that the selected trays are members of the same IMEX domain.

## Deliverables
- `CMakeLists.txt`
- `README.md`
- `scripts/preflight_imex.sh`
- `scripts/run_mnnvl_sharp.sh`
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
  - Own deterministic buffer initialization, alias fences, SHARP `multimem` kernel, byte hash kernel, and device-side helpers.
- `src/e4m3_ref.*`
  - Own host-side `E4M3` encode/decode, CPU semantic all-reduce reference, and precision metrics.
- `scripts/preflight_imex.sh`
  - Validate host prerequisites before attempting the run.
- `scripts/run_mnnvl_sharp.sh`
  - Provide a reproducible two-node launch wrapper around `mpirun`.

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
- Default build type: `RelWithDebInfo`.

## CLI and Runtime Contract
- CLI:
  - `--rack-config rack.yaml`
  - `--bytes 1073741824`
  - `--warmup 3`
  - `--iters 20`
  - `--types auto|e4m3`
  - `--reference-check-bytes <N>` with default `16777216`
  - `--json` to emit one machine-readable summary line
- Rack YAML schema:
  ```yaml
  rack:
    - GB200-Rack5-01:
        - hostname: 10.135.1.31
        - port: 4399
        - device: [0, 2]
    - GB200-Rack5-02:
        - hostname: 10.135.1.32
        - device:
            - 1
            - 3
  ```
- Rack YAML compatibility:
  - Canonical form is the sequence-of-single-key-maps form shown above.
  - `port` is optional and defaults to `4399`.
  - `device` must be a YAML sequence and may be written inline as `[0, 2]` or as a block list.
  - The parser will reject scalar or comma-separated string device values such as `"0,2"`.
- Rack YAML semantics:
  - The key under each `rack` item is the compute-tray host label and should match `hostname` output on that tray.
  - `hostname` is the SSH/control-plane address used to reach the CUDA container.
  - `port` is the SSH port for the CUDA container.
  - `device` is an ordered list of local CUDA device indices to use inside that container.
  - YAML order defines rack-entry order, tray order, and MPI host order.
  - The CUDA container is assumed to be privileged and started with host networking and IPC, such as `--net=host --ipc=host`, so CUDA, NIC, memory, IMEX, and NVLink resources match the host machine.
  - `--rack-config` should be passed as an absolute path that is readable at the same path inside every selected container; `scripts/run_mnnvl_sharp.sh` will resolve it with `realpath` and check readability over SSH before launch.
- Flag semantics:
  - `--reference-check-bytes` is the number of leading payload bytes compared against CPU references for exactness and precision metrics.
  - Set `--reference-check-bytes` equal to `--bytes` for a full-buffer CPU reference.
- Invariants:
  - `world_size == 4`
  - exactly `2` ranks per selected container
  - exactly `2` unique rack entries
  - `rack.yaml` contains exactly `2` rack entries for this v1 sample
  - each rack entry contains exactly `2` unique local CUDA device ordinals
  - `bytes` is a multiple of `4` because the implementation uses packed `e4m3x4`
- Rack-entry/rank mapping:
  - Determine local rank with `MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, ...)`.
  - Gather container OS hostnames across `MPI_COMM_WORLD`.
  - Match the local OS hostname to a rack YAML key after normalizing short hostname vs FQDN.
  - If the OS hostname cannot be matched, allow `MNNVL_HOST_LABEL` to override the host label for that process.
  - Local rank `0` maps to the first device in the matched rack entry. Local rank `1` maps to the second device in the matched rack entry.
  - The launcher must start containers in YAML order so global rank order remains deterministic.
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
  - if user partitions have been created, the selected two trays must belong to the same NVLink partition or the sample must fail preflight
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
  - a note that GPU-to-NUMA uniqueness for the selected pairs will be revalidated inside the binary
- The script will enforce the following policy:
  - every selected node must expose `channel0`
  - every selected container's `hostname` output must match the corresponding rack YAML key, unless the user documents that `MNNVL_HOST_LABEL` will be used at launch
  - every selected container must have identical normalized `nodes_config.cfg` contents
  - `nvidia-imex-ctl -N` in every selected container must show all nodes from `nodes_config.cfg` in `READY` state, or explicitly mark unreachable nodes as out of scope if the deployment is intentionally per-job
  - if `nodes_config.cfg` contains more than the 2 selected trays, that is acceptable for this sample as long as the selected trays are in the same healthy IMEX domain
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
  - selected GPU pair on each tray does not resolve to the same `HOST_NUMA_ID`
  - requested allocation size is compatible with both VMM and multicast granularity
  - imported-memory operations fail fast with a clear error if the runtime IMEX membership or permissions are inconsistent with the preflight assumptions

## Core Data Structures
- `Config`
  - parsed CLI values
  - rack config path
  - reference-check byte count
  - byte counts
  - iteration counts
  - output mode
- `RackEntry`
  - `host_label`
  - `control_hostname`
  - `ssh_port`
  - ordered local CUDA device list
- `RackConfig`
  - ordered list of `RackEntry`
  - helper lookup by host label
  - helper to emit MPI host aliases as `<host_label>:2`
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
  - `nvfp4_supported`
  - `mxfp4_supported`
- `FabricAllocation`
  - local allocation handle
  - rounded allocation size
  - exported opaque fabric handle bytes
  - local unicast base VA
  - slot pointers `uc_slots[4]`
  - multicast object handle
  - multicast alias pointer `mc_alias`
- `RunStats`
  - warmup latencies
  - measured latencies
  - end-to-end wall times
  - bytewise mismatch count
  - max absolute error
  - hash values per rank

## Memory and Handle Exchange Design
- Local allocation per rank:
  - Call `cudaSetDevice(gpu_ordinal)` first so runtime and driver APIs share the same primary context.
  - Create a local 1 GiB physical allocation with `cuMemCreate` using a device-local allocation property and fabric-shareable handle type.
  - Query allocation granularity via `cuMemGetAllocationGranularity`.
  - Query multicast granularity via `cuMulticastGetGranularity`.
  - Round `bytes` up to `alloc_bytes = align_up(bytes, max(vmm_granularity, mc_granularity))`.
- Export/import:
  - Export the local physical allocation to an opaque fabric-shareable handle.
  - Exchange fixed-size serialized handle blobs with `MPI_Allgather`.
  - Import all four handles on every rank.
- Symmetric unicast mapping:
  - Reserve one contiguous VA range of size `4 * alloc_bytes`.
  - Map the imported physical handle for global rank `r` at `base + r * alloc_bytes`.
  - Set read-write access on the local device for every slot.
  - Publish:
    - `uc_slots[0]`
    - `uc_slots[1]`
    - `uc_slots[2]`
    - `uc_slots[3]`
- Multicast object:
  - Rank 0 creates a single multicast object sized to `alloc_bytes` and configured for `4` devices.
  - Rank 0 exports the multicast shareable handle and broadcasts it via MPI.
  - All ranks import the multicast object handle.
  - Each rank adds its local GPU to the multicast object.
  - Each rank binds its own local physical allocation into the multicast object for the full range.
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
2. Discover `world_rank`, `local_rank`, hostname set, matched rack entry, and selected GPU ordinal.
3. Select CUDA device, obtain current context, query capability attributes, query `HOST_NUMA_ID`, and pin CPU affinity.
4. Validate:
   - 4 ranks total
   - 2 ranks per selected container
   - 2 rack entries in the YAML
   - 2 unique device indices per rack entry
   - every MPI host maps to exactly one rack entry
   - different NUMA CPU per selected local GPU pair
   - all GPUs support VMM, fabric handles, and multicast
5. Determine datatype support:
   - `FP8 E4M3` is enabled only when both compile-time and runtime checks pass.
   - `NVFP4` and `MXFP4` are always reported unsupported in this implementation.
6. Allocate local fabric memory, export/import handles, and build the 4-slot symmetric unicast table.
7. Create/import the multicast object and map the `mc_alias`.
8. Run the symmetric-memory smoke test:
   - each rank writes a canary to `uc_slots[self]`
   - all ranks read the canaries from all 4 slots
   - fail immediately if any readback mismatches
9. Initialize the local buffer for the supported datatype.
10. Run warmup iterations:
    - reinitialize local buffer
    - global barrier
    - alias fence
    - SHARP kernel on rank 0
    - global barrier
11. Run measured iterations:
    - same flow as warmup
    - capture CUDA event elapsed time on rank 0
    - capture wall-clock elapsed time around the collective on all ranks
12. Copy the final buffer to host and gather per-rank hashes.
13. Rank 0 reconstructs host references, compares results, and prints the report.
14. Tear down mappings, handles, and MPI state.

## Kernel Plan
- `init_e4m3_kernel(uint32_t* dst_words, size_t word_count, int world_rank)`
  - One thread writes one packed `e4m3x4` word.
  - Generate four deterministic scalar values from `(world_rank, word_index, lane)`.
  - Restrict the generated range to small finite values so the 4-way sum remains representable in `E4M3`.
  - Encode the four values into one packed 32-bit word.
- `alias_fence_kernel()`
  - Single-block helper that issues `fence.proxy.alias`.
  - Launch:
    - after unicast initialization and before the rank-0 SHARP read
    - after the SHARP store and before unicast-side validation reads
- `sharp_allreduce_e4m3_kernel(uint32_t* mc_alias_words, size_t word_count)`
  - Only rank 0 launches this kernel.
  - One thread processes one packed `e4m3x4` word.
  - Inline PTX flow:
    - `multimem.ld_reduce.add.acc::f16.e4m3x4` from `mc_alias_words[i]`
    - `multimem.st.e4m3x4` back to `mc_alias_words[i]`
  - Use a grid-stride loop for the full 1 GiB region.
- `hash_kernel(const uint32_t* words, size_t word_count, uint64_t* out_hash)`
  - Produce one 64-bit hash per rank for fast identity comparison across replicas.
- `canary_kernel(uint32_t* self_slot, uint32_t rank_tag)`
  - Write a small known header into the local slot for the smoke test.

## Datatype Handling
- Reported support matrix:
  - `FP8 E4M3`: executable
  - `NVFP4`: reported unsupported
  - `MXFP4`: reported unsupported
- Reasoning encoded into the program:
  - runtime support for SHARP requires multicast capability on the device
  - executable datatype support requires a compiled `multimem` kernel path
  - this implementation only ships an `E4M3` kernel because CUDA 13.1 does not expose a direct `NVFP4` or `MXFP4` SHARP reduction path through the chosen `multimem` API
- Output example:
  - `datatype_support: {"e4m3":"supported","nvfp4":"unsupported","mxfp4":"unsupported"}`

## Host Reference and Precision Comparison
- `src/e4m3_ref.*` will implement:
  - `uint8_t encode_e4m3(float x)`
  - `float decode_e4m3(uint8_t bits)`
  - `float round_to_fp16(float x)`
  - `void cpu_allreduce_e4m3_semantic(...)`
  - `void cpu_allreduce_e4m3_f32(...)`
- Semantic reference:
  - regenerate all 4 input buffers using the exact same deterministic formula as the device initializer
  - decode each `E4M3` element to float
  - accumulate rank contributions in `FP16` rounding after each add
  - re-encode the final result to `E4M3`
- Precision reference:
  - decode each `E4M3` element to float
  - accumulate in `float32`
  - keep the final float32 result for error reporting
- Comparison policy:
  - against semantic reference: exact byte-for-byte match is required
  - against float32 reference: report
    - max absolute decoded error
    - mean absolute decoded error
    - mismatch count after `E4M3` requantization
- Scaling plan for the CPU reference:
  - default to comparing the first `16 MiB` of payload for precision metrics to keep host cost bounded
  - always compare the full-rank hashes for whole-buffer identity
  - allow full-buffer CPU reference by setting `--reference-check-bytes` equal to `--bytes`

## Timing Plan
- Device timing:
  - rank 0 records CUDA events immediately before and after the SHARP kernel
  - report min, median, p95, and max over measured iterations
- End-to-end timing:
  - all ranks record wall clock time around:
    - pre-kernel barrier
    - alias fence
    - rank-0 kernel launch
    - post-kernel barrier
  - reduce the maximum wall time across ranks per iteration
  - report min, median, and p95 of the global max latency
- Bandwidth reporting:
  - logical all-reduce bytes = `4 * bytes`
  - effective logical bandwidth = `logical_bytes / device_time`
  - print the metric with a note that it is a logical collective bandwidth, not raw link bandwidth

## Logging and Output
- Human-readable summary:
  - rack entry and GPU selection
  - detected NUMA IDs
  - capability matrix
  - datatype support matrix
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
    - prints a reminder that NVLink partition verification is an external switch-side admin check, not a compute-node check
- `scripts/run_mnnvl_sharp.sh`
  - usage:
    - `./scripts/run_mnnvl_sharp.sh rack.yaml`
  - behavior:
    - build the project if needed
    - derive the MPI host list from `rack.yaml` with PyYAML by using each rack entry's host label with two slots
    - generate a temporary SSH config mapping each host label to its `hostname` and `port`
    - resolve `rack.yaml` to an absolute path and verify the same path is readable inside every selected container
    - verify the executable path is present and executable inside every selected container
    - run `mpirun -np 4 --host <label0>:2,<label1>:2 --map-by ppr:2:node --mca plm_rsh_args "-F <tmp-ssh-config>" ./build/mnnvl_sharp_allreduce --rack-config <abs-rack-yaml>`
    - pass through additional CLI arguments after the rack config path
    - print a warning if the preflight summary indicates that the selected containers are in a larger rack-wide IMEX domain than the job itself

## Acceptance Criteria
- The binary launches successfully with 4 ranks across 2 trays.
- The binary accepts `--rack-config rack.yaml` and rejects malformed rack YAML with actionable errors.
- The launcher can reach each privileged CUDA container using the YAML `hostname` and `port`, with `port` defaulting to `4399`.
- Preflight confirms that the selected trays are in the same IMEX domain and have access to `channel0`.
- Operator-side fabric check confirms either:
  - the rack is on the default 72-GPU NVLink partition, or
  - both trays are in the same user partition.
- The selected GPU pair on each tray is confirmed to span distinct NUMA CPUs.
- Each rank allocates and exposes a 1 GiB fabric-shareable GPU allocation.
- Every rank can read canaries from all 4 symmetric unicast slots.
- The program reports `E4M3` supported and `NVFP4`/`MXFP4` unsupported.
- The SHARP kernel executes without access faults or mapping errors.
- All 4 replicas produce identical final hashes.
- Rank 0 reports exact byte match against the CPU semantic reference for the validated region.
- Rank 0 reports numeric error metrics against the float32 reference.
- Timing statistics are printed when at least 1 measured iteration completes successfully.

## Implementation Order
1. Set up `CMakeLists.txt`, executable skeleton, CLI parsing, and `yaml-cpp` integration.
2. Implement rack YAML parsing, MPI topology discovery, rack-entry mapping, GPU selection, and NUMA validation.
3. Implement CUDA capability checks and CPU affinity pinning.
4. Implement local fabric allocation plus handle export/import.
5. Implement symmetric unicast mapping and the canary smoke test.
6. Implement multicast object creation/import, binding, and alias mapping.
7. Implement `E4M3` init, alias fence, and rank-0 SHARP kernel.
8. Implement host-side `E4M3` encode/decode and CPU references.
9. Implement result hashing, timing, and reporting.
10. Add helper scripts and README usage documentation.

## Risks and Defaults
- Default: treat rack YAML order as tray order and MPI host order. The scripts and README must state this explicitly.
- Default: assume the NVL72 rack remains on the default 72-GPU NVLink partition unless an operator states otherwise.
- Default: accept a rack-wide IMEX domain even when the job only uses 2 trays, as long as the selected trays are healthy members of that same domain.
- Default: only `E4M3` is executable in v1. FP4 support remains a reported capability decision, not a runtime path.
- Risk: CUDA 13.1 Blackwell toolchain naming may differ across environments. The configure step must fail loudly instead of guessing.
- Risk: exact SHARP semantic behavior may differ from the planned `FP16` accumulation model. If the first hardware run disagrees, the semantic reference becomes the first item to recalibrate using observed results and PTX documentation.
- Risk: full 1 GiB CPU reference is expensive. The default sampled precision comparison keeps runtime practical while still validating end-to-end correctness through hashes.
- Risk: `channel0` availability alone does not prove IMEX membership consistency. The preflight must compare `nodes_config.cfg` and `nvidia-imex-ctl -N` state across the selected containers.
- Risk: if the rack has been split into user partitions, compute-node-only checks will not prove partition compatibility. The runbook must require one switch-side `nv show sdn partition` confirmation before debugging CUDA import failures.
