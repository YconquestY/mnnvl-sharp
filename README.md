# GB200 NVL72 MNNVL SHARP All-Reduce Sample

This repository contains a standalone CUDA 13.1 + MPI sample,
`mnnvl_sharp_allreduce`, for two GB200 NVL72 compute trays. It launches four MPI
ranks, maps one rank to each selected GPU, allocates a fabric-shareable GPU
buffer per rank, builds unicast and multicast virtual mappings, and uses NVLink
SHARP through PTX `multimem.ld_reduce` for an FP8 E4M3 all-reduce.

The v1 datatype policy is explicit:

- `FP8 E4M3`: supported and executed through the compiled `multimem` kernel path.
- `NVFP4`: reported unsupported under this CUDA 13.1 raw `multimem` path.
- `MXFP4`: reported unsupported under this CUDA 13.1 raw `multimem` path.

## Rack YAML

Use the compute-tray GPU pairs from the task: GPUs `0,2` on tray 0 and GPUs
`1,3` on tray 1. The binary revalidates that each tray’s selected pair maps to
two distinct `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID` values.

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

The rack key should match `hostname` inside the container. If it does not, launch
with `MNNVL_HOST_LABEL=<rack-key>` for that process. The helper launcher assumes
hostnames match the YAML keys.

## Build

Required packages/tools:

- CUDA Toolkit 13.1 with a Blackwell GPU code target.
- CMake 3.27 or newer.
- MPI with C++ bindings.
- `yaml-cpp`.
- Python 3 and PyYAML for helper scripts.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
```

The CMake configure step queries `nvcc --list-gpu-code` and fails if no
Blackwell target is available.

## Preflight

Run local checks on a selected container:

```bash
scripts/preflight_imex.sh
```

Run rack checks across both containers:

```bash
scripts/preflight_imex.sh /absolute/path/to/rack.yaml
```

The script checks the IMEX service, `channel0`, `nodes_config.cfg`, GPU
inventory, and `nvidia-imex-ctl -N`. Compute-node checks cannot prove NVSwitch
partition compatibility, so an operator must also run this on the fabric side:

```bash
nv show sdn partition
```

For the default rack-wide setup, expect one healthy `Default Partition` covering
all 72 GPUs. If user partitions exist, both selected trays must be in the same
partition.

## Launch

After you provide the YAML file and confirm preflight, launch with:

```bash
scripts/run_mnnvl_sharp.sh /absolute/path/to/rack.yaml \
  --bytes 1073741824 \
  --warmup 3 \
  --iters 20 \
  --reference-check-bytes 16777216
```

The launcher derives:

- `mpirun -np 4`
- YAML-order host list as `<rack-key>:2,<rack-key>:2`
- `--map-by ppr:2:node`
- a temporary SSH config using each entry’s `hostname` and `port`

## What The Binary Checks

At runtime the sample validates:

- exactly four MPI ranks and two ranks per selected container
- exactly two rack entries and two unique devices per entry
- selected GPUs on each tray have distinct `HOST_NUMA_ID` values
- all selected GPUs support CUDA VMM, fabric handles, and multicast
- all ranks can read canaries from all four fabric-imported unicast slots

It then initializes the local FP8 E4M3 payload, fences aliasing, runs the rank-0
NVLink SHARP all-reduce kernel over the multicast alias, and verifies:

- all four replicas have identical full-buffer device hashes
- rank 0 matches a CPU semantic reference for `E4M3 -> FP16 accumulation -> E4M3`
- rank 0 reports numeric error against a float32 CPU reference
- rank 0 reports SHARP kernel timing with CUDA events when measured iterations
  are requested

Use `--json` for a single machine-readable summary line.
