# GB200 NVL72 MNNVL SHARP All-Reduce Sample

This repository contains a standalone CUDA 13.1 + MPI sample named
`mnnvl_sharp_allreduce`. It reads a rack YAML inventory, launches one MPI rank
per selected GPU, allocates fabric-shareable GPU memory with CUDA Driver API
VMM, maps symmetric unicast peer slots, creates an NVLink multicast alias, and
uses NVLink SHARP through selectable PTX `multimem` backends.

The datatype policy is explicit:

- `FP8 E4M3`: supported with `--sharp-backend legacy`.
- `F16`: supported with `--sharp-backend tma_async` when CUDA 13.1/PTX 9.1
  accepts `multimem.cp.reduce.async.bulk`.
- `NVFP4`: reported unsupported under this CUDA 13.1 raw `multimem` path.
- `MXFP4`: reported unsupported under this CUDA 13.1 raw `multimem` path.

## Rack YAML

The rack YAML may describe any subset of the rack, from one tray to all 18 trays.
Each tray entry may list one to four local CUDA device ordinals. YAML order is
the deterministic input order for rank selection.

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
```

`port` defaults to `4399`. The rack key should match `hostname` inside the
container. If it does not, launch that process with `MNNVL_HOST_LABEL=<rack-key>`.
The helper launcher assumes the container hostnames match the YAML keys.

## Build

Required packages/tools:

- CUDA Toolkit 13.1 with a feature-specific Blackwell target such as `sm_100a`.
- CMake 3.27 or newer.
- MPI with C++ bindings.
- `yaml-cpp`.
- Python 3 and PyYAML for helper scripts.
- matplotlib for plotting sweep output.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
```

The CMake configure step queries `nvcc` and fails if a Blackwell target required
for FP8 `multimem` support is unavailable. It also probes the PTX 9.1 TMA-based
multimem reduction instruction and compiles the `tma_async` backend only when
the local CUDA compiler accepts it.

## Preflight

Run local checks on one selected container:

```bash
scripts/preflight_imex.sh
```

Run rack checks across the YAML-selected containers:

```bash
scripts/preflight_imex.sh /absolute/path/to/rack.yaml
```

The script checks IMEX channel visibility, `channel0`, normalized
`nodes_config.cfg`, GPU inventory, and available IMEX health commands. Compute
node checks cannot prove NVSwitch partition compatibility, so an operator must
also run this on the fabric side:

```bash
nv show sdn partition
```

For the default rack-wide setup, expect one healthy `Default Partition` covering
all 72 GPUs. If user partitions exist, all selected trays must be in the same
partition.

## Launch

Launch the default full YAML inventory:

```bash
scripts/run_mnnvl_sharp.sh /absolute/path/to/rack.yaml \
  --sharp-backend legacy \
  --bytes 1073741824 \
  --warmup 3 \
  --iters 20 \
  --reference-check-bytes 16777216
```

Launch the TMA async F16 backend:

```bash
scripts/run_mnnvl_sharp.sh /absolute/path/to/rack.yaml \
  --sharp-backend tma_async \
  --types auto \
  --bytes 1073741824 \
  --warmup 3 \
  --iters 20 \
  --reference-check-bytes 16777216
```

Launch a selected rank count:

```bash
scripts/run_mnnvl_sharp.sh /absolute/path/to/rack.yaml \
  --rank-count 8 \
  --rank-selection balanced \
  --init-only \
  --json
```

Rank selection policies:

- `balanced`: slot 0 across all trays, then slot 1 across all trays, and so on.
- `prefix`: trays in YAML order, then devices in each tray's listed order.

Backend and datatype policies:

- `--types auto` selects `e4m3` for `legacy` and `f16` for `tma_async`.
- `--types e4m3 --sharp-backend tma_async` is rejected.
- `--types f16 --sharp-backend legacy` is rejected.
- `tma_async` requires `--bytes` to be 16-byte aligned.

The launcher derives `mpirun -np <rank-count>`, per-host slots, a temporary SSH
config from each entry's `hostname` and `port`, and an Open MPI rankfile so
global rank order matches the selected rank plan.

## Runtime Checks

At runtime the sample validates:

- MPI world size equals the selected rank count.
- selected ranks map to the expected `(container, GPU)` plan.
- each selected GPU resolves to a valid `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID`.
- each rank is pinned to its GPU-local NUMA CPU node.
- all selected GPUs support CUDA VMM, fabric handles, and multicast.
- all ranks can read canaries from all selected fabric-imported unicast slots.

The program reports peer-access initialization latency separately from SHARP
timing. `--init-only` stops after fabric handle exchange, peer mapping,
multicast alias setup, and the symmetric-memory smoke test.

## Sweep

Run the initialization-latency sweep:

```bash
scripts/run_init_sweep.py /absolute/path/to/rack.yaml \
  --rank-counts 4,8,16,32,64,72 \
  --repeats 3 \
  --out-dir results/init_sweep
```

The sweep runs the launcher with `--init-only --json`, writes
`results/init_sweep/init_latency.csv`, and writes
`results/init_sweep/init_latency.png` with global-max peer initialization
latency versus rank count.

Use `--json` on normal runs for a single machine-readable summary line.
