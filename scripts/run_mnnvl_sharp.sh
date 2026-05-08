#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage:
  scripts/run_mnnvl_sharp.sh rack.yaml [--rank-count N] [--rank-selection balanced|prefix] [mnnvl_sharp_allreduce args...]

The launcher builds the local target if needed, derives the selected rank plan
from rack.yaml, creates a temporary SSH config and Open MPI rankfile, verifies
the rack YAML path and executable on each selected container, and launches
one MPI rank per selected GPU.
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

[[ "${1:-}" != "-h" && "${1:-}" != "--help" ]] || { usage; exit 0; }
[[ $# -ge 1 ]] || { usage; exit 1; }

rack_yaml=$1
shift
[[ -r "$rack_yaml" ]] || die "cannot read $rack_yaml"

rank_count_arg=""
rank_selection="balanced"
scan_args=("$@")
idx=0
while [[ $idx -lt ${#scan_args[@]} ]]; do
  arg=${scan_args[$idx]}
  case "$arg" in
    --rank-count)
      next=$((idx + 1))
      [[ $next -lt ${#scan_args[@]} ]] || die "--rank-count requires a value"
      rank_count_arg=${scan_args[$next]}
      idx=$((idx + 2))
      ;;
    --rank-selection)
      next=$((idx + 1))
      [[ $next -lt ${#scan_args[@]} ]] || die "--rank-selection requires a value"
      rank_selection=${scan_args[$next]}
      idx=$((idx + 2))
      ;;
    *)
      idx=$((idx + 1))
      ;;
  esac
done
[[ "$rank_selection" == "balanced" || "$rank_selection" == "prefix" ]] || die "--rank-selection must be balanced or prefix"

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
rack_yaml_abs=$(realpath "$rack_yaml")
app_dir=${MNNVL_APP_DIR:-/home/ubuntu/mnnvl_sharp}
build_dir="$repo_root/build"
remote_build_dir="$app_dir/build"
exe="$build_dir/mnnvl_sharp_allreduce"
remote_exe="$remote_build_dir/mnnvl_sharp_allreduce"
ssh_identity=${MNNVL_SSH_IDENTITY:-/home/ubuntu/.ssh/id_rsa}
ssh_known_hosts=${MNNVL_SSH_KNOWN_HOSTS:-/home/ubuntu/.ssh/known_hosts}
ssh_user=${MNNVL_SSH_USER:-ubuntu}
local_label=${MNNVL_LOCAL_LABEL:-$(hostname -s)}
cuda_compiler=${MNNVL_CUDA_COMPILER:-/usr/local/cuda/bin/nvcc}
cuda_arch=${MNNVL_CUDA_ARCHITECTURES:-100a}
mpi_tcp_if=${MNNVL_MPI_TCP_IF_INCLUDE:-10.135.1.0/26}
skip_remote_build=${MNNVL_SKIP_REMOTE_BUILD:-0}

command -v python3 >/dev/null 2>&1 || die "python3 is required"
python3 - <<'PY' >/dev/null 2>&1 || die "PyYAML is required"
import yaml
PY
command -v cmake >/dev/null 2>&1 || die "cmake is required"
command -v mpirun >/dev/null 2>&1 || die "mpirun is required"
command -v ssh >/dev/null 2>&1 || die "ssh is required"

mkdir -p "$build_dir"
if [[ ! -x "$exe" ]]; then
  CUDACXX="$cuda_compiler" cmake -S "$repo_root" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER="$cuda_compiler" \
    -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch"
  cmake --build "$build_dir" -j"$(nproc)"
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
host_file="$tmpdir/hosts.txt"
ssh_config="$tmpdir/ssh_config"
entries_file="$tmpdir/selected_entries.tsv"
rankfile="$tmpdir/rankfile"
rank_count_file="$tmpdir/rank_count.txt"

python3 - "$rack_yaml_abs" "$rank_count_arg" "$rank_selection" \
  "$host_file" "$ssh_config" "$entries_file" "$rankfile" "$rank_count_file" \
  "$ssh_identity" "$ssh_known_hosts" "$ssh_user" <<'PY'
import sys
import yaml
from collections import OrderedDict, defaultdict

rack_yaml, rank_count_arg, selection = sys.argv[1:4]
host_file, ssh_config, entries_file, rankfile, rank_count_file = sys.argv[4:9]
ssh_identity, ssh_known_hosts, ssh_user = sys.argv[9:12]

def merge_body(label, body):
    if isinstance(body, list):
        merged = {}
        for part in body:
            if not isinstance(part, dict):
                raise SystemExit(f"{label}: body list entries must be maps")
            merged.update(part)
        return merged
    if isinstance(body, dict):
        return body
    raise SystemExit(f"{label}: body must be a map or list of maps")

with open(rack_yaml, "r", encoding="utf-8") as f:
    doc = yaml.safe_load(f)
if not isinstance(doc, dict) or not isinstance(doc.get("rack"), list):
    raise SystemExit("rack YAML must contain top-level sequence 'rack'")

entries = []
seen_labels = set()
total_devices = 0
for item in doc["rack"]:
    if not isinstance(item, dict) or len(item) != 1:
        raise SystemExit("each rack item must be a single-key map")
    label, body = next(iter(item.items()))
    short = str(label).split(".", 1)[0]
    if short in seen_labels:
        raise SystemExit(f"duplicate rack host label: {label}")
    seen_labels.add(short)
    merged = merge_body(label, body)
    host = merged.get("hostname")
    port = int(merged.get("port", 4399))
    devices = merged.get("device")
    if not isinstance(host, str):
        raise SystemExit(f"{label}: hostname must be a string")
    if not isinstance(devices, list) or not (1 <= len(devices) <= 4):
        raise SystemExit(f"{label}: device must be a YAML list with 1 to 4 entries")
    devices = [int(x) for x in devices]
    if len(set(devices)) != len(devices) or any(x < 0 for x in devices):
        raise SystemExit(f"{label}: devices must be unique non-negative integers")
    entries.append({"label": str(label), "host": host, "port": port, "devices": devices})
    total_devices += len(devices)

if not (1 <= len(entries) <= 18):
    raise SystemExit("rack YAML must contain 1 to 18 rack entries")
if total_devices > 72:
    raise SystemExit("rack YAML describes more than 72 CUDA devices")

rank_count = int(rank_count_arg) if rank_count_arg else total_devices
if not (1 <= rank_count <= total_devices):
    raise SystemExit(f"rank count {rank_count} must be between 1 and {total_devices}")

plan = []
if selection == "prefix":
    for rack_index, entry in enumerate(entries):
        for gpu in entry["devices"]:
            plan.append((rack_index, entry, gpu))
            if len(plan) == rank_count:
                break
        if len(plan) == rank_count:
            break
elif selection == "balanced":
    for slot in range(4):
        for rack_index, entry in enumerate(entries):
            if slot < len(entry["devices"]):
                plan.append((rack_index, entry, entry["devices"][slot]))
                if len(plan) == rank_count:
                    break
        if len(plan) == rank_count:
            break
else:
    raise SystemExit("--rank-selection must be balanced or prefix")

counts = OrderedDict()
selected_labels = set()
for _, entry, _ in plan:
    counts[entry["label"]] = counts.get(entry["label"], 0) + 1
    selected_labels.add(entry["label"])

with open(host_file, "w", encoding="utf-8") as f:
    f.write(",".join(f"{label}:{count}" for label, count in counts.items()))

with open(ssh_config, "w", encoding="utf-8") as f:
    for entry in entries:
        if entry["label"] not in selected_labels:
            continue
        f.write(f"Host {entry['label']}\n")
        f.write(f"  HostName {entry['host']}\n")
        f.write(f"  User {ssh_user}\n")
        f.write(f"  Port {entry['port']}\n")
        f.write(f"  IdentityFile {ssh_identity}\n")
        f.write("  IdentitiesOnly yes\n")
        f.write("  PasswordAuthentication no\n")
        f.write(f"  UserKnownHostsFile {ssh_known_hosts}\n")
        f.write("  StrictHostKeyChecking accept-new\n")
        f.write("  BatchMode yes\n\n")

with open(entries_file, "w", encoding="utf-8") as f:
    for entry in entries:
        if entry["label"] in selected_labels:
            f.write(f"{entry['label']}\t{entry['host']}\t{entry['port']}\n")

host_slots = defaultdict(int)
with open(rankfile, "w", encoding="utf-8") as f:
    for rank, (_, entry, _) in enumerate(plan):
        slot = host_slots[entry["label"]]
        host_slots[entry["label"]] += 1
        f.write(f"rank {rank}={entry['label']} slot={slot}\n")

with open(rank_count_file, "w", encoding="utf-8") as f:
    f.write(str(rank_count))
PY

rank_count=$(cat "$rank_count_file")
host_list=$(cat "$host_file")
test -x "$exe" || die "local executable is missing after build: $exe"

while IFS=$'\t' read -r label _host _port; do
  [[ -n "$label" ]] || continue
  if [[ "${label%%.*}" == "${local_label%%.*}" ]]; then
    continue
  fi
  echo "preparing $label:$app_dir"
  ssh -n -F "$ssh_config" "$label" "test -d '$app_dir'" || die "$label:$app_dir is missing"
  remote_rack_dir=$(dirname "$rack_yaml_abs")
  ssh -n -F "$ssh_config" "$label" "mkdir -p '$remote_rack_dir'"
  if ! ssh -n -F "$ssh_config" "$label" test -r "$rack_yaml_abs"; then
    scp -F "$ssh_config" "$rack_yaml_abs" "$label:$rack_yaml_abs"
  fi
  if [[ "$skip_remote_build" != "1" ]]; then
    ssh -n -F "$ssh_config" "$label" \
      "cd '$app_dir' && CUDACXX='$cuda_compiler' cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_COMPILER='$cuda_compiler' -DCMAKE_CUDA_ARCHITECTURES='$cuda_arch' && cmake --build build -j\$(nproc)"
  fi
  ssh -n -F "$ssh_config" "$label" test -x "$remote_exe"
done < "$entries_file"

binary_args=(--rack-config "$rack_yaml_abs")
if [[ -z "$rank_count_arg" ]]; then
  binary_args+=(--rank-count "$rank_count")
fi
binary_args+=("$@")

echo "launching mnnvl_sharp_allreduce rank_count=$rank_count selection=$rank_selection hosts=$host_list"
mpirun -np "$rank_count" \
  --allow-run-as-root \
  --host "$host_list" \
  --rankfile "$rankfile" \
  --bind-to none \
  --mca pml ob1 \
  --mca btl self,vader,tcp \
  --mca btl_tcp_if_include "$mpi_tcp_if" \
  --mca oob_tcp_if_include "$mpi_tcp_if" \
  --mca plm_rsh_args "-F $ssh_config" \
  "$remote_exe" "${binary_args[@]}"
