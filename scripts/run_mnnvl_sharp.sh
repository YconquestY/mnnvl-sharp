#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage:
  scripts/run_mnnvl_sharp.sh rack.yaml [mnnvl_sharp_allreduce args...]

The rack YAML order defines MPI host order. The script builds the sample if
needed, writes a temporary SSH config from rack.yaml, verifies that the rack YAML
and executable path exist inside every selected container, and launches 4 ranks.
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
entries_file="$tmpdir/entries.tsv"
host_file="$tmpdir/hosts.txt"
ssh_config="$tmpdir/ssh_config"

python3 - "$rack_yaml_abs" "$host_file" "$ssh_config" "$entries_file" \
  "$ssh_identity" "$ssh_known_hosts" "$ssh_user" <<'PY'
import sys
import yaml

rack_yaml, host_file, ssh_config, entries_file = sys.argv[1:5]
ssh_identity, ssh_known_hosts, ssh_user = sys.argv[5:8]
with open(rack_yaml, "r", encoding="utf-8") as f:
    doc = yaml.safe_load(f)
if not isinstance(doc, dict) or not isinstance(doc.get("rack"), list):
    raise SystemExit("rack YAML must contain top-level sequence 'rack'")

entries = []
for item in doc["rack"]:
    if not isinstance(item, dict) or len(item) != 1:
        raise SystemExit("each rack item must be a single-key map")
    label, body = next(iter(item.items()))
    merged = {}
    if isinstance(body, list):
        for part in body:
            if not isinstance(part, dict):
                raise SystemExit(f"{label}: body list entries must be maps")
            merged.update(part)
    elif isinstance(body, dict):
        merged = body
    else:
        raise SystemExit(f"{label}: body must be a map or list of maps")
    host = merged.get("hostname")
    port = int(merged.get("port", 4399))
    devices = merged.get("device")
    if not isinstance(host, str) or not isinstance(devices, list) or len(devices) != 2:
        raise SystemExit(f"{label}: expected hostname string and exactly two devices")
    entries.append((label, host, port))

if len(entries) != 2:
    raise SystemExit("this v1 sample expects exactly two rack entries")

with open(host_file, "w", encoding="utf-8") as f:
    f.write(",".join(f"{label}:2" for label, _, _ in entries))

with open(ssh_config, "w", encoding="utf-8") as f:
    for label, host, port in entries:
        f.write(f"Host {label}\n")
        f.write(f"  HostName {host}\n")
        f.write(f"  User {ssh_user}\n")
        f.write(f"  Port {port}\n")
        f.write(f"  IdentityFile {ssh_identity}\n")
        f.write("  IdentitiesOnly yes\n")
        f.write("  PasswordAuthentication no\n")
        f.write(f"  UserKnownHostsFile {ssh_known_hosts}\n")
        f.write("  StrictHostKeyChecking accept-new\n")
        f.write("  BatchMode yes\n\n")

with open(entries_file, "w", encoding="utf-8") as f:
    for label, host, port in entries:
        f.write(f"{label}\t{host}\t{port}\n")
PY

host_list=$(cat "$host_file")

test -x "$exe" || die "local executable is missing after build: $exe"

while IFS=$'\t' read -r label _host _port; do
  if [[ "${label%%.*}" == "${local_label%%.*}" ]]; then
    continue
  fi
  echo "preparing $label:$app_dir"
  if ! ssh -n -F "$ssh_config" "$label" test -r "$rack_yaml_abs"; then
    scp -F "$ssh_config" "$rack_yaml_abs" "$label:$rack_yaml_abs"
  fi
  ssh -n -F "$ssh_config" "$label" \
    "cd '$app_dir' && CUDACXX='$cuda_compiler' cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_COMPILER='$cuda_compiler' -DCMAKE_CUDA_ARCHITECTURES='$cuda_arch' && cmake --build build -j\$(nproc)"
  ssh -n -F "$ssh_config" "$label" test -x "$remote_exe"
done < "$entries_file"

echo "launching mnnvl_sharp_allreduce on $host_list"
mpirun -np 4 \
  --allow-run-as-root \
  --host "$host_list" \
  --map-by ppr:2:node \
  --mca pml ob1 \
  --mca btl self,vader,tcp \
  --mca btl_tcp_if_include "$mpi_tcp_if" \
  --mca oob_tcp_if_include "$mpi_tcp_if" \
  --mca plm_rsh_args "-F $ssh_config" \
  "$remote_exe" --rack-config "$rack_yaml_abs" "$@"
