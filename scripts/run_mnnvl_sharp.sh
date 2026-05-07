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
build_dir="$repo_root/build"
exe="$build_dir/mnnvl_sharp_allreduce"

command -v python3 >/dev/null 2>&1 || die "python3 is required"
python3 - <<'PY' >/dev/null 2>&1 || die "PyYAML is required"
import yaml
PY
command -v cmake >/dev/null 2>&1 || die "cmake is required"
command -v mpirun >/dev/null 2>&1 || die "mpirun is required"
command -v ssh >/dev/null 2>&1 || die "ssh is required"

mkdir -p "$build_dir"
if [[ ! -x "$exe" ]]; then
  cmake -S "$repo_root" -B "$build_dir" -DCMAKE_BUILD_TYPE=RelWithDebInfo
  cmake --build "$build_dir" -j"$(nproc)"
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
host_file="$tmpdir/hosts.txt"
ssh_config="$tmpdir/ssh_config"

python3 - "$rack_yaml_abs" "$host_file" "$ssh_config" <<'PY'
import sys
import yaml

rack_yaml, host_file, ssh_config = sys.argv[1:4]
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
        f.write(f"  Port {port}\n")
        f.write("  StrictHostKeyChecking accept-new\n")
        f.write("  BatchMode yes\n\n")
PY

host_list=$(cat "$host_file")

python3 - "$rack_yaml_abs" "$exe" <<'PY'
import subprocess
import sys
import yaml

rack_yaml, exe = sys.argv[1:3]
with open(rack_yaml, "r", encoding="utf-8") as f:
    doc = yaml.safe_load(f)
for item in doc["rack"]:
    label, body = next(iter(item.items()))
    merged = {}
    if isinstance(body, list):
        for part in body:
            merged.update(part)
    else:
        merged = body
    host = merged["hostname"]
    port = str(merged.get("port", 4399))
    for path, test_flag in [(rack_yaml, "-r"), (exe, "-x")]:
        cmd = ["ssh", "-p", port, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
               host, "test", test_flag, path]
        subprocess.check_call(cmd)
PY

echo "launching mnnvl_sharp_allreduce on $host_list"
mpirun -np 4 \
  --host "$host_list" \
  --map-by ppr:2:node \
  --mca plm_rsh_args "-F $ssh_config" \
  "$exe" --rack-config "$rack_yaml_abs" "$@"
