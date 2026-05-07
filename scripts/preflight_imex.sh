#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage:
  scripts/preflight_imex.sh
  scripts/preflight_imex.sh rack.yaml

Local mode checks the current container/node. Rack mode parses rack.yaml, checks
each selected container over SSH, and compares IMEX node membership.
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

normalize_nodes_config() {
  sed -e 's/#.*$//' -e '/^[[:space:]]*$/d' "$1" | awk '{$1=$1; print}' | sort
}

run_local_checks() {
  echo "== local host =="
  hostname

  echo "== gpu inventory =="
  nvidia-smi -L

  echo "== gpu topology =="
  nvidia-smi topo -m

  echo "== nvidia-imex service =="
  systemctl is-active nvidia-imex

  echo "== imex channels =="
  ls -la /dev/nvidia-caps-imex-channels
  test -e /dev/nvidia-caps-imex-channels/channel0

  echo "== nodes_config.cfg =="
  cat /etc/nvidia-imex/nodes_config.cfg

  echo "== nvidia-imex-ctl -N =="
  nvidia-imex-ctl -N

  cat <<'NOTE'
== nvlink partition reminder ==
Compute-node checks cannot prove NVSwitch partition membership. Ask the fabric
operator to run `nv show sdn partition` on the leader NVSwitch/NMX endpoint.
For the default GB200 NVL72 setup, expect one healthy "Default Partition" with
72 GPUs. If user partitions exist, both selected trays must be in the same one.
NOTE
}

parse_rack() {
  local rack_yaml=$1
  python3 - "$rack_yaml" <<'PY'
import sys
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    doc = yaml.safe_load(f)
if not isinstance(doc, dict) or not isinstance(doc.get("rack"), list):
    raise SystemExit("rack YAML must contain top-level sequence 'rack'")
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
    if not isinstance(host, str):
        raise SystemExit(f"{label}: hostname must be a string")
    if not isinstance(devices, list):
        raise SystemExit(f"{label}: device must be a YAML list")
    print(f"{label}\t{host}\t{port}\t{','.join(str(x) for x in devices)}")
PY
}

remote() {
  local port=$1
  local host=$2
  shift 2
  ssh -p "$port" -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$host" "$@"
}

run_rack_checks() {
  local rack_yaml=$1
  [[ -r "$rack_yaml" ]] || die "cannot read $rack_yaml"
  have python3 || die "python3 is required"
  python3 - <<'PY' >/dev/null 2>&1 || die "PyYAML is required"
import yaml
PY
  have ssh || die "ssh is required"

  local tmpdir
  tmpdir=$(mktemp -d)
  trap 'rm -rf "$tmpdir"' EXIT

  local idx=0
  local first_nodes=""
  while IFS=$'\t' read -r label host port devices; do
    echo "== rack entry $label =="
    echo "control_host: $host"
    echo "ssh_port: $port"
    echo "devices: $devices"

    local remote_hostname
    remote_hostname=$(remote "$port" "$host" hostname)
    echo "hostname: $remote_hostname"
    if [[ "${remote_hostname%%.*}" != "${label%%.*}" ]]; then
      echo "warning: hostname does not match rack key; launch with MNNVL_HOST_LABEL if intentional" >&2
    fi

    echo "-- nvidia-imex service"
    remote "$port" "$host" systemctl is-active nvidia-imex

    echo "-- imex channels"
    remote "$port" "$host" ls -la /dev/nvidia-caps-imex-channels
    remote "$port" "$host" test -e /dev/nvidia-caps-imex-channels/channel0

    echo "-- gpu inventory"
    remote "$port" "$host" nvidia-smi -L

    echo "-- imex health"
    remote "$port" "$host" nvidia-imex-ctl -N

    local nodes_file="$tmpdir/nodes_$idx.cfg"
    remote "$port" "$host" cat /etc/nvidia-imex/nodes_config.cfg >"$nodes_file"
    echo "-- normalized nodes_config.cfg"
    normalize_nodes_config "$nodes_file"
    if [[ -z "$first_nodes" ]]; then
      first_nodes="$nodes_file"
    else
      diff -u <(normalize_nodes_config "$first_nodes") <(normalize_nodes_config "$nodes_file")
    fi
    idx=$((idx + 1))
  done < <(parse_rack "$rack_yaml")

  [[ "$idx" -eq 2 ]] || die "this v1 sample expects exactly two rack entries"

  cat <<'NOTE'
== remaining checks ==
GPU-to-NUMA uniqueness for the selected pairs is checked inside the binary with
CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID.

Compute-node checks cannot prove NVSwitch partition membership. Ask the fabric
operator to run `nv show sdn partition`; the default expected state is one
healthy "Default Partition" with 72 GPUs. If user partitions exist, both
selected trays must be in the same one.
NOTE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -eq 0 ]]; then
  run_local_checks
elif [[ $# -eq 1 ]]; then
  run_rack_checks "$1"
else
  usage
  exit 1
fi
