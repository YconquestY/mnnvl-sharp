#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import statistics
import subprocess
import sys
from typing import Any

import yaml


DEFAULT_RANK_COUNTS = [4, 8, 16, 32, 64, 72]


def merge_body(label: str, body: Any) -> dict[str, Any]:
    if isinstance(body, list):
        merged: dict[str, Any] = {}
        for part in body:
            if not isinstance(part, dict):
                raise ValueError(f"{label}: body list entries must be maps")
            merged.update(part)
        return merged
    if isinstance(body, dict):
        return body
    raise ValueError(f"{label}: body must be a map or list of maps")


def available_rank_count(rack_yaml: pathlib.Path) -> int:
    with rack_yaml.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict) or not isinstance(doc.get("rack"), list):
        raise ValueError("rack YAML must contain top-level sequence 'rack'")

    total = 0
    for item in doc["rack"]:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError("each rack item must be a single-key map")
        label, body = next(iter(item.items()))
        merged = merge_body(str(label), body)
        devices = merged.get("device")
        if not isinstance(devices, list) or not (1 <= len(devices) <= 4):
            raise ValueError(f"{label}: device must be a YAML list with 1 to 4 entries")
        total += len(devices)
    if total > 72:
        raise ValueError("rack YAML describes more than 72 CUDA devices")
    return total


def parse_rank_counts(text: str) -> list[int]:
    out = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("rank counts must be positive")
        out.append(value)
    return out


def extract_json_line(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)
    raise ValueError("launcher output did not contain a JSON summary line")


def run_one(launcher: pathlib.Path,
            rack_yaml: pathlib.Path,
            rank_count: int,
            rank_selection: str,
            sharp_backend: str,
            repeat: int,
            log_path: pathlib.Path) -> dict[str, Any]:
    cmd = [
        str(launcher),
        str(rack_yaml),
        "--rank-count",
        str(rank_count),
        "--rank-selection",
        rank_selection,
        "--sharp-backend",
        sharp_backend,
        "--init-only",
        "--json",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"rank_count={rank_count} repeat={repeat} failed; see {log_path}")
    return extract_json_line(proc.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep MNNVL peer-access initialization latency.")
    parser.add_argument("rack_yaml", type=pathlib.Path)
    parser.add_argument("--rank-counts", default=",".join(str(x) for x in DEFAULT_RANK_COUNTS))
    parser.add_argument("--rank-selection", choices=["balanced", "prefix"], default="balanced")
    parser.add_argument("--sharp-backend", choices=["legacy", "tma_async"], default="legacy")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("results/init_sweep"))
    parser.add_argument("--launcher", type=pathlib.Path, default=pathlib.Path("scripts/run_mnnvl_sharp.sh"))
    args = parser.parse_args()

    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    rack_yaml = args.rack_yaml.resolve()
    launcher = args.launcher.resolve()
    total = available_rank_count(rack_yaml)
    rank_counts = [x for x in parse_rank_counts(args.rank_counts) if x <= total]
    if total >= 4 and total not in rank_counts:
        rank_counts.append(total)
    rank_counts = sorted(set(rank_counts))
    if not rank_counts:
        raise SystemExit(f"no requested rank counts fit rack.yaml total device count {total}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "init_latency.csv"
    rows: list[dict[str, Any]] = []

    for rank_count in rank_counts:
        for repeat in range(args.repeats):
            log_path = args.out_dir / f"rank_{rank_count}_repeat_{repeat}.log"
            summary = run_one(launcher, rack_yaml, rank_count, args.rank_selection,
                              args.sharp_backend, repeat, log_path)
            rows.append({
                "rank_count": rank_count,
                "repeat": repeat,
                "sharp_backend": args.sharp_backend,
                "peer_init_ms": summary["peer_init_ms"],
                "peer_init_min_ms": summary["peer_init_min_ms"],
                "peer_init_median_ms": summary["peer_init_median_ms"],
                "peer_init_p95_ms": summary["peer_init_p95_ms"],
                "peer_init_max_ms": summary["peer_init_max_ms"],
            })

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "rank_count",
            "repeat",
            "sharp_backend",
            "peer_init_ms",
            "peer_init_min_ms",
            "peer_init_median_ms",
            "peer_init_p95_ms",
            "peer_init_max_ms",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plotter = pathlib.Path(__file__).with_name("plot_init_latency.py")
    png_path = args.out_dir / "init_latency.png"
    subprocess.run([sys.executable, str(plotter), str(csv_path), "--output", str(png_path)], check=True)

    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["rank_count"]), []).append(float(row["peer_init_ms"]))
    for rank_count in sorted(grouped):
        print(f"rank_count={rank_count} peer_init_ms_median={statistics.median(grouped[rank_count]):.3f}")
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
