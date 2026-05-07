#!/usr/bin/env python3
import argparse
import csv
import pathlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot MNNVL peer initialization latency.")
    parser.add_argument("csv_path", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    grouped: dict[int, list[float]] = {}
    with args.csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped.setdefault(int(row["rank_count"]), []).append(float(row["peer_init_ms"]))

    if not grouped:
        raise SystemExit("CSV did not contain any rows")

    rank_counts = sorted(grouped)
    medians = [statistics.median(grouped[x]) for x in rank_counts]
    lows = [medians[i] - min(grouped[x]) for i, x in enumerate(rank_counts)]
    highs = [max(grouped[x]) - medians[i] for i, x in enumerate(rank_counts)]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(rank_counts, medians, yerr=[lows, highs], marker="o", capsize=4, linewidth=1.8)
    ax.set_xlabel("rank count")
    ax.set_ylabel("peer access initialization latency (ms)")
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_xticks(rank_counts)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
