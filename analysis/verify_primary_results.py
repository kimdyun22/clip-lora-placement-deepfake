#!/usr/bin/env python3
"""Recompute the primary mean and sample SD columns from the 12 public seed rows."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from build_primary_seed_results import DATASETS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, default=Path("results/primary_seed_results.csv"))
    parser.add_argument("--aggregate", type=Path, default=Path("results/primary_results.csv"))
    args = parser.parse_args()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.seeds.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["placement"]].append(row)
    with args.aggregate.open(newline="", encoding="utf-8") as handle:
        aggregates = {row["placement"]: row for row in csv.DictReader(handle)}
    if set(grouped) != set(aggregates):
        raise ValueError("placement mismatch between seed and aggregate tables")
    for placement, rows in grouped.items():
        if {int(row["seed"]) for row in rows} != {3407, 7859, 12011} or len(rows) != 3:
            raise ValueError(f"incomplete seeds for {placement}")
        checks = {
            "source_auc_mean": statistics.mean(float(row["source_frame_auc"]) for row in rows),
            "source_auc_sd": statistics.stdev(float(row["source_frame_auc"]) for row in rows),
            "macro_6_auc_mean": statistics.mean(float(row["macro_6_auc"]) for row in rows),
            "macro_6_auc_sd": statistics.stdev(float(row["macro_6_auc"]) for row in rows),
        }
        for dataset in DATASETS:
            values = [float(row[f"{dataset}_analysis_unit_auc"]) for row in rows]
            checks[f"{dataset}_auc_mean"] = statistics.mean(values)
            checks[f"{dataset}_auc_sd"] = statistics.stdev(values)
        for field, expected in checks.items():
            observed = float(aggregates[placement][field])
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=2e-16):
                raise ValueError(f"{placement}/{field}: expected {expected}, observed {observed}")
    print(f"PRIMARY_RESULTS_VERIFIED placements={len(grouped)} seed_rows={sum(map(len, grouped.values()))}")


if __name__ == "__main__":
    main()
