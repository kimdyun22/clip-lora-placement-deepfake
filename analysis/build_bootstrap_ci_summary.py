#!/usr/bin/env python3
"""Summarize existing seed-conditional paired bootstrap intervals without pooling."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


FIELDS = [
    "pair_id", "first", "second", "dataset", "n_seeds", "mean_auc_delta",
    "sample_sd_auc_delta", "ci_excludes_zero_first_better", "ci_excludes_zero_second_better",
    "ci_includes_zero", "holm_primary_reject_first_better",
    "holm_primary_reject_second_better", "holm_global_reject_first_better",
    "holm_global_reject_second_better",
]


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def summary_rows(statistics_path: Path) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with statistics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["pair_id"], row["dataset"])].append(row)
    output: list[dict[str, object]] = []
    for (pair_id, dataset), rows in sorted(grouped.items()):
        seeds = {int(row["seed"]) for row in rows}
        if seeds != {3407, 7859, 12011} or len(rows) != 3:
            raise ValueError(f"incomplete seed family: {pair_id}/{dataset}")
        deltas = [float(row["delta_first_minus_second"]) for row in rows]
        first_ci = [row for row in rows if float(row["bootstrap_delta_ci_low"]) > 0.0]
        second_ci = [row for row in rows if float(row["bootstrap_delta_ci_high"]) < 0.0]
        primary_first = [row for row in rows if as_bool(row["holm_primary_reject"]) and float(row["delta_first_minus_second"]) > 0.0]
        primary_second = [row for row in rows if as_bool(row["holm_primary_reject"]) and float(row["delta_first_minus_second"]) < 0.0]
        global_first = [row for row in rows if as_bool(row["holm_global_108_reject"]) and float(row["delta_first_minus_second"]) > 0.0]
        global_second = [row for row in rows if as_bool(row["holm_global_108_reject"]) and float(row["delta_first_minus_second"]) < 0.0]
        output.append({
            "pair_id": pair_id,
            "first": rows[0]["first"],
            "second": rows[0]["second"],
            "dataset": dataset,
            "n_seeds": len(seeds),
            "mean_auc_delta": statistics.mean(deltas),
            "sample_sd_auc_delta": statistics.stdev(deltas),
            "ci_excludes_zero_first_better": len(first_ci),
            "ci_excludes_zero_second_better": len(second_ci),
            "ci_includes_zero": len(rows) - len(first_ci) - len(second_ci),
            "holm_primary_reject_first_better": len(primary_first),
            "holm_primary_reject_second_better": len(primary_second),
            "holm_global_reject_first_better": len(global_first),
            "holm_global_reject_second_better": len(global_second),
        })
    if len(output) != 36:
        raise ValueError(f"expected 36 pair-by-dataset summaries, observed {len(output)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, default=Path("results/statistics.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/bootstrap_ci_summary.csv"))
    args = parser.parse_args()
    rows = summary_rows(args.statistics)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"BOOTSTRAP_CI_SUMMARY_WRITTEN rows={len(rows)} path={args.output}")


if __name__ == "__main__":
    main()
