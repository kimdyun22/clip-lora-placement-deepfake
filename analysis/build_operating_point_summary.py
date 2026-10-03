#!/usr/bin/env python3
"""Build manuscript-ready operating-point summaries from public seed rows."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


CANDIDATES = {
    "std_attn_out_r30_alpha120_p025": "Attn-Out",
    "std_attn_qv_r15_alpha60_p025": "Attn-Q/V",
    "std_mlp_r6_alpha24_p025": "MLP",
    "std_mlp_attn_out_r5_alpha20_p025": "MLP+Attn-Out",
}
THRESHOLD_METRICS = {
    "unit_f1_source_fixed": ("analysis_unit", "f1_source_fixed", "source_validation"),
    "unit_eer_descriptive": ("analysis_unit", "target_eer_descriptive", "target_descriptive"),
    "frame_tpr_source_fpr_0.01": ("frame", "tpr_at_source_fpr_0.01", "source_validation"),
    "frame_tpr_source_fpr_0.001": ("frame", "tpr_at_source_fpr_0.001", "source_validation"),
}
FIELDS = [
    "placement", "candidate", "dataset", "analysis_level", "metric", "mean",
    "sample_sd", "n_seeds", "threshold_origin", "threshold_mean",
    "threshold_sample_sd", "source_constraint_mean",
]


def summary_rows(raw_path: Path) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    with raw_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "target":
                continue
            if row["candidate"] not in CANDIDATES:
                raise ValueError(f"unknown candidate: {row['candidate']}")
            if row["record_type"] == "calibration":
                for metric, column in (("ece_15", "raw_ece_15"), ("brier", "brier")):
                    item = dict(row)
                    item["value"] = row[column]
                    item["analysis_level"] = "analysis_unit"
                    item["metric_name"] = metric
                    item["threshold_origin"] = "none"
                    grouped[(row["candidate"], row["dataset"], "analysis_unit", metric)].append(item)
            elif row["record_type"] == "threshold":
                level, metric, origin = THRESHOLD_METRICS[row["metric"]]
                item = dict(row)
                item["analysis_level"] = level
                item["metric_name"] = metric
                item["threshold_origin"] = origin
                grouped[(row["candidate"], row["dataset"], level, metric)].append(item)
            else:
                raise ValueError(f"unknown record type: {row['record_type']}")

    output: list[dict[str, object]] = []
    for (candidate, dataset, level, metric), rows in sorted(grouped.items()):
        seeds = {int(row["seed"]) for row in rows}
        if seeds != {3407, 7859, 12011} or len(rows) != 3:
            raise ValueError(f"{candidate}/{dataset}/{metric}: incomplete seeds {sorted(seeds)}")
        values = [float(row["value"]) for row in rows]
        thresholds = [float(row["threshold"]) for row in rows if row["threshold"]]
        constraints = [float(row["source_constraint_value"]) for row in rows if row["source_constraint_value"]]
        output.append({
            "placement": CANDIDATES[candidate],
            "candidate": candidate,
            "dataset": dataset,
            "analysis_level": level,
            "metric": metric,
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
            "n_seeds": len(seeds),
            "threshold_origin": rows[0]["threshold_origin"],
            "threshold_mean": statistics.mean(thresholds) if thresholds else "",
            "threshold_sample_sd": statistics.stdev(thresholds) if len(thresholds) > 1 else "",
            "source_constraint_mean": statistics.mean(constraints) if constraints else "",
        })
    if len(output) != 4 * 6 * 6:
        raise ValueError(f"expected 144 summary rows, observed {len(output)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("results/calibration_thresholds.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/operating_point_summary.csv"))
    args = parser.parse_args()
    rows = summary_rows(args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"OPERATING_POINT_SUMMARY_WRITTEN rows={len(rows)} path={args.output}")


if __name__ == "__main__":
    main()
