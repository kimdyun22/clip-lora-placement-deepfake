#!/usr/bin/env python3
"""Check public grouping-sensitivity summaries against their seed tables.

This is a public cross-file consistency check, not a recomputation of AUC from
private source/target frame predictions. Use the builder for that operation.
"""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

from build_analysis_unit_sensitivity import PRIMARY, CANDIDATES, SEEDS

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / "results" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12)


def keyed(rows, fields):
    result = {tuple(row[f] for f in fields): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate keys: {fields}")
    return result


def main():
    seeds = read("source_per_video_auc.csv")
    summaries = keyed(read("source_per_video_auc_summary.csv"), ("config_id",))
    old_source = keyed(read("source_unit_level_auc.csv"), ("config_id",))
    configs = {**PRIMARY, **CANDIDATES}
    expected = {(config, str(seed)) for config in configs for seed in SEEDS}
    if set(keyed(seeds, ("config_id", "seed"))) != expected or set(summaries) != {(c,) for c in configs}:
        raise ValueError("source sensitivity must contain exactly eight configurations by three seeds")
    for config in configs:
        rows = [r for r in seeds if r["config_id"] == config]
        summary, original = summaries[(config,)], old_source[(config,)]
        expected_cohort = "exact_match" if config in PRIMARY else "source_selection"
        if int(summary["n_seeds"]) != 3 or summary["cohort"] != expected_cohort:
            raise ValueError(f"incorrect source cohort/seed count: {config}")
        for row in rows:
            if row["cohort"] != expected_cohort or row["configuration"] != configs[config]:
                raise ValueError(f"incorrect configuration label: {config}")
            counts = tuple(int(row[k]) for k in ("frames", "id_groups", "individual_videos", "real_videos", "fake_videos"))
            if counts != (22354, 280, 700, 140, 560):
                raise ValueError(f"incorrect source populations: {config}")
            for metric in ("source_frame_auc", "source_id_group_auc", "source_individual_video_auc"):
                value = float(row[metric])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"invalid AUC: {config} {metric}")
        for metric in ("source_frame_auc", "source_id_group_auc", "source_individual_video_auc"):
            values = [float(r[metric]) for r in rows]
            for suffix, value in (("_mean", statistics.mean(values)), ("_sd", statistics.stdev(values))):
                if not close(value, summary[metric + suffix]):
                    raise ValueError(f"source summary mismatch: {config} {metric + suffix}")
        for new, old in (("source_frame_auc", "source_frame_auc"), ("source_id_group_auc", "source_unit_auc")):
            for suffix in ("_mean", "_sd"):
                if not close(summary[new + suffix], original[old + suffix]):
                    raise ValueError(f"source primary metric changed: {config} {new + suffix}")

    targets = keyed(read("primary_seed_results.csv"), ("candidate", "seed"))
    dfdcp = read("dfdcp_video_sensitivity.csv")
    if set(keyed(dfdcp, ("candidate", "seed"))) != {(c, str(s)) for c in PRIMARY for s in SEEDS}:
        raise ValueError("DFDCP sensitivity must cover only the 12 primary checkpoints")
    for row in dfdcp:
        if row["placement"] != PRIMARY[row["candidate"]]:
            raise ValueError("incorrect DFDCP placement label")
        if tuple(int(row[k]) for k in ("frames", "primary_id_groups", "individual_videos")) != (17222, 652, 654):
            raise ValueError("incorrect DFDCP populations")
        primary = float(row["primary_id_group_auc"])
        individual = float(row["individual_video_auc"])
        if not (math.isfinite(primary) and math.isfinite(individual) and 0 <= primary <= 1 and 0 <= individual <= 1):
            raise ValueError("invalid DFDCP AUC")
        if not close(primary, targets[(row["candidate"], row["seed"])]["DFDCP_analysis_unit_auc"]):
            raise ValueError("primary DFDCP AUC changed")
        if not close(individual - primary, row["auc_change"]):
            raise ValueError("DFDCP AUC difference mismatch")
    maximum = max(abs(float(r["auc_change"])) for r in dfdcp)
    if maximum > 0.00023:
        raise ValueError(f"DFDCP sensitivity exceeds reported bound: {maximum}")
    print(f"ANALYSIS_UNIT_SENSITIVITY_VERIFIED source_seed_rows={len(seeds)} source_summaries={len(summaries)} "
          f"dfdcp_checkpoints={len(dfdcp)} max_absolute_auc_change={maximum:.15g}")


if __name__ == "__main__":
    main()
