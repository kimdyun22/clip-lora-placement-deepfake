#!/usr/bin/env python3
"""Recompute source-threshold error overlap and verify Figure 2 coordinates."""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BITS = {"attn_out": 1, "attn_qv": 2, "mlp": 4, "mlp_attn_out": 8}
NAMES = dict(zip(BITS, ["Attn-Out", "Attn-Q/V", "MLP", "MLP+Out"]))
DATASETS = {"Celeb-DF-v2", "DFDC", "DFDCP", "UADFV", "DeeperForensics-1.0", "WildDeepfake"}
SEEDS = {3407, 7859, 12011}
TOLERANCE = 1e-9


def rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def close(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=0, abs_tol=TOLERANCE)


def indexed(name: str, fields: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    output = {}
    for row in rows(name):
        key = tuple(row[f] for f in fields)
        require(key not in output, f"Duplicate key in {name}: {key}")
        output[key] = row
    return output


def main() -> None:
    thresholds = {}
    f1_expected = {}
    for row in rows("calibration_thresholds.csv"):
        if row["record_type"] != "threshold" or row["metric"] != "unit_f1_source_fixed":
            continue
        key = (row["candidate"], int(row["seed"]))
        threshold = float(row["threshold"])
        require(math.isfinite(threshold), f"Nonfinite threshold: {key}")
        require(key not in thresholds or thresholds[key] == threshold, f"Inconsistent source threshold: {key}")
        thresholds[key] = threshold
        fkey = (*key, row["dataset"])
        require(fkey not in f1_expected, f"Duplicate F1 row: {fkey}")
        f1_expected[fkey] = float(row["value"])
    require(len(thresholds) == 12 and len(f1_expected) == 72, "Incomplete threshold/F1 coverage")

    units = {}
    f1_counts = defaultdict(Counter)
    model_ids = {}
    archive = pq.ParquetFile(RESULTS / "predictions/analysis_unit_scores.parquet")
    for batch in archive.iter_batches(batch_size=32768):
        for row in batch.to_pylist():
            dataset, seed = row["dataset"], int(row["seed"])
            placement, candidate = row["placement"], row["candidate"]
            require(dataset in DATASETS and seed in SEEDS and placement in BITS, "Unexpected prediction key")
            require(row["label"] in (0, 1) and math.isfinite(row["score"]), "Invalid prediction label/score")
            require(placement not in model_ids or model_ids[placement] == candidate, "Candidate changes within placement")
            model_ids[placement] = candidate
            threshold = thresholds[(candidate, seed)]
            prediction = int(row["score"] >= threshold)
            key = (dataset, seed, row["anonymous_analysis_unit_id"])
            label, seen, errors = units.get(key, (row["label"], 0, 0))
            bit = BITS[placement]
            require(label == row["label"] and not seen & bit, "Unpaired label or duplicate unit prediction")
            units[key] = (label, seen | bit, errors | (bit if prediction != label else 0))
            f1_counts[(candidate, seed, dataset)][(row["label"], prediction)] += 1
    require(set(f1_counts) == set(f1_expected), "Incomplete dataset/model/seed predictions")
    for key, count in f1_counts.items():
        tp, fp, fn = count[(1, 1)], count[(0, 1)], count[(1, 0)]
        denominator = 2 * tp + fp + fn
        value = 2 * tp / denominator if denominator else 0.0
        require(close(value, f1_expected[key]), f"Source-fixed F1 mismatch: {key}")

    partition = defaultdict(Counter)
    real_fake = defaultdict(Counter)
    for (dataset, seed, _), (label, seen, errors) in units.items():
        require(seen == 15, "Incomplete four-placement unit coverage")
        partition[(dataset, seed)][errors.bit_count()] += 1
        real_fake[(dataset, seed)][label] += 1
    seed_reference = indexed("error_overlap_seed_results.csv", ("dataset", "seed"))
    require(set(seed_reference) == {(d, str(s)) for d in DATASETS for s in SEEDS}, "Incomplete overlap seed rows")
    summary_values = defaultdict(list)
    for (dataset, seed), count in partition.items():
        row = seed_reference[(dataset, str(seed))]
        total = sum(count.values())
        any_error = total - count[0]
        require(any_error > 0, "Conditional overlap is undefined without errors")
        require(int(row["analysis_units"]) == total and int(row["any_error_units"]) == any_error, "Unit-count mismatch")
        require(int(row["real_units"]) == real_fake[(dataset, seed)][0] and int(row["fake_units"]) == real_fake[(dataset, seed)][1], "Class-count mismatch")
        values = {"any_error_units": any_error, "any_error_rate_pct": 100 * any_error / total}
        for k in range(5):
            require(int(row[f"wrong_{k}_units"]) == count[k], f"Error-partition mismatch: {dataset}/{seed}/{k}")
            if k:
                values[f"wrong_{k}_conditional_pct"] = 100 * count[k] / any_error
        for metric, value in values.items():
            require(close(value, float(row[metric])), f"Overlap metric mismatch: {dataset}/{seed}/{metric}")
            summary_values[(dataset, metric)].append(value)
    summary = indexed("error_overlap_summary.csv", ("dataset",))
    require(set(summary) == {(d,) for d in DATASETS}, "Incomplete overlap summaries")
    for dataset in DATASETS:
        row = summary[(dataset,)]
        counts = {int(seed_reference[(dataset, str(s))]["analysis_units"]) for s in SEEDS}
        require(len(counts) == 1 and int(row["analysis_units_per_seed"]) in counts and int(row["training_seeds"]) == 3, "Summary denominator mismatch")
        for (d, metric), values in summary_values.items():
            if d == dataset:
                require(close(statistics.mean(values), float(row[metric + "_mean"])) and close(statistics.stdev(values), float(row[metric + "_sd"])), f"Summary mean/SD mismatch: {dataset}/{metric}")

    threshold_reference = indexed("error_overlap_source_thresholds.csv", ("placement", "seed"))
    require(set(threshold_reference) == {(p, str(s)) for p in BITS for s in SEEDS}, "Incomplete released thresholds")
    for (placement, seed), row in threshold_reference.items():
        require(row["candidate"] == model_ids[placement], "Released threshold candidate mismatch")
        require(close(float(row["source_fixed_threshold"]), thresholds[(model_ids[placement], int(seed))]), "Released threshold mismatch")

    efficiency = indexed("efficiency.csv", ("placement",))
    primary = indexed("primary_results.csv", ("position",))
    figure2 = indexed("figure2_training_resources.csv", ("placement",))
    require(set(figure2) == {(p,) for p in BITS}, "Incomplete Figure 2 coordinates")
    for (placement,), row in figure2.items():
        require(row["label"] == NAMES[placement], "Figure 2 placement label mismatch")
        require((int(row["training_seeds"]), int(row["hardware_rounds"]), int(row["batch_size"])) == (3, 5, 32), "Figure 2 measurement setting mismatch")
        for field in ["training_step_ms_mean", "training_step_ms_sd", "training_peak_allocated_gib_mean", "training_peak_allocated_gib_sd"]:
            require(close(float(row[field]), float(efficiency[(placement,)][field])), f"Figure 2 resource mismatch: {placement}/{field}")
        for field in ["macro_6_auc_mean", "macro_6_auc_sd"]:
            require(close(float(row[field]), float(primary[(placement,)][field])), f"Figure 2 AUC mismatch: {placement}/{field}")
    print("ERROR_OVERLAP_VERIFIED dataset_seed_rows=18 summaries=6 thresholds=12 f1_cells=72 figure2_rows=4")


if __name__ == "__main__":
    main()
