#!/usr/bin/env python3
"""Prevalence-aware calibration for the reporting-only target datasets.

Raw ECE and Brier are prevalence-sensitive, which matters for
DeeperForensics-1.0: its evaluation manifest holds 40,669 real and 201 fake
analysis units, so an aggregate score there is dominated by the real class.
This recomputes calibration per class and as a class-balanced average, with
stratified bootstrap intervals, directly from the released predictions.

No threshold is fitted and no recalibration is applied; this is analysis of the
existing probabilities only.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results" / "predictions" / "analysis_unit_scores.parquet"

DATASETS = ("Celeb-DF-v2", "DFDC", "DFDCP", "UADFV", "DeeperForensics-1.0", "WildDeepfake")
PLACEMENT = {"attn_out": "Attn-Out", "attn_qv": "Attn-Q/V",
             "mlp": "MLP", "mlp_attn_out": "MLP+Attn-Out"}
BINS = 15


def ece(labels: np.ndarray, probs: np.ndarray, bins: int = BINS,
        weights: np.ndarray | None = None) -> float:
    """Raw expected calibration error over equal-width probability bins."""
    if len(labels) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, bins - 1)
    if weights is None:
        weights = np.full(len(labels), 1.0 / len(labels), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
    total = 0.0
    for b in range(bins):
        mask = index == b
        n = int(mask.sum())
        if n == 0:
            continue
        bin_weight = float(weights[mask].sum())
        accuracy = float(np.average(labels[mask], weights=weights[mask]))
        confidence = float(np.average(probs[mask], weights=weights[mask]))
        total += bin_weight * abs(accuracy - confidence)
    return float(total)


def balanced_weights(labels: np.ndarray) -> np.ndarray:
    """Return weights giving real and fake observations total mass 0.5 each."""
    real = labels == 0
    fake = labels == 1
    if not real.any() or not fake.any():
        raise ValueError("balanced calibration requires both classes")
    weights = np.empty(len(labels), dtype=np.float64)
    weights[real] = 0.5 / int(real.sum())
    weights[fake] = 0.5 / int(fake.sum())
    return weights


def brier(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(labels) == 0:
        return float("nan")
    return float(np.mean((probs - labels) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "prevalence_balanced_calibration.csv")
    args = parser.parse_args()

    table = pq.read_table(
        ARCHIVE,
        columns=["dataset", "placement", "seed", "label", "score", "anonymous_analysis_unit_id"],
    )
    col = {name: np.asarray(table[name].to_pylist()) for name in table.column_names}
    seed_col = col["seed"].astype(int)
    seeds = sorted({int(s) for s in col["seed"]})
    rng = np.random.default_rng(args.seed)

    cells = {}
    for placement in PLACEMENT:
        for dataset in DATASETS:
            reference_ids = None
            reference_labels = None
            for seed in seeds:
                mask = ((seed_col == seed) & (col["dataset"] == dataset)
                        & (col["placement"] == placement))
                ids = col["anonymous_analysis_unit_id"][mask]
                labels = col["label"][mask].astype(int)
                scores = col["score"][mask].astype(float)
                if reference_ids is None:
                    reference_ids, reference_labels = ids, labels
                elif not (np.array_equal(reference_ids, ids)
                          and np.array_equal(reference_labels, labels)):
                    raise RuntimeError(
                        f"unit order or labels differ across seeds: {placement}, {dataset}"
                    )
                cells[(placement, dataset, seed)] = (labels, scores)

    rows: list[dict[str, object]] = []
    for placement, name in PLACEMENT.items():
        for dataset in DATASETS:
            per_seed: dict[str, list[float]] = {
                "ece_raw": [], "ece_prevalence_balanced": [],
                "real_absolute_probability_error": [], "fake_absolute_probability_error": [],
                "brier_raw": [], "brier_real": [], "brier_fake": [], "brier_class_balanced": [],
            }
            n_real = n_fake = 0
            for seed in seeds:
                y, p = cells[(placement, dataset, seed)]
                real, fake = y == 0, y == 1
                n_real, n_fake = int(real.sum()), int(fake.sum())
                per_seed["ece_raw"].append(ece(y, p))
                per_seed["ece_prevalence_balanced"].append(
                    ece(y, p, weights=balanced_weights(y)))
                per_seed["real_absolute_probability_error"].append(float(np.mean(p[real])))
                per_seed["fake_absolute_probability_error"].append(float(np.mean(1.0 - p[fake])))
                per_seed["brier_raw"].append(brier(y, p))
                per_seed["brier_real"].append(brier(y[real], p[real]))
                per_seed["brier_fake"].append(brier(y[fake], p[fake]))
                per_seed["brier_class_balanced"].append(
                    0.5 * (brier(y[real], p[real]) + brier(y[fake], p[fake])))

            # Crossed bootstrap: resample seeds and apply one shared,
            # class-stratified unit draw to each selected seed.
            boot = {"ece_prevalence_balanced": [], "brier_class_balanced": []}
            for _ in range(args.replicates):
                picks = {"ece_prevalence_balanced": [], "brier_class_balanced": []}
                chosen_seeds = rng.choice(seeds, size=len(seeds), replace=True)
                reference_y, _ = cells[(placement, dataset, seeds[0])]
                real_reference = np.flatnonzero(reference_y == 0)
                fake_reference = np.flatnonzero(reference_y == 1)
                real_draw = rng.choice(real_reference, size=len(real_reference), replace=True)
                fake_draw = rng.choice(fake_reference, size=len(fake_reference), replace=True)
                take = np.concatenate([real_draw, fake_draw])
                for seed in chosen_seeds:
                    y, p = cells[(placement, dataset, int(seed))]
                    sampled_y, sampled_p = y[take], p[take]
                    sampled_real, sampled_fake = sampled_y == 0, sampled_y == 1
                    picks["ece_prevalence_balanced"].append(
                        ece(sampled_y, sampled_p, weights=balanced_weights(sampled_y)))
                    picks["brier_class_balanced"].append(
                        0.5 * (brier(sampled_y[sampled_real], sampled_p[sampled_real])
                               + brier(sampled_y[sampled_fake], sampled_p[sampled_fake])))
                boot["ece_prevalence_balanced"].append(
                    np.mean(picks["ece_prevalence_balanced"]))
                boot["brier_class_balanced"].append(np.mean(picks["brier_class_balanced"]))

            row: dict[str, object] = {
                "placement": name, "dataset": dataset,
                "real_units": n_real, "fake_units": n_fake,
                "fake_prevalence": repr(n_fake / (n_real + n_fake)),
                "n_seeds": len(seeds),
            }
            for key, values in per_seed.items():
                row[f"{key}_mean"] = repr(float(np.mean(values)))
                row[f"{key}_sd"] = repr(float(np.std(values, ddof=1)))
            for key, values in boot.items():
                low, high = np.percentile(values, [2.5, 97.5])
                row[f"{key}_ci_low"] = repr(float(low))
                row[f"{key}_ci_high"] = repr(float(high))
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'placement':14s} {'dataset':22s} {'fake%':>7s} {'ECE raw':>9s} {'ECE bal':>9s}"
          f" {'real APE':>9s} {'fake APE':>9s}")
    for row in rows:
        print(f"{row['placement']:14s} {row['dataset']:22s}"
              f" {float(row['fake_prevalence']) * 100:6.2f}%"
              f" {float(row['ece_raw_mean']):9.4f}"
              f" {float(row['ece_prevalence_balanced_mean']):9.4f}"
              f" {float(row['real_absolute_probability_error_mean']):9.4f}"
              f" {float(row['fake_absolute_probability_error_mean']):9.4f}")
    print(f"\nwrote {args.output}")
    print("BALANCED_CALIBRATION_DONE")


if __name__ == "__main__":
    main()
