#!/usr/bin/env python3
"""Render training-resource and source-threshold error-overlap figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PLACEMENTS = ["attn_out", "attn_qv", "mlp", "mlp_attn_out"]
NAMES = ["Attn-Out", "Attn-Q/V", "MLP", "MLP+Out"]
DATASETS = ["Celeb-DF-v2", "DFDC", "DFDCP", "UADFV", "DeeperForensics-1.0", "WildDeepfake"]
SHORT_DATASETS = ["CDF-v2", "DFDC", "DFDCP", "UADFV", "DF-1.0", "WDF"]
SEEDS = {3407, 7859, 12011}
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]


def save(fig: plt.Figure, directory: Path, name: str) -> None:
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def training_figure(results: Path, figures: Path) -> None:
    efficiency = pd.read_csv(results / "efficiency.csv").set_index("placement").loc[PLACEMENTS]
    primary = pd.read_csv(results / "primary_results.csv").set_index("position").loc[PLACEMENTS]
    if not efficiency["trainable_params"].eq(1478658).all():
        raise ValueError("Training-resource figure requires the exact-budget configurations")
    if not efficiency["rounds"].eq(5).all():
        raise ValueError("Expected five hardware rounds")
    if not primary["seeds"].eq(3).all():
        raise ValueError("Expected three training seeds")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharey=True)
    specs = [
        ("training_step_ms", "Model-only training step time (ms)", "(a) Training step time"),
        ("training_peak_allocated_gib", "Peak allocated training memory (GiB)", "(b) Training memory"),
    ]
    rows = []
    for ax, (metric, xlabel, title) in zip(axes, specs):
        for i, placement in enumerate(PLACEMENTS):
            x = float(efficiency.loc[placement, metric + "_mean"])
            xs = float(efficiency.loc[placement, metric + "_sd"])
            y = float(primary.loc[placement, "macro_6_auc_mean"])
            ys = float(primary.loc[placement, "macro_6_auc_sd"])
            ax.errorbar(x, y, xerr=xs, yerr=ys, fmt="o", markersize=6.5,
                        color=COLORS[i], ecolor=COLORS[i], capsize=3, linewidth=1.2)
            offset = [(8, -4), (8, 5), (8, -4), (-8, 8)][i]
            ax.annotate(NAMES[i], (x, y), xytext=offset, textcoords="offset points",
                        ha="right" if i == 3 else "left", va="bottom", fontsize=9)
            if ax is axes[0]:
                rows.append({
                    "placement": placement, "label": NAMES[i], "training_seeds": 3,
                    "hardware_rounds": 5, "batch_size": 32,
                    "training_step_ms_mean": x, "training_step_ms_sd": xs,
                    "training_peak_allocated_gib_mean": float(efficiency.loc[placement, "training_peak_allocated_gib_mean"]),
                    "training_peak_allocated_gib_sd": float(efficiency.loc[placement, "training_peak_allocated_gib_sd"]),
                    "macro_6_auc_mean": y, "macro_6_auc_sd": ys,
                })
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
        ax.set_ylim(0.9355, 0.9455)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_xlim(417, 505)
    axes[1].set_xlim(12.9, 18.45)
    axes[0].set_ylabel("Target Macro-6 analysis-unit AUC")
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.18, top=0.88, wspace=0.12)
    pd.DataFrame(rows).to_csv(results / "figure2_training_resources.csv", index=False, float_format="%.17g")
    save(fig, figures, "figure2")


def overlap_data(results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pq.read_table(results / "predictions/analysis_unit_scores.parquet").to_pandas()
    if set(scores["placement"]) != set(PLACEMENTS) or set(scores["seed"]) != SEEDS:
        raise ValueError("Incomplete placement or seed coverage")
    if set(scores["dataset"]) != set(DATASETS):
        raise ValueError("Incomplete target-dataset coverage")
    raw = pd.read_csv(results / "calibration_thresholds.csv")
    threshold_rows = raw[(raw.record_type == "threshold") & (raw.metric == "unit_f1_source_fixed")]
    thresholds = {}
    for (candidate, seed), part in threshold_rows.groupby(["candidate", "seed"]):
        values = part.threshold.unique()
        if len(values) != 1 or not np.isfinite(values[0]):
            raise ValueError(f"Nonunique source threshold: {candidate}/{seed}")
        thresholds[(candidate, seed)] = float(values[0])
    keys = ["dataset", "seed", "anonymous_analysis_unit_id"]
    if scores.duplicated(keys + ["placement"]).any():
        raise ValueError("Duplicate placement predictions")
    grouped = scores.groupby(keys, sort=False)
    if not grouped.label.nunique().eq(1).all() or not grouped.size().eq(4).all():
        raise ValueError("Labels or prediction coverage do not match across placements")
    scores["source_fixed_threshold"] = [thresholds[(c, int(s))] for c, s in zip(scores.candidate, scores.seed)]
    scores["wrong"] = (scores.score >= scores.source_fixed_threshold).astype(int) != scores.label
    # Each record below is a dataset/seed/unit, not an independent new video per seed.
    units = scores.groupby(keys, sort=False).agg(wrong_placements=("wrong", "sum"), label=("label", "first")).reset_index()
    rows = []
    for (dataset, seed), part in units.groupby(["dataset", "seed"]):
        counts = part.wrong_placements.value_counts()
        n = len(part)
        any_error = int(part.wrong_placements.gt(0).sum())
        if any_error == 0:
            raise ValueError(f"Undefined conditional error proportions: {dataset}/{seed}")
        row = {"dataset": dataset, "seed": int(seed), "analysis_units": n,
               "real_units": int(part.label.eq(0).sum()), "fake_units": int(part.label.eq(1).sum()),
               "any_error_units": any_error, "any_error_rate_pct": 100 * any_error / n}
        for k in range(5):
            row[f"wrong_{k}_units"] = int(counts.get(k, 0))
            if k:
                row[f"wrong_{k}_conditional_pct"] = 100 * row[f"wrong_{k}_units"] / any_error
        if sum(row[f"wrong_{k}_units"] for k in range(5)) != n:
            raise ValueError("Error partition does not sum to the dataset size")
        rows.append(row)
    seed_rows = pd.DataFrame(rows)
    summaries = []
    measures = ["any_error_units", "any_error_rate_pct"] + [f"wrong_{k}_conditional_pct" for k in range(1, 5)]
    for dataset in DATASETS:
        part = seed_rows[seed_rows.dataset == dataset]
        if set(part.seed) != SEEDS or part.analysis_units.nunique() != 1:
            raise ValueError(f"Inconsistent dataset/seed coverage: {dataset}")
        row = {"dataset": dataset, "analysis_units_per_seed": int(part.analysis_units.iloc[0]), "training_seeds": 3}
        for m in measures:
            row[m + "_mean"] = float(part[m].mean())
            row[m + "_sd"] = float(part[m].std(ddof=1))
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    seed_rows.to_csv(results / "error_overlap_seed_results.csv", index=False, float_format="%.17g")
    summary.to_csv(results / "error_overlap_summary.csv", index=False, float_format="%.17g")
    threshold_output = scores[["placement", "candidate", "seed", "source_fixed_threshold"]].drop_duplicates()
    threshold_output.sort_values(["placement", "seed"]).to_csv(results / "error_overlap_source_thresholds.csv", index=False, float_format="%.17g")
    return seed_rows, summary


def overlap_figure(seed_rows: pd.DataFrame, summary: pd.DataFrame, figures: Path) -> None:
    summary = summary.set_index("dataset").loc[DATASETS]
    y = np.arange(len(DATASETS))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.4, 4.7), sharey=True,
                              gridspec_kw={"width_ratios": [1, 1.65]})
    a.errorbar(summary.any_error_rate_pct_mean, y,
               xerr=summary.any_error_rate_pct_sd, fmt="o", color="#333333",
               ecolor="#555555", capsize=3, markersize=5.5, linewidth=1.2, zorder=3)
    a.set_yticks(y, [f"{label}  (N={int(summary.loc[d, 'analysis_units_per_seed']):,})"
                      for d, label in zip(DATASETS, SHORT_DATASETS)])
    a.set_xlim(0, 35)
    a.set_xticks([0, 10, 20, 30])
    a.set_xlabel("Units with any error (%)")
    a.set_title("(a) Any-error rate", loc="left", fontsize=11, fontweight="bold")
    a.grid(axis="x", alpha=0.2)
    left = np.zeros(len(DATASETS))
    colors = ["#DEEBF7", "#9ECAE1", "#4292C6", "#08519C"]
    for k, color in enumerate(colors, 1):
        width = summary[f"wrong_{k}_conditional_pct_mean"].to_numpy()
        b.barh(y, width, left=left, color=color, edgecolor="white", linewidth=0.8,
               height=0.67, label=f"{k} placement" + ("s" if k > 1 else ""))
        for yy, x, w in zip(y, left, width):
            b.text(x + w / 2, yy, f"{w:.1f}", ha="center", va="center",
                   color="white" if k >= 3 else "#111111", fontsize=9)
        left += width
    if not np.allclose(left, 100, atol=1e-10):
        raise ValueError("Conditional mean proportions do not sum to 100%")
    b.set_xlim(0, 100)
    b.set_xticks([0, 25, 50, 75, 100])
    b.set_xlabel("Among units with any error (%)")
    b.set_title("(b) Number of placements making an error", loc="left", fontsize=11, fontweight="bold")
    a.invert_yaxis()
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    b.tick_params(axis="y", left=False)
    fig.legend(*b.get_legend_handles_labels(), loc="lower center", ncol=4,
               bbox_to_anchor=(0.61, 0.015), frameon=False, title="Placements misclassifying the same unit")
    fig.subplots_adjust(left=0.19, right=0.985, top=0.9, bottom=0.25, wspace=0.13)
    save(fig, figures, "figure3")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "results")
    parser.add_argument("--figures", type=Path, default=root / "figure")
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    training_figure(args.results, args.figures)
    seed_rows, summary = overlap_data(args.results)
    overlap_figure(seed_rows, summary, args.figures)
    print(summary.to_string(index=False))
    print("FIGURES_AND_ERROR_OVERLAP_GENERATED")


if __name__ == "__main__":
    main()
