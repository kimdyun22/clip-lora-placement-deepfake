#!/usr/bin/env python3
"""Recompute the primary cross-dataset table from the released prediction archive.

This runs no training and reads no checkpoint. It takes the published
per-analysis-unit scores, recomputes each dataset AUC per placement and seed,
derives the unweighted Macro-6, and checks the result against
``results/primary_results.csv`` and ``results/primary_seed_results.csv``.
"""

from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARCHIVE = RESULTS / "predictions" / "analysis_unit_scores.parquet"

DATASETS = (
    "Celeb-DF-v2",
    "DFDC",
    "DFDCP",
    "UADFV",
    "DeeperForensics-1.0",
    "WildDeepfake",
)
PLACEMENT_NAME = {
    "attn_out": "Attn-Out",
    "attn_qv": "Attn-Q/V",
    "mlp": "MLP",
    "mlp_attn_out": "MLP+Attn-Out",
}
# Published values are rounded to four decimals in the manuscript tables; the
# stored artifacts keep full precision, so compare at full precision.
TOLERANCE = 5e-9


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def recompute() -> dict[tuple[str, int, str], float]:
    table = pq.read_table(ARCHIVE, columns=["dataset", "placement", "seed", "label", "score"])
    columns = {name: table[name].to_pylist() for name in table.column_names}
    buckets: dict[tuple[str, int, str], tuple[list[int], list[float]]] = {}
    for dataset, placement, seed, label, score in zip(
        columns["dataset"], columns["placement"], columns["seed"],
        columns["label"], columns["score"],
    ):
        labels, scores = buckets.setdefault((placement, int(seed), dataset), ([], []))
        labels.append(int(label))
        scores.append(float(score))
    return {key: roc_auc_score(labels, scores) for key, (labels, scores) in buckets.items()}


def main() -> None:
    if not ARCHIVE.is_file():
        raise SystemExit(f"missing prediction archive: {ARCHIVE}")

    auc = recompute()
    seeds = sorted({key[1] for key in auc})
    placements = sorted({key[0] for key in auc})
    failures: list[str] = []

    if set(placements) != set(PLACEMENT_NAME):
        failures.append(f"unexpected placements in archive: {placements}")
    if len(seeds) != 3:
        failures.append(f"expected three seeds, found {seeds}")

    # Per-seed dataset AUC and Macro-6 against the seed-level table.
    seed_rows = {
        (row["placement"], int(row["seed"])): row
        for row in read_csv("primary_seed_results.csv")
    }
    for placement, name in PLACEMENT_NAME.items():
        for seed in seeds:
            row = seed_rows.get((name, seed))
            if row is None:
                failures.append(f"no seed row for {name} seed={seed}")
                continue
            for dataset in DATASETS:
                recomputed = auc[(placement, seed, dataset)]
                published = float(row[f"{dataset}_analysis_unit_auc"])
                if abs(recomputed - published) > TOLERANCE:
                    failures.append(
                        f"{name} seed={seed} {dataset}: archive={recomputed!r} table={published!r}"
                    )
            macro = st.fmean(auc[(placement, seed, dataset)] for dataset in DATASETS)
            published_macro = float(row["macro_6_auc"])
            if abs(macro - published_macro) > TOLERANCE:
                failures.append(
                    f"{name} seed={seed} Macro-6: archive={macro!r} table={published_macro!r}"
                )

    # Three-seed mean and sample SD against the aggregate table.
    aggregate = {row["placement"]: row for row in read_csv("primary_results.csv")}
    for placement, name in PLACEMENT_NAME.items():
        row = aggregate.get(name)
        if row is None:
            failures.append(f"no aggregate row for {name}")
            continue
        for dataset in DATASETS:
            values = [auc[(placement, seed, dataset)] for seed in seeds]
            for label, computed, published in (
                ("mean", st.fmean(values), float(row[f"{dataset}_auc_mean"])),
                ("sd", st.stdev(values), float(row[f"{dataset}_auc_sd"])),
            ):
                if abs(computed - published) > TOLERANCE:
                    failures.append(
                        f"{name} {dataset} {label}: archive={computed!r} table={published!r}"
                    )
        macros = [
            st.fmean(auc[(placement, seed, dataset)] for dataset in DATASETS)
            for seed in seeds
        ]
        for label, computed, published in (
            ("mean", st.fmean(macros), float(row["macro_6_auc_mean"])),
            ("sd", st.stdev(macros), float(row["macro_6_auc_sd"])),
        ):
            if abs(computed - published) > TOLERANCE:
                failures.append(
                    f"{name} Macro-6 {label}: archive={computed!r} table={published!r}"
                )

    if failures:
        for failure in failures:
            print(f"BLOCKER {failure}")
        raise SystemExit(1)

    checks = len(PLACEMENT_NAME) * (len(seeds) * (len(DATASETS) + 1) + 2 * (len(DATASETS) + 1))
    print(
        f"PREDICTION_METRICS_VERIFIED placements={len(PLACEMENT_NAME)} "
        f"seeds={len(seeds)} datasets={len(DATASETS)} checks={checks}"
    )


if __name__ == "__main__":
    main()