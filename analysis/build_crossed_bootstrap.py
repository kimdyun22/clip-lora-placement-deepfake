#!/usr/bin/env python3
"""Seed-aware hierarchical paired bootstrap over the placement pairs.

The released DeLong tests are conditional on one trained checkpoint and one
dataset. This uses a crossed bootstrap over the two levels that the reported
average depends on: training seeds and the shared analysis units within each
dataset. The same unit resample is applied to every selected seed and placement
so both forms of pairing are preserved. DFDC and DFDCP are
averaged into one benchmark family before the cross-dataset mean so that
challenge does not carry two of six equal weights.

With three seeds the seed level is coarse; the interval is reported as an
uncertainty summary, not as a significance test.
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results" / "predictions" / "analysis_unit_scores.parquet"

DATASETS = ("Celeb-DF-v2", "DFDC", "DFDCP", "UADFV", "DeeperForensics-1.0", "WildDeepfake")
FAMILIES = (("Celeb-DF-v2",), ("DFDC", "DFDCP"), ("UADFV",),
            ("DeeperForensics-1.0",), ("WildDeepfake",))
PLACEMENT = {"attn_out": "Attn-Out", "attn_qv": "Attn-Q/V",
             "mlp": "MLP", "mlp_attn_out": "MLP+Attn-Out"}


def fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    ranks = rankdata(scores, method="average")
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def load() -> tuple[dict, list[int]]:
    table = pq.read_table(
        ARCHIVE, columns=["dataset", "placement", "seed", "label", "score", "anonymous_analysis_unit_id"]
    )
    col = {name: np.asarray(table[name].to_pylist()) for name in table.column_names}
    seeds = sorted({int(s) for s in col["seed"]})

    store: dict[tuple[int, str], dict] = {}
    for seed in seeds:
        for dataset in DATASETS:
            base = (col["seed"].astype(int) == seed) & (col["dataset"] == dataset)
            # Unit order is fixed per (seed, dataset) so every placement is
            # resampled on exactly the same units.
            reference = col["anonymous_analysis_unit_id"][base & (col["placement"] == "attn_out")]
            if len(reference) != len(set(reference.tolist())):
                raise RuntimeError(f"duplicate analysis-unit IDs: seed={seed}, dataset={dataset}")
            index = {uid: pos for pos, uid in enumerate(reference)}
            labels = np.zeros(len(reference), dtype=np.int64)
            scores = {pl: np.zeros(len(reference)) for pl in PLACEMENT}
            for placement in PLACEMENT:
                mask = base & (col["placement"] == placement)
                observed_ids = col["anonymous_analysis_unit_id"][mask]
                if (len(observed_ids) != len(reference)
                        or set(observed_ids.tolist()) != set(reference.tolist())):
                    raise RuntimeError(
                        f"unit-set mismatch: seed={seed}, dataset={dataset}, placement={placement}"
                    )
                positions = np.fromiter(
                    (index[u] for u in col["anonymous_analysis_unit_id"][mask]),
                    dtype=np.int64, count=int(mask.sum()),
                )
                scores[placement][positions] = col["score"][mask].astype(float)
                labels[positions] = col["label"][mask].astype(int)
            store[(seed, dataset)] = {
                "labels": labels,
                "scores": scores,
                "pos": np.flatnonzero(labels == 1),
                "neg": np.flatnonzero(labels == 0),
            }
    # Analysis units are crossed with, rather than nested in, training seed.
    for dataset in DATASETS:
        first = store[(seeds[0], dataset)]
        for seed in seeds[1:]:
            if not np.array_equal(first["labels"], store[(seed, dataset)]["labels"]):
                raise RuntimeError(f"label/order mismatch across seeds: {dataset}")
    return store, seeds


def family_macro(per_dataset: dict[str, float]) -> float:
    return float(np.mean([np.mean([per_dataset[d] for d in fam]) for fam in FAMILIES]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "hierarchical_bootstrap_crossed.csv")
    args = parser.parse_args()

    store, seeds = load()
    rng = np.random.default_rng(args.seed)
    names = list(PLACEMENT)
    pairs = list(itertools.combinations(names, 2))

    draws = {pair: np.empty(args.replicates) for pair in pairs}
    point = {}

    # Point estimate on the observed data.
    observed = {}
    for placement in names:
        per_seed = []
        for seed in seeds:
            per_dataset = {
                d: fast_auc(store[(seed, d)]["labels"], store[(seed, d)]["scores"][placement])
                for d in DATASETS
            }
            per_seed.append(family_macro(per_dataset))
        observed[placement] = float(np.mean(per_seed))
    for a, b in pairs:
        point[(a, b)] = observed[a] - observed[b]

    for rep in range(args.replicates):
        chosen = rng.choice(seeds, size=len(seeds), replace=True)
        macro = {pl: [] for pl in names}
        # Draw each dataset's units once per replicate and reuse that draw for
        # every selected seed and placement (crossed paired bootstrap).
        unit_draws = {}
        for dataset in DATASETS:
            reference = store[(seeds[0], dataset)]
            pos = rng.choice(reference["pos"], size=len(reference["pos"]), replace=True)
            neg = rng.choice(reference["neg"], size=len(reference["neg"]), replace=True)
            unit_draws[dataset] = np.concatenate([pos, neg])
        for seed in chosen:
            per_dataset = {pl: {} for pl in names}
            for dataset in DATASETS:
                cell = store[(int(seed), dataset)]
                take = unit_draws[dataset]
                labels = cell["labels"][take]
                for placement in names:
                    per_dataset[placement][dataset] = fast_auc(
                        labels, cell["scores"][placement][take]
                    )
            for placement in names:
                macro[placement].append(family_macro(per_dataset[placement]))
        means = {pl: float(np.mean(macro[pl])) for pl in names}
        for a, b in pairs:
            draws[(a, b)][rep] = means[a] - means[b]
        if (rep + 1) % 500 == 0:
            print(f"  {rep + 1}/{args.replicates} replicates", flush=True)

    rows = []
    for a, b in pairs:
        values = draws[(a, b)]
        low, high = np.percentile(values, [2.5, 97.5])
        rows.append({
            "first": PLACEMENT[a],
            "second": PLACEMENT[b],
            "metric": "family_balanced_macro_auc_difference",
            "replicates": args.replicates,
            "point_estimate": repr(point[(a, b)]),
            "ci_low": repr(float(low)),
            "ci_high": repr(float(high)),
            "excludes_zero": str(bool(low > 0 or high < 0)).lower(),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'pair':32s} {'delta':>10s} {'95% CI':>26s}   excludes 0")
    for row in rows:
        lo, hi = float(row["ci_low"]), float(row["ci_high"])
        print(f"{row['first'] + ' - ' + row['second']:32s} {float(row['point_estimate']):+10.5f}"
              f"   [{lo:+.5f}, {hi:+.5f}]   {row['excludes_zero']}")
    print(f"\nwrote {args.output}")
    print("HIERARCHICAL_BOOTSTRAP_DONE")


if __name__ == "__main__":
    main()
