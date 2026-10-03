#!/usr/bin/env python3
"""Report seed-wise differences and unit-only bootstrap uncertainty.

The estimand is the family-balanced target macro AUC used by the crossed
seed-by-analysis-unit bootstrap.  Seed-only resampling holds the observed
analysis units fixed.  Unit-only resampling holds all three trained seeds
fixed and applies one class-stratified unit draw per dataset to every seed and
placement.  Both preserve the paired comparison structure.  With only three
trained seeds, no seed-level confidence interval is reported: the output gives
the three observed seed-specific differences, their range, and whether their
signs agree.
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from build_crossed_bootstrap import (
    DATASETS,
    PLACEMENT,
    family_macro,
    fast_auc,
    load,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "uncertainty_decomposition.csv",
    )
    args = parser.parse_args()

    store, seeds = load()
    names = list(PLACEMENT)
    pairs = list(itertools.combinations(names, 2))

    # Cache the observed per-seed estimand.  Three seeds are too few for a
    # nominal seed-level confidence interval, so retain the values directly.
    observed_by_seed = {pl: [] for pl in names}
    for seed in seeds:
        for pl in names:
            per_dataset = {
                d: fast_auc(store[(seed, d)]["labels"], store[(seed, d)]["scores"][pl])
                for d in DATASETS
            }
            observed_by_seed[pl].append(family_macro(per_dataset))
    observed = {pl: float(np.mean(vals)) for pl, vals in observed_by_seed.items()}

    # Unit-only bootstrap.  Seeds remain fixed; every dataset draw is shared
    # by all placements and seeds, preserving both pairings.
    rng = np.random.default_rng(args.seed)
    unit_draws = {pair: np.empty(args.replicates) for pair in pairs}
    for rep in range(args.replicates):
        takes = {}
        for dataset in DATASETS:
            reference = store[(seeds[0], dataset)]
            pos = rng.choice(reference["pos"], size=len(reference["pos"]), replace=True)
            neg = rng.choice(reference["neg"], size=len(reference["neg"]), replace=True)
            takes[dataset] = np.concatenate([pos, neg])
        macro = {pl: [] for pl in names}
        for seed in seeds:
            per_dataset = {pl: {} for pl in names}
            for dataset in DATASETS:
                cell = store[(seed, dataset)]
                take = takes[dataset]
                labels = cell["labels"][take]
                for pl in names:
                    per_dataset[pl][dataset] = fast_auc(labels, cell["scores"][pl][take])
            for pl in names:
                macro[pl].append(family_macro(per_dataset[pl]))
        means = {pl: float(np.mean(macro[pl])) for pl in names}
        for pair in pairs:
            unit_draws[pair][rep] = means[pair[0]] - means[pair[1]]
        if (rep + 1) % 500 == 0:
            print(f"  unit-only {rep + 1}/{args.replicates}", flush=True)

    rows = []
    for pair in pairs:
        point = observed[pair[0]] - observed[pair[1]]
        seed_differences = np.asarray(observed_by_seed[pair[0]]) - np.asarray(
            observed_by_seed[pair[1]]
        )
        low, high = np.percentile(unit_draws[pair], [2.5, 97.5])
        signs_consistent = bool(np.all(seed_differences > 0) or np.all(seed_differences < 0))
        row = {
            "first": PLACEMENT[pair[0]],
            "second": PLACEMENT[pair[1]],
            "metric": "family_balanced_macro_auc_difference",
            "point_estimate": repr(float(point)),
        }
        for seed, value in zip(seeds, seed_differences):
            row[f"seed_{seed}_difference"] = repr(float(value))
        row.update({
            "seed_difference_range_low": repr(float(seed_differences.min())),
            "seed_difference_range_high": repr(float(seed_differences.max())),
            "signs_consistent_across_seeds": str(signs_consistent).lower(),
            "unit_bootstrap_replicates": args.replicates,
            "unit_ci_low": repr(float(low)),
            "unit_ci_high": repr(float(high)),
            "unit_ci_excludes_zero": str(bool(low > 0 or high < 0)).lower(),
        })
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")
    print("UNCERTAINTY_DECOMPOSITION_DONE")


if __name__ == "__main__":
    main()
