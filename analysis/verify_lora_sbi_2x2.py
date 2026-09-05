#!/usr/bin/env python3
"""Recompute the source-side LoRA x SBI 2x2 summary from public seed rows."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


EXPECTED = {
    "Frozen / no SBI": (0.8705857259610941, 0.0018551294321322081),
    "Frozen / SBI": (0.8674298966980777, 0.001208168696919747),
    "Attn-Out LoRA / no SBI": (0.9766213701285175, 0.001842332026563178),
    "Attn-Out LoRA / SBI": (0.9842650529159858, 0.0015554741076342427),
}
SEEDS = {3407, 7859, 12011}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?", default=Path("results/lora_sbi_2x2.csv"))
    args = parser.parse_args()

    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with args.csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["source_metric"] != "frame_auc":
                raise ValueError(f"unexpected source metric: {row['source_metric']}")
            grouped[row["condition"]].append((int(row["seed"]), float(row["source_auc"])))

    if set(grouped) != set(EXPECTED):
        raise ValueError(f"unexpected factorial cells: {sorted(grouped)}")
    for condition, expected in EXPECTED.items():
        values = grouped[condition]
        if {seed for seed, _ in values} != SEEDS or len(values) != 3:
            raise ValueError(f"{condition}: expected exactly seeds {sorted(SEEDS)}")
        aucs = [auc for _, auc in sorted(values)]
        observed = (statistics.mean(aucs), statistics.stdev(aucs))
        if any(abs(got - want) > 1e-15 for got, want in zip(observed, expected)):
            raise ValueError(f"{condition}: expected {expected}, observed {observed}")
        print(f"{condition}: {observed[0]:.6f} +/- {observed[1]:.6f}")
    print("LORA_SBI_2X2_VERIFIED")


if __name__ == "__main__":
    main()
