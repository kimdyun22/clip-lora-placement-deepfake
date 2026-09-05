#!/usr/bin/env python3
"""Recompute the three-seed adaptation-baseline summaries from public seed rows."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEEDS = {3407, 7859, 12011}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15)


def main() -> None:
    seed_rows = read_csv(RESULTS / "adaptation_baseline_seed_results.csv")
    aggregate_rows = read_csv(RESULTS / "adaptation_baselines.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in seed_rows:
        grouped[row["method"]].append(row)

    if set(grouped) != {row["method"] for row in aggregate_rows}:
        raise RuntimeError("method coverage differs between seed and aggregate tables")

    for aggregate in aggregate_rows:
        method = aggregate["method"]
        rows = grouped[method]
        seeds = {int(row["seed"]) for row in rows}
        if len(rows) != 3 or seeds != SEEDS:
            raise RuntimeError(f"invalid seed coverage for {method}: {sorted(seeds)}")
        source = [float(row["source_frame_auc"]) for row in rows]
        target = [float(row["macro_6_auc"]) for row in rows]
        observed = {
            "source_auc_mean": statistics.mean(source),
            "source_auc_sd": statistics.stdev(source),
            "macro_6_auc_mean": statistics.mean(target),
            "macro_6_auc_sd": statistics.stdev(target),
        }
        for field, value in observed.items():
            expected = float(aggregate[field])
            if not close(value, expected):
                raise RuntimeError(
                    f"{method} {field} mismatch: recomputed={value!r}, aggregate={expected!r}"
                )
    print(f"ADAPTATION_BASELINES_VERIFIED methods={len(grouped)} seed_rows={len(seed_rows)}")


if __name__ == "__main__":
    main()
