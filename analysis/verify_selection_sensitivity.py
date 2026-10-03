#!/usr/bin/env python3
"""Recompute source-selection summaries from the fifteen public seed rows."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEEDS = {3407, 7859, 12011}
METRICS = {
    "source_auc": "source_frame_auc",
    "source_eer": "source_frame_eer",
    "source_ap": "source_frame_ap",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    raw = read_csv(RESULTS / "selection_seed_results.csv")
    summary = read_csv(RESULTS / "selection_sensitivity.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        grouped[row["candidate"]].append(row)
    if set(grouped) != {row["candidate"] for row in summary}:
        raise RuntimeError("candidate coverage differs between seed and summary tables")

    for expected in summary:
        candidate = expected["candidate"]
        rows = grouped[candidate]
        if len(rows) != 3 or {int(row["seed"]) for row in rows} != SEEDS:
            raise RuntimeError(f"invalid seed coverage for {candidate}")
        for prefix, raw_field in METRICS.items():
            values = [float(row[raw_field]) for row in rows]
            observed = (statistics.mean(values), statistics.stdev(values))
            reported = (float(expected[f"{prefix}_mean"]), float(expected[f"{prefix}_sd"]))
            for name, left, right in zip(("mean", "sample_sd"), observed, reported):
                if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15):
                    raise RuntimeError(
                        f"{candidate} {prefix} {name} mismatch: recomputed={left!r}, reported={right!r}"
                    )
    print(f"SELECTION_SENSITIVITY_VERIFIED candidates={len(grouped)} seed_rows={len(raw)}")


if __name__ == "__main__":
    main()
