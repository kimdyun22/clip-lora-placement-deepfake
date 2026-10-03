#!/usr/bin/env python3
"""Recompute the final efficiency table from the five public hardware rounds."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLACEMENTS = {"attn_out", "attn_qv", "mlp", "mlp_attn_out"}
FIELDS = {
    "batch1_ms": ("batch1_ms_mean", "batch1_ms_sd"),
    "batch32_fps": ("batch32_fps_mean", "batch32_fps_sd"),
    "inference_peak_allocated_gib": (
        "inference_peak_allocated_gib_mean", "inference_peak_allocated_gib_sd"
    ),
    "training_step_ms": ("training_step_ms_mean", "training_step_ms_sd"),
    "training_peak_allocated_gib": (
        "training_peak_allocated_gib_mean", "training_peak_allocated_gib_sd"
    ),
    "projected_epoch_minutes": ("projected_epoch_minutes_mean", "projected_epoch_minutes_sd"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    raw = read_csv(RESULTS / "efficiency_round_results.csv")
    summary = read_csv(RESULTS / "efficiency.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        grouped[row["placement"]].append(row)
    if set(grouped) != PLACEMENTS or {row["placement"] for row in summary} != PLACEMENTS:
        raise RuntimeError("efficiency placement coverage is incomplete")
    by_placement = {row["placement"]: row for row in summary}
    for placement, rows in grouped.items():
        if len(rows) != 5 or {int(row["round"]) for row in rows} != {1, 2, 3, 4, 5}:
            raise RuntimeError(f"invalid round coverage for {placement}")
        expected = by_placement[placement]
        if {int(row["trainable_params"]) for row in rows} != {1_478_658}:
            raise RuntimeError(f"trainable parameter mismatch for {placement}")
        if not math.isclose(float(rows[0]["counted_gflops"]), float(expected["counted_gflops"]),
                            rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(f"FLOP mismatch for {placement}")
        for raw_field, (mean_field, sd_field) in FIELDS.items():
            values = [float(row[raw_field]) for row in rows]
            observed = (statistics.mean(values), statistics.stdev(values))
            reported = (float(expected[mean_field]), float(expected[sd_field]))
            for name, left, right in zip(("mean", "sample_sd"), observed, reported):
                if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
                    raise RuntimeError(
                        f"{placement} {raw_field} {name} mismatch: recomputed={left!r}, reported={right!r}"
                    )
    print(f"EFFICIENCY_VERIFIED placements={len(grouped)} round_rows={len(raw)}")


if __name__ == "__main__":
    main()
