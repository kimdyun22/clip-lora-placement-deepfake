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

    supplementary = read_csv(RESULTS / "supplementary_adaptation_references.csv")
    expected_supplementary = {
        "Frozen CLIP + linear head": (0.25, 4098, 1, 0.8675945906338437, 0.8431212346269804),
        "BitFit": (0.25, 202754, 1, 0.9755614486197712, 0.9097680244479976),
        "Partial fine-tuning (last blocks)": (
            0.25, 50391042, 1, 0.9526570947187585, 0.9207926926178455
        ),
    }
    if {row["method"] for row in supplementary} != set(expected_supplementary):
        raise RuntimeError("supplementary adaptation-reference coverage mismatch")
    for row in supplementary:
        expected = expected_supplementary[row["method"]]
        observed = (
            float(row["p_sbi"]), int(row["updated_parameters"]), int(row["seeds"]),
            float(row["source_auc"]), float(row["macro_6_auc"]),
        )
        if observed[:3] != expected[:3] or not close(observed[3], expected[3]) or not close(observed[4], expected[4]):
            raise RuntimeError(f"supplementary adaptation-reference mismatch: {row['method']}")
        if row["evidence_level"] != "single_seed_descriptive":
            raise RuntimeError(f"invalid evidence label: {row['method']}")

    external = read_csv(RESULTS / "external_reproduction_audit.csv")
    expected_external = {
        "Celeb-DF-v2": (0.957, 0.9566754791804362),
        "DFDC": (0.872, 0.8720923021009722),
        "DFDCP": (0.929, 0.9288893467957965),
    }
    if {row["dataset"] for row in external} != set(expected_external):
        raise RuntimeError("external reproduction dataset coverage mismatch")
    for row in external:
        published, reproduced = expected_external[row["dataset"]]
        if row["method"] != "Forensics Adapter":
            raise RuntimeError("unexpected external reproduction method")
        if not close(float(row["published_auc"]), published) or not close(float(row["reproduced_auc"]), reproduced):
            raise RuntimeError(f"external reproduction mismatch: {row['dataset']}")
        if not close(float(row["delta"]), reproduced - published):
            raise RuntimeError(f"external reproduction delta mismatch: {row['dataset']}")
        if row["disposition"] != "agreement_at_three_decimals":
            raise RuntimeError(f"external reproduction disposition mismatch: {row['dataset']}")

    print(
        "ADAPTATION_BASELINES_VERIFIED "
        f"methods={len(grouped)} seed_rows={len(seed_rows)} "
        f"supplementary_rows={len(supplementary)} external_rows={len(external)}"
    )


if __name__ == "__main__":
    main()
