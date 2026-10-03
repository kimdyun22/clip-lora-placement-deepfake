#!/usr/bin/env python3
"""Verify that the public operating-point summary is regenerated from seed rows."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from build_operating_point_summary import FIELDS, summary_rows


def normalise(row: dict[str, object]) -> dict[str, str]:
    return {field: str(row.get(field, "")) for field in FIELDS}


def same(expected: str, observed: str) -> bool:
    """Exact text match, or numeric agreement to 1e-12 (float printing differs across Python versions)."""
    if expected == observed:
        return True
    try:
        return math.isclose(float(expected), float(observed), rel_tol=0.0, abs_tol=1e-12)
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("results/calibration_thresholds.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/operating_point_summary.csv"))
    args = parser.parse_args()
    expected = [normalise(row) for row in summary_rows(args.raw)]
    with args.summary.open(newline="", encoding="utf-8") as handle:
        observed = list(csv.DictReader(handle))
    if len(observed) != len(expected) or any(
        list(row) != FIELDS or not all(same(want[field], row[field]) for field in FIELDS)
        for want, row in zip(expected, observed)
    ):
        raise ValueError("operating-point summary differs from its seed-level source")
    print(f"OPERATING_POINT_SUMMARY_VERIFIED rows={len(observed)}")


if __name__ == "__main__":
    main()
