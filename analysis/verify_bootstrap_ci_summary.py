#!/usr/bin/env python3
"""Verify the compact bootstrap summary against the 108 seed-level tests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from build_bootstrap_ci_summary import FIELDS, summary_rows


def normalise(row: dict[str, object]) -> dict[str, str]:
    return {field: str(row.get(field, "")) for field in FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, default=Path("results/statistics.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/bootstrap_ci_summary.csv"))
    args = parser.parse_args()
    expected = [normalise(row) for row in summary_rows(args.statistics)]
    with args.summary.open(newline="", encoding="utf-8") as handle:
        observed = list(csv.DictReader(handle))
    if observed != expected:
        raise ValueError("bootstrap summary differs from seed-conditional statistics")
    print(f"BOOTSTRAP_CI_SUMMARY_VERIFIED rows={len(observed)}")


if __name__ == "__main__":
    main()
