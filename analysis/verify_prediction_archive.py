#!/usr/bin/env python3
"""Validate completeness, anonymity, pairing, and checksum of the public predictions."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


PLACEMENTS = {"attn_out", "attn_qv", "mlp", "mlp_attn_out"}
SEEDS = {3407, 7859, 12011}
DATASET_COUNTS = {
    "Celeb-DF-v2": 518,
    "DFDC": 4704,
    "DFDCP": 652,
    "UADFV": 98,
    "DeeperForensics-1.0": 40870,
    "WildDeepfake": 157,
}
EXPECTED_COLUMNS = {
    "anonymous_analysis_unit_id", "dataset", "placement", "candidate",
    "seed", "label", "score", "frame_count",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="?",
                        default=Path("results/predictions/analysis_unit_scores.parquet"))
    args = parser.parse_args()
    table = pq.read_table(args.archive)
    if set(table.column_names) != EXPECTED_COLUMNS:
        raise ValueError(f"unexpected archive columns: {table.column_names}")
    data = table.to_pydict()
    expected_rows = sum(DATASET_COUNTS.values()) * len(PLACEMENTS) * len(SEEDS)
    if table.num_rows != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, observed {table.num_rows}")
    if set(data["placement"]) != PLACEMENTS or set(data["seed"]) != SEEDS:
        raise ValueError("placement or seed coverage is incomplete")
    counts = Counter(zip(data["placement"], data["seed"], data["dataset"]))
    for placement in PLACEMENTS:
        for seed in SEEDS:
            for dataset, expected in DATASET_COUNTS.items():
                if counts[(placement, seed, dataset)] != expected:
                    raise ValueError(f"count mismatch: {placement}/{seed}/{dataset}")
    identity: dict[tuple[str, str], tuple[int, int]] = {}
    for anon, dataset, label, score, frames in zip(
        data["anonymous_analysis_unit_id"], data["dataset"], data["label"],
        data["score"], data["frame_count"]
    ):
        if len(anon) != 64 or any(char not in "0123456789abcdef" for char in anon):
            raise ValueError("non-anonymous analysis-unit identifier detected")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("invalid probability score")
        key = (dataset, anon)
        value = (int(label), int(frames))
        previous = identity.setdefault(key, value)
        if previous != value:
            raise ValueError(f"inconsistent label/frame count for {key}")
    checksum_file = args.archive.parent / "SHA256SUMS.txt"
    expected_hash, expected_name = checksum_file.read_text(encoding="ascii").split()
    if expected_name != args.archive.name or sha256_file(args.archive) != expected_hash:
        raise ValueError("prediction archive checksum mismatch")
    print(f"PREDICTION_ARCHIVE_VERIFIED rows={table.num_rows} units={len(identity)}")


if __name__ == "__main__":
    main()
