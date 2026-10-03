#!/usr/bin/env python3
"""Create a de-identified primary prediction archive from existing evaluations."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


CANDIDATES = {
    "attn_out": "std_attn_out_r30_alpha120_p025",
    "attn_qv": "std_attn_qv_r15_alpha60_p025",
    "mlp": "std_mlp_r6_alpha24_p025",
    "mlp_attn_out": "std_mlp_attn_out_r5_alpha20_p025",
}
SEEDS = (3407, 7859, 12011)
SOURCE_COLUMNS = ["dataset", "analysis_unit_id", "label", "fake_probability", "frame_count"]


def anonymous_id(dataset: str, native_id: str) -> str:
    return hashlib.sha256(f"{dataset}\0{native_id}".encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Directory containing <candidate>__seed<seed>/analysis_unit_predictions.parquet")
    parser.add_argument("--output-dir", type=Path, default=Path("results/predictions"))
    args = parser.parse_args()

    tables: list[pa.Table] = []
    reference: dict[tuple[str, str], tuple[int, int]] | None = None
    for placement, candidate in CANDIDATES.items():
        for seed in SEEDS:
            source = args.source_root / f"{candidate}__seed{seed}" / "analysis_unit_predictions.parquet"
            if not source.is_file():
                raise FileNotFoundError(source)
            table = pq.read_table(source, columns=SOURCE_COLUMNS)
            data = table.to_pydict()
            current = {
                (dataset, native_id): (int(label), int(frame_count))
                for dataset, native_id, label, frame_count in zip(
                    data["dataset"], data["analysis_unit_id"], data["label"], data["frame_count"]
                )
            }
            if len(current) != table.num_rows:
                raise ValueError(f"duplicate analysis-unit identity in {source}")
            if reference is None:
                reference = current
            elif current != reference:
                raise ValueError(f"analysis-unit identity/label/frame-count mismatch in {source}")
            tables.append(pa.table({
                "anonymous_analysis_unit_id": [
                    anonymous_id(dataset, native_id)
                    for dataset, native_id in zip(data["dataset"], data["analysis_unit_id"])
                ],
                "dataset": data["dataset"],
                "placement": [placement] * table.num_rows,
                "candidate": [candidate] * table.num_rows,
                "seed": pa.array([seed] * table.num_rows, type=pa.int32()),
                "label": pa.array(data["label"], type=pa.int8()),
                "score": pa.array(data["fake_probability"], type=pa.float64()),
                "frame_count": pa.array(data["frame_count"], type=pa.int32()),
            }))

    combined = pa.concat_tables(tables)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "analysis_unit_scores.parquet"
    pq.write_table(combined, output, compression="zstd", compression_level=9)
    checksum = sha256_file(output)
    (args.output_dir / "SHA256SUMS.txt").write_text(
        f"{checksum}  {output.name}\n", encoding="ascii"
    )
    print(f"PREDICTION_ARCHIVE_WRITTEN rows={combined.num_rows} sha256={checksum} path={output}")


if __name__ == "__main__":
    main()
