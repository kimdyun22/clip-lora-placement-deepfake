#!/usr/bin/env python3
"""Rewrite the frozen experiment manifests into portable public manifests.

The frozen manifests used for training and evaluation store machine-local
absolute frame paths. This builder strips the local dataset root from every
path and emits a dataset-relative path instead, so an external reader can
point the loader at their own copy of each benchmark with ``--dataset-root``.

Nothing else changes: sample identity, frame selection, labels, splits,
analysis-unit keys and row order are copied verbatim from the frozen input.
The source manifests are not redistributed, so this script is provenance
documentation; it only runs where those inputs are available.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

# Machine-local dataset root -> canonical public directory name. The root is
# the directory the local copy was mounted under; the public name is the
# directory an external reader is expected to create under --dataset-root.
LAYOUT = {
    "ffpp_c23_train": ("/data/deepfakebench/FaceForensics++_bench/", "FaceForensics++"),
    "ffpp_c23_val": ("/data/deepfakebench/FaceForensics++_bench/", "FaceForensics++"),
    "Celeb-DF-v2_test": ("/data/deepfakebench/Celeb-DF-v2_bench/", "Celeb-DF-v2"),
    "DFDC_test": ("/data/deepfakebench/DFDC_bench/", "DFDC"),
    "DFDCP_test": ("/data/deepfakebench/DFDCP_bench/", "DFDCP"),
    "UADFV_test": ("/data/deepfakebench/UADFV_bench/", "UADFV"),
    "DeeperForensics-1.0_test": ("/data/deepfakebench/DeeperForensics-1.0/", "DeeperForensics-1.0"),
    "WildDeepfake_test": ("/data/WildDeepfake/deepfake_in_the_wild/", "WildDeepfake"),
}

# Public output name for each frozen manifest. The manifests are gzip-compressed
# because the uncompressed DeeperForensics-1.0 frame list is far larger than the
# GitHub per-file limit; the loader reads .jsonl and .jsonl.gz identically.
OUTPUT_NAME = {
    "ffpp_c23_train": "ffpp_train.jsonl.gz",
    "ffpp_c23_val": "ffpp_val.jsonl.gz",
    "Celeb-DF-v2_test": "cdfv2_test.jsonl.gz",
    "DFDC_test": "dfdc_test.jsonl.gz",
    "DFDCP_test": "dfdcp_test.jsonl.gz",
    "UADFV_test": "uadfv_test.jsonl.gz",
    "DeeperForensics-1.0_test": "df1_test.jsonl.gz",
    "WildDeepfake_test": "wdf_test.jsonl.gz",
}

# Fields carried into the public manifest, in a fixed order.
PUBLIC_FIELDS = (
    "dataset",
    "split",
    "label",
    "label_group",
    "video_id",
    "sequence_id",
    "raw_video_id",
    "frame_index",
    "sample_id",
    "frame_path",
)


def relative_path(absolute: str, local_root: str, public_name: str) -> str:
    """Strip the local dataset root and normalise onto the public layout."""
    if not absolute.startswith(local_root):
        raise ValueError(f"path outside the declared dataset root: {absolute}")
    remainder = absolute[len(local_root):].lstrip("/")
    if not remainder:
        raise ValueError(f"path collapses to the dataset root: {absolute}")
    first = remainder.split("/", 1)[0]
    if first == public_name:
        return remainder
    return f"{public_name}/{remainder}"


def convert(source: Path, destination: Path, local_root: str, public_name: str) -> int:
    rows = 0
    # mtime=0 so rebuilding identical content produces byte-identical output.
    with source.open(encoding="utf-8") as reader, destination.open("wb") as binary:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=binary, mtime=0) as writer:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                public = {key: record[key] for key in PUBLIC_FIELDS if key != "frame_path"}
                public["frame_path"] = relative_path(record["frame_path"], local_root, public_name)
                payload = json.dumps(public, ensure_ascii=False, sort_keys=False) + "\n"
                writer.write(payload.encode("utf-8"))
                rows += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Directory holding the frozen <stem>.jsonl manifests")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/manifests"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stem, (local_root, public_name) in LAYOUT.items():
        source = args.source_dir / f"{stem}.jsonl"
        if not source.is_file():
            raise SystemExit(f"missing frozen manifest: {source}")
        destination = args.output_dir / OUTPUT_NAME[stem]
        rows = convert(source, destination, local_root, public_name)
        print(f"{destination}: {rows} rows")
    print("PUBLIC_MANIFESTS_WRITTEN")


if __name__ == "__main__":
    main()