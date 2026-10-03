#!/usr/bin/env python3
"""Extract the public 12-row primary seed table from existing metric artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CANDIDATES = {
    "Attn-Out": "std_attn_out_r30_alpha120_p025",
    "Attn-Q/V": "std_attn_qv_r15_alpha60_p025",
    "MLP": "std_mlp_r6_alpha24_p025",
    "MLP+Attn-Out": "std_mlp_attn_out_r5_alpha20_p025",
}
SEEDS = (3407, 7859, 12011)
DATASETS = ("Celeb-DF-v2", "DFDC", "DFDCP", "UADFV", "DeeperForensics-1.0", "WildDeepfake")
FIELDS = ["placement", "candidate", "seed", "source_frame_auc"] + [
    f"{dataset}_analysis_unit_auc" for dataset in DATASETS
] + ["macro_6_auc"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/primary_seed_results.csv"))
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for placement, candidate in CANDIDATES.items():
        for seed in SEEDS:
            run_name = f"{candidate}__seed{seed}"
            source = json.loads((args.source_root / run_name / "metrics.json").read_text(encoding="utf-8"))
            target = json.loads((args.target_root / run_name / "metrics.json").read_text(encoding="utf-8"))
            row: dict[str, object] = {
                "placement": placement,
                "candidate": candidate,
                "seed": seed,
                "source_frame_auc": source["datasets"]["FaceForensics++"]["frame"]["auc"],
                "macro_6_auc": target["macro_analysis_unit_auc"],
            }
            for dataset in DATASETS:
                row[f"{dataset}_analysis_unit_auc"] = target["datasets"][dataset]["analysis_unit"]["auc"]
            rows.append(row)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"PRIMARY_SEED_RESULTS_WRITTEN rows={len(rows)} path={args.output}")


if __name__ == "__main__":
    main()
