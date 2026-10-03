#!/usr/bin/env python3
"""Compute reporting-only grouping sensitivities from saved frame predictions.

Private per-run frame_predictions.parquet files are required. No inference,
training, threshold fitting, or replacement of primary target groups is done.
Only numerical summaries, not frame paths or identifiers, are exported.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (3407, 7859, 12011)
PRIMARY = {
    "std_attn_out_r30_alpha120_p025": "Attn-Out",
    "std_attn_qv_r15_alpha60_p025": "Attn-Q/V",
    "std_mlp_r6_alpha24_p025": "MLP",
    "std_mlp_attn_out_r5_alpha20_p025": "MLP+Attn-Out",
}
CANDIDATES = {
    "sel_mlp_r4_alpha16_p025": "MLP",
    "sel_attn_qv_r16_alpha64_p025": "Attn-Q/V",
    "sel_attn_out_r16_alpha64_p025": "Attn-Out",
    "sel_mlp_attn_out_r4_alpha16_p025": "MLP+Attn-Out",
}


def frame_table(path: Path, dataset: str) -> pd.DataFrame:
    columns = ["sample_id", "dataset", "split", "video_id", "sequence_id",
               "frame_path", "label", "fake_probability"]
    data = pd.read_parquet(path, columns=columns, filters=[("dataset", "==", dataset)], use_threads=False)
    if data.empty or data.sample_id.duplicated().any() or data.frame_path.duplicated().any():
        raise ValueError(f"missing or duplicated frame predictions: {path}")
    scores = data.fake_probability.to_numpy(dtype=float)
    if (not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()
            or set(data.label.unique()) != {0, 1}):
        raise ValueError(f"invalid scores or labels: {path}")
    data = data.copy()
    data["id_group"] = data.sequence_id.fillna("").where(
        data.sequence_id.fillna("").ne(""), data.video_id)
    data["individual_video"] = data.frame_path.str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[0]
    if data.id_group.isna().any() or data.id_group.eq("").any():
        raise ValueError(f"empty analysis-unit identifier: {path}")
    return data


def aggregate(data: pd.DataFrame, key: str) -> pd.DataFrame:
    grouped = data.groupby(key, sort=True).agg(
        label=("label", "first"), score=("fake_probability", "mean"),
        labels=("label", "nunique"))
    if not grouped.labels.eq(1).all():
        raise ValueError(f"inconsistent labels in {key}")
    return grouped


def auc(data: pd.DataFrame) -> float:
    return float(roc_auc_score(data.label, data.score))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no results for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_sensitivity(source_root: Path, output: Path) -> None:
    rows, summaries = [], []
    for cohort, family, mapping in (
        ("source_selection", "five_candidate", CANDIDATES),
        ("exact_match", "four_way", PRIMARY),
    ):
        for config, label in mapping.items():
            seed_rows = []
            for seed in SEEDS:
                path = source_root / family / "evaluations" / f"{config}__seed{seed}" / "frame_predictions.parquet"
                data = frame_table(path, "FaceForensics++")
                ids, videos = aggregate(data, "id_group"), aggregate(data, "individual_video")
                if (len(data), len(ids), len(videos), int(videos.label.sum())) != (22354, 280, 700, 560):
                    raise ValueError(f"unexpected source population: {path}")
                if not data.split.eq("val").all():
                    raise ValueError(f"source sensitivity requires validation predictions: {path}")
                row = dict(cohort=cohort, configuration=label, config_id=config, seed=seed,
                           frames=len(data), id_groups=len(ids), individual_videos=len(videos),
                           real_videos=int(videos.label.eq(0).sum()), fake_videos=int(videos.label.eq(1).sum()),
                           source_frame_auc=float(roc_auc_score(data.label, data.fake_probability)),
                           source_id_group_auc=auc(ids), source_individual_video_auc=auc(videos))
                seed_rows.append(row)
                rows.append(row)
            summary = dict(cohort=cohort, configuration=label, config_id=config, n_seeds=3)
            for metric in ("source_frame_auc", "source_id_group_auc", "source_individual_video_auc"):
                values = [r[metric] for r in seed_rows]
                summary[metric + "_mean"] = statistics.mean(values)
                summary[metric + "_sd"] = statistics.stdev(values)
            summaries.append(summary)
    write_csv(output / "source_per_video_auc.csv", rows)
    write_csv(output / "source_per_video_auc_summary.csv", summaries)
    print(f"SOURCE_PER_VIDEO_SENSITIVITY_WRITTEN seed_rows={len(rows)} summaries={len(summaries)}")


def dfdcp_sensitivity(target_root: Path, output: Path) -> None:
    rows = []
    for config, label in PRIMARY.items():
        for seed in SEEDS:
            path = target_root / f"{config}__seed{seed}" / "frame_predictions.parquet"
            data = frame_table(path, "DFDCP")
            ids, videos = aggregate(data, "id_group"), aggregate(data, "individual_video")
            if (len(data), len(ids), len(videos), int(videos.label.sum())) != (17222, 652, 654, 424):
                raise ValueError(f"unexpected DFDCP population: {path}")
            primary_auc, video_auc = auc(ids), auc(videos)
            rows.append(dict(placement=label, candidate=config, seed=seed, frames=len(data),
                             primary_id_groups=len(ids), individual_videos=len(videos),
                             primary_id_group_auc=primary_auc, individual_video_auc=video_auc,
                             auc_change=video_auc - primary_auc))
    write_csv(output / "dfdcp_video_sensitivity.csv", rows)
    print(f"DFDCP_VIDEO_SENSITIVITY_WRITTEN checkpoints={len(rows)} "
          f"max_absolute_auc_change={max(abs(r['auc_change']) for r in rows):.15g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        help="Private run root with four_way/evaluations and five_candidate/evaluations")
    parser.add_argument("--target-root", type=Path, help="Private target evaluation run directories")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.source_root is None and args.target_root is None:
        parser.error("at least one of --source-root and --target-root is required")
    if args.source_root is not None:
        source_sensitivity(args.source_root, args.output)
    if args.target_root is not None:
        dfdcp_sensitivity(args.target_root, args.output)


if __name__ == "__main__":
    main()
