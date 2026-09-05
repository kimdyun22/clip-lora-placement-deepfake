"""Validate and summarize the canonical machine-readable result tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


REQUIRED_RESULTS = {
    "primary_results.csv": {"placement", "source_auc_mean", "macro_6_auc_mean"},
    "primary_seed_results.csv": {"placement", "seed", "source_frame_auc", "macro_6_auc"},
    "adaptation_baselines.csv": {"method", "seeds", "source_auc_mean", "macro_6_auc_mean"},
    "adaptation_baseline_seed_results.csv": {"method", "seed", "source_frame_auc", "macro_6_auc"},
    "selection_sensitivity.csv": {
        "candidate", "role", "source_auc_mean", "macro_6_auc_mean", "macro_6_auc_sd"
    },
    "selection_seed_results.csv": {"candidate", "role", "seed", "source_frame_auc"},
    "lora_sbi_2x2.csv": {"cohort", "condition", "seed", "source_auc"},
    "statistics.csv": {"seed", "dataset", "first", "second", "delong_p_raw"},
    "bootstrap_ci_summary.csv": {"pair_id", "dataset", "n_seeds", "mean_auc_delta", "ci_includes_zero"},
    "calibration_thresholds.csv": {"record_type", "candidate", "seed", "dataset"},
    "operating_point_summary.csv": {"placement", "dataset", "metric", "mean", "sample_sd"},
    "failure_cases.csv": {"case_id", "candidate", "seed", "dataset", "error_category"},
    "efficiency.csv": {"placement", "batch1_ms_mean", "batch32_fps_mean"},
    "efficiency_round_results.csv": {"placement", "round", "batch1_ms", "batch32_fps"},
    "external_reproduction_audit.csv": {
        "method", "released_checkpoint_sha256", "dataset",
        "published_auc", "reproduced_auc", "disposition",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate(results_dir: Path, protocol: Path) -> dict[str, int]:
    config = yaml.safe_load(protocol.read_text(encoding="utf-8"))
    if config.get("status") != "frozen":
        raise ValueError("analysis protocol must have status: frozen")
    counts: dict[str, int] = {}
    for name, required in REQUIRED_RESULTS.items():
        path = results_dir / name
        rows = read_csv(path)
        if not rows:
            raise ValueError(f"empty result file: {path}")
        missing = required - set(rows[0])
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        counts[name] = len(rows)
    primary = read_csv(results_dir / "primary_results.csv")
    if {row["placement"] for row in primary} != {
        "Attn-Out", "Attn-Q/V", "MLP", "MLP+Attn-Out"
    }:
        raise ValueError("primary table does not contain the four frozen placements")
    if {int(row["trainable_params"]) for row in primary} != {1_478_658}:
        raise ValueError("primary placements are not exactly parameter matched")
    for row in read_csv(results_dir / "external_reproduction_audit.csv"):
        published = float(row["published_auc"])
        reproduced = float(row["reproduced_auc"])
        if abs((reproduced - published) - float(row["delta"])) > 1e-9:
            raise ValueError(
                f"external audit delta inconsistent for {row['method']} / {row['dataset']}"
            )
        # The published reference is a reproducibility sanity check, so every
        # released row must actually agree with its published value.
        if round(published, 3) != round(reproduced, 3):
            raise ValueError(
                f"external audit row does not reproduce at three decimals: "
                f"{row['method']} / {row['dataset']} "
                f"(published {published}, local {reproduced})"
            )
        if row["disposition"] != "agreement_at_three_decimals":
            raise ValueError(
                f"external audit disposition contradicts its own values for "
                f"{row['method']} / {row['dataset']}: "
                f"recorded {row['disposition']!r}"
            )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("protocols/analysis.yaml"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    counts = validate(args.results, args.config)
    for name, count in counts.items():
        print(f"{name}: {count} rows")
    print("RESULT_ARTIFACTS_VALIDATED")


if __name__ == "__main__":
    main()
