#!/usr/bin/env python3
"""Run the non-training provenance and consistency checks for public results."""

from __future__ import annotations

import csv
import gzip
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXPECTED_PARAMETERS = {
    "Attn-Out": (30, 120, 24 * 2 * 1024 * 30 + 4098),
    "Attn-Q/V": (15, 60, 24 * 4 * 1024 * 15 + 4098),
    "MLP": (6, 24, 24 * 2 * (1024 + 4096) * 6 + 4098),
    "MLP+Attn-Out": (5, 20, 24 * (2 * (1024 + 4096) * 5 + 2 * 1024 * 5) + 4098),
}
EXPECTED_PRIMARY_REJECTIONS_BY_PAIR = {
    "std_attn_out_r30_alpha120_p025__minus__std_attn_qv_r15_alpha60_p025": 4,
    "std_attn_out_r30_alpha120_p025__minus__std_mlp_attn_out_r5_alpha20_p025": 2,
    "std_attn_out_r30_alpha120_p025__minus__std_mlp_r6_alpha24_p025": 2,
    "std_attn_qv_r15_alpha60_p025__minus__std_mlp_attn_out_r5_alpha20_p025": 2,
    "std_attn_qv_r15_alpha60_p025__minus__std_mlp_r6_alpha24_p025": 1,
    "std_mlp_r6_alpha24_p025__minus__std_mlp_attn_out_r5_alpha20_p025": 1,
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_verifier(name: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "analysis" / name)], cwd=ROOT, check=True)


def main() -> None:
    failures: list[str] = []

    primary = read_csv("primary_results.csv")
    for row in primary:
        placement = row["placement"]
        if placement not in EXPECTED_PARAMETERS:
            failures.append(f"unknown primary placement: {placement}")
            continue
        rank, alpha, count = EXPECTED_PARAMETERS[placement]
        if (int(row["rank"]), int(row["alpha"]), int(row["trainable_params"])) != (rank, alpha, count):
            failures.append(f"parameter configuration mismatch: {placement}")
        if not math.isclose(float(row["alpha"]) / float(row["rank"]), 4.0):
            failures.append(f"scaling-ratio mismatch: {placement}")
    print("PASS primary parameter counts and frozen rank/alpha configurations")

    statistics_rows = read_csv("statistics.csv")
    primary_rejects = [row for row in statistics_rows if row["holm_primary_reject"].lower() == "true"]
    global_rejects = [row for row in statistics_rows if row["holm_global_108_reject"].lower() == "true"]
    if len(statistics_rows) != 108 or len(primary_rejects) != 12 or len(global_rejects) != 11:
        failures.append(
            f"Holm count mismatch: rows={len(statistics_rows)}, primary={len(primary_rejects)}, global={len(global_rejects)}"
        )
    pair_counts = Counter(row["pair_id"] for row in primary_rejects)
    if pair_counts != Counter(EXPECTED_PRIMARY_REJECTIONS_BY_PAIR):
        failures.append(f"primary Holm pair counts mismatch: {dict(pair_counts)}")
    print(f"PASS Holm counts primary={len(primary_rejects)}/108 global={len(global_rejects)}/108")
    print("INFO primary Holm rejections by pair " + ", ".join(f"{key}={value}" for key, value in sorted(pair_counts.items())))

    sensitivity = read_csv("selection_sensitivity.csv")
    role_counts = Counter(row["role"] for row in sensitivity)
    if len(sensitivity) != 5 or role_counts != Counter({"comparable_candidate": 4, "descriptive_reference": 1}):
        failures.append(f"selection/sensitivity coverage mismatch: {role_counts}")
    for row in sensitivity:
        if not math.isclose(float(row["alpha"]) / float(row["rank"]), 4.0):
            failures.append(f"selection row does not hold alpha/r=4: {row['candidate']}")
    print("PASS source-selection and rank/budget sensitivity coverage (fixed alpha/r=4)")

    two_by_two = read_csv("lora_sbi_2x2.csv")
    source_selection = next(
        row for row in sensitivity
        if row["candidate"] == "Attn-Out r16/alpha64" and row["p_sbi"] == "0.25"
    )
    factorial_auc = sum(
        float(row["source_auc"]) for row in two_by_two if row["condition"] == "Attn-Out LoRA / SBI"
    ) / 3.0
    selection_auc = float(source_selection["source_auc_mean"])
    if math.isclose(factorial_auc, selection_auc, rel_tol=0.0, abs_tol=1e-12):
        failures.append("distinct Attn-Out pSBI=.25 cohorts were incorrectly collapsed")
    print(
        "PASS distinct-cohort provenance retained: "
        f"factorial={factorial_auc:.6f}, source_selection={selection_auc:.6f}"
    )

    for verifier in (
        "verify_primary_results.py",
        "verify_adaptation_baselines.py",
        "verify_selection_sensitivity.py",
        "verify_lora_sbi_2x2.py",
        "verify_operating_point_summary.py",
        "verify_bootstrap_ci_summary.py",
        "verify_prediction_archive.py",
        "verify_prediction_metrics.py",
        "verify_efficiency.py",
    ):
        try:
            run_verifier(verifier)
        except subprocess.CalledProcessError as exc:
            failures.append(f"{verifier} failed with exit code {exc.returncode}")

    forbidden = ("/data/", "/workspace/", "/home/")
    for path in RESULTS.rglob("*.csv"):
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in forbidden):
            failures.append(f"local absolute path in public CSV: {path.relative_to(ROOT)}")
    print("PASS no local absolute paths in public CSV artifacts")

    manifest_dir = ROOT / "datasets" / "manifests"
    manifests = sorted(manifest_dir.glob("*.jsonl.gz")) if manifest_dir.is_dir() else []
    if len(manifests) != 8:
        failures.append(f"expected eight published manifests, found {len(manifests)}")
    for path in manifests:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                frame_path = json.loads(line)["frame_path"]
                if frame_path.startswith("/") or any(token in frame_path for token in forbidden):
                    failures.append(
                        f"absolute frame path in {path.relative_to(ROOT)}:{line_number}"
                    )
                    break
    print(f"PASS published manifests carry dataset-relative frame paths ({len(manifests)} files)")

    if failures:
        for failure in failures:
            print(f"BLOCKER {failure}")
        raise SystemExit(1)
    print("PUBLIC_RESULT_AUDIT_PASS")


if __name__ == "__main__":
    main()
