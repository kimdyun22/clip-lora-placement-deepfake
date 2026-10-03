#!/usr/bin/env python3
"""Tabulate source-validation AUC at both frame and analysis-unit level.

The reported source-only selection rule ranks candidates by frame-level AUC,
while external evaluation is analysis-unit level. This builder extracts both
levels from the frozen source evaluations so the selection can be re-checked on
the same unit as the target metric. It reads existing evaluation artifacts; no
inference is run.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXACT = {
    "std_attn_out_r30_alpha120_p025": ("exact_match", "Attn-Out"),
    "std_attn_qv_r15_alpha60_p025": ("exact_match", "Attn-Q/V"),
    "std_mlp_r6_alpha24_p025": ("exact_match", "MLP"),
    "std_mlp_attn_out_r5_alpha20_p025": ("exact_match", "MLP+Attn-Out"),
}
SELECTION = {
    "sel_attn_qv_r16_alpha64_p025": ("source_selection", "Attn-Q/V r16/alpha64"),
    "sel_attn_out_r16_alpha64_p025": ("source_selection", "Attn-Out r16/alpha64"),
    "sel_mlp_r4_alpha16_p025": ("source_selection", "MLP r4/alpha16"),
    "sel_mlp_attn_out_r4_alpha16_p025": ("source_selection", "MLP+Attn-Out r4/alpha16"),
    "sel_attn_out_r16_alpha64_p010": ("source_selection", "Attn-Out r16/alpha64 pSBI=0.10"),
}
SOURCE_DATASET = "FaceForensics++"


def collect(root: Path, mapping: dict[str, tuple[str, str]]) -> dict[str, dict[str, list[float]]]:
    found: dict[str, dict[str, list[float]]] = {}
    for metrics_path in sorted(root.glob("*/metrics.json")):
        run = metrics_path.parent.name
        config = run.rsplit("__seed", 1)[0]
        if config not in mapping:
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        block = payload["datasets"][SOURCE_DATASET]
        entry = found.setdefault(config, {"frame": [], "unit": []})
        entry["frame"].append(float(block["frame"]["auc"]))
        entry["unit"].append(float(block["analysis_unit"]["auc"]))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Directory holding source_eval_four_way/ and source_eval_five_candidate/")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "source_unit_level_auc.csv")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for subdir, mapping in (("source_eval_four_way", EXACT),
                            ("source_eval_five_candidate", SELECTION)):
        root = args.source_root / subdir
        if not root.is_dir():
            raise SystemExit(f"missing evaluation directory: {root}")
        for config, entry in sorted(collect(root, mapping).items()):
            cohort, label = mapping[config]
            if len(entry["frame"]) != 3:
                raise SystemExit(f"{config}: expected three seeds, found {len(entry['frame'])}")
            rows.append({
                "cohort": cohort,
                "configuration": label,
                "config_id": config,
                "n_seeds": len(entry["frame"]),
                "source_frame_auc_mean": repr(st.fmean(entry["frame"])),
                "source_frame_auc_sd": repr(st.stdev(entry["frame"])),
                "source_unit_auc_mean": repr(st.fmean(entry["unit"])),
                "source_unit_auc_sd": repr(st.stdev(entry["unit"])),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for cohort in ("exact_match", "source_selection"):
        subset = [r for r in rows if r["cohort"] == cohort]
        if not subset:
            continue
        print(f"\n[{cohort}]")
        print(f"  {'configuration':34s} {'frame AUC':>12s} {'unit AUC':>12s}")
        for row in sorted(subset, key=lambda r: -float(r["source_unit_auc_mean"])):
            print(f"  {row['configuration']:34s} {float(row['source_frame_auc_mean']):12.6f}"
                  f" {float(row['source_unit_auc_mean']):12.6f}")
        frame_leader = max(subset, key=lambda r: float(r["source_frame_auc_mean"]))["configuration"]
        unit_leader = max(subset, key=lambda r: float(r["source_unit_auc_mean"]))["configuration"]
        agree = "unchanged" if frame_leader == unit_leader else "CHANGED"
        print(f"  frame-level leader: {frame_leader}")
        print(f"  unit-level leader : {unit_leader}   -> selection {agree}")

    print(f"\nwrote {args.output} ({len(rows)} rows)")
    print("SOURCE_UNIT_LEVEL_DONE")


if __name__ == "__main__":
    main()
