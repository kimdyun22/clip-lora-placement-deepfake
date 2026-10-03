#!/usr/bin/env python3
"""Aggregation sensitivity for the reported cross-dataset average.

Recomputes the target average under alternatives to the unweighted Macro-6:
excluding DeeperForensics-1.0, treating DFDC and DFDCP as one benchmark family,
and leaving out each dataset in turn. Reads only the released prediction
archive; no training or inference is performed.
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results" / "predictions" / "analysis_unit_scores.parquet"

DATASETS = ("Celeb-DF-v2", "DFDC", "DFDCP", "UADFV", "DeeperForensics-1.0", "WildDeepfake")
PLACEMENT = {"attn_out": "Attn-Out", "attn_qv": "Attn-Q/V",
             "mlp": "MLP", "mlp_attn_out": "MLP+Attn-Out"}
# DFDC and DFDCP come from the same challenge; grouping them prevents that
# benchmark from carrying two of six equal weights.
FAMILIES = (("Celeb-DF-v2",), ("DFDC", "DFDCP"), ("UADFV",),
            ("DeeperForensics-1.0",), ("WildDeepfake",))


def dataset_auc(path: Path) -> dict[tuple[str, int, str], float]:
    table = pq.read_table(path, columns=["dataset", "placement", "seed", "label", "score"])
    col = {name: table[name].to_pylist() for name in table.column_names}
    buckets: dict[tuple[str, int, str], tuple[list[int], list[float]]] = {}
    for ds, pl, seed, label, score in zip(
        col["dataset"], col["placement"], col["seed"], col["label"], col["score"]
    ):
        labels, scores = buckets.setdefault((pl, int(seed), ds), ([], []))
        labels.append(int(label))
        scores.append(float(score))
    return {key: roc_auc_score(lab, sc) for key, (lab, sc) in buckets.items()}


def summarise(values: list[float]) -> tuple[float, float]:
    return st.fmean(values), st.stdev(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "macro_sensitivity.csv")
    args = parser.parse_args()

    auc = dataset_auc(ARCHIVE)
    seeds = sorted({key[1] for key in auc})
    rows: list[dict[str, object]] = []

    for placement, name in PLACEMENT.items():
        def per_seed(subset: tuple[str, ...]) -> list[float]:
            return [st.fmean(auc[(placement, s, d)] for d in subset) for s in seeds]

        def family_per_seed() -> list[float]:
            return [
                st.fmean([st.fmean(auc[(placement, s, d)] for d in fam) for fam in FAMILIES])
                for s in seeds
            ]

        variants: list[tuple[str, list[float]]] = [
            ("macro_6_all_datasets", per_seed(DATASETS)),
            ("macro_5_excluding_DeeperForensics-1.0",
             per_seed(tuple(d for d in DATASETS if d != "DeeperForensics-1.0"))),
            ("macro_5_excluding_DFDCP",
             per_seed(tuple(d for d in DATASETS if d != "DFDCP"))),
            ("family_balanced_DFDC_DFDCP_merged", family_per_seed()),
        ]
        for dropped in DATASETS:
            # These two leave-one-out definitions are already represented by
            # the explicitly named Macro-5 rows above.
            if dropped in ("DeeperForensics-1.0", "DFDCP"):
                continue
            variants.append((
                f"leave_one_out_without_{dropped}",
                per_seed(tuple(d for d in DATASETS if d != dropped)),
            ))

        for variant, values in variants:
            mean, sd = summarise(values)
            rows.append({
                "placement": name,
                "variant": variant,
                "n_seeds": len(seeds),
                "mean": repr(mean),
                "sample_sd": repr(sd),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    variants = sorted({row["variant"] for row in rows})
    order = [name for name in PLACEMENT.values()]
    print(f"{'variant':44s} " + " ".join(f"{n:>14s}" for n in order) + "   leader")
    for variant in variants:
        cells = {row["placement"]: float(row["mean"]) for row in rows if row["variant"] == variant}
        leader = max(cells, key=cells.get)
        print(f"{variant:44s} " + " ".join(f"{cells[n]:14.6f}" for n in order) + f"   {leader}")
    try:
        display_path = args.output.relative_to(ROOT)
    except ValueError:
        display_path = args.output
    print(f"\nwrote {display_path} ({len(rows)} rows)")
    print("MACRO_SENSITIVITY_DONE")


if __name__ == "__main__":
    main()
