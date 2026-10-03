#!/usr/bin/env python3
"""Verify the robustness, calibration, and efficiency results.

Recomputed here, without importing the analysis scripts:
  * per-seed dataset AUCs from the prediction archive (rank-based Mann-Whitney),
    then every aggregation in macro_sensitivity.csv and the 18 seed-specific
    family-balanced differences in uncertainty_decomposition.csv;
  * the raw and prevalence-balanced ECE and Brier point values in
    prevalence_balanced_calibration.csv;
  * the eight-round benchmark means, SDs and derived spreads from its 64 rows;
  * the AUC-implied discordant-equivalent counts of source_unit_level_auc.csv.

Checked for consistency but not re-simulated (they are 10,000- and
2,000-replicate bootstraps; rerun the analysis scripts to regenerate them):
the crossed and unit-only intervals and the calibration intervals.

The merge-equivalence and benchmark files are checked against each other and
against the SHA-256 of the released scripts that produced them.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATASETS = ("Celeb-DF-v2", "DFDC", "DFDCP", "UADFV", "DeeperForensics-1.0", "WildDeepfake")
FAMILIES = (("Celeb-DF-v2",), ("DFDC", "DFDCP"), ("UADFV",),
            ("DeeperForensics-1.0",), ("WildDeepfake",))
LABEL = {"attn_out": "Attn-Out", "attn_qv": "Attn-Q/V",
         "mlp": "MLP", "mlp_attn_out": "MLP+Attn-Out"}
TOL = 1e-9

failures: list[str] = []


def check(name: str, passed: bool, detail: object = "") -> None:
    if passed:
        print(f"PASS {name}")
    else:
        failures.append(f"{name}: {detail}")
        print(f"FAIL {name}: {detail}")


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    ranks = rankdata(scores, method="average")
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = labels.size - n_pos
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ece(labels: np.ndarray, probs: np.ndarray, weights: np.ndarray, bins: int = 15) -> float:
    index = np.clip(np.digitize(probs, np.linspace(0.0, 1.0, bins + 1)[1:-1]), 0, bins - 1)
    weights = weights / weights.sum()
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            w = weights[mask]
            total += w.sum() * abs(np.average(labels[mask], weights=w) - np.average(probs[mask], weights=w))
    return float(total)


def load_archive() -> dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]]:
    frame = pq.read_table(
        RESULTS / "predictions" / "analysis_unit_scores.parquet",
        columns=["dataset", "placement", "seed", "label", "score"],
    ).to_pandas()
    cells = {}
    for (placement, seed, dataset), group in frame.groupby(["placement", "seed", "dataset"]):
        cells[(placement, int(seed), dataset)] = (
            group["label"].to_numpy(dtype=np.int64), group["score"].to_numpy(dtype=np.float64)
        )
    return cells


def verify_aggregations(cells) -> None:
    seeds = sorted({key[1] for key in cells})
    dataset_auc = {key: auc(*value) for key, value in cells.items()}

    def macro(placement: str, seed: int, subset) -> float:
        return statistics.fmean(dataset_auc[(placement, seed, d)] for d in subset)

    def family(placement: str, seed: int) -> float:
        return statistics.fmean(macro(placement, seed, fam) for fam in FAMILIES)

    definitions = {
        "macro_6_all_datasets": lambda p, s: macro(p, s, DATASETS),
        "macro_5_excluding_DeeperForensics-1.0":
            lambda p, s: macro(p, s, [d for d in DATASETS if d != "DeeperForensics-1.0"]),
        "macro_5_excluding_DFDCP": lambda p, s: macro(p, s, [d for d in DATASETS if d != "DFDCP"]),
        "family_balanced_DFDC_DFDCP_merged": family,
    }
    for dropped in ("Celeb-DF-v2", "DFDC", "UADFV", "WildDeepfake"):
        definitions[f"leave_one_out_without_{dropped}"] = (
            lambda p, s, x=dropped: macro(p, s, [d for d in DATASETS if d != x])
        )

    rows = read_csv("macro_sensitivity.csv")
    check("macro sensitivity has 32 rows and 8 distinct aggregations",
          len(rows) == 32 and {r["variant"] for r in rows} == set(definitions),
          (len(rows), sorted({r["variant"] for r in rows})))
    worst = 0.0
    for row in rows:
        placement = next(k for k, v in LABEL.items() if v == row["placement"])
        values = [definitions[row["variant"]](placement, s) for s in seeds]
        worst = max(worst, abs(statistics.fmean(values) - float(row["mean"])),
                    abs(statistics.stdev(values) - float(row["sample_sd"])))
    check(f"macro sensitivity recomputed from predictions (max abs diff {worst:.1e})", worst <= TOL, worst)
    leaders = {v: max((r for r in rows if r["variant"] == v), key=lambda r: float(r["mean"]))["placement"]
               for v in definitions}
    check("Attn-Q/V is the numerical leader under all eight aggregations",
          set(leaders.values()) == {"Attn-Q/V"}, leaders)

    decomposition = {(r["first"], r["second"]): r for r in read_csv("uncertainty_decomposition.csv")}
    crossed = {(r["first"], r["second"]): r for r in read_csv("hierarchical_bootstrap_crossed.csv")}
    pairs = {(LABEL[a], LABEL[b]) for a, b in itertools.combinations(LABEL, 2)}
    check("decomposition and crossed files cover the same six pairs",
          set(decomposition) == pairs and set(crossed) == pairs,
          (sorted(decomposition), sorted(crossed)))
    worst = 0.0
    signs_ok = True
    for (first, second), row in decomposition.items():
        a = next(k for k, v in LABEL.items() if v == first)
        b = next(k for k, v in LABEL.items() if v == second)
        differences = [family(a, s) - family(b, s) for s in seeds]
        for seed, value in zip(seeds, differences):
            worst = max(worst, abs(value - float(row[f"seed_{seed}_difference"])))
        worst = max(worst, abs(statistics.fmean(differences) - float(row["point_estimate"])),
                    abs(float(row["point_estimate"]) - float(crossed[(first, second)]["point_estimate"])))
        consistent = all(d > 0 for d in differences) or all(d < 0 for d in differences)
        signs_ok &= consistent == (row["signs_consistent_across_seeds"] == "true")
    check(f"18 seed-specific family-balanced differences and 6 point estimates (max abs diff {worst:.1e})",
          worst <= TOL, worst)
    check("sign-consistency flags recomputed", signs_ok)
    check("all six unit-only intervals include zero",
          all(float(r["unit_ci_low"]) <= 0.0 <= float(r["unit_ci_high"]) for r in decomposition.values()))
    check("all six crossed intervals include zero",
          all(float(r["ci_low"]) <= 0.0 <= float(r["ci_high"]) for r in crossed.values()))


def verify_calibration(cells) -> None:
    rows = read_csv("prevalence_balanced_calibration.csv")
    check("calibration table has 24 placement-dataset rows", len(rows) == 24, len(rows))
    seeds = sorted({key[1] for key in cells})
    worst = 0.0
    for row in rows:
        placement = next(k for k, v in LABEL.items() if v == row["placement"])
        values = defaultdict(list)
        for seed in seeds:
            y, p = cells[(placement, seed, row["dataset"])]
            real, fake = y == 0, y == 1
            balanced = np.where(real, 0.5 / real.sum(), 0.5 / fake.sum())
            values["ece_raw_mean"].append(ece(y, p, np.ones_like(p)))
            values["ece_prevalence_balanced_mean"].append(ece(y, p, balanced))
            values["brier_raw_mean"].append(float(np.mean((p - y) ** 2)))
            values["brier_class_balanced_mean"].append(
                0.5 * (float(np.mean(p[real] ** 2)) + float(np.mean((1.0 - p[fake]) ** 2))))
            counts = (int(real.sum()), int(fake.sum()))
        if counts != (int(row["real_units"]), int(row["fake_units"])):
            failures.append(f"class counts differ for {row['placement']} {row['dataset']}")
        for column, series in values.items():
            worst = max(worst, abs(statistics.fmean(series) - float(row[column])))
    check(f"raw and balanced ECE/Brier point values recomputed (max abs diff {worst:.1e})", worst <= TOL, worst)
    df10 = {(r["real_units"], r["fake_units"]) for r in rows if r["dataset"] == "DeeperForensics-1.0"}
    check("DeeperForensics-1.0 has 40,669 real and 201 fake units", df10 == {("40669", "201")}, df10)


def verify_source_units() -> None:
    expected = {
        "sel_attn_qv_r16_alpha64_p025": 2, "sel_attn_out_r16_alpha64_p025": 3,
        "sel_mlp_r4_alpha16_p025": 4, "sel_mlp_attn_out_r4_alpha16_p025": 7,
        "sel_attn_out_r16_alpha64_p010": 20,
        "std_mlp_r6_alpha24_p025": 3, "std_mlp_attn_out_r5_alpha20_p025": 5,
        "std_attn_qv_r15_alpha60_p025": 7, "std_attn_out_r30_alpha120_p025": 13,
    }
    rows = {r["config_id"]: r for r in read_csv("source_unit_level_auc.csv")}
    # 140 real x 140 fake validation units per seed, three seeds.
    counts = {k: 3 * 140 * 140 * (1.0 - float(r["source_unit_auc_mean"])) for k, r in rows.items()}
    check("nine source configurations present", set(rows) == set(expected), sorted(rows))
    check("AUC-implied discordant-equivalent counts out of 58,800",
          all(abs(counts[k] - v) < 1e-6 for k, v in expected.items()), counts)


def verify_benchmark() -> None:
    rows = read_csv("merged_inference_benchmark.csv")
    meta = json.loads((RESULTS / "merged_inference_benchmark.meta.json").read_text(encoding="utf-8"))
    keys = [(int(r["round"]), r["position"], r["state"]) for r in rows]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["placement"], r["state"])].append(r)
    check("64 unique round/condition measurements, eight per condition",
          len(rows) == 64 and len(set(keys)) == 64 and all(len(v) == 8 for v in groups.values())
          and len(groups) == 8, (len(rows), len(set(keys)), {k: len(v) for k, v in groups.items()}))
    check("every condition occupies each ordinal position once",
          all(sorted(int(r["order"]) for r in v) == list(range(1, 9)) for v in groups.values()))

    expected = {
        ("Attn-Out", "unmerged"): ("8.0103", "0.1276", "188.952", "0.516"),
        ("Attn-Out", "merged"): ("7.6516", "0.0191", "189.439", "0.222"),
        ("Attn-Q/V", "unmerged"): ("10.4798", "0.0775", "186.597", "0.174"),
        ("Attn-Q/V", "merged"): ("7.6657", "0.0181", "189.414", "0.414"),
        ("MLP", "unmerged"): ("11.0649", "0.1546", "182.246", "0.362"),
        ("MLP", "merged"): ("7.7069", "0.0078", "188.303", "0.151"),
        ("MLP+Attn-Out", "unmerged"): ("12.4554", "0.0437", "181.103", "0.128"),
        ("MLP+Attn-Out", "merged"): ("7.7970", "0.1946", "188.133", "0.276"),
    }
    observed, means, medians = {}, {}, {}
    for key, group in groups.items():
        latency = [float(r["batch1_latency_ms"]) for r in group]
        fps = [float(r["batch32_fps"]) for r in group]
        observed[key] = (f"{statistics.fmean(latency):.4f}", f"{statistics.stdev(latency):.4f}",
                         f"{statistics.fmean(fps):.3f}", f"{statistics.stdev(fps):.3f}")
        means[key], medians[key] = statistics.fmean(latency), statistics.median(latency)
    check("eight-round means and SDs recomputed from the 64 measurements", observed == expected,
          {k: v for k, v in observed.items() if v != expected.get(k)})

    def spread(values: dict, state: str) -> tuple[str, str]:
        subset = [v for (_, s), v in values.items() if s == state]
        low, high = min(subset), max(subset)
        return f"{high - low:.4f}", f"{100 * (high - low) / low:.1f}"
    derived = (spread(means, "unmerged"), spread(means, "merged"), spread(medians, "merged"))
    check("derived spreads: unmerged 4.4451 ms (55.5%), merged 0.1454 ms (1.9%), merged median 0.0689 ms (0.9%)",
          derived == (("4.4451", "55.5"), ("0.1454", "1.9"), ("0.0689", "0.9")), derived)
    reduction = 100 * (means[("Attn-Q/V", "unmerged")] - means[("Attn-Out", "unmerged")]) / means[("Attn-Q/V", "unmerged")]
    check("eight-round unmerged Attn-Out vs Attn-Q/V reduction is 23.56%", f"{reduction:.2f}" == "23.56", reduction)
    efficiency = {r["placement"]: float(r["batch1_ms_mean"]) for r in read_csv("efficiency.csv")}
    five_round = 100 * (efficiency["attn_qv"] - efficiency["attn_out"]) / efficiency["attn_qv"]
    check("five-round unmerged reduction is 23.82% (reported as 23.8%)", f"{five_round:.2f}" == "23.82", five_round)

    row_hashes = {r["position"]: r["checkpoint_sha256"] for r in rows}
    check("benchmark rows use the checkpoints recorded in its metadata",
          row_hashes == meta["checkpoint_sha256"], row_hashes)
    check("benchmark was reporting-only", meta.get("prediction_or_target_metrics_used") is False)

    for name, torch_version, devices, released_script in (
        ("merged_equivalence_seed3407_cuda.json", "2.2.2+cu121", ("cuda:0",), True),
    ):
        data = json.loads((RESULTS / name).read_text(encoding="utf-8"))
        items = {c["position"]: c for c in data["checks"]}
        check(f"{name}: environment and four placements",
              data["status"] == "VALIDATED" and data["torch"] == torch_version
              and data.get("device") in devices and set(items) == set(LABEL),
              (data["torch"], data.get("device"), sorted(items)))
        check(f"{name}: logits and features within tolerance",
              all(c["max_abs_logit_difference"] <= c["tolerance"]
                  and c["max_abs_feature_difference"] <= c["tolerance"] for c in items.values()))
        check(f"{name}: Attn-Q/V key-projection rows unchanged",
              items["attn_qv"]["max_abs_key_projection_drift"] == 0.0)
        check(f"{name}: same checkpoints as the benchmark",
              {p: c["checkpoint_sha256"] for p, c in items.items()} == meta["checkpoint_sha256"])
        if released_script:
            check(f"{name}: produced by the released equivalence test",
                  data["script_sha256"] == sha256(ROOT / "tests" / "test_merged_equivalence.py"),
                  data["script_sha256"])


def main() -> None:
    cells = load_archive()
    verify_aggregations(cells)
    verify_calibration(cells)
    verify_source_units()
    verify_benchmark()
    if failures:
        for failure in failures:
            print(f"BLOCKER {failure}")
        raise SystemExit(1)
    print("ROBUSTNESS_RESULTS_VERIFIED")


if __name__ == "__main__":
    main()
