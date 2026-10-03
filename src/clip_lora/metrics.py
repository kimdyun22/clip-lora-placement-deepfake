#!/usr/bin/env python3
"""Validation, aggregation, and metric helpers for evaluation artifacts.

The raw artifact is keyed by the stable ``sample_id`` from a frozen manifest.
This module deliberately contains no model code so its invariants can be tested
on CPU without importing CUDA or DeepfakeBench.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


REQUIRED_MANIFEST_FIELDS = {
    "dataset",
    "split",
    "sample_id",
    "video_id",
    "sequence_id",
    "frame_path",
    "frame_index",
    "label",
}

REQUIRED_PREDICTION_FIELDS = REQUIRED_MANIFEST_FIELDS | {
    "fake_logit",
    "real_logit",
    "fake_probability",
}


class ArtifactError(ValueError):
    """Raised when an evaluation artifact violates a declared invariant."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ArtifactError(f"{path}:{line_number}: row is not an object")
            yield row


def analysis_unit_id(row: Mapping[str, Any]) -> str:
    unit_id = str(row.get("sequence_id") or row.get("video_id") or "")
    if not unit_id:
        raise ArtifactError(f"sample {row.get('sample_id', '<unknown>')} has no analysis-unit ID")
    return unit_id


def validate_manifest_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    sample_ids: set[str] = set()
    frame_paths: set[str] = set()
    labels_by_unit: dict[tuple[str, str], set[int]] = defaultdict(set)
    datasets: set[str] = set()
    count = 0

    for row in rows:
        missing = REQUIRED_MANIFEST_FIELDS - set(row)
        if missing:
            raise ArtifactError(f"manifest row lacks fields: {sorted(missing)}")
        sample_id = str(row["sample_id"])
        frame_path = str(row["frame_path"])
        if not sample_id or sample_id in sample_ids:
            raise ArtifactError(f"empty or duplicate sample_id: {sample_id!r}")
        if not frame_path or frame_path in frame_paths:
            raise ArtifactError(f"empty or duplicate frame_path: {frame_path!r}")
        if not os.path.isfile(frame_path):
            raise ArtifactError(f"frame file is missing: {frame_path}")
        label = int(row["label"])
        if label not in (0, 1):
            raise ArtifactError(f"label must be binary, got {label}")
        dataset = str(row["dataset"])
        unit_key = (dataset, analysis_unit_id(row))
        labels_by_unit[unit_key].add(label)
        sample_ids.add(sample_id)
        frame_paths.add(frame_path)
        datasets.add(dataset)
        count += 1

    if count == 0:
        raise ArtifactError("manifest contains no samples")
    inconsistent = [key for key, labels in labels_by_unit.items() if len(labels) != 1]
    if inconsistent:
        raise ArtifactError(f"analysis-unit labels are inconsistent: {inconsistent[:5]}")
    return {
        "frames": count,
        "analysis_units": len(labels_by_unit),
        "datasets": sorted(datasets),
    }


def validate_prediction_row(row: Mapping[str, Any]) -> None:
    missing = REQUIRED_PREDICTION_FIELDS - set(row)
    if missing:
        raise ArtifactError(f"prediction row lacks fields: {sorted(missing)}")
    probability = float(row["fake_probability"])
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ArtifactError(f"invalid fake_probability: {probability}")
    for field in ("fake_logit", "real_logit"):
        if not math.isfinite(float(row[field])):
            raise ArtifactError(f"invalid {field}: {row[field]}")


def aggregate_prediction_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    seen_sample_ids: set[str] = set()
    for row in rows:
        validate_prediction_row(row)
        sample_id = str(row["sample_id"])
        if sample_id in seen_sample_ids:
            raise ArtifactError(f"duplicate prediction sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)
        dataset = str(row["dataset"])
        unit_id = analysis_unit_id(row)
        key = (dataset, unit_id)
        if key not in groups:
            groups[key] = {
                "dataset": dataset,
                "split": str(row["split"]),
                "analysis_unit_id": unit_id,
                "unit_kind": "sequence" if row.get("sequence_id") else "video",
                "label": int(row["label"]),
                "probability_sum": 0.0,
                "frame_count": 0,
            }
        group = groups[key]
        if int(row["label"]) != group["label"]:
            raise ArtifactError(f"inconsistent labels in analysis unit {key}")
        group["probability_sum"] += float(row["fake_probability"])
        group["frame_count"] += 1

    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        output.append(
            {
                "dataset": group["dataset"],
                "split": group["split"],
                "analysis_unit_id": group["analysis_unit_id"],
                "unit_kind": group["unit_kind"],
                "label": group["label"],
                "fake_probability": group["probability_sum"] / group["frame_count"],
                "frame_count": group["frame_count"],
            }
        )
    return output


def eer_score(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if np.unique(y).size != 2:
        raise ArtifactError("EER requires both classes")
    fpr, tpr, _ = roc_curve(y, p)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def binary_metrics(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape:
        raise ArtifactError(f"label/probability shape mismatch: {y.shape} vs {p.shape}")
    if y.size == 0 or np.unique(y).size != 2:
        raise ArtifactError("binary metrics require non-empty samples from both classes")
    return {
        "n": int(y.size),
        "real": int(np.sum(y == 0)),
        "fake": int(np.sum(y == 1)),
        "auc": float(roc_auc_score(y, p)),
        "eer": eer_score(y, p),
        "ap": float(average_precision_score(y, p)),
        "accuracy_at_0_5": float(np.mean((p >= 0.5).astype(np.int64) == y)),
    }


def metrics_from_rows(
    frame_rows: Sequence[Mapping[str, Any]],
    unit_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    datasets = sorted({str(row["dataset"]) for row in frame_rows})
    result: dict[str, Any] = {"datasets": {}}
    for dataset in datasets:
        frames = [row for row in frame_rows if row["dataset"] == dataset]
        units = [row for row in unit_rows if row["dataset"] == dataset]
        result["datasets"][dataset] = {
            "frame": binary_metrics(
                [int(row["label"]) for row in frames],
                [float(row["fake_probability"]) for row in frames],
            ),
            "analysis_unit": binary_metrics(
                [int(row["label"]) for row in units],
                [float(row["fake_probability"]) for row in units],
            ),
        }
    result["macro_analysis_unit_auc"] = float(
        np.mean(
            [
                result["datasets"][dataset]["analysis_unit"]["auc"]
                for dataset in datasets
            ]
        )
    )
    return result


def compare_artifact_indexes(
    left_rows: Iterable[Mapping[str, Any]],
    right_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    def build_index(rows: Iterable[Mapping[str, Any]]) -> dict[str, tuple[Any, ...]]:
        index: dict[str, tuple[Any, ...]] = {}
        for row in rows:
            sample_id = str(row["sample_id"])
            value = (
                str(row["dataset"]),
                str(row["frame_path"]),
                analysis_unit_id(row),
                int(row["label"]),
            )
            if sample_id in index:
                raise ArtifactError(f"duplicate sample_id in artifact: {sample_id}")
            index[sample_id] = value
        return index

    left = build_index(left_rows)
    right = build_index(right_rows)
    missing_from_right = sorted(set(left) - set(right))
    missing_from_left = sorted(set(right) - set(left))
    mismatched = sorted(key for key in set(left) & set(right) if left[key] != right[key])
    return {
        "equal": not missing_from_right and not missing_from_left and not mismatched,
        "left_count": len(left),
        "right_count": len(right),
        "missing_from_right_count": len(missing_from_right),
        "missing_from_left_count": len(missing_from_left),
        "metadata_mismatch_count": len(mismatched),
        "missing_from_right_examples": missing_from_right[:10],
        "missing_from_left_examples": missing_from_left[:10],
        "metadata_mismatch_examples": mismatched[:10],
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
def midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    output = np.empty(values.size, dtype=np.float64)
    output[order] = ranks
    return output


def fast_delong(predictions: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    model_count, sample_count = predictions.shape
    negative_count = sample_count - positive_count
    positive = predictions[:, :positive_count]
    negative = predictions[:, positive_count:]
    tx = np.empty((model_count, positive_count), dtype=np.float64)
    ty = np.empty((model_count, negative_count), dtype=np.float64)
    tz = np.empty((model_count, sample_count), dtype=np.float64)
    for model in range(model_count):
        tx[model] = midrank(positive[model])
        ty[model] = midrank(negative[model])
        tz[model] = midrank(predictions[model])
    aucs = tz[:, :positive_count].sum(axis=1) / (positive_count * negative_count)
    aucs -= (positive_count + 1.0) / (2.0 * negative_count)
    v01 = (tz[:, :positive_count] - tx) / negative_count
    v10 = 1.0 - (tz[:, positive_count:] - ty) / positive_count
    covariance = np.cov(v01) / positive_count + np.cov(v10) / negative_count
    return aucs, np.atleast_2d(covariance)


def paired_delong(labels: np.ndarray, primary: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    order = np.argsort(-labels, kind="stable")
    positive_count = int(labels.sum())
    predictions = np.vstack((primary[order], baseline[order]))
    aucs, covariance = fast_delong(predictions, positive_count)
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast)
    delta = float(aucs[0] - aucs[1])
    if variance <= 0.0:
        z = math.inf if delta != 0 else 0.0
        p_value = 0.0 if delta != 0 else 1.0
    else:
        z = delta / math.sqrt(variance)
        p_value = float(2.0 * norm.sf(abs(z)))
    return {
        "primary_auc": float(aucs[0]),
        "baseline_auc": float(aucs[1]),
        "auc_delta_primary_minus_baseline": delta,
        "variance": variance,
        "z_statistic": float(z),
        "raw_p_value": p_value,
    }


def score_plan(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.concatenate(
        (np.array([0], dtype=np.int64), np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1)
    )
    return order, starts


def paired_stratified_bootstrap(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    replicates: int,
    rng: np.random.Generator,
    chunk_size: int = 32,
) -> dict[str, np.ndarray]:
    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)
    positive_count = positive_indices.size
    negative_count = negative_indices.size
    if not positive_count or not negative_count:
        raise ArtifactError("stratified bootstrap requires both labels")
    positive_probability = np.full(positive_count, 1.0 / positive_count)
    negative_probability = np.full(negative_count, 1.0 / negative_count)
    plans = {name: score_plan(value) for name, value in scores.items()}
    output = {name: np.empty(replicates, dtype=np.float64) for name in scores}
    labels_float = labels.astype(np.float64)

    for offset in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - offset)
        positive_weights = rng.multinomial(
            positive_count, positive_probability, size=size
        ).astype(np.float64)
        negative_weights = rng.multinomial(
            negative_count, negative_probability, size=size
        ).astype(np.float64)
        weights = np.zeros((size, labels.size), dtype=np.float64)
        weights[:, positive_indices] = positive_weights
        weights[:, negative_indices] = negative_weights
        weighted_positive = weights * labels_float
        weighted_negative = weights * (1.0 - labels_float)
        for name, (order, starts) in plans.items():
            positive_group = np.add.reduceat(weighted_positive[:, order], starts, axis=1)
            negative_group = np.add.reduceat(weighted_negative[:, order], starts, axis=1)
            negatives_below = np.cumsum(negative_group, axis=1) - negative_group
            contribution = positive_group * (negatives_below + 0.5 * negative_group)
            output[name][offset : offset + size] = contribution.sum(axis=1) / (
                positive_count * negative_count
            )
    return output


def percentile_interval(values: np.ndarray, confidence_level: float) -> list[float]:
    tail = (1.0 - confidence_level) / 2.0
    return [
        float(np.quantile(values, tail, method="linear")),
        float(np.quantile(values, 1.0 - tail, method="linear")),
    ]


def holm_adjust(p_values: dict[str, float], alpha: float) -> dict[str, dict[str, Any]]:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    count = len(ordered)
    running = 0.0
    adjusted: dict[str, float] = {}
    for rank, key in enumerate(ordered):
        value = min(1.0, (count - rank) * float(p_values[key]))
        running = max(running, value)
        adjusted[key] = running
    return {
        key: {
            "raw_p_value": float(p_values[key]),
            "holm_adjusted_p_value": float(adjusted[key]),
            "reject_at_alpha": bool(adjusted[key] <= alpha),
        }
        for key in p_values
    }


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    prediction = probabilities >= threshold
    positive = labels == 1
    negative = labels == 0
    tp = int(np.sum(prediction & positive))
    fp = int(np.sum(prediction & negative))
    tn = int(np.sum(~prediction & negative))
    fn = int(np.sum(~prediction & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "n": int(labels.size),
        "real": int(negative.sum()),
        "fake": int(positive.sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": float((tp + tn) / labels.size),
        "precision": float(precision),
        "recall_tpr": float(recall),
        "f1": float(f1),
        "fpr": float(fp / negative.sum()),
    }


def grouped_descending(labels: np.ndarray, probabilities: np.ndarray):
    order = np.argsort(-probabilities, kind="stable")
    scores = probabilities[order]
    sorted_labels = labels[order]
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and scores[end] == scores[start]:
            end += 1
        group = sorted_labels[start:end]
        yield float(scores[start]), int(np.sum(group == 1)), int(np.sum(group == 0))
        start = end


def select_source_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    if set(np.unique(labels)) != {0, 1}:
        raise ArtifactError("source F1 threshold requires both labels")
    sentinel = float(np.nextafter(float(np.max(probabilities)), math.inf))
    positives = int(np.sum(labels == 1))
    tp = 0
    fp = 0
    best_threshold = sentinel
    best_f1 = 0.0
    for threshold, positive_group, negative_group in grouped_descending(labels, probabilities):
        tp += positive_group
        fp += negative_group
        fn = positives - tp
        denominator = 2 * tp + fp + fn
        f1 = (2.0 * tp / denominator) if denominator else 0.0
        # Candidates are visited highest threshold first, so exact ties retain
        # the predeclared conservative highest-threshold tie breaker.
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    metrics = classification_metrics(labels, probabilities, best_threshold)
    metrics.update(
        {
            "selection_objective": "maximize_source_f1",
            "tie_breaker": "highest_threshold",
        }
    )
    return metrics


def select_source_low_fpr_threshold(
    labels: np.ndarray, probabilities: np.ndarray, fpr_upper_bound: float
) -> dict[str, Any]:
    if set(np.unique(labels)) != {0, 1}:
        raise ArtifactError("source low-FPR threshold requires both labels")
    sentinel = float(np.nextafter(float(np.max(probabilities)), math.inf))
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    tp = 0
    fp = 0
    best_threshold = sentinel
    best_tpr = 0.0
    for threshold, positive_group, negative_group in grouped_descending(labels, probabilities):
        tp += positive_group
        fp += negative_group
        fpr = fp / negatives
        tpr = tp / positives
        if fpr <= fpr_upper_bound and tpr > best_tpr:
            best_tpr = tpr
            best_threshold = threshold
    metrics = classification_metrics(labels, probabilities, best_threshold)
    if metrics["fpr"] > fpr_upper_bound + 1e-15:
        raise ArtifactError("selected source threshold violates its FPR bound")
    metrics.update(
        {
            "source_fpr_upper_bound": float(fpr_upper_bound),
            "selection_objective": "maximize_source_tpr_subject_to_fpr_bound",
            "tie_breaker": "highest_threshold",
        }
    )
    return metrics


def descriptive_eer(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    fpr, tpr, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fnr - fpr)))
    return {
        "eer": float((fpr[index] + fnr[index]) / 2.0),
        "fpr_at_selected_roc_point": float(fpr[index]),
        "fnr_at_selected_roc_point": float(fnr[index]),
        "target_descriptive_threshold": float(thresholds[index]),
    }

