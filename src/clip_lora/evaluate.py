#!/usr/bin/env python3
"""Evaluate standard-LoRA checkpoints against frozen manifests with stable identities.

Each successful run is an immutable directory containing:

* ``frame_predictions.parquet``: one row per manifest sample;
* ``analysis_unit_predictions.parquet``: mean frame probability per video/sequence;
* ``metrics.json`` and ``run_metadata.json``;
* ``VALIDATED``: written last, only after every invariant and expectation passes.

No dataset object is reconstructed after inference. Labels, paths, unit IDs, and
predictions are emitted together from the same batch.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import pyarrow as pa
import pyarrow.json as pajson
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

from .metrics import (
    ArtifactError,
    analysis_unit_id,
    atomic_write_json,
    binary_metrics,
    sha256_file,
)
from .model import CLIPStandardLoRABackbone


RAW_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("split", pa.string()),
        ("sample_id", pa.string()),
        ("video_id", pa.string()),
        ("sequence_id", pa.string()),
        ("frame_path", pa.string()),
        ("frame_index", pa.int64()),
        ("label", pa.int8()),
        ("real_logit", pa.float64()),
        ("fake_logit", pa.float64()),
        ("fake_probability", pa.float64()),
    ]
)

UNIT_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("split", pa.string()),
        ("analysis_unit_id", pa.string()),
        ("unit_kind", pa.string()),
        ("label", pa.int8()),
        ("fake_probability", pa.float64()),
        ("frame_count", pa.int64()),
    ]
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def git_revision() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def read_manifest_table(path: Path) -> pa.Table:
    """Read a JSONL manifest into Arrow, transparently handling gzip."""
    if path.suffix == ".gz":
        with pa.CompressedInputStream(pa.OSFile(os.fspath(path), "rb"), "gzip") as stream:
            return pajson.read_json(stream).combine_chunks()
    return pajson.read_json(path).combine_chunks()


class ArrowManifestDataset(Dataset):
    """Map-style dataset backed by compact Arrow columns, not Python dictionaries."""

    def __init__(
        self,
        manifest_path: Path,
        frame_loader: Any,
        path_maps: list[tuple[str, str]] | None = None,
        dataset_root: Path | None = None,
    ):
        self.manifest_path = manifest_path
        self.table = read_manifest_table(manifest_path)
        self.frame_loader = frame_loader
        self.path_maps = path_maps or []
        self.dataset_root = dataset_root
        required = {
            "dataset",
            "split",
            "sample_id",
            "video_id",
            "sequence_id",
            "frame_path",
            "frame_index",
            "label",
        }
        missing = required - set(self.table.column_names)
        if missing:
            raise ArtifactError(f"{manifest_path} lacks columns: {sorted(missing)}")

    def __len__(self) -> int:
        return self.table.num_rows

    def _value(self, column: str, index: int) -> Any:
        value = self.table[column][index].as_py()
        return "" if value is None and column in {"video_id", "sequence_id"} else value

    def resolve_runtime_path(self, logical_path: str) -> str:
        # Dataset-relative manifests are anchored first so that an explicit path
        # map still sees the same absolute form it saw for the frozen manifests.
        if not os.path.isabs(logical_path):
            if self.dataset_root is None:
                raise ArtifactError(
                    f"relative frame path {logical_path!r} requires --dataset-root"
                )
            logical_path = os.path.join(os.fspath(self.dataset_root), logical_path)
        for source, destination in self.path_maps:
            if logical_path == source:
                return destination
            prefix = source.rstrip("/") + "/"
            if logical_path.startswith(prefix):
                return destination.rstrip("/") + "/" + logical_path[len(prefix) :]
        return logical_path

    def __getitem__(self, index: int) -> dict[str, Any]:
        frame_path = str(self._value("frame_path", index))
        runtime_path = self.resolve_runtime_path(frame_path)
        try:
            tensor = self.frame_loader(runtime_path)
        except Exception as exc:
            raise ArtifactError(
                f"failed to read manifest frame {frame_path} (runtime {runtime_path}): {exc}"
            ) from exc
        return {
            "image": tensor,
            "dataset": str(self._value("dataset", index)),
            "split": str(self._value("split", index)),
            "sample_id": str(self._value("sample_id", index)),
            "video_id": str(self._value("video_id", index)),
            "sequence_id": str(self._value("sequence_id", index)),
            "frame_path": frame_path,
            "frame_index": int(self._value("frame_index", index)),
            "label": int(self._value("label", index)),
        }

    def validate_identity(self) -> dict[str, Any]:
        sample_ids = self.table["sample_id"].to_pylist()
        frame_paths = self.table["frame_path"].to_pylist()
        if len(sample_ids) != len(set(sample_ids)):
            raise ArtifactError(f"{self.manifest_path}: duplicate sample_id")
        if len(frame_paths) != len(set(frame_paths)):
            raise ArtifactError(f"{self.manifest_path}: duplicate frame_path")
        labels_by_unit: dict[tuple[str, str], set[int]] = defaultdict(set)
        for index in range(len(self)):
            frame_path = str(frame_paths[index])
            runtime_path = self.resolve_runtime_path(frame_path)
            if not os.path.isfile(runtime_path):
                raise ArtifactError(
                    f"{self.manifest_path}: missing frame: {frame_path} "
                    f"(runtime {runtime_path})"
                )
            dataset = str(self._value("dataset", index))
            sequence = str(self._value("sequence_id", index))
            video = str(self._value("video_id", index))
            unit_id = sequence or video
            if not unit_id:
                raise ArtifactError(f"{self.manifest_path}: empty analysis-unit ID at row {index}")
            label = int(self._value("label", index))
            if label not in (0, 1):
                raise ArtifactError(f"{self.manifest_path}: non-binary label {label}")
            labels_by_unit[(dataset, unit_id)].add(label)
        inconsistent = [key for key, labels in labels_by_unit.items() if len(labels) != 1]
        if inconsistent:
            raise ArtifactError(f"{self.manifest_path}: inconsistent unit labels: {inconsistent[:5]}")
        return {"frames": len(self), "analysis_units": len(labels_by_unit)}


class DeepfakeBenchFrameLoader:
    """Exact frame preprocessing used by DeepfakeAbstractBaseDataset.load_rgb."""

    def __init__(
        self,
        resolution: int = 224,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        self.resolution = resolution
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, frame_path: str) -> torch.Tensor:
        # Keep this sequence byte-for-byte equivalent to DFB:
        # cv2.imread(BGR) -> RGB -> INTER_CUBIC square resize -> ToTensor -> normalize.
        image = cv2.imread(frame_path)
        if image is None:
            raise ArtifactError(f"cv2.imread returned None: {frame_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            (self.resolution, self.resolution),
            interpolation=cv2.INTER_CUBIC,
        )
        return self.normalize(self.to_tensor(image))


class WildDeepfakeFrameLoader:
    """Exact PIL/torchvision preprocessing in archived eval_holdout_datasets.py."""

    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __call__(self, frame_path: str) -> torch.Tensor:
        with Image.open(frame_path) as image:
            return self.transform(image.convert("RGB"))


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    return CLIPStandardLoRABackbone(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_position=args.lora_position,
    )


def normalize_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                payload = nested
                break
    if not isinstance(payload, dict) or not payload:
        raise ArtifactError("checkpoint is not a non-empty state_dict")
    if all(str(key).startswith("module.") for key in payload):
        payload = {str(key)[7:]: value for key, value in payload.items()}
    if all(str(key).startswith("backbone.") for key in payload):
        payload = {str(key)[9:]: value for key, value in payload.items()}
    return payload


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = build_model(args)
    try:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(args.checkpoint, map_location="cpu")
    state_dict = normalize_state_dict(payload)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ArtifactError(
            f"checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.to(device)
    model.eval()
    return model


def batch_rows(
    batch: dict[str, Any], logits: torch.Tensor, probabilities: torch.Tensor
) -> list[dict[str, Any]]:
    logits_np = logits.detach().to(torch.float64).cpu().numpy()
    probabilities_np = probabilities.detach().to(torch.float64).cpu().numpy()
    labels_np = batch["label"].numpy()
    frame_indices = batch["frame_index"].numpy()
    output: list[dict[str, Any]] = []
    for index in range(len(probabilities_np)):
        output.append(
            {
                "dataset": batch["dataset"][index],
                "split": batch["split"][index],
                "sample_id": batch["sample_id"][index],
                "video_id": batch["video_id"][index],
                "sequence_id": batch["sequence_id"][index],
                "frame_path": batch["frame_path"][index],
                "frame_index": int(frame_indices[index]),
                "label": int(labels_np[index]),
                "real_logit": float(logits_np[index, 0]),
                "fake_logit": float(logits_np[index, 1]),
                "fake_probability": float(probabilities_np[index]),
            }
        )
    return output


def verify_expected(metrics: dict[str, Any], expected_path: Path | None) -> dict[str, Any]:
    if expected_path is None:
        return {"enabled": False, "passed": True}
    expected = yaml.safe_load(expected_path.read_text(encoding="utf-8"))
    tolerance = float(expected["absolute_auc_tolerance"])
    comparisons: dict[str, Any] = {}
    passed = True
    for dataset, expected_auc in expected["analysis_unit_auc"].items():
        actual = float(metrics["datasets"][dataset]["analysis_unit"]["auc"])
        difference = abs(actual - float(expected_auc))
        okay = difference <= tolerance
        passed = passed and okay
        comparisons[dataset] = {
            "expected": float(expected_auc),
            "actual": actual,
            "absolute_difference": difference,
            "passed": okay,
        }
    if "macro_analysis_unit_auc" in expected:
        actual = float(metrics["macro_analysis_unit_auc"])
        expected_macro = float(expected["macro_analysis_unit_auc"])
        difference = abs(actual - expected_macro)
        okay = difference <= tolerance
        passed = passed and okay
        comparisons["macro_analysis_unit_auc"] = {
            "expected": expected_macro,
            "actual": actual,
            "absolute_difference": difference,
            "passed": okay,
        }
    result = {
        "enabled": True,
        "expected_file": os.fspath(expected_path.resolve()),
        "expected_file_sha256": sha256_file(expected_path),
        "absolute_auc_tolerance": tolerance,
        "comparisons": comparisons,
        "passed": passed,
    }
    if not passed:
        raise ArtifactError(f"expected-metric validation failed: {comparisons}")
    return result


def evaluate(args: argparse.Namespace) -> Path:
    checkpoint = args.checkpoint.resolve()
    manifests = [path.resolve() for path in args.manifest]
    output_root = args.output_root.resolve()
    final_dir = output_root / args.run_name
    partial_dir = output_root / f".{args.run_name}.partial.{os.getpid()}"
    if final_dir.exists():
        raise ArtifactError(f"immutable run directory already exists: {final_dir}")
    if partial_dir.exists():
        raise ArtifactError(f"partial run directory already exists: {partial_dir}")
    if not checkpoint.is_file():
        raise ArtifactError(f"checkpoint missing: {checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ArtifactError(f"CUDA requested but unavailable: {args.device}")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.device_ids:
        if device.type != "cuda":
            raise ArtifactError("--device-ids requires a CUDA primary device")
        unavailable = [
            index for index in args.device_ids if index < 0 or index >= torch.cuda.device_count()
        ]
        if unavailable:
            raise ArtifactError(
                f"requested CUDA devices do not exist: {unavailable}; "
                f"visible count={torch.cuda.device_count()}"
            )
        primary_index = device.index if device.index is not None else 0
        if args.device_ids[0] != primary_index:
            raise ArtifactError(
                f"first --device-ids value must equal primary device {primary_index}"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir()

    metadata: dict[str, Any] = {
        "run_name": args.run_name,
        "status": "RUNNING",
        "started_utc": utc_now(),
        "command": [os.fspath(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
        "model": {
            "kind": "standard_weight_space_lora",
            "checkpoint": os.fspath(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "lora_position": args.lora_position,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
        },
        "manifests": [
            {"path": os.fspath(path), "sha256": sha256_file(path)} for path in manifests
        ],
        "runtime_path_maps": [
            {"source": source, "destination": destination}
            for source, destination in args.path_maps
        ],
        "script_sha256": sha256_file(Path(__file__)),
        "metrics_module_sha256": sha256_file(SCRIPT_DIR / "metrics.py"),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "cuda_device_ids": args.device_ids,
            "cuda_device_names": (
                [torch.cuda.get_device_name(index) for index in args.device_ids]
                if args.device_ids
                else []
            ),
            "preprocessing_protocols": {
                "deepfakebench_datasets": {
                    "reference": "DeepfakeAbstractBaseDataset.load_rgb/__getitem__",
                    "color": "cv2 BGR to RGB",
                    "resize": "cv2.INTER_CUBIC",
                    "resolution": 224,
                    "mean": [0.5, 0.5, 0.5],
                    "std": [0.5, 0.5, 0.5],
                },
                "WildDeepfake": {
                    "reference": "archived eval_holdout_datasets.py",
                    "loader": "PIL RGB",
                    "resize": "torchvision.transforms.Resize((224,224))",
                    "mean": [0.5, 0.5, 0.5],
                    "std": [0.5, 0.5, 0.5],
                },
            },
        },
        "git": git_revision(),
    }
    atomic_write_json(partial_dir / "run_metadata.json", metadata)

    model = load_model(args, device)
    parameter_counts = {
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
    metadata["model"]["parameter_counts"] = parameter_counts
    atomic_write_json(partial_dir / "run_metadata.json", metadata)
    if args.device_ids and len(args.device_ids) > 1:
        model = torch.nn.DataParallel(
            model, device_ids=args.device_ids, output_device=args.device_ids[0]
        )
        model.eval()
    frame_labels_by_dataset: dict[str, list[int]] = defaultdict(list)
    frame_probs_by_dataset: dict[str, list[float]] = defaultdict(list)
    unit_state: dict[tuple[str, str], dict[str, Any]] = {}
    seen_sample_ids: set[str] = set()
    manifest_checks: list[dict[str, Any]] = []
    raw_path = partial_dir / "frame_predictions.parquet"
    raw_writer = pq.ParquetWriter(raw_path, RAW_SCHEMA, compression="zstd")

    try:
        for manifest_path in manifests:
            with manifest_path.open(encoding="utf-8") as manifest_handle:
                first_manifest_row = json.loads(next(manifest_handle))
            dataset_name = str(first_manifest_row["dataset"])
            frame_loader = (
                WildDeepfakeFrameLoader()
                if dataset_name == "WildDeepfake"
                else DeepfakeBenchFrameLoader()
            )
            dataset = ArrowManifestDataset(
                manifest_path, frame_loader, path_maps=args.path_maps,
                dataset_root=args.dataset_root
            )
            check = dataset.validate_identity()
            check.update(
                {
                    "path": os.fspath(manifest_path),
                    "sha256": sha256_file(manifest_path),
                }
            )
            manifest_checks.append(check)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.workers > 0,
            )
            with torch.inference_mode():
                for batch in tqdm(loader, desc=manifest_path.stem, unit="batch"):
                    images = batch["image"].to(device, non_blocking=device.type == "cuda")
                    logits, _ = model(images)
                    probabilities = torch.softmax(logits, dim=1)[:, 1]
                    if logits.ndim != 2 or logits.shape[1] != 2:
                        raise ArtifactError(f"model logits must be [B,2], got {logits.shape}")
                    rows = batch_rows(batch, logits, probabilities)
                    for row in rows:
                        sample_id = row["sample_id"]
                        if sample_id in seen_sample_ids:
                            raise ArtifactError(f"duplicate prediction sample_id: {sample_id}")
                        seen_sample_ids.add(sample_id)
                        dataset_name = row["dataset"]
                        label = int(row["label"])
                        probability = float(row["fake_probability"])
                        frame_labels_by_dataset[dataset_name].append(label)
                        frame_probs_by_dataset[dataset_name].append(probability)
                        unit_id = analysis_unit_id(row)
                        key = (dataset_name, unit_id)
                        if key not in unit_state:
                            unit_state[key] = {
                                "dataset": dataset_name,
                                "split": row["split"],
                                "analysis_unit_id": unit_id,
                                "unit_kind": "sequence" if row["sequence_id"] else "video",
                                "label": label,
                                "probability_sum": 0.0,
                                "frame_count": 0,
                            }
                        unit = unit_state[key]
                        if unit["label"] != label:
                            raise ArtifactError(f"inconsistent labels in analysis unit {key}")
                        unit["probability_sum"] += probability
                        unit["frame_count"] += 1
                    raw_writer.write_table(pa.Table.from_pylist(rows, schema=RAW_SCHEMA))
    finally:
        raw_writer.close()

    expected_frames = sum(check["frames"] for check in manifest_checks)
    if len(seen_sample_ids) != expected_frames:
        raise ArtifactError(
            f"prediction count mismatch: {len(seen_sample_ids)} vs {expected_frames}"
        )

    unit_rows: list[dict[str, Any]] = []
    unit_labels_by_dataset: dict[str, list[int]] = defaultdict(list)
    unit_probs_by_dataset: dict[str, list[float]] = defaultdict(list)
    for key in sorted(unit_state):
        unit = unit_state[key]
        probability = unit["probability_sum"] / unit["frame_count"]
        row = {
            "dataset": unit["dataset"],
            "split": unit["split"],
            "analysis_unit_id": unit["analysis_unit_id"],
            "unit_kind": unit["unit_kind"],
            "label": unit["label"],
            "fake_probability": probability,
            "frame_count": unit["frame_count"],
        }
        unit_rows.append(row)
        unit_labels_by_dataset[row["dataset"]].append(row["label"])
        unit_probs_by_dataset[row["dataset"]].append(row["fake_probability"])

    pq.write_table(
        pa.Table.from_pylist(unit_rows, schema=UNIT_SCHEMA),
        partial_dir / "analysis_unit_predictions.parquet",
        compression="zstd",
    )
    if args.write_unit_csv:
        with (partial_dir / "analysis_unit_predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=[field.name for field in UNIT_SCHEMA])
            writer.writeheader()
            writer.writerows(unit_rows)

    metrics: dict[str, Any] = {"datasets": {}}
    for dataset_name in sorted(frame_labels_by_dataset):
        metrics["datasets"][dataset_name] = {
            "frame": binary_metrics(
                frame_labels_by_dataset[dataset_name],
                frame_probs_by_dataset[dataset_name],
            ),
            "analysis_unit": binary_metrics(
                unit_labels_by_dataset[dataset_name],
                unit_probs_by_dataset[dataset_name],
            ),
        }
    metrics["macro_analysis_unit_auc"] = float(
        np.mean(
            [
                metrics["datasets"][dataset]["analysis_unit"]["auc"]
                for dataset in sorted(metrics["datasets"])
            ]
        )
    )
    metrics["expectation_validation"] = verify_expected(metrics, args.expected)
    atomic_write_json(partial_dir / "metrics.json", metrics)

    metadata.update(
        {
            "status": "VALIDATED",
            "completed_utc": utc_now(),
            "manifest_checks": manifest_checks,
            "prediction_frames": len(seen_sample_ids),
            "prediction_analysis_units": len(unit_rows),
            "artifacts": {
                "frame_predictions.parquet": sha256_file(raw_path),
                "analysis_unit_predictions.parquet": sha256_file(
                    partial_dir / "analysis_unit_predictions.parquet"
                ),
                "metrics.json": sha256_file(partial_dir / "metrics.json"),
            },
        }
    )
    atomic_write_json(partial_dir / "run_metadata.json", metadata)
    (partial_dir / "VALIDATED").write_text(
        f"validated_utc={metadata['completed_utc']}\n", encoding="utf-8"
    )
    partial_dir.replace(final_dir)
    return final_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lora-position", default="attn_out")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--device-ids",
        type=int,
        nargs="+",
        help="optional CUDA DataParallel device IDs; first must match --device",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--write-unit-csv", action="store_true")
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="SOURCE=DESTINATION",
        help="runtime-only frame path prefix mapping; logical artifact paths stay unchanged",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Root directory holding the benchmark datasets. Required when the "
             "manifest stores dataset-relative frame paths, as the published "
             "manifests under datasets/manifests do; ignored for absolute paths.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch-size must be positive and workers must be non-negative")
    if args.expected is not None:
        args.expected = args.expected.resolve()
    parsed_maps: list[tuple[str, str]] = []
    for mapping in args.path_map:
        if "=" not in mapping:
            parser.error(f"invalid --path-map {mapping!r}; expected SOURCE=DESTINATION")
        source, destination = mapping.split("=", 1)
        source = source.rstrip("/") or "/"
        destination = destination.rstrip("/") or "/"
        if not source.startswith("/") or not destination.startswith("/"):
            parser.error("--path-map prefixes must be absolute")
        parsed_maps.append((source, destination))
    args.path_maps = parsed_maps
    return args


def main() -> None:
    args = parse_args()
    try:
        final_dir = evaluate(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
    print(json.dumps({"status": "VALIDATED", "run_directory": os.fspath(final_dir)}, indent=2))


if __name__ == "__main__":
    main()
