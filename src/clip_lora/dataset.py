#!/usr/bin/env python3
"""Fail-hard FF++ training dataset backed by an immutable JSONL manifest."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ManifestDatasetError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_path_map(path: str, mapping: tuple[str, str] | None) -> str:
    if mapping is None:
        return path
    source, destination = mapping
    if path == source:
        return destination
    prefix = source.rstrip("/") + "/"
    if path.startswith(prefix):
        return destination.rstrip("/") + "/" + path[len(prefix) :]
    return path


def parse_path_map(value: str | None) -> tuple[str, str] | None:
    if not value:
        return None
    if "=" not in value:
        raise ManifestDatasetError("path map must use SOURCE=DESTINATION")
    source, destination = value.split("=", 1)
    if not source.startswith("/") or not destination.startswith("/"):
        raise ManifestDatasetError("path-map endpoints must be absolute")
    return source.rstrip("/"), destination.rstrip("/")


def anchor_path(path: str, dataset_root: str | os.PathLike[str] | None) -> str:
    """Resolve a manifest frame path against an optional dataset root.

    Absolute paths are returned unchanged, so manifests recorded with the
    original machine-local paths keep working exactly as before. Relative
    paths, as published in ``datasets/manifests``, are joined onto
    ``dataset_root``.
    """
    if os.path.isabs(path):
        return path
    if dataset_root is None:
        raise ManifestDatasetError(
            f"relative frame path {path!r} requires --dataset-root"
        )
    return os.path.join(os.fspath(dataset_root), path)


def read_manifest_lines(path: Path) -> list[str]:
    """Read a manifest as text lines, transparently handling gzip compression."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read().splitlines()
    return path.read_text(encoding="utf-8").splitlines()


class FixedManifestSBIDataset(Dataset):
    """Read exactly the listed samples and optionally apply SBI to real frames."""

    def __init__(
        self,
        manifest: str | os.PathLike[str],
        expected_frames: int,
        transform: Any,
        apply_sbi: bool,
        sbi_prob: float,
        path_map: str | None,
        expected_sha256: str | None = None,
        dataset_root: str | os.PathLike[str] | None = None,
        preflight_files: bool = True,
    ) -> None:
        self.manifest = Path(manifest).resolve()
        self.transform = transform
        self.apply_sbi = bool(apply_sbi)
        self.sbi_prob = float(sbi_prob)
        self.path_mapping = parse_path_map(path_map)
        self.dataset_root = dataset_root
        if not 0.0 <= self.sbi_prob <= 1.0:
            raise ManifestDatasetError("sbi_prob must be in [0, 1]")
        if not self.manifest.is_file():
            raise ManifestDatasetError(f"manifest missing: {self.manifest}")
        # Optional: pin one specific manifest file when a hash is being tracked.
        if expected_sha256:
            actual_hash = sha256_file(self.manifest)
            if actual_hash != expected_sha256:
                raise ManifestDatasetError(
                    f"manifest hash mismatch: expected={expected_sha256}, actual={actual_hash}"
                )

        self.records: list[dict[str, Any]] = []
        sample_ids: set[str] = set()
        for line_number, line in enumerate(
            read_manifest_lines(self.manifest), 1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestDatasetError(
                    f"invalid JSON at {self.manifest}:{line_number}"
                ) from exc
            required = {"sample_id", "frame_path", "label", "split"}
            missing = required - set(record)
            if missing:
                raise ManifestDatasetError(
                    f"missing fields at line {line_number}: {sorted(missing)}"
                )
            if record["split"] != "train" or record["label"] not in (0, 1):
                raise ManifestDatasetError(
                    f"invalid train record at line {line_number}: "
                    f"split={record['split']!r}, label={record['label']!r}"
                )
            if record["sample_id"] in sample_ids:
                raise ManifestDatasetError(f"duplicate sample_id: {record['sample_id']}")
            sample_ids.add(record["sample_id"])
            anchored = anchor_path(record["frame_path"], self.dataset_root)
            runtime_path = apply_path_map(anchored, self.path_mapping)
            item = dict(record)
            item["runtime_path"] = runtime_path
            item["landmark_path"] = runtime_path.replace(
                "/frames/", "/landmarks/"
            ).rsplit(".", 1)[0] + ".npy"
            self.records.append(item)

        if len(self.records) != expected_frames:
            raise ManifestDatasetError(
                f"manifest count mismatch: expected={expected_frames}, "
                f"actual={len(self.records)}"
            )
        if preflight_files:
            missing_paths = [
                item["runtime_path"]
                for item in self.records
                if not Path(item["runtime_path"]).is_file()
            ]
            if missing_paths:
                raise ManifestDatasetError(
                    f"{len(missing_paths)} manifest images are missing; "
                    f"examples={missing_paths[:5]}"
                )

        self.real_frames = sum(item["label"] == 0 for item in self.records)
        self.fake_frames = len(self.records) - self.real_frames
        self.real_landmarks = sum(
            item["label"] == 0 and Path(item["landmark_path"]).is_file()
            for item in self.records
        )
        if self.apply_sbi:
            try:
                module_name = os.environ.get("CLIP_LORA_SBI_MODULE", "sbi_api")
                SBI_API = importlib.import_module(module_name).SBI_API
            except ModuleNotFoundError as exc:
                raise ManifestDatasetError(
                    "pSBI requires an independently obtained Self-Blended Images "
                    "SBI_API module. Add it to PYTHONPATH or set "
                    "CLIP_LORA_SBI_MODULE to its importable module name."
                ) from exc

            self.sbi_api = SBI_API(phase="train", image_size=224)
        else:
            self.sbi_api = None

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _read_rgb(path: str) -> np.ndarray:
        image = cv2.imread(path)
        if image is None:
            raise ManifestDatasetError(f"cv2.imread failed: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.records[index]
        label = int(item["label"])
        path = item["runtime_path"]
        output_label = label

        if (
            self.apply_sbi
            and label == 0
            and random.random() < self.sbi_prob
            and Path(item["landmark_path"]).is_file()
        ):
            image = self._read_rgb(path)
            landmark = np.load(item["landmark_path"], allow_pickle=False)
            fake_blend, real_blend = self.sbi_api(image, landmark)
            if fake_blend is None or real_blend is None:
                raise ManifestDatasetError(f"SBI returned an empty sample: {path}")
            if random.random() < 0.5:
                output = Image.fromarray(fake_blend)
                output_label = 1
            else:
                output = Image.fromarray(real_blend)
                output_label = 0
        else:
            with Image.open(path) as image:
                output = image.convert("RGB")

        if self.transform is not None:
            output = self.transform(output)
        return {
            "image": output,
            "label": output_label,
            "sample_id": item["sample_id"],
        }
