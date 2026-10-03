#!/usr/bin/env python3
"""Train one standard weight-space LoRA configuration from a frozen protocol."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]

from .dataset import FixedManifestSBIDataset, sha256_file
from .model import CLIPStandardLoRABackbone


class TrainingError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise TrainingError("protocol is not frozen")
    if float(protocol["training"]["lora_scale"]) != 4.0:
        raise TrainingError("only the frozen LoRA scale 4.0 is allowed")
    return protocol


def candidate_by_id(protocol: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    matches = [item for item in protocol["placements"] if item["id"] == candidate_id]
    if len(matches) != 1:
        raise TrainingError(f"candidate not found or duplicated: {candidate_id}")
    item = dict(matches[0])
    item["lora_rank"] = item.pop("rank")
    item["lora_alpha"] = item.pop("alpha")
    item["sbi_prob"] = float(protocol["training"]["p_sbi"])
    return item


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    cv2.setRNGSeed(int(worker_seed % (2**31 - 1)))
    cv2.setNumThreads(1)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    cv2.setRNGSeed(int(seed % (2**31 - 1)))
    cv2.setNumThreads(1)


def source_hashes() -> dict[str, str]:
    paths = {
        "train_standard_lora.py": Path(__file__).resolve(),
        "model.py": Path(__file__).resolve().with_name("model.py"),
        "dataset.py": Path(__file__).resolve().with_name("dataset.py"),
    }
    return {name: sha256_file(path) for name, path in paths.items() if path.is_file()}


def build_model(candidate: dict[str, Any]) -> torch.nn.Module:
    return CLIPStandardLoRABackbone(
        lora_rank=int(candidate["lora_rank"]),
        lora_alpha=int(candidate["lora_alpha"]),
        lora_position=str(candidate["lora_position"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True, help="CUDA device such as cuda:0")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Root directory holding the benchmark datasets. Required when the "
             "manifest stores dataset-relative frame paths, as the published "
             "manifests under datasets/manifests do; ignored for absolute paths.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    candidate = candidate_by_id(protocol, args.candidate)

    if args.seed not in [int(s) for s in protocol["seeds"]]:
        raise TrainingError(f"seed {args.seed} is not frozen in the protocol")
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise TrainingError(f"CUDA device is required and unavailable: {args.device}")
    if args.smoke_batches < 1:
        raise TrainingError("--smoke-batches must be positive")

    run_name = f"{candidate['id']}__seed{args.seed}"
    final_dir = args.output_root.resolve() / run_name
    partial_dir = args.output_root.resolve() / f".{run_name}.partial.{os.getpid()}"
    if not args.smoke_test and (final_dir.exists() or partial_dir.exists()):
        raise TrainingError(f"immutable output already exists: {final_dir}")

    set_determinism(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    training = protocol["training"]
    data = protocol["data"]

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    dataset = FixedManifestSBIDataset(
        manifest=data["train_manifest"],
        expected_frames=int(data["train_frames"]),
        transform=transform,
        apply_sbi=True,
        sbi_prob=float(candidate["sbi_prob"]),
        path_map=data["runtime_path_map"],
        expected_sha256=data.get("train_manifest_sha256"),
        dataset_root=args.dataset_root,
        preflight_files=True,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=0 if args.smoke_test else int(training["workers"]),
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=False,
    )

    model = build_model(candidate).to(device)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_parameters)
    total_count = sum(p.numel() for p in model.parameters())
    if trainable_count != int(candidate["expected_trainable_params"]):
        raise TrainingError(
            f"trainable parameter mismatch for {candidate['id']}: "
            f"expected={candidate['expected_trainable_params']}, actual={trainable_count}"
        )

    optimizer = optim.AdamW(
        trainable_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training["scheduler_t_max"])
    )

    print(json.dumps({
        "event": "RUN_START",
        "run_name": run_name,
        "candidate": candidate,
        "seed": args.seed,
        "device": args.device,
        "train_frames": len(dataset),
        "real_frames": dataset.real_frames,
        "fake_frames": dataset.fake_frames,
        "real_landmarks": dataset.real_landmarks,
        "trainable_params": trainable_count,
        "protocol_sha256": sha256_file(protocol_path),
    }, sort_keys=True), flush=True)

    started = utc_now()
    epoch_records: list[dict[str, Any]] = []
    epoch_count = 1 if args.smoke_test else int(training["epochs"])
    max_batches = args.smoke_batches if args.smoke_test else None

    for epoch in range(epoch_count):
        model.train()
        loss_sum = 0.0
        batches = 0
        progress = tqdm(
            loader,
            desc=f"{run_name} epoch {epoch + 1}/{epoch_count}",
            mininterval=10.0,
            dynamic_ncols=False,
        )
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(images)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise TrainingError(f"non-finite loss at epoch={epoch + 1}, batch={batches}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, max_norm=float(training["grad_clip_norm"])
            )
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batches += 1
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
            if max_batches is not None and batches >= max_batches:
                break
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "batches": batches,
            "mean_loss": loss_sum / batches,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        epoch_records.append(record)
        print(json.dumps({"event": "EPOCH_COMPLETE", **record}, sort_keys=True), flush=True)

    if args.smoke_test:
        print(json.dumps({"event": "SMOKE_TEST_PASSED", "run_name": run_name}, sort_keys=True))
        return

    partial_dir.parent.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir()
    try:
        checkpoint = partial_dir / f"epoch_{training['epochs']}.pth"
        torch.save(model.state_dict(), checkpoint)
        metadata = {
            "status": "TRAINED",
            "run_name": run_name,
            "candidate": candidate,
            "seed": args.seed,
            "started_utc": started,
            "completed_utc": utc_now(),
            "device": args.device,
            "protocol": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "protocol_id": protocol["protocol_id"],
            "train_manifest": os.fspath(dataset.manifest),
            "train_manifest_sha256": sha256_file(dataset.manifest),
            "train_frames": len(dataset),
            "real_frames": dataset.real_frames,
            "fake_frames": dataset.fake_frames,
            "real_landmarks": dataset.real_landmarks,
            "parameter_counts": {"total": total_count, "trainable": trainable_count},
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha256_file(checkpoint),
            "sources": source_hashes(),
            "implementation": "standard_weight_space_lora",
            "environment": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_device": torch.cuda.get_device_name(device),
            },
            "epochs": epoch_records,
        }
        (partial_dir / "train_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (partial_dir / "TRAINED").write_text("TRAINED\n", encoding="utf-8")
        partial_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(partial_dir, ignore_errors=True)
        raise

    print(json.dumps(
        {"event": "RUN_COMPLETE", "run_name": run_name, "output": str(final_dir)}, sort_keys=True
    ))


if __name__ == "__main__":
    main()
