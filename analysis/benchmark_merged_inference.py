#!/usr/bin/env python3
"""Balanced merged/unmerged inference benchmark using trained checkpoints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clip_lora.merge import load_checkpoint, merged_deployment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPECS = (
    ("attn_out", "Attn-Out", 30, 120, "std_attn_out_r30_alpha120_p025__seed3407"),
    ("attn_qv", "Attn-Q/V", 15, 60, "std_attn_qv_r15_alpha60_p025__seed3407"),
    ("mlp", "MLP", 6, 24, "std_mlp_r6_alpha24_p025__seed3407"),
    ("mlp_attn_out", "MLP+Attn-Out", 5, 20,
     "std_mlp_attn_out_r5_alpha20_p025__seed3407"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nvidia_snapshot() -> str:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader",
    ]
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True).stdout.strip()
    except OSError as error:
        return f"unavailable: {error}"


def latency(model, sample, warmup: int, iterations: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            model(sample)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def throughput(model, sample, warmup: int, iterations: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            model(sample)
        torch.cuda.synchronize()
    return sample.shape[0] * iterations / (time.perf_counter() - start)


def build(spec, state: str, checkpoint_root: Path, device: torch.device):
    position, _, rank, alpha, run_id = spec
    checkpoint = checkpoint_root / run_id / "epoch_10.pth"
    model = load_checkpoint(
        checkpoint, rank=rank, alpha=alpha, position=position
    ).eval()
    if state == "merged":
        model = merged_deployment(model).eval()
    return model.to(device), checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--global-warmup", type=int, default=200)
    parser.add_argument("--batch1-warmup", type=int, default=100)
    parser.add_argument("--batch1-iterations", type=int, default=500)
    parser.add_argument("--batch32-warmup", type=int, default=100)
    parser.add_argument("--batch32-iterations", type=int, default=300)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/merged_inference_benchmark.csv")
    args = parser.parse_args()
    if args.rounds != 8:
        raise SystemExit("this balanced Latin-cycle protocol requires exactly 8 rounds")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    script_path = Path(__file__).resolve()
    checkpoint_hashes = {}
    for position, _, _, _, run_id in SPECS:
        checkpoint = args.checkpoint_root / run_id / "epoch_10.pth"
        if not checkpoint.is_file():
            raise SystemExit(f"missing checkpoint: {checkpoint}")
        checkpoint_hashes[position] = sha256(checkpoint)

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    process_start = nvidia_snapshot()

    # Stabilise kernels and clocks before any recorded condition.
    warm_model, _ = build(SPECS[0], "unmerged", args.checkpoint_root, device)
    warm_sample = torch.randn(1, 3, 224, 224, device=device)
    with torch.inference_mode():
        for _ in range(args.global_warmup):
            warm_model(warm_sample)
    torch.cuda.synchronize()
    del warm_model, warm_sample
    torch.cuda.empty_cache()

    # Eight conditions, cyclically shifted over eight rounds: every condition
    # occupies every ordinal position exactly once. Merged/unmerged states are
    # interleaved rather than measured in a fixed pair order.
    base = []
    for index, spec in enumerate(SPECS):
        base.append((spec, "unmerged" if index % 2 == 0 else "merged"))
    for index, spec in enumerate(SPECS):
        base.append((spec, "merged" if index % 2 == 0 else "unmerged"))

    rows = []
    for round_index in range(args.rounds):
        schedule = base[round_index:] + base[:round_index]
        for order_index, (spec, state) in enumerate(schedule, start=1):
            position, label, _, _, run_id = spec
            model, checkpoint = build(spec, state, args.checkpoint_root, device)
            one = torch.randn(1, 3, 224, 224, device=device)
            many = torch.randn(32, 3, 224, 224, device=device)

            torch.cuda.reset_peak_memory_stats(device)
            batch1_ms = latency(model, one, args.batch1_warmup, args.batch1_iterations)
            batch1_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats(device)
            batch32_fps = throughput(
                model, many, args.batch32_warmup, args.batch32_iterations
            )
            batch32_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            row = {
                "round": round_index + 1,
                "order": order_index,
                "placement": label,
                "position": position,
                "state": state,
                "run_id": run_id,
                "checkpoint_sha256": checkpoint_hashes[position],
                "batch1_latency_ms": repr(batch1_ms),
                "batch32_fps": repr(batch32_fps),
                "batch1_peak_gib": repr(batch1_peak),
                "batch32_peak_gib": repr(batch32_peak),
            }
            rows.append(row)
            print(
                f"round={round_index + 1} order={order_index} {label} {state} "
                f"latency_ms={batch1_ms:.4f} fps={batch32_fps:.3f}", flush=True
            )
            del model, one, many
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSummary (mean +/- sample SD across eight hardware rounds)")
    for _, label, _, _, _ in SPECS:
        for state in ("unmerged", "merged"):
            selected = [r for r in rows if r["placement"] == label and r["state"] == state]
            lat = [float(r["batch1_latency_ms"]) for r in selected]
            fps = [float(r["batch32_fps"]) for r in selected]
            print(
                f"{label:14s} {state:9s} {st.fmean(lat):.4f} +/- {st.stdev(lat):.4f} ms; "
                f"{st.fmean(fps):.3f} +/- {st.stdev(fps):.3f} fps"
            )

    properties = torch.cuda.get_device_properties(device)
    meta = {
        "status": "VALIDATED",
        "script_sha256": sha256(script_path),
        "prediction_or_target_metrics_used": False,
        "checkpoint_root": str(args.checkpoint_root),
        "checkpoint_sha256": checkpoint_hashes,
        "seed": 3407,
        "device_argument": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": properties.name,
        "device_total_memory": properties.total_memory,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "rounds": args.rounds,
        "schedule": "eight-condition cyclic Latin schedule",
        "global_warmup": args.global_warmup,
        "batch1_warmup": args.batch1_warmup,
        "batch1_iterations": args.batch1_iterations,
        "batch32_warmup": args.batch32_warmup,
        "batch32_iterations": args.batch32_iterations,
        "precision": "float32",
        "process_snapshot_start": process_start,
        "process_snapshot_end": nvidia_snapshot(),
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} and {meta_path}")
    print("MERGED_INFERENCE_BENCHMARK_V2_VALIDATED")


if __name__ == "__main__":
    main()
