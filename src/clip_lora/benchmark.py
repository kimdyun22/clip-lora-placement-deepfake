#!/usr/bin/env python3
"""Sequential five-round efficiency benchmark on one physical RTX 4090."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from fvcore.nn import FlopCountAnalysis

from .metrics import sha256_file
from .model import CLIPStandardLoRABackbone


class ProtocolError(RuntimeError):
    pass


def load_standard_lora_model(spec: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = CLIPStandardLoRABackbone(
        lora_rank=int(spec["lora_rank"]),
        lora_alpha=int(spec["lora_alpha"]),
        lora_position=str(spec["lora_position"]),
    )
    try:
        state = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(spec["checkpoint"], map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    if all(str(key).startswith("backbone.") for key in state):
        state = {str(key)[9:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device)


def count_active_parameters(model: torch.nn.Module) -> dict[str, int | str]:
    prefixes = ("clip_model.visual.", "lora_layers.", "head.")
    active = [p for name, p in model.named_parameters() if name.startswith(prefixes)]
    return {
        "loaded_total": sum(p.numel() for p in model.parameters()),
        "requires_grad_flag": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "active_forward": sum(p.numel() for p in active),
        "active_forward_trainable": sum(p.numel() for p in active if p.requires_grad),
        "classification": "CLIP visual tower + classifier head + LoRA layers",
    }


class LogitsOnly(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)[0]


def measure_flops(model: torch.nn.Module, sample: torch.Tensor) -> dict[str, Any]:
    analysis = FlopCountAnalysis(LogitsOnly(model).eval(), sample)
    total = int(analysis.total())
    unsupported = {str(k): int(v) for k, v in analysis.unsupported_ops().items()}
    return {
        "tool": "fvcore.nn.FlopCountAnalysis",
        "input_batch_size": 1,
        "counted_flops": total,
        "counted_gflops_per_frame": total / 1e9,
        "unsupported_operators": unsupported,
        "operator_coverage_complete": not unsupported,
        "convention": "fvcore operator handles; one fused multiply-add is one flop",
    }


def sample_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ProtocolError("empty measurement list")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def latency_measurement(
    model: torch.nn.Module,
    sample: torch.Tensor,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        synchronize(sample.device)
        torch.cuda.reset_peak_memory_stats(sample.device)
        timings_ms: list[float] = []
        for _ in range(iterations):
            synchronize(sample.device)
            started = time.perf_counter_ns()
            model(sample)
            synchronize(sample.device)
            timings_ms.append((time.perf_counter_ns() - started) / 1e6)
        peak_allocated = torch.cuda.max_memory_allocated(sample.device)
        peak_reserved = torch.cuda.max_memory_reserved(sample.device)
    return {
        "batch_size": int(sample.shape[0]),
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "milliseconds_per_batch": sample_stats(timings_ms),
        "primary_peak_allocated_bytes": int(peak_allocated),
        "secondary_peak_reserved_bytes": int(peak_reserved),
    }


def throughput_measurement(
    model: torch.nn.Module,
    sample: torch.Tensor,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        synchronize(sample.device)
        torch.cuda.reset_peak_memory_stats(sample.device)
        started = time.perf_counter_ns()
        for _ in range(iterations):
            model(sample)
        synchronize(sample.device)
        elapsed = (time.perf_counter_ns() - started) / 1e9
        peak_allocated = torch.cuda.max_memory_allocated(sample.device)
        peak_reserved = torch.cuda.max_memory_reserved(sample.device)
    return {
        "batch_size": int(sample.shape[0]),
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "elapsed_seconds": elapsed,
        "frames_per_second": int(sample.shape[0]) * iterations / elapsed,
        "primary_peak_allocated_bytes": int(peak_allocated),
        "secondary_peak_reserved_bytes": int(peak_reserved),
    }


def training_measurement(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    warmup: int,
    iterations: int,
    learning_rate: float,
    weight_decay: float,
    projected_steps: int,
) -> dict[str, Any]:
    device = images.device
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(images)
        loss_fn(logits, labels).backward()
        optimizer.step()

    for _ in range(warmup):
        step()
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings_ms: list[float] = []
    for _ in range(iterations):
        synchronize(device)
        started = time.perf_counter_ns()
        step()
        synchronize(device)
        timings_ms.append((time.perf_counter_ns() - started) / 1e6)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    stats = sample_stats(timings_ms)
    return {
        "batch_size": int(images.shape[0]),
        "warmup_optimizer_steps": warmup,
        "measured_optimizer_steps": iterations,
        "milliseconds_per_optimizer_step": stats,
        "effective_frames_per_second": int(images.shape[0]) / (float(stats["mean"]) / 1000.0),
        "projected_steps_per_epoch": projected_steps,
        "projected_epoch_seconds_model_compute_only": float(stats["mean"]) / 1000.0 * projected_steps,
        "primary_peak_allocated_bytes": int(peak_allocated),
        "secondary_peak_reserved_bytes": int(peak_reserved),
    }


def aggregate(protocol: dict[str, Any], protocol_hash: str, root: Path) -> None:
    per_model: dict[str, dict[str, Any]] = {}
    for spec in protocol["models"]:
        model_id = spec["id"]
        round_metrics = []
        for round_index in range(1, int(protocol["rounds"]["count"]) + 1):
            path = root / "rounds" / f"round_{round_index:02d}" / model_id / "metrics.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["protocol_sha256"] != protocol_hash:
                raise ProtocolError(f"protocol hash mismatch: {path}")
            round_metrics.append(result)

        def values(*keys: str) -> list[float]:
            output = []
            for item in round_metrics:
                value: Any = item
                for key in keys:
                    value = value[key]
                output.append(float(value))
            return output

        per_model[model_id] = {
            "label": spec["label"],
            "rounds": len(round_metrics),
            "checkpoint_sha256": round_metrics[0]["checkpoint_sha256"],
            "parameters": round_metrics[0]["parameters"],
            "flops": next(item["flops"] for item in round_metrics if item.get("flops") is not None),
            "batch1_latency_ms": sample_stats(values("inference_latency", "milliseconds_per_batch", "mean")),
            "batch32_throughput_fps": sample_stats(values("inference_throughput", "frames_per_second")),
            "inference_peak_allocated_mib": sample_stats([
                value / 2**20 for value in values("inference_throughput", "primary_peak_allocated_bytes")
            ]),
            "inference_peak_reserved_mib": sample_stats([
                value / 2**20 for value in values("inference_throughput", "secondary_peak_reserved_bytes")
            ]),
            "training_step_ms": sample_stats(values("training_compute", "milliseconds_per_optimizer_step", "mean")),
            "training_effective_fps": sample_stats(values("training_compute", "effective_frames_per_second")),
            "training_peak_allocated_mib": sample_stats([
                value / 2**20 for value in values("training_compute", "primary_peak_allocated_bytes")
            ]),
            "training_peak_reserved_mib": sample_stats([
                value / 2**20 for value in values("training_compute", "secondary_peak_reserved_bytes")
            ]),
            "projected_epoch_minutes_model_compute_only": sample_stats([
                value / 60.0 for value in values("training_compute", "projected_epoch_seconds_model_compute_only")
            ]),
        }

    report = {
        "status": "VALIDATED",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "summary_unit": "five hardware benchmark round means",
        "results": per_model,
    }
    (root / "efficiency_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = [
        "model_id", "label", "rounds", "trainable_params", "counted_gflops",
        "batch1_ms_mean", "batch1_ms_sd", "batch32_fps_mean", "batch32_fps_sd",
        "inference_peak_allocated_gib_mean", "inference_peak_allocated_gib_sd",
        "training_step_ms_mean", "training_step_ms_sd",
        "training_peak_allocated_gib_mean", "training_peak_allocated_gib_sd",
        "projected_epoch_minutes_mean", "projected_epoch_minutes_sd",
    ]
    with (root / "efficiency_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec in protocol["models"]:
            model_id, item = spec["id"], per_model[spec["id"]]
            writer.writerow({
                "model_id": model_id,
                "label": item["label"],
                "rounds": item["rounds"],
                "trainable_params": item["parameters"]["active_forward_trainable"],
                "counted_gflops": item["flops"]["counted_gflops_per_frame"],
                "batch1_ms_mean": item["batch1_latency_ms"]["mean"],
                "batch1_ms_sd": item["batch1_latency_ms"]["sample_sd"],
                "batch32_fps_mean": item["batch32_throughput_fps"]["mean"],
                "batch32_fps_sd": item["batch32_throughput_fps"]["sample_sd"],
                "inference_peak_allocated_gib_mean": item["inference_peak_allocated_mib"]["mean"] / 1024.0,
                "inference_peak_allocated_gib_sd": item["inference_peak_allocated_mib"]["sample_sd"] / 1024.0,
                "training_step_ms_mean": item["training_step_ms"]["mean"],
                "training_step_ms_sd": item["training_step_ms"]["sample_sd"],
                "training_peak_allocated_gib_mean": item["training_peak_allocated_mib"]["mean"] / 1024.0,
                "training_peak_allocated_gib_sd": item["training_peak_allocated_mib"]["sample_sd"] / 1024.0,
                "projected_epoch_minutes_mean": item["projected_epoch_minutes_model_compute_only"]["mean"],
                "projected_epoch_minutes_sd": item["projected_epoch_minutes_model_compute_only"]["sample_sd"],
            })
    (root / "EFFICIENCY_VALIDATED").write_text("VALIDATED\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_measurement":
        raise ProtocolError("protocol is not frozen before measurement")
    protocol_hash = sha256_file(protocol_path)
    device = torch.device(str(protocol["hardware"]["device"]))
    if not torch.cuda.is_available():
        raise ProtocolError("CUDA unavailable")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_name(device) != protocol["hardware"]["required_gpu"]:
        raise ProtocolError("GPU differs from frozen protocol")
    torch.backends.cuda.matmul.allow_tf32 = bool(protocol["determinism"]["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(protocol["determinism"]["cudnn_allow_tf32"])
    torch.backends.cudnn.benchmark = bool(protocol["determinism"]["cudnn_benchmark"])

    specs = {item["id"]: item for item in protocol["models"]}
    expected = set(specs)
    for order in protocol["rounds"]["orders"]:
        if set(order) != expected or len(order) != len(expected):
            raise ProtocolError(f"invalid round order: {order}")
    if args.preflight_only:
        for spec in protocol["models"]:
            checkpoint = Path(spec["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            torch.manual_seed(int(protocol["determinism"]["synthetic_input_seed"]))
            torch.cuda.manual_seed_all(int(protocol["determinism"]["synthetic_input_seed"]))
            model = load_standard_lora_model(spec, device)
            params = count_active_parameters(model)
            if params["active_forward_trainable"] != int(spec["expected_trainable_params"]):
                raise ProtocolError(f"trainable parameter mismatch: {spec['id']}")
            sample = torch.randn(2, 3, 224, 224, device=device)
            labels = torch.tensor([0, 1], device=device)
            with torch.inference_mode():
                model.eval()(sample)
            model.train()
            logits, _ = model(sample)
            torch.nn.functional.cross_entropy(logits, labels).backward()
            synchronize(device)
            del model, sample, labels, logits
            torch.cuda.empty_cache()
            gc.collect()
        print("EFFICIENCY_PREFLIGHT_PASSED")
        return
    root = Path(protocol["reporting"]["output_root"])
    if root.exists():
        raise ProtocolError(f"immutable output already exists: {root}")
    root.mkdir(parents=True)
    (root / "RUNNING").write_text("RUNNING\n", encoding="utf-8")
    rounds_root = root / "rounds"
    rounds_root.mkdir()
    seed = int(protocol["determinism"]["synthetic_input_seed"])

    for round_index, order in enumerate(protocol["rounds"]["orders"], start=1):
        round_root = rounds_root / f"round_{round_index:02d}"
        round_root.mkdir()
        (round_root / "order.json").write_text(json.dumps(order) + "\n", encoding="utf-8")
        for position_index, model_id in enumerate(order, start=1):
            spec = specs[model_id]
            checkpoint = Path(spec["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.empty_cache()
            gc.collect()
            model = load_standard_lora_model(spec, device)
            params = count_active_parameters(model)
            if params["active_forward_trainable"] != int(spec["expected_trainable_params"]):
                raise ProtocolError(f"trainable parameter mismatch: {model_id}")
            generator = torch.Generator(device=device).manual_seed(seed)
            sample1 = torch.randn(1, 3, 224, 224, generator=generator, device=device)
            sample32 = torch.randn(32, 3, 224, 224, generator=generator, device=device)
            labels32 = torch.arange(32, device=device, dtype=torch.long) % 2
            flops = measure_flops(model, sample1) if round_index == 1 else None
            result = {
                "status": "VALIDATED",
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_hash,
                "benchmark_script_sha256": sha256_file(Path(__file__).resolve()),
                "round": round_index,
                "position_in_round": position_index,
                "order": order,
                "model": spec,
                "checkpoint_sha256": sha256_file(checkpoint),
                "environment": {
                    "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device),
                    "device": str(device),
                    "precision": protocol["precision"],
                },
                "parameters": params,
                "flops": flops,
                "inference_latency": latency_measurement(
                    model, sample1,
                    int(protocol["inference"]["latency_warmup_iterations"]),
                    int(protocol["inference"]["latency_measured_iterations"]),
                ),
                "inference_throughput": throughput_measurement(
                    model, sample32,
                    int(protocol["inference"]["throughput_warmup_iterations"]),
                    int(protocol["inference"]["throughput_measured_iterations"]),
                ),
                "training_compute": training_measurement(
                    model, sample32, labels32,
                    int(protocol["training_compute"]["warmup_optimizer_steps"]),
                    int(protocol["training_compute"]["measured_optimizer_steps"]),
                    float(protocol["training_compute"]["learning_rate"]),
                    float(protocol["training_compute"]["weight_decay"]),
                    int(protocol["training_compute"]["projected_steps_per_epoch"]),
                ),
            }
            final = round_root / model_id
            partial = round_root / f".{model_id}.partial"
            partial.mkdir()
            (partial / "metrics.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (partial / "VALIDATED").write_text("VALIDATED\n", encoding="utf-8")
            partial.rename(final)
            print(json.dumps({
                "event": "EFFICIENCY_ROUND_MODEL_COMPLETE",
                "round": round_index,
                "position": position_index,
                "model_id": model_id,
            }), flush=True)
            del model, sample1, sample32, labels32, result
            torch.cuda.empty_cache()
            gc.collect()

    aggregate(protocol, protocol_hash, root)
    (root / "RUNNING").unlink()
    print(json.dumps({"status": "VALIDATED", "output": os.fspath(root)}), flush=True)


if __name__ == "__main__":
    main()
