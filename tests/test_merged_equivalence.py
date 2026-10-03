#!/usr/bin/env python3
"""Merged-weight equivalence for every LoRA placement.

For each placement this materialises W' = W + (alpha/r) B A into the frozen
CLIP weights, removes the low-rank branches, and checks that the merged model
reproduces the unmerged forward pass. Passing means the reported numbers are a
property of the adapted weights rather than of the unmerged runtime path, so a
deployed model may merge the update without changing its predictions.

Requires the CLIP ViT-L/14 weights. Run from the repository root:
    python tests/test_merged_equivalence.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clip_lora.merge import load_checkpoint, merged_deployment  # noqa: E402
from src.clip_lora.model import CLIPStandardLoRABackbone, VALID_POSITIONS  # noqa: E402

TOLERANCE = 2e-4
RANKS = {"attn_out": 30, "attn_qv": 15, "mlp": 6, "mlp_attn_out": 5}
ALPHAS = {"attn_out": 120, "attn_qv": 60, "mlp": 24, "mlp_attn_out": 20}

results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def randomise(model: CLIPStandardLoRABackbone) -> None:
    """Give B a non-zero value so the update is not the identity."""
    generator = torch.Generator().manual_seed(20260924)
    for layer in model.lora_layers:
        for pair in layer.values():
            with torch.no_grad():
                pair["B"].weight.copy_(
                    torch.randn(pair["B"].weight.shape, generator=generator) * 0.01
                )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(
    position: str,
    batch: torch.Tensor,
    checkpoint: Path | None,
    device: torch.device,
) -> dict[str, object]:
    if checkpoint is None:
        model = CLIPStandardLoRABackbone(
            lora_rank=RANKS[position], lora_alpha=ALPHAS[position],
            lora_position=position, verbose=False,
        ).eval()
        randomise(model)
        checkpoint_hash = None
    else:
        model = load_checkpoint(
            checkpoint, rank=RANKS[position], alpha=ALPHAS[position], position=position
        ).eval()
        checkpoint_hash = sha256(checkpoint)
    model = model.to(device).eval()

    with torch.no_grad():
        logits_before, feat_before = model(batch)
    key_before = torch.stack([
        b.attn.in_proj_weight[1024:2048].clone()
        for b in model.clip_model.visual.transformer.resblocks
    ])

    merged = merged_deployment(model).eval()
    with torch.no_grad():
        logits_after, feat_after = merged(batch)

    logit_delta = (logits_before - logits_after).abs().max().item()
    feat_delta = (feat_before - feat_after).abs().max().item()
    record(f"merged-weight logit equivalence [{position}]",
           logit_delta < TOLERANCE, f"max|delta| = {logit_delta:.3e}")
    record(f"merged-weight feature equivalence [{position}]",
           feat_delta < TOLERANCE, f"max|delta| = {feat_delta:.3e}")

    key_drift = None
    if position == "attn_qv":
        key_after = torch.stack([
            b.attn.in_proj_weight[1024:2048]
            for b in merged.clip_model.visual.transformer.resblocks
        ])
        drift = (key_before - key_after).abs().max().item()
        key_drift = drift
        record("merged K projection untouched [attn_qv]",
               drift == 0.0, f"max|delta| = {drift:.3e}")
    return {
        "position": position,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": checkpoint_hash,
        "max_abs_logit_difference": logit_delta,
        "max_abs_feature_difference": feat_delta,
        "max_abs_key_projection_drift": key_drift,
        "tolerance": TOLERANCE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    print("Merged-weight equivalence gates\n")
    torch.manual_seed(20260924)
    batch = torch.randn(2, 3, 224, 224, device=device)
    details = []
    for position in VALID_POSITIONS:
        checkpoint = None
        if args.checkpoint_root:
            run_ids = {
                "attn_out": "std_attn_out_r30_alpha120_p025__seed3407",
                "attn_qv": "std_attn_qv_r15_alpha60_p025__seed3407",
                "mlp": "std_mlp_r6_alpha24_p025__seed3407",
                "mlp_attn_out": "std_mlp_attn_out_r5_alpha20_p025__seed3407",
            }
            checkpoint = args.checkpoint_root / run_ids[position] / "epoch_10.pth"
            if not checkpoint.is_file():
                raise SystemExit(f"missing checkpoint: {checkpoint}")
        details.append(check(position, batch, checkpoint, device))

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} gates passed")
    if passed != len(results):
        raise SystemExit("MERGED_EQUIVALENCE_FAILED")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "status": "VALIDATED",
            "script_sha256": sha256(Path(__file__).resolve()),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "random_seed": 20260924,
            "input_shape": list(batch.shape),
            "checks": details,
        }, indent=2) + "\n")
    print("ALL_MERGED_EQUIVALENCE_GATES_PASSED")


if __name__ == "__main__":
    main()
