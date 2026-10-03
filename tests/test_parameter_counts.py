"""Identity, MLP weight-space, gradient, and parameter-count checks."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.clip_lora.model import CLIPStandardLoRABackbone  # noqa: E402

EMBED, MLP_DIM, BLOCKS = 1024, 4096, 24
PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    results.append((name, PASS if ok else FAIL, detail))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")


def batch(n: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(n, 3, 224, 224)


def gate_zero_init_identity() -> None:
    """B = 0 at init, so every placement must reproduce frozen CLIP exactly."""
    x = batch()
    for pos in ("attn_out", "attn_qv", "mlp", "mlp_attn_out"):
        model = CLIPStandardLoRABackbone(lora_rank=8, lora_alpha=32, lora_position=pos, verbose=False)
        model.eval()
        with torch.no_grad():
            adapted = model.extract_features(x)
            for layer in model.lora_layers:  # neutralise by zeroing A as well
                for pair in layer.values():
                    nn.init.zeros_(pair["A"].weight)
            frozen = model.extract_features(x)
        delta = float((adapted - frozen).abs().max())
        record(f"zero-init identity [{pos}]", delta == 0.0, f"max|delta| = {delta:.3e}")
        del model


def gate_weight_space_equality() -> None:
    """The MLP path must equal a materialised (W + s.BA) linear layer."""
    torch.manual_seed(1)
    rank, alpha = 8, 32
    scaling = alpha / rank
    model = CLIPStandardLoRABackbone(lora_rank=rank, lora_alpha=alpha, lora_position="mlp", verbose=False)
    model.eval()
    for layer in model.lora_layers:  # move B off zero so the branch is live
        for pair in layer.values():
            nn.init.normal_(pair["B"].weight, std=0.02)

    block, layer = model.clip_model.visual.transformer.resblocks[0], model.lora_layers[0]
    h_in = torch.randn(5, EMBED)
    with torch.no_grad():
        got = model._mlp(block, layer, h_in)
        w_fc = block.mlp.c_fc.weight + scaling * (layer["fc"]["B"].weight @ layer["fc"]["A"].weight)
        w_pr = block.mlp.c_proj.weight + scaling * (layer["proj"]["B"].weight @ layer["proj"]["A"].weight)
        want = F.linear(block.mlp.gelu(F.linear(h_in, w_fc, block.mlp.c_fc.bias)), w_pr, block.mlp.c_proj.bias)
    delta = float((got - want).abs().max())
    record("weight-space equality [mlp]", delta < 1e-5, f"max|delta| = {delta:.3e}")
    del model


def gate_mlp_gradient_flow() -> None:
    """Both MLP B matrices must receive a nonzero first-step gradient."""
    for pos in ("mlp", "mlp_attn_out"):
        model = CLIPStandardLoRABackbone(lora_rank=4, lora_alpha=16, lora_position=pos, verbose=False)
        model.train()
        logits, _ = model(batch())
        model.zero_grad()
        logits.sum().backward()
        layer = model.lora_layers[0]
        fc_b = float(layer["fc"]["B"].weight.grad.abs().sum())
        pr_b = float(layer["proj"]["B"].weight.grad.abs().sum())
        record(
            f"mlp gradient flow [{pos}]",
            fc_b > 0 and pr_b > 0,
            f"fc_B = {fc_b:.3e}, proj_B = {pr_b:.3e}",
        )
        del model


def gate_parameter_counts() -> None:
    """Trainable counts must match the analytic LoRA formula plus the head."""
    head = EMBED * 2 + 2 + 2 * EMBED  # linear + LayerNorm affine
    cases = [
        ("attn_out", 16, BLOCKS * 2 * EMBED * 16),
        ("attn_qv", 16, BLOCKS * 4 * EMBED * 16),
        ("mlp", 4, BLOCKS * 2 * (EMBED + MLP_DIM) * 4),
        ("mlp_attn_out", 4, BLOCKS * (2 * (EMBED + MLP_DIM) * 4 + 2 * EMBED * 4)),
    ]
    for pos, rank, expected_lora in cases:
        model = CLIPStandardLoRABackbone(lora_rank=rank, lora_alpha=4 * rank, lora_position=pos, verbose=False)
        got = sum(p.numel() for p in model.parameters() if p.requires_grad)
        want = expected_lora + head
        record(f"parameter count [{pos} r{rank}]", got == want, f"got {got:,}, expected {want:,}")
        del model


def main() -> int:
    print("Standard-LoRA verification gates\n")
    for gate in (
        gate_zero_init_identity,
        gate_weight_space_equality,
        gate_mlp_gradient_flow,
        gate_parameter_counts,
    ):
        gate()
    failed = [r for r in results if r[1] == FAIL]
    print(f"\n{len(results) - len(failed)}/{len(results)} gates passed")
    if failed:
        print("FAILED:")
        for name, _, detail in failed:
            print(f"  - {name}: {detail}")
        return 1
    print("ALL_GATES_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
