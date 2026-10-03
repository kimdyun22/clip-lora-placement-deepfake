"""Utilities for folding standard-LoRA updates into CLIP visual weights."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .model import CLIPStandardLoRABackbone, _delta


def normalise_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    """Extract the public backbone state dict from a training checkpoint."""
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                payload = nested
                break
    if not isinstance(payload, dict) or not payload:
        raise ValueError("checkpoint is not a non-empty state_dict")
    if all(str(key).startswith("module.") for key in payload):
        payload = {str(key)[7:]: value for key, value in payload.items()}
    if all(str(key).startswith("backbone.") for key in payload):
        payload = {str(key)[9:]: value for key, value in payload.items()}
    return payload


def load_checkpoint(
    checkpoint: str | Path,
    *,
    rank: int,
    alpha: int,
    position: str,
) -> CLIPStandardLoRABackbone:
    """Strictly load one standard-LoRA backbone checkpoint."""
    model = CLIPStandardLoRABackbone(
        lora_rank=rank,
        lora_alpha=alpha,
        lora_position=position,
        verbose=False,
    )
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    state = normalise_state_dict(payload)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={incompatible.missing_keys}; "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model


def merge_into_backbone(model: CLIPStandardLoRABackbone) -> None:
    """In-place fold of every ``(alpha/r) B A`` update into its base weight."""
    blocks = model.clip_model.visual.transformer.resblocks
    with torch.no_grad():
        for block, layer in zip(blocks, model.lora_layers):
            if model.lora_position == "attn_qv":
                dq = _delta(layer["q"], model.scaling)
                dv = _delta(layer["v"], model.scaling)
                block.attn.in_proj_weight.add_(
                    torch.cat([dq, torch.zeros_like(dq), dv], dim=0)
                )
            if model.lora_position in ("attn_out", "mlp_attn_out"):
                block.attn.out_proj.weight.add_(
                    _delta(layer["attn_out"], model.scaling)
                )
            if model.lora_position in ("mlp", "mlp_attn_out"):
                block.mlp.c_fc.weight.add_(_delta(layer["fc"], model.scaling))
                block.mlp.c_proj.weight.add_(_delta(layer["proj"], model.scaling))


class MergedDeployment(nn.Module):
    """Branch-free visual forward after LoRA deltas have been folded into weights."""

    def __init__(self, source: CLIPStandardLoRABackbone) -> None:
        super().__init__()
        self.clip_model = source.clip_model
        self.head = source.head
        self._renormalize = source._renormalize

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vit = self.clip_model.visual
        x = self._renormalize(pixel_values)
        x = vit.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = vit.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1) + vit.positional_embedding.to(x.dtype)
        x = vit.ln_pre(x).permute(1, 0, 2)
        for block in vit.transformer.resblocks:
            x = block(x)
        feat = vit.ln_post(x.permute(1, 0, 2)[:, 0, :])
        return self.head(feat), feat


def merged_deployment(model: CLIPStandardLoRABackbone) -> MergedDeployment:
    """Merge a model in place and return the branch-free deployment wrapper."""
    merge_into_backbone(model)
    return MergedDeployment(model)
