"""CLIP adaptation model definitions for the reported baseline configurations.

These definitions preserve the evaluated feature/head paths and bias subset.
They do not provide the complete baseline training orchestration. Inputs use
DeepfakeBench normalization; outputs are (two-class logits, pre-projection CLS).
"""
from __future__ import annotations

import math

import clip
import torch
from torch import nn

from .model import CLIP_MEAN, CLIP_STD

WIDTH = 1024


class FrozenCLIPBackbone(nn.Module):
    """Frozen ViT-L/14 with a head; optional BitFit visual-bias subset.

    BitFit selects names ending in '.bias'. Fused attn.in_proj_bias parameters
    do not match this rule and remain frozen, as do all visual weights.
    """

    def __init__(self, train_bias: bool = False):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-L/14", device="cpu")
        self.clip_model = self.clip_model.float()
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False
        if train_bias:
            for name, parameter in self.clip_model.visual.named_parameters():
                if name.endswith(".bias"):
                    parameter.requires_grad = True
        self.head = nn.Sequential(nn.LayerNorm(WIDTH), nn.Dropout(0.1), nn.Linear(WIDTH, 2))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=x.device).view(1, 3, 1, 1)
        return ((x * 0.5 + 0.5) - mean) / std

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        vit = self.clip_model.visual
        x = vit.conv1(self._normalize(x))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = vit.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1) + vit.positional_embedding.to(x.dtype)
        return vit.ln_pre(x).permute(1, 0, 2)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        vit = self.clip_model.visual
        tokens = vit.transformer(self._tokens(x)).permute(1, 0, 2)
        return vit.ln_post(tokens[:, 0, :])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        return self.head(features), features


class AdapterCLIPBackbone(FrozenCLIPBackbone):
    """Residual Linear-GELU-Linear adapters after each native CLIP block."""

    def __init__(self, adapter_dim: int = 15):
        super().__init__()
        if adapter_dim <= 0:
            raise ValueError("adapter_dim must be positive")
        self.adapter_dim = int(adapter_dim)
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(WIDTH, self.adapter_dim), nn.GELU(),
                          nn.Linear(self.adapter_dim, WIDTH))
            for _ in self.clip_model.visual.transformer.resblocks])
        for adapter in self.adapters:
            nn.init.kaiming_uniform_(adapter[0].weight, a=math.sqrt(5))
            nn.init.zeros_(adapter[2].weight)
            nn.init.zeros_(adapter[2].bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        vit = self.clip_model.visual
        x = self._tokens(x)
        for block, adapter in zip(vit.transformer.resblocks, self.adapters):
            x = block(x)
            x = x + adapter(x)
        return vit.ln_post(x.permute(1, 0, 2)[:, 0, :])


class PartialFinetuneCLIPBackbone(FrozenCLIPBackbone):
    """Train the final visual blocks, ln_post, and classifier head."""

    def __init__(self, train_last_blocks: int = 4):
        super().__init__()
        blocks = self.clip_model.visual.transformer.resblocks
        if not 0 < train_last_blocks <= len(blocks):
            raise ValueError("train_last_blocks must be within the transformer depth")
        for block in blocks[-int(train_last_blocks):]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in self.clip_model.visual.ln_post.parameters():
            parameter.requires_grad = True


class FullFinetuneCLIPBackbone(FrozenCLIPBackbone):
    """Full visual adaptation with the 1024-dimensional pre-projection head.

    Visual proj is marked trainable as in the evaluated model definition, but
    is not used by forward and receives no gradient. Actual updated parameters
    exclude that tensor. The unused text tower remains frozen here.
    """

    def __init__(self):
        super().__init__()
        for parameter in self.clip_model.visual.parameters():
            parameter.requires_grad = True


def build_baseline(kind: str, **settings) -> nn.Module:
    if kind == "frozen":
        return FrozenCLIPBackbone()
    if kind == "bitfit":
        return FrozenCLIPBackbone(train_bias=True)
    if kind == "bottleneck_adapter":
        return AdapterCLIPBackbone(adapter_dim=int(settings.get("adapter_dim", 15)))
    if kind == "partial_finetune":
        return PartialFinetuneCLIPBackbone(train_last_blocks=int(settings.get("train_last_blocks", 4)))
    if kind == "full_finetune":
        return FullFinetuneCLIPBackbone()
    raise ValueError(f"unsupported baseline: {kind}")
