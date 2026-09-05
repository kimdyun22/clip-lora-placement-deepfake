"""CLIP ViT-L/14 with standard weight-space LoRA at four placements.

Each adapted matrix uses ``W' = W + (alpha / r) B A``.  For Q/V adaptation,
the key-projection rows remain frozen.  The MLP path applies the adapted
``c_fc`` weight before GELU and the adapted ``c_proj`` weight after GELU.
"""

from __future__ import annotations

import math

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
VALID_POSITIONS = ("attn_out", "attn_qv", "mlp", "mlp_attn_out")


def _lora_pair(fan_in: int, fan_out: int, rank: int) -> nn.ModuleDict:
    """A rank-r pair with the standard LoRA initialisation: A ~ kaiming, B = 0."""
    a = nn.Linear(fan_in, rank, bias=False)
    b = nn.Linear(rank, fan_out, bias=False)
    nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
    nn.init.zeros_(b.weight)
    return nn.ModuleDict({"A": a, "B": b})


def _delta(pair: nn.ModuleDict, scaling: float) -> torch.Tensor:
    """(alpha / r) . B A, shaped like the frozen weight it is added to."""
    return (pair["B"].weight @ pair["A"].weight) * scaling


class CLIPStandardLoRABackbone(nn.Module):
    """CLIP ViT-L/14 with weight-space LoRA at a selected insertion point."""

    def __init__(
        self,
        lora_rank: int = 16,
        lora_alpha: int = 64,
        lora_position: str = "attn_out",
        verbose: bool = True,
    ) -> None:
        super().__init__()
        if lora_position not in VALID_POSITIONS:
            raise ValueError(f"unsupported lora_position: {lora_position}")

        self.lora_position = lora_position
        self.lora_rank = int(lora_rank)
        self.lora_alpha = int(lora_alpha)
        self.scaling = self.lora_alpha / self.lora_rank

        self.clip_model, _ = clip.load("ViT-L/14", device="cpu")
        self.clip_model = self.clip_model.float()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        vit = self.clip_model.visual
        embed_dim = vit.transformer.width
        mlp_dim = embed_dim * 4

        self.lora_layers = nn.ModuleList()
        for _ in vit.transformer.resblocks:
            layer = nn.ModuleDict()
            if lora_position in ("mlp", "mlp_attn_out"):
                layer["fc"] = _lora_pair(embed_dim, mlp_dim, self.lora_rank)
                layer["proj"] = _lora_pair(mlp_dim, embed_dim, self.lora_rank)
            if lora_position in ("attn_out", "mlp_attn_out"):
                layer["attn_out"] = _lora_pair(embed_dim, embed_dim, self.lora_rank)
            if lora_position == "attn_qv":
                layer["q"] = _lora_pair(embed_dim, embed_dim, self.lora_rank)
                layer["v"] = _lora_pair(embed_dim, embed_dim, self.lora_rank)
            self.lora_layers.append(layer)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 2),
        )

        if verbose:
            self._report_params()

    def _report_params(self) -> None:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  [StandardLoRA] Position: {self.lora_position}, Rank: {self.lora_rank}")
        print(
            f"  [StandardLoRA] Total: {total / 1e6:.1f}M, "
            f"Trainable: {trainable / 1e6:.2f}M ({100 * trainable / total:.2f}%)"
        )

    def _renormalise(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """DeepfakeBench [-1, 1] tensors -> CLIP normalisation."""
        device = pixel_values.device
        mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
        return ((pixel_values * 0.5 + 0.5) - mean) / std

    def _attention(self, block: nn.Module, layer: nn.ModuleDict, x_ln1: torch.Tensor) -> torch.Tensor:
        in_proj_weight = block.attn.in_proj_weight
        out_proj_weight = block.attn.out_proj.weight

        if self.lora_position == "attn_qv":
            delta_q = _delta(layer["q"], self.scaling)
            delta_v = _delta(layer["v"], self.scaling)
            in_proj_weight = in_proj_weight + torch.cat(
                [delta_q, torch.zeros_like(delta_q), delta_v], dim=0
            )
        if self.lora_position in ("attn_out", "mlp_attn_out"):
            out_proj_weight = out_proj_weight + _delta(layer["attn_out"], self.scaling)

        attn_out, _ = F.multi_head_attention_forward(
            query=x_ln1,
            key=x_ln1,
            value=x_ln1,
            embed_dim_to_check=x_ln1.shape[-1],
            num_heads=block.attn.num_heads,
            in_proj_weight=in_proj_weight,
            in_proj_bias=block.attn.in_proj_bias,
            bias_k=block.attn.bias_k,
            bias_v=block.attn.bias_v,
            add_zero_attn=block.attn.add_zero_attn,
            dropout_p=0.0,
            out_proj_weight=out_proj_weight,
            out_proj_bias=block.attn.out_proj.bias,
            training=self.training,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
        )
        return attn_out

    def _mlp(self, block: nn.Module, layer: nn.ModuleDict, x_ln2: torch.Tensor) -> torch.Tensor:
        """Standard weight-space LoRA on c_fc and c_proj.

        h = (W_fc + s B_f A_f) x
        a = GELU(h)
        y = (W_proj + s B_p A_p) a
        """
        if self.lora_position not in ("mlp", "mlp_attn_out"):
            return block.mlp(x_ln2)

        c_fc, c_proj = block.mlp.c_fc, block.mlp.c_proj
        h = F.linear(x_ln2, c_fc.weight + _delta(layer["fc"], self.scaling), c_fc.bias)
        a = block.mlp.gelu(h)
        return F.linear(a, c_proj.weight + _delta(layer["proj"], self.scaling), c_proj.bias)

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self._renormalise(pixel_values)
        vit = self.clip_model.visual

        x = vit.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = vit.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1) + vit.positional_embedding.to(x.dtype)
        x = vit.ln_pre(x).permute(1, 0, 2)

        for block, layer in zip(vit.transformer.resblocks, self.lora_layers):
            x = x + self._attention(block, layer, block.ln_1(x))
            x = x + self._mlp(block, layer, block.ln_2(x))

        x = x.permute(1, 0, 2)
        return vit.ln_post(x[:, 0, :])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        return self.head(feat), feat
