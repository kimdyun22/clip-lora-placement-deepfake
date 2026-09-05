"""Live attention, combined-path, and native frozen-parity checks."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.clip_lora.model import CLIPStandardLoRABackbone, _delta  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    results.append((name, PASS if ok else FAIL, detail))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")


def batch(n: int = 2, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, 3, 224, 224)


def make_live(model: CLIPStandardLoRABackbone, std: float = 0.02) -> None:
    """Move every B off zero so every LoRA branch actually contributes."""
    for layer in model.lora_layers:
        for pair in layer.values():
            nn.init.normal_(pair["B"].weight, std=std)


def gate_nonzero_attn_out_equality() -> None:
    """out_proj path must equal an explicitly materialised (W + s.BA) linear layer."""
    torch.manual_seed(2)
    rank, alpha = 8, 32
    model = CLIPStandardLoRABackbone(lora_rank=rank, lora_alpha=alpha, lora_position="attn_out", verbose=False)
    model.eval()
    make_live(model)

    block, layer = model.clip_model.visual.transformer.resblocks[0], model.lora_layers[0]
    x_ln1 = torch.randn(7, 3, 1024)  # (seq, batch, embed), matches internal (L,N,D) layout
    with torch.no_grad():
        got = model._attention(block, layer, x_ln1)
        materialised_out_proj = block.attn.out_proj.weight + _delta(layer["attn_out"], model.scaling)
        want, _ = F.multi_head_attention_forward(
            query=x_ln1, key=x_ln1, value=x_ln1,
            embed_dim_to_check=x_ln1.shape[-1], num_heads=block.attn.num_heads,
            in_proj_weight=block.attn.in_proj_weight, in_proj_bias=block.attn.in_proj_bias,
            bias_k=block.attn.bias_k, bias_v=block.attn.bias_v, add_zero_attn=block.attn.add_zero_attn,
            dropout_p=0.0, out_proj_weight=materialised_out_proj, out_proj_bias=block.attn.out_proj.bias,
            training=False, key_padding_mask=None, need_weights=False, attn_mask=None,
        )
    delta = float((got - want).abs().max())
    record("nonzero-B weight-space equality [attn_out]", delta < 1e-5, f"max|delta| = {delta:.3e}")
    del model


def gate_nonzero_attn_qv_equality() -> None:
    """Q/V-projected in_proj path must equal an explicitly materialised weight."""
    torch.manual_seed(3)
    rank, alpha = 8, 32
    model = CLIPStandardLoRABackbone(lora_rank=rank, lora_alpha=alpha, lora_position="attn_qv", verbose=False)
    model.eval()
    make_live(model)

    block, layer = model.clip_model.visual.transformer.resblocks[0], model.lora_layers[0]
    x_ln1 = torch.randn(7, 3, 1024)
    with torch.no_grad():
        got = model._attention(block, layer, x_ln1)
        delta_q = _delta(layer["q"], model.scaling)
        delta_v = _delta(layer["v"], model.scaling)
        materialised_in_proj = block.attn.in_proj_weight + torch.cat(
            [delta_q, torch.zeros_like(delta_q), delta_v], dim=0
        )
        want, _ = F.multi_head_attention_forward(
            query=x_ln1, key=x_ln1, value=x_ln1,
            embed_dim_to_check=x_ln1.shape[-1], num_heads=block.attn.num_heads,
            in_proj_weight=materialised_in_proj, in_proj_bias=block.attn.in_proj_bias,
            bias_k=block.attn.bias_k, bias_v=block.attn.bias_v, add_zero_attn=block.attn.add_zero_attn,
            dropout_p=0.0, out_proj_weight=block.attn.out_proj.weight, out_proj_bias=block.attn.out_proj.bias,
            training=False, key_padding_mask=None, need_weights=False, attn_mask=None,
        )
    delta = float((got - want).abs().max())
    record("nonzero-B weight-space equality [attn_qv]", delta < 1e-5, f"max|delta| = {delta:.3e}")

    # K must stay exactly frozen: the K third of in_proj_weight is untouched by delta_q/delta_v.
    embed_dim = block.attn.in_proj_weight.shape[0] // 3
    k_slice = slice(embed_dim, 2 * embed_dim)
    k_delta = float((materialised_in_proj[k_slice] - block.attn.in_proj_weight[k_slice]).abs().max())
    record("K row untouched [attn_qv]", k_delta == 0.0, f"max|delta| = {k_delta:.3e}")
    del model


def gate_nonzero_mlp_attn_out_combined() -> None:
    """mlp_attn_out: both the MLP and out_proj branches live at once, one full block."""
    torch.manual_seed(4)
    rank, alpha = 4, 16
    model = CLIPStandardLoRABackbone(lora_rank=rank, lora_alpha=alpha, lora_position="mlp_attn_out", verbose=False)
    model.eval()
    make_live(model)

    block, layer = model.clip_model.visual.transformer.resblocks[0], model.lora_layers[0]
    scaling = model.scaling
    x = torch.randn(5, 3, 1024)
    with torch.no_grad():
        x_ln1 = block.ln_1(x)
        attn_out = model._attention(block, layer, x_ln1)
        materialised_out_proj = block.attn.out_proj.weight + _delta(layer["attn_out"], scaling)
        want_attn, _ = F.multi_head_attention_forward(
            query=x_ln1, key=x_ln1, value=x_ln1,
            embed_dim_to_check=x_ln1.shape[-1], num_heads=block.attn.num_heads,
            in_proj_weight=block.attn.in_proj_weight, in_proj_bias=block.attn.in_proj_bias,
            bias_k=block.attn.bias_k, bias_v=block.attn.bias_v, add_zero_attn=block.attn.add_zero_attn,
            dropout_p=0.0, out_proj_weight=materialised_out_proj, out_proj_bias=block.attn.out_proj.bias,
            training=False, key_padding_mask=None, need_weights=False, attn_mask=None,
        )
        x_mid = x + attn_out

        x_ln2 = block.ln_2(x_mid)
        got_mlp = model._mlp(block, layer, x_ln2)
        c_fc, c_proj = block.mlp.c_fc, block.mlp.c_proj
        w_fc = c_fc.weight + _delta(layer["fc"], scaling)
        w_pr = c_proj.weight + _delta(layer["proj"], scaling)
        want_mlp = F.linear(block.mlp.gelu(F.linear(x_ln2, w_fc, c_fc.bias)), w_pr, c_proj.bias)

    delta_attn = float((attn_out - want_attn).abs().max())
    delta_mlp = float((got_mlp - want_mlp).abs().max())
    record(
        "nonzero-B combined block equality [mlp_attn_out]",
        delta_attn < 1e-5 and delta_mlp < 1e-5,
        f"attn max|delta| = {delta_attn:.3e}, mlp max|delta| = {delta_mlp:.3e}",
    )
    del model


def gate_native_frozen_parity() -> None:
    """At B = 0, extract_features must match CLIP's own unmodified block.forward,
    called directly (nn.MultiheadAttention module, not our reimplementation) --
    an independent reference this module's own code never touches."""
    x = batch(seed=5)
    for pos in ("attn_out", "attn_qv", "mlp", "mlp_attn_out"):
        model = CLIPStandardLoRABackbone(lora_rank=8, lora_alpha=32, lora_position=pos, verbose=False)
        model.eval()
        # B is zero at init already; this gate does not touch it.
        vit = model.clip_model.visual
        with torch.no_grad():
            adapted = model.extract_features(x)

            ref = model._renormalise(x)
            ref = vit.conv1(ref)
            ref = ref.reshape(ref.shape[0], ref.shape[1], -1).permute(0, 2, 1)
            cls = vit.class_embedding.to(ref.dtype) + torch.zeros(
                ref.shape[0], 1, ref.shape[-1], dtype=ref.dtype, device=ref.device
            )
            ref = torch.cat([cls, ref], dim=1) + vit.positional_embedding.to(ref.dtype)
            ref = vit.ln_pre(ref).permute(1, 0, 2)
            for block in vit.transformer.resblocks:
                ref = block(ref)  # CLIP's own ResidualAttentionBlock.forward, untouched
            ref = ref.permute(1, 0, 2)
            native = vit.ln_post(ref[:, 0, :])

        delta = float((adapted - native).abs().max())
        record(f"native frozen CLIP parity [{pos}]", delta < 1e-4, f"max|delta| = {delta:.3e}")
        del model


def main() -> int:
    print("Supplementary standard-LoRA verification gates\n")
    for gate in (
        gate_nonzero_attn_out_equality,
        gate_nonzero_attn_qv_equality,
        gate_nonzero_mlp_attn_out_combined,
        gate_native_frozen_parity,
    ):
        gate()
    failed = [r for r in results if r[1] == FAIL]
    print(f"\n{len(results) - len(failed)}/{len(results)} supplementary gates passed")
    if failed:
        print("FAILED:")
        for name, _, detail in failed:
            print(f"  - {name}: {detail}")
        return 1
    print("ALL_SUPPLEMENTARY_GATES_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
