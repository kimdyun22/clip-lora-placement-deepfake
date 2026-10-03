"""Manual CPU checks using the actual pretrained CLIP ViT-L/14 model."""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.clip_lora.baselines import build_baseline  # noqa: E402


def main():
    torch.set_num_threads(4)
    torch.manual_seed(3407)
    settings = yaml.safe_load((ROOT / "protocols/adaptation_baselines.yaml").read_text())
    x = torch.randn(1, 3, 224, 224)
    for kind, configuration in settings["models"].items():
        model = build_baseline(kind, **configuration).cpu().eval()
        with torch.no_grad():
            logits, features = model(x)
        assert logits.shape == (1, 2) and features.shape == (1, 1024), kind
        assert torch.isfinite(logits).all() and torch.isfinite(features).all(), kind
        selected = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if kind == "full_finetune":
            model.zero_grad(set_to_none=True)
            output, _ = model(x)
            output.square().sum().backward()
            active = sum(p.numel() for p in model.parameters() if p.grad is not None)
            assert model.clip_model.visual.proj.grad is None
            assert selected - active == 786432
        else:
            active = selected
        assert active == int(configuration["updated_parameters"]), (kind, active)
        if kind == "bitfit":
            biases = [p for n, p in model.clip_model.visual.named_parameters() if n.endswith("in_proj_bias")]
            assert len(biases) == 24 and sum(p.numel() for p in biases) == 73728
            assert all(not p.requires_grad for p in biases)
            assert all(p.requires_grad == n.endswith(".bias")
                       for n, p in model.clip_model.visual.named_parameters())
        print(f"PASS {kind}: logits=(1,2), features=(1,1024), updated_parameters={active}", flush=True)
        model.zero_grad(set_to_none=True)
        del model, logits, features
        if kind == "full_finetune":
            del output
        gc.collect()
    print("BASELINE_MODEL_TESTS_PASSED models=5")


if __name__ == "__main__":
    main()
