# Parameter-matched LoRA placement reveals performance-efficiency trade-offs in cross-dataset deepfake detection

This repository contains the code, protocols, and public result artifacts for a controlled study of standard weight-space LoRA placement in CLIP ViT-L/14. The primary experiment compares Attn-Out, Attn-Q/V, MLP, and MLP+Attn-Out with exactly 1,478,658 trainable parameters and three training seeds. A complementary rank-16 comparison identifies Attn-Q/V as the numerical performance-oriented configuration and Attn-Out as the compact-adaptation and lower-training-resource configuration under the evaluated conditions.

## Key results

- Attn-Q/V achieved the highest numerical six-dataset macro AUC in the exact-budget comparison.
- Attn-Out used less measured training memory and model-compute time.
- At rank 16, Attn-Out used 49.9% fewer trainable parameters than Attn-Q/V, with a source-validation AUC difference of 0.000659.
- Explicit weight merging removed placement-specific low-rank inference operations.

## Quick start

Python 3.10 and CUDA 12.1 were used for the reported experiments.

```bash
python -m pip install -r requirements.txt
python -m compileall -q src analysis tests
python analysis/validate_results.py
```

## Repository contents

| Path | Contents |
|---|---|
| `src/clip_lora/` | Model, training, evaluation, and metric code |
| `protocols/` | Training, selection, analysis, and efficiency settings |
| `results/` | Canonical tables and de-identified prediction scores |
| `analysis/` | Analysis builders and public result validation |
| `tests/` | LoRA correctness, parameter-count, and merge-equivalence tests |
| `datasets/` | Dataset setup and published frame manifests |

## Training and evaluation

The primary configuration is defined in [`protocols/primary.yaml`](protocols/primary.yaml). Dataset access and manifest usage are described in [`datasets/README.md`](datasets/README.md).

```bash
python -m src.clip_lora.train --protocol protocols/primary.yaml \
  --candidate attn_qv --seed 3407 --device cuda:0 \
  --output-root runs --dataset-root /path/to/datasets

python -m src.clip_lora.evaluate \
  --manifest datasets/manifests/cdfv2_test.jsonl.gz \
  --dataset-root /path/to/datasets \
  --checkpoint runs/attn_qv__seed3407/epoch_10.pth \
  --lora-position attn_qv --lora-rank 15 --lora-alpha 60 \
  --run-name attn_qv_cdf --output-root evaluations --device cuda:0
```

## Public results

The canonical tables are listed in [`results/README.md`](results/README.md). The de-identified archive at `results/predictions/analysis_unit_scores.parquet` contains labels and scores for four placements, three seeds, and six reporting-only target datasets. Original dataset files are not included.

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Repository code is licensed under Apache-2.0; third-party software and datasets remain subject to their original terms.
