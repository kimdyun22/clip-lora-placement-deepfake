# Parameter-matched LoRA placement reveals performance-efficiency trade-offs in cross-dataset deepfake detection

This repository accompanies a controlled comparison of standard weight-space LoRA placements in CLIP ViT-L/14. Attn-Out, Attn-Q/V, MLP, and MLP+Attn-Out are matched at exactly 1,478,658 trainable parameters. Models are trained on FaceForensics++ C23; Celeb-DF-v2, DFDC, DFDCP, UADFV, DeeperForensics-1.0, and WildDeepfake are reporting-only targets. The results show a performance-efficiency trade-off and do not establish a universally optimal placement.

## Repository structure

```text
datasets/       data-access guidance and the published frame manifests
environment/    verbatim package list from the container that produced the results
protocols/      frozen primary, selection, analysis, and efficiency settings
results/        canonical numerical artifacts used by the paper
analysis/       public provenance builders and non-training verification scripts
src/clip_lora/  model, data, training, evaluation, metrics, and analysis code,
                each runnable as a module entry point
tests/          standard-LoRA correctness and parameter-count checks
```

## Installation

Python 3.10 and CUDA 12.1 were used for the reported experiments.

```bash
python -m pip install -r requirements.txt
```

All direct scientific dependencies, including the CUDA 12.1 PyTorch wheels, are pinned in the single `requirements.txt`, and pip solves them in one resolution pass. Transitive dependencies are not individually pinned; the complete package state recorded from the experimental container is in [environment/container_pip_freeze.txt](environment/container_pip_freeze.txt). The provided `Dockerfile` runs the same command.

The pins match the container that produced the reported results; its verbatim `pip freeze` is in [environment/](environment/), together with the one documented deviation for `opencv-python`.

## Dataset setup

Obtain all datasets and DeepfakeBench-compatible preprocessed RGB frames independently. See [datasets/README.md](datasets/README.md) for the expected directory layout and for the upstream DeepfakeBench and Self-Blended Images revisions used. Training with pSBI also requires a separately obtained Self-Blended Images `SBI_API`; expose its module through `PYTHONPATH` or `CLIP_LORA_SBI_MODULE`.

The exact frame lists are published under [datasets/manifests/](datasets/manifests/). Frame paths there are dataset-relative, so pass `--dataset-root` pointing at the directory that holds your dataset copies. Manifests that store absolute paths continue to work without that flag.

## Primary configurations

The frozen configuration is [protocols/primary.yaml](protocols/primary.yaml). All four placements use `alpha/r = 4`, `p_SBI = 0.25`, and seeds 3407, 7859, and 12011:

| Placement | Rank | Alpha | Trainable parameters |
|---|---:|---:|---:|
| Attn-Out | 30 | 120 | 1,478,658 |
| Attn-Q/V | 15 | 60 | 1,478,658 |
| MLP | 6 | 24 | 1,478,658 |
| MLP+Attn-Out | 5 | 20 | 1,478,658 |

## Training

The training manifest is declared in `protocols/primary.yaml` and ships in this repository:

```bash
python -m src.clip_lora.train --protocol protocols/primary.yaml --candidate attn_qv --seed 3407 --device cuda:0 --output-root runs --dataset-root /path/to/datasets
```

## Evaluation

```bash
python -m src.clip_lora.evaluate --manifest datasets/manifests/cdfv2_test.jsonl.gz --dataset-root /path/to/datasets --checkpoint runs/attn_qv__seed3407/epoch_10.pth --lora-position attn_qv --lora-rank 15 --lora-alpha 60 --run-name attn_qv_cdf --output-root evaluations --device cuda:0
```

Evaluation writes frame- and analysis-unit prediction Parquet files, metrics, metadata, and a `VALIDATED` marker. Target results are reporting only.

## Reported analyses and efficiency

```bash
python -m src.clip_lora.analysis --config protocols/analysis.yaml --results results
python analysis/audit_public_results.py
```

The analysis command validates the canonical public tables.

The public prediction archive at `results/predictions/analysis_unit_scores.parquet` contains de-identified analysis-unit labels and scores for all four placements, three seeds, and six reporting-only target datasets. Its checksum and completeness are checked by the public audit command.

## Correctness tests

The tests cover zero-initialized identity, explicit weight-space equality, live gradients, attention parity, an unchanged K projection for Q/V LoRA, the combined MLP+Attn-Out path, and analytic parameter counts.

```bash
python tests/test_parameter_counts.py
python tests/test_lora.py
```

These full tests instantiate CLIP ViT-L/14 and require its model weights.

## Third-party components

OpenAI CLIP is installed from its upstream repository and remains subject to its own license. This repository does not redistribute DeepfakeBench, Self-Blended Images code, third-party datasets, checkpoints, or arrangement files. Users must obtain those components independently and comply with their original licenses and access terms. The repository's own code is licensed under Apache-2.0.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff).
