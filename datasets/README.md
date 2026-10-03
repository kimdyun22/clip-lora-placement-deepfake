# Dataset setup

This repository does not redistribute third-party datasets. FaceForensics++ C23 is the source domain. Celeb-DF-v2, DFDC, DFDCP, UADFV, DeeperForensics-1.0, and WildDeepfake are reporting-only targets.

Obtain the datasets and DeepfakeBench-preprocessed RGB frames independently and comply with their original licenses and access terms. The reported setup used DeepfakeBench `v1.0.3-106-gf188b1c` and Self-Blended Images commit `d6d9a351f32334ae9a3a250872c05836e0dd0f2c`.

## Published manifests

The compressed JSONL files in `manifests/` contain the exact frame lists used by the experiments. Paths are relative to a user-supplied dataset root.

| Manifest | Role | Frames | Analysis units |
|---|---|---:|---:|
| `ffpp_train.jsonl.gz` | FF++ C23 training | 114,884 | 1,439 |
| `ffpp_val.jsonl.gz` | FF++ C23 validation | 22,354 | 280 |
| `cdfv2_test.jsonl.gz` | Celeb-DF-v2 | 16,420 | 518 |
| `dfdc_test.jsonl.gz` | DFDC | 132,116 | 4,704 |
| `dfdcp_test.jsonl.gz` | DFDCP | 17,222 | 652 |
| `uadfv_test.jsonl.gz` | UADFV | 3,099 | 98 |
| `df1_test.jsonl.gz` | DeeperForensics-1.0 | 1,062,580 | 40,870 |
| `wdf_test.jsonl.gz` | WildDeepfake | 5,024 | 157 |

## Local layout

```text
<dataset-root>/
  FaceForensics++/
  Celeb-DF-v2/
  DFDC/
  DFDCP/
  UADFV/
  DeeperForensics-1.0/
  WildDeepfake/
```

Pass this directory with `--dataset-root`:

```bash
python -m src.clip_lora.evaluate \
  --manifest datasets/manifests/cdfv2_test.jsonl.gz \
  --dataset-root /path/to/datasets \
  ...
```

## SBI training dependency

pSBI training requires landmark arrays produced by DeepfakeBench preprocessing and the DeepfakeBench-adapted SBI API used by the reported runs. Obtain DeepfakeBench and check out the recorded commit:

```bash
git clone https://github.com/SCLBD/DeepfakeBench.git
cd DeepfakeBench
git checkout f188b1c105465e2e5377eb536a95022ae0e4522d
export PYTHONPATH="$PWD:${PYTHONPATH}"
export CLIP_LORA_SBI_MODULE=training.dataset.sbi_api
```

The loaded adapter file was `training/dataset/sbi_api.py` with SHA-256
`2e79f715438da6a9dc717a3f6efcca27b6153beb572a551a17be2c3c444fb65a`.
It derives from the upstream Self-Blended Images implementation at commit
`d6d9a351f32334ae9a3a250872c05836e0dd0f2c`; the corresponding upstream
`src/utils/sbi.py` SHA-256 was
`2a3e4baa51ed99edcd9ae6feebf964ad790dd6b1a6388306be2192a530f308e3`.

Before training, verify the import from the repository root in the CUDA-enabled
environment used for training:

```bash
python -c "import importlib, os; m=importlib.import_module(os.environ['CLIP_LORA_SBI_MODULE']); assert hasattr(m, 'SBI_API'); print('SBI_API_IMPORT_PASS')"
```

Every real source frame eligible for SBI must have the corresponding DeepfakeBench landmark array at the path implied by the preprocessed RGB layout. Training aborts if the configured module cannot be imported.
