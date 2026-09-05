# Dataset access and manifests

This repository does not redistribute third-party datasets. The source-domain
experiments use FaceForensics++ C23. Reporting-only target datasets are
Celeb-DF-v2, DFDC, DFDCP, UADFV, DeeperForensics-1.0, and WildDeepfake.

Obtain the original datasets, the DeepfakeBench-preprocessed RGB data, the
landmarks needed for pSBI, and the data-arrangement files independently. Users
are responsible for complying with every dataset and provider's license, access
agreement, and usage terms.

## Published manifests

`manifests/` contains the exact frame lists used for training and evaluation.
Each row keeps the sample, video and sequence identity, the frame selection and
index, the label, the split, and the analysis-unit key, in the original row
order. Only the frame path differs from the frozen internal manifests: the
machine-local dataset root has been replaced by a dataset-relative path so the
lists are portable.

| Manifest | Role | Rows | Analysis units |
|---|---|---:|---:|
| `ffpp_train.jsonl.gz` | FF++ C23 source training | 114,884 | 1,439 |
| `ffpp_val.jsonl.gz` | FF++ C23 source validation | 22,354 | 280 |
| `cdfv2_test.jsonl.gz` | Celeb-DF-v2 target | 16,420 | 518 |
| `dfdc_test.jsonl.gz` | DFDC target | 132,116 | 4,704 |
| `dfdcp_test.jsonl.gz` | DFDCP target | 17,222 | 652 |
| `uadfv_test.jsonl.gz` | UADFV target | 3,099 | 98 |
| `df1_test.jsonl.gz` | DeeperForensics-1.0 target | 1,062,580 | 40,870 |
| `wdf_test.jsonl.gz` | WildDeepfake target | 5,024 | 157 |

The files are gzip-compressed JSONL: the uncompressed DeeperForensics-1.0 frame
list alone is about 435 MB, which exceeds the GitHub per-file limit. Both
loaders read `.jsonl` and `.jsonl.gz` transparently, so decompressing them first
is optional.

## Expected local layout

Frame paths are relative to a single dataset root that you supply with
`--dataset-root`. Create or symlink the following directories under it:

```text
<dataset-root>/
  FaceForensics++/       manipulated_sequences/ , original_sequences/
  Celeb-DF-v2/           Celeb-real/ , Celeb-synthesis/ , YouTube-real/
  DFDC/                  test/frames/
  DFDCP/                 method_A/ , method_B/ , original_videos/
  UADFV/                 fake/ , real/
  DeeperForensics-1.0/   manipulated_videos/ , source_videos/
  WildDeepfake/          fake_test/ , real_test/
```

For example:

```bash
python -m src.clip_lora.evaluate \
  --manifest datasets/manifests/cdfv2_test.jsonl.gz \
  --dataset-root /path/to/datasets \
  ...
```

Absolute frame paths are still honoured unchanged, so manifests recorded with
machine-local paths keep working without `--dataset-root`.

pSBI training additionally reads a landmark file beside each frame, obtained by
replacing `/frames/` with `/landmarks/` and the image suffix with `.npy`. Those
landmark arrays come from the DeepfakeBench preprocessing step and are not
redistributed here.

## Upstream component provenance

The recorded experiments used the following third-party sources.

**DeepfakeBench** — preprocessed RGB data, data-arrangement files, and the
`bi_online_generation` hull routine used by the pSBI adapter.

- Repository: `https://github.com/SCLBD/DeepfakeBench.git`
- Commit: `f188b1c105465e2e5377eb536a95022ae0e4522d` (`git describe`: `v1.0.3-106-gf188b1c`)
- Arrangement files used: `FaceForensics++.json`, `Celeb-DF-v2.json`, `DFDC.json`,
  `DFDCP.json`, `UADFV.json`, `DeeperForensics-1.0.json`
- WildDeepfake is not covered by a DeepfakeBench arrangement file; it was read
  directly from the distributed `deepfake_in_the_wild` sequence layout, which is
  why its analysis unit is a sequence rather than a video id.
- The working copy carried local edits to `preprocessing/config.yaml`,
  `preprocessing/rearrange.py`, `training/config/train_config.yaml`,
  `training/config/test_config.yaml`, and `training/dataset/__init__.py`, which
  set local paths and register the dataset entry points.

**Self-Blended Images** — the pSBI augmentation used during source training.

- Repository: `https://github.com/mapooon/SelfBlendedImages.git`
- Commit: `d6d9a351f32334ae9a3a250872c05836e0dd0f2c`
- The `sbi_api` module supplied at runtime is an adaptation of upstream
  `src/utils/sbi.py`, rewritten
  to expose a callable API and to source its convex hull from the DeepfakeBench
  `bi_online_generation` helper. It is not redistributed here because it derives
  from third-party code; supply it through `PYTHONPATH` or
  `CLIP_LORA_SBI_MODULE`.
- Two distinct SHA-256 values are relevant. The upstream file it was adapted
  from is
  `2a3e4baa51ed99edcd9ae6feebf964ad790dd6b1a6388306be2192a530f308e3`.
  The adapted `sbi_api.py` actually loaded by the reported training runs is
  `2e79f715438da6a9dc717a3f6efcca27b6153beb572a551a17be2c3c444fb65a`,
  which is also recorded in the `train_metadata.json` written beside each
  checkpoint. Use the second value to confirm that a locally supplied module
  matches the one used here.