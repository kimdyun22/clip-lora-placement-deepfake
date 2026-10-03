# Results

Canonical machine-readable results used in the manuscript and Supplementary Information:

| Files | Contents |
|---|---|
| `primary_results.csv`, `primary_seed_results.csv` | Exact-budget four-placement results |
| `selection_sensitivity.csv`, `selection_seed_results.csv` | Source-only selection and common-rank sensitivity |
| `adaptation_baselines.csv`, `adaptation_baseline_seed_results.csv` | Three-seed CLIP adaptation baselines |
| `supplementary_adaptation_references.csv` | Single-seed descriptive adaptation references in Supplementary Table S3 |
| `external_reproduction_audit.csv` | Method-native Forensics Adapter checkpoint reproduction in Supplementary Table S3 |
| `statistics.csv` | Checkpoint- and dataset-conditional DeLong/Holm diagnostics |
| `macro_sensitivity.csv`, `hierarchical_bootstrap_crossed.csv`, `uncertainty_decomposition.csv` | Aggregation and crossed-uncertainty analyses |
| `source_unit_level_auc.csv` | Source-validation frame- and identifier-group AUC |
| `source_per_video_auc.csv`, `source_per_video_auc_summary.csv` | Source frame, identifier-group, and individual-video AUC: 24 seed rows and eight configuration summaries |
| `dfdcp_video_sensitivity.csv` | Individual-video sensitivity for 12 primary DFDCP checkpoints; primary ID-group AUC is retained |
| `calibration_thresholds.csv`, `operating_point_summary.csv`, `prevalence_balanced_calibration.csv` | Calibration and source-fixed operating points |
| `efficiency.csv`, `efficiency_round_results.csv` | Five-round model-only efficiency benchmark |
| `figure2_training_resources.csv` | Training step time, allocated memory, and Macro-6 coordinates for Figure 2 |
| `merged_inference_benchmark.csv`, `merged_inference_benchmark.meta.json` | Eight-round merged/unmerged benchmark |
| `merged_equivalence_seed3407_cuda.json` | Merge-equivalence check on trained checkpoints |
| `error_overlap_seed_results.csv`, `error_overlap_summary.csv`, `error_overlap_source_thresholds.csv` | Dataset/seed error counts, conditional overlap proportions, and model/seed source thresholds for Figure 3 |
| `failure_cases.csv` | De-identified case metadata for source-fixed prediction diagnostics |
| `predictions/analysis_unit_scores.parquet` | Target analysis-unit labels and scores |

Figure 3 quantities are calculated separately for each dataset and seed, then summarized across three seeds. The any-error rate uses all evaluated units as its denominator; conditional overlap proportions use only units misclassified by at least one placement.

All target datasets were reporting-only. Run `python analysis/validate_results.py` from the repository root to verify the public tables and prediction archive.

Analysis-unit definitions are in [`datasets/README.md`](../datasets/README.md). The source and DFDCP grouping sensitivities can be recalculated with the following command only when the private per-run `frame_predictions.parquet` files are available; the public target archive is already aggregated and cannot recover individual-video scores.

```bash
python analysis/build_analysis_unit_sensitivity.py \
  --source-root /path/to/source_runs \
  --target-root /path/to/target_evaluations --output results
```

The source root contains `four_way/evaluations/` and `five_candidate/evaluations/`; the target root contains the primary run directories. Each run directory is named `<config_id>__seed<seed>`. Public validation recomputes summary statistics from the released seed rows and checks consistency with the primary tables; it does not regenerate these AUCs without the private frame predictions. Full fine-tuning's updated-parameter count excludes the unused visual projection; BitFit retains the reported bias subset with fused QKV input biases frozen.
