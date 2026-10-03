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
| `source_unit_level_auc.csv` | Source-validation frame- and analysis-unit AUC |
| `calibration_thresholds.csv`, `operating_point_summary.csv`, `prevalence_balanced_calibration.csv` | Calibration and source-fixed operating points |
| `efficiency.csv`, `efficiency_round_results.csv` | Five-round model-only efficiency benchmark |
| `merged_inference_benchmark.csv`, `merged_inference_benchmark.meta.json` | Eight-round merged/unmerged benchmark |
| `merged_equivalence_seed3407_cuda.json` | Merge-equivalence check on trained checkpoints |
| `failure_cases.csv` | De-identified qualitative case metadata |
| `predictions/analysis_unit_scores.parquet` | Target analysis-unit labels and scores |

All target datasets were reporting-only. Run `python analysis/validate_results.py` from the repository root to verify the public tables and prediction archive.
