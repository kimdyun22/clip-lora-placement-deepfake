# Results

This directory contains the numerical results reported in the manuscript and Supplementary Information.

- `primary_results.csv`: parameter-matched four-way LoRA comparison
- `primary_seed_results.csv`: twelve seed-level rows from which the primary means and sample SDs are recomputed
- `adaptation_baselines.csv`: three-seed CLIP adaptation baselines
- `adaptation_baseline_seed_results.csv`: nine seed-level rows from which the adaptation-baseline means and sample SDs are recomputed
- `supplementary_adaptation_references.csv`: single-seed descriptive references
- `selection_sensitivity.csv`: source-only selection and rank/budget sensitivity at fixed `alpha/r = 4`
- `selection_seed_results.csv`: fifteen source-validation seed rows supporting the selection and sensitivity summary
- `lora_sbi_2x2.csv`: source-frame-AUC seed rows and cohort provenance for the Frozen/LoRA by no-SBI/SBI control
- `statistics.csv`: seed-conditional paired statistical analyses
- `bootstrap_ci_summary.csv`: pair-by-dataset summary of the existing seed-conditional bootstrap intervals; it does not pool seeds
- `calibration_thresholds.csv`: calibration and source-fixed operating points
- `operating_point_summary.csv`: three-seed mean and sample SD for F1, low-FPR TPR, descriptive target EER, ECE, and Brier score
- `failure_cases.csv`: de-identified metadata for representative failure examples
- `efficiency.csv`: final RTX 4090 model-only efficiency measurements
- `efficiency_round_results.csv`: twenty raw hardware-round summaries supporting the efficiency means and sample SDs
- `external_reproduction_audit.csv`: method-native Forensics Adapter checkpoint reference used as a reproducibility sanity check, reporting published versus local video AUC under the method's own recipe
- `predictions/analysis_unit_scores.parquet`: de-identified primary analysis-unit labels and scores (4 placements x 3 seeds x 6 targets)
- `predictions/SHA256SUMS.txt`: checksum for the prediction archive

All external datasets were used for reporting only and were not used for model or hyperparameter selection.

The Attn-Out r16/alpha64/pSBI=0.25 row in `lora_sbi_2x2.csv` belongs to the confirmatory factorial cohort. It is not the corrected source-selection cohort represented by the similarly named row in `selection_sensitivity.csv`; their independently trained checkpoints yield different three-seed summaries and must not be merged.
The path-sanitized frozen record for this control is `protocols/lora_sbi_2x2.yaml`; it retains the original protocol hash, manifest hashes, and trainer/evaluator lineage hashes.

Run `python analysis/audit_public_results.py` from the repository root to verify aggregate reconstruction, Holm counts, operating-point and bootstrap summaries, the 2x2 control, prediction completeness, and absence of local absolute paths in public CSV files.
