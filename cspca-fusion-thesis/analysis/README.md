# Reproducibility package: machine learning prediction of clinically significant prostate cancer from fused clinical and MRI features

Supplementary material to the thesis. It contains the scripts that build the derived tables, extract the imaging features, fit the models,
and run every audit, baseline and verification analysis reported in Chapters 3 to 5 and Appendices A to I.

## What this package is, and is not
- The analysis scripts (steps 7 to 14) are the code that produced the reported audit, baseline, external-validation and verification results.
- The data-preparation scripts (steps 1 to 6) were recovered from the project notebook history and written as scripts. They reproduce the derived-file shapes listed below,
  but they were not re-run end to end on the original images when the package was assembled. Compare your outputs with the expected shapes and values.
- `make_synthetic_data.py` writes a small SYNTHETIC dataset in the same file layout, so the scripts can be smoke-tested without the real data (results are meaningless clinically). Use `--n-components 20` in step 3 with it, and run steps 1 and 3 before steps 8 to 12.
- Known deviations from a clean pipeline are listed in thesis Appendix I and below. Please read them before judging a mismatch.

## Data (public, not redistributed here)
| Data | Source |
|---|---|
| PI-CAI public training and development dataset (1,500 studies, 1,476 patients) | Zenodo, doi 10.5281/zenodo.6624726 (images); annotations and clinical information: github.com/DIAGNijmegen/picai_labels |
| Prostate-MRI-US-Biopsy (external cohort) | The Cancer Imaging Archive, doi 10.7937/TCIA.2020.A61IOC1A (T2-weighted MRI plus the Biopsy Data and Target Data spreadsheets) |

Put the downloads under one folder and set `CSPCA_ROOT` to it (default: the current directory). Expected layout:
`picai_labels/clinical_information/marksheet.csv`, `picai_images/<patient_id>/<patient_id>_<study_id>_t2w.mha`, and the TCIA DICOM folder for the external cohort.

## Environment
`pip install -r requirements.txt` (Python 3.11). Run `python 00_environment_report.py` to compare versions and file fingerprints with thesis Table 3.10 and Table D.1.
All stochastic steps use seed 42 (bootstrap generators: seed 7 in step 10, seed 0 in step 11). TensorFlow GPU inference can change feature values in the last decimals.

## Run order
| Step | Script | Output | Expected check |
|---|---|---|---|
| 0 | `00_environment_report.py` | printed report | versions and shapes as in the thesis |
| 1 | `01_build_picai_clinical_table.py` | `picai_clinical_clean.csv` | 1,500 rows, 425 csPCa, no missing values |
| 2 | `02_extract_picai_features.py --images picai_images --manifest selected_t2w_files.csv` | `cnn_features_t2w.csv` | 1,476 rows x 2,049 columns |
| 3 | `03_build_fused_dataset.py` | `fused_final_dataset.csv`, `fused_final_dataset_with_pirads.csv` | 1,500 x 106 and 1,500 x 107; PCA keeps about 0.71 of the variance |
| 4 | `04_build_external_cohort.py --biopsy-file ... --target-file ... --dicom-root ...` | `external_validation/external_clinical_clean.csv`, `final_external_cohort.csv` | 1,150 x 7 and 782 x 9; prevalence 0.575 |
| 5 | `05_extract_external_features.py` | `external_validation/external_cnn_features_t2w.csv` | 782 x 2,049 |
| 6 | `06_train_algorithms.py --config A` (also B, C) | `results_config_*.csv`, predictions | see below |
| 7 | `07_locked_model_external.py [--basis unique|merged]` | locked-model predictions | internal 79/6/52/163; external 313/137/122/210 (unique basis) |
| 8 | `08_leakage_audit.py` | printed audit | 8 overlapping patients; ROC-AUC 0.884, 0.885, 0.891, 0.874, 0.883 |
| 9 | `09_leave_one_centre_out.py` | `results_loco.csv` | fused ROC-AUC 0.732 / 0.585 / 0.779 (RUMC / PCNN / ZGT) |
| 10 | `10_baselines_statistics.py` (set `CSPCA_FAST=1` for a quick test) | printed tables | logistic regression 0.757; with PI-RADS and PSA density 0.878; locked 0.884 |
| 11 | `11_external_investigation.py` | printed analysis | drop 0.138 (0.090 to 0.187); score alone 0.857 vs 0.703 |
| 12 | `12_pca_sensitivity.py` | printed table | fused ROC-AUC 0.790 at 100 components, 0.759 at 387 |
| 13 | `13_verify_confusion_matrices.py` | printed comparison | 12 of 14 reconstructed counts exact |
| 14 | `14_reproducibility_checks.py --images picai_images` | `selected_t2w_files.csv` | 1,476 vectors for 1,500 studies; PCA basis check; sensitivity results |

Step 6 expected counts (hold-out partition, 85 positive / 215 negative, default threshold): fully automated Stacking 58/27/70/145 (TP/FN/FP/TN); PI-RADS-informed Stacking 78/7/54/161;
biopsy-pathway Stacking 80/5/61/154. The fully automated XGBoost of thesis Table 4.3 (ROC-AUC 0.774, recall 0.506) was tuned in the project notebook (`notebooks/02_full_experiments.ipynb`, cell 91) with a separate grid scored on accuracy. `06_train_algorithms.py --config A --xgb-tuning accuracy_grid` repeats that search; the default Table A.1 grid, scored on ROC-AUC, gives about 0.746 (Appendix H).

## Known deviations (details in thesis Appendix I)
1. Imaging features exist for 1,476 patients, not 1,500 studies: the 24 extra studies of 23 repeat patients share one T2-weighted file's features. Which file was used came from directory order; `selected_t2w_files.csv` records it (14 of the 23 were not the first in sorted order).
2. The PCA behind `fused_final_dataset.csv` was fitted on the 1,500 merged rows, before the train/test partition (label-free). The external projection used for the reported results was fitted on the 1,476 distinct vectors; `--basis merged` in step 7 gives the consistent basis (ROC-AUC 0.742, recall 0.651, specificity 0.705).
3. Missing clinical values were filled with whole-cohort medians before partitioning.
4. PI-RADS and biopsy-pathway values for repeat patients come from the patient's first marksheet record.
5. Training label: case-level ISUP grade group >= 2 (`case_csPCa`). External label: maximum Gleason sum >= 7 over biopsy cores.
6. Patient age is not recorded in the external cohort and was imputed with the PI-CAI median.

## Licence and citation
Released under the MIT licence (see `LICENSE` in the repository root). Cite the thesis (see the repository README) when you use the code.
