# Run guide: how to run the code and what to show

There are two levels. Show the demo first, because it works in minutes and proves the code runs. Then show the full reproduction with the real public datasets, which reproduces the thesis numbers.

| Level | Data | Time | What it proves |
|---|---|---|---|
| Demo (`run_demo.py`) | Synthetic, generated for you | about 7 minutes | Every analysis step runs end to end. The numbers are meaningless clinically and will not match the thesis. |
| Full reproduction | PI-CAI and Prostate-MRI-US-Biopsy (public downloads) | Feature extraction (steps 2 and 5) is the slowest part. I have not timed it on the real data. | The numbers in the thesis, compared with the expected values below. |

## 1. Set up (once)
Use Python 3.11. The thesis results came from macOS on Apple silicon.
```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python analysis/00_environment_report.py
```
If a pinned version does not install on your machine, install the same packages without the version numbers and note the versions that `00_environment_report.py` prints. Small version differences can change the last decimals.

## 2. Demo (synthetic data)
```bash
python run_demo.py            # all steps
python run_demo.py --quick    # skips two of the three model configurations
```
It prints the end of each step's output and writes everything to `demo_output/demo_log.txt`, with one file per step in `demo_output/logs/`. Show your supervisor that file, and say clearly that the data are synthetic.

## 3. Full reproduction (real data)
### 3.1 Download the public data (not included here)
- PI-CAI public training and development dataset: Zenodo, doi 10.5281/zenodo.6624726 (images). Clinical information and labels: github.com/DIAGNijmegen/picai_labels.
- Prostate-MRI-US-Biopsy: The Cancer Imaging Archive, doi 10.7937/TCIA.2020.A61IOC1A (T2-weighted MRI, plus the Biopsy Data and Target Data spreadsheets).

Put everything under one folder and point `CSPCA_ROOT` at it. Expected layout:
```
<CSPCA_ROOT>/picai_labels/clinical_information/marksheet.csv
<CSPCA_ROOT>/picai_images/<patient_id>/<patient_id>_<study_id>_t2w.mha
<CSPCA_ROOT>/<TCIA DICOM folder for the external cohort>
```
Also copy `selected_t2w_files.csv` (the record of which T2-weighted file was used for each repeat patient) into the same folder and pass it to step 2 and step 14.

### 3.2 Run order
```bash
export CSPCA_ROOT=/path/to/your/data      # Windows PowerShell: $env:CSPCA_ROOT="C:\path\to\data"
cd analysis
python 01_build_picai_clinical_table.py
python 02_extract_picai_features.py --images "$CSPCA_ROOT/picai_images" --manifest "$CSPCA_ROOT/selected_t2w_files.csv"
python 03_build_fused_dataset.py
python 04_build_external_cohort.py --biopsy-file <Biopsy Data xlsx> --target-file <Target Data xlsx> --dicom-root <TCIA DICOM folder>
python 05_extract_external_features.py
python 06_train_algorithms.py --config A     # fully automated
python 06_train_algorithms.py --config B     # PI-RADS-informed
python 06_train_algorithms.py --config C     # biopsy-pathway
python 06_train_algorithms.py --config A --xgb-tuning accuracy_grid   # repeats notebook cell 91: the tuned XGBoost of Table 4.3
python 07_locked_model_external.py           # add --basis merged for the consistent-basis result
python 08_leakage_audit.py
python 09_leave_one_centre_out.py
python 10_baselines_statistics.py
python 11_external_investigation.py
python 12_pca_sensitivity.py
python 13_verify_confusion_matrices.py
python 14_reproducibility_checks.py --images "$CSPCA_ROOT/picai_images"
```
Do not use `--n-components 20` or `CSPCA_FAST` for the real run. They exist only for the synthetic demo.

### 3.3 Expected values (what you should see)
| Step | Expected |
|---|---|
| 1 | 1,500 rows, 425 csPCa, no missing values |
| 2 | 1,476 rows x 2,049 columns |
| 3 | 1,500 x 106 and 1,500 x 107; PCA keeps about 0.71 of the variance |
| 4 | 1,150 x 7 and 782 x 9; prevalence 0.575 |
| 5 | 782 x 2,049 |
| 6 | With `--xgb-tuning accuracy_grid`: XGBoost ROC-AUC about 0.774, recall about 0.506 (TP 43, FN 42), as recorded in notebook cell 91 (I have not re-run it on the real data). Hold-out counts (TP/FN/FP/TN), Stacking: fully automated 58/27/70/145; PI-RADS-informed 78/7/54/161; biopsy-pathway 80/5/61/154 |
| 7 | Internal 79/6/52/163; external 313/137/122/210 (unique basis) |
| 8 | 8 overlapping patients; ROC-AUC 0.884, 0.885, 0.891, 0.874, 0.883 |
| 9 | Fused ROC-AUC 0.732 / 0.585 / 0.779 (RUMC / PCNN / ZGT) |
| 10 | Logistic regression 0.757; with PI-RADS and PSA density 0.878; locked model 0.884 |
| 11 | Drop 0.138 (0.090 to 0.187); score alone 0.857 against 0.703 |
| 12 | Fused ROC-AUC 0.790 at 100 components, 0.759 at 387 |
| 13 | 12 of 14 reconstructed counts exact |
| 14 | 1,476 vectors for 1,500 studies |

### 3.4 Which step backs which part of the thesis
| Step | Thesis |
|---|---|
| 06 (A, B, C) and `notebooks/` | Algorithm comparison tables (Table 4.3 for the fully automated configuration; Table 4.8 for the PI-RADS-informed configuration) |
| 07 | Locked-model internal and external evaluation (Section 4.10, Table 4.20) |
| 08 | Leakage audit (Appendix E) |
| 09 | Leave-one-centre-out validation (Section 4.8) |
| 10 | Baselines, bootstrap intervals and paired tests (Section 4.12) |
| 11 | Investigation of the external fall (Appendix G) |
| 13 | Verification of reconstructed confusion matrices (Appendix H) |
| 14 | Feature-extraction and PCA-basis checks (Appendix I) |

## 3.5 The notebooks
`notebooks/README.md` maps all 195 cells of the main notebook to the thesis sections. It holds the code for the experiments that have no script: SMOTE, late and intermediate fusion, fine-tuning, augmentation, five-slice and ADC/HBV variants, the threshold experiment, decision curve and calibration analysis, and the data examined but not used (TCGA-PRAD and two further clinical files).

## 4. Say this to your supervisor before they find it
These points are in the thesis (Appendix I), and the code README repeats them. Raising them yourself is better than having them discovered.
- Steps 1 to 6 were rewritten as scripts from the project notebooks, which are in `notebooks/` with their outputs. They reproduce the derived-file shapes above, but they were not re-run end to end on the original images when the package was assembled. Compare your outputs with the expected values.
- The tuned XGBoost of Table 4.3 (ROC-AUC 0.774) was tuned on accuracy in notebook cell 91, whereas every other model was tuned on ROC-AUC. Run step 6 with `--xgb-tuning accuracy_grid` to repeat it. The ROC-AUC-scored grid of Table A.1 gives about 0.746. The thesis states this in Appendices A and H.
- The imaging features exist for 1,476 patients, not 1,500 studies, and the choice of T2-weighted file for 23 repeat patients depended on directory order. `selected_t2w_files.csv` records it.
- The external ROC-AUC is 0.745 on the reported basis and 0.742 on the consistent PCA basis (step 7 with `--basis merged`).
- GPU inference can change the imaging features in the last decimals.

## 5. Troubleshooting
- `FileNotFoundError` for a CSV: check `CSPCA_ROOT`, and run the earlier steps first (steps 1 and 3 before steps 6 to 12).
- TensorFlow does not install on Apple silicon: install `tensorflow` (not the `tensorflow-metal` plug-in, which the analysis does not need).
- Step 3 fails with `n_components=100 must be between 0 and min(n_samples, n_features)`: you are running on the synthetic data. Use `--n-components 20`, or let `run_demo.py` do it.
