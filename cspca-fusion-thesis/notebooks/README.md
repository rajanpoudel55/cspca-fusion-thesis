# notebooks/

These are the Jupyter notebooks in which the experiments of the thesis were run, with their outputs.
They are the record of what was run. The scripts in `../analysis/` are a cleaner restatement of the central steps.

| File | Content |
|---|---|
| `01_clinical_table.ipynb` | Reads the PI-CAI marksheet, recomputes missing PSA density, imputes medians, writes `picai_clinical_clean.csv` (analysis step 01) |
| `02_full_experiments.ipynb` | All other experiments (195 code cells, map below) |
| `02_full_experiments_code_only.py` | The code cells of the second notebook without outputs, for reading. `# %% [cell N]` marks each cell (0-based) |

## Cell map of `02_full_experiments.ipynb`
| Cells | What they do | Thesis |
|---|---|---|
| 0 to 11 | T2-weighted feature extraction with a frozen ResNet50 (central slice); writes `cnn_features_t2w.csv` | Section 3.3 |
| 12 to 17 | Merge with clinical table, standardise, PCA (capped at 100 components), fused dataset | Sections 3.4, 3.5 |
| 18 to 21 | First algorithm comparison on the fused features | Section 4.3 |
| 22 to 30 | ADC and HBV feature extraction, per-sequence PCA (35 components), multiparametric dataset | Section 4.6 |
| 31 to 37 | Models on the multiparametric dataset | Section 4.6 |
| 38 to 41 | Clinical-only, imaging-only and fused comparison across algorithms; comparison table | Sections 4.2, 4.3 |
| 42 to 47 | Five-slice averaging: features and models | Section 4.6 |
| 48 to 52 | SMOTE oversampling experiment | Section 4.4 |
| 53 to 66 | Comparison tables, ROC figures, literature-comparison figure | Figures |
| 67 to 82 | End-to-end ResNet50 classifiers: frozen backbone, fine-tuning of the last ten layers (cells 77 to 78), data augmentation (79 to 81) | Section 4.6 |
| 83 to 91 | XGBoost: fixed hyperparameters (86 to 87), accuracy-optimal threshold search (90), tuned XGBoost grid search scored on accuracy (91) | Sections 4.3, 4.9 |
| 92 to 96 | Widened randomised XGBoost search scored on ROC-AUC | Section 4.3 |
| 97 to 104 | Leave-one-centre-out validation | Section 4.8 |
| 105 to 119 | PI-RADS-informed configuration; writes `fused_final_dataset_with_pirads.csv` (cell 119) | Section 4.7 |
| 120 to 128 | Biopsy-pathway configuration; writes `fused_final_dataset_with_pirads_histopath.csv` (cell 128) | Section 4.7 |
| 129 to 131 | Rebuild and test of the Stacking model | Section 4.7 |
| 132 to 156 | External cohort: biopsy and target tables, Gleason per patient, DICOM discovery, external features (150 to 151), locked-model evaluation (152 to 156) | Sections 3.8, 4.10 |
| 157 to 160 | Internal test set rebuild; decision curve analysis (158); calibration curves (159 to 160) | Section 4.11 |
| 161 | Other locked models, including an XGBoost with the hyperparameters found in cell 91 | |
| 162 to 170 | Data examined and not used: TCIA PROSTATE-DIAGNOSIS clinical file (162 to 163), a clinical cohort from Turkey (164 to 165), TCGA-PRAD clinical files (166 to 170) | Sections 3.2, 3.14 |
| 171 to 184 | Figures for the thesis (ROC grids, bar charts, threshold plots, fusion plots) and locked-model reports | Figures |
| 185 to 194 | Later consolidated analyses: locked pipeline and Table A.1 refits (185), identifier audit (186), leave-one-centre-out with a refitted PCA (187), merge audit (188), PCA sensitivity (189), uncertainty and external analyses (190), baselines and statistics (191), confusion-matrix verification (192), reproducibility checks (193), external feature-basis comparison (194) | Sections 4.10 to 4.12; Appendices E to I |

## How to read the notebook
- Eight cells were executed out of order. Read the execution counts (`[n]` beside each cell), not the cell positions, when you need the order in which things ran.
- The notebook expects to run in the project folder (`~/prostate-cancer-research`) with the downloaded datasets and its own earlier outputs (`fused_final_dataset.csv`, `picai_clinical_clean.csv`, and so on) present.
- 24 outputs are error tracebacks from exploratory attempts. They are kept because they are part of the record.

## Two things this notebook settles
1. **Tuned XGBoost of Table 4.3 (ROC-AUC 0.774).** Cell 91 produced it: `GridSearchCV` over n_estimators 200, 300, 500; max_depth 3, 5, 6, 8; learning_rate 0.01, 0.05, 0.1; subsample 0.8, 1.0; colsample_bytree 0.8, 1.0; `scale_pos_weight` 2.5; five-fold cross-validation **scored on accuracy**. The best parameters were colsample_bytree 1.0, learning_rate 0.01, max_depth 8, n_estimators 300, subsample 0.8, giving test ROC-AUC 0.774, recall 0.506 and accuracy 0.737. Every other tuning search in the notebook is scored on ROC-AUC. The refit in cell 192 uses the ROC-AUC-scored grid of Table A.1 and gives 0.746 (recall 0.694). `analysis/06_train_algorithms.py --config A --xgb-tuning accuracy_grid` repeats the cell 91 search.
2. **Threshold experiment (Section 4.9).** Cell 90 searches the accuracy-optimal threshold on `xgb_probs` from cells 86 to 87, an XGBoost with fixed hyperparameters (300 trees, depth 6, learning rate 0.1), not the tuned model of cell 91. At 0.5 that model has accuracy 0.747, precision 0.567 and recall 0.447; at 0.74 it has accuracy 0.760, precision 0.659 and recall 0.318.

## Changes made for the public deposit
The notebook is otherwise unchanged. In `02_full_experiments.ipynb` the author's home-directory path in warnings and tracebacks was replaced by `~`, and the five-row patient-level preview printed in cell 164 (clinical file from Turkey) was removed, keeping its shape and column names.
