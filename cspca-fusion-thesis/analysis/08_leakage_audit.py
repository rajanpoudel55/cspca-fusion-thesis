"""Step 8. Patient-level separation and preprocessing audit (Section 3.12, Appendix E).
Checks: study/patient identifiers; how missing values were filled; patients present in both partitions; effect on the locked model of
(a) removing overlapping patients, (b) refitting the imaging scaler and PCA on training studies only, (c) a patient-grouped partition.
Expected (real data): 8 overlapping patients; test ROC-AUC 0.884 (0.885 without overlaps); 0.891 with PCA refitted; grouped 0.874 / 0.883."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from common import path, MARKSHEET, SEED, load_internal, split_indices, make_locked, metrics_row
fused, raw, fp, X, y = load_internal(); cc = pd.read_csv(path("picai_clinical_clean.csv")); ids = fused["patient_id"].reset_index(drop=True)
print("== 1. IDENTIFIERS"); print("studies:", len(fused), "| patients:", ids.nunique(), "| studies of repeat patients:", int(fused.duplicated("patient_id", keep=False).sum()))
print("patients in more than one centre:", int((cc[["patient_id", "center"]].drop_duplicates().groupby("patient_id")["center"].nunique() > 1).sum()))
print("== 2. IMPUTATION (full-cohort median versus the value found in the cleaned table)")
for col in ["patient_age", "psa", "psad", "prostate_volume"]:
    miss = raw.loc[raw[col].isna(), "patient_id"]; vals = cc[cc["patient_id"].isin(miss)][col].round(3).unique()[:4]; print(f"{col}: raw missing {int(raw[col].isna().sum())} | cleaned values at those rows {vals} | median {raw[col].median():.3f}")
itr, ite = split_indices(y); overlap = set(ids.iloc[itr]) & set(ids.iloc[ite]); print("== 3. PATIENT OVERLAP:", len(overlap), "patients in both partitions;", int(ids.iloc[ite].isin(overlap).sum()), "test studies affected")
sc = StandardScaler().fit(X.iloc[itr]); m0 = make_locked().fit(sc.transform(X.iloc[itr]), y.iloc[itr]); p = m0.predict_proba(sc.transform(X.iloc[ite]))[:, 1]
print("original test set          ", {k: round(v, 3) for k, v in metrics_row(y.iloc[ite], p).items() if k in ("roc_auc", "recall", "precision", "accuracy")})
keep = ~ids.iloc[ite].isin(set(ids.iloc[itr])).to_numpy(); print("without overlapping patients", {k: round(v, 3) for k, v in metrics_row(y.iloc[ite][keep], p[keep]).items() if k in ("roc_auc", "recall", "precision", "accuracy")})
print("== 4. PREPROCESSING AND PATIENT-LEVEL SPLIT SENSITIVITY")
t2w = pd.read_csv(path("cnn_features_t2w.csv")); t2w["patient_id"] = t2w["patient_id"].astype(int); fc = [c for c in t2w.columns if c.startswith("cnn_feat_")]
F = t2w.set_index("patient_id").loc[ids, fc].to_numpy(); clin = [c for c in X.columns if not c.startswith("cnn_pca_")]; k = X.shape[1] - len(clin)
def fit_eval(tr, te, refit, label):
    if refit:
        s = StandardScaler().fit(F[tr]); pc = PCA(n_components=k, random_state=SEED).fit(s.transform(F[tr]))
        Atr = np.hstack([X.iloc[tr][clin].to_numpy(), pc.transform(s.transform(F[tr]))]); Ate = np.hstack([X.iloc[te][clin].to_numpy(), pc.transform(s.transform(F[te]))])
    else: Atr, Ate = X.iloc[tr].to_numpy(), X.iloc[te].to_numpy()
    s2 = StandardScaler().fit(Atr); m = make_locked().fit(s2.transform(Atr), y.iloc[tr]); print(label, {a: round(b, 3) for a, b in metrics_row(y.iloc[te], m.predict_proba(s2.transform(Ate))[:, 1]).items() if a in ("roc_auc", "recall", "precision", "accuracy")})
fit_eval(itr, ite, True, "same split, PCA/scaler refitted on training studies only:")
g_tr, g_te = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, groups=ids)); print("grouped split: patients in both sets =", len(set(ids.iloc[g_tr]) & set(ids.iloc[g_te])))
fit_eval(g_tr, g_te, False, "patient-grouped split, original PCA features:"); fit_eval(g_tr, g_te, True, "patient-grouped split, PCA refitted on training only:")
