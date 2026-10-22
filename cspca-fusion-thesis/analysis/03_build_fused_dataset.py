"""Step 3. Fused dataset: clinical table + imaging features, standardised and reduced by PCA (100 components).
Merge is on patient_id, so the 24 additional studies of repeat patients receive their patient's feature vector (1,500 rows from 1,476 vectors).
The scaler and PCA are fitted on the merged 1,500 rows, before the train/test partition (label-free; see Section 3.12 and Appendix E).
Outputs: fused_final_dataset.csv (expected 1500 x 106) and fused_final_dataset_with_pirads.csv (expected 1500 x 107)."""
import argparse
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from common import path, MARKSHEET, SEED, get_max_pirads
ap = argparse.ArgumentParser(); ap.add_argument("--n-components", type=int, default=100); a = ap.parse_args()
clin = pd.read_csv(path("picai_clinical_clean.csv")); feats = pd.read_csv(path("cnn_features_t2w.csv")); feats["patient_id"] = feats["patient_id"].astype(int)
m = clin.merge(feats, on="patient_id", how="inner").reset_index(drop=True); fc = [c for c in m.columns if c.startswith("cnn_feat_")]
sc = StandardScaler().fit(m[fc]); pca = PCA(n_components=a.n_components, random_state=SEED).fit(sc.transform(m[fc]))
Z = pd.DataFrame(pca.transform(sc.transform(m[fc])), columns=[f"cnn_pca_{i}" for i in range(a.n_components)])
fused = pd.concat([m[["patient_id", "patient_age", "psa", "psad", "prostate_volume", "csPCa_label"]], Z], axis=1)
print("fused:", fused.shape, "| variance retained:", round(pca.explained_variance_ratio_.sum(), 3), "(expected about 0.71)"); fused.to_csv(path("fused_final_dataset.csv"), index=False)
raw = pd.read_csv(MARKSHEET); raw["max_pirads"] = raw["lesion_PIRADS"].apply(get_max_pirads)
fp = fused.merge(raw[["patient_id", "max_pirads"]].drop_duplicates("patient_id"), on="patient_id", how="left"); fp["max_pirads"] = fp["max_pirads"].fillna(fp["max_pirads"].median())
print("with PI-RADS:", fp.shape); fp.to_csv(path("fused_final_dataset_with_pirads.csv"), index=False)
