"""Step 9. Leave-one-centre-out validation (Section 4.8): Random Forest (300 trees, depth 10, balanced) on clinical, imaging and fused features.
Centre labels are merged on patient_id AFTER removing duplicate patient records (a many-to-many merge in the first run gave 1,550 rows).
Expected fused ROC-AUC (real data): RUMC 0.732, PCNN 0.585, ZGT 0.779; training sizes 700 / 1150 / 1150."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from common import path, SEED, metrics_row
fused = pd.read_csv(path("fused_final_dataset.csv")); fused["patient_id"] = fused["patient_id"].astype(int); cc = pd.read_csv(path("picai_clinical_clean.csv"))
df = fused.merge(cc[["patient_id", "center"]].drop_duplicates("patient_id"), on="patient_id", how="left"); assert len(df) == len(fused)
clin = ["patient_age", "psa", "psad", "prostate_volume"]; pcs = [c for c in df.columns if c.startswith("cnn_pca_")]; sets = {"Clinical only": clin, "Imaging only": pcs, "Fused": clin + pcs}; rows = []
rf = lambda: RandomForestClassifier(n_estimators=300, max_depth=10, random_state=SEED, class_weight="balanced")
for held in sorted(df["center"].unique()):
    tr = (df["center"] != held).to_numpy(); te = ~tr; y = df["csPCa_label"].to_numpy()
    for name, cols in sets.items():
        A = df[cols].to_numpy(); sc = StandardScaler().fit(A[tr]); m = rf().fit(sc.transform(A[tr]), y[tr]); r = metrics_row(y[te], m.predict_proba(sc.transform(A[te]))[:, 1])
        rows.append(dict(held_out=held, features=name, train=int(tr.sum()), test=int(te.sum()), test_pos=int(y[te].sum()), **{k: round(v, 3) for k, v in r.items() if k in ("accuracy", "precision", "recall", "f1", "roc_auc")}))
R = pd.DataFrame(rows); print(R.to_string(index=False)); R.to_csv(path("results_loco.csv"), index=False)
t2w = pd.read_csv(path("cnn_features_t2w.csv")); t2w["patient_id"] = t2w["patient_id"].astype(int); fc = [c for c in t2w.columns if c.startswith("cnn_feat_")]; F = t2w.set_index("patient_id").loc[df["patient_id"], fc].to_numpy()
print("\nSensitivity: PCA refitted on the training centres only (fused features)")
for held in sorted(df["center"].unique()):
    tr = (df["center"] != held).to_numpy(); te = ~tr; y = df["csPCa_label"].to_numpy(); s = StandardScaler().fit(F[tr]); p = PCA(n_components=len(pcs), random_state=SEED).fit(s.transform(F[tr]))
    A = np.hstack([df[clin].to_numpy(), p.transform(s.transform(F))]); sc = StandardScaler().fit(A[tr]); m = rf().fit(sc.transform(A[tr]), y[tr])
    print(held, "ROC-AUC", round(metrics_row(y[te], m.predict_proba(sc.transform(A[te]))[:, 1])["roc_auc"], 3), "(expected 0.727 / 0.592 / 0.778 for RUMC / PCNN / ZGT)")
