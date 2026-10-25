"""Step 14. Checks specific to this study's data handling (Appendix I).
(1) unit of the imaging features; (2) which data the PCA was fitted on; (3) external results on the as-run and the consistent PCA basis;
(4) sensitivity to removing all studies of repeat patients; (5) manifest of the T2W file chosen per patient (selected_t2w_files.csv).
Usage: python 14_reproducibility_checks.py [--images picai_images]
Expected (real data): 1476 feature vectors for 1500 studies; the PCA behind fused_final_dataset.csv matches a PCA fitted on the merged rows (max difference about 1e-3);
external as-run ROC-AUC 0.745, consistent basis 0.742 (recall 0.651, specificity 0.705), no imaging features 0.749; removing repeat patients: test ROC-AUC 0.881."""
import argparse, os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from common import SEED, path, load_internal, load_external, split_indices, make_locked, metrics_row
ap = argparse.ArgumentParser(); ap.add_argument("--images", default="picai_images"); a = ap.parse_args()
fused, raw, fp, X, y = load_internal(); t2w = pd.read_csv(path("cnn_features_t2w.csv")); t2w["patient_id"] = t2w["patient_id"].astype(int)
fc = [c for c in t2w.columns if c.startswith("cnn_feat_")]; pcs = [c for c in fused.columns if c.startswith("cnn_pca_")]
print("== 1. UNIT OF THE IMAGING FEATURES"); print("feature vectors:", len(t2w), "| unique patients:", t2w["patient_id"].nunique(), "| studies:", len(fused))
rep = fused[fused.duplicated("patient_id", keep=False)]; same = sum(1 for _, g in rep.groupby("patient_id") if np.allclose(g[pcs].to_numpy(), g[pcs].iloc[0].to_numpy())); print(f"repeat-patient groups: {rep['patient_id'].nunique()} ({len(rep)} studies); groups sharing identical imaging features: {same}")
print("== 2. WHICH DATA WAS THE PCA FITTED ON?"); F_m = t2w.set_index("patient_id").loc[fused["patient_id"], fc].to_numpy()
for name, fit_on in (("distinct patient vectors", t2w[fc].to_numpy()), ("merged study rows", F_m)):
    sc = StandardScaler().fit(fit_on); pc = PCA(n_components=len(pcs), random_state=SEED).fit(sc.transform(fit_on)); Z = pc.transform(sc.transform(F_m)); print(f"PCA fitted on {name}: max |difference| to the fused file = {np.abs(Z - fused[pcs].to_numpy()).max():.2e}")
print("== 3. LOCKED MODEL ON THE EXTERNAL COHORT"); itr, ite = split_indices(y); cols = X.columns.tolist(); sc = StandardScaler().fit(X.iloc[itr]); locked = make_locked().fit(sc.transform(X.iloc[itr]), y.iloc[itr])
for label, basis in (("as run (PCA on distinct vectors)", "unique"), ("consistent basis (PCA on merged rows)", "merged")):
    ext = load_external(basis); r = metrics_row(ext["csPCa_label"], locked.predict_proba(sc.transform(ext[cols]))[:, 1]); print(f"{label:42s}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("roc_auc", "recall", "precision", "specificity", "TP", "FN", "FP", "TN")})
nonimg = [c for c in cols if c not in pcs]; s2 = StandardScaler().fit(X.iloc[itr][nonimg]); m2 = make_locked().fit(s2.transform(X.iloc[itr][nonimg]), y.iloc[itr]); ext = load_external("merged")
print("no imaging features at all:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics_row(ext["csPCa_label"], m2.predict_proba(s2.transform(ext[nonimg]))[:, 1]).items() if k in ("roc_auc", "recall", "precision", "specificity")})
print("== 4. REMOVE ALL STUDIES OF REPEAT PATIENTS"); isrep = fused.duplicated("patient_id", keep=False).to_numpy()
for label, tr, te in (("all studies", itr, ite), ("repeat-patient studies removed", itr[~isrep[itr]], ite[~isrep[ite]])):
    s_ = StandardScaler().fit(X.iloc[tr]); m = make_locked().fit(s_.transform(X.iloc[tr]), y.iloc[tr]); r = metrics_row(y.iloc[te], m.predict_proba(s_.transform(X.iloc[te]))[:, 1]); print(f"{label:34s} train {len(tr)} test {len(te)}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("roc_auc", "recall", "TP", "FN", "FP", "TN")})
print("== 5. T2W FILE SELECTION MANIFEST")
if os.path.isdir(a.images):
    rows = []
    for pid in sorted(os.listdir(a.images)):
        fol = os.path.join(a.images, pid)
        if os.path.isdir(fol): t = [f for f in os.listdir(fol) if "t2w" in f.lower()]; rows.append(dict(patient_id=int(pid), n_t2w_files=len(t), selected_file=t[0] if t else "", first_in_sorted_order=sorted(t)[0] if t else "", selected_is_sorted_first=bool(t) and t[0] == sorted(t)[0]))
    man = pd.DataFrame(rows); man.to_csv(path("selected_t2w_files.csv"), index=False); print("patients:", len(man), "| with more than one T2W file:", int((man.n_t2w_files > 1).sum()), "| selected file not first in sorted order:", int(((man.n_t2w_files > 1) & (~man.selected_is_sorted_first)).sum()), "(expected 23 and 14)")
    print("NOTE: this lists the first file in the CURRENT directory order; the selection used for the thesis is the supplied selected_t2w_files.csv.")
else: print("image folder not found")
