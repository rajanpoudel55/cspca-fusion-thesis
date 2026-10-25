"""Step 12. Sensitivity of the Random Forest to the number of principal components (Section 3.4.3, Appendix F).
PCA and scaler are fitted on the training studies only. Expected fused ROC-AUC (real data): 0.789 (10), 0.784 (25), 0.790 (50), 0.790 (100), 0.775 (200), 0.759 (387)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score
from common import path, SEED, load_internal, split_indices
fused, raw, fp, X, y = load_internal(); t2w = pd.read_csv(path("cnn_features_t2w.csv")); t2w["patient_id"] = t2w["patient_id"].astype(int); fc = [c for c in t2w.columns if c.startswith("cnn_feat_")]
F = t2w.set_index("patient_id").loc[fused["patient_id"], fc].to_numpy(); clin = fused[["patient_age", "psa", "psad", "prostate_volume"]].to_numpy(); yy = y.to_numpy(); tr, te = split_indices(y)
s = StandardScaler().fit(F[tr]); Ztr, Zte = s.transform(F[tr]), s.transform(F[te]); kmax = min(Ztr.shape[0], Ztr.shape[1])
for k in (10, 25, 50, 100, 200, 387):
    if k > kmax: continue
    p = PCA(n_components=k, random_state=SEED).fit(Ztr)
    for name, a_tr, a_te in (("imaging only", p.transform(Ztr), p.transform(Zte)), ("fused", np.hstack([clin[tr], p.transform(Ztr)]), np.hstack([clin[te], p.transform(Zte)]))):
        sc = StandardScaler().fit(a_tr); m = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=SEED, class_weight="balanced").fit(sc.transform(a_tr), yy[tr]); pr = m.predict_proba(sc.transform(a_te))[:, 1]
        print(k, name, "AUC", round(roc_auc_score(yy[te], pr), 3), "recall", round(recall_score(yy[te], pr >= 0.5), 3), "| variance kept", round(p.explained_variance_ratio_.sum(), 2))
