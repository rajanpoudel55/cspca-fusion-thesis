"""Step 7. Locked PI-RADS-informed Stacking model: fitted on the training partition, then applied unchanged to the external cohort.
Usage: python 07_locked_model_external.py [--basis unique|merged]
Expected (real data): internal TP 79 FN 6 FP 52 TN 163 (ROC-AUC 0.884); external with --basis unique TP 313 FN 137 FP 122 TN 210 (ROC-AUC 0.745);
with --basis merged (the basis of the training features) ROC-AUC 0.742, TP 293 FN 157 FP 98 TN 234 (Appendix I)."""
import argparse, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from common import path, load_internal, load_external, split_indices, make_locked, metrics_row
ap = argparse.ArgumentParser(); ap.add_argument("--basis", choices=["unique", "merged"], default="unique"); basis = ap.parse_args().basis
fused, raw, fp, X, y = load_internal(); itr, ite = split_indices(y); cols = X.columns.tolist()
sc = StandardScaler().fit(X.iloc[itr]); model = make_locked().fit(sc.transform(X.iloc[itr]), y.iloc[itr])
pin = model.predict_proba(sc.transform(X.iloc[ite]))[:, 1]; r = metrics_row(y.iloc[ite], pin); print("INTERNAL ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
ext = load_external(basis); pex = model.predict_proba(sc.transform(ext[cols]))[:, 1]; r = metrics_row(ext["csPCa_label"], pex); print(f"EXTERNAL ({basis} basis)", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
pd.DataFrame({"y": y.iloc[ite].to_numpy(), "p": pin}).to_csv(path("locked_internal_predictions.csv"), index=False)
pd.DataFrame({"Patient Number": ext["Patient Number"], "y": ext["csPCa_label"], "p": pex}).to_csv(path(f"locked_external_predictions_{basis}.csv"), index=False)
