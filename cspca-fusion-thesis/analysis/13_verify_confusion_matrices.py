"""Step 13. Exact confusion matrices of refitted models and profile of the locked model's errors (Section 4.10.6, Appendix H).
Compares counts reconstructed from reported recall and precision with the actual predictions (six algorithms x three configurations).
Expected (real data): all rows match except the fully automated XGBoost (refit ROC-AUC 0.746) and Random Forest (35 instead of 34 true positives)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from common import SEED, load_config, load_internal, load_external, split_indices, make_locked, metrics_row
GR = {"RandomForest": (RandomForestClassifier(random_state=SEED, class_weight="balanced"), {"n_estimators": [100, 300], "max_depth": [10, 20]}), "AdaBoost": (AdaBoostClassifier(random_state=SEED), {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]}),
      "GradientBoosting": (GradientBoostingClassifier(random_state=SEED), {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1]}), "XGBoost": (XGBClassifier(scale_pos_weight=2.5, eval_metric="logloss", random_state=SEED), {"n_estimators": [200, 300], "max_depth": [3, 5, 6], "learning_rate": [0.01, 0.1]})}
RECON = {("A", "Stacking"): (58, 27, 70, 145), ("A", "XGBoost"): (43, 42, None, None), ("A", "RandomForest"): (34, 51, None, None), ("B", "Stacking"): (78, 7, 54, 161), ("B", "RandomForest"): (59, 26, 32, 183), ("B", "XGBoost"): (78, 7, 59, 156), ("B", "NaiveBayes"): (38, 47, 20, 195),
         ("C", "Stacking"): (80, 5, 61, 154), ("C", "RandomForest"): (75, 10, 50, 165), ("C", "XGBoost"): (81, 4, 67, 148), ("C", "NaiveBayes"): (84, 1, 99, 116), ("C", "GradientBoosting"): (56, 29, None, None)}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED); rows = []
for cfg in "ABC":
    X, y = load_config(cfg); itr, ite = split_indices(y); sc = StandardScaler().fit(X.iloc[itr]); Xtr, Xte = sc.transform(X.iloc[itr]), sc.transform(X.iloc[ite]); ytr, yte = y.iloc[itr].to_numpy(), y.iloc[ite].to_numpy(); best = {}
    for nm, (m, g) in GR.items(): best[nm] = GridSearchCV(m, g, cv=cv, scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr).best_estimator_
    best["NaiveBayes"] = GaussianNB().fit(Xtr, ytr)
    best["Stacking"] = StackingClassifier(estimators=[("rf", best["RandomForest"]), ("ada", best["AdaBoost"]), ("gb", best["GradientBoosting"])], final_estimator=LogisticRegression(class_weight="balanced"), cv=5).fit(Xtr, ytr)
    for nm, m in best.items(): r = metrics_row(yte, m.predict_proba(Xte)[:, 1]); rows.append(dict(cfg=cfg, model=nm, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("TP", "FN", "FP", "TN", "recall", "precision", "roc_auc")}))
R = pd.DataFrame(rows); print("== 1. EXACT COUNTS"); print(R.to_string(index=False)); print("\n== 2. RECONSTRUCTED VERSUS EXACT")
for (cfg, nm), rec in RECON.items():
    r = R[(R.cfg == cfg) & (R.model == nm)].iloc[0]; print(cfg, f"{nm:17s}", " | ".join(f"{k} {int(r[k])} vs {e}" + ("" if int(r[k]) == e else " <-- differs") for k, e in zip(("TP", "FN", "FP", "TN"), rec) if e is not None))
fused, raw, fp, X, y = load_internal(); itr, ite = split_indices(y); cols = X.columns.tolist(); scL = StandardScaler().fit(X.iloc[itr]); locked = make_locked().fit(scL.transform(X.iloc[itr]), y.iloc[itr])
pin = locked.predict_proba(scL.transform(X.iloc[ite]))[:, 1]; ext = load_external("unique"); pex = locked.predict_proba(scL.transform(ext[cols]))[:, 1]
for nm, yy, pp in (("internal", y.iloc[ite], pin), ("external", ext["csPCa_label"], pex)): tn, fp_, fn, tp = confusion_matrix(yy, (pp >= .5).astype(int)).ravel(); print(f"\n== 3. LOCKED MODEL {nm}: TP {tp} FN {fn} FP {fp_} TN {tn} (expected internal 79/6/52/163, external 313/137/122/210)")
te = X.iloc[ite].reset_index(drop=True).copy(); te["y"] = y.iloc[ite].to_numpy(); te["pred"] = (pin >= .5).astype(int); te["patient_id"] = fused["patient_id"].iloc[ite].to_numpy()
iso = raw[["patient_id"] + [c for c in ("case_ISUP", "histopath_type") if c in raw.columns]].drop_duplicates("patient_id"); te = te.merge(iso, on="patient_id", how="left")
te["group"] = np.select([(te.y == 1) & (te.pred == 1), (te.y == 1) & (te.pred == 0), (te.y == 0) & (te.pred == 1)], ["TP", "FN", "FP"], "TN")
print("\n== 4. ERROR PROFILE (internal test partition)"); print(te.groupby("group").agg(n=("y", "size"), median_psa=("psa", "median"), median_psad=("psad", "median"), median_volume=("prostate_volume", "median"), median_pirads=("max_pirads", "median")).round(2))
print(pd.crosstab(te.group, te.max_pirads.round()))
if "case_ISUP" in te.columns: print(pd.crosstab(te.group, te.case_ISUP)); print(te[te.group == "FN"][["max_pirads", "psad", "prostate_volume", "case_ISUP"]].round(2).to_string(index=False))
rule = (te.max_pirads >= 4).astype(int); tn, fp_, fn, tp = confusion_matrix(te.y, rule).ravel(); print("PI-RADS >= 4 rule:", f"TP {tp} FN {fn} FP {fp_} TN {tn}")
ex = ext[cols].copy(); ex["y"] = ext["csPCa_label"].to_numpy(); ex["pred"] = (pex >= .5).astype(int); ex["group"] = np.select([(ex.y == 1) & (ex.pred == 1), (ex.y == 1) & (ex.pred == 0), (ex.y == 0) & (ex.pred == 1)], ["TP", "FN", "FP"], "TN")
print("\n== 5. ERROR PROFILE (external cohort)"); print(ex.groupby("group").agg(n=("y", "size"), median_psa=("psa", "median"), median_psad=("psad", "median"), median_volume=("prostate_volume", "median"), median_score=("max_pirads", "median")).round(2)); print(pd.crosstab(ex.group, ex.max_pirads.round()))
