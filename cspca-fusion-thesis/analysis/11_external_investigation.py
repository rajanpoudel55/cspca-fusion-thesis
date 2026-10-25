"""Step 11. Why did performance fall on the external cohort? (Section 4.10.5, Appendix G)
A uncertainty and calibration; B input ablations (Random Forest); C cohort shift and a cohort classifier; D optimism check.
Bootstrap seed 0. Expected (real data): ROC-AUC 0.884 internal vs 0.745 external, drop 0.138 (95% CI 0.090-0.187); score alone 0.857 vs 0.703;
clinical-only drop 0.041; removing age changes the external ROC-AUC by about -0.009; the imaging features separate the cohorts with ROC-AUC 1.000."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from common import SEED, FAST, load_internal, load_external, split_indices, make_locked
fused, raw, fp, X, y = load_internal(); cols = X.columns.tolist(); pcs = [c for c in cols if c.startswith("cnn_pca_")]; clin3 = ["psa", "psad", "prostate_volume"]
itr, ite = split_indices(y); Xtr, Xte, ytr, yte = X.iloc[itr], X.iloc[ite], y.iloc[itr], y.iloc[ite]; ext = load_external("unique"); Xex = ext[cols].reset_index(drop=True); yex = ext["csPCa_label"].reset_index(drop=True)
rf = lambda: RandomForestClassifier(n_estimators=300, max_depth=10, random_state=SEED, class_weight="balanced")
sc = StandardScaler().fit(Xtr); locked = make_locked().fit(sc.transform(Xtr), ytr); pin = locked.predict_proba(sc.transform(Xte))[:, 1]; pex = locked.predict_proba(sc.transform(Xex))[:, 1]
print("== A. UNCERTAINTY"); r = np.random.default_rng(0); yi, yx = np.asarray(yte), np.asarray(yex); out, ai, ae = [], [], []
for _ in range(200 if FAST else 1000):
    i = r.integers(0, len(yi), len(yi)); j = r.integers(0, len(yx), len(yx))
    if yi[i].min() == yi[i].max() or yx[j].min() == yx[j].max(): continue
    a, b = roc_auc_score(yi[i], pin[i]), roc_auc_score(yx[j], pex[j]); ai.append(a); ae.append(b); out.append(a - b)
print("internal AUC %.3f (95%% CI %.3f-%.3f) | external AUC %.3f (%.3f-%.3f) | drop %.3f (95%% CI %.3f-%.3f)" % (roc_auc_score(yte, pin), *np.percentile(ai, [2.5, 97.5]), roc_auc_score(yex, pex), *np.percentile(ae, [2.5, 97.5]), roc_auc_score(yte, pin) - roc_auc_score(yex, pex), *np.percentile(out, [2.5, 97.5])))
for nm, yy, pp in (("internal", yte, pin), ("external", yex, pex)):
    pc = np.clip(pp, 1e-6, 1 - 1e-6); z = np.log(pc / (1 - pc)); lr = LogisticRegression(C=1e6).fit(z.reshape(-1, 1), yy); print(f"{nm}: mean predicted risk {pp.mean():.3f} vs observed {np.mean(yy):.3f} | calibration slope {lr.coef_[0][0]:.2f}, intercept {lr.intercept_[0]:.2f}")
sens_i = recall_score(yte, pin >= .5); spec_i = recall_score(1 - np.asarray(yte), pin < .5); prev_e = yex.mean()
print("precision expected at external prevalence: %.3f | observed: %.3f" % (sens_i * prev_e / (sens_i * prev_e + (1 - spec_i) * (1 - prev_e)), precision_score(yex, pex >= .5)))
thr = np.quantile(pex[np.asarray(yex) == 0], spec_i); print("external recall at the internal specificity (%.3f): %.3f" % (spec_i, np.mean(pex[np.asarray(yex) == 1] > thr)))
print("\n== B. INPUT ABLATIONS (Random Forest)")
sets = {"all inputs (as locked)": cols, "without age": [c for c in cols if c != "patient_age"], "without imaging": [c for c in cols if c not in pcs], "without PI-RADS/UCLA score": [c for c in cols if c != "max_pirads"], "clinical only (PSA, PSAD, volume)": clin3, "score only": ["max_pirads"], "imaging only": pcs}
for nm, cl in sets.items():
    s = StandardScaler().fit(Xtr[cl]); m = rf().fit(s.transform(Xtr[cl]), ytr); ai_ = roc_auc_score(yte, m.predict_proba(s.transform(Xte[cl]))[:, 1]); ae_ = roc_auc_score(yex, m.predict_proba(s.transform(Xex[cl]))[:, 1]); print(f"{nm:38s} internal {ai_:.3f} | external {ae_:.3f} | drop {ai_ - ae_:+.3f}")
print("raw score alone, AUC: internal %.3f | external %.3f" % (roc_auc_score(y, X["max_pirads"]), roc_auc_score(yex, Xex["max_pirads"])))
print("\n== C. COHORT SHIFT")
for c in clin3 + ["max_pirads"]:
    a, b = X[c].to_numpy(), Xex[c].to_numpy(); sd = np.sqrt((a.var() + b.var()) / 2); print(f"{c:16s} internal mean {a.mean():7.2f} | external mean {b.mean():7.2f} | standardised difference {(b.mean() - a.mean()) / sd:+.2f} | KS {ks_2samp(a, b).statistic:.2f}")
tab = pd.DataFrame({"internal n": X["max_pirads"].round().value_counts(), "external n": Xex["max_pirads"].round().value_counts()}).fillna(0).astype(int).sort_index()
tab["internal csPCa rate"] = [round(y[X["max_pirads"].round() == k].mean(), 3) if (X["max_pirads"].round() == k).any() else np.nan for k in tab.index]
tab["external csPCa rate"] = [round(yex[Xex["max_pirads"].round() == k].mean(), 3) if (Xex["max_pirads"].round() == k).any() else np.nan for k in tab.index]; print(tab.to_string())
D = pd.concat([X.assign(dom=0), Xex.assign(dom=1)], ignore_index=True); print("cohort-classifier AUC (0.5 = indistinguishable):")
for nm, cl in {"clinical (PSA, PSAD, volume)": clin3, "score only": ["max_pirads"], "imaging PCA components": pcs, "all inputs except age": [c for c in cols if c != "patient_age"]}.items():
    A = StandardScaler().fit_transform(D[cl]); pr = cross_val_predict(LogisticRegression(max_iter=3000), A, D["dom"], cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]; print(f"  {nm:32s} {roc_auc_score(D['dom'], pr):.3f}")
print("\n== D. OPTIMISM CHECK (Random Forest, all inputs)")
cvp = cross_val_predict(rf(), sc.transform(Xtr), ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]; m = rf().fit(sc.transform(Xtr), ytr)
print("5-fold CV AUC on training data %.3f | test-partition AUC %.3f" % (roc_auc_score(ytr, cvp), roc_auc_score(yte, m.predict_proba(sc.transform(Xte))[:, 1])))
