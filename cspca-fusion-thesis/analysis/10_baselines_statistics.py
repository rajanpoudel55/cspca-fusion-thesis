"""Step 10. Clinical and PI-RADS baselines, bootstrap intervals, paired tests, McNemar tests, repeated cross-validation (Sections 3.9.5, 3.9.6, 4.12).
Set CSPCA_FAST=1 for a quick smoke test (200 bootstrap samples, one CV repeat). Bootstrap seed 7. Expected (real data): logistic regression on age, log PSA, volume
hold-out ROC-AUC 0.757; with PI-RADS and PSA density 0.878; locked Stacking 0.884 (0.845-0.917); locked minus fully automated Stacking +0.128 (Holm p 0.008)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import t as tdist, binomtest
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import SEED, FAST, load_internal, load_external, split_indices, make_locked
fused, raw, fp, X, y = load_internal(); cols = X.columns.tolist(); itr, ite = split_indices(y); ext = load_external("unique"); Xex = ext[cols].reset_index(drop=True); yex = ext["csPCa_label"].to_numpy()
X2, Xex2 = X.assign(logpsa=np.log(X["psa"])), Xex.assign(logpsa=np.log(Xex["psa"])); Xtr2, Xte2 = X2.iloc[itr], X2.iloc[ite]; ytr_, yte_ = y.iloc[itr].to_numpy(), y.iloc[ite].to_numpy()
LRm = lambda: LogisticRegression(max_iter=2000, class_weight="balanced"); RFm = lambda: RandomForestClassifier(n_estimators=300, max_depth=10, random_state=SEED, class_weight="balanced")
fused_cols = [c for c in cols if c != "max_pirads"]; CL3 = ["patient_age", "logpsa", "prostate_volume"]
def fit_pred(cl, kind):
    s = StandardScaler().fit(Xtr2[cl]); m = kind().fit(s.transform(Xtr2[cl]), ytr_); return m.predict_proba(s.transform(Xte2[cl]))[:, 1], m.predict_proba(s.transform(Xex2[cl]))[:, 1]
spec = {"LR: age, log PSA, volume": (CL3, LRm), "LR: PSA density alone": (["psad"], LRm), "RF: clinical (age, PSA, PSAD, volume)": (["patient_age", "psa", "psad", "prostate_volume"], RFm), "LR: PI-RADS + PSA density": (["max_pirads", "psad"], LRm),
        "LR: age, log PSA, volume + PI-RADS": (CL3 + ["max_pirads"], LRm), "RF: fused (fully automated)": (fused_cols, RFm), "Stacking: fused (fully automated)": (fused_cols, make_locked), "Stacking: PI-RADS-informed (locked)": (cols, make_locked)}
P_te, P_ex = {}, {}
for nm, (cl, kind) in spec.items(): P_te[nm], P_ex[nm] = fit_pred(cl, kind); print("fitted:", nm)
def rules(D): return {"Rule: PSA density >= 0.15": (D["psad"] >= .15).astype(float).to_numpy(), "Rule: PI-RADS >= 3": (D["max_pirads"] >= 3).astype(float).to_numpy(), "Rule: PI-RADS >= 4": (D["max_pirads"] >= 4).astype(float).to_numpy(),
    "Rule: PI-RADS >= 4, or 3 with PSA density >= 0.15": ((D["max_pirads"] >= 4) | ((D["max_pirads"] >= 3) & (D["psad"] >= .15))).astype(float).to_numpy(), "PI-RADS score alone (ROC-AUC only)": D["max_pirads"].to_numpy(dtype=float)}
P_te.update(rules(Xte2)); P_ex.update(rules(Xex2))
rng = np.random.default_rng(7); B = 200 if FAST else 2000
def strat_idx(v):
    pos, neg = np.where(v == 1)[0], np.where(v == 0)[0]; return [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(B)]
IDX_te, IDX_ex = strat_idx(yte_), strat_idx(yex)
def metr(v, s, thr=0.5):
    h = s >= thr; tp = (h & (v == 1)).sum(); fn = ((~h) & (v == 1)).sum(); fp_ = (h & (v == 0)).sum(); tn = ((~h) & (v == 0)).sum(); return tp / max(tp + fn, 1), tn / max(tn + fp_, 1), tp / max(tp + fp_, 1)
def summarize(P, v, IDX, names):
    rows = []
    for nm in names:
        s = P[nm]; aucs = np.array([roc_auc_score(v[i], s[i]) for i in IDX]); row = [nm, f"{roc_auc_score(v, s):.3f} ({np.percentile(aucs, 2.5):.3f}-{np.percentile(aucs, 97.5):.3f})"]
        if "score alone" in nm: row += ["-", "-", "-"]
        else:
            ms = np.array([metr(v[i], s[i]) for i in IDX]); m0 = metr(v, s); row += [f"{m0[k]:.3f} ({np.percentile(ms[:, k], 2.5):.3f}-{np.percentile(ms[:, k], 97.5):.3f})" for k in range(3)]
        rows.append(row)
    return pd.DataFrame(rows, columns=["Model", "ROC-AUC (95% CI)", "Recall", "Specificity", "Precision"])
def paired(P, v, IDX, a, b):
    da = np.array([roc_auc_score(v[i], P[a][i]) - roc_auc_score(v[i], P[b][i]) for i in IDX]); d0 = roc_auc_score(v, P[a]) - roc_auc_score(v, P[b])
    return d0, np.percentile(da, 2.5), np.percentile(da, 97.5), min(1.0, 2 * max(min((da <= 0).mean(), (da >= 0).mean()), 1 / B))
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
    return adj
LK, ST, RFF, RFC = "Stacking: PI-RADS-informed (locked)", "Stacking: fused (fully automated)", "RF: fused (fully automated)", "RF: clinical (age, PSA, PSAD, volume)"
planned = [("LR: age, log PSA, volume", RFF), ("LR: age, log PSA, volume", ST), (RFC, RFF), (ST, "LR: age, log PSA, volume"), (LK, ST), (LK, "PI-RADS score alone (ROC-AUC only)"), (LK, "LR: PI-RADS + PSA density"), (LK, "LR: age, log PSA, volume + PI-RADS")]
print("\n=== 1. HOLD-OUT (95% bootstrap intervals)"); print(summarize(P_te, yte_, IDX_te, list(P_te)).to_string(index=False))
res = [paired(P_te, yte_, IDX_te, a, b) for a, b in planned]; adj = holm([r[3] for r in res]); print("\n=== 2. PLANNED PAIRED COMPARISONS OF ROC-AUC (first minus second; Holm-adjusted p)")
for (a, b), r, q in zip(planned, res, adj): print(f"{a}  vs  {b}: diff {r[0]:+.3f} (95% CI {r[1]:+.3f} to {r[2]:+.3f}) | p {r[3]:.4f} | Holm p {q:.4f}")
def mcnemar(v, sa, sb):
    ha, hb = sa >= .5, sb >= .5; out = []
    for lab, mask, f in (("recall", v == 1, lambda h: h), ("specificity", v == 0, lambda h: ~h)):
        ca, cb = f(ha)[mask], f(hb)[mask]; b_, c_ = int((ca & ~cb).sum()), int((~ca & cb).sum()); out.append(f"{lab}: model-only-correct {b_}, rule-only-correct {c_}, exact McNemar p {binomtest(min(b_, c_), b_ + c_, .5).pvalue if b_ + c_ else 1.0:.4f}")
    return "; ".join(out)
print("\n=== 3. EXACT McNEMAR: locked model versus rules")
for r_ in ("Rule: PI-RADS >= 3", "Rule: PI-RADS >= 4", "Rule: PI-RADS >= 4, or 3 with PSA density >= 0.15"): print(r_, "->", mcnemar(yte_, P_te[LK], P_te[r_]))
print("\n=== 4. REPEATED 5-FOLD CV on all studies (corrected resampled t-test, Nadeau and Bengio)")
nrep = 1 if FAST else 3; rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=nrep, random_state=SEED); yall = y.to_numpy()
cvm = {"LR: age, log PSA, volume": (CL3, LRm), "LR: PI-RADS + PSA density": (["max_pirads", "psad"], LRm), "LR: age, log PSA, volume + PI-RADS": (CL3 + ["max_pirads"], LRm), "RF: clinical": (["patient_age", "psa", "psad", "prostate_volume"], RFm), "RF: fused (fully automated)": (fused_cols, RFm), "RF: PI-RADS-informed": (cols, RFm)}
fa = {k: [] for k in cvm}
for tr, te in rs.split(X2, yall):
    for k, (cl, kind) in cvm.items(): s = StandardScaler().fit(X2.iloc[tr][cl]); m = kind().fit(s.transform(X2.iloc[tr][cl]), yall[tr]); fa[k].append(roc_auc_score(yall[te], m.predict_proba(s.transform(X2.iloc[te][cl]))[:, 1]))
kf = 5 * nrep; ratio = 0.25; tc = tdist.ppf(.975, kf - 1)
for k, v_ in fa.items(): v_ = np.array(v_); se = np.sqrt((1 / kf + ratio) * v_.var(ddof=1)); print(f"{k:38s} mean ROC-AUC {v_.mean():.3f} (95% CI {v_.mean() - tc * se:.3f}-{v_.mean() + tc * se:.3f})")
for a, b in (("RF: fused (fully automated)", "LR: age, log PSA, volume"), ("RF: fused (fully automated)", "RF: clinical"), ("RF: PI-RADS-informed", "RF: fused (fully automated)"), ("RF: PI-RADS-informed", "LR: PI-RADS + PSA density"), ("RF: PI-RADS-informed", "LR: age, log PSA, volume + PI-RADS")):
    d_ = np.array(fa[a]) - np.array(fa[b]); se = np.sqrt((1 / kf + ratio) * d_.var(ddof=1)); p = 2 * tdist.sf(abs(d_.mean() / se), kf - 1) if se > 0 else float("nan"); print(f"{a}  minus  {b}: {d_.mean():+.3f} (95% CI {d_.mean() - tc * se:+.3f} to {d_.mean() + tc * se:+.3f}) | corrected t-test p {p:.4f}")
print("\n=== 5. EXTERNAL COHORT: baselines fixed on the training partition")
names_ex = [LK, "LR: age, log PSA, volume", "LR: PSA density alone", "LR: PI-RADS + PSA density", "LR: age, log PSA, volume + PI-RADS", "Rule: PSA density >= 0.15", "Rule: PI-RADS >= 3", "Rule: PI-RADS >= 4", "Rule: PI-RADS >= 4, or 3 with PSA density >= 0.15", "PI-RADS score alone (ROC-AUC only)"]
print(summarize(P_ex, yex, IDX_ex, names_ex).to_string(index=False))
for nm in names_ex[1:]: r = paired(P_ex, yex, IDX_ex, LK, nm); print(f"locked model minus {nm}: {r[0]:+.3f} (95% CI {r[1]:+.3f} to {r[2]:+.3f}) | p {r[3]:.4f}")
