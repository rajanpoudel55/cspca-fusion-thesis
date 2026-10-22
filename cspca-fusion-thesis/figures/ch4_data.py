# -*- coding: utf-8 -*-
# Reported results (thesis Tables 4.1-4.9) and values derived arithmetically from them.
import math
P, N = 85, 215                      # test partition: 20% of 1,500 studies, 425 positive -> 85 positive, 215 negative
T42 = {  # model: acc, prec, rec, f1, auc   (Table 4.2, fused features, no PI-RADS)
 "Random Forest": (.730,.531,.400,.456,.765), "AdaBoost": (.743,.591,.306,.403,.758), "Tuned XGBoost": (.737,.538,.506,.518,.774),
 "Gradient Boosting": (.740,.559,.388,.458,.736), "Extra Trees": (.717,.500,.047,.086,.734), "SVM": (.700,.475,.565,.516,.734),
 "Multi-Layer Perceptron": (.720,.507,.412,.455,.717), "Linear Discriminant Analysis": (.720,.508,.388,.440,.717),
 "Logistic Regression": (.653,.426,.647,.514,.714), "Naive Bayes": (.703,.462,.282,.350,.713), "Stacking Ensemble": (.677,.453,.682,.545,.763),
 "k-Nearest Neighbours": (.700,.273,.035,.063,.683), "Decision Tree": (.620,.378,.529,.441,.598)}
T46 = {  # Table 4.6, fused + PI-RADS + biopsy pathway
 "Gradient Boosting": (.823,.700,.659,.679,.888), "Random Forest": (.800,.600,.882,.714,.888), "Naive Bayes": (.667,.459,.988,.627,.889),
 "Stacking Ensemble": (.780,.567,.941,.708,.893), "AdaBoost": (.797,.643,.635,.639,.884), "XGBoost": (.763,.547,.953,.695,.880),
 "Extra Trees": (.790,.696,.459,.553,.880), "Logistic Regression": (.753,.540,.882,.670,.870), "Linear Discriminant Analysis": (.780,.607,.635,.621,.866),
 "SVM": (.770,.562,.859,.679,.862), "Multi-Layer Perceptron": (.790,.631,.624,.627,.854), "k-Nearest Neighbours": (.733,.632,.141,.231,.830),
 "Decision Tree": (.723,.507,.847,.634,.796)}
FAMILY = {"Linear": ["Logistic Regression", "Linear Discriminant Analysis"],
          "Tree ensembles": ["Random Forest", "Extra Trees", "Gradient Boosting", "AdaBoost", "Tuned XGBoost", "XGBoost"],
          "Stacking": ["Stacking Ensemble"], "Single decision tree": ["Decision Tree"],
          "Support vector machine": ["SVM"], "Instance-based and probabilistic": ["k-Nearest Neighbours", "Naive Bayes"],
          "Neural network": ["Multi-Layer Perceptron"]}

def counts(rec, prec, pos=P, neg=N):
    tp = round(rec * pos); fn = pos - tp
    fp = round(tp * (1 / prec - 1)); tn = neg - fp
    return tp, fn, fp, tn
def derived(row, pos=P, neg=N):
    a, p, r, f, u = row
    tp, fn, fp, tn = counts(r, p, pos, neg)
    spec = tn / neg; bal = (r + spec) / 2; flagged = (tp + fp) / (pos + neg)
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, spec=spec, bal=bal, flagged=flagged, acc_rec=(tp + tn) / (pos + neg))
def wilson(k, n, z=1.96):
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h
def ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: -vals[i]); rk = [0] * len(vals); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]: j += 1
        for t in range(i, j + 1): rk[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return rk
def spearman(x, y):
    rx, ry = ranks(x), ranks(y); n = len(x); mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry)); den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den
# external cohort
EXT = dict(n=782, prev=.575, acc=.669, prec=.720, rec=.696, auc=.745)
EXT_POS = round(EXT["prev"] * EXT["n"]); EXT_NEG = EXT["n"] - EXT_POS
EXT_C = counts(EXT["rec"], EXT["prec"], EXT_POS, EXT_NEG)

if __name__ == "__main__":
    print("--- reconstruction check: reconstructed accuracy vs reported accuracy (Table 4.2 and 4.6)")
    mx = 0
    for nm, T in (("4.2", T42), ("4.6", T46)):
        for k, v in T.items():
            d = derived(v); diff = abs(d["acc_rec"] - v[0]); mx = max(mx, diff)
            print(nm, f"{k:30s} TP{d['tp']:>3} FN{d['fn']:>3} FP{d['fp']:>3} TN{d['tn']:>3}  acc_rec {d['acc_rec']:.3f} vs {v[0]:.3f}  spec {d['spec']:.3f} bal {d['bal']:.3f} flagged {d['flagged']*100:4.1f}%")
    print("max |reconstructed - reported accuracy|:", round(mx, 4))
    tp, fn, fp, tn = EXT_C
    print("external: pos", EXT_POS, "neg", EXT_NEG, "TP/FN/FP/TN", EXT_C, "acc", round((tp+tn)/EXT['n'],3), "spec", round(tn/EXT_NEG,3), "NPV", round(tn/(tn+fn),3), "bal", round((EXT['rec']+tn/EXT_NEG)/2,3))
    print("XGB thresholds: default", counts(.506,.538), "-> acc", round((counts(.506,.538)[0]+counts(.506,.538)[3])/300,3), "| 0.74:", counts(.318,.659), "-> acc", round((counts(.318,.659)[0]+counts(.318,.659)[3])/300,3))
    names42 = list(T42); names46 = list(T46)
    print("Spearman AUC vs recall 4.2:", round(spearman([T42[k][4] for k in names42],[T42[k][2] for k in names42]),2), "| 4.6:", round(spearman([T46[k][4] for k in names46],[T46[k][2] for k in names46]),2))
    print("Wilson recall CIs:", {k: tuple(round(x,3) for x in wilson(round(T46[k][2]*P), P)) for k in ("Stacking Ensemble","Random Forest","XGBoost","Gradient Boosting")}, "| external", tuple(round(x,3) for x in wilson(EXT_C[0], EXT_POS)))
    # paired change 4.2 -> 4.6
    mp = {"Tuned XGBoost": "XGBoost"}
    d_auc = []; d_rec = []
    for k, v in T42.items():
        v2 = T46[mp.get(k, k)]; d_auc.append(v2[4]-v[4]); d_rec.append(v2[2]-v[2])
        print(f"{k:30s} dAUC {v2[4]-v[4]:+.3f} dRec {v2[2]-v[2]:+.3f} dPrec {v2[1]-v[1]:+.3f} dF1 {v2[3]-v[3]:+.3f}")
    print("mean dAUC", round(sum(d_auc)/13,3), "mean dRec", round(sum(d_rec)/13,3), "| AUC improved in", sum(1 for x in d_auc if x>0), "of 13; recall improved in", sum(1 for x in d_rec if x>0))
    for fam, ms in FAMILY.items():
        a = [T42[m][4] for m in ms if m in T42]; r = [T42[m][2] for m in ms if m in T42]
        a6 = [T46[m][4] for m in ms if m in T46]; r6 = [T46[m][2] for m in ms if m in T46]
        print(fam, "| 4.2 mean AUC", round(sum(a)/len(a),3), "mean rec", round(sum(r)/len(r),3), "| 4.6 mean AUC", round(sum(a6)/len(a6),3), "mean rec", round(sum(r6)/len(r6),3))

def hm_se(auc, n1, n2):
    """Hanley-McNeil (1982) standard error of an AUC given the numbers of positive (n1) and negative (n2) cases."""
    q1 = auc / (2 - auc); q2 = 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc * auc) + (n2 - 1) * (q2 - auc * auc)) / (n1 * n2)
    return math.sqrt(var)
def hm_ci(auc, n1, n2, z=1.96):
    se = hm_se(auc, n1, n2); return max(0, auc - z * se), min(1, auc + z * se)
