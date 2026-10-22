"""Step 6. Thirteen algorithms with grid-search tuning (Table A.1) on one configuration: A fully automated, B PI-RADS-informed, C biopsy-pathway.
Usage: python 06_train_algorithms.py --config B
Procedure: stratified 80/20 partition (seed 42); standardisation fitted on the training partition; five-fold stratified shuffled CV (seed 42), ROC-AUC scoring; default threshold 0.5.
Known issue: the fully automated XGBoost of Table 4.3 (ROC-AUC 0.774) came from an earlier tuning run with other settings and is NOT reproduced here
(this grid gives about 0.746; Appendix H). All other rows checked in Appendix H were reproduced exactly with these settings."""
import argparse, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from common import SEED, path, load_config, split_indices, metrics_row
GRIDS = {
 "LogisticRegression": (LogisticRegression(max_iter=1000, class_weight="balanced"), {"C": [0.01, 0.1, 1, 10]}),
 "DecisionTree": (DecisionTreeClassifier(random_state=SEED, class_weight="balanced"), {"max_depth": [5, 10, None]}),
 "RandomForest": (RandomForestClassifier(random_state=SEED, class_weight="balanced"), {"n_estimators": [100, 300], "max_depth": [10, 20]}),
 "ExtraTrees": (ExtraTreesClassifier(random_state=SEED, class_weight="balanced"), {"n_estimators": [100, 300], "max_depth": [10, 20]}),
 "GradientBoosting": (GradientBoostingClassifier(random_state=SEED), {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1]}),
 "AdaBoost": (AdaBoostClassifier(random_state=SEED), {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]}),
 "SVM": (SVC(probability=True, class_weight="balanced", random_state=SEED), {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]}),
 "KNN": (KNeighborsClassifier(), {"n_neighbors": [5, 10, 15]}),
 "NaiveBayes": (GaussianNB(), {}), "LDA": (LinearDiscriminantAnalysis(), {}),
 "MLP": (MLPClassifier(max_iter=1000, random_state=SEED), {"hidden_layer_sizes": [(50,), (100,)], "alpha": [0.001, 0.01]}),
 "XGBoost": (XGBClassifier(scale_pos_weight=2.5, eval_metric="logloss", random_state=SEED), {"n_estimators": [200, 300], "max_depth": [3, 5, 6], "learning_rate": [0.01, 0.1]})}
ap = argparse.ArgumentParser(); ap.add_argument("--config", choices=["A", "B", "C"], required=True); cfg = ap.parse_args().config
X, y = load_config(cfg); itr, ite = split_indices(y); sc = StandardScaler().fit(X.iloc[itr]); Xtr, Xte = sc.transform(X.iloc[itr]), sc.transform(X.iloc[ite]); ytr, yte = y.iloc[itr].to_numpy(), y.iloc[ite].to_numpy()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED); best = {}; params = {}
for name, (m, g) in GRIDS.items():
    if g: gs = GridSearchCV(m, g, cv=cv, scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr); best[name], params[name] = gs.best_estimator_, gs.best_params_
    else: best[name], params[name] = m.fit(Xtr, ytr), {}
best["Stacking"] = StackingClassifier(estimators=[("rf", best["RandomForest"]), ("ada", best["AdaBoost"]), ("gb", best["GradientBoosting"])], final_estimator=LogisticRegression(class_weight="balanced"), cv=5).fit(Xtr, ytr)
rows, preds = [], {}
for name, m in best.items():
    p = m.predict_proba(Xte)[:, 1]; preds[name] = p; rows.append(dict(config=cfg, model=name, best_params=str(params.get(name, "")), **metrics_row(yte, p)))
R = pd.DataFrame(rows); print(R.drop(columns="best_params").round(3).to_string(index=False)); R.to_csv(path(f"results_config_{cfg}.csv"), index=False)
pd.DataFrame(preds).assign(y_true=yte, row=ite).to_csv(path(f"predictions_config_{cfg}.csv"), index=False)
