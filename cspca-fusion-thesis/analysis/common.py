"""Shared helpers for the csPCa reproducibility package.

Set the environment variable CSPCA_ROOT to the folder that holds the data and derived CSV files
(default: the current directory). All stochastic steps use SEED = 42 unless a script states otherwise.
"""
import os, hashlib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

SEED = 42
ROOT = os.environ.get("CSPCA_ROOT", ".")
FAST = bool(os.environ.get("CSPCA_FAST"))          # smoke-test switch: fewer resamples and repeats
def path(*p): return os.path.join(ROOT, *p)
MARKSHEET = path("picai_labels", "clinical_information", "marksheet.csv")

def get_max_pirads(v):
    """Maximum lesion-level PI-RADS score in a marksheet string such as '4, 3'."""
    if pd.isna(v): return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)): return int(v)       # column stored as numbers
    s = [int(float(x)) for x in str(v).replace(";", ",").split(",") if x.strip().replace(".", "", 1).isdigit()]
    return max(s) if s else np.nan

def sha256_prefix(f, n=16): return hashlib.sha256(open(f, "rb").read()).hexdigest()[:n]

def load_internal():
    """Returns fused (as saved), raw marksheet, fused+PI-RADS table, design matrix X (configuration B) and labels y."""
    fused = pd.read_csv(path("fused_final_dataset.csv")); fused["patient_id"] = fused["patient_id"].astype(int)
    raw = pd.read_csv(MARKSHEET); raw["max_pirads"] = raw["lesion_PIRADS"].apply(get_max_pirads)
    # PI-RADS is taken from the first marksheet record of each patient (affects the 24 extra studies of repeat patients)
    fp = fused.merge(raw[["patient_id", "max_pirads"]].drop_duplicates("patient_id"), on="patient_id", how="left")
    fp["max_pirads"] = fp["max_pirads"].fillna(fp["max_pirads"].median())
    X = fp.drop(columns=["patient_id", "csPCa_label"]).reset_index(drop=True); y = fused["csPCa_label"].reset_index(drop=True)
    return fused, raw, fp, X, y

def load_config(cfg):
    """Design matrices: A fully automated, B PI-RADS-informed, C biopsy-pathway (not a pre-biopsy model)."""
    fused, raw, fp, X_b, y = load_internal()
    if cfg == "A": return fused.drop(columns=["patient_id", "csPCa_label"]), y
    if cfg == "B": return X_b, y
    fph = fp.merge(raw[["patient_id", "histopath_type"]].drop_duplicates("patient_id"), on="patient_id", how="left")
    fph["histopath_type"] = fph["histopath_type"].fillna("No_Biopsy_Needed")
    fph = pd.get_dummies(fph, columns=["histopath_type"], drop_first=True)
    return fph.drop(columns=["patient_id", "csPCa_label"]), y

def split_indices(y):
    """The single stratified 80/20 hold-out partition (made on studies, seed 42)."""
    return train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)

def make_locked():
    """The locked Stacking model of the external validation (Table A.1)."""
    return StackingClassifier(estimators=[
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=10, random_state=SEED, class_weight="balanced")),
        ("ada", AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=SEED)),
        ("gb", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED))],
        final_estimator=LogisticRegression(class_weight="balanced"), cv=5)

def load_external(basis="unique"):
    """External cohort in the locked model's feature space.
    basis='unique': PCA fitted on the distinct PI-CAI feature vectors (the basis used for the results in the thesis).
    basis='merged': PCA fitted on the merged study rows, i.e. the basis of the training features (Appendix I)."""
    fused = pd.read_csv(path("fused_final_dataset.csv")); fused["patient_id"] = fused["patient_id"].astype(int)
    t2w = pd.read_csv(path("cnn_features_t2w.csv")); t2w["patient_id"] = t2w["patient_id"].astype(int)
    fc = [c for c in t2w.columns if c.startswith("cnn_feat_")]; pcs = [c for c in fused.columns if c.startswith("cnn_pca_")]
    fit_on = t2w[fc].to_numpy() if basis == "unique" else t2w.set_index("patient_id").loc[fused["patient_id"], fc].to_numpy()
    sc = StandardScaler().fit(fit_on); pca = PCA(n_components=len(pcs), random_state=SEED).fit(sc.transform(fit_on))
    ecnn = pd.read_csv(path("external_validation", "external_cnn_features_t2w.csv")); fe = [c for c in ecnn.columns if c.startswith("cnn_feat_")]
    e = pd.DataFrame(pca.transform(sc.transform(ecnn[fe].to_numpy())), columns=pcs); e["Patient Number"] = ecnn["Patient Number"].values
    coh = pd.read_csv(path("external_validation", "final_external_cohort.csv")); ext = coh.merge(e, on="Patient Number", how="inner")
    ext["patient_age"] = pd.read_csv(path("picai_clinical_clean.csv"))["patient_age"].median()   # age is not recorded externally
    return ext

def metrics_row(y, p, thr=0.5):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    h = (np.asarray(p) >= thr).astype(int); tn, fp_, fn, tp = confusion_matrix(y, h).ravel()
    return dict(accuracy=accuracy_score(y, h), precision=precision_score(y, h, zero_division=0), recall=recall_score(y, h), f1=f1_score(y, h),
                specificity=float(tn / (tn + fp_)), roc_auc=roc_auc_score(y, p), TP=int(tp), FN=int(fn), FP=int(fp_), TN=int(tn))
