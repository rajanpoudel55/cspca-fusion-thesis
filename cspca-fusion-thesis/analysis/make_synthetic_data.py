"""Writes a small SYNTHETIC dataset with the same file structure as the real one, so the scripts can be smoke-tested without the PI-CAI and TCIA data.
The numbers it produces mean nothing clinically. Usage: CSPCA_ROOT=/tmp/synth python make_synthetic_data.py"""
import os
import numpy as np, pandas as pd
from common import path
rng = np.random.default_rng(0); n_pat, n_rep, nf = 380, 20, 64
pid = np.arange(10000, 10000 + n_pat); studies = np.concatenate([pid, rng.choice(pid, n_rep, replace=False)]); n = len(studies)
latent = rng.normal(size=n); center = rng.choice(["RUMC", "ZGT", "PCNN"], n, p=[.53, .235, .235])
psa = np.round(np.exp(2.0 + 0.3 * latent + rng.normal(0, .4, n)), 1); vol = np.round(np.exp(4.0 - 0.2 * latent + rng.normal(0, .3, n)), 0); psad = psa / vol
risk = 1 / (1 + np.exp(-(1.6 * latent - 1.1))); y = (rng.random(n) < risk).astype(int)
pirads = np.clip(np.round(3 + 1.2 * latent + 0.8 * y + rng.normal(0, .7, n)), 1, 5).astype(int)
mark = pd.DataFrame({"patient_id": studies, "study_id": np.arange(1000000, 1000000 + n), "patient_age": rng.integers(50, 82, n), "psa": psa, "psad": psad, "prostate_volume": vol, "center": center,
    "case_csPCa": np.where(y == 1, "YES", "NO"), "case_ISUP": np.where(y == 1, rng.choice([2, 3, 4, 5], n, p=[.6, .25, .08, .07]), rng.choice([0, 1], n, p=[.8, .2])),
    "lesion_PIRADS": [(f"{p}, {max(1, p - 1)}" if r2 < .15 else str(p)) if r > .05 else np.nan for p, r, r2 in zip(pirads, rng.random(n), rng.random(n))], "histopath_type": np.where(rng.random(n) > .2, rng.choice(["MRBx", "SysBx", "SysBx+MRBx"], n), None)})
mark.loc[rng.choice(n, 30, replace=False), "psad"] = np.nan; mark.loc[rng.choice(n, 10, replace=False), "psa"] = np.nan; mark.loc[rng.choice(n, 8, replace=False), "prostate_volume"] = np.nan
os.makedirs(os.path.dirname(path("picai_labels", "clinical_information", "marksheet.csv")), exist_ok=True); mark.to_csv(path("picai_labels", "clinical_information", "marksheet.csv"), index=False)
lat_u = pd.Series(latent, index=studies).groupby(level=0).first(); feats = pd.DataFrame(rng.normal(size=(n_pat, nf)) + 0.15 * lat_u.loc[pid].to_numpy()[:, None] * rng.normal(size=nf), columns=[f"cnn_feat_{j}" for j in range(nf)]); feats.insert(0, "patient_id", pid)
feats.to_csv(path("cnn_features_t2w.csv"), index=False)
ne = 150; le = rng.normal(0.3, 1, ne); ye = (rng.random(ne) < 1 / (1 + np.exp(-(1.2 * le)))).astype(int); pe = np.round(np.exp(1.9 + 0.25 * le + rng.normal(0, .4, ne)), 1); ve = np.round(np.exp(3.8 + rng.normal(0, .3, ne)), 0)
coh = pd.DataFrame({"Patient Number": np.arange(1, ne + 1), "psa": pe, "prostate_volume": ve, "max_gleason": np.where(ye == 1, 7, 6), "psad": pe / ve, "csPCa_label": ye, "max_pirads": np.clip(np.round(3.6 + 0.8 * le + rng.normal(0, .8, ne)), 0, 5), "series_path": "n/a", "num_images": 30})
os.makedirs(path("external_validation"), exist_ok=True); coh.to_csv(path("external_validation", "final_external_cohort.csv"), index=False)
ef = pd.DataFrame(rng.normal(size=(ne, nf)), columns=[f"cnn_feat_{j}" for j in range(nf)]); ef.insert(0, "Patient Number", coh["Patient Number"]); ef.to_csv(path("external_validation", "external_cnn_features_t2w.csv"), index=False); print("synthetic data written to", os.path.abspath(path(".")))
