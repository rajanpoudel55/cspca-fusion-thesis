"""Prints software versions, platform and file fingerprints (rows x columns and SHA-256 prefix) for the derived tables."""
import sys, platform, importlib, os
import pandas as pd
from common import path, MARKSHEET, sha256_prefix
print(sys.version.split()[0], "|", platform.platform(), "|", platform.machine())
for m in ["numpy", "pandas", "scipy", "sklearn", "xgboost", "tensorflow", "keras", "SimpleITK", "pydicom", "cv2", "matplotlib", "PIL", "imblearn"]:
    try: print(f"{m:12s}", importlib.import_module(m).__version__)
    except Exception: print(f"{m:12s} not available")
FILES = [MARKSHEET, path("picai_clinical_clean.csv"), path("cnn_features_t2w.csv"), path("fused_final_dataset.csv"), path("fused_final_dataset_with_pirads.csv"),
         path("external_validation", "external_clinical_clean.csv"), path("external_validation", "final_external_cohort.csv"), path("external_validation", "external_cnn_features_t2w.csv"), path("selected_t2w_files.csv")]
for f in FILES:
    if os.path.exists(f): d = pd.read_csv(f); print(f"{os.path.relpath(f, path('.')):62s} {d.shape[0]:5d} x {d.shape[1]:4d}  {sha256_prefix(f)}")
    else: print(f"{os.path.relpath(f, path('.')):62s} not found")
print("Note: feature tables can differ in the last decimals on other hardware, so compare shapes first and checksums second.")
