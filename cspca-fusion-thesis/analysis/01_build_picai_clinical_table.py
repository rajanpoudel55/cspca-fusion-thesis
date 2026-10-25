"""Step 1. PI-CAI clinical table (code recovered from the project notebook).
Input : picai_labels/clinical_information/marksheet.csv      Output: picai_clinical_clean.csv
Rules : PSA density is recomputed as PSA / prostate volume where missing; remaining gaps in PSA, PSA density and prostate volume are
        filled with the median of the whole cohort (before any partition); label = case_csPCa == 'YES' (ISUP grade group >= 2)."""
import pandas as pd
from common import path, MARKSHEET
c = pd.read_csv(MARKSHEET)
c["psad"] = c["psad"].fillna(c["psa"] / c["prostate_volume"])
for col in ("psa", "psad", "prostate_volume"): c[col] = c[col].fillna(c[col].median())
c["csPCa_label"] = (c["case_csPCa"] == "YES").astype(int)
out = c[["patient_id", "study_id", "patient_age", "psa", "psad", "prostate_volume", "center", "csPCa_label"]]
print(out.isnull().sum().to_string()); print("rows:", len(out), "| csPCa:", int(out.csPCa_label.sum()), "(expected 1500 and 425)")
out.to_csv(path("picai_clinical_clean.csv"), index=False)
