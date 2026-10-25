"""Step 4. External cohort (Prostate-MRI-US-Biopsy, TCIA): per-patient clinical table, T2W series catalogue and final cohort (code recovered from the notebook).
Usage: python 04_build_external_cohort.py --biopsy-file <Biopsy Data spreadsheet> --target-file <Target Data spreadsheet> --dicom-root <.../prostate_mri_us_biopsy>
Outputs: external_validation/external_clinical_clean.csv (expected 1150 x 7) and external_validation/final_external_cohort.csv (expected 782 x 9).
Rules : csPCa_label = maximum Gleason sum over biopsy cores >= 7 (note: the training label is case-level ISUP grade group >= 2);
        max_pirads = maximum UCLA score (similar to PI-RADS v2) per patient from the target data, median-filled where absent;
        age is not recorded; PSA density = PSA / prostate volume. T2W series = first series folder whose name contains 't2' and 'spc'
        (directory order); patients with ultrasound only (369) have no T2W series and are excluded."""
import argparse, os
import pandas as pd
from common import path
ap = argparse.ArgumentParser(); ap.add_argument("--biopsy-file", required=True); ap.add_argument("--target-file", required=True); ap.add_argument("--dicom-root", required=True); a = ap.parse_args()
rd = lambda f: pd.read_excel(f) if f.lower().endswith(("xls", "xlsx")) else pd.read_csv(f)
biopsy_df, target_df = rd(a.biopsy_file), rd(a.target_file)
biopsy_df["gleason_score"] = biopsy_df["Primary Gleason"] + biopsy_df["Secondary Gleason"]
pb = biopsy_df.groupby("Patient Number").agg(psa=("PSA (ng/mL)", "first"), prostate_volume=("Prostate Volume (CC)", "first"), max_gleason=("gleason_score", "max")).reset_index()
pb["psad"] = pb["psa"] / pb["prostate_volume"]; pb["csPCa_label"] = (pb["max_gleason"] >= 7).astype(int)
pt = target_df.groupby("Patient ID").agg(max_pirads=("UCLA Score (Similar to PIRADS v2)", "max")).reset_index().rename(columns={"Patient ID": "Patient Number"})
ec = pb.merge(pt, on="Patient Number", how="left"); ec["max_pirads"] = ec["max_pirads"].fillna(ec["max_pirads"].median())
os.makedirs(path("external_validation"), exist_ok=True); ec.to_csv(path("external_validation", "external_clinical_clean.csv"), index=False); print("external clinical:", ec.shape, "(expected 1150 x 7)")
catalog = []
for patient in sorted(f for f in os.listdir(a.dicom_root) if os.path.isdir(os.path.join(a.dicom_root, f))):
    found = None
    for study in sorted(os.listdir(os.path.join(a.dicom_root, patient))):
        sp = os.path.join(a.dicom_root, patient, study)
        if not os.path.isdir(sp): continue
        for series in sorted(os.listdir(sp)):
            ser = os.path.join(sp, series)
            if os.path.isdir(ser) and "t2" in series.lower() and "spc" in series.lower():
                found = {"patient": patient, "series_path": ser, "num_images": len([f for f in os.listdir(ser) if f.endswith(".dcm")])}; break
        if found: break
    if found: catalog.append(found)
final = ec.merge(pd.DataFrame(catalog).rename(columns={"patient": "Patient Number"}), on="Patient Number", how="inner")
print("final external cohort:", final.shape, "(expected 782 x 9) | csPCa prevalence:", round(final.csPCa_label.mean(), 3), "(expected 0.575)")
final.to_csv(path("external_validation", "final_external_cohort.csv"), index=False)
