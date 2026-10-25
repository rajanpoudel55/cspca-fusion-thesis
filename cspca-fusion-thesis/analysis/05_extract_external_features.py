"""Step 5. Frozen ResNet50 features for the external cohort: same preprocessing as step 2, DICOM series read with SimpleITK.
Input: external_validation/final_external_cohort.csv    Output: external_validation/external_cnn_features_t2w.csv (expected 782 x 2049)."""
import numpy as np, pandas as pd, SimpleITK as sitk, cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from common import path
base = ResNet50(weights="imagenet", include_top=False, pooling="avg")
def load_middle_slice_dicom(series_folder, size=(224, 224)):
    r = sitk.ImageSeriesReader(); r.SetFileNames(r.GetGDCMSeriesFileNames(series_folder)); arr = sitk.GetArrayFromImage(r.Execute()); s = arr[arr.shape[0] // 2]
    s = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8); return cv2.cvtColor(cv2.resize(s, size), cv2.COLOR_GRAY2RGB)
coh = pd.read_csv(path("external_validation", "final_external_cohort.csv")); rows = []
for i, r in coh.iterrows():
    try:
        f = base.predict(preprocess_input(np.expand_dims(load_middle_slice_dicom(r["series_path"]), 0).astype(np.float32)), verbose=0).flatten()
        row = {"Patient Number": r["Patient Number"]}; row.update({f"cnn_feat_{j}": v for j, v in enumerate(f)}); rows.append(row)
    except Exception as e: print("failed", r["Patient Number"], e)
    if i % 100 == 0: print(f"{i}/{len(coh)}")
df = pd.DataFrame(rows); print(df.shape, "(expected 782 x 2049)"); df.to_csv(path("external_validation", "external_cnn_features_t2w.csv"), index=False)
