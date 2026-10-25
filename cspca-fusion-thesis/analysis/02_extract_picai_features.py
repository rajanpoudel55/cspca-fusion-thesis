"""Step 2. Frozen ResNet50 features from the central axial slice of one T2-weighted volume per PI-CAI patient.
Usage: python 02_extract_picai_features.py --images picai_images [--manifest selected_t2w_files.csv]

The thesis results used ONE T2W file per patient folder, chosen as the first file returned by os.listdir (an unspecified order).
For the 23 patients with two T2W files this selection depends on the file system, so the exact selection used in the thesis is
recorded in selected_t2w_files.csv; pass it with --manifest to reproduce the features. Without a manifest the sorted-first file is used."""
import argparse, os
import numpy as np, pandas as pd, SimpleITK as sitk, cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from common import path
ap = argparse.ArgumentParser(); ap.add_argument("--images", default="picai_images"); ap.add_argument("--manifest", default=None); ap.add_argument("--out", default=path("cnn_features_t2w.csv")); a = ap.parse_args()
base = ResNet50(weights="imagenet", include_top=False, pooling="avg")      # 2,048-dimensional global-average-pooled descriptor
def load_middle_slice(mha_path, size=(224, 224)):
    arr = sitk.GetArrayFromImage(sitk.ReadImage(mha_path)); s = arr[arr.shape[0] // 2]       # central axial slice
    s = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)                       # min-max to 8 bit
    return cv2.cvtColor(cv2.resize(s, size), cv2.COLOR_GRAY2RGB)                               # 224 x 224 x 3
def features(p): return base.predict(preprocess_input(np.expand_dims(load_middle_slice(p), 0).astype(np.float32)), verbose=0).flatten()
chosen = dict(zip(pd.read_csv(a.manifest)["patient_id"].astype(str), pd.read_csv(a.manifest)["selected_file"])) if a.manifest else {}
rows = []
for i, pid in enumerate(sorted(os.listdir(a.images))):
    fol = os.path.join(a.images, pid)
    if not os.path.isdir(fol): continue
    t2w = sorted(f for f in os.listdir(fol) if "t2w" in f.lower())
    if not t2w: continue
    f = chosen.get(pid, t2w[0]) if chosen else t2w[0]
    try:
        row = {"patient_id": int(pid)}; row.update({f"cnn_feat_{j}": v for j, v in enumerate(features(os.path.join(fol, f)))}); rows.append(row)
    except Exception as e: print("failed", pid, e)
    if i % 100 == 0: print(f"{i} folders processed")
df = pd.DataFrame(rows); print(df.shape, "(expected 1476 x 2049)"); df.to_csv(a.out, index=False)
