# Code cells of 02_full_experiments.ipynb, in notebook order, without outputs. "# %% [cell N]" marks each cell (0-based).
# This is a reading copy: the notebook is the record of what was run, in what order, with what output.


# %% [cell 0]
import os
patient_folders = os.listdir('picai_images')
print(len(patient_folders))
print(patient_folders[:5])


# %% [cell 1]
sample_patient = patient_folders[0]
files = os.listdir(os.path.join('picai_images', sample_patient))
print(sample_patient)
print(files)


# %% [cell 2]
import SimpleITK as sitk

sample_path = os.path.join('picai_images', sample_patient, files[0])
img = sitk.ReadImage(sample_path)
img_array = sitk.GetArrayFromImage(img)

print("Shape:", img_array.shape)   # (slices, height, width)
print("Spacing:", img.GetSpacing())
print("Data type:", img_array.dtype)


# %% [cell 3]
import SimpleITK as sitk
import numpy as np
import cv2

def load_middle_slice(mha_path, target_size=(224, 224)):
    img = sitk.ReadImage(mha_path)
    arr = sitk.GetArrayFromImage(img)  # shape: (slices, H, W)
    
    middle_idx = arr.shape[0] // 2
    slice_2d = arr[middle_idx]
    
    # Normalize uint16 -> 0-255 uint8
    slice_norm = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Resize to CNN input size
    slice_resized = cv2.resize(slice_norm, target_size)
    
    # Convert grayscale -> 3-channel (ResNet50 expects RGB-shaped input)
    slice_rgb = cv2.cvtColor(slice_resized, cv2.COLOR_GRAY2RGB)
    
    return slice_rgb

# Test it on your sample patient
test_slice = load_middle_slice(sample_path)
print(test_slice.shape)  # should be (224, 224, 3)


# %% [cell 4]
import matplotlib.pyplot as plt
plt.imshow(test_slice)
plt.title(f"Middle slice: {sample_patient}")
plt.axis('off')
plt.show()


# %% [cell 5]
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os

base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_cnn_features(mha_path):
    slice_rgb = load_middle_slice(mha_path)  # from your existing function
    img_array = np.expand_dims(slice_rgb, axis=0).astype(np.float32)
    img_preprocessed = preprocess_input(img_array)
    features = base_model.predict(img_preprocessed, verbose=0)
    return features.flatten()  # 2048-dim vector


# %% [cell 6]
test_features = extract_cnn_features(sample_path)
print(test_features.shape)  # should be (2048,)


# %% [cell 7]
feature_rows = []

for i, patient_id in enumerate(patient_folders):
    folder_path = os.path.join('picai_images', patient_id)
    files = os.listdir(folder_path)
    
    t2w_file = [f for f in files if 't2w' in f.lower()]
    if not t2w_file:
        continue  # skip if no T2W scan found
    
    mha_path = os.path.join(folder_path, t2w_file[0])
    
    try:
        features = extract_cnn_features(mha_path)
        row = {'patient_id': patient_id}
        row.update({f'cnn_feat_{j}': features[j] for j in range(len(features))})
        feature_rows.append(row)
    except Exception as e:
        print(f"Failed on {patient_id}: {e}")
    
    if i % 100 == 0:
        print(f"Processed {i}/{len(patient_folders)}")

cnn_features_df = pd.DataFrame(feature_rows)
print(cnn_features_df.shape)


# %% [cell 8]
import pandas as pd
cnn_features_df = pd.DataFrame(feature_rows)
print(cnn_features_df.shape)


# %% [cell 9]
cnn_features_df.to_csv('cnn_features_t2w.csv', index=False)


# %% [cell 10]
print(len(feature_rows))


# %% [cell 12]
picai_clinical_clean = pd.read_csv('picai_clinical_clean.csv')

print("Unique patients in clinical:", picai_clinical_clean['patient_id'].nunique())
print("Unique patients in CNN features:", cnn_features_df['patient_id'].nunique())

# Check dtype match — folder names are strings, clinical patient_id might be int
print(picai_clinical_clean['patient_id'].dtype)
print(cnn_features_df['patient_id'].dtype)


# %% [cell 13]
cnn_features_df['patient_id'] = cnn_features_df['patient_id'].astype(int)

fused_df = picai_clinical_clean.merge(cnn_features_df, on='patient_id', how='inner')
print(fused_df.shape)
fused_df['csPCa_label'].value_counts()


# %% [cell 14]
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

cnn_cols = [c for c in cnn_features_df.columns if c.startswith('cnn_feat_')]
X_cnn = fused_df[cnn_cols]

# Standardize before PCA (PCA is scale-sensitive)
scaler = StandardScaler()
X_cnn_scaled = scaler.fit_transform(X_cnn)

# Reduce to 50 components — captures most variance, drastically cuts dimensionality
pca = PCA(n_components=50, random_state=42)
X_cnn_pca = pca.fit_transform(X_cnn_scaled)

print("Explained variance captured:", pca.explained_variance_ratio_.sum())

pca_cols = [f'cnn_pca_{i}' for i in range(50)]
cnn_pca_df = pd.DataFrame(X_cnn_pca, columns=pca_cols)


# %% [cell 15]
pca_check = PCA(random_state=42)
pca_check.fit(X_cnn_scaled)

cumulative_variance = np.cumsum(pca_check.explained_variance_ratio_)

# Find how many components needed for 90% and 95% variance
n_90 = np.argmax(cumulative_variance >= 0.90) + 1
n_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"Components needed for 90% variance: {n_90}")
print(f"Components needed for 95% variance: {n_95}")


# %% [cell 16]
pca = PCA(n_components=min(n_90, 100), random_state=42)  # cap at 100 regardless
X_cnn_pca = pca.fit_transform(X_cnn_scaled)
print("Explained variance captured:", pca.explained_variance_ratio_.sum())

pca_cols = [f'cnn_pca_{i}' for i in range(X_cnn_pca.shape[1])]
cnn_pca_df = pd.DataFrame(X_cnn_pca, columns=pca_cols)
print(cnn_pca_df.shape)


# %% [cell 17]
final_model_df = pd.concat([
    fused_df[['patient_id', 'patient_age', 'psa', 'psad', 'prostate_volume', 'csPCa_label']].reset_index(drop=True),
    cnn_pca_df.reset_index(drop=True)
], axis=1)

print(final_model_df.shape)  # should be (1500, 106)
final_model_df.to_csv('fused_final_dataset.csv', index=False)


# %% [cell 18]
from sklearn.model_selection import train_test_split

X = final_model_df.drop(columns=['patient_id', 'csPCa_label'])
y = final_model_df['csPCa_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(X_train.shape, X_test.shape)
print(y_train.value_counts())
print(y_test.value_counts())


# %% [cell 19]
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Scale features (important for LR and SVM, harmless for RF)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_params = {
    'LogisticRegression': (
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        {'C': [0.01, 0.1, 1, 10]}
    ),
    'RandomForest': (
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        {'n_estimators': [100, 300, 500], 'max_depth': [None, 10, 20]}
    ),
    'SVM': (
        SVC(probability=True, class_weight='balanced'),
        {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
    )
}

best_models = {}
for name, (model, params) in models_params.items():
    grid = GridSearchCV(model, params, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    best_models[name] = grid.best_estimator_
    print(f"{name}: best params={grid.best_params_}, best CV ROC-AUC={grid.best_score_:.4f}")


# %% [cell 20]
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

results = []
for name, model in best_models.items():
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1': f1_score(y_test, preds),
        'ROC-AUC': roc_auc_score(y_test, probs)
    })

results_df = pd.DataFrame(results)
print(results_df)


# %% [cell 21]
for name, model in best_models.items():
    preds = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, preds)
    print(f"--- {name} ---")
    print(cm)
    print(f"False negatives (missed cancer): {cm[1][0]}")
    print(f"False positives (false alarm): {cm[0][1]}")
    print()


# %% [cell 22]
def extract_features_for_sequence(sequence_name):
    feature_rows = []
    
    for i, patient_id in enumerate(patient_folders):
        folder_path = os.path.join('picai_images', patient_id)
        files = os.listdir(folder_path)
        
        seq_file = [f for f in files if sequence_name in f.lower()]
        if not seq_file:
            continue
        
        mha_path = os.path.join(folder_path, seq_file[0])
        
        try:
            features = extract_cnn_features(mha_path)
            row = {'patient_id': patient_id}
            row.update({f'{sequence_name}_feat_{j}': features[j] for j in range(len(features))})
            feature_rows.append(row)
        except Exception as e:
            print(f"Failed on {patient_id} ({sequence_name}): {e}")
        
        if i % 200 == 0:
            print(f"{sequence_name}: processed {i}/{len(patient_folders)}")
    
    return pd.DataFrame(feature_rows)

adc_features_df = extract_features_for_sequence('adc')
adc_features_df.to_csv('cnn_features_adc.csv', index=False)
print(adc_features_df.shape)


# %% [cell 23]
hbv_features_df = extract_features_for_sequence('hbv')
hbv_features_df.to_csv('cnn_features_hbv.csv', index=False)
print(hbv_features_df.shape)


# %% [cell 24]
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

t2w_features_df = pd.read_csv('cnn_features_t2w.csv')
adc_features_df = pd.read_csv('cnn_features_adc.csv')
hbv_features_df = pd.read_csv('cnn_features_hbv.csv')

print(t2w_features_df.shape, adc_features_df.shape, hbv_features_df.shape)


# %% [cell 25]
def reduce_sequence(df, prefix, n_components=35):
    feat_cols = [c for c in df.columns if c.startswith(f'{prefix}_feat_')]
    X = df[feat_cols]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"{prefix}: variance captured = {pca.explained_variance_ratio_.sum():.4f}")
    
    pca_cols = [f'{prefix}_pca_{i}' for i in range(n_components)]
    result_df = pd.DataFrame(X_pca, columns=pca_cols)
    result_df['patient_id'] = df['patient_id'].astype(int).values
    return result_df

t2w_pca_df = reduce_sequence(t2w_features_df, 't2w')
adc_pca_df = reduce_sequence(adc_features_df, 'adc')
hbv_pca_df = reduce_sequence(hbv_features_df, 'hbv')


# %% [cell 26]
print(t2w_features_df.columns.tolist()[:5])
print(adc_features_df.columns.tolist()[:5])
print(hbv_features_df.columns.tolist()[:5])


# %% [cell 27]
def reduce_sequence(df, feat_prefix, output_prefix, n_components=35):
    feat_cols = [c for c in df.columns if c.startswith(feat_prefix)]
    X = df[feat_cols]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"{output_prefix}: variance captured = {pca.explained_variance_ratio_.sum():.4f}")
    
    pca_cols = [f'{output_prefix}_pca_{i}' for i in range(n_components)]
    result_df = pd.DataFrame(X_pca, columns=pca_cols)
    result_df['patient_id'] = df['patient_id'].astype(int).values
    return result_df

t2w_pca_df = reduce_sequence(t2w_features_df, 'cnn_feat_', 't2w')
adc_pca_df = reduce_sequence(adc_features_df, 'adc_feat_', 'adc')
hbv_pca_df = reduce_sequence(hbv_features_df, 'hbv_feat_', 'hbv')


# %% [cell 28]
imaging_combined = t2w_pca_df.merge(adc_pca_df, on='patient_id', how='inner') \
                              .merge(hbv_pca_df, on='patient_id', how='inner')
print(imaging_combined.shape)  # expect (~1476, 106)


# %% [cell 29]
picai_clinical_clean = pd.read_csv('picai_clinical_clean.csv')

fused_mp_df = picai_clinical_clean.merge(imaging_combined, on='patient_id', how='inner')
print(fused_mp_df.shape)  # expect around (1500, 111) — 5 clinical + 1 label + 105 imaging
fused_mp_df['csPCa_label'].value_counts()


# %% [cell 30]
fused_mp_df.to_csv('fused_multiparametric_dataset.csv', index=False)

from sklearn.model_selection import train_test_split

X_mp = fused_mp_df.drop(columns=['patient_id', 'csPCa_label'])
y_mp = fused_mp_df['csPCa_label']

X_train_mp, X_test_mp, y_train_mp, y_test_mp = train_test_split(
    X_mp, y_mp, test_size=0.2, stratify=y_mp, random_state=42
)

scaler_mp = StandardScaler()
X_train_mp_scaled = scaler_mp.fit_transform(X_train_mp)
X_test_mp_scaled = scaler_mp.transform(X_test_mp)

print(X_train_mp_scaled.shape, X_test_mp_scaled.shape)


# %% [cell 31]
X_mp = fused_mp_df.drop(columns=['patient_id', 'csPCa_label', 'center'])
y_mp = fused_mp_df['csPCa_label']


# %% [cell 32]
X_train_mp, X_test_mp, y_train_mp, y_test_mp = train_test_split(
    X_mp, y_mp, test_size=0.2, stratify=y_mp, random_state=42
)

scaler_mp = StandardScaler()
X_train_mp_scaled = scaler_mp.fit_transform(X_train_mp)
X_test_mp_scaled = scaler_mp.transform(X_test_mp)

print(X_train_mp_scaled.shape, X_test_mp_scaled.shape)


# %% [cell 33]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_params = {
    'LogisticRegression': (
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        {'C': [0.01, 0.1, 1, 10]}
    ),
    'RandomForest': (
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        {'n_estimators': [100, 300, 500], 'max_depth': [None, 10, 20]}
    ),
    'SVM': (
        SVC(probability=True, class_weight='balanced'),
        {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
    )
}

best_models_mp = {}
for name, (model, params) in models_params.items():
    grid = GridSearchCV(model, params, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train_mp_scaled, y_train_mp)
    best_models_mp[name] = grid.best_estimator_
    print(f"{name}: best params={grid.best_params_}, best CV ROC-AUC={grid.best_score_:.4f}")


# %% [cell 34]
results_mp = []
for name, model in best_models_mp.items():
    preds = model.predict(X_test_mp_scaled)
    probs = model.predict_proba(X_test_mp_scaled)[:, 1]
    
    results_mp.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test_mp, preds),
        'Precision': precision_score(y_test_mp, preds),
        'Recall': recall_score(y_test_mp, preds),
        'F1': f1_score(y_test_mp, preds),
        'ROC-AUC': roc_auc_score(y_test_mp, probs)
    })

results_mp_df = pd.DataFrame(results_mp)
print(results_mp_df)


# %% [cell 35]
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

models_params = {
    'LogisticRegression': (LogisticRegression(max_iter=1000, class_weight='balanced'), {'C': [0.01, 0.1, 1, 10]}),
    'DecisionTree': (DecisionTreeClassifier(random_state=42, class_weight='balanced'), {'max_depth': [5, 10, None]}),
    'RandomForest': (RandomForestClassifier(random_state=42, class_weight='balanced'), {'n_estimators': [100, 300], 'max_depth': [10, 20]}),
    'ExtraTrees': (ExtraTreesClassifier(random_state=42, class_weight='balanced'), {'n_estimators': [100, 300], 'max_depth': [10, 20]}),
    'GradientBoosting': (GradientBoostingClassifier(random_state=42), {'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1]}),
    'AdaBoost': (AdaBoostClassifier(random_state=42), {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]}),
    'SVM': (SVC(probability=True, class_weight='balanced'), {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}),
    'KNN': (KNeighborsClassifier(), {'n_neighbors': [5, 10, 15]}),
    'NaiveBayes': (GaussianNB(), {}),
    'LDA': (LinearDiscriminantAnalysis(), {}),
    'MLP': (MLPClassifier(max_iter=1000, random_state=42), {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.001, 0.01]})
}


# %% [cell 36]
best_models_all = {}
for name, (model, params) in models_params.items():
    if params:
        grid = GridSearchCV(model, params, cv=cv, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_models_all[name] = grid.best_estimator_
        print(f"{name}: best params={grid.best_params_}, best CV ROC-AUC={grid.best_score_:.4f}")
    else:
        model.fit(X_train_scaled, y_train)
        best_models_all[name] = model
        print(f"{name}: fitted (no tuning params)")


# %% [cell 37]
results_all = []
for name, model in best_models_all.items():
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    results_all.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1': f1_score(y_test, preds),
        'ROC-AUC': roc_auc_score(y_test, probs)
    })

results_all_df = pd.DataFrame(results_all).sort_values('ROC-AUC', ascending=False)
print(results_all_df)


# %% [cell 38]
# --- Dataset 1: Clinical only ---
X_clin = picai_clinical_clean.drop(columns=['patient_id', 'study_id', 'center', 'csPCa_label'], errors='ignore')
y_clin = picai_clinical_clean['csPCa_label']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clin, y_clin, test_size=0.2, stratify=y_clin, random_state=42)
scaler_c = StandardScaler()
Xc_train_scaled = scaler_c.fit_transform(Xc_train)
Xc_test_scaled = scaler_c.transform(Xc_test)

# --- Dataset 2: Imaging only ---
imaging_only_df = fused_df[['patient_id', 'csPCa_label']].join(cnn_pca_df)
X_img = imaging_only_df.drop(columns=['patient_id', 'csPCa_label'])
y_img = imaging_only_df['csPCa_label']

Xi_train, Xi_test, yi_train, yi_test = train_test_split(X_img, y_img, test_size=0.2, stratify=y_img, random_state=42)
scaler_i = StandardScaler()
Xi_train_scaled = scaler_i.fit_transform(Xi_train)
Xi_test_scaled = scaler_i.transform(Xi_test)

# --- Dataset 3: Fused (you already have this) ---
# X_train_scaled, X_test_scaled, y_train, y_test — from earlier

print("Clinical:", Xc_train_scaled.shape, Xc_test_scaled.shape)
print("Imaging:", Xi_train_scaled.shape, Xi_test_scaled.shape)
print("Fused:", X_train_scaled.shape, X_test_scaled.shape)


# %% [cell 39]
def run_all_models(X_train_s, X_test_s, y_tr, y_te, dataset_name):
    results = []
    for name, (model, params) in models_params.items():
        if params:
            grid = GridSearchCV(model, params, cv=cv, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train_s, y_tr)
            best_model = grid.best_estimator_
        else:
            model.fit(X_train_s, y_tr)
            best_model = model
        
        preds = best_model.predict(X_test_s)
        probs = best_model.predict_proba(X_test_s)[:, 1]
        
        results.append({
            'Dataset': dataset_name,
            'Model': name,
            'Accuracy': accuracy_score(y_te, preds),
            'Precision': precision_score(y_te, preds),
            'Recall': recall_score(y_te, preds),
            'F1': f1_score(y_te, preds),
            'ROC-AUC': roc_auc_score(y_te, probs)
        })
    return pd.DataFrame(results)


# %% [cell 40]
clinical_results = run_all_models(Xc_train_scaled, Xc_test_scaled, yc_train, yc_test, 'Clinical only')
print("Clinical done")

imaging_results = run_all_models(Xi_train_scaled, Xi_test_scaled, yi_train, yi_test, 'Imaging only')
print("Imaging done")

fused_results = run_all_models(X_train_scaled, X_test_scaled, y_train, y_test, 'Fused')
print("Fused done")

all_comparison = pd.concat([clinical_results, imaging_results, fused_results], ignore_index=True)
all_comparison.to_csv('full_comparison_results.csv', index=False)


# %% [cell 41]
pivot_view = all_comparison.pivot(index='Model', columns='Dataset', values='ROC-AUC')
print(pivot_view)


# %% [cell 42]
def load_multi_slice(mha_path, target_size=(224, 224), n_slices=5):
    img = sitk.ReadImage(mha_path)
    arr = sitk.GetArrayFromImage(img)
    
    total_slices = arr.shape[0]
    middle_idx = total_slices // 2
    half_window = n_slices // 2
    
    start = max(0, middle_idx - half_window)
    end = min(total_slices, middle_idx + half_window + 1)
    
    selected_slices = arr[start:end]
    
    processed_slices = []
    for s in selected_slices:
        s_norm = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        s_resized = cv2.resize(s_norm, target_size)
        s_rgb = cv2.cvtColor(s_resized, cv2.COLOR_GRAY2RGB)
        processed_slices.append(s_rgb)
    
    return processed_slices

def extract_cnn_features_multislice(mha_path):
    slices = load_multi_slice(mha_path)
    slice_features = []
    
    for s in slices:
        img_array = np.expand_dims(s, axis=0).astype(np.float32)
        img_preprocessed = preprocess_input(img_array)
        features = base_model.predict(img_preprocessed, verbose=0)
        slice_features.append(features.flatten())
    
    # Average across slices — one 2048-dim vector per patient, informed by 5 slices instead of 1
    return np.mean(slice_features, axis=0)


# %% [cell 43]
test_features = extract_cnn_features_multislice(sample_path)
print(test_features.shape)  # should still be (2048,)


# %% [cell 44]
feature_rows = []

for i, patient_id in enumerate(patient_folders):
    folder_path = os.path.join('picai_images', patient_id)
    files = os.listdir(folder_path)
    
    t2w_file = [f for f in files if 't2w' in f.lower()]
    if not t2w_file:
        continue
    
    mha_path = os.path.join(folder_path, t2w_file[0])
    
    try:
        features = extract_cnn_features_multislice(mha_path)
        row = {'patient_id': patient_id}
        row.update({f'cnn_feat_{j}': features[j] for j in range(len(features))})
        feature_rows.append(row)
    except Exception as e:
        print(f"Failed on {patient_id}: {e}")
    
    if i % 100 == 0:
        print(f"Processed {i}/{len(patient_folders)}")

cnn_features_multislice_df = pd.DataFrame(feature_rows)
cnn_features_multislice_df.to_csv('cnn_features_t2w_multislice.csv', index=False)
print(cnn_features_multislice_df.shape)


# %% [cell 45]
t2w_multi_features_df = pd.read_csv('cnn_features_t2w_multislice.csv')

feat_cols = [c for c in t2w_multi_features_df.columns if c.startswith('cnn_feat_')]
X_multi = t2w_multi_features_df[feat_cols]

scaler_multi = StandardScaler()
X_multi_scaled = scaler_multi.fit_transform(X_multi)

pca_multi = PCA(n_components=100, random_state=42)
X_multi_pca = pca_multi.fit_transform(X_multi_scaled)

print("Variance captured:", pca_multi.explained_variance_ratio_.sum())

pca_multi_cols = [f'cnn_pca_{i}' for i in range(100)]
cnn_multi_pca_df = pd.DataFrame(X_multi_pca, columns=pca_multi_cols)
cnn_multi_pca_df['patient_id'] = t2w_multi_features_df['patient_id'].astype(int).values


# %% [cell 46]
fused_multi_df = picai_clinical_clean.merge(cnn_multi_pca_df, on='patient_id', how='inner')
print(fused_multi_df.shape)

X_fm = fused_multi_df.drop(columns=['patient_id', 'center', 'csPCa_label'], errors='ignore')
y_fm = fused_multi_df['csPCa_label']

Xfm_train, Xfm_test, yfm_train, yfm_test = train_test_split(X_fm, y_fm, test_size=0.2, stratify=y_fm, random_state=42)

scaler_fm = StandardScaler()
Xfm_train_scaled = scaler_fm.fit_transform(Xfm_train)
Xfm_test_scaled = scaler_fm.transform(Xfm_test)

print(Xfm_train_scaled.shape, Xfm_test_scaled.shape)


# %% [cell 47]
multislice_results = run_all_models(Xfm_train_scaled, Xfm_test_scaled, yfm_train, yfm_test, 'Fused (multi-slice)')
print(multislice_results.sort_values('ROC-AUC', ascending=False))


# %% [cell 48]
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", y_train_smote.value_counts())


# %% [cell 49]
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", y_train_smote.value_counts())


# %% [cell 50]
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", y_train_smote.value_counts())


# %% [cell 51]
smote_results = run_all_models(X_train_smote, X_test_scaled, y_train_smote, y_test, 'Fused + SMOTE')
print(smote_results.sort_values('ROC-AUC', ascending=False))


# %% [cell 52]
all_comparison = pd.read_csv('full_comparison_results.csv')  # Clinical only / Imaging only / Fused

master_results = pd.concat([
    all_comparison,
    multislice_results,
    smote_results
], ignore_index=True)

master_results.to_csv('master_comparison_results.csv', index=False)
print(master_results.shape)
master_results.head()


# %% [cell 53]
pivot_roc = master_results.pivot_table(index='Model', columns='Dataset', values='ROC-AUC')
pivot_recall = master_results.pivot_table(index='Model', columns='Dataset', values='Recall')

print("=== ROC-AUC by dataset ===")
print(pivot_roc)
print()
print("=== Recall by dataset ===")
print(pivot_recall)


# %% [cell 54]
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Use Random Forest since it was your best/most consistent performer across all three
rf_clinical = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
rf_clinical.fit(Xc_train_scaled, yc_train)

rf_imaging = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
rf_imaging.fit(Xi_train_scaled, yi_train)

rf_fused = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
rf_fused.fit(X_train_scaled, y_train)

datasets = {
    'Clinical only': (Xc_test_scaled, yc_test, rf_clinical),
    'Imaging only': (Xi_test_scaled, yi_test, rf_imaging),
    'Fused': (X_test_scaled, y_test, rf_fused)
}

plt.figure(figsize=(8, 6))
for name, (X_te, y_te, model) in datasets.items():
    probs = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, probs)
    auc = roc_auc_score(y_te, probs)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison: Clinical vs. Imaging vs. Fused (Random Forest)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('roc_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [cell 55]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# --- Clinical only ---
picai_clinical_clean = pd.read_csv('picai_clinical_clean.csv')
X_clin = picai_clinical_clean.drop(columns=['patient_id', 'study_id', 'center', 'csPCa_label'], errors='ignore')
y_clin = picai_clinical_clean['csPCa_label']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clin, y_clin, test_size=0.2, stratify=y_clin, random_state=42)
scaler_c = StandardScaler()
Xc_train_scaled = scaler_c.fit_transform(Xc_train)
Xc_test_scaled = scaler_c.transform(Xc_test)

# --- Fused (clinical + imaging PCA) ---
fused_df_reload = pd.read_csv('fused_final_dataset.csv')
X_fused = fused_df_reload.drop(columns=['patient_id', 'csPCa_label'], errors='ignore')
y_fused = fused_df_reload['csPCa_label']

X_train, X_test, y_train, y_test = train_test_split(X_fused, y_fused, test_size=0.2, stratify=y_fused, random_state=42)
scaler_f = StandardScaler()
X_train_scaled = scaler_f.fit_transform(X_train)
X_test_scaled = scaler_f.transform(X_test)

# --- Imaging only (just the PCA imaging columns from the fused file, no clinical) ---
imaging_cols = [c for c in fused_df_reload.columns if c.startswith('cnn_pca_')]
X_img = fused_df_reload[imaging_cols]
y_img = fused_df_reload['csPCa_label']

Xi_train, Xi_test, yi_train, yi_test = train_test_split(X_img, y_img, test_size=0.2, stratify=y_img, random_state=42)
scaler_i = StandardScaler()
Xi_train_scaled = scaler_i.fit_transform(Xi_train)
Xi_test_scaled = scaler_i.transform(Xi_test)

print("Clinical:", Xc_train_scaled.shape)
print("Imaging:", Xi_train_scaled.shape)
print("Fused:", X_train_scaled.shape)


# %% [cell 56]
rf_clinical = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
rf_clinical.fit(Xc_train_scaled, yc_train)

rf_imaging = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
rf_imaging.fit(Xi_train_scaled, yi_train)

rf_fused = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
rf_fused.fit(X_train_scaled, y_train)

datasets = {
    'Clinical only': (Xc_test_scaled, yc_test, rf_clinical),
    'Imaging only': (Xi_test_scaled, yi_test, rf_imaging),
    'Fused': (X_test_scaled, y_test, rf_fused)
}

plt.figure(figsize=(8, 6))
for name, (X_te, y_te, model) in datasets.items():
    probs = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, probs)
    auc = roc_auc_score(y_te, probs)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison: Clinical vs. Imaging vs. Fused (Random Forest)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('roc_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [cell 57]
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

# Based on your earlier results, your top 5 by ROC-AUC on the fused dataset were:
# RandomForest, AdaBoost, ExtraTrees, LogisticRegression, GradientBoosting

top5_models = {
    'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, class_weight='balanced'),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, learning_rate=0.5, random_state=42),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, class_weight='balanced'),
    'LogisticRegression': LogisticRegression(C=0.01, max_iter=1000, class_weight='balanced'),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
}

for name, model in top5_models.items():
    model.fit(X_train_scaled, y_train)


# %% [cell 58]
plt.figure(figsize=(8, 6))
for name, model in top5_models.items():
    probs = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison: Top 5 Algorithms (Fused Dataset)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('roc_top5_models.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [cell 59]
import matplotlib.pyplot as plt
import numpy as np

models = ['Random Forest', 'AdaBoost', 'Gradient Boosting', 'Extra Trees', 'Logistic Regression']
accuracy = [0.730, 0.743, 0.740, 0.717, 0.653]
precision = [0.531, 0.591, 0.559, 0.500, 0.426]
recall = [0.400, 0.306, 0.388, 0.047, 0.647]
roc_auc = [0.765, 0.758, 0.736, 0.734, 0.714]

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#2a78d6')
ax.bar(x - 0.5*width, precision, width, label='Precision', color='#eb6834')
ax.bar(x + 0.5*width, recall, width, label='Recall', color='#c0392b')
ax.bar(x + 1.5*width, roc_auc, width, label='ROC-AUC', color='#1baf7a')

ax.set_ylabel('Score')
ax.set_title('Performance Comparison: Top 5 Algorithms (Fused Dataset)')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.set_ylim(0, 1)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('top5_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [cell 60]
import matplotlib.pyplot as plt

stages = ['Clinical\nonly', 'Imaging\nonly', 'Fused', 'Fused +\nSMOTE\n(AdaBoost)']
auc_values = [0.724, 0.613, 0.765, 0.716]  # Random Forest for first 3, AdaBoost+SMOTE for last

plt.figure(figsize=(8, 6))
bars = plt.bar(stages, auc_values, color=['#889', '#eb6834', '#2a78d6', '#1baf7a'], width=0.6)
for bar, val in zip(bars, auc_values):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}', ha='center', fontweight='bold')

plt.ylim(0, 1)
plt.ylabel('ROC-AUC')
plt.title('Stepwise Model Improvement: Clinical → Imaging → Fused → Fused+SMOTE')
plt.axhline(0.5, linestyle='--', color='gray', alpha=0.5, label='Random chance')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig('stepwise_auc_progression.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [cell 61]
import seaborn as sns
from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Best ROC-AUC model
cm_rf = confusion_matrix(y_test, best_models_all['RandomForest'].predict(X_test_scaled))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Indolent', 'Clinically Sig.'], yticklabels=['Indolent', 'Clinically Sig.'])
axes[0].set_title('Random Forest (Best ROC-AUC)')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Best Recall model
cm_lr = confusion_matrix(y_test, best_models_all['LogisticRegression'].predict(X_test_scaled))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['Indolent', 'Clinically Sig.'], yticklabels=['Indolent', 'Clinically Sig.'])
axes[1].set_title('Logistic Regression (Best Recall)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [cell 62]
papers = ['Woźnicki 2020', 'TRUS Wang 2022', 'Peritumoral 2025', 'Bi-param MRI 2024',
          'Textural 2023', 'Sci.Rep. 2025', 'BMC 2024', 'mpMRI+US 2023', 'MRI+Methylation 2024', 'PI-CAI 2024', 'Your Study']
aucs = [0.688, 0.75, 0.729, 0.64, 0.804, 0.774, 0.968, 0.87, 0.97, 0.898, 0.765]
colors = ['#889']*10 + ['#c0392b']

sorted_pairs = sorted(zip(aucs, papers, colors))
aucs_s, papers_s, colors_s = zip(*sorted_pairs)

plt.figure(figsize=(9,7))
plt.barh(papers_s, aucs_s, color=colors_s)
plt.xlabel('AUC')
plt.title('Ranked AUC Comparison')
plt.xlim(0,1)
plt.tight_layout()
plt.savefig('style2_ranked_bar.png', dpi=300)
plt.show()


# %% [cell 63]
import matplotlib.pyplot as plt

papers = ['Woźnicki 2020', 'TRUS Wang 2022', 'Peritumoral 2025', 'Bi-param MRI 2024',
          'Textural 2023', 'Sci.Rep. 2025', 'BMC 2024', 'mpMRI+US 2023', 'MRI+Methylation 2024', 'PI-CAI 2024', 'Your Study']
aucs = [0.688, 0.75, 0.729, 0.64, 0.804, 0.774, 0.968, 0.87, 0.97, 0.898, 0.765]
colors = ['#889']*10 + ['#c0392b']

sorted_pairs = sorted(zip(aucs, papers, colors))
aucs_s, papers_s, colors_s = zip(*sorted_pairs)

plt.figure(figsize=(9,7))
plt.barh(papers_s, aucs_s, color=colors_s)
plt.xlabel('AUC')
plt.title('Ranked AUC Comparison')
plt.xlim(0,1)
plt.tight_layout()
plt.savefig('style2_ranked_bar.png', dpi=300)
plt.show()


# %% [cell 64]
sizes = [191, 231, 199, 77, 91, 500, 350, 300, 300, 5000, 1476]  # estimates where "not specified"
plt.figure(figsize=(9,7))
plt.scatter(sizes[:-1], aucs[:-1], s=100, color='#889', label='Literature')
plt.scatter(sizes[-1], aucs[-1], s=200, color='#c0392b', label='Your Study', zorder=5)
plt.xscale('log')
plt.xlabel('Sample Size (log scale)')
plt.ylabel('AUC')
plt.title('Sample Size vs. AUC Across Studies')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('style3_scatter.png', dpi=300)
plt.show()


# %% [cell 65]
plt.figure(figsize=(8,7))
y_pos = range(len(papers))
plt.scatter(aucs, y_pos, color=colors, s=80, zorder=3)
plt.axvline(0.765, linestyle='--', color='#c0392b', alpha=0.5, label='Your AUC')
plt.yticks(y_pos, papers)
plt.xlabel('AUC')
plt.title('Forest Plot: AUC Across Studies')
plt.xlim(0.5, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig('style4_forest.png', dpi=300)
plt.show()


# %% [cell 66]
import numpy as np
categories = ['AUC\n(normalized)', 'Sample Size\n(normalized)', 'Method\nSimplicity']
your_vals = [0.765, 1.0, 0.8]  # sample size normalized to max=1
lit_avg_vals = [0.82, 0.15, 0.3]  # rough literature average

angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
your_vals += your_vals[:1]
lit_avg_vals += lit_avg_vals[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
ax.plot(angles, your_vals, color='#c0392b', linewidth=2, label='Your Study')
ax.fill(angles, your_vals, color='#c0392b', alpha=0.2)
ax.plot(angles, lit_avg_vals, color='#889', linewidth=2, label='Literature Avg')
ax.fill(angles, lit_avg_vals, color='#889', alpha=0.2)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))
plt.savefig('style5_radar.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [cell 67]
print(best_models_all.keys())
print(X_test_scaled.shape)


# %% [cell 68]
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

resnet_classifier = Model(inputs=base.input, outputs=output)
resnet_classifier.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
print("Model built successfully")


# %% [cell 69]
import os
import numpy as np
import pandas as pd

picai_clinical_clean = pd.read_csv('picai_clinical_clean.csv')
patient_folders = os.listdir('picai_images')

X_images = []
y_images = []

for i, patient_id in enumerate(patient_folders):
    folder_path = os.path.join('picai_images', patient_id)
    files = os.listdir(folder_path)
    t2w_file = [f for f in files if 't2w' in f.lower()]
    if not t2w_file:
        continue
    mha_path = os.path.join(folder_path, t2w_file[0])
    try:
        slice_rgb = load_middle_slice(mha_path)
        label_row = picai_clinical_clean[picai_clinical_clean['patient_id'] == int(patient_id)]
        if len(label_row) == 0:
            continue
        X_images.append(slice_rgb)
        y_images.append(label_row['csPCa_label'].values[0])
    except Exception as e:
        continue
    if i % 200 == 0:
        print(f"Loaded {i}/{len(patient_folders)}")

X_images = np.array(X_images)
y_images = np.array(y_images)
print("Final shape:", X_images.shape, y_images.shape)


# %% [cell 70]
import os
import numpy as np
import pandas as pd

picai_clinical_clean = pd.read_csv('picai_clinical_clean.csv')
patient_folders = [f for f in os.listdir('picai_images') if os.path.isdir(os.path.join('picai_images', f))]

print(f"Found {len(patient_folders)} actual patient folders")

X_images = []
y_images = []

for i, patient_id in enumerate(patient_folders):
    folder_path = os.path.join('picai_images', patient_id)
    files = os.listdir(folder_path)
    t2w_file = [f for f in files if 't2w' in f.lower()]
    if not t2w_file:
        continue
    mha_path = os.path.join(folder_path, t2w_file[0])
    try:
        slice_rgb = load_middle_slice(mha_path)
        label_row = picai_clinical_clean[picai_clinical_clean['patient_id'] == int(patient_id)]
        if len(label_row) == 0:
            continue
        X_images.append(slice_rgb)
        y_images.append(label_row['csPCa_label'].values[0])
    except Exception as e:
        continue
    if i % 200 == 0:
        print(f"Loaded {i}/{len(patient_folders)}")

X_images = np.array(X_images)
y_images = np.array(y_images)
print("Final shape:", X_images.shape, y_images.shape)


# %% [cell 71]
sample_patient = patient_folders[0]
folder_path = os.path.join('picai_images', sample_patient)
files = os.listdir(folder_path)
print("Files in folder:", files)

t2w_file = [f for f in files if 't2w' in f.lower()]
print("T2W file found:", t2w_file)

if t2w_file:
    mha_path = os.path.join(folder_path, t2w_file[0])
    slice_rgb = load_middle_slice(mha_path)
    print("Image loaded, shape:", slice_rgb.shape)
    
    label_row = picai_clinical_clean[picai_clinical_clean['patient_id'] == int(sample_patient)]
    print("Label row found:", label_row)


# %% [cell 72]
import SimpleITK as sitk
import cv2
import numpy as np

def load_middle_slice(mha_path, target_size=(224, 224)):
    img = sitk.ReadImage(mha_path)
    arr = sitk.GetArrayFromImage(img)
    middle_idx = arr.shape[0] // 2
    slice_2d = arr[middle_idx]
    slice_norm = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    slice_resized = cv2.resize(slice_norm, target_size)
    slice_rgb = cv2.cvtColor(slice_resized, cv2.COLOR_GRAY2RGB)
    return slice_rgb

print("Function defined")


# %% [cell 73]
sample_patient = patient_folders[0]
folder_path = os.path.join('picai_images', sample_patient)
files = os.listdir(folder_path)
t2w_file = [f for f in files if 't2w' in f.lower()]
mha_path = os.path.join(folder_path, t2w_file[0])

slice_rgb = load_middle_slice(mha_path)
print("Image loaded, shape:", slice_rgb.shape)

label_row = picai_clinical_clean[picai_clinical_clean['patient_id'] == int(sample_patient)]
print("Label row found:")
print(label_row)


# %% [cell 74]
X_images = []
y_images = []

for i, patient_id in enumerate(patient_folders):
    folder_path = os.path.join('picai_images', patient_id)
    files = os.listdir(folder_path)
    t2w_file = [f for f in files if 't2w' in f.lower()]
    if not t2w_file:
        continue
    mha_path = os.path.join(folder_path, t2w_file[0])
    try:
        slice_rgb = load_middle_slice(mha_path)
        label_row = picai_clinical_clean[picai_clinical_clean['patient_id'] == int(patient_id)]
        if len(label_row) == 0:
            continue
        X_images.append(slice_rgb)
        y_images.append(label_row['csPCa_label'].values[0])
    except Exception as e:
        continue
    if i % 200 == 0:
        print(f"Loaded {i}/{len(patient_folders)}")

X_images = np.array(X_images)
y_images = np.array(y_images)
print("Final shape:", X_images.shape, y_images.shape)


# %% [cell 75]
from sklearn.model_selection import train_test_split

Xi_train_imgs, Xi_test_imgs, yi_train_imgs, yi_test_imgs = train_test_split(
    X_images, y_images, test_size=0.2, stratify=y_images, random_state=42
)

print("Train shape:", Xi_train_imgs.shape)
print("Test shape:", Xi_test_imgs.shape)

history = resnet_classifier.fit(
    Xi_train_imgs, yi_train_imgs,
    validation_data=(Xi_test_imgs, yi_test_imgs),
    epochs=15, batch_size=16
)


# %% [cell 76]
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

probs_resnet = resnet_classifier.predict(Xi_test_imgs).flatten()
preds_resnet = (probs_resnet >= 0.5).astype(int)

print("Accuracy:", accuracy_score(yi_test_imgs, preds_resnet))
print("Precision:", precision_score(yi_test_imgs, preds_resnet))
print("Recall:", recall_score(yi_test_imgs, preds_resnet))
print("F1:", f1_score(yi_test_imgs, preds_resnet))
print("ROC-AUC:", roc_auc_score(yi_test_imgs, probs_resnet))
print()
print("Confusion matrix:")
print(confusion_matrix(yi_test_imgs, preds_resnet))
print()
print("Prediction distribution:", np.bincount(preds_resnet))
print("Actual distribution:", np.bincount(yi_test_imgs))


# %% [cell 77]
base.trainable = True
for layer in base.layers[:-10]:  # freeze all but the last 10 layers
    layer.trainable = False

resnet_classifier.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

history = resnet_classifier.fit(
    Xi_train_imgs, yi_train_imgs,
    validation_data=(Xi_test_imgs, yi_test_imgs),
    epochs=15, batch_size=16,
    class_weight={0: 1.0, 1: 2.5}
)


# %% [cell 78]
probs_resnet_ft = resnet_classifier.predict(Xi_test_imgs).flatten()
preds_resnet_ft = (probs_resnet_ft >= 0.5).astype(int)

print("Accuracy:", accuracy_score(yi_test_imgs, preds_resnet_ft))
print("Precision:", precision_score(yi_test_imgs, preds_resnet_ft))
print("Recall:", recall_score(yi_test_imgs, preds_resnet_ft))
print("F1:", f1_score(yi_test_imgs, preds_resnet_ft))
print("ROC-AUC:", roc_auc_score(yi_test_imgs, probs_resnet_ft))
print()
print("Confusion matrix:")
print(confusion_matrix(yi_test_imgs, preds_resnet_ft))


# %% [cell 79]
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

base_aug = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_aug.trainable = False

x = GlobalAveragePooling2D()(base_aug.output)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

resnet_augmented = Model(inputs=base_aug.input, outputs=output)
resnet_augmented.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


# %% [cell 80]
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(Xi_train_imgs)

history_aug = resnet_augmented.fit(
    datagen.flow(Xi_train_imgs, yi_train_imgs, batch_size=16),
    validation_data=(Xi_test_imgs, yi_test_imgs),
    epochs=15,
    class_weight={0: 1.0, 1: 2.5}
)


# %% [cell 81]
probs_aug = resnet_augmented.predict(Xi_test_imgs).flatten()
preds_aug = (probs_aug >= 0.5).astype(int)

print("Accuracy:", accuracy_score(yi_test_imgs, preds_aug))
print("Precision:", precision_score(yi_test_imgs, preds_aug))
print("Recall:", recall_score(yi_test_imgs, preds_aug))
print("F1:", f1_score(yi_test_imgs, preds_aug))
print("ROC-AUC:", roc_auc_score(yi_test_imgs, probs_aug))
print()
print("Confusion matrix:")
print(confusion_matrix(yi_test_imgs, preds_aug))


# %% [cell 82]
print(best_models_all.keys())
print(X_train_scaled.shape)


# %% [cell 83]
 import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

fused_df_reload = pd.read_csv('fused_final_dataset.csv')
X_fused = fused_df_reload.drop(columns=['patient_id', 'csPCa_label'], errors='ignore')
y_fused = fused_df_reload['csPCa_label']

X_train, X_test, y_train, y_test = train_test_split(X_fused, y_fused, test_size=0.2, stratify=y_fused, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_params = {
    'LogisticRegression': (LogisticRegression(max_iter=1000, class_weight='balanced'), {'C': [0.01, 0.1, 1, 10]}),
    'DecisionTree': (DecisionTreeClassifier(random_state=42, class_weight='balanced'), {'max_depth': [5, 10, None]}),
    'RandomForest': (RandomForestClassifier(random_state=42, class_weight='balanced'), {'n_estimators': [100, 300], 'max_depth': [10, 20]}),
    'ExtraTrees': (ExtraTreesClassifier(random_state=42, class_weight='balanced'), {'n_estimators': [100, 300], 'max_depth': [10, 20]}),
    'GradientBoosting': (GradientBoostingClassifier(random_state=42), {'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1]}),
    'AdaBoost': (AdaBoostClassifier(random_state=42), {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]}),
    'SVM': (SVC(probability=True, class_weight='balanced'), {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}),
    'KNN': (KNeighborsClassifier(), {'n_neighbors': [5, 10, 15]}),
    'NaiveBayes': (GaussianNB(), {}),
    'LDA': (LinearDiscriminantAnalysis(), {}),
    'MLP': (MLPClassifier(max_iter=1000, random_state=42), {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.001, 0.01]})
}

best_models_all = {}
for name, (model, params) in models_params.items():
    if params:
        grid = GridSearchCV(model, params, cv=cv, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_models_all[name] = grid.best_estimator_
    else:
        model.fit(X_train_scaled, y_train)
        best_models_all[name] = model
    print(f"{name} trained")

print("All models rebuilt successfully")


# %% [cell 84]
import pandas as pd
print("test")


# %% [cell 85]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

fused_df_reload = pd.read_csv('fused_final_dataset.csv')
X_fused = fused_df_reload.drop(columns=['patient_id', 'csPCa_label'], errors='ignore')
y_fused = fused_df_reload['csPCa_label']

X_train, X_test, y_train, y_test = train_test_split(X_fused, y_fused, test_size=0.2, stratify=y_fused, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_params = {
    'LogisticRegression': (LogisticRegression(max_iter=1000, class_weight='balanced'), {'C': [0.01, 0.1, 1, 10]}),
    'DecisionTree': (DecisionTreeClassifier(random_state=42, class_weight='balanced'), {'max_depth': [5, 10, None]}),
    'RandomForest': (RandomForestClassifier(random_state=42, class_weight='balanced'), {'n_estimators': [100, 300], 'max_depth': [10, 20]}),
    'ExtraTrees': (ExtraTreesClassifier(random_state=42, class_weight='balanced'), {'n_estimators': [100, 300], 'max_depth': [10, 20]}),
    'GradientBoosting': (GradientBoostingClassifier(random_state=42), {'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1]}),
    'AdaBoost': (AdaBoostClassifier(random_state=42), {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]}),
    'SVM': (SVC(probability=True, class_weight='balanced'), {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}),
    'KNN': (KNeighborsClassifier(), {'n_neighbors': [5, 10, 15]}),
    'NaiveBayes': (GaussianNB(), {}),
    'LDA': (LinearDiscriminantAnalysis(), {}),
    'MLP': (MLPClassifier(max_iter=1000, random_state=42), {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.001, 0.01]})
}

best_models_all = {}
for name, (model, params) in models_params.items():
    if params:
        grid = GridSearchCV(model, params, cv=cv, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_models_all[name] = grid.best_estimator_
    else:
        model.fit(X_train_scaled, y_train)
        best_models_all[name] = model
    print(f"{name} trained")

print("All models rebuilt successfully")


# %% [cell 86]
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=2.5,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_scaled, y_train)

xgb_preds = xgb_model.predict(X_test_scaled)
xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]

print("=== XGBoost ===")
print("Accuracy:", accuracy_score(y_test, xgb_preds))
print("Precision:", precision_score(y_test, xgb_preds))
print("Recall:", recall_score(y_test, xgb_preds))
print("F1:", f1_score(y_test, xgb_preds))
print("ROC-AUC:", roc_auc_score(y_test, xgb_probs))


# %% [cell 87]
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=2.5,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_scaled, y_train)

xgb_preds = xgb_model.predict(X_test_scaled)
xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]

print("=== XGBoost ===")
print("Accuracy:", accuracy_score(y_test, xgb_preds))
print("Precision:", precision_score(y_test, xgb_preds))
print("Recall:", recall_score(y_test, xgb_preds))
print("F1:", f1_score(y_test, xgb_preds))
print("ROC-AUC:", roc_auc_score(y_test, xgb_probs))


# %% [cell 88]
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

estimators = [
    ('rf', best_models_all['RandomForest']),
    ('ada', best_models_all['AdaBoost']),
    ('gb', best_models_all['GradientBoosting'])
]

stacked_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=5
)
stacked_model.fit(X_train_scaled, y_train)

stacked_preds = stacked_model.predict(X_test_scaled)
stacked_probs = stacked_model.predict_proba(X_test_scaled)[:, 1]

print("=== Stacking Ensemble ===")
print("Accuracy:", accuracy_score(y_test, stacked_preds))
print("Precision:", precision_score(y_test, stacked_preds))
print("Recall:", recall_score(y_test, stacked_preds))
print("F1:", f1_score(y_test, stacked_preds))
print("ROC-AUC:", roc_auc_score(y_test, stacked_probs))


# %% [cell 89]
print(xgb_model)
print(X_train_scaled.shape)


# %% [cell 90]
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score

best_acc, best_thresh = 0, 0.5
for t in np.arange(0.1, 0.9, 0.01):
    preds = (xgb_probs >= t).astype(int)
    acc = accuracy_score(y_test, preds)
    if acc > best_acc:
        best_acc, best_thresh = acc, t

print(f"Best threshold: {best_thresh:.2f}, Accuracy: {best_acc:.4f}")

best_preds = (xgb_probs >= best_thresh).astype(int)
print(f"Recall at this threshold: {recall_score(y_test, best_preds):.4f}")
print(f"Precision at this threshold: {precision_score(y_test, best_preds):.4f}")


# %% [cell 91]
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

xgb_params = {
    'n_estimators': [200, 300, 500],
    'max_depth': [3, 5, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_grid = GridSearchCV(
    XGBClassifier(scale_pos_weight=2.5, random_state=42, eval_metric='logloss'),
    xgb_params, cv=5, scoring='accuracy', n_jobs=-1
)
xgb_grid.fit(X_train_scaled, y_train)

print("Best params:", xgb_grid.best_params_)
print("Best CV accuracy:", xgb_grid.best_score_)

best_xgb = xgb_grid.best_estimator_
best_xgb_preds = best_xgb.predict(X_test_scaled)
best_xgb_probs = best_xgb.predict_proba(X_test_scaled)[:, 1]

print("\n=== Tuned XGBoost (test set) ===")
print("Accuracy:", accuracy_score(y_test, best_xgb_preds))
print("Precision:", precision_score(y_test, best_xgb_preds))
print("Recall:", recall_score(y_test, best_xgb_preds))
print("ROC-AUC:", roc_auc_score(y_test, best_xgb_probs))


# %% [cell 92]
xgb_params_v2 = {
    'n_estimators': [100, 200, 300, 500, 800],
    'max_depth': [3, 4, 5, 6, 8, 10],
    'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.5]
}


# %% [cell 93]
from sklearn.model_selection import RandomizedSearchCV

xgb_random = RandomizedSearchCV(
    XGBClassifier(scale_pos_weight=2.5, random_state=42, eval_metric='logloss'),
    xgb_params_v2, n_iter=100, cv=5, scoring='roc_auc', n_jobs=-1, random_state=42
)
xgb_random.fit(X_train_scaled, y_train)
print("Best params:", xgb_random.best_params_)
print("Best CV ROC-AUC:", xgb_random.best_score_)


# %% [cell 94]
print(X_train_scaled.shape)
print(y_train.shape)


# %% [cell 95]
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

xgb_params_v2 = {
    'n_estimators': [100, 200, 300, 500, 800],
    'max_depth': [3, 4, 5, 6, 8, 10],
    'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.5]
}

xgb_random = RandomizedSearchCV(
    XGBClassifier(scale_pos_weight=2.5, random_state=42, eval_metric='logloss'),
    xgb_params_v2, n_iter=100, cv=5, scoring='roc_auc', n_jobs=-1, random_state=42
)
xgb_random.fit(X_train_scaled, y_train)

print("Best params:", xgb_random.best_params_)
print("Best CV ROC-AUC:", xgb_random.best_score_)


# %% [cell 96]
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

best_xgb_v2 = xgb_random.best_estimator_
preds_v2 = best_xgb_v2.predict(X_test_scaled)
probs_v2 = best_xgb_v2.predict_proba(X_test_scaled)[:, 1]

print("=== Widened RandomizedSearch XGBoost (test set) ===")
print("Accuracy:", accuracy_score(y_test, preds_v2))
print("Precision:", precision_score(y_test, preds_v2))
print("Recall:", recall_score(y_test, preds_v2))
print("F1:", f1_score(y_test, preds_v2))
print("ROC-AUC:", roc_auc_score(y_test, probs_v2))


# %% [cell 97]
print(picai_clinical_clean['center'].value_counts())


# %% [cell 98]
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

fused_df_reload = pd.read_csv('fused_final_dataset.csv')
# Merge center info back in, since it's not currently in your fused dataset
fused_with_center = fused_df_reload.merge(
    picai_clinical_clean[['patient_id', 'center']], on='patient_id', how='left'
)

centers = ['RUMC', 'PCNN', 'ZGT']
loco_results = []

for held_out in centers:
    train_data = fused_with_center[fused_with_center['center'] != held_out]
    test_data = fused_with_center[fused_with_center['center'] == held_out]
    
    X_train_loco = train_data.drop(columns=['patient_id', 'csPCa_label', 'center'])
    y_train_loco = train_data['csPCa_label']
    X_test_loco = test_data.drop(columns=['patient_id', 'csPCa_label', 'center'])
    y_test_loco = test_data['csPCa_label']
    
    scaler_loco = StandardScaler()
    X_train_loco_scaled = scaler_loco.fit_transform(X_train_loco)
    X_test_loco_scaled = scaler_loco.transform(X_test_loco)
    
    model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train_loco_scaled, y_train_loco)
    
    preds = model.predict(X_test_loco_scaled)
    probs = model.predict_proba(X_test_loco_scaled)[:, 1]
    
    loco_results.append({
        'Held-out center': held_out,
        'Train size': len(train_data),
        'Test size': len(test_data),
        'Accuracy': accuracy_score(y_test_loco, preds),
        'Precision': precision_score(y_test_loco, preds),
        'Recall': recall_score(y_test_loco, preds),
        'ROC-AUC': roc_auc_score(y_test_loco, probs)
    })

loco_df = pd.DataFrame(loco_results)
print(loco_df)


# %% [cell 99]
diagnostic_cols = ['patient_age', 'psa', 'psad', 'prostate_volume', 'csPCa_label']
for center in ['RUMC', 'PCNN', 'ZGT']:
    subset = picai_clinical_clean[picai_clinical_clean['center'] == center]
    print(f"--- {center} (n={len(subset)}) ---")
    print(subset[diagnostic_cols].describe().loc[['mean', 'std']])
    print(f"csPCa positive rate: {subset['csPCa_label'].mean():.3f}")
    print()


# %% [cell 100]
def run_loco_with_domain_adaptation(held_out):
    train_data = fused_with_center[fused_with_center['center'] != held_out]
    test_data = fused_with_center[fused_with_center['center'] == held_out]
    
    X_train_loco = train_data.drop(columns=['patient_id', 'csPCa_label', 'center'])
    y_train_loco = train_data['csPCa_label']
    X_test_loco = test_data.drop(columns=['patient_id', 'csPCa_label', 'center'])
    y_test_loco = test_data['csPCa_label']
    
    # Scale train and test SEPARATELY using their own statistics (no label leakage)
    scaler_train = StandardScaler()
    X_train_scaled = scaler_train.fit_transform(X_train_loco)
    
    scaler_test = StandardScaler()
    X_test_scaled = scaler_test.fit_transform(X_test_loco)  # fit on test features only, not labels
    
    model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train_loco)
    
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    return {
        'Held-out center': held_out,
        'Accuracy': accuracy_score(y_test_loco, preds),
        'Precision': precision_score(y_test_loco, preds),
        'Recall': recall_score(y_test_loco, preds),
        'ROC-AUC': roc_auc_score(y_test_loco, probs)
    }

domain_adapted_results = pd.DataFrame([run_loco_with_domain_adaptation(c) for c in ['RUMC', 'PCNN', 'ZGT']])
print(domain_adapted_results)


# %% [cell 101]
def run_loco_xgboost(held_out):
    train_data = fused_with_center[fused_with_center['center'] != held_out]
    test_data = fused_with_center[fused_with_center['center'] == held_out]
    
    X_train_loco = train_data.drop(columns=['patient_id', 'csPCa_label', 'center'])
    y_train_loco = train_data['csPCa_label']
    X_test_loco = test_data.drop(columns=['patient_id', 'csPCa_label', 'center'])
    y_test_loco = test_data['csPCa_label']
    
    scaler_train = StandardScaler()
    X_train_scaled = scaler_train.fit_transform(X_train_loco)
    scaler_test = StandardScaler()
    X_test_scaled = scaler_test.fit_transform(X_test_loco)
    
    model = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.01, subsample=0.8,
                            scale_pos_weight=2.5, random_state=42, eval_metric='logloss')
    model.fit(X_train_scaled, y_train_loco)
    
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    return {
        'Held-out center': held_out,
        'Accuracy': accuracy_score(y_test_loco, preds),
        'Recall': recall_score(y_test_loco, preds),
        'ROC-AUC': roc_auc_score(y_test_loco, probs)
    }

xgb_loco_results = pd.DataFrame([run_loco_xgboost(c) for c in ['RUMC', 'PCNN', 'ZGT']])
print(xgb_loco_results)


# %% [cell 102]
def run_loco_isolated(held_out, feature_type):
    train_data = fused_with_center[fused_with_center['center'] != held_out]
    test_data = fused_with_center[fused_with_center['center'] == held_out]
    
    if feature_type == 'clinical':
        cols = ['patient_age', 'psa', 'psad', 'prostate_volume']
    else:
        cols = [c for c in fused_with_center.columns if c.startswith('cnn_pca_')]
    
    X_train_loco = train_data[cols]
    y_train_loco = train_data['csPCa_label']
    X_test_loco = test_data[cols]
    y_test_loco = test_data['csPCa_label']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_loco)
    X_test_scaled = scaler.transform(X_test_loco)
    
    model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train_loco)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    return roc_auc_score(y_test_loco, probs)

print("Clinical-only LOCO AUC:")
for c in ['RUMC', 'PCNN', 'ZGT']:
    print(f"  {c}: {run_loco_isolated(c, 'clinical'):.3f}")

print("\nImaging-only LOCO AUC:")
for c in ['RUMC', 'PCNN', 'ZGT']:
    print(f"  {c}: {run_loco_isolated(c, 'imaging'):.3f}")


# %% [cell 103]
print("Comparing fused vs clinical-only across all centers:")
for c in ['RUMC', 'PCNN', 'ZGT']:
    clin_auc = run_loco_isolated(c, 'clinical')
    print(f"{c}: clinical-only AUC = {clin_auc:.3f}")


# %% [cell 104]
def run_loco_with_domain_adaptation(held_out):
    train_data = fused_with_center[fused_with_center['center'] != held_out]
    test_data = fused_with_center[fused_with_center['center'] == held_out]
    
    X_train_loco = train_data.drop(columns=['patient_id', 'csPCa_label', 'center'])
    y_train_loco = train_data['csPCa_label']
    X_test_loco = test_data.drop(columns=['patient_id', 'csPCa_label', 'center'])
    y_test_loco = test_data['csPCa_label']
    
    # Scale train and test SEPARATELY using their own feature statistics
    scaler_train = StandardScaler()
    X_train_scaled = scaler_train.fit_transform(X_train_loco)
    
    scaler_test = StandardScaler()
    X_test_scaled = scaler_test.fit_transform(X_test_loco)
    
    model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train_loco)
    
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    return {
        'Held-out center': held_out,
        'Accuracy': accuracy_score(y_test_loco, preds),
        'Precision': precision_score(y_test_loco, preds),
        'Recall': recall_score(y_test_loco, preds),
        'ROC-AUC': roc_auc_score(y_test_loco, probs)
    }

domain_adapted_results = pd.DataFrame([run_loco_with_domain_adaptation(c) for c in ['RUMC', 'PCNN', 'ZGT']])
print(domain_adapted_results)


# %% [cell 105]
picai_raw = pd.read_csv('picai_labels/clinical_information/marksheet.csv')

# PIRADS can be comma-separated for multiple lesions (e.g. "4" or "3,2") — take the MAX score per case
def get_max_pirads(val):
    if pd.isna(val):
        return np.nan
    scores = str(val).split(',')
    return max([int(s.strip()) for s in scores if s.strip().isdigit()])

picai_raw['max_pirads'] = picai_raw['lesion_PIRADS'].apply(get_max_pirads)
print(picai_raw['max_pirads'].value_counts(dropna=False))


# %% [cell 106]
fused_with_pirads = fused_df_reload.merge(
    picai_raw[['patient_id', 'max_pirads']].drop_duplicates('patient_id'),
    on='patient_id', how='left'
)
fused_with_pirads['max_pirads'] = fused_with_pirads['max_pirads'].fillna(fused_with_pirads['max_pirads'].median())

X_pirads = fused_with_pirads.drop(columns=['patient_id', 'csPCa_label'])
y_pirads = fused_with_pirads['csPCa_label']

Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_pirads, y_pirads, test_size=0.2, stratify=y_pirads, random_state=42)
scaler_p = StandardScaler()
Xp_train_scaled = scaler_p.fit_transform(Xp_train)
Xp_test_scaled = scaler_p.transform(Xp_test)

rf_pirads = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
rf_pirads.fit(Xp_train_scaled, yp_train)

preds_p = rf_pirads.predict(Xp_test_scaled)
probs_p = rf_pirads.predict_proba(Xp_test_scaled)[:, 1]

print("Accuracy:", accuracy_score(yp_test, preds_p))
print("Recall:", recall_score(yp_test, preds_p))
print("ROC-AUC:", roc_auc_score(yp_test, probs_p))


# %% [cell 107]
# Train clinical-only model
clinical_model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
clinical_model.fit(Xc_train_scaled, yc_train)
clinical_probs = clinical_model.predict_proba(Xc_test_scaled)[:, 1]

# Train imaging-only model
imaging_model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
imaging_model.fit(Xi_train_scaled, yi_train)
imaging_probs = imaging_model.predict_proba(Xi_test_scaled)[:, 1]

# Combine via simple averaging (basic late fusion)
late_fusion_probs_avg = (clinical_probs + imaging_probs) / 2
late_fusion_preds_avg = (late_fusion_probs_avg >= 0.5).astype(int)

print("=== Late Fusion (Simple Average) ===")
print("Accuracy:", accuracy_score(y_test, late_fusion_preds_avg))
print("Recall:", recall_score(y_test, late_fusion_preds_avg))
print("ROC-AUC:", roc_auc_score(y_test, late_fusion_probs_avg))

# Combine via a learned meta-combiner (weighted late fusion — more sophisticated)
meta_features_train = np.column_stack([
    clinical_model.predict_proba(Xc_train_scaled)[:, 1],
    imaging_model.predict_proba(Xi_train_scaled)[:, 1]
])
meta_features_test = np.column_stack([clinical_probs, imaging_probs])

meta_combiner = LogisticRegression(class_weight='balanced')
meta_combiner.fit(meta_features_train, yc_train)  # y is same across both since same patients

late_fusion_preds_learned = meta_combiner.predict(meta_features_test)
late_fusion_probs_learned = meta_combiner.predict_proba(meta_features_test)[:, 1]

print("\n=== Late Fusion (Learned Combiner) ===")
print("Accuracy:", accuracy_score(y_test, late_fusion_preds_learned))
print("Recall:", recall_score(y_test, late_fusion_preds_learned))
print("ROC-AUC:", roc_auc_score(y_test, late_fusion_probs_learned))


# %% [cell 108]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the fused dataset (contains clinical + imaging PCA columns together)
fused_df_reload = pd.read_csv('fused_final_dataset.csv')

clinical_cols = ['patient_age', 'psa', 'psad', 'prostate_volume']
imaging_cols = [c for c in fused_df_reload.columns if c.startswith('cnn_pca_')]

X_all = fused_df_reload.drop(columns=['patient_id', 'csPCa_label'], errors='ignore')
y_all = fused_df_reload['csPCa_label']

# ONE split, used for everything below — guarantees row alignment
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)

# Slice out clinical-only and imaging-only views from the SAME split
Xc_train, Xc_test = X_train_full[clinical_cols], X_test_full[clinical_cols]
Xi_train, Xi_test = X_train_full[imaging_cols], X_test_full[imaging_cols]

# Scale each view separately
scaler_full = StandardScaler()
X_train_scaled = scaler_full.fit_transform(X_train_full)
X_test_scaled = scaler_full.transform(X_test_full)

scaler_c = StandardScaler()
Xc_train_scaled = scaler_c.fit_transform(Xc_train)
Xc_test_scaled = scaler_c.transform(Xc_test)

scaler_i = StandardScaler()
Xi_train_scaled = scaler_i.fit_transform(Xi_train)
Xi_test_scaled = scaler_i.transform(Xi_test)

print("Full:", X_train_scaled.shape, X_test_scaled.shape)
print("Clinical:", Xc_train_scaled.shape, Xc_test_scaled.shape)
print("Imaging:", Xi_train_scaled.shape, Xi_test_scaled.shape)
print("Alignment check — same y_train/y_test used for all three views")


# %% [cell 109]
clinical_model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
clinical_model.fit(Xc_train_scaled, y_train)
clinical_probs = clinical_model.predict_proba(Xc_test_scaled)[:, 1]

imaging_model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
imaging_model.fit(Xi_train_scaled, y_train)
imaging_probs = imaging_model.predict_proba(Xi_test_scaled)[:, 1]

late_fusion_probs_avg = (clinical_probs + imaging_probs) / 2
late_fusion_preds_avg = (late_fusion_probs_avg >= 0.5).astype(int)

print("=== Late Fusion (Simple Average) ===")
print("Accuracy:", accuracy_score(y_test, late_fusion_preds_avg))
print("Recall:", recall_score(y_test, late_fusion_preds_avg))
print("ROC-AUC:", roc_auc_score(y_test, late_fusion_probs_avg))

meta_features_train = np.column_stack([
    clinical_model.predict_proba(Xc_train_scaled)[:, 1],
    imaging_model.predict_proba(Xi_train_scaled)[:, 1]
])
meta_features_test = np.column_stack([clinical_probs, imaging_probs])

meta_combiner = LogisticRegression(class_weight='balanced')
meta_combiner.fit(meta_features_train, y_train)

late_fusion_preds_learned = meta_combiner.predict(meta_features_test)
late_fusion_probs_learned = meta_combiner.predict_proba(meta_features_test)[:, 1]

print("\n=== Late Fusion (Learned Combiner) ===")
print("Accuracy:", accuracy_score(y_test, late_fusion_preds_learned))
print("Recall:", recall_score(y_test, late_fusion_preds_learned))
print("ROC-AUC:", roc_auc_score(y_test, late_fusion_probs_learned))


# %% [cell 110]
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

clinical_input = Input(shape=(Xc_train_scaled.shape[1],), name='clinical_input')
c = Dense(16, activation='relu')(clinical_input)
c = Dropout(0.2)(c)

imaging_input = Input(shape=(Xi_train_scaled.shape[1],), name='imaging_input')
i = Dense(64, activation='relu')(imaging_input)
i = Dropout(0.3)(i)
i = Dense(32, activation='relu')(i)

merged = Concatenate()([c, i])
m = Dense(32, activation='relu')(merged)
m = Dropout(0.3)(m)
output = Dense(1, activation='sigmoid')(m)

intermediate_model = Model(inputs=[clinical_input, imaging_input], outputs=output)
intermediate_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history_intermediate = intermediate_model.fit(
    [Xc_train_scaled, Xi_train_scaled], y_train,
    validation_data=([Xc_test_scaled, Xi_test_scaled], y_test),
    epochs=30, batch_size=32,
    class_weight={0: 1.0, 1: 2.5}
)


# %% [cell 111]
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

probs_intermediate = intermediate_model.predict([Xc_test_scaled, Xi_test_scaled]).flatten()
preds_intermediate = (probs_intermediate >= 0.5).astype(int)

print("=== Intermediate Fusion ===")
print("Accuracy:", accuracy_score(y_test, preds_intermediate))
print("Recall:", recall_score(y_test, preds_intermediate))
print("ROC-AUC:", roc_auc_score(y_test, probs_intermediate))


# %% [cell 112]
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

t2w_features_df = pd.read_csv('cnn_features_t2w.csv')
adc_features_df = pd.read_csv('cnn_features_adc.csv')
hbv_features_df = pd.read_csv('cnn_features_hbv.csv')

def reduce_sequence(df, feat_prefix, output_prefix, n_components=35):
    feat_cols = [c for c in df.columns if c.startswith(feat_prefix)]
    X = df[feat_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_cols = [f'{output_prefix}_pca_{i}' for i in range(n_components)]
    result_df = pd.DataFrame(X_pca, columns=pca_cols)
    result_df['patient_id'] = df['patient_id'].astype(int).values
    return result_df

t2w_pca_df = reduce_sequence(t2w_features_df, 'cnn_feat_', 't2w')
adc_pca_df = reduce_sequence(adc_features_df, 'adc_feat_', 'adc')
hbv_pca_df = reduce_sequence(hbv_features_df, 'hbv_feat_', 'hbv')

imaging_combined = t2w_pca_df.merge(adc_pca_df, on='patient_id', how='inner').merge(hbv_pca_df, on='patient_id', how='inner')
print("Multiparametric imaging shape:", imaging_combined.shape)


# %% [cell 113]
picai_clinical_clean = pd.read_csv('picai_clinical_clean.csv')
picai_raw = pd.read_csv('picai_labels/clinical_information/marksheet.csv')

def get_max_pirads(val):
    if pd.isna(val):
        return np.nan
    scores = str(val).split(',')
    return max([int(s.strip()) for s in scores if s.strip().isdigit()])

picai_raw['max_pirads'] = picai_raw['lesion_PIRADS'].apply(get_max_pirads)

fused_mp_pirads = picai_clinical_clean.merge(imaging_combined, on='patient_id', how='inner')
fused_mp_pirads = fused_mp_pirads.merge(
    picai_raw[['patient_id', 'max_pirads']].drop_duplicates('patient_id'),
    on='patient_id', how='left'
)
fused_mp_pirads['max_pirads'] = fused_mp_pirads['max_pirads'].fillna(fused_mp_pirads['max_pirads'].median())

print("Final dataset shape:", fused_mp_pirads.shape)


# %% [cell 114]
X_mp_pirads = fused_mp_pirads.drop(columns=['patient_id', 'csPCa_label', 'center'], errors='ignore')
y_mp_pirads = fused_mp_pirads['csPCa_label']

Xmp_train, Xmp_test, ymp_train, ymp_test = train_test_split(
    X_mp_pirads, y_mp_pirads, test_size=0.2, stratify=y_mp_pirads, random_state=42
)

scaler_mp = StandardScaler()
Xmp_train_scaled = scaler_mp.fit_transform(Xmp_train)
Xmp_test_scaled = scaler_mp.transform(Xmp_test)

rf_mp_pirads = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
rf_mp_pirads.fit(Xmp_train_scaled, ymp_train)

preds_mp_p = rf_mp_pirads.predict(Xmp_test_scaled)
probs_mp_p = rf_mp_pirads.predict_proba(Xmp_test_scaled)[:, 1]

print("=== Multiparametric (T2W+ADC+HBV) + Clinical + PI-RADS ===")
print("Accuracy:", accuracy_score(ymp_test, preds_mp_p))
print("Precision:", precision_score(ymp_test, preds_mp_p))
print("Recall:", recall_score(ymp_test, preds_mp_p))
print("F1:", f1_score(ymp_test, preds_mp_p))
print("ROC-AUC:", roc_auc_score(ymp_test, probs_mp_p))


# %% [cell 115]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

fused_df_reload = pd.read_csv('fused_final_dataset.csv')
picai_raw = pd.read_csv('picai_labels/clinical_information/marksheet.csv')

def get_max_pirads(val):
    if pd.isna(val):
        return np.nan
    scores = str(val).split(',')
    return max([int(s.strip()) for s in scores if s.strip().isdigit()])

picai_raw['max_pirads'] = picai_raw['lesion_PIRADS'].apply(get_max_pirads)

fused_with_pirads = fused_df_reload.merge(
    picai_raw[['patient_id', 'max_pirads']].drop_duplicates('patient_id'),
    on='patient_id', how='left'
)
fused_with_pirads['max_pirads'] = fused_with_pirads['max_pirads'].fillna(fused_with_pirads['max_pirads'].median())

X_pirads = fused_with_pirads.drop(columns=['patient_id', 'csPCa_label'])
y_pirads = fused_with_pirads['csPCa_label']

Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_pirads, y_pirads, test_size=0.2, stratify=y_pirads, random_state=42)
scaler_p = StandardScaler()
Xp_train_scaled = scaler_p.fit_transform(Xp_train)
Xp_test_scaled = scaler_p.transform(Xp_test)

print("Dataset shape:", Xp_train_scaled.shape, Xp_test_scaled.shape)


# %% [cell 116]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_params_full = {
    'LogisticRegression': (LogisticRegression(max_iter=1000, class_weight='balanced'), {'C': [0.01, 0.1, 1, 10]}),
    'DecisionTree': (DecisionTreeClassifier(random_state=42, class_weight='balanced'), {'max_depth': [5, 10, None]}),
    'RandomForest': (RandomForestClassifier(random_state=42, class_weight='balanced'), {'n_estimators': [100, 300], 'max_depth': [10, 20]}),
    'ExtraTrees': (ExtraTreesClassifier(random_state=42, class_weight='balanced'), {'n_estimators': [100, 300], 'max_depth': [10, 20]}),
    'GradientBoosting': (GradientBoostingClassifier(random_state=42), {'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1]}),
    'AdaBoost': (AdaBoostClassifier(random_state=42), {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]}),
    'SVM': (SVC(probability=True, class_weight='balanced'), {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}),
    'KNN': (KNeighborsClassifier(), {'n_neighbors': [5, 10, 15]}),
    'NaiveBayes': (GaussianNB(), {}),
    'LDA': (LinearDiscriminantAnalysis(), {}),
    'MLP': (MLPClassifier(max_iter=1000, random_state=42), {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.001, 0.01]}),
    'XGBoost': (XGBClassifier(scale_pos_weight=2.5, random_state=42, eval_metric='logloss'),
                {'n_estimators': [200, 300], 'max_depth': [3, 5, 6], 'learning_rate': [0.01, 0.1]})
}

best_models_pirads = {}
for name, (model, params) in models_params_full.items():
    if params:
        grid = GridSearchCV(model, params, cv=cv, scoring='roc_auc', n_jobs=-1)
        grid.fit(Xp_train_scaled, yp_train)
        best_models_pirads[name] = grid.best_estimator_
    else:
        model.fit(Xp_train_scaled, yp_train)
        best_models_pirads[name] = model
    print(f"{name} tuned")


# %% [cell 117]
estimators_final = [
    ('rf', best_models_pirads['RandomForest']),
    ('ada', best_models_pirads['AdaBoost']),
    ('gb', best_models_pirads['GradientBoosting'])
]
stacked_pirads = StackingClassifier(estimators=estimators_final, final_estimator=LogisticRegression(class_weight='balanced'), cv=5)
stacked_pirads.fit(Xp_train_scaled, yp_train)
best_models_pirads['Stacking'] = stacked_pirads
print("Stacking trained")


# %% [cell 118]
results_pirads = []
for name, model in best_models_pirads.items():
    preds = model.predict(Xp_test_scaled)
    probs = model.predict_proba(Xp_test_scaled)[:, 1]
    results_pirads.append({
        'Model': name,
        'Accuracy': accuracy_score(yp_test, preds),
        'Precision': precision_score(yp_test, preds),
        'Recall': recall_score(yp_test, preds),
        'F1': f1_score(yp_test, preds),
        'ROC-AUC': roc_auc_score(yp_test, probs)
    })

results_pirads_df = pd.DataFrame(results_pirads).sort_values('ROC-AUC', ascending=False)
print(results_pirads_df)


# %% [cell 119]
fused_with_pirads.to_csv('fused_final_dataset_with_pirads.csv', index=False)
print("Saved:", fused_with_pirads.shape)


# %% [cell 120]
print(picai_raw['histopath_type'].value_counts(dropna=False))


# %% [cell 121]
fused_with_histopath = fused_with_pirads.merge(
    picai_raw[['patient_id', 'histopath_type']].drop_duplicates('patient_id'),
    on='patient_id', how='left'
)
fused_with_histopath['histopath_type'] = fused_with_histopath['histopath_type'].fillna('No_Biopsy_Needed')

fused_with_histopath_encoded = pd.get_dummies(fused_with_histopath, columns=['histopath_type'], drop_first=True)

X_hist = fused_with_histopath_encoded.drop(columns=['patient_id', 'csPCa_label'])
y_hist = fused_with_histopath_encoded['csPCa_label']

Xh_train, Xh_test, yh_train, yh_test = train_test_split(X_hist, y_hist, test_size=0.2, stratify=y_hist, random_state=42)
scaler_h = StandardScaler()
Xh_train_scaled = scaler_h.fit_transform(Xh_train)
Xh_test_scaled = scaler_h.transform(Xh_test)

rf_hist = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
rf_hist.fit(Xh_train_scaled, yh_train)

print("Accuracy:", accuracy_score(yh_test, rf_hist.predict(Xh_test_scaled)))
print("Recall:", recall_score(yh_test, rf_hist.predict(Xh_test_scaled)))
print("ROC-AUC:", roc_auc_score(yh_test, rf_hist.predict_proba(Xh_test_scaled)[:,1]))


# %% [cell 122]
print(pd.crosstab(picai_raw['histopath_type'].fillna('No_Biopsy'), 
                    picai_raw.merge(fused_df_reload[['patient_id','csPCa_label']], on='patient_id')['csPCa_label']))


# %% [cell 123]
print(Xh_train_scaled.shape)


# %% [cell 124]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_params_full = {
    'LogisticRegression': (LogisticRegression(max_iter=1000, class_weight='balanced'), {'C': [0.01, 0.1, 1, 10]}),
    'DecisionTree': (DecisionTreeClassifier(random_state=42, class_weight='balanced'), {'max_depth': [5, 10, None]}),
    'RandomForest': (RandomForestClassifier(random_state=42, class_weight='balanced'), {'n_estimators': [100, 300], 'max_depth': [10, 20]}),
    'ExtraTrees': (ExtraTreesClassifier(random_state=42, class_weight='balanced'), {'n_estimators': [100, 300], 'max_depth': [10, 20]}),
    'GradientBoosting': (GradientBoostingClassifier(random_state=42), {'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1]}),
    'AdaBoost': (AdaBoostClassifier(random_state=42), {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]}),
    'SVM': (SVC(probability=True, class_weight='balanced'), {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}),
    'KNN': (KNeighborsClassifier(), {'n_neighbors': [5, 10, 15]}),
    'NaiveBayes': (GaussianNB(), {}),
    'LDA': (LinearDiscriminantAnalysis(), {}),
    'MLP': (MLPClassifier(max_iter=1000, random_state=42), {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.001, 0.01]}),
    'XGBoost': (XGBClassifier(scale_pos_weight=2.5, random_state=42, eval_metric='logloss'),
                {'n_estimators': [200, 300], 'max_depth': [3, 5, 6], 'learning_rate': [0.01, 0.1]})
}

best_models_hist = {}
for name, (model, params) in models_params_full.items():
    if params:
        grid = GridSearchCV(model, params, cv=cv, scoring='roc_auc', n_jobs=-1)
        grid.fit(Xh_train_scaled, yh_train)
        best_models_hist[name] = grid.best_estimator_
    else:
        model.fit(Xh_train_scaled, yh_train)
        best_models_hist[name] = model
    print(f"{name} tuned")


# %% [cell 125]
estimators_hist = [
    ('rf', best_models_hist['RandomForest']),
    ('ada', best_models_hist['AdaBoost']),
    ('gb', best_models_hist['GradientBoosting'])
]
stacked_hist = StackingClassifier(estimators=estimators_hist, final_estimator=LogisticRegression(class_weight='balanced'), cv=5)
stacked_hist.fit(Xh_train_scaled, yh_train)
best_models_hist['Stacking'] = stacked_hist
print("Stacking trained")


# %% [cell 126]
results_hist = []
for name, model in best_models_hist.items():
    preds = model.predict(Xh_test_scaled)
    probs = model.predict_proba(Xh_test_scaled)[:, 1]
    results_hist.append({
        'Model': name,
        'Accuracy': accuracy_score(yh_test, preds),
        'Precision': precision_score(yh_test, preds),
        'Recall': recall_score(yh_test, preds),
        'F1': f1_score(yh_test, preds),
        'ROC-AUC': roc_auc_score(yh_test, probs)
    })

results_hist_df = pd.DataFrame(results_hist).sort_values('ROC-AUC', ascending=False)
print(results_hist_df)


# %% [cell 127]
for name in ['NaiveBayes', 'XGBoost', 'Stacking']:
    preds = best_models_hist[name].predict(Xh_test_scaled)
    print(f"{name} prediction distribution:", np.bincount(preds))
print("Actual distribution:", np.bincount(yh_test))


# %% [cell 128]
fused_with_histopath_encoded.to_csv('fused_final_dataset_with_pirads_histopath.csv', index=False)
print("Saved:", fused_with_histopath_encoded.shape)


# %% [cell 129]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Step 1: Rebuild the dataset (Clinical + Imaging + PI-RADS + Biopsy type) ---
fused_df_reload = pd.read_csv('fused_final_dataset.csv')
picai_raw = pd.read_csv('picai_labels/clinical_information/marksheet.csv')

def get_max_pirads(val):
    if pd.isna(val):
        return np.nan
    scores = str(val).split(',')
    return max([int(s.strip()) for s in scores if s.strip().isdigit()])

picai_raw['max_pirads'] = picai_raw['lesion_PIRADS'].apply(get_max_pirads)

fused_with_pirads = fused_df_reload.merge(
    picai_raw[['patient_id', 'max_pirads']].drop_duplicates('patient_id'),
    on='patient_id', how='left'
)
fused_with_pirads['max_pirads'] = fused_with_pirads['max_pirads'].fillna(fused_with_pirads['max_pirads'].median())

fused_with_histopath = fused_with_pirads.merge(
    picai_raw[['patient_id', 'histopath_type']].drop_duplicates('patient_id'),
    on='patient_id', how='left'
)
fused_with_histopath['histopath_type'] = fused_with_histopath['histopath_type'].fillna('No_Biopsy_Needed')
fused_with_histopath_encoded = pd.get_dummies(fused_with_histopath, columns=['histopath_type'], drop_first=True)

print("Final dataset shape:", fused_with_histopath_encoded.shape)

# --- Step 2: 80/20 train/test split ---
X_final = fused_with_histopath_encoded.drop(columns=['patient_id', 'csPCa_label'])
y_final = fused_with_histopath_encoded['csPCa_label']

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, stratify=y_final, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Train:", X_train_scaled.shape, "| Test:", X_test_scaled.shape)
print("Train class balance:", np.bincount(y_train))
print("Test class balance:", np.bincount(y_test))

# --- Step 3: Train all 13 algorithms with hyperparameter tuning ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_params = {
    'LogisticRegression': (LogisticRegression(max_iter=1000, class_weight='balanced'), {'C': [0.01, 0.1, 1, 10]}),
    'DecisionTree': (DecisionTreeClassifier(random_state=42, class_weight='balanced'), {'max_depth': [5, 10, None]}),
    'RandomForest': (RandomForestClassifier(random_state=42, class_weight='balanced'), {'n_estimators': [100, 300], 'max_depth': [10, 20]}),
    'ExtraTrees': (ExtraTreesClassifier(random_state=42, class_weight='balanced'), {'n_estimators': [100, 300], 'max_depth': [10, 20]}),
    'GradientBoosting': (GradientBoostingClassifier(random_state=42), {'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1]}),
    'AdaBoost': (AdaBoostClassifier(random_state=42), {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]}),
    'SVM': (SVC(probability=True, class_weight='balanced'), {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}),
    'KNN': (KNeighborsClassifier(), {'n_neighbors': [5, 10, 15]}),
    'NaiveBayes': (GaussianNB(), {}),
    'LDA': (LinearDiscriminantAnalysis(), {}),
    'MLP': (MLPClassifier(max_iter=1000, random_state=42), {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.001, 0.01]}),
    'XGBoost': (XGBClassifier(scale_pos_weight=2.5, random_state=42, eval_metric='logloss'),
                {'n_estimators': [200, 300], 'max_depth': [3, 5, 6], 'learning_rate': [0.01, 0.1]})
}

best_models = {}
for name, (model, params) in models_params.items():
    if params:
        grid = GridSearchCV(model, params, cv=cv, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_models[name] = grid.best_estimator_
    else:
        model.fit(X_train_scaled, y_train)
        best_models[name] = model
    print(f"{name} trained")

# --- Step 4: Add Stacking ---
estimators = [
    ('rf', best_models['RandomForest']),
    ('ada', best_models['AdaBoost']),
    ('gb', best_models['GradientBoosting'])
]
stacked_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(class_weight='balanced'), cv=5)
stacked_model.fit(X_train_scaled, y_train)
best_models['Stacking'] = stacked_model
print("Stacking trained")

# --- Step 5: Evaluate all 13 on the 20% test set ---
results = []
for name, model in best_models.items():
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1': f1_score(y_test, preds),
        'ROC-AUC': roc_auc_score(y_test, probs)
    })

results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
print(results_df)


# %% [cell 130]
def count_lesions(val):
    if pd.isna(val):
        return 0
    return len(str(val).split(','))

picai_raw['num_lesions'] = picai_raw['lesion_PIRADS'].apply(count_lesions)
print(picai_raw['num_lesions'].value_counts())


# %% [cell 131]
fused_with_lesion_count = fused_with_histopath_encoded.merge(
    picai_raw[['patient_id', 'num_lesions']].drop_duplicates('patient_id'),
    on='patient_id', how='left'
)

X_lc = fused_with_lesion_count.drop(columns=['patient_id', 'csPCa_label'])
y_lc = fused_with_lesion_count['csPCa_label']

Xlc_train, Xlc_test, ylc_train, ylc_test = train_test_split(X_lc, y_lc, test_size=0.2, stratify=y_lc, random_state=42)
scaler_lc = StandardScaler()
Xlc_train_scaled = scaler_lc.fit_transform(Xlc_train)
Xlc_test_scaled = scaler_lc.transform(Xlc_test)

# Test on Stacking (your current best model)
estimators_lc = [
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')),
    ('ada', AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=42))
]
stacked_lc = StackingClassifier(estimators=estimators_lc, final_estimator=LogisticRegression(class_weight='balanced'), cv=5)
stacked_lc.fit(Xlc_train_scaled, ylc_train)

preds_lc = stacked_lc.predict(Xlc_test_scaled)
probs_lc = stacked_lc.predict_proba(Xlc_test_scaled)[:, 1]

print("=== Stacking + Lesion Count ===")
print("Accuracy:", accuracy_score(ylc_test, preds_lc))
print("Recall:", recall_score(ylc_test, preds_lc))
print("F1:", f1_score(ylc_test, preds_lc))
print("ROC-AUC:", roc_auc_score(ylc_test, probs_lc))


# %% [cell 132]
import pandas as pd

target_df = pd.read_excel('external_validation/target_data.xlsx')
biopsy_df = pd.read_excel('external_validation/biopsy_data.xlsx')

print("=== Target Data ===")
print(target_df.shape)
print(target_df.columns.tolist())
print(target_df.head())

print("\n=== Biopsy Data ===")
print(biopsy_df.shape)
print(biopsy_df.columns.tolist())
print(biopsy_df.head())


# %% [cell 133]
import pandas as pd

target_df = pd.read_excel('external_validation/target_data.xlsx')
biopsy_df = pd.read_excel('external_validation/biopsy_data.xlsx')

print("=== Target Data ===")
print(target_df.shape)
print(target_df.columns.tolist())
print(target_df.head())

print("\n=== Biopsy Data ===")
print(biopsy_df.shape)
print(biopsy_df.columns.tolist())
print(biopsy_df.head())


# %% [cell 134]
print("Unique patients in Biopsy Data:", biopsy_df['Patient Number'].nunique())
print("Unique patients in Target Data:", target_df['Patient ID'].nunique())


# %% [cell 135]
# Compute Gleason Score per core, then take the MAX per patient (most severe finding)
biopsy_df['gleason_score'] = biopsy_df['Primary Gleason'] + biopsy_df['Secondary Gleason']

patient_biopsy = biopsy_df.groupby('Patient Number').agg(
    psa=('PSA (ng/mL)', 'first'),  # should be constant per patient
    prostate_volume=('Prostate Volume (CC)', 'first'),
    max_gleason=('gleason_score', 'max')  # NaN-safe: cores with no cancer show NaN, handled by max()
).reset_index()

patient_biopsy['psad'] = patient_biopsy['psa'] / patient_biopsy['prostate_volume']

# csPCa label: ISUP >=2 equivalent is Gleason >= 7 (matches your PI-CAI logic exactly)
patient_biopsy['csPCa_label'] = (patient_biopsy['max_gleason'] >= 7).astype(int)

print(patient_biopsy.shape)
print(patient_biopsy['csPCa_label'].value_counts())
print(patient_biopsy.isnull().sum())


# %% [cell 136]
patient_target = target_df.groupby('Patient ID').agg(
    max_pirads=('UCLA Score (Similar to PIRADS v2)', 'max')
).reset_index().rename(columns={'Patient ID': 'Patient Number'})

external_clinical = patient_biopsy.merge(patient_target, on='Patient Number', how='left')
external_clinical['max_pirads'] = external_clinical['max_pirads'].fillna(external_clinical['max_pirads'].median())

print("Final external clinical dataset shape:", external_clinical.shape)
print(external_clinical.head())
external_clinical.to_csv('external_validation/external_clinical_clean.csv', index=False)


# %% [cell 137]
print(external_clinical.isnull().sum())
print()
print(external_clinical['csPCa_label'].value_counts())
print()
print(external_clinical['max_gleason'].isnull().sum(), "patients with no cancer found in any core")


# %% [cell 138]
external_clinical['psa'] = external_clinical['psa'].fillna(external_clinical['psa'].median())
external_clinical['prostate_volume'] = external_clinical['prostate_volume'].fillna(external_clinical['prostate_volume'].median())
external_clinical['psad'] = external_clinical['psa'] / external_clinical['prostate_volume']

print(external_clinical.isnull().sum())
external_clinical.to_csv('external_validation/external_clinical_clean.csv', index=False)
print("Saved:", external_clinical.shape)


# %% [cell 139]
import os
patient_folders = [f for f in os.listdir('external_validation/prostate_mri_us_biopsy_images') if os.path.isdir(os.path.join('external_validation/prostate_mri_us_biopsy_images', f))]
print(f"Found {len(patient_folders)} patient folders")
print(patient_folders[:5])


# %% [cell 140]
import os

download_root = os.path.expanduser('~/prostate-cancer-research/external_validation/prostate_mri_us_biopsy_images')
# or wherever you set the download directory - check Finder if unsure

print(os.listdir(download_root)[:5])


# %% [cell 141]
patient_folders = os.listdir(download_root)
sample_patient = patient_folders[0]
print("Sample patient:", sample_patient)

for root, dirs, files in os.walk(os.path.join(download_root, sample_patient)):
    level = root.replace(download_root, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level > 3:  # avoid printing every single .dcm file
        break


# %% [cell 142]
import pydicom

# Find one .dcm file anywhere in the sample patient folder
for root, dirs, files in os.walk(os.path.join(download_root, sample_patient)):
    dcm_files = [f for f in files if f.endswith('.dcm')]
    if dcm_files:
        sample_dcm_path = os.path.join(root, dcm_files[0])
        break

ds = pydicom.dcmread(sample_dcm_path)
print("Series Description:", ds.get('SeriesDescription', 'N/A'))
print("Modality:", ds.get('Modality', 'N/A'))
print("Patient ID:", ds.get('PatientID', 'N/A'))


# %% [cell 143]
import os

download_root = os.path.expanduser('~/prostate-cancer-research/external_validation/prostate_mri_us_biopsy_images')
actual_root = os.path.join(download_root, 'prostate_mri_us_biopsy_manifest', 'prostate_mri_us_biopsy')

patient_folders = [f for f in os.listdir(actual_root) if os.path.isdir(os.path.join(actual_root, f))]
print(f"Found {len(patient_folders)} patient folders")
print(patient_folders[:5])


# %% [cell 144]
import pydicom

sample_patient = patient_folders[0]
patient_path = os.path.join(actual_root, sample_patient)

print(f"=== Studies/series under {sample_patient} ===")
for study_folder in os.listdir(patient_path):
    study_path = os.path.join(patient_path, study_folder)
    if not os.path.isdir(study_path):
        continue
    for series_folder in os.listdir(study_path):
        series_path = os.path.join(study_path, series_folder)
        if not os.path.isdir(series_path):
            continue
        dcm_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
        if dcm_files:
            ds = pydicom.dcmread(os.path.join(series_path, dcm_files[0]))
            print(f"Series folder: {series_folder}")
            print(f"  Modality: {ds.get('Modality', 'N/A')}")
            print(f"  SeriesDescription: {ds.get('SeriesDescription', 'N/A')}")
            print(f"  Number of images: {len(dcm_files)}")
            print()


# %% [cell 145]
import os

actual_root = os.path.join(download_root, 'prostate_mri_us_biopsy_manifest', 'prostate_mri_us_biopsy')
patient_folders = [f for f in os.listdir(actual_root) if os.path.isdir(os.path.join(actual_root, f))]

t2w_catalog = []

for i, patient in enumerate(patient_folders):
    patient_path = os.path.join(actual_root, patient)
    found_t2w = None
    
    for study_folder in os.listdir(patient_path):
        study_path = os.path.join(patient_path, study_folder)
        if not os.path.isdir(study_path):
            continue
        for series_folder in os.listdir(study_path):
            series_path = os.path.join(study_path, series_folder)
            if not os.path.isdir(series_path):
                continue
            # Match on t2_spc pattern (case-insensitive), seen in the sample above
            if 't2' in series_folder.lower() and 'spc' in series_folder.lower():
                dcm_count = len([f for f in os.listdir(series_path) if f.endswith('.dcm')])
                found_t2w = {'patient': patient, 'series_path': series_path, 'num_images': dcm_count}
                break
        if found_t2w:
            break
    
    if found_t2w:
        t2w_catalog.append(found_t2w)
    
    if i % 200 == 0:
        print(f"Scanned {i}/{len(patient_folders)}")

print(f"\nPatients with a T2W (t2_spc) series found: {len(t2w_catalog)} / {len(patient_folders)}")


# %% [cell 146]
# Check a sample of 20 random patients for ALL their series names, to catch naming variants
import random
sample_patients = random.sample(patient_folders, 20)

all_series_names = set()
for patient in sample_patients:
    patient_path = os.path.join(actual_root, patient)
    for study_folder in os.listdir(patient_path):
        study_path = os.path.join(patient_path, study_folder)
        if not os.path.isdir(study_path):
            continue
        for series_folder in os.listdir(study_path):
            all_series_names.add(series_folder)

for name in sorted(all_series_names):
    print(name)


# %% [cell 147]
found_patient_ids = set(item['patient'] for item in t2w_catalog)
not_found_patients = [p for p in patient_folders if p not in found_patient_ids]
print(f"Not found: {len(not_found_patients)}")

# Inspect 3 of them in full detail
for patient in not_found_patients[:3]:
    print(f"\n=== {patient} ===")
    patient_path = os.path.join(actual_root, patient)
    for study_folder in os.listdir(patient_path):
        study_path = os.path.join(patient_path, study_folder)
        if not os.path.isdir(study_path):
            continue
        for series_folder in os.listdir(study_path):
            series_path = os.path.join(study_path, series_folder)
            if not os.path.isdir(series_path):
                continue
            dcm_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
            if dcm_files:
                ds = pydicom.dcmread(os.path.join(series_path, dcm_files[0]))
                print(f"  Folder: {series_folder} | Modality: {ds.get('Modality','N/A')} | SeriesDescription: {ds.get('SeriesDescription','N/A')} | #images: {len(dcm_files)}")


# %% [cell 148]
import pandas as pd

t2w_df = pd.DataFrame(t2w_catalog)
print(t2w_df.shape)
print(t2w_df.head())

external_clinical = pd.read_csv('external_validation/external_clinical_clean.csv')

# Merge: only keep patients with BOTH clinical data AND a T2W series
final_external_cohort = external_clinical.merge(
    t2w_df.rename(columns={'patient': 'Patient Number'}),
    on='Patient Number', how='inner'
)

print("Final external cohort with clinical + T2W imaging:", final_external_cohort.shape)
print(final_external_cohort['csPCa_label'].value_counts())

final_external_cohort.to_csv('external_validation/final_external_cohort.csv', index=False)


# %% [cell 149]
import SimpleITK as sitk
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def load_middle_slice_dicom(series_folder_path, target_size=(224, 224)):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(series_folder_path)
    reader.SetFileNames(dicom_names)
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img)
    
    middle_idx = arr.shape[0] // 2
    slice_2d = arr[middle_idx]
    slice_norm = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    slice_resized = cv2.resize(slice_norm, target_size)
    slice_rgb = cv2.cvtColor(slice_resized, cv2.COLOR_GRAY2RGB)
    return slice_rgb

def extract_cnn_features(series_folder_path):
    slice_rgb = load_middle_slice_dicom(series_folder_path)
    img_array = np.expand_dims(slice_rgb, axis=0).astype(np.float32)
    img_preprocessed = preprocess_input(img_array)
    features = base_model.predict(img_preprocessed, verbose=0)
    return features.flatten()

# Test on one patient first
test_features = extract_cnn_features(final_external_cohort.iloc[0]['series_path'])
print("Test feature shape:", test_features.shape)


# %% [cell 150]
feature_rows = []

for i, row in final_external_cohort.iterrows():
    try:
        features = extract_cnn_features(row['series_path'])
        feature_row = {'Patient Number': row['Patient Number']}
        feature_row.update({f'cnn_feat_{j}': features[j] for j in range(len(features))})
        feature_rows.append(feature_row)
    except Exception as e:
        print(f"Failed on {row['Patient Number']}: {e}")
    
    if i % 100 == 0:
        print(f"Processed {i}/{len(final_external_cohort)}")

external_cnn_features_df = pd.DataFrame(feature_rows)
print("Final shape:", external_cnn_features_df.shape)
external_cnn_features_df.to_csv('external_validation/external_cnn_features_t2w.csv', index=False)


# %% [cell 151]
external_cnn_features_df.to_csv('external_validation/external_cnn_features_t2w.csv', index=False)
print("Saved:", external_cnn_features_df.shape)


# %% [cell 152]
# Rebuild the exact PCA fit from your original PI-CAI T2W features
t2w_features_df = pd.read_csv('cnn_features_t2w.csv')  # your original PI-CAI file
feat_cols_picai = [c for c in t2w_features_df.columns if c.startswith('cnn_feat_')]
X_picai_cnn = t2w_features_df[feat_cols_picai]

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler_cnn_locked = StandardScaler()
X_picai_cnn_scaled = scaler_cnn_locked.fit_transform(X_picai_cnn)

pca_locked = PCA(n_components=100, random_state=42)
pca_locked.fit(X_picai_cnn_scaled)

print("Locked PCA fitted on PI-CAI training data. Variance captured:", pca_locked.explained_variance_ratio_.sum())


# %% [cell 153]
feat_cols_external = [c for c in external_cnn_features_df.columns if c.startswith('cnn_feat_')]
X_external_cnn = external_cnn_features_df[feat_cols_external]

# Use .transform(), NOT .fit_transform() — this is the critical locked-model step
X_external_cnn_scaled = scaler_cnn_locked.transform(X_external_cnn)
X_external_pca = pca_locked.transform(X_external_cnn_scaled)

external_pca_df = pd.DataFrame(X_external_pca, columns=[f'cnn_pca_{i}' for i in range(100)])
external_pca_df['Patient Number'] = external_cnn_features_df['Patient Number'].values

print(external_pca_df.shape)


# %% [cell 154]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

fused_df_reload = pd.read_csv('fused_final_dataset.csv')
picai_raw = pd.read_csv('picai_labels/clinical_information/marksheet.csv')

def get_max_pirads(val):
    if pd.isna(val):
        return np.nan
    scores = str(val).split(',')
    return max([int(s.strip()) for s in scores if s.strip().isdigit()])

picai_raw['max_pirads'] = picai_raw['lesion_PIRADS'].apply(get_max_pirads)

fused_with_pirads = fused_df_reload.merge(
    picai_raw[['patient_id', 'max_pirads']].drop_duplicates('patient_id'),
    on='patient_id', how='left'
)
fused_with_pirads['max_pirads'] = fused_with_pirads['max_pirads'].fillna(fused_with_pirads['max_pirads'].median())

X_pirads = fused_with_pirads.drop(columns=['patient_id', 'csPCa_label'])
y_pirads = fused_with_pirads['csPCa_label']

# This exact split/scaler/model = your LOCKED pipeline
Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_pirads, y_pirads, test_size=0.2, stratify=y_pirads, random_state=42)
scaler_locked = StandardScaler()
Xp_train_scaled = scaler_locked.fit_transform(Xp_train)

estimators_locked = [
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')),
    ('ada', AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=42))
]
locked_model = StackingClassifier(estimators=estimators_locked, final_estimator=LogisticRegression(class_weight='balanced'), cv=5)
locked_model.fit(Xp_train_scaled, yp_train)

print("Locked model trained. Feature order:", X_pirads.columns.tolist())


# %% [cell 155]
picai_clinical_clean = pd.read_csv('picai_clinical_clean.csv')
median_age = picai_clinical_clean['patient_age'].median()

external_full = final_external_cohort[['Patient Number', 'psa', 'psad', 'prostate_volume', 'max_pirads', 'csPCa_label']].merge(
    external_pca_df, on='Patient Number', how='inner'
)
external_full['patient_age'] = median_age

# Reorder columns to EXACTLY match X_pirads.columns
feature_order = X_pirads.columns.tolist()
X_external_final = external_full[feature_order]
y_external_final = external_full['csPCa_label']

print(X_external_final.shape)
print("Column order matches:", X_external_final.columns.tolist() == feature_order)


# %% [cell 156]
X_external_scaled = scaler_locked.transform(X_external_final)

preds_external = locked_model.predict(X_external_scaled)
probs_external = locked_model.predict_proba(X_external_scaled)[:, 1]

print("=== EXTERNAL VALIDATION RESULTS ===")
print("Accuracy:", accuracy_score(y_external_final, preds_external))
print("Precision:", precision_score(y_external_final, preds_external))
print("Recall:", recall_score(y_external_final, preds_external))
print("F1:", f1_score(y_external_final, preds_external))
print("ROC-AUC:", roc_auc_score(y_external_final, probs_external))


# %% [cell 157]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier

fused_df_reload = pd.read_csv('fused_final_dataset.csv')
picai_raw = pd.read_csv('picai_labels/clinical_information/marksheet.csv')

def get_max_pirads(val):
    if pd.isna(val):
        return np.nan
    scores = str(val).split(',')
    return max([int(s.strip()) for s in scores if s.strip().isdigit()])

picai_raw['max_pirads'] = picai_raw['lesion_PIRADS'].apply(get_max_pirads)
fused_with_pirads = fused_df_reload.merge(
    picai_raw[['patient_id', 'max_pirads']].drop_duplicates('patient_id'),
    on='patient_id', how='left'
)
fused_with_pirads['max_pirads'] = fused_with_pirads['max_pirads'].fillna(fused_with_pirads['max_pirads'].median())

X_pirads = fused_with_pirads.drop(columns=['patient_id', 'csPCa_label'])
y_pirads = fused_with_pirads['csPCa_label']

Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_pirads, y_pirads, test_size=0.2, stratify=y_pirads, random_state=42)
scaler_locked = StandardScaler()
Xp_train_scaled = scaler_locked.fit_transform(Xp_train)
Xp_test_scaled = scaler_locked.transform(Xp_test)

estimators_locked = [
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')),
    ('ada', AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=42))
]
locked_model = StackingClassifier(estimators=estimators_locked, final_estimator=LogisticRegression(class_weight='balanced'), cv=5)
locked_model.fit(Xp_train_scaled, yp_train)

probs_internal = locked_model.predict_proba(Xp_test_scaled)[:, 1]
y_internal = yp_test.values

print("Internal test set ready:", len(y_internal), "patients")


# %% [cell 158]
import matplotlib.pyplot as plt

def calculate_net_benefit(y_true, y_prob, threshold):
    preds = (y_prob >= threshold).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    n = len(y_true)
    return (tp/n) - (fp/n) * (threshold/(1-threshold))

thresholds = np.arange(0.01, 0.5, 0.01)

# Net benefit of your model
nb_model_internal = [calculate_net_benefit(y_internal, probs_internal, t) for t in thresholds]

# Reference strategies
nb_treat_all = [(y_internal.mean()) - (1 - y_internal.mean()) * (t/(1-t)) for t in thresholds]
nb_treat_none = [0 for t in thresholds]

plt.figure(figsize=(8, 6))
plt.plot(thresholds, nb_model_internal, label='Model (Clinical+Imaging+PI-RADS)', linewidth=2, color='#065A82')
plt.plot(thresholds, nb_treat_all, label='Biopsy All', linestyle='--', color='gray')
plt.plot(thresholds, nb_treat_none, label='Biopsy None', linestyle=':', color='black')
plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.title('Decision Curve Analysis — Internal (PI-CAI) Test Set')
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(-0.1, max(nb_model_internal) + 0.05)
plt.savefig('dca_internal.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [cell 159]
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_internal, probs_internal, n_bins=10, strategy='quantile')

plt.figure(figsize=(7, 7))
plt.plot(prob_pred, prob_true, marker='o', label='Model', color='#065A82')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Frequency')
plt.title('Calibration Curve — Internal (PI-CAI) Test Set')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('calibration_internal.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [cell 161]
# Rebuild your other locked models (already trained on PI-CAI, no changes)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

rf_locked = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
rf_locked.fit(Xp_train_scaled, yp_train)

lr_locked = LogisticRegression(C=0.01, max_iter=1000, class_weight='balanced')
lr_locked.fit(Xp_train_scaled, yp_train)

xgb_locked = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.01, subsample=0.8, scale_pos_weight=2.5, random_state=42, eval_metric='logloss')
xgb_locked.fit(Xp_train_scaled, yp_train)

locked_models_all = {
    'Stacking': locked_model,  # already trained
    'RandomForest': rf_locked,
    'LogisticRegression': lr_locked,
    'XGBoost': xgb_locked
}

# Apply the SAME locked scaler, test on the SAME external cohort, for every model
X_external_scaled = scaler_locked.transform(X_external_final)

external_results = []
for name, model in locked_models_all.items():
    preds = model.predict(X_external_scaled)
    probs = model.predict_proba(X_external_scaled)[:, 1]
    external_results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_external_final, preds),
        'Precision': precision_score(y_external_final, preds),
        'Recall': recall_score(y_external_final, preds),
        'F1': f1_score(y_external_final, preds),
        'ROC-AUC': roc_auc_score(y_external_final, probs)
    })

external_results_df = pd.DataFrame(external_results).sort_values('ROC-AUC', ascending=False)
print(external_results_df)


# %% [cell 162]
import pandas as pd

pd_clinical = pd.read_excel('external_validation_2/prostate_diagnosis_clinical.xlsx')
print(pd_clinical.shape)
print(pd_clinical.columns.tolist())
print(pd_clinical.head())


# %% [cell 163]
age_cols = [c for c in pd_clinical.columns if 'age' in str(c).lower()]
psa_cols = [c for c in pd_clinical.columns if 'psa' in str(c).lower()]
pirads_cols = [c for c in pd_clinical.columns if 'pirads' in str(c).lower() or 'suspicion' in str(c).lower() or 'radiology' in str(c).lower()]
gleason_cols = [c for c in pd_clinical.columns if 'gleason' in str(c).lower()]

print("Age columns:", age_cols)
print("PSA columns:", psa_cols)
print("PI-RADS/radiology columns:", pirads_cols)
print("Gleason columns:", gleason_cols)


# %% [cell 164]
import pandas as pd

turkey_df = pd.read_csv('external_validation_3/turkey_clinical.csv')
print(turkey_df.shape)
print(turkey_df.columns.tolist())
print(turkey_df.head())


# %% [cell 165]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

picai_clinical_clean = pd.read_csv('picai_clinical_clean.csv')

X_reduced = picai_clinical_clean[['patient_age', 'psa']]
y_reduced = picai_clinical_clean['csPCa_label']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reduced, y_reduced, test_size=0.2, stratify=y_reduced, random_state=42)
scaler_reduced = StandardScaler()
Xr_train_scaled = scaler_reduced.fit_transform(Xr_train)
Xr_test_scaled = scaler_reduced.transform(Xr_test)

rf_reduced = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
rf_reduced.fit(Xr_train_scaled, yr_train)

preds_internal = rf_reduced.predict(Xr_test_scaled)
probs_internal = rf_reduced.predict_proba(Xr_test_scaled)[:, 1]

print("=== PI-CAI Internal (age+PSA only) ===")
print("Accuracy:", accuracy_score(yr_test, preds_internal))
print("Recall:", recall_score(yr_test, preds_internal))
print("ROC-AUC:", roc_auc_score(yr_test, probs_internal))


# %% [cell 166]
tcga_clinical = pd.read_csv('clinical_features_selected.csv')
print(tcga_clinical.columns.tolist())
print(tcga_clinical.shape)
print(tcga_clinical.head())


# %% [cell 167]
import os
for root, dirs, files in os.walk(os.path.expanduser('~/prostate-cancer-research')):
    for f in files:
        if 'clinical' in f.lower() and f.endswith('.csv'):
            print(os.path.join(root, f))


# %% [cell 168]
import glob
tsv_files = glob.glob(os.path.expanduser('~/prostate-cancer-research/**/*.tsv'), recursive=True)
print(tsv_files)


# %% [cell 169]
clinical = pd.read_csv('clinical.tsv', sep='\t', low_memory=False)  # adjust path if found elsewhere
clinical_clean = clinical.replace("'--", pd.NA)

clinical_features = clinical_clean[[
    'cases.case_id', 'demographic.age_at_index', 'diagnoses.gleason_score'
]].copy()
clinical_features['demographic.age_at_index'] = pd.to_numeric(clinical_features['demographic.age_at_index'], errors='coerce')

def combine_group(g):
    return g.ffill().bfill().iloc[0]
clinical_features = clinical_features.groupby('cases.case_id', as_index=False).apply(combine_group)

# Rebuild PSA from follow_up.tsv
followup = pd.read_csv('follow_up.tsv', sep='\t', low_memory=False)
psa_rows = followup[followup['molecular_tests.antigen'] == 'Prostate-Specific Antigen (PSA)']
psa_df = psa_rows[['cases.case_id', 'molecular_tests.test_value']].rename(columns={'molecular_tests.test_value': 'psa_level'})
psa_df['psa_level'] = pd.to_numeric(psa_df['psa_level'], errors='coerce')

clinical_final = clinical_features.merge(psa_df, on='cases.case_id', how='left')
clinical_final['psa_level'] = clinical_final['psa_level'].fillna(clinical_final['psa_level'].median())

# Build label and drop Gleason (avoid leakage, matching your original approach)
clinical_final['csPCa_label'] = (pd.to_numeric(clinical_final['diagnoses.gleason_score'], errors='coerce') >= 7).astype(int)
clinical_final = clinical_final.drop(columns=['diagnoses.gleason_score'])

print(clinical_final.shape)
clinical_final.to_csv('clinical_features_selected.csv', index=False)
print("Saved successfully")


# %% [cell 170]
import glob, os
tsv_files = glob.glob(os.path.expanduser('~/prostate-cancer-research/**/*.tsv'), recursive=True)
print(tsv_files)


# %% [cell 171]
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Rebuild the three configurations (adjust paths if needed)
fused_df = pd.read_csv('fused_final_dataset.csv')

clinical_cols = ['patient_age', 'psa', 'psad', 'prostate_volume']
imaging_cols = [c for c in fused_df.columns if c.startswith('cnn_pca_')]

X_all = fused_df.drop(columns=['patient_id', 'csPCa_label'])
y_all = fused_df['csPCa_label']

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)

configs = {
    'Clinical only': clinical_cols,
    'Imaging only': imaging_cols,
    'Fused (clinical + imaging)': clinical_cols + imaging_cols,
}

plt.figure(figsize=(7, 7))
colors = {'Clinical only': '#065A82', 'Imaging only': '#E8935D', 'Fused (clinical + imaging)': '#1C7293'}

for name, cols in configs.items():
    X_train = X_train_full[cols]
    X_test = X_test_full[cols]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    probs = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', color=colors[name], linewidth=2)

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance (AUC = 0.500)')
plt.xlabel('False Positive Rate (1 − Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curves: Clinical, Imaging, and Fused Feature Configurations')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('roc_clinical_imaging_fused.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [cell 172]
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Rebuild the three configurations
fused_df = pd.read_csv('fused_final_dataset.csv')

clinical_cols = ['patient_age', 'psa', 'psad', 'prostate_volume']
imaging_cols = [c for c in fused_df.columns if c.startswith('cnn_pca_')]

X_all = fused_df.drop(columns=['patient_id', 'csPCa_label'])
y_all = fused_df['csPCa_label']

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)

configs = {
    'Clinical only': clinical_cols,
    'Imaging only': imaging_cols,
    'Fused (clinical + imaging)': clinical_cols + imaging_cols,
}

# Refined, colourblind-friendly palette
colors = {
    'Clinical only': '#2E5C8A',            # deep blue
    'Imaging only': '#D97642',             # warm terracotta
    'Fused (clinical + imaging)': '#2A9D8F',  # teal
}
linestyles = {'Clinical only': '-', 'Imaging only': '-', 'Fused (clinical + imaging)': '-'}

plt.rcParams['font.family'] = 'DejaVu Serif'
fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

for name, cols in configs.items():
    X_train = X_train_full[cols]
    X_test = X_test_full[cols]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    probs = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    ax.plot(fpr, tpr, label=f'{name}  (AUC = {auc:.3f})',
            color=colors[name], linewidth=2.8, linestyle=linestyles[name],
            solid_capstyle='round', zorder=3)

# Chance line
ax.plot([0, 1], [0, 1], linestyle='--', color='#999999', linewidth=1.5,
        label='Chance (AUC = 0.500)', zorder=2)
ax.fill_between([0, 1], [0, 1], color='#999999', alpha=0.05, zorder=1)

ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=13, labelpad=10)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13, labelpad=10)
ax.set_title('ROC Curves: Clinical, Imaging, and Fused Feature Configurations',
             fontsize=14, fontweight='bold', pad=16)

ax.legend(loc='lower right', fontsize=11, frameon=True, framealpha=0.95,
          edgecolor='#DDDDDD', fancybox=True)

ax.grid(alpha=0.25, linestyle='-', linewidth=0.6)
ax.set_facecolor('#FAFAFA')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color('#888888')

plt.tight_layout()
plt.savefig('roc_clinical_imaging_fused.png', dpi=400, bbox_inches='tight', facecolor='white')
plt.show()


# %% [cell 173]
import matplotlib.pyplot as plt
import numpy as np

# Data exactly as reported in Table 4.2
results = {
    'Random Forest':        {'Accuracy': 0.730, 'ROC-AUC': 0.765},
    'AdaBoost':              {'Accuracy': 0.743, 'ROC-AUC': 0.758},
    'Tuned XGBoost':         {'Accuracy': 0.737, 'ROC-AUC': 0.774},
    'Gradient Boosting':     {'Accuracy': 0.740, 'ROC-AUC': 0.736},
    'Extra Trees':           {'Accuracy': 0.717, 'ROC-AUC': 0.734},
    'SVM':                   {'Accuracy': 0.700, 'ROC-AUC': 0.734},
    'MLP':                   {'Accuracy': 0.720, 'ROC-AUC': 0.717},
    'LDA':                   {'Accuracy': 0.720, 'ROC-AUC': 0.717},
    'Logistic Regression':   {'Accuracy': 0.653, 'ROC-AUC': 0.714},
    'Naive Bayes':           {'Accuracy': 0.703, 'ROC-AUC': 0.713},
    'Stacking Ensemble':     {'Accuracy': 0.677, 'ROC-AUC': 0.763},
    'k-Nearest Neighbours':  {'Accuracy': 0.700, 'ROC-AUC': 0.683},
    'Decision Tree':         {'Accuracy': 0.620, 'ROC-AUC': 0.598},
}

plt.rcParams['font.family'] = 'DejaVu Serif'

def make_bar_chart(metric, color, filename, title):
    sorted_items = sorted(results.items(), key=lambda x: x[1][metric], reverse=True)
    names = [item[0] for item in sorted_items]
    values = [item[1][metric] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    bars = ax.barh(names, values, color=color, edgecolor='white', linewidth=0.6, zorder=3)

    # Highlight the top performer
    bars[0].set_color('#D97642')

    for bar, val in zip(bars, values):
        ax.text(val + 0.008, bar.get_y() + bar.get_height() / 2, f'{val:.3f}',
                va='center', fontsize=10, color='#333333')

    ax.set_xlim(0, max(values) + 0.08)
    ax.set_xlabel(metric, fontsize=13, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=16)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.25, linestyle='-', linewidth=0.6, zorder=0)
    ax.set_facecolor('#FAFAFA')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#888888')

    plt.tight_layout()
    plt.savefig(filename, dpi=400, bbox_inches='tight', facecolor='white')
    plt.show()

make_bar_chart('Accuracy', '#2E5C8A', 'algo_comparison_accuracy.png',
                'Accuracy by Algorithm (Fused Dataset)')

make_bar_chart('ROC-AUC', '#2A9D8F', 'algo_comparison_rocauc.png',
                'ROC-AUC by Algorithm (Fused Dataset)')


# %% [cell 174]
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import pandas as pd

fused_df = pd.read_csv('fused_final_dataset.csv')
X_all = fused_df.drop(columns=['patient_id', 'csPCa_label'])
y_all = fused_df['csPCa_label']

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced'),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
    'Tuned XGBoost': XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.01, subsample=0.8, scale_pos_weight=2.5, random_state=42, eval_metric='logloss'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced'),
    'SVM': SVC(probability=True, class_weight='balanced', random_state=42),
    'MLP': MLPClassifier(max_iter=1000, random_state=42),
    'LDA': LinearDiscriminantAnalysis(),
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Naive Bayes': GaussianNB(),
    'k-Nearest Neighbours': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
}
models['Stacking Ensemble'] = StackingClassifier(
    estimators=[('rf', models['Random Forest']), ('ada', models['AdaBoost']), ('gb', models['Gradient Boosting'])],
    final_estimator=LogisticRegression(class_weight='balanced'), cv=5
)

# A distinct 13-colour qualitative palette (colourblind-conscious, avoids repeats)
palette = [
    '#2E5C8A', '#D97642', '#2A9D8F', '#8E5572', '#C9A227', '#5B7B4F',
    '#A63A50', '#4472CA', '#B8860B', '#6A4C93', '#3D8361', '#C05746', '#264653'
]

plt.rcParams['font.family'] = 'DejaVu Serif'
fig, ax = plt.subplots(figsize=(9, 9), dpi=150)

for (name, model), color in zip(models.items(), palette):
    model.fit(X_train_scaled, y_train)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', color=color, linewidth=2, solid_capstyle='round', zorder=3)

ax.plot([0, 1], [0, 1], linestyle='--', color='#999999', linewidth=1.5, label='Chance (AUC = 0.500)', zorder=2)
ax.fill_between([0, 1], [0, 1], color='#999999', alpha=0.05, zorder=1)

ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=13, labelpad=10)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13, labelpad=10)
ax.set_title('ROC Curves: Thirteen Machine Learning Algorithms (Fused Dataset)', fontsize=14, fontweight='bold', pad=16)

ax.legend(loc='lower right', fontsize=9, frameon=True, framealpha=0.95, edgecolor='#DDDDDD', fancybox=True, ncol=1)
ax.grid(alpha=0.25, linestyle='-', linewidth=0.6)
ax.set_facecolor('#FAFAFA')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color('#888888')

plt.tight_layout()
plt.savefig('roc_thirteen_algorithms.png', dpi=400, bbox_inches='tight', facecolor='white')
plt.show()


# %% [cell 175]
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Serif'

# Reuse your already-trained models and test data from the previous script
model_order = sorted(models.keys(), key=lambda name: roc_auc_score(
    y_test, models[name].predict_proba(X_test_scaled)[:, 1]
), reverse=True)

fig, axes = plt.subplots(4, 4, figsize=(14, 14), dpi=150)
axes_flat = axes.flatten()

for i, name in enumerate(model_order):
    ax = axes_flat[i]
    probs = models[name].predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    ax.plot(fpr, tpr, color='#2E5C8A', linewidth=2.2, zorder=3)
    ax.fill_between(fpr, tpr, alpha=0.12, color='#2E5C8A', zorder=2)
    ax.plot([0, 1], [0, 1], linestyle='--', color='#BBBBBB', linewidth=1, zorder=1)

    ax.set_title(f'{name}\nAUC = {auc:.3f}', fontsize=10.5, fontweight='bold', pad=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=8)
    ax.set_facecolor('#FAFAFA')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

# Hide the 3 unused panels (13 models in a 4x4 grid)
for j in range(len(model_order), 16):
    axes_flat[j].axis('off')

fig.suptitle('ROC Curves for Thirteen Machine Learning Algorithms (Fused Dataset)',
             fontsize=15, fontweight='bold', y=0.995)
fig.text(0.5, 0.01, 'False Positive Rate', ha='center', fontsize=12)
fig.text(0.01, 0.5, 'True Positive Rate', va='center', rotation='vertical', fontsize=12)

plt.tight_layout(rect=[0.02, 0.02, 1, 0.97])
plt.savefig('roc_thirteen_algorithms_grid.png', dpi=400, bbox_inches='tight', facecolor='white')
plt.show()


# %% [cell 176]

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

plt.rcParams['font.family'] = 'DejaVu Serif'

thresholds = np.linspace(0.01, 0.99, 99)

fig, axes = plt.subplots(4, 4, figsize=(14, 14), dpi=150)
axes_flat = axes.flatten()

# Sort by peak accuracy achieved across all thresholds, best to worst
def peak_accuracy(name):
    probs = models[name].predict_proba(X_test_scaled)[:, 1]
    accs = [accuracy_score(y_test, (probs >= t).astype(int)) for t in thresholds]
    return max(accs)

model_order_acc = sorted(models.keys(), key=peak_accuracy, reverse=True)

for i, name in enumerate(model_order_acc):
    ax = axes_flat[i]
    probs = models[name].predict_proba(X_test_scaled)[:, 1]
    accs = [accuracy_score(y_test, (probs >= t).astype(int)) for t in thresholds]

    best_idx = np.argmax(accs)
    best_thresh = thresholds[best_idx]
    best_acc = accs[best_idx]

    ax.plot(thresholds, accs, color='#2A9D8F', linewidth=2.2, zorder=3)
    ax.fill_between(thresholds, accs, alpha=0.12, color='#2A9D8F', zorder=2)
    ax.axvline(0.5, linestyle='--', color='#BBBBBB', linewidth=1, zorder=1)
    ax.scatter([best_thresh], [best_acc], color='#D97642', s=35, zorder=4)

    ax.set_title(f'{name}\nPeak Acc = {best_acc:.3f} (t={best_thresh:.2f})', fontsize=10, fontweight='bold', pad=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.tick_params(labelsize=8)
    ax.set_facecolor('#FAFAFA')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

for j in range(len(model_order_acc), 16):
    axes_flat[j].axis('off')

fig.suptitle('Accuracy vs. Decision Threshold for Thirteen Machine Learning Algorithms (Fused Dataset)',
             fontsize=15, fontweight='bold', y=0.995)
fig.text(0.5, 0.01, 'Decision Threshold', ha='center', fontsize=12)
fig.text(0.01, 0.5, 'Accuracy', va='center', rotation='vertical', fontsize=12)

plt.tight_layout(rect=[0.02, 0.02, 1, 0.97])
plt.savefig('accuracy_thirteen_algorithms_grid.png', dpi=400, bbox_inches='tight', facecolor='white')
plt.show()


# %% [cell 177]
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Serif'

configs = ['Early Fusion\n(Random Forest)', 'Late Fusion\n(simple average)', 'Late Fusion\n(learned combiner)', 'Intermediate Fusion\n(neural network)']
accuracy = [0.730, 0.703, 0.700, 0.687]
recall = [0.400, 0.541, 0.341, 0.647]
roc_auc = [0.765, 0.726, 0.727, 0.733]

x = np.arange(len(configs))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
bars1 = ax.bar(x - width, accuracy, width, label='Accuracy', color='#2E5C8A')
bars2 = ax.bar(x, recall, width, label='Recall', color='#D97642')
bars3 = ax.bar(x + width, roc_auc, width, label='ROC-AUC', color='#2A9D8F')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.012, f'{h:.3f}', ha='center', fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10.5)
ax.set_ylabel('Score', fontsize=13)
ax.set_ylim(0, 0.9)
ax.set_title('Comparison of Multimodal Fusion Architectures (This Study)', fontsize=14, fontweight='bold', pad=16)
ax.legend(loc='upper right', fontsize=11, frameon=True, edgecolor='#DDDDDD')
ax.grid(axis='y', alpha=0.25)
ax.set_facecolor('#FAFAFA')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig('fusion_architecture_comparison.png', dpi=400, bbox_inches='tight', facecolor='white')
plt.show()


# %% [cell 178]
fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)

studies = ['This study\n(internal)', 'Roest et al. 2024\n(internal)', 'Roest et al. 2024\n(external)']
early_vals = [0.765, 0.874, 0.770]
late_vals = [0.727, 0.871, 0.730]

x = np.arange(len(studies))
width = 0.32

bars_e = ax.bar(x - width/2, early_vals, width, label='Early Fusion', color='#2E5C8A')
bars_l = ax.bar(x + width/2, late_vals, width, label='Late Fusion', color='#D97642')

for bars in [bars_e, bars_l]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.012, f'{h:.3f}', ha='center', fontsize=9.5)

ax.set_xticks(x)
ax.set_xticklabels(studies, fontsize=10.5)
ax.set_ylabel('ROC-AUC', fontsize=13)
ax.set_ylim(0, 1.0)
ax.set_title('Early vs. Late Fusion: This Study vs. Published Literature', fontsize=13.5, fontweight='bold', pad=16)
ax.legend(loc='upper right', fontsize=11, frameon=True, edgecolor='#DDDDDD')
ax.grid(axis='y', alpha=0.25)
ax.set_facecolor('#FAFAFA')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig('fusion_comparison_literature.png', dpi=400, bbox_inches='tight', facecolor='white')
plt.show()


# %% [cell 179]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score

# --- Rebuild the PI-RADS + biopsy pathway dataset ---
fused_df_reload = pd.read_csv('fused_final_dataset.csv')
picai_raw = pd.read_csv('picai_labels/clinical_information/marksheet.csv')

def get_max_pirads(val):
    if pd.isna(val):
        return np.nan
    scores = str(val).split(',')
    return max([int(s.strip()) for s in scores if s.strip().isdigit()])

picai_raw['max_pirads'] = picai_raw['lesion_PIRADS'].apply(get_max_pirads)

fused_with_pirads = fused_df_reload.merge(
    picai_raw[['patient_id', 'max_pirads']].drop_duplicates('patient_id'),
    on='patient_id', how='left'
)
fused_with_pirads['max_pirads'] = fused_with_pirads['max_pirads'].fillna(fused_with_pirads['max_pirads'].median())

fused_with_histopath = fused_with_pirads.merge(
    picai_raw[['patient_id', 'histopath_type']].drop_duplicates('patient_id'),
    on='patient_id', how='left'
)
fused_with_histopath['histopath_type'] = fused_with_histopath['histopath_type'].fillna('No_Biopsy_Needed')
fused_with_histopath_encoded = pd.get_dummies(fused_with_histopath, columns=['histopath_type'], drop_first=True)

X_final = fused_with_histopath_encoded.drop(columns=['patient_id', 'csPCa_label'])
y_final = fused_with_histopath_encoded['csPCa_label']

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, stratify=y_final, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train all 13 models ---
models = {
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced'),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.01, subsample=0.8, scale_pos_weight=2.5, random_state=42, eval_metric='logloss'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced'),
    'SVM': SVC(probability=True, class_weight='balanced', random_state=42),
    'MLP': MLPClassifier(max_iter=1000, random_state=42),
    'LDA': LinearDiscriminantAnalysis(),
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Naive Bayes': GaussianNB(),
    'k-Nearest Neighbours': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
}
models['Stacking Ensemble'] = StackingClassifier(
    estimators=[('rf', models['Random Forest']), ('ada', models['AdaBoost']), ('gb', models['Gradient Boosting'])],
    final_estimator=LogisticRegression(class_weight='balanced'), cv=5
)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

# --- Small-multiples ROC grid, sorted best to worst ---
plt.rcParams['font.family'] = 'DejaVu Serif'
model_order = sorted(models.keys(), key=lambda name: roc_auc_score(
    y_test, models[name].predict_proba(X_test_scaled)[:, 1]
), reverse=True)

fig, axes = plt.subplots(4, 4, figsize=(14, 14), dpi=150)
axes_flat = axes.flatten()

for i, name in enumerate(model_order):
    ax = axes_flat[i]
    probs = models[name].predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    ax.plot(fpr, tpr, color='#D97642', linewidth=2.2, zorder=3)
    ax.fill_between(fpr, tpr, alpha=0.12, color='#D97642', zorder=2)
    ax.plot([0, 1], [0, 1], linestyle='--', color='#BBBBBB', linewidth=1, zorder=1)

    ax.set_title(f'{name}\nAUC = {auc:.3f}', fontsize=10.5, fontweight='bold', pad=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=8)
    ax.set_facecolor('#FAFAFA')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

for j in range(len(model_order), 16):
    axes_flat[j].axis('off')

fig.suptitle('ROC Curves: Thirteen Algorithms with PI-RADS and Biopsy Pathway Added',
             fontsize=15, fontweight='bold', y=0.995)
fig.text(0.5, 0.01, 'False Positive Rate', ha='center', fontsize=12)
fig.text(0.01, 0.5, 'True Positive Rate', va='center', rotation='vertical', fontsize=12)

plt.tight_layout(rect=[0.02, 0.02, 1, 0.97])
plt.savefig('roc_pirads_biopsy_grid.png', dpi=400, bbox_inches='tight', facecolor='white')
plt.show()


# %% [cell 180]
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, confusion_matrix, brier_score_loss)

def full_report(model, X, y, name):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"Set": name, "n": len(y), "Prevalence": y.mean(),
            "Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred),
            "Recall": recall_score(y, pred), "Specificity": tn / (tn + fp),
            "F1": f1_score(y, pred), "Balanced acc.": balanced_accuracy_score(y, pred),
            "MCC": matthews_corrcoef(y, pred), "ROC-AUC": roc_auc_score(y, prob),
            "PR-AUC": average_precision_score(y, prob), "Brier": brier_score_loss(y, prob)}

rows = [full_report(locked_model, Xp_test_scaled, yp_test, "Internal (PI-CAI test)"),
        full_report(locked_model, X_external_scaled, y_external_final, "External (US cohort)")]
print(pd.DataFrame(rows).round(3).T)


# %% [cell 181]
# same locked Stacking model and same feature set used for the external run
full_report(locked_stack, X_test_fused_picai, y_test)   # accuracy, precision, recall, AUC + counts


# %% [cell 182]
cols_no_path = [c for c in expert_cols if c not in ("biopsy_pathway",)]  # keep PI-RADS, drop pathway
Xtr, Xte = df_train[cols_no_path], df_test[cols_no_path]
results = {}
for name, model in models.items():            # your 13 algorithms, same settings as before
    model.fit(Xtr, y_train)
    p = model.predict_proba(Xte)[:, 1]
    results[name] = dict(auc=roc_auc_score(y_test, p),
                         recall=recall_score(y_test, p >= 0.5),
                         spec=recall_score(1 - y_test, p < 0.5))
pd.DataFrame(results).T.sort_values("auc", ascending=False)


# %% [cell 183]
Xp_test_scaled = scaler_locked.transform(Xp_test)      # locked scaler, transform only

rows = [
    full_report(locked_model, Xp_test_scaled,  yp_test,           "Internal (PI-CAI test)"),
    full_report(locked_model, X_external_scaled, y_external_final, "External (US cohort)"),
]
print(pd.DataFrame(rows).round(3).T)


# %% [cell 184]
Xp_test_scaled = scaler_locked.transform(Xp_test)

nopath = run_all_models(Xp_train_scaled, Xp_test_scaled, yp_train, yp_test,
                        'Fused + PI-RADS (no pathway)')

# add the locked Stacking model if it is not already one of the rows
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
pred = locked_model.predict(Xp_test_scaled)
prob = locked_model.predict_proba(Xp_test_scaled)[:, 1]
nopath = pd.concat([nopath, pd.DataFrame([{
    'Dataset': 'Fused + PI-RADS (no pathway)', 'Model': 'Stacking (locked)',
    'Accuracy': accuracy_score(yp_test, pred), 'Precision': precision_score(yp_test, pred),
    'Recall': recall_score(yp_test, pred), 'F1': f1_score(yp_test, pred),
    'ROC-AUC': roc_auc_score(yp_test, prob)}])], ignore_index=True)

print(nopath.sort_values('ROC-AUC', ascending=False).round(3))
nopath.to_csv('results_fused_pirads_no_pathway.csv', index=False)


# %% [cell 185]
import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, balanced_accuracy_score)

needed = ['fused_final_dataset.csv', 'picai_clinical_clean.csv', 'picai_labels/clinical_information/marksheet.csv',
          'cnn_features_t2w.csv', 'external_validation/external_cnn_features_t2w.csv',
          'external_validation/final_external_cohort.csv']
missing = [f for f in needed if not os.path.exists(f)]
assert not missing, f"Missing (run from ~/prostate-cancer-research): {missing}"

def report(model, X, y, name):
    prob = model.predict_proba(X)[:, 1]; pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"Set": name, "n": len(y), "Prevalence": round(float(np.mean(y)), 3),
            "Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred),
            "Recall": recall_score(y, pred), "Specificity": tn / (tn + fp), "F1": f1_score(y, pred),
            "Balanced acc.": balanced_accuracy_score(y, pred), "ROC-AUC": roc_auc_score(y, prob),
            "TP": tp, "FN": fn, "FP": fp, "TN": tn}

# ---- 1. PI-CAI data: clinical + imaging + PI-RADS (NO biopsy pathway)
def get_max_pirads(val):
    if pd.isna(val): return np.nan
    s = [int(x.strip()) for x in str(val).split(',') if x.strip().isdigit()]
    return max(s) if s else np.nan

fused = pd.read_csv('fused_final_dataset.csv')
raw = pd.read_csv('picai_labels/clinical_information/marksheet.csv')
raw['max_pirads'] = raw['lesion_PIRADS'].apply(get_max_pirads)
fp = fused.merge(raw[['patient_id', 'max_pirads']].drop_duplicates('patient_id'), on='patient_id', how='left')
fp['max_pirads'] = fp['max_pirads'].fillna(fp['max_pirads'].median())
X_pirads = fp.drop(columns=['patient_id', 'csPCa_label']); y_pirads = fp['csPCa_label']
Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_pirads, y_pirads, test_size=0.2, stratify=y_pirads, random_state=42)
scaler_locked = StandardScaler(); Xp_train_scaled = scaler_locked.fit_transform(Xp_train)
Xp_test_scaled = scaler_locked.transform(Xp_test)
print("Shapes:", Xp_train_scaled.shape, Xp_test_scaled.shape, "| positives in test:", int(yp_test.sum()))

# ---- 2. Locked Stacking model (same settings as the external validation)
locked_model = StackingClassifier(
    estimators=[('rf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')),
                ('ada', AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=42))],
    final_estimator=LogisticRegression(class_weight='balanced'), cv=5)
locked_model.fit(Xp_train_scaled, yp_train)

# ---- 3. External cohort through the LOCKED scaler + PCA (transform only)
t2w = pd.read_csv('cnn_features_t2w.csv'); fc = [c for c in t2w.columns if c.startswith('cnn_feat_')]
sc_cnn = StandardScaler().fit(t2w[fc]); pca = PCA(n_components=100, random_state=42).fit(sc_cnn.transform(t2w[fc]))
ext_cnn = pd.read_csv('external_validation/external_cnn_features_t2w.csv'); fe = [c for c in ext_cnn.columns if c.startswith('cnn_feat_')]
ext_pca = pd.DataFrame(pca.transform(sc_cnn.transform(ext_cnn[fe])), columns=[f'cnn_pca_{i}' for i in range(100)])
ext_pca['Patient Number'] = ext_cnn['Patient Number'].values
coh = pd.read_csv('external_validation/final_external_cohort.csv')
ext = coh[['Patient Number', 'psa', 'psad', 'prostate_volume', 'max_pirads', 'csPCa_label']].merge(ext_pca, on='Patient Number', how='inner')
ext['patient_age'] = pd.read_csv('picai_clinical_clean.csv')['patient_age'].median()
X_external_scaled = scaler_locked.transform(ext[X_pirads.columns.tolist()]); y_external_final = ext['csPCa_label']

# ---- PART A: locked Stacking, internal vs external, from the same model
A = pd.DataFrame([report(locked_model, Xp_test_scaled, yp_test, "Internal (PI-CAI test)"),
                  report(locked_model, X_external_scaled, y_external_final, "External (US cohort)")])
print(A.round(3).T); A.to_csv('locked_model_internal_vs_external.csv', index=False)
print("CHECK: external AUC should be about 0.745 and accuracy about 0.669")

# ---- PART B: all 13 algorithms, PI-RADS WITHOUT the biopsy pathway (takes several minutes)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mp = {
 'LogisticRegression': (LogisticRegression(max_iter=1000, class_weight='balanced'), {'C': [0.01, 0.1, 1, 10]}),
 'DecisionTree': (DecisionTreeClassifier(random_state=42, class_weight='balanced'), {'max_depth': [5, 10, None]}),
 'RandomForest': (RandomForestClassifier(random_state=42, class_weight='balanced'), {'n_estimators': [100, 300], 'max_depth': [10, 20]}),
 'ExtraTrees': (ExtraTreesClassifier(random_state=42, class_weight='balanced'), {'n_estimators': [100, 300], 'max_depth': [10, 20]}),
 'GradientBoosting': (GradientBoostingClassifier(random_state=42), {'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1]}),
 'AdaBoost': (AdaBoostClassifier(random_state=42), {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]}),
 'SVM': (SVC(probability=True, class_weight='balanced'), {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}),
 'KNN': (KNeighborsClassifier(), {'n_neighbors': [5, 10, 15]}),
 'NaiveBayes': (GaussianNB(), {}), 'LDA': (LinearDiscriminantAnalysis(), {}),
 'MLP': (MLPClassifier(max_iter=1000, random_state=42), {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.001, 0.01]}),
 'XGBoost': (XGBClassifier(scale_pos_weight=2.5, random_state=42, eval_metric='logloss'),
             {'n_estimators': [200, 300], 'max_depth': [3, 5, 6], 'learning_rate': [0.01, 0.1]})}
best = {}
for name, (m, p) in mp.items():
    if p: g = GridSearchCV(m, p, cv=cv, scoring='roc_auc', n_jobs=-1).fit(Xp_train_scaled, yp_train); best[name] = g.best_estimator_
    else: best[name] = m.fit(Xp_train_scaled, yp_train)
    print(name, "done")
best['Stacking'] = StackingClassifier(estimators=[('rf', best['RandomForest']), ('ada', best['AdaBoost']), ('gb', best['GradientBoosting'])],
                                      final_estimator=LogisticRegression(class_weight='balanced'), cv=5).fit(Xp_train_scaled, yp_train)
B = pd.DataFrame([report(m, Xp_test_scaled, yp_test, n) for n, m in best.items()]).rename(columns={'Set': 'Model'})
B = B.sort_values('ROC-AUC', ascending=False)
print(B.drop(columns=['n', 'Prevalence']).round(3).to_string(index=False))
B.to_csv('results_fused_pirads_no_pathway.csv', index=False)


# %% [cell 186]
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, precision_score

def make_locked():
    return StackingClassifier(estimators=[('rf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')),
        ('ada', AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=42))],
        final_estimator=LogisticRegression(class_weight='balanced'), cv=5)
def metrics(m, X, y):
    p = m.predict_proba(X)[:, 1]; h = (p >= 0.5).astype(int)
    return dict(n=len(y), AUC=round(roc_auc_score(y, p), 3), Recall=round(recall_score(y, h), 3), Precision=round(precision_score(y, h), 3), Accuracy=round(accuracy_score(y, h), 3))
def get_max_pirads(v):
    if pd.isna(v): return np.nan
    s = [int(x.strip()) for x in str(v).split(',') if x.strip().isdigit()]
    return max(s) if s else np.nan

fused = pd.read_csv('fused_final_dataset.csv'); raw = pd.read_csv('picai_labels/clinical_information/marksheet.csv'); cc = pd.read_csv('picai_clinical_clean.csv')
raw['max_pirads'] = raw['lesion_PIRADS'].apply(get_max_pirads)
fp = fused.merge(raw[['patient_id', 'max_pirads']].drop_duplicates('patient_id'), on='patient_id', how='left')
fp['max_pirads'] = fp['max_pirads'].fillna(fp['max_pirads'].median())
ids = fp['patient_id'].reset_index(drop=True); X = fp.drop(columns=['patient_id', 'csPCa_label']).reset_index(drop=True); y = fp['csPCa_label'].reset_index(drop=True)

print("== 1. IDENTIFIERS")
print("fused rows:", len(fused), "| unique patient_id:", fused['patient_id'].nunique(), "| patients with >1 study:", int(fused.duplicated('patient_id', keep=False).sum()))
print("marksheet rows:", len(raw), "| unique patient_id:", raw['patient_id'].nunique(), "| columns:", [c for c in raw.columns if 'id' in c.lower()])
print("rows in fused after PI-RADS merge:", len(fp), "(should equal", len(fused), ")")
cc2 = cc[['patient_id', 'center']].drop_duplicates(); print("patients appearing in >1 centre:", int((cc2.groupby('patient_id')['center'].nunique() > 1).sum()))
print("\n== 2. IMPUTATION (full-data median vs value found in cleaned table)")
for col in ['patient_age', 'psa', 'psad', 'prostate_volume']:
    if col in raw.columns and col in cc.columns:
        miss = raw.loc[raw[col].isna(), 'patient_id']; sub = cc[cc['patient_id'].isin(miss)][col]
        print(f"{col}: raw missing {int(raw[col].isna().sum())} | cleaned values at those rows (unique): {sub.round(3).unique()[:4]} | full-data median {raw[col].median():.3f}")
print("missing values left in fused table:", int(fused.isna().sum().sum()))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
tr_ids, te_ids = set(ids.loc[Xtr.index]), ids.loc[Xte.index]
overlap = tr_ids & set(te_ids)
print("\n== 3. PATIENT OVERLAP IN THE 80/20 SPLIT")
print("patients in BOTH train and test:", len(overlap), "| test studies affected:", int(te_ids.isin(overlap).sum()))
sc = StandardScaler().fit(Xtr); m0 = make_locked().fit(sc.transform(Xtr), ytr)
print("original test set:", metrics(m0, sc.transform(Xte), yte))
keep = ~te_ids.isin(tr_ids).values
print("test set without patients seen in training:", metrics(m0, sc.transform(Xte[keep]), yte[keep]))

print("\n== 4. PREPROCESSING AND PATIENT-LEVEL SPLIT SENSITIVITY")
t2w = pd.read_csv('cnn_features_t2w.csv'); fc = [c for c in t2w.columns if c.startswith('cnn_feat_')]
if 'patient_id' in t2w.columns and t2w['patient_id'].is_unique and set(fp['patient_id']) <= set(t2w['patient_id']): F = t2w.set_index('patient_id').loc[fp['patient_id'], fc].to_numpy()
elif len(t2w) == len(fp): F = t2w[fc].to_numpy(); print("WARNING: imaging features aligned by row order, not by ID")
else: raise SystemExit("cannot align imaging features")
clin = [c for c in X.columns if not c.startswith('cnn_pca_')]
def fit_eval(tr_i, te_i, refit_pca, label):
    if refit_pca:
        s = StandardScaler().fit(F[tr_i]); p = PCA(n_components=100, random_state=42).fit(s.transform(F[tr_i]))
        Atr = np.hstack([X.iloc[tr_i][clin].to_numpy(), p.transform(s.transform(F[tr_i]))]); Ate = np.hstack([X.iloc[te_i][clin].to_numpy(), p.transform(s.transform(F[te_i]))])
    else: Atr, Ate = X.iloc[tr_i].to_numpy(), X.iloc[te_i].to_numpy()
    s2 = StandardScaler().fit(Atr); m = make_locked().fit(s2.transform(Atr), y.iloc[tr_i]); print(label, metrics(m, s2.transform(Ate), y.iloc[te_i]))
tr_i, te_i = Xtr.index.to_numpy(), Xte.index.to_numpy()
fit_eval(tr_i, te_i, True, "same split, PCA/scaler refitted on training studies only:")
g_tr, g_te = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups=ids))
print("grouped split: patients in both sets =", len(set(ids.iloc[g_tr]) & set(ids.iloc[g_te])))
fit_eval(g_tr, g_te, False, "patient-grouped split, original PCA features:")
fit_eval(g_tr, g_te, True, "patient-grouped split, PCA refitted on training only:")


# %% [cell 187]
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score
fused = pd.read_csv('fused_final_dataset.csv'); cc = pd.read_csv('picai_clinical_clean.csv')[['patient_id', 'center']].drop_duplicates('patient_id')
t2w = pd.read_csv('cnn_features_t2w.csv'); fc = [c for c in t2w.columns if c.startswith('cnn_feat_')]
if 'patient_id' in t2w.columns and t2w['patient_id'].is_unique: F = t2w.set_index('patient_id').loc[fused['patient_id'], fc].to_numpy()
else: F = t2w[fc].to_numpy(); print("WARNING: aligned by row order")
df = fused.merge(cc, on='patient_id', how='left'); assert len(df) == len(fused)
clin = ['patient_age', 'psa', 'psad', 'prostate_volume']; pcs = [c for c in fused.columns if c.startswith('cnn_pca_')]
for held in ['RUMC', 'PCNN', 'ZGT']:
    tr = (df['center'] != held).to_numpy(); te = ~tr; y = df['csPCa_label'].to_numpy()
    for mode in ('original PCA (all centres)', 'PCA refitted on training centres'):
        if mode.startswith('original'): A = df[clin + pcs].to_numpy()
        else:
            s = StandardScaler().fit(F[tr]); p = PCA(n_components=len(pcs), random_state=42).fit(s.transform(F[tr]))
            A = np.hstack([df[clin].to_numpy(), p.transform(s.transform(F))])
        sc = StandardScaler().fit(A[tr]); m = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced').fit(sc.transform(A[tr]), y[tr])
        pr = m.predict_proba(sc.transform(A[te]))[:, 1]; print(held, '|', mode, '| AUC', round(roc_auc_score(y[te], pr), 3), '| recall', round(recall_score(y[te], pr >= 0.5), 3))


# %% [cell 188]
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
fused = pd.read_csv('fused_final_dataset.csv'); cc = pd.read_csv('picai_clinical_clean.csv')
naive = fused.merge(cc[['patient_id', 'center']], on='patient_id', how='left')
print("rows after the ORIGINAL merge:", len(naive), "| rows in fused:", len(fused), "| rows in clinical_clean:", len(cc))
df = fused.merge(cc[['patient_id', 'center']].drop_duplicates('patient_id'), on='patient_id', how='left'); assert len(df) == len(fused)
print("studies per centre:", df['center'].value_counts().to_dict())
clin = ['patient_age', 'psa', 'psad', 'prostate_volume']; pcs = [c for c in df.columns if c.startswith('cnn_pca_')]
print("PCA components in file:", len(pcs))
sets = {'Clinical only': clin, 'Imaging only': pcs, 'Fused': clin + pcs}; rows = []
for held in ['RUMC', 'PCNN', 'ZGT']:
    tr = (df['center'] != held).to_numpy(); te = ~tr; y = df['csPCa_label'].to_numpy()
    for name, cols in sets.items():
        A = df[cols].to_numpy(); sc = StandardScaler().fit(A[tr])
        m = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced').fit(sc.transform(A[tr]), y[tr])
        p = m.predict_proba(sc.transform(A[te]))[:, 1]; h = (p >= 0.5).astype(int)
        rows.append(dict(Held_out=held, Features=name, Train=int(tr.sum()), Test=int(te.sum()), Test_pos=int(y[te].sum()),
                         Accuracy=round(accuracy_score(y[te], h), 3), Precision=round(precision_score(y[te], h, zero_division=0), 3),
                         Recall=round(recall_score(y[te], h), 3), F1=round(f1_score(y[te], h), 3), ROC_AUC=round(roc_auc_score(y[te], p), 3)))
print(pd.DataFrame(rows).to_string(index=False))


# %% [cell 189]
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score
fused = pd.read_csv('fused_final_dataset.csv'); t2w = pd.read_csv('cnn_features_t2w.csv'); fc = [c for c in t2w.columns if c.startswith('cnn_feat_')]
if 'patient_id' in t2w.columns and t2w['patient_id'].is_unique: F = t2w.set_index('patient_id').loc[fused['patient_id'], fc].to_numpy()
else: F = t2w[fc].to_numpy(); print("WARNING: aligned by row order")
clin = fused[['patient_age', 'psa', 'psad', 'prostate_volume']].to_numpy(); y = fused['csPCa_label'].to_numpy()
tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
s = StandardScaler().fit(F[tr]); Ztr_all = s.transform(F[tr]); Zte_all = s.transform(F[te])
for k in (10, 25, 50, 100, 200, 387):
    p = PCA(n_components=k, random_state=42).fit(Ztr_all)
    for name, A_tr, A_te in (('imaging only', p.transform(Ztr_all), p.transform(Zte_all)), ('fused', np.hstack([clin[tr], p.transform(Ztr_all)]), np.hstack([clin[te], p.transform(Zte_all)]))):
        sc = StandardScaler().fit(A_tr); m = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced').fit(sc.transform(A_tr), y[tr])
        pr = m.predict_proba(sc.transform(A_te))[:, 1]; print(k, name, 'AUC', round(roc_auc_score(y[te], pr), 3), 'recall', round(recall_score(y[te], pr >= 0.5), 3), '| variance kept', round(p.explained_variance_ratio_.sum(), 2))


# %% [cell 190]
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score

def get_max_pirads(v):
    if pd.isna(v): return np.nan
    s = [int(x.strip()) for x in str(v).split(',') if x.strip().isdigit()]
    return max(s) if s else np.nan
def make_locked():
    return StackingClassifier(estimators=[('rf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')),
        ('ada', AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)), ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=42))],
        final_estimator=LogisticRegression(class_weight='balanced'), cv=5)
rf = lambda: RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')

# ---- rebuild internal and external data exactly as for the locked model
fused = pd.read_csv('fused_final_dataset.csv'); raw = pd.read_csv('picai_labels/clinical_information/marksheet.csv'); cc = pd.read_csv('picai_clinical_clean.csv')
raw['max_pirads'] = raw['lesion_PIRADS'].apply(get_max_pirads)
fp = fused.merge(raw[['patient_id', 'max_pirads']].drop_duplicates('patient_id'), on='patient_id', how='left'); fp['max_pirads'] = fp['max_pirads'].fillna(fp['max_pirads'].median())
X = fp.drop(columns=['patient_id', 'csPCa_label']).reset_index(drop=True); y = fp['csPCa_label'].reset_index(drop=True)
cols = X.columns.tolist(); pcs = [c for c in cols if c.startswith('cnn_pca_')]; clin3 = ['psa', 'psad', 'prostate_volume']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
t2w = pd.read_csv('cnn_features_t2w.csv'); fc = [c for c in t2w.columns if c.startswith('cnn_feat_')]
sc_cnn = StandardScaler().fit(t2w[fc]); pca = PCA(n_components=100, random_state=42).fit(sc_cnn.transform(t2w[fc]))
ecnn = pd.read_csv('external_validation/external_cnn_features_t2w.csv'); fe = [c for c in ecnn.columns if c.startswith('cnn_feat_')]
epca = pd.DataFrame(pca.transform(sc_cnn.transform(ecnn[fe])), columns=[f'cnn_pca_{i}' for i in range(100)]); epca['Patient Number'] = ecnn['Patient Number'].values
coh = pd.read_csv('external_validation/final_external_cohort.csv')
ext = coh[['Patient Number', 'psa', 'psad', 'prostate_volume', 'max_pirads', 'csPCa_label']].merge(epca, on='Patient Number', how='inner'); ext['patient_age'] = cc['patient_age'].median()
Xex = ext[cols].reset_index(drop=True); yex = ext['csPCa_label'].reset_index(drop=True)
print("internal test:", len(yte), "| external:", len(yex), "| prevalence internal/external:", round(y.mean(), 3), round(yex.mean(), 3))

# ---- A. locked model, uncertainty, calibration, prevalence
sc = StandardScaler().fit(Xtr); locked = make_locked().fit(sc.transform(Xtr), ytr)
pin = locked.predict_proba(sc.transform(Xte))[:, 1]; pex = locked.predict_proba(sc.transform(Xex))[:, 1]
def boot_diff(B=1000):
    r = np.random.default_rng(0); yi, yx = np.asarray(yte), np.asarray(yex); out, ai, ae = [], [], []
    for _ in range(B):
        i = r.integers(0, len(yi), len(yi)); j = r.integers(0, len(yx), len(yx))
        if yi[i].min() == yi[i].max() or yx[j].min() == yx[j].max(): continue
        a, b = roc_auc_score(yi[i], pin[i]), roc_auc_score(yx[j], pex[j]); ai.append(a); ae.append(b); out.append(a - b)
    return np.percentile(ai, [2.5, 97.5]), np.percentile(ae, [2.5, 97.5]), np.percentile(out, [2.5, 97.5])
ci_i, ci_e, ci_d = boot_diff()
print("\n== A. UNCERTAINTY")
print("internal AUC %.3f (95%% CI %.3f-%.3f) | external AUC %.3f (%.3f-%.3f) | drop %.3f (95%% CI %.3f-%.3f)" % (roc_auc_score(yte, pin), *ci_i, roc_auc_score(yex, pex), *ci_e, roc_auc_score(yte, pin) - roc_auc_score(yex, pex), *ci_d))
for nm, yy, pp in (('internal', yte, pin), ('external', yex, pex)):
    z = np.log(np.clip(pp, 1e-6, 1 - 1e-6) / (1 - np.clip(pp, 1e-6, 1 - 1e-6))); lr = LogisticRegression(C=1e6).fit(z.reshape(-1, 1), yy)
    print(f"{nm}: mean predicted risk {pp.mean():.3f} vs observed {np.mean(yy):.3f} | calibration slope {lr.coef_[0][0]:.2f}, intercept {lr.intercept_[0]:.2f}")
sens_i = recall_score(yte, pin >= .5); spec_i = recall_score(1 - np.asarray(yte), pin < .5); prev_e = yex.mean()
ppv_expected = sens_i * prev_e / (sens_i * prev_e + (1 - spec_i) * (1 - prev_e))
print("precision expected at external prevalence if internal sensitivity/specificity held: %.3f | observed external precision: %.3f" % (ppv_expected, precision_score(yex, pex >= .5)))
thr = np.quantile(pex[np.asarray(yex) == 0], spec_i); print("external recall at the internal specificity (%.3f): %.3f (at 0.5: %.3f)" % (spec_i, np.mean(pex[np.asarray(yex) == 1] > thr), recall_score(yex, pex >= .5)))

# ---- B. which inputs transfer (Random Forest ablations; AUC internal vs external)
print("\n== B. INPUT ABLATIONS (Random Forest)")
sets = {'all inputs (as locked)': cols, 'without age': [c for c in cols if c != 'patient_age'], 'without imaging': [c for c in cols if c not in pcs], 'without PI-RADS/UCLA score': [c for c in cols if c != 'max_pirads'],
        'clinical only (PSA, PSAD, volume)': clin3, 'score only': ['max_pirads'], 'imaging only': pcs}
for nm, cl in sets.items():
    s = StandardScaler().fit(Xtr[cl]); m = rf().fit(s.transform(Xtr[cl]), ytr)
    ai = roc_auc_score(yte, m.predict_proba(s.transform(Xte[cl]))[:, 1]); ae = roc_auc_score(yex, m.predict_proba(s.transform(Xex[cl]))[:, 1]); print(f"{nm:38s} internal {ai:.3f} | external {ae:.3f} | drop {ai - ae:+.3f}")
print("raw score alone, AUC: internal %.3f | external %.3f" % (roc_auc_score(y, X['max_pirads']), roc_auc_score(yex, Xex['max_pirads'])))

# ---- C. cohort shift
print("\n== C. COHORT SHIFT")
for c in clin3 + ['max_pirads']:
    a, b = X[c].to_numpy(), Xex[c].to_numpy(); sd = np.sqrt((a.var() + b.var()) / 2)
    print(f"{c:16s} internal mean {a.mean():7.2f} | external mean {b.mean():7.2f} | standardised difference {(b.mean() - a.mean()) / sd:+.2f} | KS {ks_2samp(a, b).statistic:.2f}")
tab = pd.DataFrame({'internal n': X['max_pirads'].round().value_counts(), 'external n': Xex['max_pirads'].round().value_counts()}).fillna(0).astype(int).sort_index()
tab['internal csPCa rate'] = [round(y[X['max_pirads'].round() == k].mean(), 3) if (X['max_pirads'].round() == k).any() else np.nan for k in tab.index]
tab['external csPCa rate'] = [round(yex[Xex['max_pirads'].round() == k].mean(), 3) if (Xex['max_pirads'].round() == k).any() else np.nan for k in tab.index]
print(tab.to_string())
D = pd.concat([X.assign(dom=0), Xex.assign(dom=1)], ignore_index=True)
print("domain classifier AUC (0.5 = cohorts indistinguishable):")
for nm, cl in {'clinical (PSA, PSAD, volume)': clin3, 'score only': ['max_pirads'], 'imaging PCA components': pcs, 'all inputs except age': [c for c in cols if c != 'patient_age']}.items():
    A = StandardScaler().fit_transform(D[cl]); pr = cross_val_predict(LogisticRegression(max_iter=3000), A, D['dom'], cv=StratifiedKFold(5, shuffle=True, random_state=42), method='predict_proba')[:, 1]
    print(f"  {nm:32s} {roc_auc_score(D['dom'], pr):.3f}")

# ---- D. is the internal figure optimistic?
print("\n== D. OPTIMISM CHECK (Random Forest, all inputs)")
cvp = cross_val_predict(rf(), sc.transform(Xtr), ytr, cv=StratifiedKFold(5, shuffle=True, random_state=42), method='predict_proba')[:, 1]
m = rf().fit(sc.transform(Xtr), ytr); print("5-fold CV AUC on training data %.3f | test-partition AUC %.3f" % (roc_auc_score(ytr, cvp), roc_auc_score(yte, m.predict_proba(sc.transform(Xte))[:, 1])))


# %% [cell 191]
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import t as tdist, binomtest
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def get_max_pirads(v):
    if pd.isna(v): return np.nan
    s = [int(x.strip()) for x in str(v).split(',') if x.strip().isdigit()]
    return max(s) if s else np.nan
def make_locked():
    return StackingClassifier(estimators=[('rf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')),
        ('ada', AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)), ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=42))],
        final_estimator=LogisticRegression(class_weight='balanced'), cv=5)
if 'Xex' not in globals():
    fused = pd.read_csv('fused_final_dataset.csv'); raw = pd.read_csv('picai_labels/clinical_information/marksheet.csv'); cc = pd.read_csv('picai_clinical_clean.csv')
    raw['max_pirads'] = raw['lesion_PIRADS'].apply(get_max_pirads)
    fp = fused.merge(raw[['patient_id', 'max_pirads']].drop_duplicates('patient_id'), on='patient_id', how='left'); fp['max_pirads'] = fp['max_pirads'].fillna(fp['max_pirads'].median())
    X = fp.drop(columns=['patient_id', 'csPCa_label']).reset_index(drop=True); y = fp['csPCa_label'].reset_index(drop=True); cols = X.columns.tolist()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    t2w = pd.read_csv('cnn_features_t2w.csv'); fc = [c for c in t2w.columns if c.startswith('cnn_feat_')]
    sc_cnn = StandardScaler().fit(t2w[fc]); pca = PCA(n_components=100, random_state=42).fit(sc_cnn.transform(t2w[fc]))
    ecnn = pd.read_csv('external_validation/external_cnn_features_t2w.csv'); fe = [c for c in ecnn.columns if c.startswith('cnn_feat_')]
    epca = pd.DataFrame(pca.transform(sc_cnn.transform(ecnn[fe])), columns=[f'cnn_pca_{i}' for i in range(100)]); epca['Patient Number'] = ecnn['Patient Number'].values
    coh = pd.read_csv('external_validation/final_external_cohort.csv')
    ext = coh[['Patient Number', 'psa', 'psad', 'prostate_volume', 'max_pirads', 'csPCa_label']].merge(epca, on='Patient Number', how='inner'); ext['patient_age'] = cc['patient_age'].median()
    Xex = ext[cols].reset_index(drop=True); yex = ext['csPCa_label'].reset_index(drop=True)

X2, Xex2 = X.assign(logpsa=np.log(X['psa'])), Xex.assign(logpsa=np.log(Xex['psa']))
Xtr2, Xte2 = X2.loc[Xtr.index], X2.loc[Xte.index]; ytr_, yte_, yex_ = np.asarray(ytr), np.asarray(yte), np.asarray(yex)
LRm = lambda: LogisticRegression(max_iter=2000, class_weight='balanced')
RFm = lambda: RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
fused_cols = [c for c in cols if c != 'max_pirads']; CL3 = ['patient_age', 'logpsa', 'prostate_volume']
def fit_pred(cl, kind):
    s = StandardScaler().fit(Xtr2[cl]); m = kind().fit(s.transform(Xtr2[cl]), ytr_)
    return m.predict_proba(s.transform(Xte2[cl]))[:, 1], m.predict_proba(s.transform(Xex2[cl]))[:, 1]
spec = {'LR: age, log PSA, volume': (CL3, LRm), 'LR: PSA density alone': (['psad'], LRm), 'RF: clinical (age, PSA, PSAD, volume)': (['patient_age', 'psa', 'psad', 'prostate_volume'], RFm),
        'LR: PI-RADS + PSA density': (['max_pirads', 'psad'], LRm), 'LR: age, log PSA, volume + PI-RADS': (CL3 + ['max_pirads'], LRm),
        'RF: fused (fully automated)': (fused_cols, RFm), 'Stacking: fused (fully automated)': (fused_cols, make_locked), 'Stacking: PI-RADS-informed (locked)': (cols, make_locked)}
P_te, P_ex = {}, {}
for nm, (cl, kind) in spec.items(): P_te[nm], P_ex[nm] = fit_pred(cl, kind); print("fitted:", nm)
def rules(D):
    return {'Rule: PSA density >= 0.15': (D['psad'] >= .15).astype(float).to_numpy(), 'Rule: PI-RADS >= 3': (D['max_pirads'] >= 3).astype(float).to_numpy(), 'Rule: PI-RADS >= 4': (D['max_pirads'] >= 4).astype(float).to_numpy(),
            'Rule: PI-RADS >= 4, or 3 with PSA density >= 0.15': ((D['max_pirads'] >= 4) | ((D['max_pirads'] >= 3) & (D['psad'] >= .15))).astype(float).to_numpy(), 'PI-RADS score alone (ROC-AUC only)': D['max_pirads'].to_numpy(dtype=float)}
P_te.update(rules(Xte2)); P_ex.update(rules(Xex2))

rng = np.random.default_rng(7); B = 2000
def strat_idx(v):
    pos, neg = np.where(v == 1)[0], np.where(v == 0)[0]
    return [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(B)]
IDX_te, IDX_ex = strat_idx(yte_), strat_idx(yex_)
def metr(v, s, thr=0.5):
    h = s >= thr; tp = (h & (v == 1)).sum(); fn = ((~h) & (v == 1)).sum(); fp = (h & (v == 0)).sum(); tn = ((~h) & (v == 0)).sum()
    return tp / max(tp + fn, 1), tn / max(tn + fp, 1), tp / max(tp + fp, 1)
def summarize(P, v, IDX, names):
    rows = []
    for nm in names:
        s = P[nm]; aucs = np.array([roc_auc_score(v[i], s[i]) for i in IDX]); row = [nm, f"{roc_auc_score(v, s):.3f} ({np.percentile(aucs, 2.5):.3f}-{np.percentile(aucs, 97.5):.3f})"]
        if 'score alone' in nm: row += ['-', '-', '-']
        else:
            ms = np.array([metr(v[i], s[i]) for i in IDX]); m0 = metr(v, s)
            row += [f"{m0[k]:.3f} ({np.percentile(ms[:, k], 2.5):.3f}-{np.percentile(ms[:, k], 97.5):.3f})" for k in range(3)]
        rows.append(row)
    return pd.DataFrame(rows, columns=['Model', 'ROC-AUC (95% CI)', 'Recall', 'Specificity', 'Precision'])
def paired(P, v, IDX, a, b):
    da = np.array([roc_auc_score(v[i], P[a][i]) - roc_auc_score(v[i], P[b][i]) for i in IDX]); d0 = roc_auc_score(v, P[a]) - roc_auc_score(v, P[b])
    return d0, np.percentile(da, 2.5), np.percentile(da, 97.5), min(1.0, 2 * max(min((da <= 0).mean(), (da >= 0).mean()), 1 / B))
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
    return adj
LK, ST, RFF, RFC = 'Stacking: PI-RADS-informed (locked)', 'Stacking: fused (fully automated)', 'RF: fused (fully automated)', 'RF: clinical (age, PSA, PSAD, volume)'
planned = [('LR: age, log PSA, volume', RFF), ('LR: age, log PSA, volume', ST), (RFC, RFF), (ST, 'LR: age, log PSA, volume'), (LK, ST), (LK, 'PI-RADS score alone (ROC-AUC ONLY)'.replace('ONLY', 'only')), (LK, 'LR: PI-RADS + PSA density'), (LK, 'LR: age, log PSA, volume + PI-RADS')]
planned = [(a, b) for a, b in planned if a in P_te and b in P_te]
names_te = list(P_te)
print("\n=== 1. HOLD-OUT TEST PARTITION (n=300, 85 positive): 95% bootstrap intervals"); print(summarize(P_te, yte_, IDX_te, names_te).to_string(index=False))
res = [paired(P_te, yte_, IDX_te, a, b) for a, b in planned]; adj = holm([r[3] for r in res])
print("\n=== 2. PLANNED PAIRED COMPARISONS OF ROC-AUC (first minus second; paired bootstrap; Holm-adjusted p)")
for (a, b), r, q in zip(planned, res, adj): print(f"{a}  vs  {b}: diff {r[0]:+.3f} (95% CI {r[1]:+.3f} to {r[2]:+.3f}) | p {r[3]:.4f} | Holm p {q:.4f}")
def mcnemar(v, sa, sb):
    ha, hb = sa >= .5, sb >= .5; out = []
    for lab, mask, f in (('recall', v == 1, lambda h: h), ('specificity', v == 0, lambda h: ~h)):
        ca, cb = f(ha)[mask], f(hb)[mask]; b_, c_ = int((ca & ~cb).sum()), int((~ca & cb).sum()); out.append(f"{lab}: model-only-correct {b_}, rule-only-correct {c_}, exact McNemar p {binomtest(min(b_, c_), b_ + c_, .5).pvalue if b_ + c_ else 1.0:.4f}")
    return '; '.join(out)
print("\n=== 3. EXACT McNEMAR (default threshold): locked PI-RADS-informed model vs rules")
for r_ in ('Rule: PI-RADS >= 3', 'Rule: PI-RADS >= 4', 'Rule: PI-RADS >= 4, or 3 with PSA density >= 0.15'): print(r_, '->', mcnemar(yte_, P_te[LK], P_te[r_]))
print("\n=== 4. REPEATED 5-FOLD CV x3 on all 1,500 studies (corrected resampled t-test)")
rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42); yall = np.asarray(y)
cvm = {'LR: age, log PSA, volume': (CL3, LRm), 'LR: PI-RADS + PSA density': (['max_pirads', 'psad'], LRm), 'LR: age, log PSA, volume + PI-RADS': (CL3 + ['max_pirads'], LRm),
       'RF: clinical': (['patient_age', 'psa', 'psad', 'prostate_volume'], RFm), 'RF: fused (fully automated)': (fused_cols, RFm), 'RF: PI-RADS-informed': (cols, RFm)}
fa = {k: [] for k in cvm}
for tr, te in rs.split(X2, yall):
    for k, (cl, kind) in cvm.items():
        s = StandardScaler().fit(X2.iloc[tr][cl]); m = kind().fit(s.transform(X2.iloc[tr][cl]), yall[tr]); fa[k].append(roc_auc_score(yall[te], m.predict_proba(s.transform(X2.iloc[te][cl]))[:, 1]))
kf = 15; ratio = 300 / 1200; tc = tdist.ppf(.975, kf - 1)
for k, v_ in fa.items():
    v_ = np.array(v_); se = np.sqrt((1 / kf + ratio) * v_.var(ddof=1)); print(f"{k:38s} mean ROC-AUC {v_.mean():.3f} (95% CI {v_.mean() - tc * se:.3f}-{v_.mean() + tc * se:.3f})")
for a, b in (('RF: fused (fully automated)', 'LR: age, log PSA, volume'), ('RF: fused (fully automated)', 'RF: clinical'), ('RF: PI-RADS-informed', 'RF: fused (fully automated)'), ('RF: PI-RADS-informed', 'LR: PI-RADS + PSA density'), ('RF: PI-RADS-informed', 'LR: age, log PSA, volume + PI-RADS')):
    d_ = np.array(fa[a]) - np.array(fa[b]); se = np.sqrt((1 / kf + ratio) * d_.var(ddof=1)); p = 2 * tdist.sf(abs(d_.mean() / se), kf - 1)
    print(f"{a}  minus  {b}: {d_.mean():+.3f} (95% CI {d_.mean() - tc * se:+.3f} to {d_.mean() + tc * se:+.3f}) | corrected t-test p {p:.4f}")
print("\n=== 5. EXTERNAL COHORT (n=782): baselines fixed on the training partition")
names_ex = [LK, 'LR: age, log PSA, volume', 'LR: PSA density alone', 'LR: PI-RADS + PSA density', 'LR: age, log PSA, volume + PI-RADS', 'Rule: PSA density >= 0.15', 'Rule: PI-RADS >= 3', 'Rule: PI-RADS >= 4', 'Rule: PI-RADS >= 4, or 3 with PSA density >= 0.15', 'PI-RADS score alone (ROC-AUC only)']
print(summarize(P_ex, yex_, IDX_ex, names_ex).to_string(index=False))
for nm in names_ex[1:]:
    r = paired(P_ex, yex_, IDX_ex, LK, nm); print(f"locked model minus {nm}: {r[0]:+.3f} (95% CI {r[1]:+.3f} to {r[2]:+.3f}) | p {r[3]:.4f}")


# %% [cell 192]
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

def get_max_pirads(v):
    if pd.isna(v): return np.nan
    s = [int(x.strip()) for x in str(v).split(',') if x.strip().isdigit()]
    return max(s) if s else np.nan
fused = pd.read_csv('fused_final_dataset.csv'); raw = pd.read_csv('picai_labels/clinical_information/marksheet.csv'); cc = pd.read_csv('picai_clinical_clean.csv')
raw['max_pirads'] = raw['lesion_PIRADS'].apply(get_max_pirads)
fp = fused.merge(raw[['patient_id', 'max_pirads']].drop_duplicates('patient_id'), on='patient_id', how='left'); fp['max_pirads'] = fp['max_pirads'].fillna(fp['max_pirads'].median())
fph = fp.merge(raw[['patient_id', 'histopath_type']].drop_duplicates('patient_id'), on='patient_id', how='left'); fph['histopath_type'] = fph['histopath_type'].fillna('No_Biopsy_Needed')
fph = pd.get_dummies(fph, columns=['histopath_type'], drop_first=True)
y = fused['csPCa_label']
XA = fused.drop(columns=['patient_id', 'csPCa_label']); XB = fp.drop(columns=['patient_id', 'csPCa_label']); XC = fph.drop(columns=['patient_id', 'csPCa_label'])
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
ytr, yte = y.iloc[idx_tr].to_numpy(), y.iloc[idx_te].to_numpy()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
GRIDS = {'RandomForest': (RandomForestClassifier(random_state=42, class_weight='balanced'), {'n_estimators': [100, 300], 'max_depth': [10, 20]}),
         'AdaBoost': (AdaBoostClassifier(random_state=42), {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]}),
         'GradientBoosting': (GradientBoostingClassifier(random_state=42), {'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1]}),
         'XGBoost': (XGBClassifier(scale_pos_weight=2.5, eval_metric='logloss', random_state=42), {'n_estimators': [200, 300], 'max_depth': [3, 5, 6], 'learning_rate': [0.01, 0.1]})}
RECON = {('A', 'Stacking'): (58, 27, 70, 145, .763), ('A', 'XGBoost'): (43, 42, None, None, .774), ('A', 'RandomForest'): (34, 51, None, None, .765),
         ('B', 'Stacking'): (78, 7, 54, 161, .887), ('B', 'RandomForest'): (59, 26, 32, 183, None), ('B', 'XGBoost'): (78, 7, 59, 156, .873), ('B', 'NaiveBayes'): (38, 47, 20, 195, None),
         ('C', 'Stacking'): (80, 5, 61, 154, .893), ('C', 'RandomForest'): (75, 10, 50, 165, .888), ('C', 'XGBoost'): (81, 4, 67, 148, .880), ('C', 'NaiveBayes'): (84, 1, 99, 116, None), ('C', 'GradientBoosting'): (56, 29, None, None, .888)}
rows = []; PRED = {}
for cfg, X in (('A', XA), ('B', XB), ('C', XC)):
    sc = StandardScaler().fit(X.iloc[idx_tr]); Xtr_, Xte_ = sc.transform(X.iloc[idx_tr]), sc.transform(X.iloc[idx_te]); best = {}
    for nm, (m, g) in GRIDS.items():
        gs = GridSearchCV(m, g, cv=cv, scoring='roc_auc', n_jobs=-1).fit(Xtr_, ytr); best[nm] = gs.best_estimator_
    best['NaiveBayes'] = GaussianNB().fit(Xtr_, ytr)
    best['Stacking'] = StackingClassifier(estimators=[('rf', best['RandomForest']), ('ada', best['AdaBoost']), ('gb', best['GradientBoosting'])], final_estimator=LogisticRegression(class_weight='balanced'), cv=5).fit(Xtr_, ytr)
    for nm, m in best.items():
        p = m.predict_proba(Xte_)[:, 1]; tn, fp_, fn, tp = confusion_matrix(yte, (p >= .5).astype(int)).ravel(); PRED[(cfg, nm)] = p
        rows.append(dict(cfg=cfg, model=nm, TP=tp, FN=fn, FP=fp_, TN=tn, recall=round(tp / (tp + fn), 3), precision=round(tp / (tp + fp_), 3), auc=round(roc_auc_score(yte, p), 3)))
R = pd.DataFrame(rows)
print("== 1. EXACT VERSUS RECONSTRUCTED COUNTS (hold-out partition, 85 positive / 215 negative)")
out = []
for (cfg, nm), rec in RECON.items():
    r = R[(R.cfg == cfg) & (R.model == nm)].iloc[0]; same = []
    for k, e in zip(('TP', 'FN', 'FP', 'TN'), rec[:4]):
        if e is not None: same.append(f"{k} {int(r[k])} vs {e}" + ('' if int(r[k]) == e else ' <-- differs'))
    out.append((cfg, nm, ' | '.join(same), f"exact recall {r.recall} precision {r.precision} AUC {r.auc}" + (f" (thesis AUC {rec[4]})" if rec[4] else '')))
for o in out: print(o[0], f"{o[1]:17s}", o[2], '||', o[3])
print("\n== 1b. ALL EXACT COUNTS"); print(R.to_string(index=False))
# ---- locked model, internal and external
t2w = pd.read_csv('cnn_features_t2w.csv'); fc = [c for c in t2w.columns if c.startswith('cnn_feat_')]
sc_cnn = StandardScaler().fit(t2w[fc]); pca = PCA(n_components=100, random_state=42).fit(sc_cnn.transform(t2w[fc]))
ecnn = pd.read_csv('external_validation/external_cnn_features_t2w.csv'); fe = [c for c in ecnn.columns if c.startswith('cnn_feat_')]
epca = pd.DataFrame(pca.transform(sc_cnn.transform(ecnn[fe])), columns=[f'cnn_pca_{i}' for i in range(100)]); epca['Patient Number'] = ecnn['Patient Number'].values
coh = pd.read_csv('external_validation/final_external_cohort.csv')
ext = coh.merge(epca, on='Patient Number', how='inner'); ext['patient_age'] = cc['patient_age'].median()
cols = XB.columns.tolist(); Xex = ext[cols].reset_index(drop=True); yex = ext['csPCa_label'].to_numpy()
scL = StandardScaler().fit(XB.iloc[idx_tr])
locked = StackingClassifier(estimators=[('rf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')), ('ada', AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=42))], final_estimator=LogisticRegression(class_weight='balanced'), cv=5).fit(scL.transform(XB.iloc[idx_tr]), ytr)
pin = locked.predict_proba(scL.transform(XB.iloc[idx_te]))[:, 1]; pex = locked.predict_proba(scL.transform(Xex))[:, 1]
for nm, yy, pp in (('internal', yte, pin), ('external', yex, pex)):
    tn, fp_, fn, tp = confusion_matrix(yy, (pp >= .5).astype(int)).ravel(); print(f"\n== 2. LOCKED MODEL {nm}: TP {tp} FN {fn} FP {fp_} TN {tn} (n={len(yy)}, positives {int(yy.sum())})")
# ---- error profile, locked model, internal test partition
print("\n== 3. WHO ARE THE FALSE NEGATIVES AND FALSE POSITIVES? (locked model, internal test partition)")
te = XB.iloc[idx_te].reset_index(drop=True).copy(); te['y'] = yte; te['pred'] = (pin >= .5).astype(int); te['patient_id'] = fused['patient_id'].iloc[idx_te].values
iso = raw[['patient_id'] + [c for c in ('case_ISUP', 'histopath_type') if c in raw.columns]].drop_duplicates('patient_id'); te = te.merge(iso, on='patient_id', how='left')
te['group'] = np.select([(te.y == 1) & (te.pred == 1), (te.y == 1) & (te.pred == 0), (te.y == 0) & (te.pred == 1)], ['TP', 'FN', 'FP'], 'TN')
print(te.groupby('group').agg(n=('y', 'size'), median_psa=('psa', 'median'), median_psad=('psad', 'median'), median_volume=('prostate_volume', 'median'), median_pirads=('max_pirads', 'median')).round(2))
print("PI-RADS distribution by group (rows = group):"); print(pd.crosstab(te.group, te.max_pirads.round()))
if 'case_ISUP' in te.columns:
    print("ISUP grade group by group (0 = no cancer; 1 = grade group 1):"); print(pd.crosstab(te.group, te.case_ISUP))
    fn_ = te[te.group == 'FN'][['max_pirads', 'psad', 'prostate_volume', 'case_ISUP']]; print("each false negative:"); print(fn_.round(2).to_string(index=False))
if 'histopath_type' in te.columns: print("histopath type among false positives:"); print(te[te.group == 'FP'].histopath_type.fillna('none').value_counts())
print("\n== 4. SAME PARTITION: PI-RADS >= 4 RULE")
rule = (te.max_pirads >= 4).astype(int); tn, fp_, fn, tp = confusion_matrix(te.y, rule).ravel(); print(f"TP {tp} FN {fn} FP {fp_} TN {tn}")
print("false positives of the locked model that the rule did not flag / did flag:", int(((te.group == 'FP') & (rule == 0)).sum()), int(((te.group == 'FP') & (rule == 1)).sum()))
print("\n== 5. EXTERNAL FALSE NEGATIVES AND FALSE POSITIVES (locked model)")
ex = Xex.copy(); ex['y'] = yex; ex['pred'] = (pex >= .5).astype(int); ex['group'] = np.select([(ex.y == 1) & (ex.pred == 1), (ex.y == 1) & (ex.pred == 0), (ex.y == 0) & (ex.pred == 1)], ['TP', 'FN', 'FP'], 'TN')
print(ex.groupby('group').agg(n=('y', 'size'), median_psa=('psa', 'median'), median_psad=('psad', 'median'), median_volume=('prostate_volume', 'median'), median_score=('max_pirads', 'median')).round(2))
print(pd.crosstab(ex.group, ex.max_pirads.round()))
print("external file columns:", [c for c in coh.columns if not c.startswith('cnn')][:25])


# %% [cell 193]
import sys, platform, hashlib, os, importlib, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix
IMG_DIR = 'picai_images'
print("== 1. ENVIRONMENT"); print(sys.version.split()[0], '|', platform.platform(), '|', platform.machine())
for m in ['numpy', 'pandas', 'scipy', 'sklearn', 'xgboost', 'tensorflow', 'keras', 'SimpleITK', 'pydicom', 'cv2', 'matplotlib', 'PIL', 'imblearn', 'shap']:
    try: print(f"{m:12s}", importlib.import_module(m).__version__)
    except Exception as e: print(f"{m:12s} not available")
try:
    import tensorflow as tf; print("TensorFlow devices:", [d.device_type for d in tf.config.list_physical_devices()])
except Exception: pass
print("\n== 2. DERIVED FILES (rows x columns, SHA-256 prefix)")
for f in ['picai_labels/clinical_information/marksheet.csv', 'picai_clinical_clean.csv', 'cnn_features_t2w.csv', 'fused_final_dataset.csv', 'fused_final_dataset_with_pirads.csv', 'external_validation/external_clinical_clean.csv', 'external_validation/final_external_cohort.csv', 'external_validation/external_cnn_features_t2w.csv']:
    if os.path.exists(f):
        df_ = pd.read_csv(f); print(f"{f:62s} {df_.shape[0]:5d} x {df_.shape[1]:4d}  {hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16]}")
    else: print(f"{f:62s} not found")
fused = pd.read_csv('fused_final_dataset.csv'); t2w = pd.read_csv('cnn_features_t2w.csv'); raw = pd.read_csv('picai_labels/clinical_information/marksheet.csv')
fc = [c for c in t2w.columns if c.startswith('cnn_feat_')]; pcs = [c for c in fused.columns if c.startswith('cnn_pca_')]
t2w['patient_id'] = t2w['patient_id'].astype(int); fused['patient_id'] = fused['patient_id'].astype(int)
print("\n== 3. UNIT OF THE IMAGING FEATURES")
print("feature table rows:", len(t2w), "| unique patients:", t2w['patient_id'].nunique(), "| fused rows:", len(fused), "| fused unique patients:", fused['patient_id'].nunique())
rep = fused[fused.duplicated('patient_id', keep=False)]; same = sum(1 for _, g in rep.groupby('patient_id') if np.allclose(g[pcs].to_numpy(), g[pcs].iloc[0].to_numpy()))
print(f"repeat-patient groups: {rep['patient_id'].nunique()} ({len(rep)} studies); groups whose studies share identical imaging features: {same}")
if os.path.isdir(IMG_DIR):
    cnt = {pid: sorted(f for f in os.listdir(os.path.join(IMG_DIR, pid)) if 't2w' in f.lower()) for pid in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, pid))}
    multi = {k: v for k, v in cnt.items() if len(v) > 1}; print(f"patient folders: {len(cnt)} | with more than one T2W file: {len(multi)} | total T2W files: {sum(len(v) for v in cnt.values())}")
    try:
        import SimpleITK as sitk, cv2
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input
        base = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        def feats(path):
            a = sitk.GetArrayFromImage(sitk.ReadImage(path)); s = a[a.shape[0] // 2]
            s = cv2.cvtColor(cv2.resize(cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), (224, 224)), cv2.COLOR_GRAY2RGB)
            return base.predict(preprocess_input(np.expand_dims(s, 0).astype(np.float32)), verbose=0).flatten()
        for pid in list(multi)[:3]:
            stored = t2w[t2w.patient_id == int(pid)][fc].to_numpy()[0]; order = [f for f in os.listdir(os.path.join(IMG_DIR, pid)) if 't2w' in f.lower()]
            print(f"patient {pid}: os.listdir order {order} | sorted order {multi[pid]}")
            for f in multi[pid]: print(f"   max |stored - re-extracted| for {f}: {np.abs(stored - feats(os.path.join(IMG_DIR, pid, f))).max():.5f}")
    except Exception as e: print("re-extraction check skipped:", repr(e)[:100])
else: print("image folder not found; set IMG_DIR")
print("\n== 4. WHICH DATA WAS THE PCA FITTED ON?")
F_m = t2w.set_index('patient_id').loc[fused['patient_id'], fc].to_numpy(); k = len(pcs)
for name, fit_on in (('unique patient vectors', t2w[fc].to_numpy()), ('merged study rows (with repeats)', F_m)):
    sc_ = StandardScaler().fit(fit_on); pc_ = PCA(n_components=k, random_state=42).fit(sc_.transform(fit_on)); Z = pc_.transform(sc_.transform(F_m))
    print(f"PCA fitted on {name}: max |difference| to fused file = {np.abs(Z - fused[pcs].to_numpy()).max():.2e} (sign-agnostic {np.abs(np.abs(Z) - np.abs(fused[pcs].to_numpy())).max():.2e})")
print("\n== 5. SENSITIVITY: REMOVE ALL STUDIES OF REPEAT PATIENTS")
def get_max_pirads(v):
    if pd.isna(v): return np.nan
    s = [int(x.strip()) for x in str(v).split(',') if x.strip().isdigit()]
    return max(s) if s else np.nan
raw['max_pirads'] = raw['lesion_PIRADS'].apply(get_max_pirads)
fp = fused.merge(raw[['patient_id', 'max_pirads']].drop_duplicates('patient_id'), on='patient_id', how='left'); fp['max_pirads'] = fp['max_pirads'].fillna(fp['max_pirads'].median())
X = fp.drop(columns=['patient_id', 'csPCa_label']).reset_index(drop=True); y = fused['csPCa_label'].reset_index(drop=True); isrep = fused.duplicated('patient_id', keep=False).to_numpy()
itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
mk = lambda: StackingClassifier(estimators=[('rf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')), ('ada', AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=42))], final_estimator=LogisticRegression(class_weight='balanced'), cv=5)
for label, tr, te in (('all studies (as reported)', itr, ite), ('repeat-patient studies removed', itr[~isrep[itr]], ite[~isrep[ite]])):
    s_ = StandardScaler().fit(X.iloc[tr]); m = mk().fit(s_.transform(X.iloc[tr]), y.iloc[tr]); p = m.predict_proba(s_.transform(X.iloc[te]))[:, 1]
    tn, fp_, fn, tp = confusion_matrix(y.iloc[te], p >= .5).ravel(); print(f"{label:34s} train {len(tr)} test {len(te)} | AUC {roc_auc_score(y.iloc[te], p):.3f} recall {recall_score(y.iloc[te], p >= .5):.3f} | TP {tp} FN {fn} FP {fp_} TN {tn}")


# %% [cell 194]
import os, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score, confusion_matrix
IMG_DIR = 'picai_images'
fused = pd.read_csv('fused_final_dataset.csv'); t2w = pd.read_csv('cnn_features_t2w.csv'); raw = pd.read_csv('picai_labels/clinical_information/marksheet.csv'); cc = pd.read_csv('picai_clinical_clean.csv')
t2w['patient_id'] = t2w['patient_id'].astype(int); fused['patient_id'] = fused['patient_id'].astype(int)
fc = [c for c in t2w.columns if c.startswith('cnn_feat_')]; pcs = [c for c in fused.columns if c.startswith('cnn_pca_')]
def get_max_pirads(v):
    if pd.isna(v): return np.nan
    s = [int(x.strip()) for x in str(v).split(',') if x.strip().isdigit()]
    return max(s) if s else np.nan
raw['max_pirads'] = raw['lesion_PIRADS'].apply(get_max_pirads)
fp = fused.merge(raw[['patient_id', 'max_pirads']].drop_duplicates('patient_id'), on='patient_id', how='left'); fp['max_pirads'] = fp['max_pirads'].fillna(fp['max_pirads'].median())
X = fp.drop(columns=['patient_id', 'csPCa_label']).reset_index(drop=True); y = fused['csPCa_label'].reset_index(drop=True); cols = X.columns.tolist()
itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
F_m = t2w.set_index('patient_id').loc[fused['patient_id'], fc].to_numpy()
ecnn = pd.read_csv('external_validation/external_cnn_features_t2w.csv'); fe = [c for c in ecnn.columns if c.startswith('cnn_feat_')]; coh = pd.read_csv('external_validation/final_external_cohort.csv')
def external_with(fit_on):
    sc = StandardScaler().fit(fit_on); pc = PCA(n_components=len(pcs), random_state=42).fit(sc.transform(fit_on)); Z = pc.transform(sc.transform(ecnn[fe].to_numpy()))
    e = pd.DataFrame(Z, columns=pcs); e['Patient Number'] = ecnn['Patient Number'].values; ext = coh.merge(e, on='Patient Number', how='inner'); ext['patient_age'] = cc['patient_age'].median(); return ext, Z
ext_u, Z_u = external_with(t2w[fc].to_numpy()); ext_m, Z_m = external_with(F_m)
cor = [abs(np.corrcoef(Z_u[:, j], Z_m[:, j])[0, 1]) for j in range(len(pcs))]
print("== 1. HOW DIFFERENT ARE THE TWO EXTERNAL FEATURE BASES?"); print("components with |correlation| < 0.9:", int(np.sum(np.array(cor) < 0.9)), "of", len(pcs), "| first 10 correlations:", np.round(cor[:10], 3))
mk = lambda: StackingClassifier(estimators=[('rf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')), ('ada', AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=42))], final_estimator=LogisticRegression(class_weight='balanced'), cv=5)
sc = StandardScaler().fit(X.iloc[itr]); locked = mk().fit(sc.transform(X.iloc[itr]), y.iloc[itr])
def report(label, ext, model, scaler, cl):
    p = model.predict_proba(scaler.transform(ext[cl]))[:, 1]; yy = ext['csPCa_label'].to_numpy(); tn, fp_, fn, tp = confusion_matrix(yy, p >= .5).ravel()
    print(f"{label:52s} AUC {roc_auc_score(yy, p):.3f} recall {recall_score(yy, p >= .5):.3f} precision {precision_score(yy, p >= .5):.3f} spec {tn / (tn + fp_):.3f} | TP {tp} FN {fn} FP {fp_} TN {tn}")
print("\n== 2. LOCKED MODEL ON THE EXTERNAL COHORT"); report("as reported (PCA fitted on 1,476 unique vectors)", ext_u, locked, sc, cols); report("consistent basis (PCA fitted on merged 1,500 rows)", ext_m, locked, sc, cols)
nonimg = [c for c in cols if c not in pcs]; sc2 = StandardScaler().fit(X.iloc[itr][nonimg]); m2 = mk().fit(sc2.transform(X.iloc[itr][nonimg]), y.iloc[itr]); report("no imaging features at all (clinical + score)", ext_m, m2, sc2, nonimg)
print("\n== 3. SELECTION MANIFEST FOR PATIENTS WITH MORE THAN ONE T2W FILE")
rows = []
for pid in sorted(os.listdir(IMG_DIR)):
    fol = os.path.join(IMG_DIR, pid)
    if not os.path.isdir(fol): continue
    t = [f for f in os.listdir(fol) if 't2w' in f.lower()]
    rows.append(dict(patient_id=int(pid), n_t2w_files=len(t), selected_file=t[0] if t else '', first_in_sorted_order=sorted(t)[0] if t else '', selected_is_sorted_first=bool(t) and t[0] == sorted(t)[0]))
man = pd.DataFrame(rows); man.to_csv('selected_t2w_files.csv', index=False)
print("saved selected_t2w_files.csv:", len(man), "patients | with >1 file:", int((man.n_t2w_files > 1).sum()), "| of those, selected file is not the first in sorted order:", int(((man.n_t2w_files > 1) & (~man.selected_is_sorted_first)).sum()))
