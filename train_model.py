"""
ThyroScan KBS – Combined Model Training
========================================
Combines TWO datasets:
  1. hypothyroid.csv  (Garavan, 3772 rows) → binary: Normal / Hypothyroid
  2. thyroidDF.csv    (UCI/Kaggle, 9172 rows) → 3-class: Negative / Hypothyroid / Hyperthyroid

Final unified classes: Negative (Normal), Hypothyroid, Hyperthyroid
Features (common to both): age, sex, TSH, T3, TT4, T4U, FTI, pregnant

Dataset 1 label facts (verified from notebook):
  P = 3481 = NORMAL (negative for disease)
  N =  291 = HYPOTHYROID

Dataset 2 label encoding (LabelEncoder on Hyperthyroid/Hypothyroid/Negative):
  Hyperthyroid = 0, Hypothyroid = 1, Negative = 2
"""

import sys, os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib

RANDOM_STATE = 42
MODEL_DIR    = "models"

# ── Unified feature columns ───────────────────────────────────────
FEATURES = ['age', 'sex', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'pregnant']

# ── Class labels ─────────────────────────────────────────────────
CLASSES = ['Negative', 'Hypothyroid', 'Hyperthyroid']

KNOWLEDGE_BASE = {
    "Negative": {
        "description": "Normal thyroid function (euthyroid) — no intervention needed",
        "tsh_range": [0.35, 4.5], "t4_range": [0.8, 1.8], "t3_range": [1.2, 3.1],
        "symptoms": [],
        "rules": ["TSH 0.35–4.5 mIU/L with normal T3/TT4 → euthyroid"],
    },
    "Hypothyroid": {
        "description": "Underactive thyroid – insufficient hormone production",
        "tsh_range": [4.5, 150], "t4_range": [0.0, 0.8], "t3_range": [0.0, 1.2],
        "symptoms": ["fatigue", "weight_gain", "cold_intolerance",
                     "dry_skin", "hair_loss", "depression", "slow_heart_rate"],
        "rules": [
            "TSH > 4.5 mIU/L suggests hypothyroidism",
            "Low T3/TT4 with elevated TSH confirms primary hypothyroidism",
        ],
    },
    "Hyperthyroid": {
        "description": "Overactive thyroid – excess hormone production",
        "tsh_range": [0.0, 0.35], "t4_range": [1.8, 6.0], "t3_range": [3.1, 10.0],
        "symptoms": ["weight_loss", "heat_intolerance", "tremors",
                     "rapid_heart_rate", "anxiety", "insomnia"],
        "rules": [
            "TSH < 0.35 mIU/L suggests hyperthyroidism",
            "Elevated T3/TT4 with suppressed TSH confirms hyperthyroidism",
        ],
    },
}


# ════════════════════════════════════════════════════════
#  DATASET 1: hypothyroid.csv  (binary)
# ════════════════════════════════════════════════════════
def load_dataset1(csv_path):
    print(f"\n[Dataset 1] Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Raw shape: {df.shape}")
    print(f"  Raw class counts:\n{df['binaryClass'].value_counts().to_string()}")

    # Replace '?' first
    df = df.replace("?", np.nan)

    df["label"] = df["binaryClass"].map({"P": "Hypothyroid", "N": "Negative"})

    # Encode booleans
    df = df.replace({"t": 1, "f": 0, "F": 0, "M": 1})

    # Drop irrelevant cols
    for col in ["TBG", "referral source", "binaryClass"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Cast to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rename columns to match dataset2 naming
    rename = {}
    # hypothyroid.csv uses exact same column names already
    df = df.rename(columns=rename)

    # Keep only shared features + label + pregnant
    # hypothyroid.csv has 'pregnant' as a boolean col already
    available = [f for f in FEATURES if f in df.columns]
    df = df[available + ["label"]].copy()

    # Impute
    imputer = SimpleImputer(strategy="mean")
    feat_cols = [f for f in FEATURES if f in df.columns]
    df[feat_cols] = pd.DataFrame(
        imputer.fit_transform(df[feat_cols]), columns=feat_cols, index=df.index)

    df = df.dropna(subset=["label"])
    print(f"  After processing: {df.shape}")
    print(f"  Classes:\n{df['label'].value_counts().to_string()}")
    return df


# ════════════════════════════════════════════════════════
#  DATASET 2: thyroidDF.csv  (3-class)
# ════════════════════════════════════════════════════════
def load_dataset2(csv_path):
    print(f"\n[Dataset 2] Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Raw shape: {df.shape}")

    # Drop cols not needed
    for col in ["TBG", "patient_id", "referral_source",
                "TBG_measured", "TSH_measured", "T3_measured",
                "TT4_measured", "T4U_measured", "FTI_measured"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Map target labels
    label_map = {
        '-': 'Negative',
        'A': 'Hyperthyroid', 'AK': 'Hyperthyroid', 'B': 'Hyperthyroid',
        'C': 'Hyperthyroid', 'C|I': 'Hyperthyroid', 'D': 'Hyperthyroid',
        'D|R': 'Hyperthyroid',
        'E': 'Hypothyroid', 'F': 'Hypothyroid', 'FK': 'Hypothyroid',
        'G': 'Hypothyroid', 'GK': 'Hypothyroid', 'GI': 'Hypothyroid',
        'GKJ': 'Hypothyroid', 'H|K': 'Hypothyroid',
    }
    df["label"] = df["target"].map(label_map)
    df = df.dropna(subset=["label"])
    df = df.drop(columns=["target"])

    # Encode booleans t/f
    df = df.replace({"t": 1, "f": 0})
    # Encode sex F→0, M→1
    df = df.replace({"F": 0, "M": 1})

    # Rename columns to unified names
    rename_map = {
        "on_thyroxine": "on_thyroxine",
        "query_on_thyroxine": "query_on_thyroxine",
        "on_antithyroid_meds": "on_antithyroid_meds",
    }
    df = df.rename(columns=rename_map)

    # Cast to numeric
    for col in df.columns:
        if col != "label":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove outliers (as in notebook)
    for col in ["age", "TSH", "T3", "TT4", "T4U", "FTI"]:
        if col in df.columns:
            mean, std = df[col].mean(), df[col].std()
            df = df[df[col] < (mean + 3 * std)]
    df = df[df["age"] <= 100]

    # Keep only shared features + label
    available = [f for f in FEATURES if f in df.columns]
    df = df[available + ["label"]].copy()

    # Impute missing
    imputer = SimpleImputer(strategy="mean")
    feat_cols = [f for f in FEATURES if f in df.columns]
    df[feat_cols] = pd.DataFrame(
        imputer.fit_transform(df[feat_cols]), columns=feat_cols, index=df.index)

    df = df.dropna(subset=["label"])
    print(f"  After processing: {df.shape}")
    print(f"  Classes:\n{df['label'].value_counts().to_string()}")
    return df


# ════════════════════════════════════════════════════════
#  TRAIN
# ════════════════════════════════════════════════════════
def train(csv1=None, csv2=None):
    frames = []

    if csv1 and os.path.exists(csv1):
        frames.append(load_dataset1(csv1))
    else:
        print(f"[Dataset 1] Not found: {csv1} — skipping")

    if csv2 and os.path.exists(csv2):
        frames.append(load_dataset2(csv2))
    else:
        print(f"[Dataset 2] Not found: {csv2} — skipping")

    if not frames:
        print("ERROR: No datasets found. Provide hypothyroid.csv and/or thyroidDF.csv")
        sys.exit(1)

    # Combine
    combined = pd.concat(frames, ignore_index=True)
    print(f"\n[Combined] Shape: {combined.shape}")
    print(f"[Combined] Classes:\n{combined['label'].value_counts().to_string()}")

    # Ensure all feature columns exist
    for f in FEATURES:
        if f not in combined.columns:
            combined[f] = 0.0

    X = combined[FEATURES].values
    y = combined["label"].values

    # Encode labels to int
    le = LabelEncoder()
    le.fit(CLASSES)  # fix order: Negative=0, Hypothyroid=1, Hyperthyroid=2
    y_enc = le.transform(y)
    print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=RANDOM_STATE, stratify=y_enc)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("\nTraining Random Forest (300 trees, balanced weights)…")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=le.classes_))

    cv = cross_val_score(model, scaler.transform(X), y_enc, cv=5, scoring="accuracy")
    print(f"Cross-val: {cv.mean()*100:.2f}% +/- {cv.std()*100:.2f}%")

    # Feature importance
    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    print("\nFeature importances:")
    for name, score in imp:
        print(f"  {name:<12} {score:.4f}")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "thyroid_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(le,     os.path.join(MODEL_DIR, "label_encoder.pkl"))

    meta = {
        "accuracy":           float(acc),
        "cv_mean":            float(cv.mean()),
        "cv_std":             float(cv.std()),
        "classes":            list(le.classes_),
        "class_mapping":      {str(i): c for i, c in enumerate(le.classes_)},
        "feature_names":      FEATURES,
        "feature_importance": dict(zip(FEATURES, model.feature_importances_.tolist())),
        "datasets":           [csv1 or "", csv2 or ""],
        "n_samples":          int(len(combined)),
        "knowledge_base":     KNOWLEDGE_BASE,
    }
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved → {MODEL_DIR}/thyroid_model.pkl")
    print(f"Saved → {MODEL_DIR}/scaler.pkl")
    print(f"Saved → {MODEL_DIR}/label_encoder.pkl")
    print(f"Saved → {MODEL_DIR}/model_meta.json")
    print("\nDone! Run: python app.py")


if __name__ == "__main__":
    # Usage: python train_model.py [hypothyroid.csv] [thyroidDF.csv]
    csv1 = sys.argv[1] if len(sys.argv) > 1 else "hypothyroid.csv"
    csv2 = sys.argv[2] if len(sys.argv) > 2 else "thyroidDF.csv"
    train(csv1, csv2)
