"""
train_model.py — Train an XGBoost classifier for Loan Approval Prediction.

Workflow (following ML best‑practices):
1.  Load & inspect loan_data.csv
2.  Clean outliers / sanity checks
3.  Encode categoricals (LabelEncoder for binary, OrdinalEncoder for education,
    OneHotEncoder for nominal multi‑class columns)
4.  Split BEFORE fitting any preprocessing (strict featurization ordering)
5.  Scale numerical features (StandardScaler fitted on train only)
6.  Handle class imbalance via scale_pos_weight
7.  Train XGBoost with reasonable defaults
8.  Evaluate (accuracy, F1, ROC‑AUC, confusion matrix)
9.  Persist: model, preprocessor pipeline, and feature names → artifacts/
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")

# ── 1. Load ──────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "loan_data.csv")
df = pd.read_csv(DATA_PATH)
print(f"✔ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Target distribution (loan_status):\n{df['loan_status'].value_counts().to_string()}\n")

# ── 2. Sanity / outlier handling ─────────────────────────────────────────────
# person_age > 100 is clearly noise (max is 144)
outlier_mask = df["person_age"] > 100
print(f"  Dropping {outlier_mask.sum()} rows with person_age > 100")
df = df[~outlier_mask].reset_index(drop=True)

# ── 3. Separate features & target ───────────────────────────────────────────
TARGET = "loan_status"
y = df[TARGET].values
X = df.drop(columns=[TARGET])

# ── 4. Encoding — define column groups ───────────────────────────────────────
binary_cols = ["person_gender", "previous_loan_defaults_on_file"]
ordinal_cols = ["person_education"]
nominal_cols = ["person_home_ownership", "loan_intent"]
numeric_cols = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]

# Education has a natural order
education_order = [["High School", "Associate", "Bachelor", "Master", "Doctorate"]]

# Binary encode
le_gender = LabelEncoder()
X["person_gender"] = le_gender.fit_transform(X["person_gender"])  # female=0, male=1

le_defaults = LabelEncoder()
X["previous_loan_defaults_on_file"] = le_defaults.fit_transform(
    X["previous_loan_defaults_on_file"]
)  # No=0, Yes=1

# Ordinal encode education
oe_education = OrdinalEncoder(categories=education_order)
X["person_education"] = oe_education.fit_transform(X[["person_education"]]).astype(int)

# One‑hot encode nominal columns
X = pd.get_dummies(X, columns=nominal_cols, drop_first=False, dtype=int)

feature_names = X.columns.tolist()
print(f"  Features after encoding: {len(feature_names)}")

# ── 5. Train / Test split BEFORE scaling ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# Scale numerical features — fit on train only
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ── 6. Handle class imbalance ───────────────────────────────────────────────
neg, pos = np.bincount(y_train)
scale_pos = neg / pos
print(f"  Class balance → neg={neg:,} pos={pos:,}  scale_pos_weight={scale_pos:.2f}\n")

# ── 7. Train XGBoost ────────────────────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("✔ Model trained")

# ── 8. Evaluate ─────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print(f"\n{'═'*50}")
print(f"  Accuracy : {acc:.4f}")
print(f"  F1‑Score : {f1:.4f}")
print(f"  ROC‑AUC  : {roc:.4f}")
print(f"{'═'*50}")
print(f"\n  Confusion Matrix:\n{cm}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Approved (0)', 'Rejected (1)'])}")

# ── 9. Save artifacts ───────────────────────────────────────────────────────
ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)

joblib.dump(model, os.path.join(ARTIFACTS, "xgb_model.joblib"))
joblib.dump(scaler, os.path.join(ARTIFACTS, "scaler.joblib"))
joblib.dump(le_gender, os.path.join(ARTIFACTS, "le_gender.joblib"))
joblib.dump(le_defaults, os.path.join(ARTIFACTS, "le_defaults.joblib"))
joblib.dump(oe_education, os.path.join(ARTIFACTS, "oe_education.joblib"))
joblib.dump(feature_names, os.path.join(ARTIFACTS, "feature_names.joblib"))

print(f"\n✔ Artifacts saved to {ARTIFACTS}/")
