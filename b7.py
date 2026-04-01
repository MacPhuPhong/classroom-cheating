import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import GroupShuffleSplit

# =====================================================
# CONFIG
# =====================================================
DATASET_CSV = "/media/pphong/D:/git&github/classroom-cheating/out_b6/temporal_dataset.csv"
MODEL_OUT   = "/media/pphong/D:/git&github/classroom-cheating/models/temporal_xgboost_cheating.pkl"
FIG_DIR     = "/media/pphong/D:/git&github/classroom-cheating/out_b6/figures"

TEST_SIZE = 0.25
RANDOM_STATE = 42

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATASET_CSV)
print(f"Loaded dataset: {df.shape}")

# -----------------------------------------------------
# BASIC SANITY CHECK
# -----------------------------------------------------
assert "person_id" in df.columns
assert "label" in df.columns

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().reset_index(drop=True)

print("\nLabel distribution:")
print(df["label"].value_counts())

# =====================================================
# FEATURES / LABEL
# =====================================================
NON_FEATURE_COLS = [
    "video_id", "person_id",
    "window_start", "window_end",
    "label"
]

X = df.drop(columns=NON_FEATURE_COLS)
y = df["label"].astype(int)
groups = df["person_id"]  # chống leakage theo người

print(f"\nFeature dimension: {X.shape[1]}")

# =====================================================
# SPLIT TRAIN / TEST (GROUP BY PERSON)
# =====================================================
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

train_persons = set(groups.iloc[train_idx])
test_persons  = set(groups.iloc[test_idx])

print(f"\nTrain persons: {len(train_persons)}")
print(f"Test persons : {len(test_persons)}")
print("Person overlap:", train_persons & test_persons)

assert len(train_persons & test_persons) == 0, "❌ PERSON LEAKAGE DETECTED"

# =====================================================
# TRAIN XGBOOST
# =====================================================
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

# =====================================================
# EVALUATE
# =====================================================
y_pred = model.predict(X_test)

# -----------------------------
# CONFUSION MATRIX (HEATMAP)
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["NORMAL", "CHEATING"],
    yticklabels=["NORMAL", "CHEATING"]
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix (Person-level split)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=200)
plt.show()

# -----------------------------
# CLASSIFICATION REPORT (TABLE)
# -----------------------------
report = classification_report(
    y_test, y_pred,
    target_names=["NORMAL", "CHEATING"],
    output_dict=True
)

df_report = pd.DataFrame(report).transpose().round(4)
print("\nClassification report:")
print(df_report)

df_report.to_csv(os.path.join(FIG_DIR, "classification_report.csv"))

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
fi = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

plt.figure(figsize=(7,5))
fi.tail(10).plot(kind="barh")
plt.xlabel("Importance score")
plt.title("Top-10 Feature Importances (XGBoost)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "feature_importance_top10.png"), dpi=200)
plt.show()

# =====================================================
# LEAKAGE HEURISTIC CHECK
# =====================================================
top5_ratio = (fi.sort_values(ascending=False).head(5) / fi.sum())

plt.figure(figsize=(5,4))
top5_ratio.plot(kind="bar")
plt.ylabel("Importance ratio")
plt.title("Top-5 Feature Importance Ratio\n(Leakage heuristic)")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "leakage_heuristic.png"), dpi=200)
plt.show()

print("\nTop-5 feature importance ratio:")
print(top5_ratio.round(3))

# =====================================================
# SAVE MODEL
# =====================================================
joblib.dump(model, MODEL_OUT)
print(f"\nModel saved to: {MODEL_OUT}")
print(f"Figures saved to: {FIG_DIR}")
