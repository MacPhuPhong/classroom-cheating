
# (n_samples, 51) cho ML models như SVM, DT, RF, XGBoost.

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

# ====== Cấu hình ======
INPUT_DIR = "/media/pphong/D:/git&github/classroom-cheating/keypoint_id_csv"   # Thư mục chứa các file CSV sau tracking (normal + cheating)
OUTPUT_DATASET = "/media/pphong/D:/git&github/classroom-cheating/dataset/xgboost_dataset.csv"
os.makedirs("dataset", exist_ok=True)

# ====== 6.1. Gộp toàn bộ CSV ======
csv_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith("_tracked.csv")]
print(f"Đã tìm thấy {len(csv_files)} file CSV cần gộp...")

all_data = []
for file in tqdm(csv_files, desc="Đang gộp dữ liệu"):
    df = pd.read_csv(file)
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
print(f"Tổng số dòng sau gộp: {len(data)}")

# ====== 6.2. Loại bỏ keypoint không quan trọng (3 điểm phần chân) ======
# Theo bài báo trang 10: "lower body keypoints less significant"
# Các điểm phần chân YOLO-Pose: 15 (right knee), 16 (right ankle), 17 (left ankle)
remove_ids = [15, 16, 17]
cols_to_drop = []
for i in remove_ids:
    cols_to_drop += [f"kp{i}_x", f"kp{i}_y", f"kp{i}_conf"]

data = data.drop(columns=cols_to_drop, errors="ignore")

# ====== 6.3. Chuẩn hóa dữ liệu về 51 features ======
# Sau khi bỏ 3 keypoints -> 17*3 = 51 cột đặc trưng
feature_cols = [c for c in data.columns if c.startswith("kp") and "_conf" in c or "_x" in c or "_y" in c]
if "label" not in data.columns:
    raise ValueError("File đầu vào không có cột 'label'. Hãy chắc chắn rằng bước 4-5 đã gán nhãn đúng.")

X = data[feature_cols].astype(float)
y = data["label"].astype(int)

print(f"Tổng số đặc trưng còn lại: {X.shape[1]} (mong đợi ~51)")

# ====== 6.4. Train/Test Split (80/20) ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape[0]} mẫu, Test: {X_test.shape[0]} mẫu")

# ====== 6.5. Huấn luyện mô hình XGBoost ======
print("Đang huấn luyện XGBoost...")
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ====== 6.6. Đánh giá mô hình ======
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== Kết quả mô hình XGBoost ==========")
print(f"Accuracy : {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall   : {rec*100:.2f}%")
print(f"F1-score : {f1*100:.2f}%")
print("=============================================")
print(classification_report(y_test, y_pred, target_names=["Non-cheating", "Cheating"]))

# ====== 6.7. Lưu dataset chuẩn hóa ======
dataset = pd.concat([X, y], axis=1)
dataset.to_csv(OUTPUT_DATASET, index=False)
print(f" Đã lưu dataset chuẩn hóa tại: {OUTPUT_DATASET}")
