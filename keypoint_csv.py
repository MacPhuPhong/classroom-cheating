import os
import pandas as pd
from ultralytics import YOLO

# --- Cấu hình ---
MODEL_PATH = r"/media/pphong/D:/git&github/classroom-cheating/yolo11s-pose.pt"
SAVE_NORMAL = r"/media/pphong/D:/git&github/classroom-cheating/dataset_path/Normal"
SAVE_SUSPICIOUS = r"/media/pphong/D:/git&github/classroom-cheating/dataset_path/Suspicious"
CSV_PATH = r"/media/pphong/D:/git&github/classroom-cheating/dataset_path/CSV_keypoint/keypoints_dataset.csv"

# --- Load model ---
model = YOLO(MODEL_PATH)

data_records = []

def extract_from_folder(folder, label):
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        path = os.path.join(folder, fname)
        results = model(path, verbose=False)
        r = results[0]

        if r.keypoints is None or len(r.keypoints.xy) == 0:
            continue

        kps = r.keypoints.xy[0].cpu().numpy().tolist()
        # Giữ keypoints phần thân trên (0–10)
        keypoints = [kp for idx, kp in enumerate(kps) if idx <= 10]

        record = {"image": fname, "label": label}
        for i, kp in enumerate(keypoints):
            if len(kp) >= 2:
                record[f"x{i}"] = kp[0]
                record[f"y{i}"] = kp[1]
        data_records.append(record)

# --- Chạy ---
extract_from_folder(SAVE_NORMAL, "Normal")
extract_from_folder(SAVE_SUSPICIOUS, "Suspicious")

# --- Xuất CSV ---
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
df = pd.DataFrame(data_records)
df.to_csv(CSV_PATH, index=False)

print(f"\nDONE — {len(df)} samples saved to {CSV_PATH}")
