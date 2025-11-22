# -*- coding: utf-8 -*-
# STEP 4 - Skeleton Extraction (tương thích YOLOv8-Pose hoặc YOLO-Pose gốc)

import cv2
import os
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# ====== Cấu hình hệ thống ======
MODEL_PATH = "/media/pphong/D:/git&github/classroom-cheating/yolo11s-pose.pt"  # Model YOLO-Pose (Ultralytics)
INPUT_DIR = "/media/pphong/D:/git&github/classroom-cheating/output_video/normal"  # Thư mục chứa video clip
OUTPUT_CSV = "/media/pphong/D:/git&github/classroom-cheating/keypoint_csv/normal/normal_keypoints.csv"
LABEL = 0  # 0 = non-cheating, 1 = cheating
CONF_THRES = 0.35  # Ngưỡng confidence
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ====== Load mô hình YOLO-Pose ======
model = YOLO(MODEL_PATH)

# ====== Lấy danh sách video cần xử lý ======
video_list = [f for f in os.listdir(INPUT_DIR) if f.endswith(".mp4")]
print(f"Tìm thấy {len(video_list)} clip cần xử lý...")

# ====== Mảng lưu dữ liệu keypoints ======
all_rows = []

# ====== Hàm xử lý 1 video ======
def process_video(video_path):
    """
    Đọc từng frame, chạy YOLO-Pose, lấy keypoints của từng người,
    lưu ra CSV (mỗi dòng = 1 người / 1 frame).
    """
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
              desc=f"Xử lý {os.path.basename(video_path)}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=CONF_THRES, verbose=False)

            # Nếu không phát hiện người
            if len(results) == 0 or results[0].keypoints is None:
                pbar.update(1)
                continue

            result = results[0]
            kps = result.keypoints.data.cpu().numpy()  # shape = (n_person, n_kp, 3)
            n_kp = kps.shape[1]  # số lượng keypoints thực tế

            for person in kps:
                row = {}
                for i, (x, y, conf) in enumerate(person):
                    row[f"kp{i}_x"] = float(x)
                    row[f"kp{i}_y"] = float(y)
                    row[f"kp{i}_conf"] = float(conf)

                row["label"] = LABEL
                row["video"] = os.path.basename(video_path)
                row["frame"] = frame_idx
                row["num_keypoints"] = n_kp  # lưu thêm số lượng keypoints để đối chiếu
                all_rows.append(row)

            frame_idx += 1
            pbar.update(1)

    cap.release()

# ====== Xử lý tất cả video ======
for vid in video_list:
    process_video(os.path.join(INPUT_DIR, vid))

# ====== Xuất CSV ======
df = pd.DataFrame(all_rows)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Đã lưu file: {OUTPUT_CSV}")
print(f"Tổng số dòng (người-frame): {len(df)}")
if len(df) > 0:
    print(f"Phát hiện {int(df['num_keypoints'].iloc[0])} keypoints / người.")
