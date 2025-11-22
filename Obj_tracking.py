# -*- coding: utf-8 -*-
# STEP 5. Object Tracking - tương thích file keypoints mới (bước 4)

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# ====== Cấu hình ======
INPUT_CSV = "/media/pphong/D:/git&github/classroom-cheating/keypoint_csv/normal/normal_keypoints.csv"
OUTPUT_CSV = "/media/pphong/D:/git&github/classroom-cheating/keypoint_id_csv/normal_keypoints_tracked.csv"
IOU_THRESHOLD = 0.35  # Ngưỡng IoU để xác định cùng người

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ====== Đọc dữ liệu keypoints ======
df = pd.read_csv(INPUT_CSV)
print(f"Đọc {len(df)} dòng từ {INPUT_CSV}")

# ====== Xác định số lượng keypoints thực tế ======
keypoint_cols = [c for c in df.columns if "_x" in c]
n_kp = len(keypoint_cols)
if n_kp == 0:
    raise ValueError("Không tìm thấy cột keypoint nào trong file CSV!")
print(f"Số keypoints thực tế: {n_kp}")

# ====== Hàm tính bounding box từ keypoints ======
def get_bbox(row):
    """Tính toạ độ bounding box bao quanh tất cả keypoints hợp lệ"""
    xs, ys, confs = [], [], []
    for i in range(n_kp):
        if f"kp{i}_x" in row and f"kp{i}_y" in row and f"kp{i}_conf" in row:
            x, y, c = row[f"kp{i}_x"], row[f"kp{i}_y"], row[f"kp{i}_conf"]
            xs.append(x)
            ys.append(y)
            confs.append(c)

    # Lọc bỏ keypoints yếu
    valid_points = [(x, y) for x, y, c in zip(xs, ys, confs) if c > 0.3]
    if not valid_points:
        return (0, 0, 0, 0)

    xs, ys = zip(*valid_points)
    return (min(xs), min(ys), max(xs), max(ys))  # (xmin, ymin, xmax, ymax)

# ====== Hàm tính IoU ======
def compute_iou(boxA, boxB):
    """Tính Intersection over Union giữa 2 bounding box"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# ====== Tracking từng video ======
tracked_rows = []

for video_name, video_df in tqdm(df.groupby("video"), desc="Tracking theo video"):
    video_df = video_df.sort_values("frame").reset_index(drop=True)

    prev_boxes = []
    prev_ids = []
    next_person_id = 0

    for idx, row in video_df.iterrows():
        box = get_bbox(row)

        # Frame đầu tiên → gán ID mới
        if not prev_boxes:
            row["person_id"] = next_person_id
            prev_boxes.append(box)
            prev_ids.append(next_person_id)
            next_person_id += 1
        else:
            # Tính IoU với các box ở frame trước
            ious = [compute_iou(box, prev_box) for prev_box in prev_boxes]
            max_iou = max(ious) if ious else 0
            best_idx = ious.index(max_iou) if ious else -1

            if max_iou >= IOU_THRESHOLD:
                row["person_id"] = prev_ids[best_idx]
            else:
                row["person_id"] = next_person_id
                next_person_id += 1

            prev_boxes = [box]
            prev_ids = [row["person_id"]]

        tracked_rows.append(row)

# ====== Lưu kết quả ======
tracked_df = pd.DataFrame(tracked_rows)
tracked_df.to_csv(OUTPUT_CSV, index=False)

print(f"Đã lưu file tracking: {OUTPUT_CSV}")
print(f"Tổng số dòng sau tracking: {len(tracked_df)}")
print(f"Các cột: {list(tracked_df.columns)}")
