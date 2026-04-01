import os
os.environ["QT_QPA_PLATFORM"] = "xcb"   # tránh lỗi Qt trên Linux

import cv2
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict
from ultralytics import YOLO

# =====================================================
# CONFIG
# =====================================================
VIDEO_PATH = "/media/pphong/D:/git&github/classroom-cheating/datavideo/video4.mp4"
MODEL_PATH = "/media/pphong/D:/git&github/classroom-cheating/models/temporal_xgboost_cheating.pkl"
POSE_MODEL = "yolo11s-pose.pt"

WINDOW_SIZE = 20
WINDOW_STRIDE = 20
IOU_TH = 0.5

# =====================================================
# LOAD MODELS
# =====================================================
pose_model = YOLO(POSE_MODEL)
clf = joblib.load(MODEL_PATH)

# =====================================================
# IOU
# =====================================================
def iou(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)

    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (areaA + areaB - inter + 1e-6)

# =====================================================
# FEATURE EXTRACTION (B3)
# =====================================================
def extract_features_from_kp(kp):
    """
    kp: (17,2) COCO keypoints
    """
    nose = kp[0]
    le, re = kp[1], kp[2]
    ls, rs = kp[5], kp[6]
    lh, rh = kp[11], kp[12]
    lw, rw = kp[9], kp[10]

    mid_eye = (le + re) / 2
    mid_shoulder = (ls + rs) / 2
    mid_hip = (lh + rh) / 2

    head_yaw = nose[0] - mid_eye[0]
    body_down = nose[1] - mid_hip[1]
    body_lean_x = mid_shoulder[0] - mid_hip[0]

    shoulder_width = np.linalg.norm(ls - rs) + 1e-6
    reach = max(
        np.linalg.norm(lw - mid_shoulder),
        np.linalg.norm(rw - mid_shoulder)
    ) / shoulder_width

    return head_yaw, body_down, body_lean_x, reach

# =====================================================
# MAIN
# =====================================================
cap = cv2.VideoCapture(VIDEO_PATH)

tracks = {}                  # pid -> bbox
kp_by_pid = {}               # pid -> keypoints
next_pid = 0

history = defaultdict(list)  # pid -> list of feature dict
pred_frames = defaultdict(dict)

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # ---------------- B1: YOLO Pose ----------------
    res = pose_model(frame, conf=0.4, verbose=False)[0]

    detections = []
    if res.keypoints is not None:
        for kp, box in zip(res.keypoints.xy, res.boxes.xyxy):
            detections.append((box.cpu().numpy(), kp.cpu().numpy()))

    # ---------------- B2: IOU Tracking ----------------
    new_tracks = {}
    new_kp = {}

    for box, kp in detections:
        matched = False
        for pid, pbox in tracks.items():
            if iou(box, pbox) > IOU_TH:
                new_tracks[pid] = box
                new_kp[pid] = kp
                matched = True
                break

        if not matched:
            new_tracks[next_pid] = box
            new_kp[next_pid] = kp
            next_pid += 1

    tracks = new_tracks
    kp_by_pid = new_kp

    # ---------------- B3 + B6 ----------------
    for pid, box in tracks.items():
        kp = kp_by_pid.get(pid)
        if kp is None:
            continue

        head_yaw, body_down, body_lean_x, reach = extract_features_from_kp(kp)

        history[pid].append({
            "frame": frame_idx,
            "head_yaw": head_yaw,
            "body_down": body_down,
            "body_lean_x": body_lean_x,
            "reach": reach
        })

        if len(history[pid]) >= WINDOW_SIZE:
            w = history[pid][-WINDOW_SIZE:]

            row = {
                "mean_head_yaw": np.mean([f["head_yaw"] for f in w]),
                "max_head_yaw": np.max(np.abs([f["head_yaw"] for f in w])),
                "mean_body_down": np.mean([f["body_down"] for f in w]),
                "max_body_down": np.max([f["body_down"] for f in w]),
                "mean_body_lean_x": np.mean([f["body_lean_x"] for f in w]),
                "mean_reach": np.mean([f["reach"] for f in w]),
            }

            X = pd.DataFrame([row])  # QUAN TRỌNG
            pred = int(clf.predict(X)[0])

            for f in range(w[0]["frame"], w[-1]["frame"] + 1):
                pred_frames[pid][f] = max(pred_frames[pid].get(f, 0), pred)

    # ---------------- VISUALIZE ----------------
    for pid, box in tracks.items():
        x1, y1, x2, y2 = map(int, box)
        is_cheat = pred_frames.get(pid, {}).get(frame_idx, 0) == 1

        color = (0, 0, 255) if is_cheat else (0, 255, 0)
        text = "CHEATING (ML)" if is_cheat else "NORMAL"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {pid} {text}",
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Cheating Detection - End2End", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
