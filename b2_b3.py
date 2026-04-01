import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os, glob, re

# =====================
# CONFIG
# =====================
VIDEO_DIR = "/media/pphong/D:/git&github/classroom-cheating/datavideo"

OUT_B2_VIDEO = "/media/pphong/D:/git&github/classroom-cheating/out_b2/video"
OUT_B2_CSV   = "/media/pphong/D:/git&github/classroom-cheating/out_b2/csv"
OUT_B3_DIR   = "/media/pphong/D:/git&github/classroom-cheating/out_b3"
os.makedirs(OUT_B2_VIDEO, exist_ok=True)
os.makedirs(OUT_B2_CSV, exist_ok=True)
os.makedirs(OUT_B3_DIR, exist_ok=True)

MODEL_YOLO_PATH = "yolo11s-pose.pt"
CONF_THRESHOLD = 0.3
FRAME_SKIP = 1  # khuyến nghị = 1 để bắt thao tác nhanh

# Tracker params
IOU_TH = 0.30
IOU_LOWER = 0.10
DIST_NORM_TH = 0.70
MAX_AGE = 15

KP_CONF_TH = 0.25     # ngưỡng tin cậy keypoints
WRIST_CONF_TH = 0.35  # ngưỡng tin cậy cổ tay

# =====================
# Utils
# =====================
def iou_xyxy(a, b) -> float:
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-6)

def box_center_diag(box):
    x1,y1,x2,y2 = box
    cx = (x1+x2)/2; cy = (y1+y2)/2
    diag = float(np.hypot(x2-x1, y2-y1)) + 1e-6
    return cx, cy, diag

def safe_mid(p1, c1, p2, c2, th):
    if c1 >= th and c2 >= th:
        return (p1 + p2) / 2.0
    return None

# =====================
# Pose feature compute
# =====================
def compute_features(kp_xy, kp_cf, bbox):
    # COCO indices
    nose, le, re = kp_xy[0], kp_xy[1], kp_xy[2]
    ls, rs = kp_xy[5], kp_xy[6]
    lh, rh = kp_xy[11], kp_xy[12]
    lw, rw = kp_xy[9], kp_xy[10]

    nose_c, le_c, re_c = kp_cf[0], kp_cf[1], kp_cf[2]
    ls_c, rs_c = kp_cf[5], kp_cf[6]
    lh_c, rh_c = kp_cf[11], kp_cf[12]
    lw_c, rw_c = kp_cf[9], kp_cf[10]

    eye_mid = safe_mid(le, le_c, re, re_c, KP_CONF_TH)
    sh_mid  = safe_mid(ls, ls_c, rs, rs_c, KP_CONF_TH)
    hip_mid = safe_mid(lh, lh_c, rh, rh_c, KP_CONF_TH)

    x1,y1,x2,y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    shoulder_width = np.nan
    if ls_c >= KP_CONF_TH and rs_c >= KP_CONF_TH:
        shoulder_width = abs(ls[0] - rs[0])

    scale = max(
        shoulder_width if np.isfinite(shoulder_width) else 0.0,
        0.25 * bw,
        20.0
    )

    # ---------- HEAD ----------
    head_yaw = np.nan
    if nose_c >= KP_CONF_TH and eye_mid is not None:
        head_yaw = (nose[0] - eye_mid[0]) / scale

    # ---------- BODY ----------
    body_lean_x = np.nan
    if sh_mid is not None and hip_mid is not None:
        body_lean_x = (sh_mid[0] - hip_mid[0]) / scale

    body_down_lean = np.nan
    if nose_c >= KP_CONF_TH and hip_mid is not None:
        body_down_lean = (nose[1] - hip_mid[1]) / scale

    # ---------- BODY ROTATION (KEY FIX) ----------
    body_rot_shrink = np.nan
    if np.isfinite(shoulder_width):
        body_rot_shrink = shoulder_width / (bw + 1e-6)

    # ---------- DIAGONAL CRAWL ----------
    crawl_diag = np.nan
    if body_lean_x is not np.nan and body_down_lean is not np.nan:
        crawl_diag = np.hypot(body_lean_x, body_down_lean)

    # ---------- REACH ----------
    reach_norm = np.nan
    reach_dx = np.nan
    reach_dy = np.nan

    if sh_mid is not None:
        wrists = []
        if lw_c >= WRIST_CONF_TH:
            wrists.append(lw)
        if rw_c >= WRIST_CONF_TH:
            wrists.append(rw)

        if wrists:
            dists = [np.linalg.norm(w - sh_mid) for w in wrists]
            idx = int(np.argmax(dists))
            w = wrists[idx]

            reach_norm = dists[idx] / scale
            reach_dx = (w[0] - sh_mid[0]) / scale
            reach_dy = (w[1] - sh_mid[1]) / scale

    return {
        "head_yaw": head_yaw,
        "body_lean_x": body_lean_x,
        "body_down_lean": body_down_lean,
        "body_rot_shrink": body_rot_shrink,
        "crawl_diag": crawl_diag,

        "reach_norm": reach_norm,
        "reach_dx": reach_dx,
        "reach_dy": reach_dy,

        "shoulder_width": shoulder_width,

        "lw_x": float(lw[0]), "lw_y": float(lw[1]),
        "rw_x": float(rw[0]), "rw_y": float(rw[1]),

        "nose_conf": nose_c, "le_conf": le_c, "re_conf": re_c,
        "ls_conf": ls_c, "rs_conf": rs_c,
        "lh_conf": lh_c, "rh_conf": rh_c,
        "lw_conf": lw_c, "rw_conf": rw_c,
    }

# =====================
# MAIN
# =====================
def process_all_videos():
    model = YOLO(MODEL_YOLO_PATH)
    video_paths = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    print(f"Found {len(video_paths)} videos")

    for video_path in video_paths:
        base = os.path.basename(video_path)
        m = re.search(r"(\d+)", base)
        if not m:
            print(f" Cannot parse video_id from {base}, skip")
            continue
        video_id = m.group(1)

        print(f"\n Processing video {video_id}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Cannot open video")
            continue

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = cap.get(cv2.CAP_PROP_FPS)

        out_vid = os.path.join(OUT_B2_VIDEO, f"tracking_video_{video_id}.mp4")
        out_trk = os.path.join(OUT_B2_CSV,   f"tracking_csv_{video_id}.csv")
        out_feat= os.path.join(OUT_B3_DIR,   f"features_csv_{video_id}.csv")

        writer = cv2.VideoWriter(out_vid, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

        tracks = {}  # id -> {"bbox": np.array([x1,y1,x2,y2]), "age": int}
        next_id = 0

        step2_rows = []
        step3_rows = []

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                writer.write(frame)
                continue

            results = model(frame, verbose=False)

            det_boxes, det_kps_xy, det_kps_cf = [], [], []

            for r in results:
                if r.boxes is None or r.keypoints is None:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                kps_xy = r.keypoints.xy.cpu().numpy()
                kps_cf = r.keypoints.conf.cpu().numpy()

                for i, box in enumerate(boxes):
                    if confs[i] < CONF_THRESHOLD:
                        continue
                    det_boxes.append(box.astype(float))
                    det_kps_xy.append(kps_xy[i].astype(float))
                    det_kps_cf.append(kps_cf[i].astype(float))

            # -------- assignment (IoU + center distance) --------
            track_ids = list(tracks.keys())
            pairs = []
            for tid in track_ids:
                tbox = tracks[tid]["bbox"]
                tcx, tcy, tdiag = box_center_diag(tbox)
                for di, dbox in enumerate(det_boxes):
                    iou = iou_xyxy(tbox, dbox)
                    dcx, dcy, _ = box_center_diag(dbox)
                    dist_norm = float(np.hypot(dcx-tcx, dcy-tcy) / tdiag)
                    if (iou >= IOU_TH) or (iou >= IOU_LOWER and dist_norm <= DIST_NORM_TH):
                        score = float(iou - 0.05 * dist_norm)
                        pairs.append((score, tid, di))

            pairs.sort(reverse=True, key=lambda x: x[0])

            assigned_tracks = set()
            assigned_dets = set()
            det_to_id = {}

            for score, tid, di in pairs:
                if tid in assigned_tracks or di in assigned_dets:
                    continue
                det_to_id[di] = tid
                assigned_tracks.add(tid)
                assigned_dets.add(di)

            # new IDs for unmatched dets
            for di in range(len(det_boxes)):
                if di not in det_to_id:
                    det_to_id[di] = next_id
                    tracks[next_id] = {"bbox": det_boxes[di], "age": 0}
                    next_id += 1

            # update matched
            updated = set(det_to_id.values())
            for di, pid in det_to_id.items():
                tracks[pid]["bbox"] = det_boxes[di]
                tracks[pid]["age"] = 0

            # age++ for unmatched tracks
            to_del = []
            for tid in list(tracks.keys()):
                if tid not in updated:
                    tracks[tid]["age"] += 1
                    if tracks[tid]["age"] > MAX_AGE:
                        to_del.append(tid)
            for tid in to_del:
                del tracks[tid]

            # -------- save rows --------
            for di, pid in det_to_id.items():
                box = det_boxes[di]
                kp_xy = det_kps_xy[di]
                kp_cf = det_kps_cf[di]

                x1, y1, x2, y2 = map(int, box)

                step2_rows.append({
                    "frame": frame_idx,
                    "person_id": int(pid),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })

                feat = compute_features(kp_xy, kp_cf, box)
                feat_row = {
                    "frame": frame_idx,
                    "person_id": int(pid),
                    **feat
                }
                step3_rows.append(feat_row)

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"ID {pid}", (x1, max(25, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            writer.write(frame)

        cap.release()
        writer.release()

        pd.DataFrame(step2_rows).to_csv(out_trk, index=False)
        pd.DataFrame(step3_rows).to_csv(out_feat, index=False)

        print(f" Saved: {out_vid}")
        print(f" Saved: {out_trk}")
        print(f" Saved: {out_feat}")

    print("\nALL VIDEOS DONE")

if __name__ == "__main__":
    process_all_videos()
