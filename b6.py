import pandas as pd
import numpy as np
import os, glob, re

# =====================================================
# CONFIG
# =====================================================
B3_DIR = "/media/pphong/D:/git&github/classroom-cheating/out_b3"
B4_DIR = "/media/pphong/D:/git&github/classroom-cheating/out_b4"
OUT_CSV = "/media/pphong/D:/git&github/classroom-cheating/out_b6/temporal_dataset.csv"
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

WINDOW_SIZE = 20
WINDOW_STRIDE = 20

# =====================================================
# HELPERS
# =====================================================
def parse_video_id(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return m.group(1) if m else None

def window_label_from_segments(df_seg, ws, we):
    """
    df_seg: rows of B4 for 1 person_id
    """
    hit = df_seg[
        (df_seg.label == 1) &
        (df_seg.start_frame <= ws) &
        (df_seg.end_frame >= we)
    ]
    return 1 if not hit.empty else 0

# =====================================================
# MAIN
# =====================================================
rows_out = []

feature_files = sorted(glob.glob(os.path.join(B3_DIR, "features_csv_*.csv")))
print(f"Found {len(feature_files)} feature files")

for feat_path in feature_files:
    video_id = parse_video_id(feat_path)
    if not video_id:
        continue

    label_csv = os.path.join(B4_DIR, f"labels_csv_{video_id}.csv")
    if not os.path.exists(label_csv):
        print(f" Missing labels for video {video_id}, skip")
        continue

    print(f"\n Processing video {video_id}")

    df_feat = pd.read_csv(feat_path)
    df_lab  = pd.read_csv(label_csv)

    # enforce schema
    df_feat["frame"] = df_feat["frame"].astype(int)
    df_feat["person_id"] = df_feat["person_id"].astype(int)

    df_lab = df_lab[["person_id","start_frame","end_frame","label"]]
    for c in ["person_id","start_frame","end_frame","label"]:
        df_lab[c] = df_lab[c].astype(int)

    for pid, g in df_feat.groupby("person_id"):
        g = g.sort_values("frame").reset_index(drop=True)
        frames = g.frame.values
        if len(frames) < WINDOW_SIZE:
            continue

        segs = df_lab[df_lab.person_id == pid]

        for i in range(0, len(frames) - WINDOW_SIZE + 1, WINDOW_STRIDE):
            ws = frames[i]
            we = frames[i + WINDOW_SIZE - 1]

            label = window_label_from_segments(segs, ws, we)

            # =========================
            # FEATURE AGGREGATION
            # =========================
            w = g.iloc[i:i+WINDOW_SIZE]

            row = {
                "video_id": video_id,
                "person_id": pid,
                "window_start": ws,
                "window_end": we,

                # head
                "mean_head_yaw": w.head_yaw.mean(),
                "max_head_yaw": w.head_yaw.abs().max(),

                # body
                "mean_body_down": w.body_down_lean.mean(),
                "max_body_down": w.body_down_lean.max(),

                # lean
                "mean_body_lean_x": w.body_lean_x.mean(),

                # interaction
                "mean_reach": w.reach_norm.mean() if "reach_norm" in w else 0.0,

                "label": label
            }

            rows_out.append(row)

# =====================================================
# SAVE
# =====================================================
df_out = pd.DataFrame(rows_out)
df_out.to_csv(OUT_CSV, index=False)

print(f"\n Saved temporal dataset: {OUT_CSV}")
print(df_out["label"].value_counts())