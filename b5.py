import cv2
import pandas as pd
import os, glob, re

VIDEO_DIR = "/media/pphong/D:/git&github/classroom-cheating/datavideo"
TRACK_DIR = "/media/pphong/D:/git&github/classroom-cheating/out_b2/csv"
LABEL_DIR = "/media/pphong/D:/git&github/classroom-cheating/out_b4"
OUT_DIR   = "/media/pphong/D:/git&github/classroom-cheating/out_b5"
os.makedirs(OUT_DIR, exist_ok=True)

SHOW_REASON = True

video_paths = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
print(f"Found {len(video_paths)} videos")

for video_path in video_paths:
    base = os.path.basename(video_path)
    m = re.search(r"(\d+)", base)
    if not m:
        print(f" Cannot parse video_id from {base}, skip")
        continue
    video_id = m.group(1)

    track_csv = os.path.join(TRACK_DIR, f"tracking_csv_{video_id}.csv")
    label_csv = os.path.join(LABEL_DIR, f"labels_csv_{video_id}.csv")
    out_video = os.path.join(OUT_DIR, f"cheating_visualization_{video_id}.mp4")

    if not os.path.exists(track_csv) or not os.path.exists(label_csv):
        print(f" Missing CSV for video {video_id}, skip")
        continue

    print(f"\n Processing video {video_id}")

    df_track = pd.read_csv(track_csv)
    df_label = pd.read_csv(label_csv)

    df_track["frame"] = df_track["frame"].astype(int)
    for c in ["start_frame","end_frame","highlight_start_frame","highlight_end_frame","person_id","label"]:
        if c in df_label.columns:
            df_label[c] = df_label[c].astype(int)

    has_hl = ("highlight_start_frame" in df_label.columns) and ("highlight_end_frame" in df_label.columns)
    has_reason = "reason" in df_label.columns

    track_by_frame = {fr: grp for fr, grp in df_track.groupby("frame")}
    label_by_pid = {pid: grp for pid, grp in df_label.groupby("person_id")}

    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        rows = track_by_frame.get(frame_idx, None)
        if rows is None:
            writer.write(frame)
            continue

        for _, r in rows.iterrows():
            pid = int(r.person_id)
            x1, y1, x2, y2 = map(int, [r.x1, r.y1, r.x2, r.y2])

            is_cheating = False
            reason = None

            lab = label_by_pid.get(pid, None)
            if lab is not None:
                lab1 = lab[lab["label"] == 1]
                if not lab1.empty:
                    if has_hl:
                        hit = lab1[(lab1.highlight_start_frame <= frame_idx) & (lab1.highlight_end_frame >= frame_idx)]
                    else:
                        hit = lab1[(lab1.start_frame <= frame_idx) & (lab1.end_frame >= frame_idx)]

                    if not hit.empty:
                        is_cheating = True
                        if has_reason:
                            reason = str(hit.iloc[0]["reason"])

            if is_cheating:
                color = (0, 0, 255)
                text = "CHEATING" if not (SHOW_REASON and reason) else f"CHEATING-{reason}"
            else:
                color = (0, 255, 0)
                text = "NORMAL"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {pid} - {text}", (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f" Saved: {out_video}")

print("\nALL VIDEOS PROCESSED")
