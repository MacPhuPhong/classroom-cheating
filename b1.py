import cv2
from ultralytics import YOLO
import os

# =====================
# CONFIG
# =====================
VIDEO_PATH = "/media/pphong/D:/git&github/classroom-cheating/datavideo/video12.mp4"
MODEL_YOLO_PATH = "/media/pphong/D:/git&github/classroom-cheating/yolo11s-pose.pt"   # PHẢI là pose model

CONF_THRESHOLD = 0.3
FRAME_SKIP = 2                       # xử lý mỗi 2 frame
SAVE_VIDEO = True

OUTPUT_VIDEO = "/media/pphong/D:/git&github/classroom-cheating/out_b1/output_video12.mp4"


# =====================
# MAIN
# =====================
def step1_yolo_pose(video_path):
    print("Loading YOLO pose model...")
    model = YOLO(MODEL_YOLO_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS / FRAME_SKIP, (W, H))

    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        results = model(frame, verbose=False)

        annotated = frame.copy()

        for r in results:
            if r.boxes is None or r.keypoints is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            kps = r.keypoints.xy.cpu().numpy()      # (N, 17, 2)

            for i, box in enumerate(boxes):
                if confs[i] < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box)

                # --- VẼ BOX ---
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # --- VẼ KEYPOINT ---
                for (x, y) in kps[i]:
                    cv2.circle(annotated, (int(x), int(y)), 3, (0, 0, 255), -1)

                # --- TEXT ---
                cv2.putText(
                    annotated,
                    f"Person {i}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("STEP 1 - YOLO POSE", annotated)

        if SAVE_VIDEO:
            writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("STEP 1 DONE")


# =====================
# RUN
# =====================
step1_yolo_pose(VIDEO_PATH)