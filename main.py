import cv2
import os
import pandas as pd
from ultralytics import YOLO
import xgboost as xgb
import numpy as np
import cvzone

VIDEO_PATH = r"/media/pphong/D:/git&github/classroom-cheating/data_cheating/video3.mp4"
MODEL_YOLO_PATH = "yolo11n.pt"
MODEL_XGB_PATH = r"/media/pphong/D:/git&github/classroom-cheating/trained_model.json"

CONF_THRESHOLD = 0.55
SUSPICIOUS_THRESHOLD = 0.35
DEBUG_SAVE_DIR = "debug_failed_preds"
os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)

def prepare_features_from_kp(kp_xy, trained_features):
    data = {}
    for j, (x, y) in enumerate(kp_xy):
        data[f"x{j}"] = float(x)
        data[f"y{j}"] = float(y)
    df = pd.DataFrame([data])
    df = df.reindex(columns=trained_features, fill_value=0)
    return df

def detect_suspicious_behavior(video_path):
    print(" Loading YOLO pose model...")
    model_yolo = YOLO(MODEL_YOLO_PATH)

    print(" Loading XGBoost model...")
    model = xgb.Booster()
    model.load_model(MODEL_XGB_PATH)
    trained_features = model.feature_names
    print(f" XGBoost loaded with {len(trained_features)} features")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(" Cannot open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f" Total frames: {total_frames}")

    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1
        if frame_idx % 2 != 0:
            continue

        frame = cv2.resize(frame, (1018, 600))
        results = model_yolo(frame, verbose=False)
        annotated_frame = results[0].plot(boxes=False)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes.xyxy is not None else []
            confs = r.boxes.conf.cpu().numpy().tolist() if r.boxes.conf is not None else []
            keypoints = r.keypoints.xyn.cpu().numpy().tolist() if r.keypoints is not None else []

            for idx, box in enumerate(boxes):
                if idx >= len(confs) or idx >= len(keypoints):
                    continue
                if confs[idx] < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box)
                kp = keypoints[idx]

                df = prepare_features_from_kp(kp, trained_features)
                dmatrix = xgb.DMatrix(df)

                pred_prob = float(model.predict(dmatrix)[0])

                #  IN KIỂM TRA XÁC SUẤT
                print(f"Frame {frame_idx}: prob={pred_prob:.3f}")

                #  ĐẢO LOGIC HIỂN THỊ CHO CHẮC
                # Giả sử pred_prob là "xác suất Suspicious"
                pred_label = 1 if pred_prob >= SUSPICIOUS_THRESHOLD else 0

                if pred_label == 1:
                    color = (0, 0, 255)  #  Suspicious
                    label_text = f"Suspicious ({pred_prob:.2f})"
                else:
                    color = (0, 255, 0)  #  Normal
                    label_text = f"Normal ({pred_prob:.2f})"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cvzone.putTextRect(
                    annotated_frame, label_text,
                    (x1, max(0, y1 - 10)), scale=1, thickness=1, colorR=color
                )

                if pred_label == 1 and pred_prob > 0.7:
                    debug_path = os.path.join(DEBUG_SAVE_DIR, f"frame{frame_idx}_p{idx}_prob{pred_prob:.2f}.jpg")
                    cv2.imwrite(debug_path, annotated_frame)

        cv2.imshow("Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_suspicious_behavior(VIDEO_PATH)
