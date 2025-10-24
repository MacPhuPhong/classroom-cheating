import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ===============================
# --- CẤU HÌNH ---
# ===============================
VIDEO_PATH = r"/media/pphong/D:/git&github/classroom-cheating/vid_output/vid_output_3.mp4"
SAVE_NORMAL = r"/media/pphong/D:/git&github/classroom-cheating/dataset_path/Normal"
SAVE_SUSPICIOUS = r"/media/pphong/D:/git&github/classroom-cheating/dataset_path/Suspicious"
CSV_PATH = r"/media/pphong/D:/git&github/classroom-cheating/dataset_path/CSV_keypoint/keypoints_dataset.csv"
MODEL_PATH = r"/media/pphong/D:/git&github/classroom-cheating/yolo11s-pose.pt"
OUTPUT_VIDEO = r"/media/pphong/D:/git&github/classroom-cheating/vid_output/vid_processed.mp4"

DEBUG = True  # hiển thị trong khi chạy

# --- Tạo thư mục ---
os.makedirs(SAVE_NORMAL, exist_ok=True)
os.makedirs(SAVE_SUSPICIOUS, exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

# --- Load YOLO Pose ---
model = YOLO(MODEL_PATH)

# ===============================
# --- HÀM HỖ TRỢ ---
# ===============================
def detect_color_boxes(frame, color, min_area=3000):
    """Phát hiện khung viền đỏ hoặc xanh."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if color == "red":
        lower1, upper1 = np.array([0,100,100]), np.array([10,255,255])
        lower2, upper2 = np.array([160,100,100]), np.array([179,255,255])
        mask = cv2.bitwise_or(cv2.inRange(hsv,lower1,upper1),
                              cv2.inRange(hsv,lower2,upper2))
    elif color == "green":
        lower, upper = np.array([35,70,70]), np.array([85,255,255])
        mask = cv2.inRange(hsv,lower,upper)
    else:
        raise ValueError("Color must be 'red' or 'green'")
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask = cv2.dilate(mask,kernel,iterations=2)
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        boxes.append((x,y,x+w,y+h))
    return boxes

def shrink_box(box, margin=6):
    """Co khung lại một chút để loại viền màu."""
    x1,y1,x2,y2 = box
    return (x1+margin, y1+margin, x2-margin, y2-margin)

def overlap_ratio(a,b):
    """Tính phần trăm chồng lấn giữa 2 khung (để lọc trùng)."""
    x1,y1,x2,y2=a; X1,Y1,X2,Y2=b
    xi1,yi1=max(x1,X1),max(y1,Y1)
    xi2,yi2=min(x2,X2),min(y2,Y2)
    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
    if inter==0: return 0
    areaA=(x2-x1)*(y2-y1); areaB=(X2-X1)*(Y2-Y1)
    return inter/float(min(areaA,areaB))

def detect_persons_in_box(color_box, frame, model, conf_thres=0.7):
    """Chạy YOLO trên vùng khung để tìm người và keypoints."""
    x1,y1,x2,y2 = map(int, shrink_box(color_box,6))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return []

    results = model(crop, verbose=False)
    r = results[0]
    persons = []

    if r.boxes is not None and len(r.boxes) > 0:
        for i, (b, conf, cls) in enumerate(zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)):
            if conf < conf_thres:
                continue
            if int(cls) != 0:  # chỉ giữ class "person"
                continue
            bx1,by1,bx2,by2 = map(int, b.tolist())
            if (bx2-bx1)<20 or (by2-by1)<40:
                continue

            # Map to full frame coordinates
            full_box = (bx1+x1, by1+y1, bx2+x1, by2+y1)
            # Lấy keypoints
            try:
                keypoints = r.keypoints.xy[i].cpu().numpy().tolist()
            except Exception:
                keypoints = []
            persons.append((full_box, keypoints))
    return persons

# ===============================
# --- CHẠY VIDEO ---
# ===============================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_idx = 0
person_idx = 0
data_records = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    red_boxes = detect_color_boxes(frame, "red")
    green_boxes = detect_color_boxes(frame, "green")

    persons = []
    for rb in red_boxes:
        for (box, kp) in detect_persons_in_box(rb, frame, model):
            persons.append(("Suspicious", box, kp))
    for gb in green_boxes:
        for (box, kp) in detect_persons_in_box(gb, frame, model):
            persons.append(("Normal", box, kp))

    for label, box, kps in persons:
        x1,y1,x2,y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Lưu ảnh
        save_dir = SAVE_SUSPICIOUS if label=="Suspicious" else SAVE_NORMAL
        filename = f"{label.lower()}_{frame_idx:05d}_{person_idx}.jpg"
        cv2.imwrite(os.path.join(save_dir, filename), crop)
        person_idx += 1

        # Ghi vào CSV
        record = {"image": filename, "label": label}
        for i, kp in enumerate(kps):
            if len(kp) >= 2:
                record[f"x{i}"] = kp[0]
                record[f"y{i}"] = kp[1]
        data_records.append(record)

        # Vẽ box lên frame
        color = (0,0,255) if label=="Suspicious" else (0,255,0)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)
    if DEBUG:
        cv2.imshow("Processed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"✅ Frame {frame_idx}: saved {person_idx} crops")

cap.release()
out.release()
cv2.destroyAllWindows()

# --- Lưu CSV ---
df = pd.DataFrame(data_records)
df.to_csv(CSV_PATH, index=False)
print(f"\n🎬 DONE — {person_idx} persons saved.")
print(f"🧾 Keypoints CSV saved: {CSV_PATH}")
print(f"Output video: {OUTPUT_VIDEO}")
