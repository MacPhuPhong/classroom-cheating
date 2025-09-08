import os 
import cv2
from ultralytics import YOLO
import pandas as pd

# YOLO model
model = YOLO("yolo11s-pose.pt")

# Video path 
cap = cv2.VideoCapture('vid_output.mp4')

# Get video properties
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
seconds = round(frames / fps)

frame_total = 1000
i = 0 
a = 0

all_data = []

while cap.isOpened():
    # set the position in milliseconds
    cap.set(cv2.CAP_PROP_POS_MSEC, (i * ((seconds/frame_total)*1000)))
    flag, frame = cap.read()
    if not flag:
        break
    # save full frame image
    pa = r'D:\git&github\classroom-cheating\dataset_path\images'
    os.makedirs(pa, exist_ok=True)
    image_path = os.path.join(pa, f'img_{i}.jpg')
    cv2.imwrite(image_path, frame)
    
    # run yolo detection
    results = model(frame, verbose=False)
    
    for r in results:
        bound_box = r.boxes.xyxy  # get bounding boxes
        conf = r.boxes.conf.tolist()  # confidence score
        keypoints = r.keypoints.xyn.tolist()  # human keypoints
        
        for index, box in enumerate(bound_box):
            if conf[index] > 0.75:
                x1, y1, x2, y2 = box.tolist()
                cropped_person = frame[int(y1):int(y2), int(x1):int(x2)]
                
                # Chọn thư mục Normal để lưu
                op = r'D:\git&github\classroom-cheating\dataset_path\Normal'
                os.makedirs(op, exist_ok=True)
                output_path = os.path.join(op, f'person_nn_{a}.jpg')
                
                data = {'image_name': f'person_nn_{a}.jpg'}
                
                # save keypoint data
                for j in range(len(keypoints[index])):
                    data[f'x{j}'] = keypoints[index][j][0]
                    data[f'y{j}'] = keypoints[index][j][1]
                
                # Thêm label mặc định = Normal
                data['label'] = 'Normal'
                
                all_data.append(data)
                cv2.imwrite(output_path, cropped_person)
                a += 1
    i += 1

print(f"Total frames processed: {i-1}, Total cropped images saved: {a-1}")
cap.release()
cv2.destroyAllWindows()

# Convert to DataFrame
df = pd.DataFrame(all_data)

csv_file_path = r'D:\git&github\classroom-cheating\dataset_path\normal_keypoint.csv'   

# Lưu file, nếu đã tồn tại thì append
if not os.path.isfile(csv_file_path):
    df.to_csv(csv_file_path, index=False)
else:
    df.to_csv(csv_file_path, mode='a', header=False, index=False)
    
print(f"Keypoint data with labels saved to {csv_file_path}")
