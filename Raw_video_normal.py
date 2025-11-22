import cv2 
import os
from tqdm import tqdm # Thư viện hiển thị thanh tiến trình

# =============== Cấu hình ===============
INPUT_VIDEO = "/media/pphong/D:/git&github/classroom-cheating/data_cheating/normal1.mp4"           # Đường dẫn video đầu vào
OUTPUT_DIR = "/media/pphong/D:/git&github/classroom-cheating/output_video/normal"          # Đường dẫn video đầu ra
CLIP_DURATION = 5          # Thời lượng mỗi clip (giây)
TARGET_RES = (1280, 720)   # Độ phân giải mục tiêu (width, height)
FPS_TARGET = 30            # FPS mục tiêu

# =============== Tao thu muc luu video ===============
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============== Mở video đầu vào ===============
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))        # lay fps goc - lenh get de lay thong tin - CAP_PROP_FPS la thuoc tinh fps

if fps <= 0:
    fps = FPS_TARGET     # Neu fps goc = 0, dat mac dinh la FPS_TARGET
    
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # lay tong so frame trong video
duration = frame_total / fps                          # tinh thoi luong video (giay)
print(f"tong frame: {frame_total}, fps: {fps}, thoi luong video: {duration:.2f} giay")      # in thong tin video


# =============== doc va ghi clips video ===============
frames_per_clip = CLIP_DURATION * fps   # so frame moi clip nghia la thoi luong clip * fps
num_clips = int(frame_total / frames_per_clip) + 1     # tinh so clip can cat
print(f"cat thanh cong {num_clips}, moi clip {CLIP_DURATION} giay ({frames_per_clip} frames)")

# =============== doc va ghi clip =================
clip_idx = 0    # chi so clip hien tai
frame_idx = 0   # chi so frame hien tai trong clip
out_writer = None  # doi tuong ghi video

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # dinh dang video mp4

pbar = tqdm(total=frame_total, desc="Dang xu ly video")  # tao thanh tien trinh - tqdm de hien thi tien trinh - total la tong so frame - desc la mo ta tien trinh

while True:
    ret , frame = cap.read()    # ret: bieu thi viec doc frame co thanh cong hay khong, frame: khung hinh doc duoc
    if not ret:
        break
    
    # chuan hoa kich thuoc frame
    frame = cv2.resize(frame, TARGET_RES)
    
    # Tạo writer mới nếu bắt đầu clip mới
    if frame_idx % frames_per_clip == 0:
        if out_writer is not None:
            out_writer.release()
        clip_idx += 1
        clip_name = f"{OUTPUT_DIR}/normal_clip_{clip_idx:03d}.mp4"
        out_writer = cv2.VideoWriter(clip_name, fourcc, FPS_TARGET, TARGET_RES)
        print(f"--> Bắt đầu clip {clip_idx}")

    # Ghi frame vào clip
    out_writer.write(frame)
    frame_idx += 1
    pbar.update(1)

# ====== Kết thúc ======
pbar.close()
cap.release()
if out_writer is not None:
    out_writer.release()

print(f" Hoàn tất! Đã tạo {clip_idx} clip tại: {OUTPUT_DIR}")
    