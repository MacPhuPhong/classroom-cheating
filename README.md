# 🎓 XGBoost Exam Cheating Detection Dataset

Phát hiện hành vi gian lận trong phòng thi bằng YOLO-Pose Skeleton Extraction, IoU Tracking, và XGBoost Classification.

## ✨ Giới thiệu

Dự án này xây dựng pipeline đầy đủ để thu thập, xử lý và chuẩn hóa dữ liệu nhằm huấn luyện mô hình XGBoost phát hiện hành vi gian lận trong phòng thi dựa trên keypoint skeleton.
Toàn bộ quy trình được mô phỏng từ việc quay video, cắt clip, trích xuất keypoints, tracking đối tượng cho đến chuẩn hóa 51 đặc trưng cho huấn luyện. 

---

## 🧩 Mục lục
1. 🎬 Mô phỏng & Thu thập dữ liệu
2. 📹 Xử lý video thô
3. 🦴 Trích xuất skeleton bằng YOLO-Pose
4. 🧭 IoU Tracking
5. 📊 Chuẩn hóa dữ liệu cho XGBoost
6. 🗂️ Cấu trúc dataset cuối
7. 🔗 Pipeline tổng quan

---

## 🎬 1. Mô phỏng & Thu thập dữ liệu

### 🏫 Phòng thi mô phỏng
- Bàn ghế bố trí như phòng thi thật  
- Mỗi cảnh quay: 1–3 thí sinh mô phỏng tự nhiên  
- Mỗi clip có 2 người tham gia  

### 🕵️ Các hành vi thu thập
- 👀 Nhìn bài người khác  
- 🔄 Đổi bài  
- 📄 Dùng tài liệu  
- 🤫 Trao đổi bằng lời/ra hiệu  
- ✍️ Sao chép bài  
- 🙂 Hành vi bình thường (non-cheating)

### 🎞️ Thời lượng video
- Mỗi video: 1–3 phút  
- 50% bình thường – 50% gian lận  

---

## 📹 2. Xử lý video thô

### 🛠 Các bước xử lý
- Chuẩn hóa video → 1280×720  
- Loại bỏ âm thanh  
- Cắt thành 710 clips, mỗi clip 4–6 giây (~100–130 frames)

### 🏷️ Nhãn clips
- 400 clips non-cheating  
- 310 clips cheating  
- Cheating gồm 5 hành vi chi tiết  

---

## 🦴 3. Trích xuất skeleton bằng YOLO-Pose

### 🎯 Keypoint Extraction
- Mỗi frame: YOLO-Pose phát hiện người  
- Mỗi người → 18 keypoints × (x, y, confidence)

### 🗂️ Cấu trúc file CSV
kp0_x, kp0_y, kp0_conf, ..., kp17_x, kp17_y, kp17_conf, label

### 🏷️ Nhãn frame
- 1 = gian lận  
- 0 = bình thường  

---

## 🧭 4. IoU Tracking

### 🚨 Vấn đề
YOLO-Pose độc lập từng frame → dữ liệu có thể tráo người.

### ✅ Giải pháp
- So sánh IoU giữa box ở frame t và t+1  
- IoU ≥ 0.5 → cùng một người  
- Gán ID cố định xuyên suốt video  

### 🎉 Kết quả
Person_1: frame001 → frame120  
Person_2: frame001 → frame120  

---

## 📊 5. Chuẩn hóa dữ liệu cho XGBoost

### 🔧 Chuyển đổi frame → vector đặc trưng
- Ban đầu: 54 thông số  
- Loại bỏ 3 keypoints không quan trọng  
- Còn lại 51 features mỗi frame  

### 📐 Cấu trúc đầu vào
- Input matrix: (n_samples × 51)  
- Một dòng = 1 frame  

### 🧮 Tổng hợp dataset
- Gộp toàn bộ CSV  
- Mỗi dòng chứa 51 features + 1 label  

---

## 🗂️ 6. Cấu trúc dataset cuối

| Loại | Số lượng |
|------|---------|
| Cheating | 10,277 |
| Non-cheating | 35,258 |
| **Tổng** | **45,535** |

### 📌 Train/Test Split
- 80% training  
- 20% testing  

---

## 🔗 7. Pipeline tổng quan

🎥 VIDEO  
↓  
🦴 YOLO-Pose  
↓  
🧭 IoU Tracking  
↓  
📊 Chuẩn hóa 51 features  
↓  
🗂️ Gộp CSV  
↓  
🔥 Train/Test Split  
↓  
💡 XGBoost Training