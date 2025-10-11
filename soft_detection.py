# updated TKsoft.py
import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox, Scrollbar
from PIL import Image, ImageTk
import threading

# 🔹 Dùng YOLO detect người
model = YOLO("yolov8n.pt")

class HighlightApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Person Tracking - Green & Red Boxes")
        self.root.geometry("1150x750")
        self.root.resizable(False, False)

        # Video variables
        self.capture = None
        self.video_writer = None
        self.running = False
        self.paused = False
        self.frame = None
        self.annotated_frame = None
        self.original_frame = None
        self.fps = 30
        self.out_w, self.out_h = 960, 540

        # Tracking info
        self.selected_ids = set()
        self.track_ids_ui = set()
        self.checkbuttons = {}
        self.check_vars = {}
        self.color_mode = True  # luôn bật highlight logic

        self.setup_ui()

    def setup_ui(self):
        # --- Video display ---
        video_frame = tk.Frame(self.root)
        video_frame.pack(pady=5)
        self.video_label = tk.Label(video_frame)
        self.video_label.pack()

        # --- Track ID checkboxes ---
        track_id_frame = tk.Frame(self.root)
        track_id_frame.pack(padx=10, pady=10, fill="x")
        canvas = tk.Canvas(track_id_frame, height=60)
        h_scroll = Scrollbar(track_id_frame, orient="horizontal", command=canvas.xview)
        canvas.configure(xscrollcommand=h_scroll.set)
        h_scroll.pack(side="bottom", fill="x")
        canvas.pack(side="top", fill="x")
        self.track_id_inner = tk.Frame(canvas)
        canvas.create_window((0, 0), window=self.track_id_inner, anchor="nw")
        self.track_id_inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # --- Buttons ---
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Start", command=self.start_video,
                  bg="#4CAF50", fg="white", width=15).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Pause", command=self.pause_video,
                  width=10).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Resume", command=self.resume_video,
                  width=10).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Quit", command=self.quit_app,
                  bg="#f44336", fg="white", width=10).pack(side="left", padx=5)

    # --- Các hàm UI điều khiển ---
    def update_track_id_checkboxes(self, track_ids):
        for track_id in track_ids:
            if track_id not in self.track_ids_ui:
                self.track_ids_ui.add(track_id)
                var = tk.IntVar()
                cb = tk.Checkbutton(
                    self.track_id_inner,
                    text=f"Highlight ID {track_id}",
                    variable=var,
                    command=self.update_selected_ids
                )
                cb.pack(side="left", padx=5)
                self.checkbuttons[track_id] = cb
                self.check_vars[track_id] = var

    def update_selected_ids(self):
        self.selected_ids = {tid for tid, var in self.check_vars.items() if var.get() == 1}

    def pause_video(self):
        self.paused = True

    def resume_video(self):
        self.paused = False

    def quit_app(self):
        self.running = False
        if self.capture:
            self.capture.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        self.root.quit()

    # --- Xử lý video ---
    def start_video(self):
        input_path = "/media/pphong/D:/git&github/classroom-cheating/data_cheating/video6.mp4"
        output_path = "/media/pphong/D:/git&github/classroom-cheating/vid_output/vid_output_6.mp4"

        self.capture = cv2.VideoCapture(input_path)
        if not self.capture.isOpened():
            messagebox.showerror("Error", f"Không thể mở video:\n{input_path}")
            return

        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS)) or 30
        self.video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps, (self.out_w, self.out_h)
        )

        self.running = True
        self.paused = False
        threading.Thread(target=self.process_video).start()

    def process_video(self):
        delay = int(1000 / self.fps)

        def update():
            if not self.running:
                return

            if not self.paused:
                ret, frame = self.capture.read()
                if not ret:
                    self.capture.release()
                    return

                frame = cv2.resize(frame, (self.out_w, self.out_h))
                results = model.track(frame, persist=True, classes=[0])  # chỉ detect person

                if results and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.int().cpu().tolist()
                    track_ids = results[0].boxes.id.int().cpu().tolist()

                    self.update_track_id_checkboxes(track_ids)

                    # Mỗi người 1 khung duy nhất
                    filtered = {}
                    for box, tid in zip(boxes, track_ids):
                        x1, y1, x2, y2 = box
                        area = (x2 - x1) * (y2 - y1)
                        if tid not in filtered or area > filtered[tid][1]:
                            filtered[tid] = (box, area)

                    for tid, (box, _) in filtered.items():
                        x1, y1, x2, y2 = box
                        color = (0, 0, 255) if tid in self.selected_ids else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ID:{tid}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # hiển thị
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
                self.video_label.config(image=imgtk)
                self.video_label.image = imgtk
                self.video_writer.write(frame)

            self.root.after(delay, update)

        update()

if __name__ == "__main__":
    root = tk.Tk()
    app = HighlightApp(root)
    root.mainloop()
