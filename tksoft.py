import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox, Scrollbar
from PIL import Image, ImageTk
import threading

model = YOLO("yolov8n.pt")
names = model.model.names

class BlurApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Track ID Blur Tool with Dual View")
        self.root.geometry("1150x780")
        self.root.resizable(False, False)

        self.capture = None
        self.video_writer = None
        self.running = False
        self.paused = False
        self.frame = None
        self.annotated_frame = None
        self.original_frame = None
        self.blur_mode = False

        self.out_w, self.out_h = 480, 360
        self.fps = 30

        self.selected_ids = set()
        self.track_ids_ui = set()
        self.checkbuttons = {}
        self.check_vars = {}

        self.setup_ui()

    def setup_ui(self):
        # === Frame for Both Video Views ===
        video_frame = tk.Frame(self.root)
        video_frame.pack(pady=5)

        self.video_label_original = tk.Label(video_frame, text="Original Frame")
        self.video_label_original.pack(side="left", padx=5)

        self.video_label = tk.Label(video_frame, text="Blurred Frame")
        self.video_label.pack(side="left", padx=5)

        # === Track ID Checkboxes ===
        track_id_frame = tk.Frame(self.root)
        track_id_frame.pack(padx=10, fill="x")

        canvas = tk.Canvas(track_id_frame, height=60)
        h_scroll = Scrollbar(track_id_frame, orient="horizontal", command=canvas.xview)
        canvas.configure(xscrollcommand=h_scroll.set)

        h_scroll.pack(side="bottom", fill="x")
        canvas.pack(side="top", fill="x")

        self.track_id_inner = tk.Frame(canvas)
        canvas.create_window((0, 0), window=self.track_id_inner, anchor="nw")
        self.track_id_inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # === Control Buttons ===
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Start", command=self.start_video).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Pause", command=self.pause_video).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Resume", command=self.resume_video).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Start Blurring", command=self.enable_blur).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Stop Blurring", command=self.disable_blur).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Quit", command=self.quit_app).pack(side="left", padx=5)

    def update_track_id_checkboxes(self, track_ids):
        for track_id in track_ids:
            if track_id not in self.track_ids_ui:
                self.track_ids_ui.add(track_id)
                var = tk.IntVar()
                cb = tk.Checkbutton(self.track_id_inner, text=f"ID {track_id}", variable=var,
                                    command=self.update_selected_ids)
                cb.pack(side="left", padx=5)
                self.checkbuttons[track_id] = cb
                self.check_vars[track_id] = var

    def update_selected_ids(self):
        self.selected_ids.clear()
        for track_id, var in self.check_vars.items():
            if var.get() == 1:
                self.selected_ids.add(track_id)

    def enable_blur(self):
        self.blur_mode = True
        messagebox.showinfo("Blur", "Blurring started for selected IDs.")

    def disable_blur(self):
        self.blur_mode = False
        messagebox.showinfo("Blur", "Blurring stopped.")

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

    def start_video(self):
        self.running = True
        self.paused = False
        self.selected_ids.clear()
        self.track_ids_ui.clear()
        self.checkbuttons.clear()
        self.check_vars.clear()

        self.capture = cv2.VideoCapture("vid3.mp4")
        if not self.capture.isOpened():
            messagebox.showerror("Error", "Cannot open video file.")
            return

        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0:
            self.fps = 30

        self.video_writer = cv2.VideoWriter("vid_output.mp4",
                                            cv2.VideoWriter_fourcc(*"mp4v"),
                                            self.fps, (self.out_w, self.out_h))

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
                self.original_frame = frame.copy()

                results = model.track(frame, persist=True, classes=[0])

                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.int().cpu().tolist()
                    class_ids = results[0].boxes.cls.int().cpu().tolist()
                    track_ids = results[0].boxes.id.int().cpu().tolist()

                    self.update_track_id_checkboxes(track_ids)

                    for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                        x1, y1, x2, y2 = box
                        roi = frame[y1:y2, x1:x2]

                        if self.blur_mode and track_id in self.selected_ids:
                            blur = cv2.blur(roi, (45, 45))
                            frame[y1:y2, x1:x2] = blur
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f'ID:{track_id}', (x1, y2 + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(frame, names[class_id], (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                self.annotated_frame = frame.copy()
                self.video_writer.write(frame)
                self.display_frame(self.annotated_frame, self.original_frame)
            else:
                if self.annotated_frame is not None:
                    self.display_frame(self.annotated_frame, self.original_frame)

            self.root.after(delay, update)

        update()

    def display_frame(self, annotated, original):
        def to_imgtk(cv_img):
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            return ImageTk.PhotoImage(pil_img)

        self.imgtk_annotated = to_imgtk(annotated)
        self.imgtk_original = to_imgtk(original)

        self.video_label.config(image=self.imgtk_annotated)
        self.video_label.image = self.imgtk_annotated

        self.video_label_original.config(image=self.imgtk_original)
        self.video_label_original.image = self.imgtk_original

if __name__ == "__main__":
    root = tk.Tk()
    app = BlurApp(root)
    root.mainloop()
