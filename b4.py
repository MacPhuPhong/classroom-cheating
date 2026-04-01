# ============================================================
# B4 INTERACTIVE LABELING APP (TKINTER)
# + MULTI-ID SELECT
# + SEEK BAR (TIME SLIDER)
# ============================================================

import os, re
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# ============================================================
# CONFIG
# ============================================================
B3_DIR    = "/media/pphong/D:/git&github/classroom-cheating/out_b3"
TRACK_DIR = "/media/pphong/D:/git&github/classroom-cheating/out_b2/csv"
OUT_DIR   = "/media/pphong/D:/git&github/classroom-cheating/out_b4"
os.makedirs(OUT_DIR, exist_ok=True)

K_YAW = 4.0
K_DOWN = 4.0

# ============================================================
# HELPERS
# ============================================================
def parse_video_id(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return m.group(1) if m else None

def rolling_median(s, k=5):
    return s.rolling(k, center=True, min_periods=1).median()

def compute_rule_labels(features_csv):
    df = pd.read_csv(features_csv)
    df["frame"] = df["frame"].astype(int)
    df["person_id"] = df["person_id"].astype(int)

    labels = {}
    max_frame = df["frame"].max()

    for pid in df.person_id.unique():
        g = df[df.person_id == pid].sort_values("frame")
        yaw = rolling_median(g["head_yaw"].abs().fillna(0)).values
        down = rolling_median(g["body_down_lean"].fillna(0)).values

        lab = np.zeros(max_frame + 1, dtype=np.uint8)

        yaw_med, yaw_mad = np.median(yaw), np.median(np.abs(yaw - np.median(yaw))) + 1e-6
        down_med, down_mad = np.median(down), np.median(np.abs(down - np.median(down))) + 1e-6

        for f, y, d in zip(g.frame, yaw, down):
            if (y - yaw_med) / yaw_mad > K_YAW or (d - down_med) / down_mad > K_DOWN:
                lab[f] = 1

        labels[pid] = lab

    return labels

# ============================================================
# APP
# ============================================================
class LabelApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Cheating Labeler (Multi-ID + Seek)")

        self.cap = None
        self.video_id = None
        self.total_frames = 0
        self.cur_frame = 0
        self.playing = False

        self.df_track = None
        self.labels = {}
        self.overrides = []

        self.active_pids = set()
        self.mark_a = None
        self.mark_b = None

        self.id_vars = {}

        self._build_ui()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------
    def _build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(top, text="Open Video", command=self.open_video).pack(side=tk.LEFT)
        ttk.Button(top, text="Play / Pause", command=self.toggle_play).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=4)

        self.status = tk.StringVar(value="Ready")
        ttk.Label(top, textvariable=self.status).pack(side=tk.LEFT, padx=15)

        self.video_label = ttk.Label(self.root)
        self.video_label.pack()

        # -------- SEEK BAR --------
        seek = ttk.Frame(self.root)
        seek.pack(fill=tk.X, padx=8)

        self.frame_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(
            seek, from_=0, to=100,
            orient=tk.HORIZONTAL,
            command=self.on_seek
        )
        self.slider.pack(fill=tk.X)

        self.frame_info = ttk.Label(seek, text="Frame: 0 / 0")
        self.frame_info.pack(anchor=tk.E)

        # -------- ID PANEL --------
        self.id_box = ttk.LabelFrame(self.root, text="Select person_id (multi)")
        self.id_box.pack(fill=tk.X, padx=6, pady=4)

        # -------- CONTROL --------
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill=tk.X, padx=6, pady=6)

        ttk.Button(ctrl, text="Mark A", command=self.set_a).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Mark B", command=self.set_b).pack(side=tk.LEFT)

        self.label_var = tk.StringVar(value="CHEATING")
        ttk.Combobox(ctrl, textvariable=self.label_var,
                     values=["NORMAL", "CHEATING"], width=10).pack(side=tk.LEFT, padx=6)

        self.reason_var = tk.StringVar(value="MANUAL")
        ttk.Combobox(ctrl, textvariable=self.reason_var,
                     values=["LOOK_BACK", "CRAWL", "INTERACT", "MANUAL"], width=10).pack(side=tk.LEFT, padx=6)

        ttk.Button(ctrl, text="Apply Override", command=self.apply_override).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=6)

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------
    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi")])
        if not path:
            return

        vid = parse_video_id(path)
        if not vid:
            messagebox.showerror("Error", "Video filename must contain numeric ID")
            return

        feat_csv = os.path.join(B3_DIR, f"features_csv_{vid}.csv")
        track_csv = os.path.join(TRACK_DIR, f"tracking_csv_{vid}.csv")

        if not os.path.exists(feat_csv) or not os.path.exists(track_csv):
            messagebox.showerror("Error", "Missing CSV files")
            return

        self.cap = cv2.VideoCapture(path)
        self.video_id = vid
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cur_frame = 0

        self.slider.config(to=self.total_frames - 1)

        self.df_track = pd.read_csv(track_csv)
        self.df_track["frame"] = self.df_track["frame"].astype(int)

        self.labels = compute_rule_labels(feat_csv)
        self.overrides.clear()
        self.active_pids.clear()

        self.status.set(f"Loaded video {vid}")
        self.render()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.play_loop()

    def play_loop(self):
        if not self.playing:
            return
        self.cur_frame += 1
        if self.cur_frame >= self.total_frames:
            self.playing = False
            return
        self.slider.set(self.cur_frame)
        self.render()
        self.root.after(35, self.play_loop)

    def on_seek(self, val):
        self.cur_frame = int(float(val))
        self.render()

    # --------------------------------------------------------
    # RENDER
    # --------------------------------------------------------
    def render(self):
        if not self.cap:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame)
        ok, frame = self.cap.read()
        if not ok:
            return

        draw = frame.copy()

        rows = self.df_track[self.df_track.frame == self.cur_frame]
        self.update_id_panel(rows.person_id.unique())

        for _, r in rows.iterrows():
            pid = int(r.person_id)
            x1,y1,x2,y2 = map(int, [r.x1,r.y1,r.x2,r.y2])

            label = self.get_label(pid, self.cur_frame)
            color = (0,0,255) if label == 1 else (0,255,0)
            thick = 4 if pid in self.active_pids else 2

            cv2.rectangle(draw, (x1,y1), (x2,y2), color, thick)
            cv2.putText(draw, f"ID {pid}",
                        (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        img = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)).resize((960,540))
        )
        self.video_label.configure(image=img)
        self.video_label.image = img

        self.frame_info.config(text=f"Frame: {self.cur_frame} / {self.total_frames}")

    # --------------------------------------------------------
    # ID MULTI-SELECT
    # --------------------------------------------------------
    def update_id_panel(self, pids):
        for w in self.id_box.winfo_children():
            w.destroy()
        self.id_vars.clear()

        for pid in sorted(pids):
            var = tk.IntVar(value=1 if pid in self.active_pids else 0)
            cb = tk.Checkbutton(
                self.id_box, text=f"ID {pid}",
                variable=var,
                command=lambda p=pid, v=var: self.toggle_pid(p, v)
            )
            cb.pack(side=tk.LEFT, padx=4)
            self.id_vars[pid] = var

    def toggle_pid(self, pid, var):
        if var.get() == 1:
            self.active_pids.add(pid)
        else:
            self.active_pids.discard(pid)
        self.status.set(f"Active IDs: {sorted(self.active_pids)}")
        self.render()

    # --------------------------------------------------------
    # MANUAL LABEL
    # --------------------------------------------------------
    def set_a(self):
        if not self.active_pids:
            messagebox.showwarning("No ID", "Select at least one ID")
            return
        self.mark_a = self.cur_frame
        self.status.set(f"A = {self.mark_a}")

    def set_b(self):
        if not self.active_pids:
            messagebox.showwarning("No ID", "Select at least one ID")
            return
        self.mark_b = self.cur_frame
        self.status.set(f"B = {self.mark_b}")

    def apply_override(self):
        if not self.active_pids or self.mark_a is None or self.mark_b is None:
            messagebox.showwarning("Missing", "Select IDs and A/B")
            return

        a,b = sorted([self.mark_a, self.mark_b])
        label = 1 if self.label_var.get() == "CHEATING" else 0
        reason = self.reason_var.get()

        self.overrides.append({
            "pids": list(self.active_pids),
            "a": a,
            "b": b,
            "label": label,
            "reason": reason
        })
        self.status.set("Override applied")

    def undo(self):
        if self.overrides:
            self.overrides.pop()
            self.status.set("Undo last override")

    # --------------------------------------------------------
    # LABEL RESOLVE
    # --------------------------------------------------------
    def get_label(self, pid, frame):
        for o in reversed(self.overrides):
            if pid in o["pids"] and o["a"] <= frame <= o["b"]:
                return o["label"]

        if pid in self.labels and frame < len(self.labels[pid]):
            return int(self.labels[pid][frame])
        return 0

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------
    def export_csv(self):
        rows = []
        for o in self.overrides:
            for pid in o["pids"]:
                rows.append({
                    "person_id": pid,
                    "start_frame": o["a"],
                    "end_frame": o["b"],
                    "label": o["label"],
                    "reason": o["reason"],
                    "source": "MANUAL"
                })

        out = os.path.join(OUT_DIR, f"labels_csv_{self.video_id}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        messagebox.showinfo("Saved", f"Saved {out}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1150x820")
    LabelApp(root)
    root.mainloop()
