import tkinter as tk
from tkinter import ttk
import json
import os

SETTINGS_FILE = "settings.json"

def load_settings():
    defaults = {
        "BUFFER_SIZE": 30,
        "MOTION_THRESHOLD": 15,
        "DILATION_ITERATIONS": 1,
        "WAITING_INTENSITY": 10.0,
        "VOTING_WINDOW": 3,
        "TRACK_BUFFER": 3000,
        "MATCH_THRESH": 0.8,
        "YOLO_CONF": 0.25,
        "YOLO_IOU": 0.1,
        "TRACKER_TYPE": "botsort.yaml"
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
                defaults.update(saved)
        except Exception:
            pass
    return defaults
def save_tracker_yaml(buffer, thresh):
    """Overwrite botsort.yaml with new parameters."""
    content = f"""# Optimized for High-End GPU Accuracy
tracker_type: botsort
track_high_thresh: 0.35      # Lowered slightly to pick up tracks earlier
track_low_thresh: 0.1
new_track_thresh: 0.4       # Easier to start a track
track_buffer: {int(buffer)}
match_thresh: {float(thresh)}
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.5      # Increased for better ReID matching
with_reid: True
model: auto                 # Required by ultralytics >= 8.3.x for ReID model selection
fuse_score: True
"""
    with open("botsort.yaml", "w") as f:
        f.write(content)

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)

class TuningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EagleVision - Control Panel")
        self.root.geometry("420x680")
        self.root.configure(bg='#f7f7f7')
        
        self.settings = load_settings()
        self.controls = {}
        
        self.create_widgets()
 
    def create_widgets(self):
        # ── Scrollable Container ─────────────────────────────────────────────
        canvas = tk.Canvas(self.root, bg='#f7f7f7', highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
 
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=400)
        canvas.configure(yscrollcommand=scrollbar.set)
 
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Header
        header = tk.Label(self.scrollable_frame, text="EagleVision Tuner", font=('Helvetica', 12, 'bold'), bg='#f7f7f7', fg='#222')
        header.pack(pady=5)
 
        # 1. Motion & Heatmap Group
        motion_group = ttk.LabelFrame(self.scrollable_frame, text=" 1. Motion Sensitivity ", padding="5")
        motion_group.pack(fill="x", pady=2, padx=5)
        
        self.add_slider(motion_group, "MOTION_THRESHOLD", "Threshold", 1, 50, 1, True, "Sensitivity.")
        self.add_slider(motion_group, "DILATION_ITERATIONS", "Dilation", 0, 5, 1, True, "Cluster size.")
        self.add_slider(motion_group, "BUFFER_SIZE", "History", 5, 150, 1, True, "Heatmap frames.")
 
        # 2. Activity Logic Group
        activity_group = ttk.LabelFrame(self.scrollable_frame, text=" 2. Activity Logic ", padding="5")
        activity_group.pack(fill="x", pady=2, padx=5)
        
        self.add_slider(activity_group, "WAITING_INTENSITY", "Waiting Floor", 0, 150, 1, False, "Active min.")
 
        # 3. Stability Group
        track_group = ttk.LabelFrame(self.scrollable_frame, text=" 3. ID Stability ", padding="5")
        track_group.pack(fill="x", pady=2, padx=5)
        
        self.add_slider(track_group, "VOTING_WINDOW", "Stability", 1, 30, 1, True, "Anti-flicker.")
        self.add_slider(track_group, "TRACK_BUFFER", "Memory", 100, 200000, 100, True, "Lost frames.")
        self.add_slider(track_group, "MATCH_THRESH", "Leniency", 0.1, 1.0, 0.05, False, "Association.")
 
        # 4. YOLO Detection Group
        yolo_group = ttk.LabelFrame(self.scrollable_frame, text=" 4. YOLO Detections ", padding="5")
        yolo_group.pack(fill="x", pady=2, padx=5)
 
        self.add_slider(yolo_group, "YOLO_CONF", "Conf", 0.01, 1.0, 0.01, False, "Detection.")
        self.add_slider(yolo_group, "YOLO_IOU", "IoU", 0.01, 0.9, 0.01, False, "Overlap.")
 
        # 5. Tracker Selection Group
        tracker_group = ttk.LabelFrame(self.scrollable_frame, text=" 5. Tracking Algorithm ", padding="5")
        tracker_group.pack(fill="x", pady=2, padx=5)
        
        self.tracker_var = tk.StringVar(value=self.settings.get("TRACKER_TYPE", "botsort.yaml"))
        
        s = ttk.Style()
        s.configure('Small.TRadiobutton', font=('Helvetica', 8))
        
        ttk.Radiobutton(
            tracker_group, text="BoT-SORT (Balance)", 
            variable=self.tracker_var, value="botsort.yaml",
            command=self.update_tracker_type, style='Small.TRadiobutton'
        ).pack(anchor="w")
        
        ttk.Radiobutton(
            tracker_group, text="StrongSORT++ (Accuracy)", 
            variable=self.tracker_var, value="strongsort.yaml",
            command=self.update_tracker_type, style='Small.TRadiobutton'
        ).pack(anchor="w")
 
        ttk.Radiobutton(
            tracker_group, text="Deep-OC-SORT (Robust Occlusion)", 
            variable=self.tracker_var, value="deepocsort.yaml",
            command=self.update_tracker_type, style='Small.TRadiobutton'
        ).pack(anchor="w")

        # Buttons
        ttk.Button(self.scrollable_frame, text="Reset to Defaults", command=self.reset_defaults).pack(pady=10)
        
        # Small Footer
        tk.Label(self.scrollable_frame, text="Instant Live Tuning Active", font=('Helvetica', 7, 'italic'), fg='gray').pack()

    def add_slider(self, parent, key, label, min_val, max_val, res, is_int, note):
        container = ttk.Frame(parent)
        container.pack(fill="x", pady=1)
        
        lbl_frame = ttk.Frame(container)
        lbl_frame.pack(fill="x")
        
        tk.Label(lbl_frame, text=label, font=('Helvetica', 9, 'bold')).pack(side="left")
        tk.Label(lbl_frame, text=note, font=('Helvetica', 8), fg='gray').pack(side="left", padx=5)
        
        val_var = tk.DoubleVar(value=self.settings.get(key, 0))
        
        slider = ttk.Scale(
            container, from_=min_val, to=max_val, 
            orient="horizontal", variable=val_var,
            command=lambda val, k=key, is_i=is_int: self.on_change(k, val, is_i)
        )
        slider.pack(side="left", fill="x", expand=True, padx=5)
        
        label_val = ttk.Label(container, text=f"{val_var.get()}", width=6, font=('Helvetica', 8))
        label_val.pack(side="right")
        
        self.controls[key] = (val_var, label_val)

    def on_change(self, key, value, is_int):
        if is_int:
            value = int(float(value))
        else:
            value = round(float(value), 2)
            
        self.settings[key] = value
        save_settings(self.settings)
        
        if key in ["TRACK_BUFFER", "MATCH_THRESH"]:
            save_tracker_yaml(self.settings["TRACK_BUFFER"], self.settings["MATCH_THRESH"])
        
        val_var, label_val = self.controls[key]
        label_val.config(text=f"{value}")

    def update_tracker_type(self):
        new_type = self.tracker_var.get()
        self.settings["TRACKER_TYPE"] = new_type
        save_settings(self.settings)
        print(f"[Settings] Switched tracker to: {new_type}")

    def reset_defaults(self):
        defaults = {
            "BUFFER_SIZE": 30,
            "MOTION_THRESHOLD": 15,
            "DILATION_ITERATIONS": 1,
            "WAITING_INTENSITY": 10.0,
            "VOTING_WINDOW": 3,
            "TRACK_BUFFER": 3000,
            "MATCH_THRESH": 0.8,
            "YOLO_CONF": 0.25,
            "YOLO_IOU": 0.1,
            "TRACKER_TYPE": "botsort.yaml"
        }
        self.settings = defaults
        save_settings(self.settings)
        save_tracker_yaml(3000, 0.8)
        for key, val in defaults.items():
            if key in self.controls:
                val_var, label_val = self.controls[key]
                val_var.set(val)
                label_val.config(text=f"{val}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TuningGUI(root)
    root.mainloop()