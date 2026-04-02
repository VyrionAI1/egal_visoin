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
        "MATCH_THRESH": 0.8
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
    content = f"""# Optimized by EagleVision GUI
tracker_type: botsort
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: {int(buffer)}
match_thresh: {float(thresh)}
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: False
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
        self.root.title("EagleVision - Heatmap Control Panel")
        self.root.geometry("500x750")
        self.root.configure(bg='#f0f0f0')
        
        self.settings = load_settings()
        self.controls = {}
        
        self.create_widgets()

    def create_widgets(self):
        # Header
        header = tk.Label(self.root, text="EagleVision Heatmap Tuner", font=('Helvetica', 16, 'bold'), bg='#f0f0f0', fg='#333')
        header.pack(pady=10)

        # Main Container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)

        # 1. Motion & Heatmap Group
        motion_group = ttk.LabelFrame(main_frame, text=" 1. Motion & Heatmap Sensitivity ", padding="10")
        motion_group.pack(fill="x", pady=5)
        
        self.add_slider(motion_group, "MOTION_THRESHOLD", "Motion Threshold", 1, 50, 1, True, "Binary sensitivity.")
        self.add_slider(motion_group, "DILATION_ITERATIONS", "Dilation Blur", 0, 5, 1, True, "Size of motion clusters.")
        self.add_slider(motion_group, "BUFFER_SIZE", "Heatmap History", 5, 150, 1, True, "Rolling frames to average.")

        # 2. Activity Logic Group
        activity_group = ttk.LabelFrame(main_frame, text=" 2. Activity Classification (Intensity Logic) ", padding="10")
        activity_group.pack(fill="x", pady=5)
        
        self.add_slider(activity_group, "WAITING_INTENSITY", "Waiting Floor", 0, 150, 1, False, "Min pixels to be 'Active'.")

        # 3. Stability Group
        track_group = ttk.LabelFrame(main_frame, text=" 3. Stability & ID Optimization ", padding="10")
        track_group.pack(fill="x", pady=5)
        
        self.add_slider(track_group, "VOTING_WINDOW", "Stability Window", 1, 30, 1, True, "Smoother labels (counter-flicker).")
        self.add_slider(track_group, "TRACK_BUFFER", "ID Memory (Frames)", 100, 5000, 100, True, "Keep ID alive if hidden.")
        self.add_slider(track_group, "MATCH_THRESH", "Match Leniency", 0.1, 1.0, 0.05, False, "BoT-SORT strictness.")

        # Quick Guide
        info_frame = ttk.LabelFrame(main_frame, text=" Heatmap Guide ", padding="10")
        info_frame.pack(fill="both", expand=True, pady=10)
        
        guide_text = (
            "• IF TOO MUCH NOISE: Increase 'Motion Threshold'.\n"
            "• IF LABELS FLICKER: Increase 'Stability Window'.\n\n"
            "Settings are saved instantly to settings.json and botsort.yaml."
        )
        tk.Label(info_frame, text=guide_text, justify="left", wraplength=450, font=('Helvetica', 9)).pack()

        ttk.Button(self.root, text="Reset to Factory Defaults", command=self.reset_defaults).pack(pady=10)

    def add_slider(self, parent, key, label, min_val, max_val, res, is_int, note):
        container = ttk.Frame(parent)
        container.pack(fill="x", pady=2)
        
        lbl_frame = ttk.Frame(container)
        lbl_frame.pack(fill="x")
        
        tk.Label(lbl_frame, text=label, font=('Helvetica', 10, 'bold')).pack(side="left")
        tk.Label(lbl_frame, text=note, font=('Helvetica', 8, 'italic'), fg='gray').pack(side="left", padx=10)
        
        val_var = tk.DoubleVar(value=self.settings.get(key, 0))
        
        slider = ttk.Scale(
            container, from_=min_val, to=max_val, 
            orient="horizontal", variable=val_var,
            command=lambda val, k=key, is_i=is_int: self.on_change(k, val, is_i)
        )
        slider.pack(side="left", fill="x", expand=True, padx=5)
        
        label_val = ttk.Label(container, text=f"{val_var.get()}", width=6)
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

    def reset_defaults(self):
        defaults = {
            "BUFFER_SIZE": 30,
            "MOTION_THRESHOLD": 15,
            "DILATION_ITERATIONS": 1,
            "WAITING_INTENSITY": 10.0,
            "VOTING_WINDOW": 3,
            "TRACK_BUFFER": 3000,
            "MATCH_THRESH": 0.8
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
