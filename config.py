import json
import os

# EagleVision Configuration Parameters

# --- Video Processing Settings ---
BUFFER_SIZE = 30           # Number of frames to accumulate for the heatmap (approx 1-2 seconds)
BLUR_KERNEL = (7, 7)       # Smoothing kernel size for motion detection
MOTION_THRESHOLD = 15      # Binary threshold for frame differencing (higher = less sensitive)
DILATION_ITERATIONS = 1    # How many times to dilate motion pixels

# --- Activity Classification Thresholds ---
# Adjust these to fine-tune "Waiting vs Active" detection
BUFFER_SIZE = 30           # Number of frames to accumulate for the heatmap
BLUR_KERNEL = (7, 7)       # Smoothing kernel size for motion detection
MOTION_THRESHOLD = 15      # Binary threshold for frame differencing
DILATION_ITERATIONS = 1    # How many times to dilate motion pixels

# --- Activity Classification Thresholds ---
WAITING_INTENSITY = 10.0   # If average heatmap intensity is below this, status is "WAITING"
VOTING_WINDOW = 3          # Smoothing window

# --- Tracking Sensitivity (ByteTrack) ---
TRACK_BUFFER = 100000          # How many frames to remember lost objects
MATCH_THRESH = 0.8         # Matching threshold
DEVICE=0
# --- YOLO Settings ---
MODEL_PATH = 'models/best2.pt'
CLASSIFICATION_MODEL_PATH = 'models/classfy.pt'
TRACKED_CLASSES = [0, 1] # COCO / Custom indices: Excavators (0), Trucks (1)

# --- Kafka & Backend Settings ---
KAFKA_SERVER = 'localhost:9092'
KAFKA_TOPIC = 'equipment-telemetry'
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///db/eaglevision.db")

# --- Live Tuning Logic ---
def load_live_settings():
    """Real-time parameter loading from settings.json."""
    global BUFFER_SIZE, MOTION_THRESHOLD, DILATION_ITERATIONS, WAITING_INTENSITY
    settings_file = "settings.json"
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r") as f:
                data = json.load(f)
                BUFFER_SIZE = int(data.get("BUFFER_SIZE", BUFFER_SIZE))
                MOTION_THRESHOLD = int(data.get("MOTION_THRESHOLD", MOTION_THRESHOLD))
                DILATION_ITERATIONS = int(data.get("DILATION_ITERATIONS", DILATION_ITERATIONS))
                BUFFER_SIZE = int(data.get("BUFFER_SIZE", BUFFER_SIZE))
                MOTION_THRESHOLD = int(data.get("MOTION_THRESHOLD", MOTION_THRESHOLD))
                DILATION_ITERATIONS = int(data.get("DILATION_ITERATIONS", DILATION_ITERATIONS))
                WAITING_INTENSITY = float(data.get("WAITING_INTENSITY", WAITING_INTENSITY))
                global VOTING_WINDOW
                VOTING_WINDOW = int(data.get("VOTING_WINDOW", VOTING_WINDOW))
                global TRACK_BUFFER, MATCH_THRESH
                TRACK_BUFFER = int(data.get("TRACK_BUFFER", TRACK_BUFFER))
                MATCH_THRESH = float(data.get("MATCH_THRESH", MATCH_THRESH))
        except Exception:
            pass # Use defaults if there's an error

# Initialize with live settings
load_live_settings()
