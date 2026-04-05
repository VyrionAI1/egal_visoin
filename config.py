import json
import os

# EagleVision Configuration Parameters

# --- Video Processing Settings ---
BUFFER_SIZE = 30           # Number of frames to accumulate for the heatmap (~1-2 seconds at 30fps)
BLUR_KERNEL = (7, 7)       # Smoothing kernel size for motion detection
MOTION_THRESHOLD = 15      # Binary threshold for frame differencing (higher = less sensitive)
DILATION_ITERATIONS = 1    # How many times to dilate motion pixels

# --- Activity Classification Thresholds ---
WAITING_INTENSITY = 10.0   # If average heatmap intensity is below this, status is "WAITING"
VOTING_WINDOW = 10         # Temporal voting window (larger = more stable labels, less reactive)
                           # Raised from 3 → 10 to prevent flickering activity labels during ID handoff

# --- Tracking Sensitivity ---
# Note: Core tracker thresholds are in botsort.yaml
# These are kept for reference / live-tuning via settings.json
TRACK_BUFFER = 8000         # Max frames a lost track is remembered (matches botsort.yaml)
MATCH_THRESH = 0.8         # IoU match threshold (matches botsort.yaml match_thresh)
TRACKER_TYPE = 'botsort.yaml' # Choice of tracker: botsort.yaml or bytetrack.yaml

# --- Device & Hardware ---
DEVICE = 0                 # 0 = first GPU, 'cpu' = CPU

# --- YOLO Settings ---
MODEL_PATH = 'models/best.pt'
CLASSIFICATION_MODEL_PATH = 'models/classfy.pt'
TRACKED_CLASSES = [0, 1]   # Custom model indices: Excavators (0), Trucks (1)
YOLO_CONF = 0.25           # Confidence threshold for detection
YOLO_IOU = 0.1             # NMS IoU threshold for detection
YOLO_IMGSZ = 640           # Reverted to 640 as requested

# --- Kafka & Backend Settings ---
KAFKA_SERVER = 'localhost:9092'
KAFKA_TOPIC = 'equipment-telemetry'
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///db/eaglevision.db")

# --- Live Tuning Logic ---
def load_live_settings():
    """Real-time parameter loading from settings.json."""
    global BUFFER_SIZE, MOTION_THRESHOLD, DILATION_ITERATIONS, WAITING_INTENSITY
    global VOTING_WINDOW, TRACK_BUFFER, MATCH_THRESH, YOLO_CONF, YOLO_IOU, TRACKER_TYPE
    settings_file = "settings.json"
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r") as f:
                data = json.load(f)
                BUFFER_SIZE = int(data.get("BUFFER_SIZE", BUFFER_SIZE))
                MOTION_THRESHOLD = int(data.get("MOTION_THRESHOLD", MOTION_THRESHOLD))
                DILATION_ITERATIONS = int(data.get("DILATION_ITERATIONS", DILATION_ITERATIONS))
                WAITING_INTENSITY = float(data.get("WAITING_INTENSITY", WAITING_INTENSITY))
                VOTING_WINDOW = int(data.get("VOTING_WINDOW", VOTING_WINDOW))
                TRACK_BUFFER = int(data.get("TRACK_BUFFER", TRACK_BUFFER))
                MATCH_THRESH = float(data.get("MATCH_THRESH", MATCH_THRESH))
                YOLO_CONF = float(data.get("YOLO_CONF", YOLO_CONF))
                YOLO_IOU = float(data.get("YOLO_IOU", YOLO_IOU))
                TRACKER_TYPE = data.get("TRACKER_TYPE", TRACKER_TYPE)
        except Exception:
            pass  # Use defaults if there's an error

# Initialize with live settings on import
load_live_settings()
