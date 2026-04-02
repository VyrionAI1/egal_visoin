import cv2
import numpy as np
from ultralytics import YOLO
from confluent_kafka import Producer
import json
import time
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from collections import deque
import config # Centralized Configuration module

# Kafka Configuration
kafka_config = {'bootstrap.servers': config.KAFKA_SERVER}
try:
    producer = Producer(kafka_config)
except Exception as e:
    print(f"Warning: Kafka not connected ({e}). Running in standalone mode.")
    producer = None

def delivery_report(err, msg):
    if err is not None:
        print(f'Message delivery failed: {err}')

class ConstructionAnalyzer:
    def __init__(self, model_path=config.MODEL_PATH, buffer_size=config.BUFFER_SIZE):
        self.model = YOLO(model_path)
        self.classifier = YOLO(config.CLASSIFICATION_MODEL_PATH)
        self.buffer_size = buffer_size
        self.diff_buffer = [] 
        self.equipment_stats = {} 
        self.prev_gray = None
        # Cumulative time stats
        self.class_time_stats = {} 
        # Temporal Voting Queues for stability {obj_id: deque(maxlen=config.VOTING_WINDOW)}
        self.activity_queues = {}

    def get_motion_mask(self, frame):
        """Calculate global motion mask using frame differencing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, config.BLUR_KERNEL, 0)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros_like(gray)
        
        frame_diff = cv2.absdiff(self.prev_gray, gray)
        _, motion_mask = cv2.threshold(frame_diff, config.MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Morphological clean-up
        kernel = np.ones((5,5), np.uint8)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=config.DILATION_ITERATIONS)
        
        self.prev_gray = gray
        return motion_mask

    def analyze_activity(self, roi_heatmap, roi_img, obj_id, cls_idx):
        """Combined classification logic using both Heatmap intensity AND the new Classification Model."""
        # Check motion intensity first for "Waiting" state
        if roi_heatmap is not None and roi_heatmap.size > 0:
            intensity = np.mean(roi_heatmap)
            if intensity < config.WAITING_INTENSITY:
                return "Waiting", "Inactive"
        
        # --- EXCAVATOR Logic (Class 0) ---
        if cls_idx == 0:
            if roi_img is not None and roi_img.size > 0:
                # Use the new YOLO classification model
                res = self.classifier(roi_img, verbose=False)
                if res and len(res) > 0:
                    probs = res[0].probs
                    top1_idx = int(probs.top1)
                    voted_activity = res[0].names[top1_idx]
                    # Format activity name (e.g., "Swinging", "Dumping", "Digging")
                    return voted_activity.capitalize(), "Active"
            
            return "Active", "Active"

        # --- TRUCK Logic (Class 1) ---
        elif cls_idx == 1:
            return "Active", "Active"
                
        return "Waiting", "Inactive"

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        
        # Video Writer Setup
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("out")
        out_dir.mkdir(exist_ok=True)
        out_path = f"out/output_bytetrack_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Splitting view: color frame + heatmap = 2 * width
        video_writer = cv2.VideoWriter(out_path, fourcc, fps, (w * 2, h))

        print(f"Starting BoT-SORT analysis on {video_path}...")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame_id += 1
                # Periodically reload settings for live-tuning
                if frame_id % 30 == 0:
                    config.load_live_settings()
                    
                # 0. Restore Heatmap accumulation
                current_motion = self.get_motion_mask(frame)
                self.diff_buffer.append(current_motion)
                if len(self.diff_buffer) > config.BUFFER_SIZE:
                    self.diff_buffer.pop(0)

                heatmap_accum = None
                if len(self.diff_buffer) == config.BUFFER_SIZE:
                    heatmap_accum = np.mean(self.diff_buffer, axis=0).astype(np.uint8)

                # tracker='botsort.yaml' points to our local optimized config
                results = self.model.track(
                    frame, 
                    persist=True, 
                    classes=config.TRACKED_CLASSES, 
                    tracker="botsort.yaml", 
                    conf=0.15,
                    verbose=False,
                    device=config.DEVICE
                )
                
                # Process tracked objects
                current_boxes = []
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    cls_indices = results[0].boxes.cls.cpu().numpy().astype(int)
                    
                    for box, obj_id, cls_idx in zip(boxes, ids, cls_indices):
                        # Map track data and CLAMP to frame boundaries
                        x1, y1, x2, y2 = map(int, box)
                        h_frame, w_frame = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w_frame, x2), min(h_frame, y2)
                        current_boxes.append((x1, y1, x2, y2))
                        
                        obj_name = self.model.names[cls_idx]
                        
                        # 2. Extract ROI Crops & Analyze
                        raw_activity, raw_status = "Waiting", "Inactive"
                        roi_heatmap = None
                        roi_img = frame[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else None
                        
                        if heatmap_accum is not None:
                            roi_heatmap = heatmap_accum[y1:y2, x1:x2]
                            raw_activity, raw_status = self.analyze_activity(roi_heatmap, roi_img, obj_id, cls_idx)
                        elif roi_img is not None:
                            # If heatmap not ready yet, we can still attempt classification if it's an excavator
                            raw_activity, raw_status = self.analyze_activity(None, roi_img, obj_id, cls_idx)
                        
                        # 3. Temporal Voting (Stability)
                        if obj_id not in self.activity_queues:
                            self.activity_queues[obj_id] = deque(maxlen=config.VOTING_WINDOW)
                        
                        self.activity_queues[obj_id].append((raw_activity, raw_status))
                        
                        # Determine consensus activity
                        from collections import Counter
                        voted_activity, voted_status = Counter(self.activity_queues[obj_id]).most_common(1)[0][0]

                        # 4. Utilization Statistics (using Voted states)
                        if obj_id not in self.equipment_stats:
                            self.equipment_stats[obj_id] = {
                                "class_name": obj_name,
                                "working_frames": 0,
                                "waiting_frames": 0
                            }
                        
                        if voted_status == "Active": 
                            self.equipment_stats[obj_id]["working_frames"] += 1
                        else:
                            self.equipment_stats[obj_id]["waiting_frames"] += 1

                        w_secs = self.equipment_stats[obj_id]["working_frames"] / fps if fps > 0 else 0.0
                        i_secs = self.equipment_stats[obj_id]["waiting_frames"] / fps if fps > 0 else 0.0
                        total_secs = w_secs + i_secs
                        u_pct = (w_secs / total_secs * 100) if total_secs > 0 else 0.0

                        # Format timestamp
                        ms = int((frame_id / fps) * 1000) if fps > 0 else 0
                        h, r = divmod(ms / 1000, 3600)
                        m, s = divmod(r, 60)
                        ts_str = f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{int(ms%1000):03d}"

                        # 5. Kafka Telemetry FORMAT UPDATE
                        # Only pipe to database/dashboard if it has logged at least 1 full second
                        if int(w_secs) > 0 or int(i_secs) > 0:
                            payload = {
                                "frame_id": frame_id,
                                "equipment_id": f"{obj_name}-{obj_id}",
                                "equipment_class": obj_name.lower(),
                                "timestamp": ts_str,
                                "utilization": {
                                    "current_state": "ACTIVE" if voted_status == "Active" else "INACTIVE",
                                    "current_activity": voted_activity.upper(),
                                    "motion_source": "articulated" if voted_activity.upper() in ["DIGGING", "SWINGING", "DUMPING"] else "tracks_only"
                                },
                                "time_analytics": {
                                    "total_tracked_seconds": round(total_secs, 1),
                                    "total_active_seconds": round(w_secs, 1),
                                    "total_idle_seconds": round(i_secs, 1),
                                    "utilization_percent": round(u_pct, 1)
                                }
                            }
                            
                            if producer:
                                producer.produce(config.KAFKA_TOPIC, json.dumps(payload), callback=delivery_report)
                        
                        # 6. Visualization
                        color = (0, 255, 0) if voted_status == "Active" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        


                        # Simplified label logic
                        label = f"{obj_name} {obj_id}: {voted_activity}"
                        if voted_status == "Inactive":
                            label = f"{obj_name} {obj_id}: {voted_activity} (Inactive)"
                            
                        # Blue color in BGR is (255, 0, 0)
                        cv2.putText(frame, label, (x1, y1-5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # 7. Draw Top Summary Header (Cumulative time)
                self.draw_summary(frame, fps)

                # 8. Create Split View (Color Frame + Masked Heatmap)
                heatmap_view = np.zeros_like(frame)
                if heatmap_accum is not None:
                    heatmap_full = cv2.applyColorMap(heatmap_accum, cv2.COLORMAP_JET)
                    # Create mask for current detections
                    mask = np.zeros(heatmap_accum.shape, dtype=np.uint8)
                    if results[0].boxes.id is not None:
                        for (x1, y1, x2, y2) in current_boxes:
                            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    
                    heatmap_view = cv2.bitwise_and(heatmap_full, heatmap_full, mask=mask)
                
                # Combine side-by-side
                combined_view = cv2.hconcat([frame, heatmap_view])

                cv2.imshow("EagleVision - BoT-SORT Analysis", combined_view)
                video_writer.write(combined_view)
                cv2.imwrite("out/current_frame.jpg", combined_view)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                if producer: producer.poll(0)
        finally:
            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()
            if producer: producer.flush()

    def draw_summary(self, frame, fps):
        """Draw a semi-transparent summary box showing every individual object's time at the bottom-right."""
        h_frame, w_frame = frame.shape[:2]
        overlay = frame.copy()
        
        # Calculate valid objects (exclude if both work and wait seconds are 0)
        valid_objs = []
        for obj_id in sorted(self.equipment_stats.keys()):
            stats = self.equipment_stats[obj_id]
            w_frames, i_frames = stats["working_frames"], stats["waiting_frames"]
            w_secs = int(w_frames / fps) if fps > 0 else 0
            i_secs = int(i_frames / fps) if fps > 0 else 0
            
            if w_secs == 0 and i_secs == 0:
                continue
            valid_objs.append((obj_id, stats, w_secs, i_secs))
            
        if not valid_objs:
            return  # Draw nothing if no valid data
            
        num_objs = len(valid_objs)
        h_box, w_box = 40 + (num_objs * 25), 450
        
        # Bottom-right positioning
        x_start, y_start = w_frame - w_box - 10, h_frame - h_box - 10
        x_end, y_end = w_frame - 10, h_frame - 10
        
        # Draw Box
        cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, "--- INDIVIDUAL EQUIPMENT TIME ---", (x_start + 10, y_start + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_text = y_start + 45
        for obj_id, stats, w_secs, i_secs in valid_objs:
            name = stats["class_name"]
            
            w_time = f"{w_secs//60}:{w_secs%60:02d}"
            i_time = f"{i_secs//60}:{i_secs%60:02d}"
            
            text = f"ID {obj_id} ({name}): {w_time} Work | {i_time} Wait"
            color = (0, 255, 0) if w_secs > 0 else (200, 200, 200)
            cv2.putText(frame, text, (x_start + 10, y_text), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_text += 25

if __name__ == "__main__":
    # Auto-run the GUI in a separate process for real-time control
    gui_process = subprocess.Popen([sys.executable, "setting.py"])
    
    try:
        analyzer = ConstructionAnalyzer()
        analyzer.analyze_video("videos/v.mp4")
    finally:
        # Give OpenCV a moment to release its writer lock
        time.sleep(0.5)
        # Kill the GUI so the terminal is properly returned to the user
        gui_process.terminate()
        print("Analysis complete! Video saved successfully.")
