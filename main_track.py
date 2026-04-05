import cv2
import numpy as np
from ultralytics import YOLO
import json
import time
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from collections import deque
import config  # Centralized Configuration module

# ─────────────────────────────────────────────────────────────────────────────
# IDStabilizer — Post-processing Anti-ID-Switch Layer (Copy from main.py)
# ─────────────────────────────────────────────────────────────────────────────
class IDStabilizer:
    """
    Pose-change-aware ID stabilizer — same 4-layer logic as main.py.
    See main.py IDStabilizer docstring for full explanation.
    """

    def __init__(
        self,
        max_lost_frames: int = 300,
        iou_thresh: float = 0.05,
        dist_thresh: float = 500,
        history_len: int = 15,
        anchor_radius: float = 350,
    ):
        self.max_lost_frames = max_lost_frames
        self.iou_thresh       = iou_thresh
        self.dist_thresh      = dist_thresh
        self.history_len      = history_len
        self.anchor_radius    = anchor_radius

        self.track_history: dict[int, deque] = {}
        self.lost_tracks: dict[int, dict]    = {}
        self.id_map: dict[int, int]          = {}
        self._sid_cls: dict[int, int]        = {}
        self._anchor: dict[int, list]        = {}

        self.next_stable_id: int = 1

    @staticmethod
    def _iou(a, b) -> float:
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter  = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0
        areaA  = (a[2]-a[0]) * (a[3]-a[1])
        areaB  = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (areaA + areaB - inter + 1e-6)

    @staticmethod
    def _center(box):
        return (box[0]+box[2])/2, (box[1]+box[3])/2

    @staticmethod
    def _center_dist(a, b) -> float:
        cxA, cyA = (a[0]+a[2])/2, (a[1]+a[3])/2
        cxB, cyB = (b[0]+b[2])/2, (b[1]+b[3])/2
        return ((cxA-cxB)**2 + (cyA-cyB)**2) ** 0.5

    def _update_anchor(self, sid: int, box):
        cx, cy = self._center(box)
        if sid not in self._anchor:
            self._anchor[sid] = [cx, cy, 1]
        else:
            acx, acy, n = self._anchor[sid]
            n = min(n + 1, 60)
            self._anchor[sid] = [acx + (cx - acx) / n, acy + (cy - acy) / n, n]

    def _anchor_dist(self, sid: int, box) -> float:
        if sid not in self._anchor:
            return float('inf')
        acx, acy, _ = self._anchor[sid]
        cx, cy = self._center(box)
        return ((cx - acx)**2 + (cy - acy)**2) ** 0.5

    def _avg_velocity(self, history: deque):
        pts = list(history)
        if len(pts) < 2:
            return (0.0, 0.0, 0.0, 0.0)
        n   = len(pts) - 1
        vx  = ((pts[-1][0]+pts[-1][2])/2 - (pts[0][0]+pts[0][2])/2) / n
        vy  = ((pts[-1][1]+pts[-1][3])/2 - (pts[0][1]+pts[0][3])/2) / n
        dwx = ((pts[-1][2]-pts[-1][0]) - (pts[0][2]-pts[0][0])) / (2*n)
        dwy = ((pts[-1][3]-pts[-1][1]) - (pts[0][3]-pts[0][1])) / (2*n)
        return (vx, vy, dwx, dwy)

    def _predict_box(self, last_box, vel, frames_ahead: int):
        vx, vy = vel[0], vel[1]
        dwx    = vel[2] if len(vel) > 2 else 0.0
        dwy    = vel[3] if len(vel) > 3 else 0.0
        fx, fy = vx * frames_ahead, vy * frames_ahead
        return (
            last_box[0] + fx - dwx * frames_ahead,
            last_box[1] + fy - dwy * frames_ahead,
            last_box[2] + fx + dwx * frames_ahead,
            last_box[3] + fy + dwy * frames_ahead,
        )

    def update(self, raw_ids, boxes, cls_indices) -> list[int]:
        stable_ids      = []
        current_raw_set = set(raw_ids)

        for sid in list(self.lost_tracks):
            self.lost_tracks[sid]["frames_lost"] += 1
            if self.lost_tracks[sid]["frames_lost"] > self.max_lost_frames:
                del self.lost_tracks[sid]
                for raw, s in list(self.id_map.items()):
                    if s == sid:
                        del self.id_map[raw]

        lost_by_class: dict[int, list] = {}
        for lost_sid, info in self.lost_tracks.items():
            lost_by_class.setdefault(info["cls_idx"], []).append(lost_sid)

        new_det_by_class: dict[int, int] = {}
        for raw_id, cls_idx in zip(raw_ids, cls_indices):
            if raw_id not in self.id_map:
                new_det_by_class[cls_idx] = new_det_by_class.get(cls_idx, 0) + 1

        for raw_id, box, cls_idx in zip(raw_ids, boxes, cls_indices):
            box = tuple(float(v) for v in box)

            # Layer 1: raw-ID continuity
            if raw_id in self.id_map:
                sid = self.id_map[raw_id]
            else:
                best_sid     = None
                best_score   = -1.0
                match_reason = "new"
                same_class_lost = lost_by_class.get(cls_idx, [])

                # Layer 2: single-instance fast-path
                if (len(same_class_lost) == 1
                        and new_det_by_class.get(cls_idx, 0) == 1):
                    best_sid     = same_class_lost[0]
                    best_score   = 1.0
                    match_reason = "single-instance-fastpath"
                else:
                    # Layer 3: zone-anchor
                    for lost_sid in same_class_lost:
                        adist = self._anchor_dist(lost_sid, box)
                        if adist < self.anchor_radius:
                            score = 1.0 - adist / self.anchor_radius
                            if score > best_score:
                                best_score, best_sid = score, lost_sid
                                match_reason = f"zone-anchor(d={adist:.0f}px)"

                    # Layer 4: trajectory prediction
                    for lost_sid in same_class_lost:
                        info = self.lost_tracks[lost_sid]
                        pred = self._predict_box(info["last_box"], info["vel"], info["frames_lost"])
                        iou  = self._iou(box, pred)
                        dist = self._center_dist(box, pred)
                        if dist > self.dist_thresh and iou < self.iou_thresh:
                            continue
                        score = iou * 0.6 + max(0.0, 1.0 - dist / self.dist_thresh) * 0.4
                        if score > best_score:
                            best_score, best_sid = score, lost_sid
                            match_reason = f"trajectory(iou={iou:.2f},d={dist:.0f}px)"

                if best_sid is not None:
                    sid = best_sid
                    del self.lost_tracks[best_sid]
                    if cls_idx in lost_by_class and best_sid in lost_by_class[cls_idx]:
                        lost_by_class[cls_idx].remove(best_sid)
                    print(f"[IDStabilizer] raw={raw_id} → stable={sid} via {match_reason}")
                else:
                    sid = self.next_stable_id
                    self.next_stable_id += 1
                    print(f"[IDStabilizer] raw={raw_id} → NEW stable={sid} (class={cls_idx})")
                self.id_map[raw_id] = sid

            if sid not in self.track_history:
                self.track_history[sid] = deque(maxlen=self.history_len)
            self.track_history[sid].append(box)
            self._update_anchor(sid, box)
            stable_ids.append(sid)

        for raw_id, sid in list(self.id_map.items()):
            if raw_id not in current_raw_set:
                if sid not in self.lost_tracks:
                    hist = self.track_history.get(sid, deque())
                    vel  = self._avg_velocity(hist)
                    last = hist[-1] if hist else (0.0, 0.0, 0.0, 0.0)
                    self.lost_tracks[sid] = {
                        "last_box":    last,
                        "vel":         vel,
                        "frames_lost": 0,
                        "cls_idx":     self._sid_cls.get(sid, -1),
                    }
        return stable_ids

    def register_cls(self, sid: int, cls_idx: int):
        self._sid_cls[sid] = cls_idx


try:
    from boxmot import StrongSORT, DeepOCSORT
    HAS_BOXMOT = True
except ImportError as e:
    HAS_BOXMOT = False
    BOXMOT_ERR = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# TrackingAnalyzer (Speed Optimized: No Heatmap, No Activity)
# ─────────────────────────────────────────────────────────────────────────────
class TrackingAnalyzer:
    def __init__(self, model_path=config.MODEL_PATH):
        self.model = YOLO(model_path)
        self.stabilizer = IDStabilizer(
            max_lost_frames=config.TRACK_BUFFER,
            iou_thresh=0.15,
            dist_thresh=300,
            history_len=15,
        )
        self.boxmot_tracker = None
        self.current_tracker_file = config.TRACKER_TYPE

    def _get_boxmot_tracker(self):
        """Initialize or return the selected tracker from boxmot."""
        if not HAS_BOXMOT:
            print(f"[Warning] Advanced tracker requested but BoxMOT import failed: {BOXMOT_ERR}")
            return None
        
        # Re-initialize if the tracker type changed in settings
        if self.boxmot_tracker is None or self.current_tracker_file != config.TRACKER_TYPE:
            print(f"[BoxMOT] Initializing {config.TRACKER_TYPE}...")
            try:
                # ── Choose Tracker Class ──
                if config.TRACKER_TYPE == 'strongsort.yaml':
                    self.boxmot_tracker = StrongSORT(
                        model_weights=Path('osnet_x0_25_msmt17.pt'),
                        device=config.DEVICE,
                        fp16=True
                    )
                elif config.TRACKER_TYPE == 'deepocsort.yaml':
                    self.boxmot_tracker = DeepOCSORT(
                        model_weights=Path('osnet_x0_25_msmt17.pt'),
                        device=config.DEVICE,
                        fp16=True
                    )
                else:
                    return None
                
                self.current_tracker_file = config.TRACKER_TYPE
            except Exception as e:
                print(f"[BoxMOT Error] Failed to init {config.TRACKER_TYPE}: {e}")
                return None
        return self.boxmot_tracker

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Error] Cannot open video: {video_path}")
            return

        frame_id = 0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"out/tracking_only_{timestamp}.mp4"
        Path("out").mkdir(exist_ok=True)
        video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        print(f"[EagleVision] Starting tracking-only (NO HEATMAP) on {video_path}...")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame_id += 1

                # Live-tune every 30 frames
                if frame_id % 30 == 0:
                    config.load_live_settings()

                # ── Choose Tracker Logic (Native vs. BoxMOT) ──────────────────
                if config.TRACKER_TYPE != self.current_tracker_file:
                    print(f"[EagleVision] Switching tracker: {self.current_tracker_file} -> {config.TRACKER_TYPE}")
                    # Clear native predictor to force re-init if using BoT-SORT/ByteTrack
                    self.model.predictor = None 
                    self.current_tracker_file = config.TRACKER_TYPE
                
                if config.TRACKER_TYPE in ['strongsort.yaml', 'deepocsort.yaml'] and HAS_BOXMOT:
                    # ── StrongSORT path ──
                    results = self.model.predict(
                        frame,
                        classes=config.TRACKED_CLASSES,
                        conf=0.1,  # Lower conf to give StrongSORT more data
                        imgsz=config.YOLO_IMGSZ,
                        verbose=False,
                        device=config.DEVICE
                    )
                    dets = results[0].boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, cls]
                    tracks = self._get_boxmot_tracker().update(dets, frame) # [x1, y1, x2, y2, id, conf, cls, ind]
                    
                    if tracks.size > 0:
                        raw_boxes   = tracks[:, 0:4]
                        raw_ids     = tracks[:, 4].astype(int)
                        cls_indices = tracks[:, 6].astype(int)
                    else:
                        raw_boxes, raw_ids, cls_indices = [], [], []

                else:
                    # ── BoT-SORT / ByteTrack path (Native) ──
                    tk_file = config.TRACKER_TYPE
                    if tk_file not in ['botsort.yaml', 'bytetrack.yaml']:
                        tk_file = 'botsort.yaml' # Safe fallback to avoid YOLO crash
                    
                    results = self.model.track(
                        frame,
                        persist=True,
                        classes=config.TRACKED_CLASSES,
                        tracker=tk_file,
                        conf=0.1,
                        iou=config.YOLO_IOU,
                        imgsz=config.YOLO_IMGSZ,
                        agnostic_nms=True,
                        verbose=False,
                        device=config.DEVICE,
                    )
                    if results[0].boxes.id is not None:
                        raw_boxes   = results[0].boxes.xyxy.cpu().numpy()
                        raw_ids     = results[0].boxes.id.cpu().numpy().astype(int)
                        cls_indices = results[0].boxes.cls.cpu().numpy().astype(int)
                    else:
                        raw_boxes, raw_ids, cls_indices = [], [], []

                # ── Post-processing: ID Stabilization ────────────────────────
                if len(raw_ids) > 0:
                    stable_ids = self.stabilizer.update(
                        raw_ids.tolist() if isinstance(raw_ids, np.ndarray) else raw_ids,
                        raw_boxes.tolist() if isinstance(raw_boxes, np.ndarray) else raw_boxes,
                        cls_indices.tolist() if isinstance(cls_indices, np.ndarray) else cls_indices
                    )
                    
                    for box, obj_id, cls_idx in zip(raw_boxes, stable_ids, cls_indices):
                        x1, y1, x2, y2 = map(int, box)
                        obj_name = self.model.names[cls_idx]
                        self.stabilizer.register_cls(int(obj_id), int(cls_idx))

                        # Visualization: Blue for Excavator, Green for Truck (example)
                        color = (255, 0, 0) if cls_idx == 0 else (0, 180, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"{obj_name} #{obj_id}"
                        cv2.putText(
                            frame, label, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                        )

                # Overlays & Output
                t_name = config.TRACKER_TYPE.replace('.yaml', '').upper()
                cv2.putText(frame, f"Tracker: {t_name} | Frame: {frame_id}", (15, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("EagleVision Tracker", frame)
                video_writer.write(frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        finally:
            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()
            print(f"Tracking complete. Video saved to: {out_path}")

if __name__ == "__main__":
    # Ensure gui and main can be run together
    gui_process = subprocess.Popen([sys.executable, "setting.py"])
    try:
        analyzer = TrackingAnalyzer()
        video = "videos/v2.mp4"
        if not Path(video).exists():
             print(f"Video {video} not found. Please check paths.")
        else:
             analyzer.analyze_video(video)
    finally:
        time.sleep(0.5)
        gui_process.terminate()