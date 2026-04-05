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
from collections import deque, Counter
import config  # Centralized Configuration module

# ─────────────────────────────────────────────────────────────────────────────
# Kafka Setup
# ─────────────────────────────────────────────────────────────────────────────
kafka_config = {'bootstrap.servers': config.KAFKA_SERVER}
try:
    producer = Producer(kafka_config)
except Exception as e:
    print(f"Warning: Kafka not connected ({e}). Running in standalone mode.")
    producer = None

def delivery_report(err, msg):
    if err is not None:
        print(f'Message delivery failed: {err}')


# ─────────────────────────────────────────────────────────────────────────────
# IDStabilizer — Post-processing Anti-ID-Switch Layer
# ─────────────────────────────────────────────────────────────────────────────
class IDStabilizer:
    """
    Pose-change-aware ID stabilizer for large construction machines.

    ROOT CAUSE OF ID SWITCHES ON ROTATING EXCAVATORS:
    When an excavator rotates its body or swings its arm, the bounding box
    centroid can jump 200–400px even though the machine never left the frame.
    Standard IoU + center-distance matching fails because both metrics measure
    geometric overlap between the *predicted* position and the *new* box —
    but the prediction assumes linear motion, which breaks completely during
    in-place rotation.

    FOUR-LAYER MATCHING STRATEGY (applied in order):
    ─────────────────────────────────────────────────
    1. RAW-ID CONTINUITY  (zero cost)
       If the raw tracker already knows this ID, trust it and keep the mapping.
       This is the normal case — no pose change happening.

    2. SINGLE-INSTANCE FAST-PATH  (pose-change resistant)
       If there is exactly 1 lost track of class C and exactly 1 new detection
       of class C that has no existing mapping, they MUST be the same machine
       (you only have 1–2 machines per class on site). Match them directly,
       regardless of how far the box has moved.
       → Fixes: excavator rotates in place, box jumps 300px, normal matching
         fails, stabilizer assigns new ID. Fast-path catches it.

    3. ZONE-ANCHOR MATCHING  (spatial memory)
       Each stable ID accumulates a running "anchor centroid" — the mean of
       all box centers it has ever occupied. New detections are matched to the
       anchor if they land within `anchor_radius` pixels, even when the
       predicted position (based on recent velocity) is far away.
       → Fixes: excavator parked at one side of the pit all day. Its anchor
         is well-known. Even after a tracker reset, a detection near the
         anchor snaps back to the correct stable ID.

    4. TRAJECTORY PREDICTION  (motion-based fallback)
       Classic linear extrapolation from history. Used when layers 2 and 3
       both fail. Handles objects that are genuinely moving across the frame.
    """

    def __init__(
        self,
        max_lost_frames: int = 120,   # ~4 s at 30 fps
        iou_thresh: float = 0.05,     # very lenient — pose change shrinks IoU to near zero
        dist_thresh: float = 500,     # large radius — excavator arm swing can move box 400px
        history_len: int = 15,        # longer history for stable velocity estimate
        anchor_radius: float = 350,   # px — zone-anchor match radius
    ):
        self.max_lost_frames = max_lost_frames
        self.iou_thresh       = iou_thresh
        self.dist_thresh      = dist_thresh
        self.history_len      = history_len
        self.anchor_radius    = anchor_radius

        # {stable_id: deque([box, ...])}
        self.track_history: dict[int, deque] = {}
        # {stable_id: {"last_box", "vel", "frames_lost", "cls_idx"}}
        self.lost_tracks: dict[int, dict]    = {}
        # {raw_tracker_id: stable_id}
        self.id_map: dict[int, int]          = {}
        # {stable_id: cls_idx}
        self._sid_cls: dict[int, int]        = {}
        # {stable_id: (anchor_cx, anchor_cy, n_samples)} — running mean centroid
        self._anchor: dict[int, list]        = {}

        self.next_stable_id: int = 1

    # ── geometry helpers ──────────────────────────────────────────────────────

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
        """Update the running mean centroid (anchor) for a stable ID."""
        cx, cy = self._center(box)
        if sid not in self._anchor:
            self._anchor[sid] = [cx, cy, 1]
        else:
            acx, acy, n = self._anchor[sid]
            n += 1
            # Clamp n so anchor slowly drifts with the machine if it moves
            n = min(n, 60)
            self._anchor[sid] = [acx + (cx - acx) / n, acy + (cy - acy) / n, n]

    def _anchor_dist(self, sid: int, box) -> float:
        """Distance from box center to the stored anchor of a stable ID."""
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

    # ── main API ──────────────────────────────────────────────────────────────

    def update(self, raw_ids, boxes, cls_indices) -> list[int]:
        stable_ids      = []
        current_raw_set = set(raw_ids)

        # ── 1. Age lost tracks; prune expired ones ────────────────────────────
        for sid in list(self.lost_tracks):
            self.lost_tracks[sid]["frames_lost"] += 1
            if self.lost_tracks[sid]["frames_lost"] > self.max_lost_frames:
                del self.lost_tracks[sid]
                for raw, s in list(self.id_map.items()):
                    if s == sid:
                        del self.id_map[raw]

        # Build per-class index of lost tracks for the fast-path
        # {cls_idx: [list of lost_sid]}
        lost_by_class: dict[int, list] = {}
        for lost_sid, info in self.lost_tracks.items():
            c = info["cls_idx"]
            lost_by_class.setdefault(c, []).append(lost_sid)

        # Count unmapped new detections per class for the fast-path gate
        # {cls_idx: count of new (unmapped) detections}
        new_det_by_class: dict[int, int] = {}
        for raw_id, cls_idx in zip(raw_ids, cls_indices):
            if raw_id not in self.id_map:
                new_det_by_class[cls_idx] = new_det_by_class.get(cls_idx, 0) + 1

        # ── 2. Associate each incoming detection ──────────────────────────────
        for raw_id, box, cls_idx in zip(raw_ids, boxes, cls_indices):
            box = tuple(float(v) for v in box)

            # ── LAYER 1: raw-ID continuity ────────────────────────────────────
            if raw_id in self.id_map:
                sid = self.id_map[raw_id]

            else:
                best_sid   = None
                best_score = -1.0
                match_reason = "new"

                same_class_lost = lost_by_class.get(cls_idx, [])

                # ── LAYER 2: single-instance fast-path ────────────────────────
                # Exactly 1 lost track of this class AND exactly 1 new detection
                # of this class → they must be the same machine. Match directly.
                if (len(same_class_lost) == 1
                        and new_det_by_class.get(cls_idx, 0) == 1):
                    best_sid     = same_class_lost[0]
                    best_score   = 1.0
                    match_reason = "single-instance-fastpath"

                else:
                    # ── LAYER 3: zone-anchor matching ─────────────────────────
                    # Check if the new box lands near any lost track's anchor
                    for lost_sid in same_class_lost:
                        adist = self._anchor_dist(lost_sid, box)
                        if adist < self.anchor_radius:
                            score = 1.0 - adist / self.anchor_radius
                            if score > best_score:
                                best_score   = score
                                best_sid     = lost_sid
                                match_reason = f"zone-anchor(d={adist:.0f}px)"

                    # ── LAYER 4: trajectory prediction ────────────────────────
                    # Standard IoU + distance against predicted position
                    for lost_sid in same_class_lost:
                        info = self.lost_tracks[lost_sid]
                        pred = self._predict_box(
                            info["last_box"], info["vel"], info["frames_lost"]
                        )
                        iou  = self._iou(box, pred)
                        dist = self._center_dist(box, pred)

                        if dist > self.dist_thresh and iou < self.iou_thresh:
                            continue

                        score = iou * 0.6 + max(0.0, 1.0 - dist / self.dist_thresh) * 0.4
                        if score > best_score:
                            best_score   = score
                            best_sid     = lost_sid
                            match_reason = f"trajectory(iou={iou:.2f},d={dist:.0f}px)"

                if best_sid is not None:
                    sid = best_sid
                    del self.lost_tracks[best_sid]
                    # Remove from lost_by_class so it can't be matched twice
                    if cls_idx in lost_by_class and best_sid in lost_by_class[cls_idx]:
                        lost_by_class[cls_idx].remove(best_sid)
                    print(f"[IDStabilizer] raw={raw_id} → stable={sid} "
                          f"via {match_reason}")
                else:
                    sid = self.next_stable_id
                    self.next_stable_id += 1
                    print(f"[IDStabilizer] raw={raw_id} → NEW stable={sid} (class={cls_idx})")

                self.id_map[raw_id] = sid

            # ── update history and anchor ─────────────────────────────────────
            if sid not in self.track_history:
                self.track_history[sid] = deque(maxlen=self.history_len)
            self.track_history[sid].append(box)
            self._update_anchor(sid, box)

            stable_ids.append(sid)

        # ── 3. Mark tracks absent this frame as "lost" ────────────────────────
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
        """Record which equipment class belongs to each stable ID."""
        self._sid_cls[sid] = cls_idx


try:
    from boxmot import StrongSORT, DeepOCSORT
    HAS_BOXMOT = True
except ImportError as e:
    HAS_BOXMOT = False
    BOXMOT_ERR = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# ConstructionAnalyzer
# ─────────────────────────────────────────────────────────────────────────────
class ConstructionAnalyzer:
    def __init__(self, model_path=config.MODEL_PATH, buffer_size=config.BUFFER_SIZE):
        self.model      = YOLO(model_path)
        self.classifier = YOLO(config.CLASSIFICATION_MODEL_PATH)
        self.buffer_size = buffer_size
        self.diff_buffer: list = []
        self.equipment_stats: dict = {}
        self.prev_gray = None
        self.class_time_stats: dict = {}
        # Temporal voting queues  {stable_id: deque}
        self.activity_queues: dict = {}
        # ID Stabilizer (post-processing anti-switch layer)
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

    # ── motion heatmap ────────────────────────────────────────────────────────

    def get_motion_mask(self, frame):
        """Calculate global motion mask using frame differencing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, config.BLUR_KERNEL, 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros_like(gray)

        frame_diff = cv2.absdiff(self.prev_gray, gray)
        _, motion_mask = cv2.threshold(
            frame_diff, config.MOTION_THRESHOLD, 255, cv2.THRESH_BINARY
        )
        kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv2.dilate(
            motion_mask, kernel, iterations=config.DILATION_ITERATIONS
        )
        self.prev_gray = gray
        return motion_mask

    # ── activity classification ───────────────────────────────────────────────

    def analyze_activity(self, roi_heatmap, roi_img, obj_id, cls_idx):
        """Combined classification: heatmap intensity + YOLO classifier."""
        # Waiting check (low motion)
        if roi_heatmap is not None and roi_heatmap.size > 0:
            if np.mean(roi_heatmap) < config.WAITING_INTENSITY:
                return "Waiting", "Inactive"

        # Excavator (class 0) – use classifier model
        if cls_idx == 0:
            if roi_img is not None and roi_img.size > 0:
                res = self.classifier(roi_img, verbose=False)
                if res and len(res) > 0:
                    probs    = res[0].probs
                    top1_idx = int(probs.top1)
                    activity = res[0].names[top1_idx].capitalize()
                    return activity, "Active"
            return "Active", "Active"

        # Truck (class 1)
        elif cls_idx == 1:
            return "Active", "Active"

        return "Waiting", "Inactive"

    # ── main loop ─────────────────────────────────────────────────────────────

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_id = 0

        # Video Writer Setup
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir    = Path("out")
        out_dir.mkdir(exist_ok=True)
        out_path   = f"out/output_stable_{timestamp}.mp4"
        fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(out_path, fourcc, fps, (w * 2, h))

        print(f"[EagleVision] Starting stable-ID analysis on {video_path} ...")
        print(f"[EagleVision] IDStabilizer active  |  BoT-SORT tracker")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_id += 1

                # Live-tune every 30 frames
                if frame_id % 30 == 0:
                    config.load_live_settings()

                # ── Heatmap accumulation ──────────────────────────────────────
                current_motion = self.get_motion_mask(frame)
                self.diff_buffer.append(current_motion)
                if len(self.diff_buffer) > config.BUFFER_SIZE:
                    self.diff_buffer.pop(0)

                heatmap_accum = None
                if len(self.diff_buffer) == config.BUFFER_SIZE:
                    heatmap_accum = np.mean(
                        self.diff_buffer, axis=0
                    ).astype(np.uint8)

                # ── Choose Tracker Logic (Native vs. BoxMOT) ──────────────────
                if config.TRACKER_TYPE != self.current_tracker_file:
                    print(f"[EagleVision] Switching tracker: {self.current_tracker_file} -> {config.TRACKER_TYPE}")
                    self.model.predictor = None # Reset native engine
                    self.current_tracker_file = config.TRACKER_TYPE

                if config.TRACKER_TYPE in ['strongsort.yaml', 'deepocsort.yaml'] and HAS_BOXMOT:
                    results = self.model.predict(
                        frame,
                        classes=config.TRACKED_CLASSES,
                        conf=0.1,
                        imgsz=config.YOLO_IMGSZ,
                        verbose=False,
                        device=config.DEVICE
                    )
                    dets = results[0].boxes.data.cpu().numpy()
                    tracks = self._get_boxmot_tracker().update(dets, frame)
                    if tracks.size > 0:
                        raw_boxes, raw_ids, cls_indices = tracks[:, 0:4], tracks[:, 4].astype(int), tracks[:, 6].astype(int)
                    else:
                        raw_boxes, raw_ids, cls_indices = [], [], []
                else:
                    # ── BoT-SORT tracker call (Native) ─────────────────────────
                    tk_file = config.TRACKER_TYPE
                    if tk_file not in ['botsort.yaml', 'bytetrack.yaml']:
                        tk_file = 'botsort.yaml' # Fallback
                    
                    results = self.model.track(
                        frame,
                        persist=True,
                        classes=config.TRACKED_CLASSES,
                        tracker=tk_file,
                        conf=0.1,           # Lower confidence for better tracker association
                        iou=config.YOLO_IOU, # From config
                        imgsz=config.YOLO_IMGSZ, # High resolution for accuracy
                        agnostic_nms=True,   # suppress cross-class overlaps
                        verbose=False,
                        device=config.DEVICE,
                    )
                    if results[0].boxes.id is not None:
                        raw_boxes   = results[0].boxes.xyxy.cpu().numpy()
                        raw_ids     = results[0].boxes.id.cpu().numpy().astype(int)
                        cls_indices = results[0].boxes.cls.cpu().numpy().astype(int)
                    else:
                        raw_boxes, raw_ids, cls_indices = [], [], []

                # ── Process detections ────────────────────────────────────────
                current_boxes = []

                if len(raw_ids) > 0:
                    # ── IDStabilizer: remap raw IDs → stable IDs ──────────────
                    stable_ids = self.stabilizer.update(
                        raw_ids.tolist() if isinstance(raw_ids, np.ndarray) else raw_ids,
                        raw_boxes.tolist() if isinstance(raw_boxes, np.ndarray) else raw_boxes,
                        cls_indices.tolist() if isinstance(cls_indices, np.ndarray) else cls_indices
                    )
                    # Register class for stable ReID persistence
                    for sid, cls_idx in zip(stable_ids, cls_indices):
                        self.stabilizer.register_cls(int(sid), int(cls_idx))

                    for box, obj_id, cls_idx in zip(raw_boxes, stable_ids, cls_indices):
                        # Clamp to frame boundaries
                        x1, y1, x2, y2 = map(int, box)
                        h_frame, w_frame = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w_frame, x2), min(h_frame, y2)
                        current_boxes.append((x1, y1, x2, y2))

                        obj_name = self.model.names[cls_idx]

                        # ── Activity analysis ─────────────────────────────────
                        raw_activity, raw_status = "Waiting", "Inactive"
                        roi_img = frame[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else None

                        if heatmap_accum is not None:
                            roi_heatmap = heatmap_accum[y1:y2, x1:x2]
                            raw_activity, raw_status = self.analyze_activity(
                                roi_heatmap, roi_img, obj_id, cls_idx
                            )
                        elif roi_img is not None:
                            raw_activity, raw_status = self.analyze_activity(
                                None, roi_img, obj_id, cls_idx
                            )

                        # ── Temporal voting (stability) ───────────────────────
                        if obj_id not in self.activity_queues:
                            self.activity_queues[obj_id] = deque(
                                maxlen=config.VOTING_WINDOW
                            )
                        self.activity_queues[obj_id].append((raw_activity, raw_status))
                        voted_activity, voted_status = Counter(
                            self.activity_queues[obj_id]
                        ).most_common(1)[0][0]

                        # ── Utilization stats ─────────────────────────────────
                        if obj_id not in self.equipment_stats:
                            self.equipment_stats[obj_id] = {
                                "class_name": obj_name,
                                "working_frames": 0,
                                "waiting_frames": 0,
                            }
                        if voted_status == "Active":
                            self.equipment_stats[obj_id]["working_frames"] += 1
                        else:
                            self.equipment_stats[obj_id]["waiting_frames"] += 1

                        w_secs    = self.equipment_stats[obj_id]["working_frames"] / fps if fps > 0 else 0.0
                        i_secs    = self.equipment_stats[obj_id]["waiting_frames"] / fps if fps > 0 else 0.0
                        total_secs = w_secs + i_secs
                        u_pct     = (w_secs / total_secs * 100) if total_secs > 0 else 0.0

                        # ── Timestamp ─────────────────────────────────────────
                        ms     = int((frame_id / fps) * 1000) if fps > 0 else 0
                        hh, r  = divmod(ms / 1000, 3600)
                        mm, ss = divmod(r, 60)
                        ts_str = f"{int(hh):02d}:{int(mm):02d}:{int(ss):02d}.{int(ms%1000):03d}"

                        # ── Kafka telemetry ───────────────────────────────────
                        if int(w_secs) > 0 or int(i_secs) > 0:
                            payload = {
                                "frame_id": frame_id,
                                "equipment_id": f"{obj_name}-{obj_id}",
                                "equipment_class": obj_name.lower(),
                                "timestamp": ts_str,
                                "utilization": {
                                    "current_state": "ACTIVE" if voted_status == "Active" else "INACTIVE",
                                    "current_activity": voted_activity.upper(),
                                    "motion_source": (
                                        "articulated"
                                        if voted_activity.upper() in ["DIGGING", "SWINGING", "DUMPING"]
                                        else "tracks_only"
                                    ),
                                },
                                "time_analytics": {
                                    "total_tracked_seconds": round(total_secs, 1),
                                    "total_active_seconds": round(w_secs, 1),
                                    "total_idle_seconds": round(i_secs, 1),
                                    "utilization_percent": round(u_pct, 1),
                                },
                            }
                            if producer:
                                producer.produce(
                                    config.KAFKA_TOPIC,
                                    json.dumps(payload),
                                    callback=delivery_report,
                                )

                        # ── Visualization ─────────────────────────────────────
                        color = (0, 255, 0) if voted_status == "Active" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        label = f"{obj_name} #{obj_id}: {voted_activity}"
                        if voted_status == "Inactive":
                            label += " (Idle)"
                        cv2.putText(
                            frame, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 50, 0), 2,
                        )

                # ── Summary overlay ───────────────────────────────────────────
                self.draw_summary(frame, fps)

                # ── Heatmap split view ────────────────────────────────────────
                heatmap_view = np.zeros_like(frame)
                if heatmap_accum is not None:
                    heatmap_full = cv2.applyColorMap(heatmap_accum, cv2.COLORMAP_JET)
                    mask = np.zeros(heatmap_accum.shape, dtype=np.uint8)
                    for (x1, y1, x2, y2) in current_boxes:
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    heatmap_view = cv2.bitwise_and(heatmap_full, heatmap_full, mask=mask)

                combined_view = cv2.hconcat([frame, heatmap_view])
                cv2.imshow("EagleVision – Stable ID Tracking", combined_view)
                video_writer.write(combined_view)
                cv2.imwrite("out/current_frame.jpg", combined_view)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if producer:
                    producer.poll(0)

        finally:
            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()
            if producer:
                producer.flush()

    # ── summary panel ─────────────────────────────────────────────────────────

    def draw_summary(self, frame, fps):
        """Semi-transparent summary showing every stable object's cumulative time."""
        h_frame, w_frame = frame.shape[:2]
        overlay = frame.copy()

        valid_objs = []
        for obj_id in sorted(self.equipment_stats):
            stats   = self.equipment_stats[obj_id]
            w_secs  = int(stats["working_frames"] / fps) if fps > 0 else 0
            i_secs  = int(stats["waiting_frames"] / fps) if fps > 0 else 0
            if w_secs == 0 and i_secs == 0:
                continue
            valid_objs.append((obj_id, stats, w_secs, i_secs))

        if not valid_objs:
            return

        num_objs         = len(valid_objs)
        h_box, w_box     = 40 + (num_objs * 25), 460
        x_start, y_start = w_frame - w_box - 10, h_frame - h_box - 10
        x_end,   y_end   = w_frame - 10, h_frame - 10

        cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        t_name = config.TRACKER_TYPE.replace('.yaml', '').upper()
        cv2.putText(
            frame, f"── TRACKER: {t_name} ──",
            (x_start + 10, y_start + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )
        y_text = y_start + 45
        for obj_id, stats, w_secs, i_secs in valid_objs:
            name   = stats["class_name"]
            w_time = f"{w_secs//60}:{w_secs%60:02d}"
            i_time = f"{i_secs//60}:{i_secs%60:02d}"
            text   = f"ID {obj_id} ({name}): {w_time} Work | {i_time} Wait"
            color  = (0, 255, 0) if w_secs > 0 else (180, 180, 180)
            cv2.putText(
                frame, text, (x_start + 10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1,
            )
            y_text += 25


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gui_process = subprocess.Popen([sys.executable, "setting.py"])
    try:
        analyzer = ConstructionAnalyzer()
        analyzer.analyze_video("videos/v.mp4")
    finally:
        time.sleep(0.5)
        gui_process.terminate()
        print("Analysis complete! Video saved successfully.")