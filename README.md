
# EagleVision: Construction Equipment Monitoring & Analytics

EagleVision is a high-performance system designed for real-time monitoring of construction equipment. It combines deep learning for object detection, advanced temporal heatmap analysis, and a specialized secondary YOLO classification model to distinguish between stationary equipment and complex articulated motion (like digging, swinging, and dumping).

## 📸 Output Preview

![EagleVision Output](out/background.jpg)


## 🎥 Demonstration Video

[![EagleVision Demo Video](https://img.youtube.com/vi/ejC5qRzuI4Q/maxresdefault.jpg)](https://youtu.be/ejC5qRzuI4Q)

## 🚀 Key Features

### 1. Robust Computer Vision (Phase 1)
- **Object Detection & Tracking**: Uses YOLO object detection combined with **BoT-SORT** (incorporating Global Motion Compensation) to track **Excavators** and **Dump Trucks** with extremely high ID stability, robust to occlusions and camera jitter.
- **Temporal Heatmap Analysis**: The system accumulates motion masks over a rolling buffer to visualize "activity density" and easily bypass deep-learning checks when equipment is completely stationary ("Waiting").
- **Split-Screen Analytics**: Real-time output renders side-by-side: bounding boxes + tracking labels on the left, and the dense motion heatmap on the right.

### 2. Intelligent Activity Classification
- **Deep Learning Action Classifier**: When an excavator is detected as active, its cropped bounding box is passed to a secondary, specialized classification model (`classfy.pt`) to actively recognize **Swinging**, **Dumping**, or **Digging** with high accuracy.
- **Temporal Stability Voting**: Instead of flickering frame-by-frame, tracking maintains a temporal voting queue on activities, ensuring completely smoothed and confident activity polling.
- **Per-Object Diagnostics**: A transparent summary box dynamically tallies the active "Working" vs. "Waiting" time for every tracked machine individually in real-time.

### 3. Live Tuning & Control
- **Live Tuner GUI (`gui.py`)**: A standalone application module that auto-launches alongside the pipeline. It features real-time sliders to adjust motion sensitivity, waiting thresholds, and live ID-retention bounds (which safely rewrites `botsort.yaml` instantly without breaking the stream).

### 4. Distributed Data Pipeline
- **Kafka Streaming**: Streams real-time telemetry (equipment ID, status, activity, utilization) to a Kafka topic.
- **Data Persistence**: A dedicated consumer service persists all telemetry into a database (SQLite/Postgres).
- **Analytics Dashboard**: A **Streamlit** Web-UI providing realtime counters, historical utilization trends, and activity breakdown charts.

### 5. Automatic Recording
- **Clean Shutdowns & Exports**: The system fully traps and synchronizes quits. Hitting 'q' securely closes all running subprocesses (including the GUI) and finalizes a high-quality `.mp4` video output of the session (e.g., `output_bytetrack_20260402_0100.mp4`).

## 🛠 Project Structure

- `main.py`: The core CV Processor. Handles video input, tracking, action classification polling, and UI drawing.
- `gui.py`: The Live Tuner GUI. Auto-spawned by the main run to tweak `botsort.yaml` and `settings.json`.
- `config.py` & `settings.json`: System-wide logic parameters and live-tuning states.
- `botsort.yaml`: Global Motion Compensation configuration parameters, tuned live.
- `consumer.py` & `database.py`: The telemetry backbone for persisting Kafka streams.
- `dashboard.py`: The analytics Streamlit app.
- `classfy.pt` & `best.pt`: Standard and Activity-centric computer vision weights.

## 🏁 Getting Started

### Prerequisites
- Python 3.9+
- Docker (optional, for full microservices stack)

### Standalone Local Execution
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Processor**:
   ```bash
   python main.py
   # (This will automatically spawn your camera feed AND the control GUI)
   ```
3. **Control**:
   - Use the **Tuner GUI** sliders on a secondary window to refine ID tracking leniency or heatmap noise suppression.
   - Press **`q`** on the video feed to beautifully stop streaming, safely close the GUI, and export the analysis video.

### Distributed Stack
To run Kafka, Zookeeper, the DB, and Streamlit all as containers:
```bash
docker-compose up --build
```
*(Check `dashboard.py` locally via `streamlit run dashboard.py` if running offline).*

## 📐 Technical Methodology
EagleVision employs a dual-stage architecture. By using a fast YOLO detector paired with BoT-SORT's resilient motion compensator, bounding boxes are reliably anchored. Instead of forcing heavy logic into one layer, Eagle Vision isolates complex poses into a distinct sub-classifier (`classfy.pt`), while maintaining algorithmic speed through mathematical temporal heatmaps that preemptively filter out "Waiting" machines from computation entirely.

## ⚠️ Proof of Concept Limitations
Please note that this project is intended as a **Proof of Concept**. The current pipeline has some known limitations:
- **Camera Movement**: System performance and activity classification accuracy may degrade significantly if the camera is actively moving or vibrating.
- **Class Overlap/Intersection**: When there is intersection or heavy overlap between "moving" and "waiting" classes, or complex interactions between multiple machines, bounding box accuracy and overall detection accuracy can be negatively impacted.

## 🛣️ Commercial Roadmap & Future Analytics Pipeline
To transition from the current region-based motion and classification prototype to an robust commercial product, the following advanced vision techniques are planned for the immediate roadmap:

### 1. Excavator Keypoint Tracking (Pose Estimation)
Transitioning from standard bounding boxes and region heatmaps to full structural **Keypoint Tracking**. 
- By tracking specific joints on the excavator (cab, boom, stick, bucket), the system will precisely calculate the geometry of the articulated arm over time.
- This unlocks **deterministic** activity classification (e.g. true "arm-only" motion). Instead of classifying a cropped image, the pipeline will define "Digging" strictly by the kinetic angles and extension velocity of the tracked mechanical joints.

### 2. Instance Segmentation for Dump Trucks
Upgrading dump truck tracking to **Instance Segmentation** masks (e.g., YOLO26-seg).
- Pixel-perfect masking solves the "Class Overlap" limitations when trucks interact closely with active excavators during loading phases.
- Segmentation prevents motion bleed, ensuring that excavator bucket movements passing perfectly *in front* of a stationary truck body aren't incorrectly flagged as the truck itself moving.
