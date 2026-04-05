
# EagleVision: Construction Equipment Monitoring & Analytics

EagleVision is a high-performance system designed for real-time monitoring of construction equipment. It combines deep learning for object detection, advanced temporal heatmap analysis, and a specialized secondary YOLO classification model to distinguish between stationary equipment and complex articulated motion (like digging, swinging, and dumping).

## 🧪 Model Training (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1v10q56t3xePM5d2VELDXCam7t2kToET0?usp=sharing)
## 📸 Output Preview
![Setting](out/setting.png)
![EagleVision Output](out/background.jpg)


## 🎥 Demonstration Video

[![EagleVision Demo Video](out/background.jpg)](https://www.youtube.com/watch?v=7ekC4IPvhvU&feature=youtu.be)

The model training notebook is available on Google Colab. It walks through the full training pipeline for the activity classification and detection models used in EagleVision.

## 🚀 Key Features

### 1. Advanced Tracking & ID Stability
- **Multi-Tracker Support**: Real-time switching between **BoT-SORT**, **StrongSORT++**, and **Deep-OC-SORT**.
  - **BoT-SORT**: Balanced performance with Global Motion Compensation (GMC). Optimized for high-end GPUs.
  - **StrongSORT++**: accuracy-focused using heavy ReID features.
  - **Deep-OC-SORT**: Optimized for heavy occlusions and non-linear machine movements (e.g., excavator rotations).
- **Proprietary ID Stabilization Layer**: A post-processing anti-ID-switch layer that handles the "Centroid Jump" problem during machine rotations.
  #### 4-Layer Matching Strategy:
  - **Layer 1: Raw-ID Continuity**: Zero-cost mapping for standard tracker persistence.
  - **Layer 2: Single-Instance Fast-Path**: Pose-change resistant logic that matches 1 lost track with 1 new detection of the same class when unique on site.
  - **Layer 3: Zone-Anchor Matching**: Spatial memory using a running mean centroid (anchor) to snap detections back to stable IDs.
  - **Layer 4: Trajectory Prediction**: Physics-aware fallback using linear extrapolation and box expansion.

### 2. Intelligent Activity Classification
- **Deep Learning Action Classifier**: When an excavator is detected as active, its cropped bounding box is passed to a secondary, specialized classification model (`classfy.pt`) to recognized **Swinging**, **Dumping**, or **Digging**.
- **Temporal Heatmap Analysis**: Accumulates motion masks to visualize "activity density" and bypass deep-learning checks when equipment is completely stationary ("Waiting").
- **Stability Voting**: A temporal voting queue ensures smoothed, flicker-free activity status.

### 3. Distributed Data Pipeline
- **Kafka Streaming**: Streams real-time telemetry (ID, status, activity, utilization) to a Kafka topic.
- **Analytics Dashboard**: A **Streamlit** Web-UI providing realtime counters, historical utilization trends, and activity breakdown charts.
- **Data Persistence**: Telemetry is automatically persisted into a database (SQLite/Postgres).

### 4. Live Tuning GUI
- **Instant Live Tuning**: Adjust motion sensitivity, waiting thresholds, and tracker parameters (like `track_buffer` and `match_thresh`) on-the-fly.
- **Dynamic Config**: Safely rewrites `.yaml` configurations and `settings.json` reflected instantly in the video stream.

### 5. Flexible Analysis Modes
- **Full Pipeline (`main.py`)**: Complete analytics with Heatmaps, Action Classification, and Kafka streaming.
- **Tracking-Only Mode (`main_track.py`)**: Speed-optimized mode focused strictly on high-performance tracking and ID stability (no heatmap/activity processing).

## 🛠 Project Structure

The project is organized into a modular core and supporting microservices:

- **Core Processors**:
  - `main.py`: The primary engine. orchestrates YOLO detection, specialized BoT-SORT/StrongSORT/Deep-OC-SORT tracking, activity heatmapping, and real-time Kafka telemetry.
  - `main_track.py`: Optimized, high-performance tracking-only entry point with anti-ID-switch stabilization layer.
- **System Control & UI**:
  - `setting.py`: Interactive GUI Control Panel for real-time parameter tuning (confidence thresholds, motion sensitivity, tracker buffers).
  - `dashboard.py`: Streamlit-driven analytics dashboard for real-time fleet activity monitoring and historical utilization reports.
- **Configuration & Persistence**:
  - `config.py` & `settings.json`: Centralized control logic for system parameters and live state persistence.
- **Tracker Configuration Subsystem**:
  - `botsort.yaml`: High-stability Global Motion Compensation (GMC) parameters.
  - `bytetrack.yaml`: High-speed detection-only tracking without appearance association.
  - `deepocsort.yaml`: Robust occlusion handling for non-linear equipment motion.
  - `strongsort.yaml`: Heavy ReID (Feature Appearance) based identity persistence.
- **Telemetry & Backend**:
  - `consumer.py`: Kafka consumer service to ingest and process real-time status updates.
  - `database.py`: ORM management for persisting analytics into SQLite (dev) or PostgreSQL (prod).
- **Inference & Models**:
  - `models/`: Pre-trained weights for heavy equipment detection (`best.pt`) and excavator activity classification (`classfy.pt`).
- **DevOps & Deployment**:
  - `Dockerfile` & `docker-compose.yml`: Comprehensive container stack containing Kafka, Zookeeper, and Database services.
  - `run_all.bat`: One-click utility script to launch the full microservices stack and core pipeline.

## ⚙️ Requirements

### 💻 System Specifications
- **Python**: 3.9+ (Active LTS)
- **OS**: Windows / Linux (Docker-ready recommended)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (Recommended for real-time multi-stream inference).

### 📦 Key Dependencies
- `ultralytics`: YOLOv11 detection and tracking backbone.
- `confluent-kafka`: Distributed streaming for real-time telemetry.
- `streamlit`: High-performance analytics visualization.
- `sqlalchemy`: Database abstraction and lifecycle management.
- `opencv-python`: Core video stream processing and GUI controls.

## 🏁 Getting Started

### Quick Start
A
1. **Prepare Environment**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Pipeline**:
   ```bash
   python main.py
   ```
3.**run Full System** :
     ```bash
   run_all.bat
   ```
B. **Run High-Speed Tracking Only**:
   ```bash
   python main_track.py
   ```

## 📐 Technical Methodology
EagleVision employs a dual-stage architecture. By using a fast YOLO detector paired with resilient motion compensators (BoT-SORT/Deep-OC-SORT), bounding boxes are reliably anchored. Instead of forcing heavy logic into one layer, Eagle Vision isolates complex poses into a distinct sub-classifier (`classfy.pt`), while maintaining algorithmic speed through mathematical temporal heatmaps that filter out "Waiting" machines.

## ⚠️ Proof of Concept Limitations
- **Camera Movement**: Accuracy may degrade if the camera is actively moving or vibrating.
- **Class Overlap**: Heavy overlap between equipment may impact detection accuracy.

## 🛣️ Commercial Roadmap
- **Excavator Pose Estimation**: Transitioning to keypoint tracking for deterministic activity classification.
- **Instance Segmentation**: Pixel-perfect masking for trucks to solve overlap issues in loading zones.
