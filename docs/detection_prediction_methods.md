# Detection and Prediction Methods

This document explains the exact methods and techniques currently used in this project for crowd detection and future crowd prediction.

## 1) System Pipeline Overview

The current runtime pipeline is:
1. Read video frame
2. Detect persons with YOLO
3. Track persons with ID persistence and re-identification
4. Compute current crowd features (count, density, elapsed progress)
5. Compute action and crowd-attribute features from tracking behavior
6. Predict future crowd with trained neural network
7. Apply alert gating to reduce false positives
8. Render overlays and write prediction logs

### Quick Architecture Diagram

```text
Video Input
  |
  v
[YOLO Person Detection]
  |
  v
[Tracker + ReID + Stable IDs]
  |
  +--> Current Crowd Count
  +--> Density Ratio
  +--> Elapsed Progress
        |
        v
    [NN Crowd Predictor]
        |
        v
 [Alert Gating: confidence + threshold + streak]
        |
        +--> Future Prediction Panel (on-screen)
        +--> prediction_output_log.csv
```

---

## 2) Detection Module (YOLO)

### 2.1 Model
- Framework: Ultralytics YOLO
- Class used: `YOLO` from `ultralytics`
- Current checkpoint: `models/yolov8n.pt`
- Person-only filtering: class id `0`

### 2.2 Detection Strategy
Detection runs in two passes to improve recall:
1. Full-frame pass:
   - confidence threshold around 0.22
   - minimum bbox area filter to remove tiny/noisy detections
2. Zoomed top-region pass:
   - upper frame region is resized for distant persons
   - lower confidence threshold around 0.16
   - mapped back to original coordinates

### 2.3 Post-processing
- Non-maximum suppression (NMS) is applied to merged boxes.
- NMS reduces duplicate overlapping detections.
- Final output: cleaned list of person bounding boxes per frame.

### 2.4 Why this YOLO setup
- Improves small/distant person detection (important in crowd scenes).
- Balances precision vs recall by using two confidence levels.
- Maintains real-time feasibility on practical hardware.

---

## 3) Tracking and Counting Stability

### 3.1 Core Tracker
- Custom tracker class: `SimpleTracker`
- Matching method: nearest-center distance (greedy association)
- Track lifecycle:
  - new track creation
  - missed-frame handling
  - track deletion after max missed threshold

### 3.2 Stable Display IDs
- Display IDs are assigned after track confirmation rules (hit count and area checks).
- IDs are not intentionally recycled in the visualization flow.

### 3.3 Re-identification (ReID-like behavior)
To reduce duplicate counting for the same person:
- When a track expires, its last state is stored as retired memory.
- New detections are matched to retired memory by:
  - center distance threshold
  - area similarity ratio
  - recent-age threshold
- If matched, prior display ID is restored.

### 3.4 Result
- Better unique-count consistency
- Reduced ID fragmentation across occlusion/re-entry events

---

## 4) Prediction Module (Neural Network)

### 4.1 Prediction Target
- Predict future crowd count after a horizon (`horizon_steps`) based on recent history.

### 4.2 Input Features
The NN uses a context window with:
1. Recent crowd count sequence
2. Recent density sequence
3. Elapsed video ratio feature
4. Current action feature vector
5. Current crowd-attribute feature vector

This gives temporal + spatial + behavior context to prediction.

### 4.2.1 Count Signal Update
The current count series is built from a fused signal:
- tracker-confirmed count
- detector box count
- coverage-aware blending

This reduces undercount during longer gaps/occlusions where tracker confirmations may lag.

### 4.3 Model
- Library: scikit-learn
- Model: `MLPRegressor`
- Pipeline: `StandardScaler` + `MLPRegressor`
- Hidden layers: multi-layer dense network (example: 96, 48)
- Activation: ReLU
- Optimizer: Adam
- Early stopping enabled for generalization

### 4.4 Training Data
- Trained from demo videos generated/sampled from crowd footage
- Sliding-window dataset construction:
  - input window length = `window_size`
  - target = crowd count at `window + horizon_steps`

### 4.5 Model Artifacts
- Trained model: `models/crowd_predictor.joblib`
- Model metadata: `models/crowd_predictor_meta.json`
- Training metrics: `models/crowd_predictor_metrics.txt`

---

## 5) Prediction Activation Policy

Prediction is intentionally activated only after progress crosses 10% of video duration.
- Before that: warmup/analysis region
- After that: active NN-based prediction and logging

This avoids unreliable early-frame predictions.

---

## 6) False Alert Reduction Techniques

The project includes explicit false-alert controls:
1. Confidence threshold gate
2. Dynamic incoming threshold (`delta` must exceed crowd-relative threshold)
3. Consecutive streak requirement before final incoming signal

So incoming alerts are not raised from one noisy frame.

---

## 7) Runtime Output and Explainability

### 7.1 Red Prediction Panel
The panel displays:
- Current count
- Future predicted count
- Delta (future - current)
- Incoming status
- Confidence
- Trend hint
- Gate/threshold/streak status
- Prediction mode

### 7.2 Structured Log Output
`prediction_output_log.csv` stores detailed per-step prediction diagnostics for auditability.

---

## 8) Current Practical Limitation

The system is strong for prototype-level forecasting, but crowd forecasting remains difficult in open-world conditions.
Performance depends on:
- camera viewpoint
- occlusion severity
- density regime changes
- training data diversity

For production reliability, continue dataset expansion, recalibration, and site-specific validation.

---

## 9) Summary of Techniques Used

Detection side:
- YOLOv8 person detection
- two-pass detection (full frame + zoom region)
- NMS cleanup
- area/confidence filtering

Tracking side:
- nearest-center association
- confirmed track thresholds
- short-memory re-identification

Prediction side:
- feed-forward neural network (MLP)
- scaled temporal feature windows
- horizon forecasting
- confidence and gating-based alert suppression
