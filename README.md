# 🚦 Crowd Flow Management & Prediction System

> Real-time crowd detection, tracking, and neural-network-powered density forecasting — built on YOLOv8 and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-purple?logo=opencv)
![scikit-learn](https://img.shields.io/badge/Model-scikit--learn-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)


## 📸 System Preview

<p align="center">
  <img width="617" height="341" alt="Crowd Detection Overview" src="https://github.com/user-attachments/assets/cf801dae-bc2e-4a11-a538-b405ebe98bbf" />
</p>

<p align="center">
  <img width="614" height="342" alt="Flow Tracking Visualization" src="https://github.com/user-attachments/assets/24864ee4-f68f-475a-9772-44e5a5fdde83" />
</p>

<p align="center">
  <img width="636" height="370" alt="Hotspot & Density Overlay" src="https://github.com/user-attachments/assets/b4ba0384-b3aa-4d11-99cf-e9cfafc679d8" />
</p>

<p align="center">
  <img width="501" height="276" alt="Prediction Output Panel" src="https://github.com/user-attachments/assets/0f14a5a3-a83b-4caa-adef-747f8f83e3c3" />
</p>

---

## 📁 Project Structure

<img width="672" height="841" alt="image" src="https://github.com/user-attachments/assets/ff4442dd-e8ec-4552-9466-4688e7401e41" />


---

## ✅ Current Capabilities

| Feature | Description |
|--------|-------------|
| 👤 Person Detection & Tracking | Detects and tracks individuals from uploaded video |
| 🆔 Unique ID Counting | Improved ID consistency for accurate crowd counting |
| 📊 Crowd Statistics | Max crowd, average crowd, and live risk status |
| 🗺️ Live Overlay | Real-time crowd flow and hotspot visualisation |
| 🤖 Neural Prediction | MLP-based crowd forecasting with fallback regression |
| 📍 Capacity Estimation | Area-based safe capacity with utilisation status |

---

## 🧠 Prediction Pipeline (Phase 1)

A trainable **MLP neural network** (scikit-learn) predicts future crowd counts based on fused time-series signals.

### Input Features
- Fused count signal (tracker + detections)
- Density history
- Elapsed video progress
- Action features: motion, entry pressure, area coverage
- Attribute features: area spread, occlusion profile

### Output
- Predicted crowd count after a fixed horizon
- Confidence score and trend hint (shown in popup)
- Fallback to regression if trained model is unavailable

> ⏱️ Prediction starts after **~10–15s warmup** and activates once video progress crosses **10%** (from the 11th percentile onward). Live logs are written to `prediction_output_log.csv`.

---

## 🚀 Getting Started

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 2 — Train the Predictor

**Option A: Train from your own demo videos (currently supports 13 clips)**

```bash
python train_crowd_predictor.py \
  --videos data/crowd_videos/demo_training/*.mp4 \
  --sample-every 4 \
  --window-size 24 \
  --horizon-steps 15 \
  --optimize-level standard
```

**Option B: Auto-generate demo clips from existing videos**

```bash
python train_crowd_predictor.py \
  --prepare-demo \
  --demo-clips 13 \
  --sample-every 4 \
  --window-size 24 \
  --horizon-steps 15 \
  --optimize-level standard
```

**Option C: Build a longer demo video (for 10% progress trigger testing)**

```bash
python build_long_demo_video.py
```

#### 📦 Artifacts Generated

```
models/
├── crowd_predictor.joblib
├── crowd_predictor_meta.json
├── crowd_predictor_metrics.txt
├── demo_training_report.json
├── demo_training_series.json
├── demo_training_results_summary.md
├── demo_model_evaluation.json
```

---

### Step 3 — Evaluate the Retrained Model

```bash
python evaluate_demo_model.py
```

#### 📄 Evaluation Outputs

```
models/
├── demo_model_evaluation.txt
├── demo_model_evaluation.csv
├── demo_model_evaluation.json
├── demo_model_evaluation_summary.md
```

---

### Step 4 — Run the Main App

```bash
python main.py
```

- ✅ If `models/crowd_predictor.joblib` **exists** → Neural prediction is used
- ⚠️ If not found → Automatic fallback prediction is used

---

## 🔍 Inspecting Models

### Neural Predictor

| Resource | Path |
|----------|------|
| Metrics | [`models/crowd_predictor_metrics.txt`](models/crowd_predictor_metrics.txt) |
| Evaluation Summary | [`models/demo_model_evaluation.txt`](models/demo_model_evaluation.txt) |
| Explainability Report | [`docs/training_explainability.html`](docs/training_explainability.html) |

### YOLO Detection Model

| Resource | Path |
|----------|------|
| Model Script | [`models/yolo_model.py`](models/yolo_model.py) |
| Pretrained Weights | [`models/yolov8n.pt`](models/yolov8n.pt) |

> 📝 **Note:** YOLO fine-tuning requires a labeled dataset (YAML config + annotation labels). This repo currently ships with inference weights and runtime integration only — no annotation dataset yet.

**Quick model inspection:**

```bash
python inspect_models.py
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [`docs/demo_training_output_fields.md`](docs/demo_training_output_fields.md) | Training/report field definitions |
| [`docs/model_training_and_metrics_guide.md`](docs/model_training_and_metrics_guide.md) | Neural vs YOLO training & metrics guide |
| [`docs/nn_model_updated_accuracy_report.md`](docs/nn_model_updated_accuracy_report.md) | Neural-only accuracy report |
| [`docs/training_explainability.html`](docs/training_explainability.html) | Feature explainability visualisation |
| [`models/demo_training_results_summary.md`](models/demo_training_results_summary.md) | Training run summary |
| [`models/demo_model_evaluation_summary.md`](models/demo_model_evaluation_summary.md) | Evaluation summary |

---

## 📍 Capacity Module (Phase 2 Foundation)

`processing/location_capacity.py` provides location-aware crowd capacity estimation.

**Supported area shapes:** Rectangle · Circle · Ellipse

**Utilisation Status Levels:**

```
LOW → MODERATE → HIGH → CRITICAL → OVER_CAPACITY
```

This module forms the base for **CCTV + geolocation + capacity-aware forecasting**, enabling density-rule-based safe capacity estimation per zone.

---

## 🗺️ Recommended Next Steps

- [ ] Expand training set diversity — vary lighting, viewpoint, and density regimes
- [ ] Retrain and compare evaluation metrics after each dataset update
- [ ] Tune `--window-size` and `--horizon-steps` per operational lead-time target
- [ ] Add external context features (events, weather, time slots)
- [ ] Build labeled annotation pipeline to enable YOLO fine-tuning

---

---

<p align="center">Made with 🧠 for smarter, safer spaces.</p>
