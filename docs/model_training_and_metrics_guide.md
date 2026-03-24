# Model Training and Metrics Guide (Separate: Neural and YOLO)

This document explains, separately for each model:
- how the model is trained,
- what metrics are used,
- where to see current accuracy,
- how to update metrics after retraining.

## 1) Neural Network Model (Crowd Predictor)

### 1.1 What this model does
- Forecasts future crowd count after a fixed horizon.
- Uses time-series features from video frames.
- Current setup in this project:
  - window_size: 24
  - horizon_steps: 15
  - feature_version: 2 (count + density + elapsed + action + attribute features)

### 1.2 How it is trained
Training script: train_crowd_predictor.py

Training flow:
1. Read training videos (currently demo clips).
2. Run YOLO person detection on sampled frames.
3. Run tracker and fuse detector/tracker count for robust count signal.
4. Build feature vectors per frame:
   - count history,
   - density history,
   - elapsed ratio,
   - action features,
   - attribute features.
5. Train scikit-learn MLP regressor pipeline.
6. Save model + metadata + training reports.

Run training:

```powershell
D:/projects/Crowd_System_AI/.venv/Scripts/python.exe train_crowd_predictor.py --videos data/crowd_videos/demo_training/*.mp4 --sample-every 4 --window-size 24 --horizon-steps 15 --optimize-level standard
```

### 1.3 Neural metrics (meaning)
- MAE: Mean Absolute Error (in people count). Lower is better.
- RMSE: Root Mean Squared Error (penalizes large misses). Lower is better.
- MAPE: Mean Absolute Percentage Error. Lower is better, but can be unstable at low counts.
- Incoming Accuracy %: Binary trend accuracy for crowd-incoming signal. Higher is better.

Useful formula summary:

$$
MAE = \frac{1}{N}\sum_{i=1}^{N} |y_i - \hat{y}_i|
$$

$$
RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}
$$

$$
MAPE = \frac{100}{N}\sum_{i=1}^{N}\left|\frac{y_i-\hat{y}_i}{\max(y_i,1)}\right|
$$

### 1.4 Current neural accuracy (latest available)
From models/crowd_predictor_metrics.txt (training split summary):
- samples: 8070
- train_samples: 6456
- val_samples: 1614
- mae: 0.722
- rmse: 1.089
- mape_percent: 42.445
- best_config_name: std_wider

From models/demo_model_evaluation.json (13-video evaluation summary):
- avg_mae: 2.835
- avg_rmse: 3.277
- avg_mape_percent: 39.957
- avg_incoming_accuracy_percent: 75.23

### 1.5 How to update neural accuracy files
1. Retrain model:

```powershell
D:/projects/Crowd_System_AI/.venv/Scripts/python.exe train_crowd_predictor.py --videos data/crowd_videos/demo_training/*.mp4 --sample-every 4 --window-size 24 --horizon-steps 15 --optimize-level standard
```

2. Re-evaluate model:

```powershell
D:/projects/Crowd_System_AI/.venv/Scripts/python.exe evaluate_demo_model.py
```

3. Check updated files:
- models/crowd_predictor_metrics.txt
- models/demo_training_results_summary.md
- models/demo_model_evaluation.json
- models/demo_model_evaluation.csv
- models/demo_model_evaluation.txt

---

## 2) YOLO Model (Person Detection)

### 2.1 What this model does
- Detects people in each frame.
- Current loader: models/yolo_model.py
- Current checkpoint used in runtime: models/yolov8n.pt

### 2.2 How it is currently used/trained in this repository
- Current repository state uses pretrained YOLO weights for inference.
- There is no local labeled dataset configuration (dataset yaml + label files) stored for YOLO fine-tuning in this repo.
- Because of that, YOLO retraining metrics (mAP/precision/recall on your custom dataset) are not currently generated here.

### 2.3 YOLO metrics (meaning)
- mAP@0.5: mean average precision at IoU 0.5. Higher is better.
- mAP@0.5:0.95: stricter averaged mAP over IoU thresholds. Higher is better.
- Precision: fraction of predicted detections that are correct. Higher is better.
- Recall: fraction of true objects that were detected. Higher is better.

### 2.4 Current YOLO accuracy status
- Custom training accuracy is not available yet in this repository because YOLO has not been fine-tuned on a labeled local dataset.
- Runtime model in use is pretrained yolov8n.pt.

### 2.5 How to update YOLO accuracy (after dataset is ready)
Required first:
1. Prepare dataset with labels.
2. Create dataset yaml file (paths, class names).

Then train/validate YOLO and save metrics:

```powershell
# Example with Ultralytics CLI (adjust paths)
# yolo train model=models/yolov8n.pt data=path/to/dataset.yaml epochs=50 imgsz=640
# yolo val model=path/to/best.pt data=path/to/dataset.yaml
```

Expected metrics to record in docs/artifacts after YOLO fine-tuning:
- precision
- recall
- mAP@0.5
- mAP@0.5:0.95

Recommended artifact to add after YOLO fine-tune:
- models/yolo_finetune_metrics.json (or .txt)

---

## 3) Quick Model Visibility Commands

Inspect both model details:

```powershell
D:/projects/Crowd_System_AI/.venv/Scripts/python.exe inspect_models.py
```

This prints:
- neural model metadata and saved training metrics,
- YOLO loader and checkpoint source.
