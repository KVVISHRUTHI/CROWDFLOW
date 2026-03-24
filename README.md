# Crowd Flow Management Prediction System

## Current Capabilities
- Person detection and tracking from uploaded video.
- Unique-ID based counting with improved ID consistency.
- Max crowd, average crowd, and risk status output.
- Live overlay with crowd flow and hotspot indicators.

## New Prediction Pipeline (Phase 1)
This project now includes a trainable neural-network crowd predictor.

- Training input: fused count + density + elapsed + action/attribute time-series from multiple videos.
- Model: MLP neural network (scikit-learn).
- Output: predicted crowd count after a fixed horizon.
- Runtime behavior: if trained model is unavailable, app falls back to regression.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Predictor from Demo Videos (Current: 13 Clips)
```bash
python train_crowd_predictor.py --videos data/crowd_videos/demo_training/*.mp4 --sample-every 4 --window-size 24 --horizon-steps 15 --optimize-level standard
```

If you do not have enough videos, auto-generate demo clips from existing videos:
```bash
python train_crowd_predictor.py --prepare-demo --demo-clips 13 --sample-every 4 --window-size 24 --horizon-steps 15 --optimize-level standard
```

Create a longer demo (for prediction trigger testing on 10% progress):
```bash
python build_long_demo_video.py
```

Artifacts created:
- models/crowd_predictor.joblib
- models/crowd_predictor_meta.json
- models/crowd_predictor_metrics.txt
- models/demo_training_report.json
- models/demo_training_series.json
- models/demo_training_results_summary.md
- models/demo_model_evaluation.json

### 3. Evaluate Retrained Predictor
```bash
python evaluate_demo_model.py
```

This writes:
- models/demo_model_evaluation.txt
- models/demo_model_evaluation.csv
- models/demo_model_evaluation.json
- models/demo_model_evaluation_summary.md

### 4. Run Main App with Prediction
```bash
python main.py
```

If `models/crowd_predictor.joblib` exists, neural prediction is used.
If not, automatic fallback prediction is used.

Runtime prediction starts after an initial analysis warmup (about 10-15 seconds).
Runtime now starts future prediction only after video progress crosses 10% (from the 11th percent onward),
and writes live prediction logs to `prediction_output_log.csv`.

Prediction input now includes:
- Fused count signal (tracker + detections)
- Density history
- Elapsed progress
- Action features (motion/entry pressure/coverage)
- Attribute features (area spread/occlusion profile)

The popup displays crowd incoming signal, confidence, and trend hint.

## How to View the Current Models

### Neural model (retrained)
1. Open metrics in [models/crowd_predictor_metrics.txt](models/crowd_predictor_metrics.txt).
2. Open evaluation summary in [models/demo_model_evaluation.txt](models/demo_model_evaluation.txt).
3. Open explainability report in [docs/training_explainability.html](docs/training_explainability.html).

### YOLO model (detection model)
Current detection model is loaded from pretrained checkpoint:
- [models/yolo_model.py](models/yolo_model.py)
- weights: [models/yolov8n.pt](models/yolov8n.pt)

To print both model details quickly:
```bash
python inspect_models.py
```

Note: YOLO fine-tuning needs labeled dataset files (dataset yaml + labels). This repository currently has inference weights and runtime integration, but no annotation dataset yet.

## Documentation
- Training/report field definitions: docs/demo_training_output_fields.md
- Separate neural vs YOLO training/metrics guide: docs/model_training_and_metrics_guide.md
- Neural-only updated accuracy report: docs/nn_model_updated_accuracy_report.md
- Generated report outputs:
	- models/demo_training_report.json
	- models/demo_training_report.csv
	- models/demo_training_results_summary.md
	- models/demo_model_evaluation_summary.md
	- docs/training_explainability.html

## Capacity Module (Phase 2 Foundation)
Added `processing/location_capacity.py` for location-based capacity estimation:
- Area estimation (rectangle/circle/ellipse)
- Safe capacity estimation based on density rules
- Utilization status (LOW to OVER_CAPACITY)

This module is the base for CCTV + geolocation + capacity-aware forecasting.

## Recommended Next Steps
1. Expand training set diversity beyond demo clips (lighting, viewpoint, density regimes).
2. Retrain and compare evaluation metrics after each dataset update.
3. Tune `window-size` and `horizon-steps` per operational lead-time target.
4. Add external context features (events/weather/time slots).
5. Add labeled dataset pipeline if YOLO fine-tuning is required.