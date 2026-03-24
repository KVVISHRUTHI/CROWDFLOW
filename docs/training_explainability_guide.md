# Training Explainability HTML Guide

This guide explains all content shown in [docs/training_explainability.html](docs/training_explainability.html).

## Purpose of the HTML Page

The page is a visual report for understanding:
1. How the NN crowd model was trained from the current demo video set (now 13 clips)
2. What configuration was used
3. What metrics were obtained
4. Whether the model quality is acceptable for current stage

It is designed for quick review, demo presentation, and project documentation.

## Section-by-Section Explanation

## 1) Training Process

This section describes the full flow used to create the model:
1. Demo videos are prepared
2. Person counts and density are extracted frame-by-frame
3. Action and crowd-attribute features are extracted from motion/tracking behavior
4. Sliding windows are created for supervised learning
5. MLP neural network is trained and validated
6. Model and reports are saved in models folder

This gives traceability from raw videos to final predictor.

## 2) Training Configuration

This block shows the key hyperparameters used in that run:
- `sample_every`: how often frames were sampled from video
- `window_size`: history length used as input to NN
- `horizon_steps`: how far ahead the model predicts
- `resize`: frame resolution used during extraction

If performance changes, this section helps compare runs quickly.

Also shown in latest config:
- `feature_version`
- `action_feature_size`
- `attribute_feature_size`

## 3) Validation KPIs (Training)

These are model quality indicators from train/validation split:
- `samples`: total training rows
- `train_samples`: rows used for training
- `val_samples`: rows used for validation
- `MAE`: mean absolute error
- `MAPE %`: percentage error (relative)

Interpretation rule:
- Lower MAE and lower MAPE are better.

## 4) Per-Video Training Summary (Current Demo Videos)

This table summarizes each training video’s crowd profile:
- `Samples`: sampled frames from that video
- `Count Min/Max/Mean`: crowd count range and average
- `Density Mean`: average occupancy ratio estimate

This section tells whether the training set is diverse or narrow.

## 5) Post-Training Evaluation Summary

This section shows aggregated metrics after evaluating model behavior on the demo set:
- `avg_mae`
- `avg_rmse`
- `avg_mape_percent`
- `avg_incoming_accuracy_percent`

This is a more practical quality view beyond training split metrics.

## 6) Per-Video Evaluation Rows

This table breaks performance by each demo video:
- MAE, RMSE, MAPE per video
- Incoming crowd classification accuracy per video

Use this to identify which scenarios are difficult for the model.

## 7) Sampled Series Preview

This section shows extracted sequence previews:
- count series preview
- density series preview
- action series preview
- attribute series preview

It confirms that feature extraction is working and data is not empty/corrupted.

## Data Sources Used by the HTML

The page is generated from these files:
- [models/crowd_predictor_metrics.txt](models/crowd_predictor_metrics.txt)
- [models/demo_training_report.json](models/demo_training_report.json)
- [models/demo_training_series.json](models/demo_training_series.json)
- [models/demo_model_evaluation.json](models/demo_model_evaluation.json)

If one of these files is missing, the HTML may show partial content.

## How to Regenerate the HTML

Run:

```powershell
python generate_training_explainability_html.py
```

Output file:
- [docs/training_explainability.html](docs/training_explainability.html)

## How to Present This in Review Meetings

Recommended order:
1. Training Process
2. Training Configuration
3. Validation KPIs
4. Per-Video Summary
5. Post-Training Evaluation
6. Per-Video Evaluation
7. Sampled Series Preview

This order gives a clear narrative from process to evidence.

## Practical Note

This page explains the current run and current demo data. As you add more real videos, regenerate the report so metrics and tables reflect the latest model state.

## How to View Current Retrained Models

Neural model:
- Metrics: [models/crowd_predictor_metrics.txt](models/crowd_predictor_metrics.txt)
- Evaluation summary: [models/demo_model_evaluation.txt](models/demo_model_evaluation.txt)
- HTML report: [docs/training_explainability.html](docs/training_explainability.html)

YOLO model in current project state:
- Loader: [models/yolo_model.py](models/yolo_model.py)
- Checkpoint: [models/yolov8n.pt](models/yolov8n.pt)
- Runtime detector logic: [processing/detection.py](processing/detection.py)

For a quick terminal summary of both models, run:

```powershell
python inspect_models.py
```
