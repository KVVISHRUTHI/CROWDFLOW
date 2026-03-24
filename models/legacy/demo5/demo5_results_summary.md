# Demo-5 Training and Prediction Summary (Legacy)

This file is retained for historical reference.

Current training/evaluation outputs for the 13-video run:
- models/demo_training_results_summary.md
- models/demo_model_evaluation.txt
- models/demo_model_evaluation_summary.md

## Training Proof
- Demo videos used: 5
- Trained model: models/crowd_predictor.joblib
- Window size: 8
- Horizon steps: 4

## Validation Metrics (Training Split/Validation Split)
- samples: 95
- train_samples: 76
- val_samples: 19
- mae: 2.857
- mape_percent: 21.291

Source: models/crowd_predictor_metrics.txt

## Demo-5 Evaluation Metrics (Post-training evaluation)
- demo_video_count: 5
- videos_evaluated: 5
- avg_mae: 6.489
- avg_rmse: 7.178
- avg_mape_percent: 48.118
- avg_incoming_accuracy_percent: 63.333

Source: models/demo5_model_evaluation.txt

## Runtime Prediction Mode Proof
From prediction_output_log.csv after 10% progress, `prediction_mode` is:
- NN_ONLY_DEMO

This confirms runtime prediction is using the trained NN-based ensemble path.

False-alert reduction is enabled using:
- confidence threshold gating
- dynamic incoming threshold
- consecutive streak requirement before final incoming alert
