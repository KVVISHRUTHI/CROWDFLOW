# Demo Training Output Fields

This document describes the training outputs created after running the demo-based training command.

## Artifacts
- models/crowd_predictor.joblib: trained neural predictor
- models/crowd_predictor_meta.json: model runtime settings
- models/crowd_predictor_metrics.txt: global validation metrics
- models/demo_training_report.json: detailed training report
- models/demo_training_report.csv: per-video summary for quick review
- models/demo_training_series.json: archived extracted sequences used for training
- models/demo_training_results_summary.md: condensed training + candidate ranking summary
- models/demo_model_evaluation.json: post-training evaluation summary and per-video rows
- models/demo_model_evaluation.txt: readable evaluation snapshot
- models/demo_model_evaluation_summary.md: markdown evaluation summary

## Metrics Fields (crowd_predictor_metrics.txt)
- samples: total sliding-window training rows used
- train_samples: rows used for model fitting
- val_samples: rows used for validation
- mae: mean absolute error on validation split
- rmse: root mean squared error on validation split
- mape_percent: mean absolute percentage error on validation split
- selection_score: composite selector score used for model candidate comparison
- best_config_name: winning candidate name
- best_hidden_layers: hidden layer topology of winning candidate
- best_alpha: regularization value for winning candidate
- best_learning_rate_init: initial learning rate for winning candidate
- best_max_iter: max iterations for winning candidate
- candidate_count: number of tested candidate models
- shuffle_split: 1 when shuffled split is enabled
- selection_leaderboard: list of candidate metrics (name, mae, rmse, mape_percent, score)

## Per-Video Report Fields (demo_training_report.json / .csv)
- video_path: source video path used in training
- samples: number of sampled frames from that video
- count_min: minimum detected people count among sampled frames
- count_max: maximum detected people count among sampled frames
- count_mean: average detected people count among sampled frames
- density_mean: average frame occupancy ratio from person boxes
- elapsed_start: first elapsed ratio in sampled timeline
- elapsed_end: last elapsed ratio in sampled timeline

## Series Archive Fields (demo_training_series.json)
- counts: fused crowd count signal used for training
- densities: occupancy ratio sequence
- elapsed: elapsed-progress sequence
- actions: per-step action feature vector
	- action[0]: normalized motion speed
	- action[1]: moving-track ratio
	- action[2]: entry pressure estimate
	- action[3]: tracker coverage
- attributes: per-step crowd-attribute vector
	- attribute[0]: normalized average box area
	- attribute[1]: area variation (coefficient)
	- attribute[2]: spatial spread indicator
	- attribute[3]: occlusion ratio proxy

## Evaluation Summary Fields (demo_model_evaluation.*)
- model_path: evaluated model artifact path
- window_size: history window used in that run
- horizon_steps: prediction lead steps used in that run
- demo_video_count: videos available in evaluation set
- videos_evaluated: videos that had enough samples for evaluation
- avg_mae: average mean absolute error across videos
- avg_rmse: average root mean squared error across videos
- avg_mape_percent: average percentage error across videos
- avg_incoming_accuracy_percent: average incoming-signal classification accuracy

## Runtime Prediction Log Fields (prediction_output_log.csv)
- frame: frame index where output was logged
- elapsed_percent: percent of total video completed
- current_count: current crowd estimate
- future_count: NN-only future crowd prediction
- delta: future_count - current_count
- incoming_raw: 1 if raw gate condition passes (before streak filtering)
- incoming: 1 if final incoming signal is active after streak filtering
- incoming_threshold: dynamic delta threshold for incoming detection
- incoming_streak: consecutive raw-pass frames count
- confidence_percent: ensemble confidence heuristic score
- alert_score: confidence-weighted normalized alert strength
- risk_hint: STRONG_INCREASE, LIKELY_INCREASE, LIKELY_DROP, or STABLE
- prediction_mode: NN_ONLY_DEMO or NN_WARMUP_OR_UNAVAILABLE
- nn_pred: neural model prediction
- nn_ready: 1 when NN prediction is active after 10% and warmup window
- density_ratio: current estimated occupancy ratio from detections
- status: system risk status label
