# Neural Network Model Updated Accuracy Report

This document is a dedicated, NN-only report for the latest trained crowd predictor.
All values below are copied from the current model metric files in the repository.

## Model Context
- Model file: models/crowd_predictor.joblib
- Window size: 24
- Horizon steps: 15
- Evaluated demo videos: 13

## Source Files Used
- models/crowd_predictor_metrics.txt
- models/demo_model_evaluation.json
- models/demo_model_evaluation.txt

## A) Training Split Metrics (from crowd_predictor_metrics.txt)

| Metric | Value |
|---|---:|
| samples | 8070 |
| train_samples | 6456 |
| val_samples | 1614 |
| mae | 0.722 |
| rmse | 1.089 |
| mape_percent | 42.445 |
| best_config_name | std_wider |

Meaning:
- mae is average absolute error in predicted people count.
- mape_percent is percentage error and can look large when true counts are small.

## B) Full Evaluation Summary (from demo_model_evaluation.*)

| Metric | Value |
|---|---:|
| avg_mae | 2.835 |
| avg_rmse | 3.277 |
| avg_mape_percent | 39.957 |
| avg_incoming_accuracy_percent | 75.23 |

Meaning:
- avg_mae: average absolute crowd-count prediction error across evaluated videos.
- avg_rmse: error metric that penalizes larger misses.
- avg_mape_percent: average percentage error.
- avg_incoming_accuracy_percent: binary trend accuracy for incoming crowd signal.

## C) Per-Video Metrics (from demo_model_evaluation.json)

| Video | Samples | MAE | RMSE | MAPE % | Incoming Accuracy % |
|---:|---:|---:|---:|---:|---:|
| 1 | 111 | 8.027 | 8.382 | 50.741 | 45.045 |
| 2 | 111 | 7.225 | 7.612 | 50.335 | 48.649 |
| 3 | 111 | 5.739 | 6.592 | 52.630 | 60.360 |
| 4 | 111 | 2.180 | 2.627 | 15.674 | 68.468 |
| 5 | 111 | 1.423 | 1.821 | 12.325 | 74.775 |
| 6 | 72 | 0.639 | 1.014 | 52.778 | 93.056 |
| 7 | 36 | 4.139 | 4.438 | 14.524 | 75.000 |
| 8 | 126 | 1.143 | 1.584 | 89.021 | 84.921 |
| 9 | 363 | 3.887 | 4.501 | 34.879 | 61.708 |
| 10 | 23 | 0.870 | 1.383 | 26.739 | 86.957 |
| 11 | 1786 | 0.314 | 0.646 | 26.105 | 97.424 |
| 12 | 1231 | 0.522 | 0.878 | 41.079 | 92.770 |
| 13 | 3223 | 0.752 | 1.128 | 52.612 | 88.861 |

## D) Accuracy Interpretation
- Primary quality metric for this project: MAE (people count units).
- Current evaluated MAE: 2.835 means average miss is about 2 to 3 people.
- Incoming trend accuracy: 75.23% means trend classification is correct in about 3 out of 4 evaluated samples.

## E) How to Refresh This Report After Next Retraining
1. Retrain model.
2. Re-run evaluation.
3. Replace values in this file from the latest metric files listed above.

Commands:

```powershell
D:/projects/Crowd_System_AI/.venv/Scripts/python.exe train_crowd_predictor.py --videos data/crowd_videos/demo_training/*.mp4 --sample-every 4 --window-size 24 --horizon-steps 15 --optimize-level standard
D:/projects/Crowd_System_AI/.venv/Scripts/python.exe evaluate_demo_model.py
```
