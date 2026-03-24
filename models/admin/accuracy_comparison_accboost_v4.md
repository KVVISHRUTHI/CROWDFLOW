# Accuracy Comparison: Baseline vs Candidate V4

## Baseline (current active model)

- Model: `models/crowd_predictor.joblib`
- Evaluation source: `models/demo_model_evaluation.json`
- Avg MAE: 2.835
- Avg RMSE: 3.277
- Avg MAPE%: 39.957
- Avg incoming accuracy%: 75.230

## Candidate V4 (safe experiment)

- Model: `models/crowd_predictor_accboost_v4.joblib`
- Evaluation source: `models/demo_model_evaluation_accboost_v4_onbase.json`
- Avg MAE: 2.770
- Avg RMSE: 3.214
- Avg MAPE%: 39.806
- Avg incoming accuracy%: 76.736

## Delta (V4 - Baseline)

- MAE: -0.065 (better)
- RMSE: -0.063 (better)
- MAPE%: -0.151 (better)
- Incoming accuracy%: +1.506 (better)

## Notes

- This run was isolated and did not overwrite the active model artifacts.
- To promote V4, replace `models/crowd_predictor.joblib` only after your approval.
