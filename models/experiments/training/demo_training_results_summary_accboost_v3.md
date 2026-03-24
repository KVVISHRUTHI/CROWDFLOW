# Demo Training Results Summary

- Demo videos used: 13
- Total extracted samples: 8564
- Window size: 24
- Horizon steps: 15
- sample_every: 4
- optimize_level: standard
- shuffle_split: True

- selection_target: incoming_accuracy
- incoming_threshold_ratio: 0.12
- incoming_threshold_min: 2

## Best Validation Metrics

- MAE: 0.722
- RMSE: 1.089
- MAPE%: 42.445
- Incoming accuracy%: 92.627
- Best model config: std_wider
- Hidden layers: 128-64
- Alpha: 0.0008
- Learning rate init: 0.0008

## Candidate Leaderboard

1. std_balanced -> score -92.3531, MAE 0.7379, RMSE 1.1284, MAPE 43.7854
2. std_wider -> score -92.6017, MAE 0.7221, RMSE 1.0886, MAPE 42.4452
3. std_regularized -> score -92.1053, MAE 0.7562, RMSE 1.0938, MAPE 46.8498
4. std_dense -> score -92.1672, MAE 0.7488, RMSE 1.111, MAPE 45.7414
