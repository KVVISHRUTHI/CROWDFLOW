# Demo Training Results Summary

- Demo videos used: 13
- Total extracted samples: 8564
- Window size: 24
- Horizon steps: 15
- sample_every: 4
- optimize_level: aggressive
- shuffle_split: True

- selection_target: composite
- incoming_threshold_ratio: 0.12
- incoming_threshold_min: 2

- max_rows_per_series: 600

## Best Validation Metrics

- MAE: 0.946
- RMSE: 1.487
- MAPE%: 32.874
- Incoming accuracy%: 90.682
- Best model config: agg_slim_highreg
- Hidden layers: 80-40
- Alpha: 0.0018
- Learning rate init: 0.0012

## Candidate Leaderboard

1. agg_balanced -> score 1.5814, MAE 0.9867, RMSE 1.5283, MAPE 36.5404
2. agg_wider -> score 1.6082, MAE 1.0023, RMSE 1.554, MAPE 37.2859
3. agg_regularized -> score 1.5619, MAE 0.9764, RMSE 1.5227, MAPE 35.7124
4. agg_dense -> score 1.4998, MAE 0.9332, RMSE 1.5061, MAPE 34.0714
5. agg_slim_highreg -> score 1.498, MAE 0.9462, RMSE 1.4872, MAPE 32.8743
