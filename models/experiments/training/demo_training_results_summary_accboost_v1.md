# Demo Training Results Summary

- Demo videos used: 13
- Total extracted samples: 11420
- Window size: 24
- Horizon steps: 15
- sample_every: 3
- optimize_level: aggressive
- shuffle_split: True

## Best Validation Metrics

- MAE: 0.737
- RMSE: 1.146
- MAPE%: 41.571
- Best model config: agg_regularized
- Hidden layers: 96-64-32
- Alpha: 0.0012
- Learning rate init: 0.0007

## Candidate Leaderboard

1. agg_balanced -> score 1.3645, MAE 0.7546, RMSE 1.1358, MAPE 43.9538
2. agg_wider -> score 1.3667, MAE 0.7536, RMSE 1.1569, MAPE 43.9526
3. agg_regularized -> score 1.3245, MAE 0.7369, RMSE 1.1458, MAPE 41.5714
4. agg_dense -> score 1.3491, MAE 0.7448, RMSE 1.1408, MAPE 43.3238
5. agg_slim_highreg -> score 1.3698, MAE 0.7564, RMSE 1.1562, MAPE 43.9999
