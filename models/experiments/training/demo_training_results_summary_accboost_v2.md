# Demo Training Results Summary

- Demo videos used: 13
- Total extracted samples: 8564
- Window size: 24
- Horizon steps: 15
- sample_every: 4
- optimize_level: aggressive
- shuffle_split: True

## Best Validation Metrics

- MAE: 0.725
- RMSE: 1.083
- MAPE%: 42.955
- Best model config: agg_wider
- Hidden layers: 128-64
- Alpha: 0.0007
- Learning rate init: 0.0008

## Candidate Leaderboard

1. agg_balanced -> score 1.3918, MAE 0.7618, RMSE 1.125, MAPE 46.1323
2. agg_wider -> score 1.3165, MAE 0.7245, RMSE 1.0832, MAPE 42.9553
3. agg_regularized -> score 1.3888, MAE 0.7562, RMSE 1.0938, MAPE 46.8498
4. agg_dense -> score 1.3641, MAE 0.7443, RMSE 1.0866, MAPE 45.6788
5. agg_slim_highreg -> score 1.3732, MAE 0.7517, RMSE 1.1147, MAPE 45.4302
