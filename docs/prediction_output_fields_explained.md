# Prediction Output Fields Explained

This file explains every prediction output field used by the system.

## 1) Red Panel Fields (On-Screen)

### Current
- Meaning: Current crowd estimate from fused tracker+detection signal.
- Source: Coverage-aware blend of active tracked persons and current detections.

### Future
- Meaning: Predicted near-future crowd count.
- Source: Trained NN predictor output.

### Delta
- Meaning: `Future - Current`.
- Interpretation:
  - Positive: crowd expected to increase
  - Zero: stable expectation
  - Negative: crowd expected to decrease

### Upcoming Crowd (YES/NO)
- Meaning: Final operational decision for incoming crowd.
- YES only if all gates pass.

### Gate (PASS/HOLD)
- PASS: decision gates satisfied.
- HOLD: at least one gate blocked the alert.

### Confidence (%)
- Meaning: Confidence score from model reliability and recent temporal stability.
- Higher value means prediction is more trustworthy.

### Incoming Probability (%)
- Meaning: Probability-like score derived from normalized alert score.
- Used for operator readability, not a certified probability.

### Trend
- Values: `STRONG_INCREASE`, `LIKELY_INCREASE`, `LIKELY_DROP`, `STABLE`.
- Meaning: Directional behavior summary from prediction delta and confidence.

### Mode
- `NN_ONLY_DEMO`: prediction running from trained NN flow.
- `NN_WARMUP_OR_UNAVAILABLE`: NN not active yet (before activation or not ready).

### State
- `ACTIVE`: prediction phase after 10% progress and NN ready.
- `WARMUP`: before full activation.

### Threshold
- Meaning: Dynamic minimum delta required to consider incoming crowd.
- Helps suppress weak/noisy spikes.

### Streak
- Meaning: Consecutive frames where raw incoming condition passed.
- Used for false-alert reduction.

### Score
- Meaning: Confidence-weighted alert strength.
- Larger score indicates stronger incoming signal.

### Gate Reason
- Main values:
  - `BEFORE_10_PERCENT`
  - `NN_NOT_READY`
  - `DELTA_BELOW_THRESHOLD`
  - `CONFIDENCE_TOO_LOW`
  - `STREAK_NOT_MET`
  - `PASS`

### Recommended Action
- Values:
  - `NORMAL`
  - `WATCH`
  - `PREPARE`
  - `INTERVENE`
- Meaning: Suggested operator action based on final signal strength.

### Started at X% video progress
- Meaning: Current elapsed progress marker.

---

## 2) CSV File Fields

File: `prediction_output_log.csv`

### frame
- Frame index where prediction row was logged.

### elapsed_percent
- Video progress percentage for this row.

### current_count
- Current crowd estimate.

### future_count
- NN predicted future crowd.

### delta
- Difference between future and current count.

### incoming_raw
- Raw pre-streak gate result (`1` or `0`).

### incoming
- Final decision after full gating (`1` or `0`).

### incoming_threshold
- Dynamic threshold used for raw gate.

### incoming_streak
- Consecutive raw-pass count.

### confidence_percent
- Confidence score for this row.

### alert_score
- Normalized confidence-weighted alert strength.

### risk_hint
- Directional hint (`STRONG_INCREASE`, etc.).

### prediction_mode
- Active prediction mode (`NN_ONLY_DEMO`, etc.).

### nn_pred
- Raw NN prediction value.

### nn_ready
- Whether NN was ready/active (`1` or `0`).

### density_ratio
- Estimated occupancy ratio from detection boxes.

### status
- Global risk status (`SAFE`, `HIGH`, `CRITICAL`, etc.).

---

## 3) JSONL File Fields

File: `prediction_output_log.jsonl`

- Contains same core information as CSV plus structured readability.
- Each line is one JSON object for one prediction step.
- Best for API-like processing and analytics scripts.

---

## 4) Alert Events File Fields

File: `prediction_alert_events.txt`

### frame
- Frame where final incoming switched from NO to YES.

### elapsed_percent
- Progress percent at alert transition.

### current_count, future_count, delta
- Core crowd movement values at alert time.

### confidence_percent
- Confidence at alert transition.

### action_recommendation
- Suggested action for this event.

### gate_reason
- Usually `PASS` for saved alert events.

---

## 5) Decision Logic Summary

Final `Upcoming Crowd = YES` only when all conditions pass:
1. Prediction active phase (after 10% progress)
2. NN ready
3. Delta >= dynamic threshold
4. Confidence >= configured minimum
5. Streak >= configured minimum

If any condition fails, panel shows `Upcoming Crowd: NO` with `Gate: HOLD` and a clear `Gate Reason`.
