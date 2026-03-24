import json
import os
import argparse
from typing import Dict, List, Optional, Sequence

import numpy as np

from processing.crowd_predictor import CrowdPredictor


def _mape(y_true: List[int], y_pred: List[int]) -> float:
    y_t = np.array(y_true, dtype=np.float32)
    y_p = np.array(y_pred, dtype=np.float32)
    return float(np.mean(np.abs((y_t - y_p) / np.maximum(y_t, 1.0))) * 100.0)


def _rmse(y_true: List[int], y_pred: List[int]) -> float:
    y_t = np.array(y_true, dtype=np.float32)
    y_p = np.array(y_pred, dtype=np.float32)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def evaluate_series(
    predictor: CrowdPredictor,
    counts: List[int],
    densities: List[float],
    elapsed: List[float],
    actions: Optional[Sequence[Sequence[float]]] = None,
    attributes: Optional[Sequence[Sequence[float]]] = None,
) -> Dict:
    window = predictor.window_size
    horizon = predictor.horizon_steps

    y_true = []
    y_pred = []
    incoming_true = []
    incoming_pred = []

    for i in range(window, len(counts) - horizon):
        if elapsed[i] <= 0.10:
            continue

        hist_counts = counts[:i]
        hist_densities = densities[:i]
        current = counts[i]
        actual_future = counts[i + horizon]

        action_features = actions[i] if actions is not None and i < len(actions) else None
        attribute_features = attributes[i] if attributes is not None and i < len(attributes) else None

        pred_future = predictor.predict(
            hist_counts,
            hist_densities,
            elapsed[i],
            action_features=action_features,
            attribute_features=attribute_features,
        )

        delta_true = actual_future - current
        delta_pred = pred_future - current

        threshold = max(2, int(round(current * 0.12)))
        incoming_true.append(1 if delta_true >= threshold else 0)
        incoming_pred.append(1 if delta_pred >= threshold else 0)

        y_true.append(actual_future)
        y_pred.append(pred_future)

    if not y_true:
        return {
            "samples": 0,
            "mae": None,
            "rmse": None,
            "mape_percent": None,
            "incoming_accuracy_percent": None,
        }

    mae = float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))
    rmse = _rmse(y_true, y_pred)
    mape = _mape(y_true, y_pred)

    acc = float(
        np.mean(np.array(incoming_true, dtype=np.int32) == np.array(incoming_pred, dtype=np.int32)) * 100.0
    )

    return {
        "samples": int(len(y_true)),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape_percent": round(mape, 3),
        "incoming_accuracy_percent": round(acc, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained crowd predictor on archived demo series.")
    parser.add_argument("--model-path", default="models/crowd_predictor.joblib")
    parser.add_argument("--series-path", default="models/demo_training_series.json")
    parser.add_argument(
        "--output-prefix",
        default="models/reports/current/demo_model_evaluation",
        help="Output prefix without extension, for example models/demo_model_evaluation_candidate.",
    )
    args = parser.parse_args()

    model_path = args.model_path
    series_path = args.series_path

    if not os.path.exists(model_path):
        raise RuntimeError(f"Missing trained model: {model_path}")
    if not os.path.exists(series_path):
        raise RuntimeError(f"Missing series archive: {series_path}. Run training first.")

    predictor = CrowdPredictor(model_path=model_path)
    if not predictor.is_trained():
        raise RuntimeError("Model file exists but failed to load predictor.")

    with open(series_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    series_list = data.get("series", [])
    if not series_list:
        raise RuntimeError("No demo series in archive.")

    per_video = []
    for idx, s in enumerate(series_list, start=1):
        metrics = evaluate_series(
            predictor,
            counts=[int(v) for v in s["counts"]],
            densities=[float(v) for v in s["densities"]],
            elapsed=[float(v) for v in s["elapsed"]],
            actions=s.get("actions"),
            attributes=s.get("attributes"),
        )
        metrics["video_index"] = idx
        per_video.append(metrics)

    valid = [m for m in per_video if m["samples"] > 0]

    def avg(key: str):
        vals = [float(m[key]) for m in valid if m[key] is not None]
        return round(float(sum(vals) / len(vals)), 3) if vals else None

    summary = {
        "model_path": model_path,
        "window_size": predictor.window_size,
        "horizon_steps": predictor.horizon_steps,
        "demo_video_count": len(series_list),
        "videos_evaluated": len(valid),
        "avg_mae": avg("mae"),
        "avg_rmse": avg("rmse"),
        "avg_mape_percent": avg("mape_percent"),
        "avg_incoming_accuracy_percent": avg("incoming_accuracy_percent"),
    }

    out_json = f"{args.output_prefix}.json"
    out_csv = f"{args.output_prefix}.csv"
    out_txt = f"{args.output_prefix}.txt"
    out_md = f"{args.output_prefix}_summary.md"
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_video": per_video}, f, indent=2)

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("video_index,samples,mae,rmse,mape_percent,incoming_accuracy_percent\n")
        for row in per_video:
            f.write(
                f"{row['video_index']},{row['samples']},{row['mae']},{row['rmse']},{row['mape_percent']},{row['incoming_accuracy_percent']}\n"
            )

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("DEMO MODEL EVALUATION\n")
        f.write("=====================\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Demo Model Evaluation Summary\n\n")
        f.write(f"- Demo videos in archive: {summary.get('demo_video_count')}\n")
        f.write(f"- Videos evaluated: {summary.get('videos_evaluated')}\n")
        f.write(f"- Average MAE: {summary.get('avg_mae')}\n")
        f.write(f"- Average RMSE: {summary.get('avg_rmse')}\n")
        f.write(f"- Average MAPE%: {summary.get('avg_mape_percent')}\n")
        f.write(f"- Avg incoming accuracy%: {summary.get('avg_incoming_accuracy_percent')}\n\n")
        f.write("## Per Video\n\n")
        for row in per_video:
            f.write(
                f"- Video {row['video_index']}: samples={row['samples']}, mae={row['mae']}, rmse={row['rmse']}, "
                f"mape={row['mape_percent']}, incoming_acc={row['incoming_accuracy_percent']}\n"
            )

    print("Evaluation completed.")
    print(f"Summary: {summary}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_txt}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
