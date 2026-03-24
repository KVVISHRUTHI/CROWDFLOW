import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from models.yolo_model import load_model
from processing.crowd_predictor import CrowdPredictor
from processing.detection import detect_people
from processing.hybrid_tracker import HybridTracker
from processing.tracker import SimpleTracker
from processing.video_processing import load_video, read_frame
from prepare_demo_videos import ensure_demo_clips


def extract_series(
    video_path: str,
    model,
    sample_every: int,
    resize_w: int,
    resize_h: int,
) -> Tuple[List[int], List[float], List[float], List[List[float]], List[List[float]]]:
    cap = load_video(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    counts = []
    densities = []
    elapsed_values = []
    action_features = []
    attribute_features = []
    frame_idx = 0
    detection_step = 0
    previous_centers: Dict[int, Tuple[int, int]] = {}

    try:
        tracker = HybridTracker()
    except Exception:
        tracker = SimpleTracker()

    def _update_tracker_compat(boxes, frame):
        try:
            return tracker.update(boxes, frame=frame)
        except TypeError:
            return tracker.update(boxes)

    def _compute_action_attribute_features(objects, boxes, frame_shape):
        nonlocal previous_centers

        current_centers: Dict[int, Tuple[int, int]] = {}
        displacements = []
        for x1, y1, x2, y2, obj_id in objects:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            current_centers[int(obj_id)] = (cx, cy)
            prev = previous_centers.get(int(obj_id))
            if prev is not None:
                displacements.append(float(np.hypot(cx - prev[0], cy - prev[1])))

        frame_h, frame_w = frame_shape[:2]
        frame_area = max(1.0, float(frame_h * frame_w))
        frame_diag = max(1.0, float(np.hypot(frame_w, frame_h)))

        tracked_count = len(objects)
        detected_count = len(boxes)
        avg_speed_px = float(np.mean(displacements)) if displacements else 0.0
        moving_ratio = float(np.mean(np.array(displacements) > 2.0)) if displacements else 0.0
        avg_speed_norm = min(1.0, avg_speed_px / max(1.0, frame_diag * 0.08))

        track_coverage = tracked_count / max(1, detected_count)
        entry_pressure = max(0.0, (detected_count - tracked_count) / max(1, detected_count))

        box_areas = [max(0, x2 - x1) * max(0, y2 - y1) for x1, y1, x2, y2 in boxes]
        avg_box_area_norm = (float(np.mean(box_areas)) / frame_area) if box_areas else 0.0

        area_cv = 0.0
        if box_areas:
            mean_area = float(np.mean(box_areas))
            if mean_area > 1e-6:
                area_cv = float(np.std(box_areas) / mean_area)

        spread_norm = 0.0
        if current_centers:
            centers = np.array(list(current_centers.values()), dtype=np.float32)
            center_std = float(np.mean(np.std(centers, axis=0)))
            spread_norm = min(1.0, center_std / max(1.0, frame_diag * 0.35))

        occlusion_ratio = max(0.0, min(1.0, 1.0 - track_coverage))

        previous_centers = current_centers
        return [
            round(avg_speed_norm, 6),
            round(moving_ratio, 6),
            round(entry_pressure, 6),
            round(min(1.0, track_coverage), 6),
        ], [
            round(avg_box_area_norm, 6),
            round(area_cv, 6),
            round(spread_norm, 6),
            round(occlusion_ratio, 6),
        ]

    while True:
        ret, frame = read_frame(cap)
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_every != 0:
            continue

        frame = cv2.resize(frame, (resize_w, resize_h))
        detection_step += 1
        use_zoom = detection_step % 2 == 0
        boxes = detect_people(model, frame, use_zoom=use_zoom)
        objects = _update_tracker_compat(boxes, frame)

        frame_area = max(1, frame.shape[0] * frame.shape[1])
        box_area_sum = 0
        for x1, y1, x2, y2 in boxes:
            box_area_sum += max(0, x2 - x1) * max(0, y2 - y1)

        density_ratio = min(1.0, box_area_sum / frame_area)
        elapsed_ratio = frame_idx / total_frames if total_frames > 0 else 0.0
        actions, attributes = _compute_action_attribute_features(objects, boxes, frame.shape)

        tracked_count = len(objects)
        detected_count = len(boxes)
        coverage = tracked_count / max(1, detected_count)
        if detected_count == 0:
            fused_count = tracked_count
        elif coverage < 0.70:
            fused_count = max(tracked_count, detected_count)
        else:
            fused_count = max(tracked_count, int(round(0.70 * tracked_count + 0.30 * detected_count)))

        counts.append(int(fused_count))
        densities.append(float(density_ratio))
        elapsed_values.append(float(elapsed_ratio))
        action_features.append(actions)
        attribute_features.append(attributes)

    cap.release()
    return counts, densities, elapsed_values, action_features, attribute_features


def resolve_video_paths(patterns: List[str]) -> List[str]:
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    unique_paths = sorted(set(paths))
    if not unique_paths:
        raise RuntimeError("No videos matched. Check --videos pattern.")
    return unique_paths


def main():
    parser = argparse.ArgumentParser(description="Train crowd predictor from multiple videos.")
    parser.add_argument(
        "--videos",
        nargs="+",
        default=["data/crowd_videos/demo_training/*.mp4"],
        help="One or more glob patterns for training videos.",
    )
    parser.add_argument("--sample-every", type=int, default=4, help="Use every Nth frame for training data.")
    parser.add_argument("--resize-w", type=int, default=1366)
    parser.add_argument("--resize-h", type=int, default=768)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--horizon-steps", type=int, default=15)
    parser.add_argument("--model-path", default="models/crowd_predictor.joblib")
    parser.add_argument(
        "--optimize-level",
        default="standard",
        choices=["quick", "standard", "aggressive"],
        help="Candidate model search depth for better accuracy.",
    )
    parser.add_argument(
        "--no-shuffle-split",
        action="store_true",
        help="Disable shuffled train/validation split (enabled by default).",
    )
    parser.add_argument(
        "--selection-target",
        default="composite",
        choices=["composite", "incoming_accuracy", "mape"],
        help="Validation objective used for candidate model selection.",
    )
    parser.add_argument(
        "--incoming-threshold-ratio",
        type=float,
        default=0.12,
        help="Relative threshold ratio for incoming-event accuracy target.",
    )
    parser.add_argument(
        "--incoming-threshold-min",
        type=int,
        default=2,
        help="Minimum absolute threshold for incoming-event accuracy target.",
    )
    parser.add_argument(
        "--prepare-demo",
        action="store_true",
        help="Generate demo training clips from existing videos before training.",
    )
    parser.add_argument("--demo-clips", type=int, default=13, help="Number of demo clips to generate.")
    parser.add_argument(
        "--max-rows-per-series",
        type=int,
        default=0,
        help="Optional cap for training rows per video series (0 disables capping).",
    )
    parser.add_argument(
        "--artifact-tag",
        default="",
        help="Optional tag appended to output artifact filenames to avoid overwriting existing results.",
    )
    args = parser.parse_args()

    def _tagged_name(base_name: str) -> str:
        if not args.artifact_tag:
            return base_name
        root, ext = os.path.splitext(base_name)
        return f"{root}_{args.artifact_tag}{ext}"

    is_experiment = bool(args.artifact_tag)
    metrics_dir = os.path.join("models", "experiments", "candidates") if is_experiment else "models"
    report_dir = os.path.join("models", "experiments", "training") if is_experiment else os.path.join("models", "reports", "current")
    series_dir = os.path.join("models", "experiments", "training") if is_experiment else "models"

    if args.prepare_demo:
        generated = ensure_demo_clips(
            source_pattern="data/crowd_videos/*.mp4",
            output_dir="data/crowd_videos/demo_training",
            target_count=args.demo_clips,
        )
        print(f"Generated/available demo clips: {len(generated)}")
        args.videos = ["data/crowd_videos/demo_training/*.mp4"]

    video_paths = resolve_video_paths(args.videos)
    print("Training videos:")
    for p in video_paths:
        print(f"  - {p}")

    model = load_model()
    all_series = []
    per_video_report = []

    for video_path in video_paths:
        counts, densities, elapsed_values, actions, attributes = extract_series(
            video_path,
            model,
            sample_every=args.sample_every,
            resize_w=args.resize_w,
            resize_h=args.resize_h,
        )
        if len(counts) < args.window_size + args.horizon_steps:
            print(f"Skipping {video_path} (too short: {len(counts)} samples)")
            continue

        print(f"{video_path}: {len(counts)} samples")
        all_series.append(
            {
                "counts": counts,
                "densities": densities,
                "elapsed": elapsed_values,
                "actions": actions,
                "attributes": attributes,
            }
        )
        per_video_report.append(
            {
                "video_path": video_path,
                "samples": len(counts),
                "count_min": int(min(counts)),
                "count_max": int(max(counts)),
                "count_mean": round(float(sum(counts) / len(counts)), 3),
                "density_mean": round(float(sum(densities) / len(densities)), 6),
                "elapsed_start": round(float(elapsed_values[0]), 6),
                "elapsed_end": round(float(elapsed_values[-1]), 6),
            }
        )

    if not all_series:
        raise RuntimeError("No usable training series found.")

    if args.optimize_level == "quick":
        candidate_configs = [
            {
                "name": "quick_default",
                "hidden_layer_sizes": (96, 48),
                "alpha": 0.0005,
                "learning_rate_init": 0.0010,
                "max_iter": 900,
            },
            {
                "name": "quick_wider",
                "hidden_layer_sizes": (128, 64),
                "alpha": 0.0008,
                "learning_rate_init": 0.0008,
                "max_iter": 1050,
            },
        ]
    elif args.optimize_level == "aggressive":
        candidate_configs = [
            {
                "name": "agg_balanced",
                "hidden_layer_sizes": (96, 48),
                "alpha": 0.0004,
                "learning_rate_init": 0.0010,
                "max_iter": 1100,
            },
            {
                "name": "agg_wider",
                "hidden_layer_sizes": (128, 64),
                "alpha": 0.0007,
                "learning_rate_init": 0.0008,
                "max_iter": 1300,
            },
            {
                "name": "agg_regularized",
                "hidden_layer_sizes": (96, 64, 32),
                "alpha": 0.0012,
                "learning_rate_init": 0.0007,
                "max_iter": 1400,
            },
            {
                "name": "agg_dense",
                "hidden_layer_sizes": (160, 80),
                "alpha": 0.0009,
                "learning_rate_init": 0.0006,
                "max_iter": 1500,
            },
            {
                "name": "agg_slim_highreg",
                "hidden_layer_sizes": (80, 40),
                "alpha": 0.0018,
                "learning_rate_init": 0.0012,
                "max_iter": 1000,
            },
        ]
    else:
        candidate_configs = [
            {
                "name": "std_balanced",
                "hidden_layer_sizes": (96, 48),
                "alpha": 0.0005,
                "learning_rate_init": 0.0010,
                "max_iter": 1000,
            },
            {
                "name": "std_wider",
                "hidden_layer_sizes": (128, 64),
                "alpha": 0.0008,
                "learning_rate_init": 0.0008,
                "max_iter": 1200,
            },
            {
                "name": "std_regularized",
                "hidden_layer_sizes": (96, 64, 32),
                "alpha": 0.0012,
                "learning_rate_init": 0.0007,
                "max_iter": 1300,
            },
            {
                "name": "std_dense",
                "hidden_layer_sizes": (144, 72),
                "alpha": 0.0010,
                "learning_rate_init": 0.00075,
                "max_iter": 1300,
            },
        ]

    predictor = CrowdPredictor(
        model_path=args.model_path,
        window_size=args.window_size,
        horizon_steps=args.horizon_steps,
    )
    # Force current run config even if older meta exists on disk.
    predictor.window_size = int(args.window_size)
    predictor.horizon_steps = int(args.horizon_steps)
    metrics = predictor.fit_from_series_list(
        all_series,
        candidate_configs=candidate_configs,
        shuffle_split=not args.no_shuffle_split,
        random_state=42,
        selection_target=args.selection_target,
        incoming_threshold_ratio=args.incoming_threshold_ratio,
        incoming_threshold_min=args.incoming_threshold_min,
        max_rows_per_series=args.max_rows_per_series,
    )
    predictor.save()

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(series_dir, exist_ok=True)

    metrics_path = os.path.join(metrics_dir, _tagged_name("crowd_predictor_metrics.txt"))
    with open(metrics_path, "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    report_json_path = os.path.join(report_dir, _tagged_name("demo_training_report.json"))
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "training_config": {
                    "sample_every": args.sample_every,
                    "resize_w": args.resize_w,
                    "resize_h": args.resize_h,
                    "window_size": args.window_size,
                    "horizon_steps": args.horizon_steps,
                    "optimize_level": args.optimize_level,
                    "shuffle_split": not args.no_shuffle_split,
                    "selection_target": args.selection_target,
                    "incoming_threshold_ratio": args.incoming_threshold_ratio,
                    "incoming_threshold_min": args.incoming_threshold_min,
                    "max_rows_per_series": args.max_rows_per_series,
                    "candidate_models": candidate_configs,
                    "feature_version": predictor.feature_version,
                    "action_feature_size": predictor.action_feature_size,
                    "attribute_feature_size": predictor.attribute_feature_size,
                    "model_path": args.model_path,
                },
                "metrics": metrics,
                "demo_video_count": len(per_video_report),
                "videos": per_video_report,
            },
            f,
            indent=2,
        )

    series_json_path = os.path.join(series_dir, _tagged_name("demo_training_series.json"))
    with open(series_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "training_config": {
                    "sample_every": args.sample_every,
                    "resize_w": args.resize_w,
                    "resize_h": args.resize_h,
                    "window_size": args.window_size,
                    "horizon_steps": args.horizon_steps,
                    "optimize_level": args.optimize_level,
                    "shuffle_split": not args.no_shuffle_split,
                    "selection_target": args.selection_target,
                    "incoming_threshold_ratio": args.incoming_threshold_ratio,
                    "incoming_threshold_min": args.incoming_threshold_min,
                    "max_rows_per_series": args.max_rows_per_series,
                    "candidate_models": candidate_configs,
                    "feature_version": predictor.feature_version,
                    "action_feature_size": predictor.action_feature_size,
                    "attribute_feature_size": predictor.attribute_feature_size,
                    "model_path": args.model_path,
                },
                "demo_video_count": len(per_video_report),
                "series": all_series,
            },
            f,
            indent=2,
        )

    report_csv_path = os.path.join(report_dir, _tagged_name("demo_training_report.csv"))
    with open(report_csv_path, "w", encoding="utf-8") as f:
        f.write("video_path,samples,count_min,count_max,count_mean,density_mean,elapsed_start,elapsed_end\n")
        for row in per_video_report:
            f.write(
                f"{row['video_path']},{row['samples']},{row['count_min']},{row['count_max']},{row['count_mean']},{row['density_mean']},{row['elapsed_start']},{row['elapsed_end']}\n"
            )

    summary_md_path = os.path.join(report_dir, _tagged_name("demo_training_results_summary.md"))
    total_samples = sum(int(v["samples"]) for v in per_video_report)
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Demo Training Results Summary\n\n")
        f.write(f"- Demo videos used: {len(per_video_report)}\n")
        f.write(f"- Total extracted samples: {total_samples}\n")
        f.write(f"- Window size: {args.window_size}\n")
        f.write(f"- Horizon steps: {args.horizon_steps}\n")
        f.write(f"- sample_every: {args.sample_every}\n")
        f.write(f"- optimize_level: {args.optimize_level}\n")
        f.write(f"- shuffle_split: {not args.no_shuffle_split}\n\n")
        f.write(f"- selection_target: {args.selection_target}\n")
        f.write(f"- incoming_threshold_ratio: {args.incoming_threshold_ratio}\n")
        f.write(f"- incoming_threshold_min: {args.incoming_threshold_min}\n\n")
        f.write(f"- max_rows_per_series: {args.max_rows_per_series}\n\n")
        f.write("## Best Validation Metrics\n\n")
        f.write(f"- MAE: {metrics.get('mae')}\n")
        f.write(f"- RMSE: {metrics.get('rmse')}\n")
        f.write(f"- MAPE%: {metrics.get('mape_percent')}\n")
        f.write(f"- Incoming accuracy%: {metrics.get('incoming_accuracy_percent')}\n")
        f.write(f"- Best model config: {metrics.get('best_config_name')}\n")
        f.write(f"- Hidden layers: {metrics.get('best_hidden_layers')}\n")
        f.write(f"- Alpha: {metrics.get('best_alpha')}\n")
        f.write(f"- Learning rate init: {metrics.get('best_learning_rate_init')}\n")

        leaderboard = metrics.get("selection_leaderboard", [])
        if leaderboard:
            f.write("\n## Candidate Leaderboard\n\n")
            for idx, row in enumerate(leaderboard, start=1):
                f.write(
                    f"{idx}. {row.get('name')} -> score {row.get('score')}, MAE {row.get('mae')}, RMSE {row.get('rmse')}, MAPE {row.get('mape_percent')}\n"
                )

    print("Training completed.")
    print(f"Model saved to: {args.model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Demo report saved to: {report_json_path}")
    print(f"Demo report CSV saved to: {report_csv_path}")
    print(f"Series archive saved to: {series_json_path}")
    print(f"Summary saved to: {summary_md_path}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
