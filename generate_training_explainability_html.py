import json
import os
from pathlib import Path


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_metrics_txt(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.strip().split(":", 1)
            k = k.strip()
            v = v.strip()
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def _safe(v):
    if v is None:
        return "N/A"
    return str(v)


def build_html():
    metrics = _read_metrics_txt("models/crowd_predictor_metrics.txt")
    training = _read_json("models/reports/current/demo_training_report.json") or _read_json("models/demo_training_report.json") or {}
    series = _read_json("models/demo_training_series.json") or {}
    evaluation = _read_json("models/reports/current/demo_model_evaluation.json") or _read_json("models/demo_model_evaluation.json") or {}

    config = training.get("training_config", {})
    videos = training.get("videos", [])
    eval_summary = evaluation.get("summary", {})
    eval_rows = evaluation.get("per_video", [])
    demo_video_count = int(eval_summary.get("demo_video_count") or len(videos) or len(series.get("series", [])) or 0)

    series_info_rows = []
    for i, item in enumerate(series.get("series", []), start=1):
        counts = item.get("counts", [])
        densities = item.get("densities", [])
        actions = item.get("actions", [])
        attributes = item.get("attributes", [])
        if not counts:
            continue
        series_info_rows.append(
            {
                "video_index": i,
                "count_samples": len(counts),
                "count_preview": ", ".join(str(c) for c in counts[:20]),
                "density_preview": ", ".join(f"{d:.4f}" for d in densities[:10]),
                "action_preview": ", ".join(str(v) for v in actions[:3]),
                "attribute_preview": ", ".join(str(v) for v in attributes[:3]),
            }
        )

    video_rows_html = "".join(
        f"<tr><td>{_safe(v.get('video_path'))}</td><td>{_safe(v.get('samples'))}</td><td>{_safe(v.get('count_min'))}</td><td>{_safe(v.get('count_max'))}</td><td>{_safe(v.get('count_mean'))}</td><td>{_safe(v.get('density_mean'))}</td></tr>"
        for v in videos
    )

    eval_rows_html = "".join(
        f"<tr><td>{_safe(v.get('video_index'))}</td><td>{_safe(v.get('samples'))}</td><td>{_safe(v.get('mae'))}</td><td>{_safe(v.get('rmse'))}</td><td>{_safe(v.get('mape_percent'))}</td><td>{_safe(v.get('incoming_accuracy_percent'))}</td></tr>"
        for v in eval_rows
    )

    series_rows_html = "".join(
        f"<tr><td>{_safe(v.get('video_index'))}</td><td>{_safe(v.get('count_samples'))}</td><td>{_safe(v.get('count_preview'))}</td><td>{_safe(v.get('density_preview'))}</td><td>{_safe(v.get('action_preview'))}</td><td>{_safe(v.get('attribute_preview'))}</td></tr>"
        for v in series_info_rows
    )

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Crowd Model Training Explainability</title>
  <style>
    body {{ font-family: Segoe UI, Tahoma, Arial, sans-serif; margin: 24px; background: #f7f8fb; color: #1f2430; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    .card {{ background: #fff; border: 1px solid #d9dfeb; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; }}
    .kpi {{ background: #0f172a; color: #f8fafc; border-radius: 8px; padding: 10px; }}
    .kpi .label {{ font-size: 12px; opacity: 0.8; }}
    .kpi .value {{ font-size: 22px; font-weight: 700; margin-top: 6px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border: 1px solid #d9dfeb; padding: 8px; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ background: #eef3ff; }}
    .muted {{ color: #5f6b7a; font-size: 13px; }}
    code {{ background: #eef3ff; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Crowd Model Training Explainability</h1>
  <p class=\"muted\">This page explains how the NN predictor was trained from {demo_video_count} demo videos and shows the resulting metrics.</p>

  <div class=\"card\">
    <h2>Training Process</h2>
    <ol>
      <li>Demo clips are prepared in <code>data/crowd_videos/demo_training</code>.</li>
      <li>For sampled frames, system detects persons using YOLO and extracts count and density sequences.</li>
      <li>Sliding windows are built using count + density + elapsed-progress + action/attribute features.</li>
      <li>A neural network regressor (MLP) is trained and validated.</li>
      <li>Model and reports are written to <code>models/</code>.</li>
    </ol>
  </div>

  <div class=\"card\">
    <h2>Training Configuration</h2>
    <div class=\"grid\">
      <div class=\"kpi\"><div class=\"label\">sample_every</div><div class=\"value\">{_safe(config.get('sample_every'))}</div></div>
      <div class=\"kpi\"><div class=\"label\">window_size</div><div class=\"value\">{_safe(config.get('window_size'))}</div></div>
      <div class=\"kpi\"><div class=\"label\">horizon_steps</div><div class=\"value\">{_safe(config.get('horizon_steps'))}</div></div>
      <div class=\"kpi\"><div class=\"label\">feature_version</div><div class=\"value\">{_safe(config.get('feature_version'))}</div></div>
      <div class=\"kpi\"><div class=\"label\">action_feature_size</div><div class=\"value\">{_safe(config.get('action_feature_size'))}</div></div>
      <div class=\"kpi\"><div class=\"label\">attribute_feature_size</div><div class=\"value\">{_safe(config.get('attribute_feature_size'))}</div></div>
      <div class=\"kpi\"><div class=\"label\">resize</div><div class=\"value\">{_safe(config.get('resize_w'))}x{_safe(config.get('resize_h'))}</div></div>
    </div>
  </div>

  <div class=\"card\">
    <h2>Validation KPIs (Training)</h2>
    <div class=\"grid\">
      <div class=\"kpi\"><div class=\"label\">samples</div><div class=\"value\">{_safe(metrics.get('samples'))}</div></div>
      <div class=\"kpi\"><div class=\"label\">train_samples</div><div class=\"value\">{_safe(metrics.get('train_samples'))}</div></div>
      <div class=\"kpi\"><div class=\"label\">val_samples</div><div class=\"value\">{_safe(metrics.get('val_samples'))}</div></div>
      <div class=\"kpi\"><div class=\"label\">MAE</div><div class=\"value\">{_safe(metrics.get('mae'))}</div></div>
      <div class=\"kpi\"><div class=\"label\">MAPE %</div><div class=\"value\">{_safe(metrics.get('mape_percent'))}</div></div>
    </div>
  </div>

  <div class=\"card\">
    <h2>Per-Video Training Summary ({demo_video_count} Demo Videos)</h2>
    <table>
      <thead><tr><th>Video Path</th><th>Samples</th><th>Count Min</th><th>Count Max</th><th>Count Mean</th><th>Density Mean</th></tr></thead>
      <tbody>{video_rows_html}</tbody>
    </table>
  </div>

  <div class=\"card\">
    <h2>Post-Training Evaluation Summary</h2>
    <p class="muted">Average metrics across the demo videos from <code>models/reports/current/demo_model_evaluation.json</code>.</p>
    <div class=\"grid\">
      <div class=\"kpi\"><div class=\"label\">videos_evaluated</div><div class=\"value\">{_safe(eval_summary.get('videos_evaluated'))}</div></div>
      <div class=\"kpi\"><div class=\"label\">avg_mae</div><div class=\"value\">{_safe(eval_summary.get('avg_mae'))}</div></div>
      <div class=\"kpi\"><div class=\"label\">avg_rmse</div><div class=\"value\">{_safe(eval_summary.get('avg_rmse'))}</div></div>
      <div class=\"kpi\"><div class=\"label\">avg_mape_percent</div><div class=\"value\">{_safe(eval_summary.get('avg_mape_percent'))}</div></div>
      <div class=\"kpi\"><div class=\"label\">avg_incoming_accuracy_percent</div><div class=\"value\">{_safe(eval_summary.get('avg_incoming_accuracy_percent'))}</div></div>
    </div>
  </div>

  <div class=\"card\">
    <h2>Per-Video Evaluation Rows</h2>
    <table>
      <thead><tr><th>Video Index</th><th>Samples</th><th>MAE</th><th>RMSE</th><th>MAPE %</th><th>Incoming Accuracy %</th></tr></thead>
      <tbody>{eval_rows_html}</tbody>
    </table>
  </div>

  <div class=\"card\">
    <h2>Sampled Series Preview</h2>
    <p class=\"muted\">Preview of extracted training series values (counts, densities, actions, and attributes) from demo videos.</p>
    <table>
      <thead><tr><th>Video Index</th><th>Count Samples</th><th>Count Preview (first 20)</th><th>Density Preview (first 10)</th><th>Action Preview (first 3)</th><th>Attribute Preview (first 3)</th></tr></thead>
      <tbody>{series_rows_html}</tbody>
    </table>
  </div>
</body>
</html>
"""

    out = Path("docs") / "training_explainability.html"
    out.write_text(html, encoding="utf-8")
    print(f"Generated: {out}")


if __name__ == "__main__":
    build_html()
