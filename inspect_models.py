import json
import os

import joblib

from models.yolo_model import load_model


def read_meta(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def inspect_neural():
    model_path = "models/crowd_predictor.joblib"
    meta_path = "models/crowd_predictor_meta.json"
    metrics_path = "models/crowd_predictor_metrics.txt"

    print("=== Neural Model (Crowd Predictor) ===")
    if not os.path.exists(model_path):
        print(f"Missing: {model_path}")
        return

    pipeline = joblib.load(model_path)
    meta = read_meta(meta_path)

    print(f"Model file: {model_path}")
    print(f"Pipeline type: {type(pipeline).__name__}")
    print(f"Features expected: {getattr(pipeline, 'n_features_in_', 'N/A')}")
    print(f"Window size: {meta.get('window_size', 'N/A')}")
    print(f"Horizon steps: {meta.get('horizon_steps', 'N/A')}")
    print(f"Feature version: {meta.get('feature_version', 'N/A')}")
    print(f"Action feature size: {meta.get('action_feature_size', 'N/A')}")
    print(f"Attribute feature size: {meta.get('attribute_feature_size', 'N/A')}")

    if os.path.exists(metrics_path):
        print(f"Metrics file: {metrics_path}")
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                print("  " + line.strip())


def inspect_yolo():
    print("\n=== YOLO Model ===")
    model = load_model()
    try:
        model_name = getattr(model.model, "yaml", {}).get("yaml_file", "N/A")
    except Exception:
        model_name = "N/A"

    print("Model loader: models/yolo_model.py")
    print("Weights source: models/yolov8n.pt")
    print(f"Ultralytics model class: {type(model).__name__}")
    print(f"Backbone config source: {model_name}")


if __name__ == "__main__":
    inspect_neural()
    inspect_yolo()
