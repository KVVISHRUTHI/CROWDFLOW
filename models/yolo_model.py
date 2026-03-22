from ultralytics import YOLO

def load_model():
    model = YOLO("models/yolov8n.pt")   # or yolov8n.pt
    return model