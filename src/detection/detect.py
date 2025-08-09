from ultralytics import YOLO
import cv2
import os
from src import config

def load_model(model_path=config.YOLO_MODEL_PATH):
    print(f"Loading YOLO model from {model_path}...")
    return YOLO(model_path)

def run_detection(video_path, output_json):
    model = load_model()
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    results_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = model(frame)[0]
        frame_results = []

        for box in detections.boxes:
            cls_id = int(box.cls.cpu().numpy())
            conf = float(box.conf.cpu().numpy())
            xyxy = box.xyxy.cpu().numpy().tolist()[0]
            frame_results.append({
                "class_id": cls_id,
                "confidence": conf,
                "bbox": xyxy
            })

        results_data.append({
            "frame": frame_id,
            "objects": frame_results
        })

        frame_id += 1

    cap.release()

    # Save to JSON
    import json
    with open(output_json, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"Detection results saved to {output_json}")
