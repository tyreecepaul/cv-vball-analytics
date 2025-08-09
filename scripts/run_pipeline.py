import os
from src import config
from src.detection.detect import run_detection
from src.tracking.track import track_objects

if __name__ == "__main__":
    video_path = os.path.join(config.RAW_VIDEO_DIR, "match.mp4")
    detection_output = os.path.join(config.OUTPUT_DIR, "detections.json")
    tracking_output = os.path.join(config.OUTPUT_DIR, "tracked_data.json")

    run_detection(video_path, detection_output)
    track_objects(detection_output, tracking_output)
