import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_VIDEO_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
MODEL_DIR = os.path.join(DATA_DIR, "models")

# Model settings
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolo.pt")  # Replace with actual
CONF_THRESHOLD = 0.25

# Court dimensions (in meters)
COURT_WIDTH = 9
COURT_LENGTH = 18
