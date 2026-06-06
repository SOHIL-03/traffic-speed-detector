"""
config.py
All tunable settings for the Vehicle Speed Detection system.
Change values here; every other file imports from here.
"""

import os
import torch

# ── Models ────────────────────────────────────────────────
MODEL_PATH       = "yolov8n.pt"
PLATE_MODEL_PATH = "license_plate_detector.pt"

# ── Detection ─────────────────────────────────────────────
CAR_LENGTH_METERS    = 4.5        # used to estimate speed
SPEED_THRESHOLD      = 35         # km/h — overspeed alert level
CAPTURE_DELAY_FRAMES = 40         # frames to wait before saving proof image
PLATE_SCAN_INTERVAL  = 8          # scan each vehicle every N frames
INFERENCE_WIDTH      = 640        # resize frame for YOLO (None = original)

# YOLO class IDs to track
VEHICLE_CLASSES = [2, 5, 7]       # 2=car  5=bus  7=truck

# Detection line position (fraction of frame height)
LINE_Y_FRACTION = 0.55

# ── Output ────────────────────────────────────────────────
DATA_DIR = "data"
SAVE_DIR = os.path.join(DATA_DIR, f"thresh_{SPEED_THRESHOLD}")
CSV_PATH = os.path.join(SAVE_DIR, "detections.csv")

# ── Device ────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
USE_GPU_OCR = DEVICE == "cuda"
