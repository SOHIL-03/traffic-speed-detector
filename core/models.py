"""
core/models.py
Loads all ML models once and exposes them as module-level singletons.
Import this module; it handles first-time download and GPU placement.
"""

import os
import sys
from ultralytics import YOLO
import easyocr

from config import (
    MODEL_PATH, PLATE_MODEL_PATH,
    DEVICE, USE_GPU_OCR
)


def _check_plate_model():
    if not os.path.exists(PLATE_MODEL_PATH):
        msg = (
            "\n" + "=" * 60 + "\n"
            "  ERROR: Plate detector model not found.\n\n"
            "  Download it manually (one time only):\n"
            "  1. Open this URL in your browser:\n"
            "     https://github.com/Muhammad-Zeerak-Khan/\n"
            "     Automatic-License-Plate-Recognition-using-YOLOv8/\n"
            "     blob/main/license_plate_detector.pt\n"
            "  2. Click 'Download raw file' (top-right button)\n"
            f"  3. Save it to: {os.path.abspath(PLATE_MODEL_PATH)}\n"
            + "=" * 60
        )
        raise FileNotFoundError(msg)


def load_models():
    """
    Load and return (vehicle_model, plate_model, ocr_reader).
    Safe to call multiple times — models are cached after first load.
    """
    print(f"[models] Device: {DEVICE.upper()}")

    if DEVICE == "cpu":
        print(
            "[models] WARNING: CUDA not available.\n"
            "  Reinstall PyTorch with CUDA:\n"
            "  pip uninstall torch torchvision torchaudio -y\n"
            "  pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu121"
        )

    print("[models] Loading vehicle tracker (YOLOv8n)...")
    vehicle_model = YOLO(MODEL_PATH)
    vehicle_model.to(DEVICE)

    print("[models] Loading plate detector...")
    _check_plate_model()
    plate_model = YOLO(PLATE_MODEL_PATH)
    plate_model.to(DEVICE)

    print(f"[models] Loading EasyOCR (gpu={USE_GPU_OCR})...")
    ocr_reader = easyocr.Reader(["en"], gpu=USE_GPU_OCR)

    print("[models] All models ready.")
    return vehicle_model, plate_model, ocr_reader
