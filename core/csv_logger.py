"""
core/csv_logger.py
Handles writing detection results to CSV.
All vehicles are logged when speed is first calculated.
Speeders additionally get an image_file path filled in later.
"""

import csv
import os
from datetime import datetime

import pandas as pd

from config import CSV_PATH, SAVE_DIR, SPEED_THRESHOLD


HEADERS = [
    "vehicle_id", "speed_kmh", "number_plate",
    "direction", "overspeed", "timestamp", "image_file"
]

_logged_ids: set = set()


def init():
    """Create/overwrite the CSV with headers. Call once per run."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    _logged_ids.clear()
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(HEADERS)


def log_vehicle(track_id: int, speed_kmh: float, plate: str, direction: str):
    """
    Append one row for this vehicle (called as soon as speed is known).
    Skips silently if already logged.
    """
    if track_id in _logged_ids:
        return
    _logged_ids.add(track_id)

    overspeed = "YES" if speed_kmh >= SPEED_THRESHOLD else "NO"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            track_id,
            f"{speed_kmh:.1f}",
            plate or "UNKNOWN",
            direction,
            overspeed,
            timestamp,
            "",   # image_file — filled in by update_image_file()
        ])


def update_image_file(track_id: int, filename: str):
    """
    After a proof image is saved for a speeder, fill in its filename
    in the existing CSV row for that vehicle.
    """
    try:
        df = pd.read_csv(CSV_PATH)
        mask = df["vehicle_id"] == track_id
        if mask.any():
            df.loc[mask, "image_file"] = filename
            df.to_csv(CSV_PATH, index=False)
    except Exception as e:
        print(f"[csv_logger] Warning: could not update image_file — {e}")


def read_all() -> pd.DataFrame:
    """Return the full detections CSV as a DataFrame."""
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame(columns=HEADERS)
    return pd.read_csv(CSV_PATH)


def reset():
    _logged_ids.clear()
