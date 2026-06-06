"""
core/detector.py
Runs the full per-frame detection pipeline.

Fixes vs previous version:
  1. vehicle_boxes is cleared each frame (only active tracks drawn)
  2. ByteTrack confidence thresholds tuned to reduce ID switches
  3. Stale track pruning — boxes older than MAX_STALE_FRAMES removed
  4. conf threshold added to YOLO call to reduce ghost detections
  5. imgsz set to 32-multiple to avoid YOLO padding artefacts
"""

import os
import time
import cv2
from datetime import datetime

from config import (
    DEVICE, INFERENCE_WIDTH, VEHICLE_CLASSES,
    SPEED_THRESHOLD, CAPTURE_DELAY_FRAMES,
    PLATE_SCAN_INTERVAL, SAVE_DIR, LINE_Y_FRACTION,
)
from core.tracker import SpeedTracker
from core         import ocr as ocr_module
from core         import csv_logger

# A track not seen for this many frames is removed from the display
MAX_STALE_FRAMES = 10


class Detector:

    def __init__(self, vehicle_model, plate_model, ocr_reader, video_path: str):
        self._vehicle_model = vehicle_model
        self._video_path    = video_path

        ocr_module.init(plate_model, ocr_reader)
        csv_logger.init()

        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.fps    = self._cap.get(cv2.CAP_PROP_FPS) or 30
        self.orig_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Round inference size to nearest 32 (YOLO requirement)
        if INFERENCE_WIDTH and INFERENCE_WIDTH < self.orig_w:
            self._scale = INFERENCE_WIDTH / self.orig_w
            raw_w = INFERENCE_WIDTH
            raw_h = int(self.orig_h * self._scale)
        else:
            self._scale = 1.0
            raw_w = self.orig_w
            raw_h = self.orig_h

        self._inf_w = (raw_w // 32) * 32
        self._inf_h = (raw_h // 32) * 32

        self._tracker = SpeedTracker(self.fps, self.orig_h)

        self._frame_number       = 0
        # track_id -> (x1,y1,x2,y2, last_seen_frame)
        self._vehicle_boxes      = {}
        self._capture_pending    = {}
        self._captured_ids       = set()
        self._plate_last_scanned = {}

        self._fps_t0      = time.time()
        self._fps_count   = 0
        self._display_fps = 0.0

    # ── Public ─────────────────────────────────────────────

    def run(self):
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            self._frame_number += 1
            annotated, stats = self._process(frame)
            yield annotated, stats

    def release(self):
        ocr_module.stop()
        self._cap.release()

    def get_total_frames(self):
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Private ────────────────────────────────────────────

    def _process(self, frame):
        fn            = self._frame_number
        height, width = frame.shape[:2]
        line_y        = self._tracker.line_y_px

        # ── Inference resize ──────────────────────────────
        if self._scale < 1.0:
            inf_frame = cv2.resize(frame, (self._inf_w, self._inf_h))
        else:
            inf_frame = frame

        # ── YOLO tracking ─────────────────────────────────
        # conf=0.4  → ignore weak detections that cause ghost boxes
        # iou=0.5   → merge overlapping boxes aggressively
        results = self._vehicle_model.track(
            inf_frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
            verbose=False,
            device=DEVICE,
            imgsz=self._inf_w,
            conf=0.4,
            iou=0.5,
        )

        # IDs seen THIS frame — used to prune stale boxes
        active_ids = set()
        new_detections = []

        if (
            results[0].boxes is not None
            and results[0].boxes.id is not None
            and len(results[0].boxes.id) > 0
        ):
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids   = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                ix1, iy1, ix2, iy2 = box
                x1 = int(ix1 / self._scale)
                y1 = int(iy1 / self._scale)
                x2 = int(ix2 / self._scale)
                y2 = int(iy2 / self._scale)

                active_ids.add(track_id)
                # Store box + frame it was last seen
                self._vehicle_boxes[track_id] = (x1, y1, x2, y2, fn)

                # Throttled plate scan
                if track_id not in self._captured_ids:
                    last = self._plate_last_scanned.get(track_id, -999)
                    if fn - last >= PLATE_SCAN_INTERVAL:
                        ocr_module.submit_plate_crop(
                            track_id, frame, x1, y1, x2, y2
                        )
                        self._plate_last_scanned[track_id] = fn

                # Speed update
                result = self._tracker.update(track_id, x1, y1, x2, y2, fn)
                if result:
                    new_detections.append(result)
                    plate = ocr_module.best_plate(track_id)
                    csv_logger.log_vehicle(
                        track_id,
                        result["speed_kmh"],
                        plate,
                        result["direction"],
                    )
                    if (
                        result["overspeed"]
                        and track_id not in self._capture_pending
                        and track_id not in self._captured_ids
                    ):
                        self._capture_pending[track_id] = fn

        # ── Prune stale tracks ────────────────────────────
        # Remove boxes for vehicles not seen for MAX_STALE_FRAMES
        stale = [
            tid for tid, val in self._vehicle_boxes.items()
            if fn - val[4] > MAX_STALE_FRAMES
        ]
        for tid in stale:
            del self._vehicle_boxes[tid]

        # ── Annotate & save ───────────────────────────────
        annotated = self._draw(frame, line_y, width)
        self._save_proofs(annotated, fn)

        # ── FPS ───────────────────────────────────────────
        self._fps_count += 1
        elapsed = time.time() - self._fps_t0
        if elapsed >= 1.0:
            self._display_fps = self._fps_count / elapsed
            self._fps_count   = 0
            self._fps_t0      = time.time()

        stats = {
            "fps":        round(self._display_fps, 1),
            "device":     DEVICE.upper(),
            "frame":      fn,
            "new_speeds": new_detections,
        }
        return annotated, stats

    def _draw(self, frame, line_y, width):
        out = frame.copy()

        cv2.line(out, (0, line_y), (width, line_y), (0, 0, 255), 3)

        cv2.putText(
            out, f"Threshold: {SPEED_THRESHOLD} km/h",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        cv2.putText(
            out,
            f"FPS: {self._display_fps:.1f}  |  Device: {DEVICE.upper()}",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

        # Only draw ACTIVE boxes (stale ones already pruned)
        for track_id, box_data in self._vehicle_boxes.items():
            x1, y1, x2, y2, _ = box_data

            color = (
                (0, 0, 255) if self._tracker.is_overspeeding(track_id)
                else (0, 255, 0)
            )
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"ID {track_id}"
            speed = self._tracker.get_speed(track_id)
            if speed is not None:
                label += f" | {speed:.1f} km/h"
            plate = ocr_module.best_plate(track_id)
            if plate:
                label += f" | {plate}"

            cv2.putText(
                out, label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        return out

    def _save_proofs(self, annotated_frame, fn):
        for track_id in list(self._capture_pending.keys()):
            if track_id in self._captured_ids:
                del self._capture_pending[track_id]
                continue

            if fn - self._capture_pending[track_id] < CAPTURE_DELAY_FRAMES:
                continue

            speed = self._tracker.get_speed(track_id)
            if speed is None:
                continue

            plate     = ocr_module.best_plate(track_id) or "UNKNOWN"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = (
                f"vehicle_{track_id}_{speed:.1f}kmh_"
                f"{plate}_{timestamp}.jpg"
            )
            filepath = os.path.join(SAVE_DIR, filename)

            if track_id in self._vehicle_boxes:
                bx1, by1, bx2, by2, _ = self._vehicle_boxes[track_id]
                pad  = 100
                crop = annotated_frame[
                    max(0, by1 - pad): min(annotated_frame.shape[0], by2 + pad),
                    max(0, bx1 - pad): min(annotated_frame.shape[1], bx2 + pad),
                ]
            else:
                crop = annotated_frame

            os.makedirs(SAVE_DIR, exist_ok=True)
            cv2.imwrite(filepath, crop)
            csv_logger.update_image_file(track_id, filename)
            print(f"[proof] Saved {filepath}")

            self._captured_ids.add(track_id)
            del self._capture_pending[track_id]