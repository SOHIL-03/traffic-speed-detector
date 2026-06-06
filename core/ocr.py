"""
core/ocr.py
Plate region detection (YOLO) + text reading (EasyOCR).
Runs OCR in a background thread so it never blocks the main loop.
"""

import re
import threading
import queue
import cv2

from config import DEVICE


# ── Shared state (populated by init()) ────────────────────
_plate_model = None
_ocr_reader  = None

ocr_queue   = queue.Queue(maxsize=30)
ocr_results = {}          # track_id -> {plate_text: count}
ocr_lock    = threading.Lock()
_ocr_thread = None


def init(plate_model, ocr_reader):
    """Call once after models are loaded to wire everything up."""
    global _plate_model, _ocr_reader, _ocr_thread
    _plate_model = plate_model
    _ocr_reader  = ocr_reader
    _ocr_thread  = threading.Thread(
        target=_ocr_worker, daemon=True, name="OCR-Worker"
    )
    _ocr_thread.start()


def stop():
    """Send sentinel to cleanly stop the OCR background thread."""
    ocr_queue.put(None)
    if _ocr_thread:
        _ocr_thread.join()


# ── Internal ──────────────────────────────────────────────

def _ocr_worker():
    while True:
        item = ocr_queue.get()
        if item is None:
            break
        track_id, crop = item
        text = _read_plate_from_crop(crop)
        if text:
            with ocr_lock:
                counts = ocr_results.setdefault(track_id, {})
                counts[text] = counts.get(text, 0) + 1
        ocr_queue.task_done()


def _clean_plate_text(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def _read_plate_from_crop(crop) -> str:
    if crop is None or crop.size == 0:
        return ""

    h, w = crop.shape[:2]
    if w < 100:
        scale = 100 / w
        crop = cv2.resize(
            crop,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)

    best, best_conf = "", 0.0
    for (_, text, conf) in _ocr_reader.readtext(gray):
        cleaned = _clean_plate_text(text)
        if len(cleaned) >= 4 and conf > best_conf:
            best, best_conf = cleaned, conf

    return best


# ── Public API ────────────────────────────────────────────

def submit_plate_crop(track_id: int, frame, x1, y1, x2, y2):
    """
    Detect the plate region inside the vehicle crop and queue it for OCR.
    Non-blocking — drops the crop if the queue is full.
    """
    if _plate_model is None:
        return

    vehicle_crop = frame[y1:y2, x1:x2]
    if vehicle_crop.size == 0:
        return

    results = _plate_model(vehicle_crop, verbose=False, device=DEVICE)

    if results[0].boxes is None or len(results[0].boxes) == 0:
        return

    confs  = results[0].boxes.conf.cpu().numpy()
    best_i = int(confs.argmax())
    px1, py1, px2, py2 = map(
        int, results[0].boxes.xyxy[best_i].cpu().numpy()
    )

    plate_crop = vehicle_crop[py1:py2, px1:px2]
    if plate_crop.size == 0:
        return

    try:
        ocr_queue.put_nowait((track_id, plate_crop))
    except queue.Full:
        pass


def best_plate(track_id: int) -> str:
    """Return the most-voted plate text seen for this vehicle."""
    with ocr_lock:
        counts = ocr_results.get(track_id, {})
    return max(counts, key=counts.get) if counts else ""


def reset():
    """Clear all OCR results (call between video runs)."""
    with ocr_lock:
        ocr_results.clear()
