# 🚗 Vehicle Speed Detection

Detects cars, buses and trucks in a traffic video, estimates their speed,
reads number plates with EasyOCR, and logs every detection to a CSV.

## Project Structure

```
speed_cam/
├── app.py              ← Streamlit UI entry point
├── main.py             ← CLI entry point (no browser needed)
├── config.py           ← All settings in one place
├── requirements.txt
├── core/
│   ├── models.py       ← Loads YOLO + EasyOCR models
│   ├── detector.py     ← Per-frame detection pipeline
│   ├── tracker.py      ← Speed calculation (both directions)
│   ├── ocr.py          ← Plate detection + OCR background thread
│   └── csv_logger.py   ← CSV read/write helpers
└── ui/
    ├── sidebar.py      ← Streamlit sidebar controls
    ├── metrics.py      ← Live metrics row
    └── dashboard.py    ← Analytics tab (table, charts, gallery)
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install PyTorch with CUDA (RTX 3060 — CUDA 12.x)

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU is available:
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### 3. Download the plate detector model

1. Open: https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/blob/main/license_plate_detector.pt
2. Click **"Download raw file"**
3. Save as `license_plate_detector.pt` in the project folder

## Run

### Streamlit app (recommended)
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

### CLI (no browser)
```bash
python main.py
python main.py --video traffic.mp4 --threshold 50
```
Press **Q** in the video window to stop.

## Output

- **CSV**: `data/thresh_<N>/detections.csv`
- **Proof images**: `data/thresh_<N>/vehicle_<id>_<speed>kmh_<plate>_<ts>.jpg`

### CSV columns

| Column | Description |
|---|---|
| vehicle_id | ByteTrack tracking ID |
| speed_kmh | Estimated speed |
| number_plate | OCR result (UNKNOWN if not read) |
| direction | `up` (outgoing) or `down` (incoming) |
| overspeed | `YES` / `NO` |
| timestamp | When speed was calculated |
| image_file | Proof image filename (speeders only) |
