"""
app.py  —  Streamlit entry point
Run with:  streamlit run app.py

Tabs:
  1. 🎥 Live Detection  — video feed + real-time metrics
  2. 📊 Dashboard       — analytics, CSV table, proof gallery
"""

import time
import cv2
import streamlit as st
import numpy as np

# ── Page config (must be first Streamlit call) ────────────
st.set_page_config(
    page_title="Vehicle Speed Detection",
    page_icon="🚗",
    layout="wide",
)

import config                        # noqa: E402 — after set_page_config
from ui      import sidebar          # noqa: E402
from ui      import metrics as met   # noqa: E402
from ui      import dashboard        # noqa: E402
from core.models import load_models  # noqa: E402
from core.detector import Detector   # noqa: E402


# ── Session-state defaults ────────────────────────────────
if "running"         not in st.session_state:
    st.session_state.running         = False
if "vehicle_count"   not in st.session_state:
    st.session_state.vehicle_count   = 0
if "overspeed_count" not in st.session_state:
    st.session_state.overspeed_count = 0
if "models_loaded"   not in st.session_state:
    st.session_state.models_loaded   = False
if "models"          not in st.session_state:
    st.session_state.models          = None


# ── Sidebar ───────────────────────────────────────────────
settings = sidebar.render()

# Apply sidebar settings to config so Detector picks them up
config.SPEED_THRESHOLD      = settings["speed_threshold"]
config.CAR_LENGTH_METERS    = settings["car_length"]
config.INFERENCE_WIDTH      = settings["inference_width"]
config.PLATE_SCAN_INTERVAL  = settings["plate_scan_every"]
config.CAPTURE_DELAY_FRAMES = settings["capture_delay"]
config.LINE_Y_FRACTION      = settings["line_position"]
config.SAVE_DIR             = f"data/thresh_{settings['speed_threshold']}"
config.CSV_PATH             = f"{config.SAVE_DIR}/detections.csv"


# ── Header ────────────────────────────────────────────────
st.title("🚗 Vehicle Speed Detection")
st.caption(
    "Detects cars · buses · trucks | "
    "Reads number plates | "
    "Logs every vehicle to CSV"
)


# ── Tabs ──────────────────────────────────────────────────
tab_live, tab_dash = st.tabs(["🎥 Live Detection", "📊 Dashboard"])


# ══════════════════════════════════════════════════════════
# TAB 1 — LIVE DETECTION
# ══════════════════════════════════════════════════════════
with tab_live:

    # ── GPU / CPU notice ──────────────────────────────────
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.success(f"✅ GPU detected: **{gpu_name}** — running on CUDA")
    else:
        st.warning(
            "⚠️ No CUDA GPU found — running on CPU (slow). "
            "Reinstall PyTorch with CUDA:\n\n"
            "```\n"
            "pip uninstall torch torchvision torchaudio -y\n"
            "pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu121\n"
            "```"
        )

    # ── Load models ───────────────────────────────────────
    if not st.session_state.models_loaded:
        with st.spinner("Loading ML models (first run may take a minute)…"):
            try:
                st.session_state.models       = load_models()
                st.session_state.models_loaded = True
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()

    vehicle_model, plate_model, ocr_reader = st.session_state.models

    # ── Controls ──────────────────────────────────────────
    col_start, col_stop, col_gap = st.columns([1, 1, 6])

    with col_start:
        start_clicked = st.button(
            "▶️ Start", type="primary",
            disabled=st.session_state.running or not settings["video_path"],
        )
    with col_stop:
        stop_clicked = st.button(
            "⏹ Stop",
            disabled=not st.session_state.running,
        )

    if not settings["video_path"]:
        st.info("👈 Upload a video or enter a path in the sidebar to begin.")

    if start_clicked:
        st.session_state.running         = True
        st.session_state.vehicle_count   = 0
        st.session_state.overspeed_count = 0

    if stop_clicked:
        st.session_state.running = False

    # ── Live feed ─────────────────────────────────────────
    metrics_placeholder = st.empty()
    video_placeholder   = st.empty()
    alert_placeholder   = st.empty()

    if st.session_state.running and settings["video_path"]:

        det = Detector(
            vehicle_model,
            plate_model,
            ocr_reader,
            settings["video_path"],
        )
        total_frames = det.get_total_frames()

        try:
            for annotated_frame, stats in det.run():

                if not st.session_state.running:
                    break

                # Count unique vehicles from new speed results
                for r in stats["new_speeds"]:
                    st.session_state.vehicle_count += 1
                    if r["overspeed"]:
                        st.session_state.overspeed_count += 1

                        # Flash alert
                        alert_placeholder.error(
                            f"🚨 **OVERSPEED** — Vehicle {r['track_id']} | "
                            f"{r['speed_kmh']:.1f} km/h | "
                            f"Direction: {'↓ Incoming' if r['direction'] == 'down' else '↑ Outgoing'}"
                        )

                # Metrics row
                with metrics_placeholder.container():
                    met.render(
                        fps            = stats["fps"],
                        device         = stats["device"],
                        frame          = stats["frame"],
                        total_frames   = total_frames,
                        vehicle_count  = st.session_state.vehicle_count,
                        overspeed_count= st.session_state.overspeed_count,
                    )

                # Video frame (BGR → RGB for Streamlit)
                rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(
                    rgb, channels="RGB", use_container_width=True
                )

        finally:
            det.release()
            st.session_state.running = False
            st.success("✅ Detection complete. Check the **📊 Dashboard** tab for results.")

    elif not st.session_state.running:
        # Show placeholder when stopped
        video_placeholder.image(
            np.zeros((360, 640, 3), dtype=np.uint8),
            caption="Feed will appear here when detection starts.",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ══════════════════════════════════════════════════════════
with tab_dash:
    st.subheader("📊 Detection Analytics")

    col_refresh, _ = st.columns([1, 7])
    with col_refresh:
        if st.button("🔄 Refresh"):
            st.rerun()

    dashboard.render(
        csv_path        = config.CSV_PATH,
        save_dir        = config.SAVE_DIR,
        speed_threshold = settings["speed_threshold"],
    )
