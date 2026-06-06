"""
ui/sidebar.py
Renders the Streamlit sidebar and returns the user's settings.
"""

import os
import streamlit as st


def render() -> dict:
    """
    Draw the sidebar and return a settings dict:
    {
        video_path:        str,
        speed_threshold:   int,
        car_length:        float,
        inference_width:   int,
        plate_scan_every:  int,
        capture_delay:     int,
        line_position:     float,
    }
    """
    st.sidebar.title("⚙️ Settings")

    # ── Video source ──────────────────────────────────────
    st.sidebar.header("📹 Video Source")

    source_type = st.sidebar.radio(
        "Input type",
        ["Upload file", "Local path"],
        horizontal=True,
    )

    video_path = None

    if source_type == "Upload file":
        uploaded = st.sidebar.file_uploader(
            "Choose a video", type=["mp4", "avi", "mov", "mkv"]
        )
        if uploaded:
            tmp_path = os.path.join("data", "uploaded_video.mp4")
            os.makedirs("data", exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())
            video_path = tmp_path
            st.sidebar.success(f"Uploaded: {uploaded.name}")
    else:
        video_path = st.sidebar.text_input(
            "Video file path", value="traffic.mp4"
        )

    # ── Detection settings ────────────────────────────────
    st.sidebar.header("🚦 Detection")

    speed_threshold = st.sidebar.slider(
        "Speed threshold (km/h)",
        min_value=10, max_value=200, value=35, step=5,
    )

    car_length = st.sidebar.number_input(
        "Vehicle length (meters)",
        min_value=2.0, max_value=20.0, value=4.5, step=0.5,
        help="Used to estimate speed. 4.5 m = typical car, 12 m = bus.",
    )

    line_position = st.sidebar.slider(
        "Detection line position",
        min_value=0.1, max_value=0.9, value=0.55, step=0.05,
        help="Fraction of frame height where the measurement line sits.",
    )

    # ── Performance settings ──────────────────────────────
    st.sidebar.header("⚡ Performance")

    inference_width = st.sidebar.select_slider(
        "Inference width (px)",
        options=[320, 416, 480, 640, 720, 1080],
        value=640,
        help="Smaller = faster but less accurate.",
    )

    plate_scan_every = st.sidebar.slider(
        "Scan plate every N frames",
        min_value=1, max_value=30, value=8,
        help="Higher = faster but may miss plates.",
    )

    capture_delay = st.sidebar.slider(
        "Proof image delay (frames)",
        min_value=10, max_value=90, value=40,
        help="Frames to wait before saving the evidence photo.",
    )

    # ── Info box ──────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.caption(
        "🖥️ Tracks cars, buses & trucks.\n\n"
        "🟢 Green box = normal speed\n"
        "🔴 Red box = over threshold\n\n"
        "📸 Proof images saved to `data/` folder."
    )

    return {
        "video_path":       video_path,
        "speed_threshold":  speed_threshold,
        "car_length":       car_length,
        "inference_width":  inference_width,
        "plate_scan_every": plate_scan_every,
        "capture_delay":    capture_delay,
        "line_position":    line_position,
    }
