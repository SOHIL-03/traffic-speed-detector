"""
ui/metrics.py
Renders the live metrics row at the top of the main panel.
"""

import streamlit as st


def render(
    fps: float,
    device: str,
    frame: int,
    total_frames: int,
    vehicle_count: int,
    overspeed_count: int,
):
    """
    Display a row of metric cards.
    Call inside st.empty() or directly — updates in place.
    """
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("🎞️ FPS",         f"{fps:.1f}")
    c2.metric("🖥️ Device",       device)
    c3.metric(
        "📹 Progress",
        f"{frame}/{total_frames}" if total_frames else str(frame),
    )
    c4.metric("🚗 Vehicles seen", vehicle_count)
    c5.metric("🚨 Overspeed",     overspeed_count)
