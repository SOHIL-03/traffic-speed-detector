"""
ui/dashboard.py
Analytics dashboard tab:
  - Live detections table
  - Speed distribution chart
  - Direction breakdown chart
  - Proof image gallery (speeders only)
"""

import os
import streamlit as st
import pandas as pd


def render(csv_path: str, save_dir: str, speed_threshold: int):
    """Render the full analytics dashboard from the current CSV."""

    if not os.path.exists(csv_path):
        st.info("No detections yet. Start the detector to see results here.")
        return

    df = pd.read_csv(csv_path)

    if df.empty:
        st.info("CSV exists but is empty — no vehicles have crossed the line yet.")
        return

    # ── Summary cards ─────────────────────────────────────
    total      = len(df)
    overspeed  = (df["overspeed"] == "YES").sum()
    unique     = df["vehicle_id"].nunique()
    avg_speed  = df["speed_kmh"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total detections", total)
    c2.metric("🚨 Overspeed",      overspeed)
    c3.metric("Unique vehicles",   unique)
    c4.metric("Avg speed (km/h)",  f"{avg_speed:.1f}")

    st.divider()

    # ── Detections table ──────────────────────────────────
    st.subheader("📋 Detections Log")

    show_df = df.copy()
    show_df["overspeed"] = show_df["overspeed"].apply(
        lambda v: "🚨 YES" if v == "YES" else "✅ NO"
    )
    show_df["direction"] = show_df["direction"].apply(
        lambda v: "↓ Incoming" if v == "down" else ("↑ Outgoing" if v == "up" else v)
    )

    filter_os = st.checkbox("Show overspeed only", value=False)
    if filter_os:
        show_df = show_df[show_df["overspeed"] == "🚨 YES"]

    st.dataframe(
        show_df[[
            "vehicle_id", "speed_kmh", "number_plate",
            "direction", "overspeed", "timestamp",
        ]],
        use_container_width=True,
        hide_index=True,
    )

    # Download button
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name="detections.csv",
        mime="text/csv",
    )

    st.divider()

    # ── Charts ────────────────────────────────────────────
    col_chart, col_dir = st.columns(2)

    with col_chart:
        st.subheader("📊 Speed Distribution")
        hist_data = df["speed_kmh"].dropna()
        if not hist_data.empty:
            import numpy as np
            bins   = list(range(0, int(hist_data.max()) + 10, 10))
            counts, edges = np.histogram(hist_data, bins=bins)
            hist_df = pd.DataFrame({
                "Speed range": [f"{int(e)}–{int(edges[i+1])}" for i, e in enumerate(edges[:-1])],
                "Count": counts,
            })
            st.bar_chart(hist_df.set_index("Speed range"))

            # threshold line note
            st.caption(f"Threshold: {speed_threshold} km/h")

    with col_dir:
        st.subheader("🔀 Direction Split")
        dir_counts = df["direction"].value_counts().rename(
            index={"up": "↑ Outgoing", "down": "↓ Incoming"}
        )
        if not dir_counts.empty:
            st.bar_chart(dir_counts)

    st.divider()

    # ── Proof image gallery ───────────────────────────────
    st.subheader("📸 Proof Images (Speeders)")

    speeders = df[
        (df["overspeed"] == "YES") & (df["image_file"].notna()) & (df["image_file"] != "")
    ]

    if speeders.empty:
        st.info("No proof images saved yet.")
        return

    cols = st.columns(3)
    for i, (_, row) in enumerate(speeders.iterrows()):
        img_path = os.path.join(save_dir, row["image_file"])
        if os.path.exists(img_path):
            with cols[i % 3]:
                st.image(
                    img_path,
                    caption=(
                        f"ID {row['vehicle_id']} | "
                        f"{row['speed_kmh']} km/h | "
                        f"{row.get('number_plate', '—')}"
                    ),
                    use_container_width=True,
                )
