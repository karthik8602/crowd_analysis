import streamlit as st
import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLO

# --------------------------------------
# PAGE CONFIG
# --------------------------------------
st.set_page_config(page_title="Crowd Behavior Analysis", layout="wide")
st.title("🧠 Real-Time Crowd Behavior Analysis")

# --------------------------------------
# SIDEBAR
# --------------------------------------
st.sidebar.header("Controls")

crowd_threshold = st.sidebar.slider("👥 Crowd Alert Threshold", 1, 50, 10)
motion_threshold = st.sidebar.slider("⚡ Movement Intensity Threshold (%)", 1, 100, 30)

start_btn = st.sidebar.button("▶️ Start", key="start_button")
stop_btn = st.sidebar.button("⏹ Stop", key="stop_button")

# --------------------------------------
# STATE
# --------------------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["time", "count", "motion", "fps"])

if start_btn:
    st.session_state.running = True
    st.session_state.data = pd.DataFrame(columns=["time", "count", "motion", "fps"])  # reset
elif stop_btn:
    st.session_state.running = False

# --------------------------------------
# LOAD YOLO
# --------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --------------------------------------
# LAYOUT
# --------------------------------------
col1, col2 = st.columns([2, 1])
frame_placeholder = col1.empty()
info_placeholder = col1.empty()
chart_placeholder = col2.empty()
motion_chart_placeholder = col2.empty()
summary_placeholder = col2.empty()
alert_placeholder = st.sidebar.empty()

# --------------------------------------
# MAIN LOOP
# --------------------------------------
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Camera not detected.")
            break

        start_time = time.time()
        results = model(frame, verbose=False)
        annotated = results[0].plot()
        count = len(results[0].boxes)

        # --- Optical Flow for motion intensity ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_percent = 0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_mask = magnitude > 2.0  # ignore very small pixel movement
            motion_percent = (np.sum(motion_mask) / motion_mask.size) * 100  # % of pixels moving
        prev_gray = gray

        fps = 1 / (time.time() - start_time)

        # --- ALERT LOGIC ---
        alert_message = []
        border_color = (0, 255, 0)  # green

        if count > crowd_threshold:
            alert_message.append(f"🚨 Crowd exceeded limit ({count})")
            border_color = (0, 0, 255)
        if motion_percent > motion_threshold:
            alert_message.append(f"⚡ Large movement detected ({motion_percent:.1f}%)")
            border_color = (0, 0, 255)

        if alert_message:
            alert_placeholder.error(" | ".join(alert_message))
        else:
            alert_placeholder.success(f"✅ Normal Crowd ({count}) | Movement: {motion_percent:.1f}%")

        # Visual cue border
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], annotated.shape[0]), border_color, 15)

        # Display frame and info
        frame_placeholder.image(annotated, channels="BGR", use_container_width=True)
        info_placeholder.markdown(f"### 👥 {count} People | ⚡ Motion {motion_percent:.1f}% | 🎥 {fps:.1f} FPS")

        # Log data
        st.session_state.data.loc[len(st.session_state.data)] = {
            "time": time.time(),
            "count": count,
            "motion": motion_percent,
            "fps": fps,
        }

        # Graph 1 — Count + FPS
        if len(st.session_state.data) > 5:
            chart_placeholder.line_chart(
                st.session_state.data[["count", "fps"]].tail(50),
                use_container_width=True,
            )
            # Graph 2 — Motion %
            motion_chart_placeholder.line_chart(
                st.session_state.data[["motion"]].tail(50),
                use_container_width=True,
            )

        if not st.session_state.running:
            break

        time.sleep(0.05)

    cap.release()

# --------------------------------------
# SUMMARY
# --------------------------------------
if not st.session_state.running and not st.session_state.data.empty:
    df = st.session_state.data
    avg_fps = df["fps"].mean()
    max_count = df["count"].max()
    avg_motion = df["motion"].mean()

    summary_placeholder.markdown(f"""
    ### 📊 Session Summary
    - **Average FPS:** {avg_fps:.2f}
    - **Max Crowd Count:** {int(max_count)}
    - **Average Motion:** {avg_motion:.1f}%
    """)
    st.success("✅ Analysis complete.")
else:
    st.info("Click **Start** to begin crowd behavior analysis.")
