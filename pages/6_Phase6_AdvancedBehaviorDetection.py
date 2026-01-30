import streamlit as st
import cv2
import time
import numpy as np
import pandas as pd
import tempfile
from ultralytics import YOLO
from collections import deque

# --- Page Config ---
st.set_page_config(page_title="Phase 6: Advanced Behavior Detection", layout="wide")

st.title("⚙️ Phase 6: Advanced Behavior Detection")
st.markdown("This phase uses **YOLOv8 Pose Estimation** to detect people, track crowd movement intensity, and raise alerts for abnormal motion.")

# --- Sidebar Controls ---
st.sidebar.header("🎛️ Controls")
run = st.sidebar.checkbox("Start Camera", value=False)
show_keypoints = st.sidebar.checkbox("Show Pose Keypoints", value=True)
movement_threshold = st.sidebar.slider("Movement Sensitivity", 10, 200, 60, step=5)
max_points = st.sidebar.slider("Chart Data Window", 10, 100, 50)

# --- Load Model ---
@st.cache_resource
def load_model():
    return YOLO("yolov8n-pose.pt")

model = load_model()

# --- Data storage ---
movement_data = deque(maxlen=max_points)
time_data = deque(maxlen=max_points)

# --- Streamlit placeholders ---
video_placeholder = st.empty()
chart_placeholder = st.empty()
alert_placeholder = st.empty()

prev_keypoints = None
cap = None

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Unable to access the webcam.")
        st.stop()

    st.sidebar.success("✅ Camera started. Close checkbox to stop.")

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Failed to read frame.")
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Run YOLOv8 Pose detection
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        keypoints = results[0].keypoints.xy.cpu().numpy() if len(results[0].keypoints) else None

        movement_intensity = 0
        if prev_keypoints is not None and keypoints is not None:
            # Calculate average keypoint movement
            movement_intensity = np.mean(np.linalg.norm(keypoints - prev_keypoints, axis=2)) * 100

        prev_keypoints = keypoints

        # --- Store data ---
        movement_data.append(movement_intensity)
        time_data.append(time.strftime("%H:%M:%S"))

        # --- Display camera feed ---
        if show_keypoints:
            video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
        else:
            video_placeholder.image(frame, channels="BGR", use_container_width=True)

        # --- Movement Chart ---
        df = pd.DataFrame({"Time": list(time_data), "Movement": list(movement_data)})
        chart_placeholder.line_chart(df.set_index("Time"))

        # --- Intelligent Alert ---
        if len(movement_data) > 5:
            avg_recent_movement = np.mean(list(movement_data)[-5:])
            if avg_recent_movement > movement_threshold * 2:  # Only big changes trigger
                alert_placeholder.error("🚨 High movement detected! Possible unusual activity.")
            else:
                alert_placeholder.info("✅ Normal crowd movement.")

        # Delay to reduce CPU load
        time.sleep(0.05)

    cap.release()
    st.sidebar.warning("🛑 Camera stopped.")
else:
    st.info("👆 Enable 'Start Camera' from sidebar to begin live detection.")
