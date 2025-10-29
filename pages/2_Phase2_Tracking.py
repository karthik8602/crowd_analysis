import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time

st.set_page_config(page_title="Crowd Analysis", layout="wide")

# Sidebar
st.sidebar.header("Controls")
start_btn = st.sidebar.button("Start", key="start_button")
stop_btn = st.sidebar.button("Stop", key="stop_button")

# Model
model = YOLO("yolov8n.pt")

# Video stream placeholder
frame_placeholder = st.empty()
info_placeholder = st.empty()

# Control variables (store in session_state to persist between reruns)
if "running" not in st.session_state:
    st.session_state.running = False

if start_btn:
    st.session_state.running = True
elif stop_btn:
    st.session_state.running = False

# Open camera only when running
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("No camera frame detected.")
            break

        # Run YOLO
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Display frame and analytics
        frame_placeholder.image(annotated_frame, channels="BGR")
        info_placeholder.markdown(f"### Crowd Count: {len(results[0].boxes)}  |  FPS: {fps:.2f}")

        # Check for stop (on next rerun)
        if not st.session_state.running:
            break

    cap.release()
    st.session_state.running = False
else:
    st.info("Click 'Start' to begin live crowd analysis.")
