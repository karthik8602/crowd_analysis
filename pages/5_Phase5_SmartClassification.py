import streamlit as st
import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crowd Movement Analysis", layout="wide")

st.title("Phase 5: Advanced Movement & Alert System")

# Sidebar
st.sidebar.header("Controls")
threshold = st.sidebar.slider("Movement Sensitivity", 10000, 200000, 80000, step=5000)
st.sidebar.info("Higher value = less sensitive (detects only large movements).")

# Initialize session states
if "cap" not in st.session_state:
    st.session_state.cap = None
if "movement_values" not in st.session_state:
    st.session_state.movement_values = deque(maxlen=50)
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# Start and Stop buttons
col1, col2 = st.columns(2)
if col1.button("Start Detection"):
    st.session_state.cap = cv2.VideoCapture(0)
    st.session_state.is_running = True

if col2.button("Stop Detection"):
    if st.session_state.cap:
        st.session_state.cap.release()
    st.session_state.is_running = False
    st.success("Detection stopped.")

# Frame display
frame_placeholder = st.empty()
chart_placeholder = st.empty()

prev_frame = None

while st.session_state.is_running and st.session_state.cap.isOpened():
    ret, frame = st.session_state.cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_frame is None:
        prev_frame = gray
        continue

    diff = cv2.absdiff(prev_frame, gray)
    _, thresh_img = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    movement = np.sum(thresh_img)

    st.session_state.movement_values.append(movement)

    color = (0, 0, 255) if movement > threshold else (0, 255, 0)
    text = "ALERT: Excessive Movement" if movement > threshold else "Normal Movement"

    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.rectangle(frame, (20, 20), (600, 80), color, 2)

    frame_placeholder.image(frame, channels="BGR")

    # Plot movement graph
    fig, ax = plt.subplots()
    ax.plot(st.session_state.movement_values, label="Movement Intensity", color='blue')
    ax.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    ax.legend()
    ax.set_title("Movement Over Time")
    ax.set_xlabel("Frame Count")
    ax.set_ylabel("Movement Value")
    chart_placeholder.pyplot(fig)

    prev_frame = gray

# Back button
st.markdown("---")
if st.button("Back to Home"):
    st.switch_page("main.py")
