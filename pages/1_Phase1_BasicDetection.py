import streamlit as st
import cv2
import time
import pandas as pd
from ultralytics import YOLO

# --------------------------------------
# PAGE CONFIG
# --------------------------------------
st.set_page_config(page_title="Crowd Analysis", layout="wide")
st.title("🎥 Real-Time Crowd Counting & Analytics")

# --------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------
st.sidebar.header("Controls")

# Use unique keys so Streamlit can track properly
start_btn = st.sidebar.button("▶️ Start", key="start_button")
stop_btn = st.sidebar.button("⏹ Stop", key="stop_button")

# --------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["time", "count", "fps"])

# Handle button clicks
if start_btn:
    st.session_state.running = True
    st.session_state.data = pd.DataFrame(columns=["time", "count", "fps"])  # reset data
elif stop_btn:
    st.session_state.running = False

# --------------------------------------
# LOAD YOLO MODEL (cached for performance)
# --------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --------------------------------------
# STREAMLIT LAYOUT
# --------------------------------------
col1, col2 = st.columns([2, 1])
frame_placeholder = col1.empty()
info_placeholder = col1.empty()
chart_placeholder = col2.empty()
summary_placeholder = col2.empty()

# --------------------------------------
# MAIN LOOP
# --------------------------------------
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ No camera frame detected.")
            break

        start_time = time.time()
        results = model(frame, verbose=False)
        annotated = results[0].plot()
        end_time = time.time()

        fps = 1 / (end_time - start_time)
        count = len(results[0].boxes)

        # Display live video + info
        frame_placeholder.image(annotated, channels="BGR", use_container_width=True)
        info_placeholder.markdown(f"### 👥 Crowd Count: {count}  |  ⚡ FPS: {fps:.2f}")

        # Store data
        st.session_state.data.loc[len(st.session_state.data)] = {
            "time": time.time(),
            "count": count,
            "fps": fps,
        }

        # Live chart (last 50 points)
        if len(st.session_state.data) > 2:
            chart_placeholder.line_chart(
                st.session_state.data[["count", "fps"]].tail(50),
                use_container_width=True,
            )

        # Check if Stop was pressed
        if not st.session_state.running:
            break

        # Short delay to avoid overloading Streamlit refresh
        time.sleep(0.05)

    cap.release()

# --------------------------------------
# SHOW SUMMARY AFTER STOP
# --------------------------------------
if not st.session_state.running and not st.session_state.data.empty:
    df = st.session_state.data
    avg_fps = df["fps"].mean()
    max_count = df["count"].max()
    avg_count = df["count"].mean()
    duration = df["time"].iloc[-1] - df["time"].iloc[0]

    summary_placeholder.markdown(f"""
    ### 📊 Session Summary
    - **Average FPS:** {avg_fps:.2f}  
    - **Max Crowd Count:** {int(max_count)}  
    - **Average Crowd Count:** {avg_count:.1f}  
    - **Duration:** {duration:.1f} seconds  
    """)
    st.success("✅ Analysis complete. You can click 'Start' again to restart.")
else:
    st.info("Click **Start** to begin live crowd analysis.")
