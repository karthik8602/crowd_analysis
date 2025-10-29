import streamlit as st

st.set_page_config(page_title="Crowd Analysis Project", layout="centered")

st.title("🎥 AI Crowd Behavior Analysis Tool")
st.markdown("Choose which project phase you want to explore:")

st.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=100)

# --- Buttons for Navigation ---
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🧩 Phase 1: Basic Detection"):
        st.switch_page("pages/1_Phase1_BasicDetection.py")

with col2:
    if st.button("📊 Phase 3: Dashboard"):
        st.switch_page("pages/3_Phase3_Dashboard.py")

with col3:
    if st.button("⚡ Phase 4: Behavior Detection"):
        st.switch_page("pages/4_Phase4_BehaviorDetection.py")

st.divider()

if st.button("🧠 Phase 5: Smart Classification", use_container_width=True):
    st.switch_page("pages/5_Phase5_SmartClassification.py")

st.markdown("---")
st.info("Each phase runs as its own Streamlit page for clean demos and stable performance.")
