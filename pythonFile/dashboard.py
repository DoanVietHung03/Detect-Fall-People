# dashboard.py
import streamlit as st
import requests
import os
from streamlit_autorefresh import st_autorefresh

API_URL = "http://localhost:8000"

st.set_page_config(layout="wide", page_title="AI Surveillance Center", page_icon="📹")

# CSS Tùy chỉnh cho đẹp
st.markdown("""
    <style>
        .stImage img { border-radius: 8px; border: 2px solid #ddd; }
        div[data-testid="stMetricValue"] { font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

st.title("📹 Smart Surveillance Center (Client-Server)")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Camera Configuration")
    
    # Quét file video
    video_folder = "samples"
    if not os.path.exists(video_folder): os.makedirs(video_folder)
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mkv'))]
    selected_video = st.selectbox("Video Source", video_files)

    st.divider()
    st.header("🎛️ Parameters Config")
    
    # State giữ giá trị slider
    if 'conf' not in st.session_state: st.session_state.conf = 0.7
    if 'lstm' not in st.session_state: st.session_state.lstm = 0.7

    def on_change_settings():
        """Gửi setting lên server ngay khi kéo slider"""
        try:
            payload = {"conf": st.session_state.conf, "lstm": st.session_state.lstm}
            requests.post(f"{API_URL}/update_settings", json=payload, timeout=1)
            st.toast("Basic settings updated!", icon="✅")
        except:
            st.toast("Server Connection Failed!", icon="❌")

    conf = st.slider("YOLO Confidence", 0.1, 1.0, key="conf", on_change=on_change_settings)
    lstm = st.slider("Fall Threshold (LSTM)", 0.1, 1.0, key="lstm", on_change=on_change_settings)

# --- MAIN UI ---
col_video, col_info = st.columns([3, 1.2])

with col_video:
    if selected_video:
        video_path = os.path.join(video_folder, selected_video)
        # URL Video Stream
        stream_url = f"{API_URL}/video_feed?video_path={video_path}"
        
        st.markdown(
            f"""
            <div style="border: 2px solid #4CAF50; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                <img src="{stream_url}" width="100%" />
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("No video files found in 'samples' folder.", icon="i️")

with col_info:
    # 1. AUTO REFRESH LOGIC (Chạy mỗi 2 giây)
    st_autorefresh(interval=2000, limit=None, key="status_refresher")

    st.subheader("📡 Status")
    status_ph = st.empty()
    
    st.divider()
    st.subheader("📸 Evidence (Latest)")
    gallery_ph = st.empty()

    # --- LOGIC GỌI API NGẦM ---
    try:
        # A. Lấy trạng thái cảnh báo
        status_res = requests.get(f"{API_URL}/status", timeout=0.5).json()
        with status_ph.container():
            if status_res.get("fall_detected"):
                st.error("🚨 WARNING: FALL DETECTED!", icon="⚠️")
            else:
                st.success("✅ Safe Area", icon="🛡️")

        # B. Lấy Gallery (Chỉ của video đang chọn)
        if selected_video:
            current_video_name = os.path.splitext(selected_video)[0]
            
            gallery_res = requests.get(
                f"{API_URL}/gallery", 
                params={"video_name": current_video_name}, 
                timeout=0.5
            ).json()
            
            images = gallery_res.get("images", [])
            
            with gallery_ph.container():
                if not images:
                    st.info("No events captured yet.")
                else:
                    # Display in a 2-column grid
                    cols = st.columns(2)
                    for idx, img_rel_path in enumerate(images):
                        # Full image URL
                        img_url = f"{API_URL}/snapshots/{img_rel_path}"
                        # caption ngắn gọn
                        caption = img_rel_path.split("/")[-1] 
                        cols[idx % 2].image(img_url, caption=caption, width='stretch')

    except Exception:
        status_ph.warning("Connecting to AI Server...")