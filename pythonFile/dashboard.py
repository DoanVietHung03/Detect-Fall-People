# dashboard.py
import streamlit as st
import requests
import os
import cv2
import numpy as np
from streamlit_autorefresh import st_autorefresh
from streamlit_drawable_canvas import st_canvas
from PIL import Image

API_URL = "http://localhost:8000"

st.set_page_config(layout="wide", page_title="AI Surveillance Center", page_icon="📹")

# --- CSS Tùy chỉnh ---
st.markdown("""
    <style>
        .stImage img { border-radius: 8px; border: 2px solid #ddd; }
        div[data-testid="stMetricValue"] { font-size: 18px; }
        /* Làm nút bấm to hơn cho dễ bấm */
        div.stButton > button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

st.title("📹 Smart Surveillance Center")

# --- QUẢN LÝ TRẠNG THÁI (SESSION STATE) ---
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'selected_video_path_prev' not in st.session_state:
    st.session_state.selected_video_path_prev = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Camera Selection")
    
    # Định nghĩa danh sách Camera (Kết nối tới tên service trong Docker)
    CAMERA_LIST = {
        "☕ Coffee Room Cam": "rtsp://rtsp-server:8554/cam_coffee",
        "🏠 Home Cam 01": "rtsp://rtsp-server:8554/cam_home"
    }

    # Chọn Camera thay vì chọn Folder/File
    selected_cam_name = st.selectbox("Select Active Camera", list(CAMERA_LIST.keys()))
    
    # Lấy đường dẫn RTSP tương ứng
    selected_video_path = CAMERA_LIST[selected_cam_name]
    selected_video_name = selected_cam_name # Dùng tên hiển thị để đặt tên folder snapshot

    # Reset trạng thái khi đổi cam
    if selected_video_path != st.session_state.get('selected_video_path_prev'):
        st.session_state.is_playing = False
        st.session_state.selected_video_path_prev = selected_video_path

    st.divider()
    st.header("🎛️ Parameters")
    
    if 'conf' not in st.session_state: st.session_state.conf = 0.7
    if 'lstm' not in st.session_state: st.session_state.lstm = 0.7

    def on_change_settings():
        try:
            payload = {"conf": st.session_state.conf, "lstm": st.session_state.lstm}
            requests.post(f"{API_URL}/update_settings", json=payload, timeout=1)
            st.toast("Settings updated!", icon="✅")
        except:
            st.toast("Connection Failed!", icon="❌")

    conf = st.slider("YOLO Confidence", 0.1, 1.0, key="conf", on_change=on_change_settings)
    lstm = st.slider("Fall Threshold", 0.1, 1.0, key="lstm", on_change=on_change_settings)

# --- TẠO TABS ---
tab_mon, tab_zone = st.tabs(["📡 Live Monitor", "🔧 Zone Setup (Anti-False Alarm)"])

# === TAB 1: MONITOR ===
with tab_mon:
    col_video, col_info = st.columns([3, 1.2])
    
    with col_video:
        if selected_video_path and (os.path.exists(selected_video_path) or "rtsp://" in selected_video_path):
            
            # --- LOGIC NÚT BẤM START/STOP ---
            if st.session_state.is_playing:
                # Nút DỪNG
                if st.button("⏹️ STOP PROCESSING", type="secondary"):
                    st.session_state.is_playing = False
                    st.rerun()
                
                # Đang chạy -> Hiện luồng video từ Server
                stream_url = f"{API_URL}/video_feed?video_path={selected_video_path}"
                st.markdown(
                    f"""
                    <div style="border: 2px solid #4CAF50; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                        <img src="{stream_url}" width="100%" />
                    </div>
                    """, unsafe_allow_html=True
                )
            else:
                # Nút BẮT ĐẦU
                if st.button("▶️ START ANALYSIS", type="primary"):
                    st.session_state.is_playing = True
                    st.rerun()
                
                # Chưa chạy -> Hiện ảnh thumbnail tĩnh (Frame đầu tiên)
                cap = cv2.VideoCapture(selected_video_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame, caption="Video Preview (Click Start to Analyze)", use_container_width=True)
                else:
                    st.warning("Cannot read video file.")

        else:
            st.info("Please select a valid video from the sidebar.", icon="👈")

    with col_info:    
        st.subheader("Status")
        status_ph = st.empty()
        st.divider()
        st.subheader("Evidence")
        gallery_ph = st.empty()

        # Logic hiển thị trạng thái và ảnh
        if st.session_state.is_playing:
            st_autorefresh(interval=3000, limit=None, key="status_refresher")
            try:
                status_res = requests.get(f"{API_URL}/status", timeout=0.5).json()
                with status_ph.container():
                    if status_res.get("fall_detected"):
                        st.error("🚨 FALL DETECTED!", icon="⚠️")
                    else:
                        st.success("✅ Safe Area", icon="🛡️")
                        
                # Kiểm tra: Dashboard đang Play NHƯNG Server báo đã tắt (is_active = False)
                is_server_active = status_res.get("is_active", False)

                if not is_server_active:
                    # Để tránh trường hợp vừa bấm Start server chưa kịp bật True
                    # Ta có thể check thêm hoặc chấp nhận độ trễ của autorefresh (2s là đủ để server start)
                    st.session_state.is_playing = False
                    st.rerun()
                    
                try:
                    if selected_video_name:
                        current_video_key = os.path.splitext(selected_video_name)[0]
                        gallery_res = requests.get(f"{API_URL}/gallery", params={"video_name": current_video_key}, timeout=0.5).json()
                        images = gallery_res.get("images", [])
                        with gallery_ph.container():
                            if not images: st.info("No events.")
                            else:
                                cols = st.columns(2)
                                for idx, img_rel_path in enumerate(images):
                                    img_url = f"{API_URL}/snapshots/{img_rel_path}"
                                    cols[idx % 2].image(img_url, caption=img_rel_path.split("/")[-1], use_container_width=True)
                except Exception as e:
                    print(f"Gallery fetch error: {e}")
            except:
                status_ph.warning("Connecting to Server...")
        else:
            status_ph.info("System Standby")
            gallery_ph.info("Click Start to view evidence.")

# === TAB 2: ZONE CONFIG ===
with tab_zone:
    st.info("Instructions: Draw safe zones (e.g., bed, sofa) over the image. AI will NOT trigger alarms in these zones.", icon="💡")
    
    col_draw, col_ctrl = st.columns([3, 1])
    
    bg_image = None
    if selected_video_path and (os.path.exists(selected_video_path) or "rtsp://" in selected_video_path):
        cap = cv2.VideoCapture(selected_video_path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bg_image = Image.fromarray(frame)
        cap.release()

    with col_draw:
        if bg_image:
            # --- TRY-EXCEPT ĐỂ XỬ LÝ LỖI PHIÊN BẢN THƯ VIỆN ---
            try:
                cv2.waitKey(0)
                canvas_result = st_canvas(
                    fill_color="rgba(0, 255, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="#00FF00",
                    background_image=bg_image,
                    update_streamlit=True,
                    height=480,
                    width=640,
                    drawing_mode="polygon",
                    key="canvas",
                )
            except Exception as e:
                st.error(f"Error drawing canvas: {e}")
                canvas_result = None
        else:
            st.warning("Please select a valid video to load the frame for zone setup.", icon="👈")

    with col_ctrl:
        st.write("### 🛠 Tools")
        st.write("- Click and drag to draw polygons.")
        st.write("- Right-click to close polygon.")
        
        if st.button("💾 Apply Safe Zones", type="primary"):
            if canvas_result and canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                zones_data = []
                for obj in objects:
                    if obj["type"] == "path":
                        raw_path = obj["path"] 
                        points = []
                        for item in raw_path:
                            if item[0] in ['M', 'L']:
                                points.append([int(item[1]), int(item[2])])
                        if len(points) > 2:
                            zones_data.append(points)
                
                if zones_data:
                    try:
                        requests.post(f"{API_URL}/update_zones", json={"zones": zones_data})
                        st.success(f"✅ Updated {len(zones_data)} safe zones!")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("No zones drawn!")