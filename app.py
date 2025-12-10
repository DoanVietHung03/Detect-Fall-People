import streamlit as st
import cv2
import os
import glob
import shutil
import tempfile
from inference import FallDetector

# --- CẤU HÌNH ---
st.set_page_config(page_title="Hệ thống Phát hiện Ngã", layout="wide", page_icon="🚨")
VIDEO_DIR = "samples"
SNAPSHOT_DIR = "snapshots"

# --- CSS TÙY CHỈNH ---
st.markdown("""
    <style>
        .stImage { border: 2px solid #ddd; border-radius: 5px; }
        div[data-testid="stMetricValue"] { font-size: 20px; }
        div.stButton > button:first-child {
            width: 100%; text-align: left; padding-left: 15px; border: 1px solid #eee;
        }
        div.stButton > button:hover {
            border-color: #ff4b4b; color: #ff4b4b;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🚨 AI Surveillance Fall Detection")

# --- QUẢN LÝ STATE ---
if 'selected_video_path' not in st.session_state:
    st.session_state['selected_video_path'] = None

# --- HÀM HỖ TRỢ ---
def clear_history():
    if os.path.exists(SNAPSHOT_DIR):
        try:
            shutil.rmtree(SNAPSHOT_DIR)
            os.makedirs(SNAPSHOT_DIR)
        except Exception: pass
    else:
        os.makedirs(SNAPSHOT_DIR)

def get_video_files():
    if not os.path.exists(VIDEO_DIR): os.makedirs(VIDEO_DIR); return []
    exts = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    files = []
    for ext in exts: files.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))
    return sorted([os.path.basename(f) for f in files])

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Cấu hình Model")
    conf_thresh = st.slider("Độ tin cậy (Confidence)", 0.3, 1.0, 0.85, 0.05) 
    lstm_thresh = st.slider("Ngưỡng nhạy ngã (LSTM)", 0.5, 0.99, 0.75, 0.05)
    st.caption("Ngưỡng càng cao, AI càng ít báo ảo nhưng có thể bỏ sót.")

    st.divider()
    st.subheader("📂 Danh sách Video")
    video_files = get_video_files()
    if video_files:
        for vid_name in video_files:
            if st.button(f"▶ {vid_name}", key=vid_name):
                st.session_state['selected_video_path'] = os.path.join(VIDEO_DIR, vid_name)
                clear_history()
                st.rerun()
    
    st.divider()
    uploaded_file = st.file_uploader("Tải video lên", type=['mp4', 'avi'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        if st.session_state['selected_video_path'] != tfile.name:
             st.session_state['selected_video_path'] = tfile.name
             clear_history()
             st.rerun()

# ================= MAIN UI =================
col_video, col_alert = st.columns([3, 1.2])

with col_alert:
    st.subheader("📋 Trạng thái & Bằng chứng")
    # KHỞI TẠO CÁC PLACEHOLDER CỐ ĐỊNH (Quan trọng!)
    status_ph = st.empty()       # 1. Khung hiển thị trạng thái Realtime
    st.divider()
    gallery_ph = st.empty()      # 2. Khung hiển thị Ảnh bằng chứng (Dùng st.empty thay vì container)
    st.divider()
    stop_btn = st.button("⏹ DỪNG HỆ THỐNG", type="primary")

video_path = st.session_state.get('selected_video_path')

if video_path and not stop_btn:
    st.info(f"Đang xử lý: **{os.path.basename(video_path)}**")
    
    detector = FallDetector(conf_threshold=conf_thresh, lstm_threshold=lstm_thresh)
    cap = cv2.VideoCapture(video_path)
    frame_ph = col_video.empty()
    
    frame_count = 0
    SKIP_RATE = 2  # Xử lý mỗi 2 frame để tăng tốc độ

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        if frame_count % (SKIP_RATE + 1) != 0:
            frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width='content')
            continue

        detector.conf_threshold = conf_thresh
        detector.lstm_threshold = lstm_thresh
        processed_frame, fall_count, _ = detector.process_frame(frame)

        # 1. Hiển thị Video
        frame_ph.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", width='content')

        # 2. Hiển thị Trạng thái (Ghi đè nội dung cũ của status_ph)
        if fall_count > 0:
            status_ph.error(f"🚨 CẢNH BÁO: {fall_count} NGƯỜI NGÃ!", icon="⚠️")
        else:
            status_ph.success("✅ Khu vực an toàn", icon="🛡️")

        # 3. Hiển thị Gallery (Dùng context manager của gallery_ph)
        with gallery_ph.container():
            if os.path.exists(SNAPSHOT_DIR):
                images = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, '*.jpg')))
                
                if not images:
                    st.info("Chưa ghi nhận sự cố nào.", icon="📝")
                else:
                    st.warning(f"📸 Đã lưu {len(images)} hồ sơ sự cố:")
                    for img_path in images:
                        file_name = os.path.basename(img_path)
                        display_name = file_name.replace("fall_evidence_", "").replace(".jpg", "")
                        # FIX LỖI WARNING VÀNG: Dùng use_container_width=True
                        st.image(img_path, caption=f"ID: {display_name}", width='content')

    cap.release()
    st.success("Đã kết thúc video.")

elif stop_btn:
    st.write("Đã dừng hệ thống.")
else:
    col_video.info("👈 Chọn video để bắt đầu.")