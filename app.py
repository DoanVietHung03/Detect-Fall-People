import streamlit as st
import numpy as np
import cv2
import os
import glob
import shutil
import tempfile
import threading
import queue
import time
from inference import FallDetector

# --- CẤU HÌNH ---
st.set_page_config(page_title="Hệ thống Phát hiện Ngã", layout="wide", page_icon="⚡")
VIDEO_DIR = "samples"
SNAPSHOT_DIR = "snapshots"

# --- CSS ---
st.markdown("""
    <style>
        .stImage { border: 2px solid #ddd; border-radius: 5px; }
        div[data-testid="stMetricValue"] { font-size: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- QUẢN LÝ STATE ---
if 'selected_video_path' not in st.session_state:
    st.session_state['selected_video_path'] = None
if 'stop_thread' not in st.session_state:
    st.session_state['stop_thread'] = False

# --- CLASS XỬ LÝ ĐA LUỒNG (THREADING) ---
class VideoProcessor(threading.Thread):
    def __init__(self, video_path, conf_thresh, lstm_thresh, frame_queue, result_queue):
        super().__init__()
        self.video_path = video_path
        self.conf_thresh = conf_thresh
        self.lstm_thresh = lstm_thresh
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stopped = False
        self.detector = None

    def run(self):
        # Khởi tạo model
        self.detector = FallDetector(conf_threshold=self.conf_thresh, lstm_threshold=self.lstm_thresh)
        cap = cv2.VideoCapture(self.video_path)
        
        # 1. LẤY FPS GỐC CỦA VIDEO
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None: 
            fps = 30 # Fallback nếu không đọc được
        
        frame_duration = 1.0 / fps # Ví dụ 30fps -> 0.033s mỗi frame

        SKIP_FRAMES = 0 # Không skip frame nữa
        frame_idx = 0

        while not self.stopped and cap.isOpened():
            start_time = time.time() # <--- BẤM GIỜ BẮT ĐẦU

            ret, frame = cap.read()
            if not ret: break
            
            frame_idx += 1
            if SKIP_FRAMES > 0 and frame_idx % (SKIP_FRAMES + 1) != 0:
                continue

            # Resize
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                new_h = int(h * scale)
                resized_frame = cv2.resize(frame, (640, new_h))
            else:
                resized_frame = frame
              
            # Tăng độ tương phản (Gamma Correction)  
            gamma = 1.8 # Thử chỉnh từ 1.2 đến 2.0
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced_frame = cv2.LUT(resized_frame, table)

            # Update threshold & Process
            self.detector.conf_threshold = self.conf_thresh
            self.detector.lstm_threshold = self.lstm_thresh
            processed_frame, fall_count, snap_dir = self.detector.process_frame(enhanced_frame)

            # Put to Queue
            if self.result_queue.full():
                try: self.result_queue.get_nowait()
                except queue.Empty: pass
            
            self.result_queue.put({
                'frame': processed_frame,
                'fall_count': fall_count,
                'snap_dir': snap_dir,
                'has_new_fall': fall_count > 0
            })
            
            # 2. LOGIC ĐỒNG BỘ TỐC ĐỘ (SYNC FPS)
            # Tính thời gian đã trôi qua cho việc xử lý frame này
            processing_time = time.time() - start_time
            
            # Nếu xử lý quá nhanh (nhanh hơn thời gian thực của frame), thì ngủ một chút
            if processing_time < frame_duration:
                time.sleep(frame_duration - processing_time)

        cap.release()
        self.stopped = True

    def stop(self):
        self.stopped = True

# --- UI FUNCTIONS ---
def clear_history():
    if os.path.exists(SNAPSHOT_DIR):
        try: shutil.rmtree(SNAPSHOT_DIR); os.makedirs(SNAPSHOT_DIR)
        except: pass
    else: os.makedirs(SNAPSHOT_DIR)

def get_video_files():
    if not os.path.exists(VIDEO_DIR): os.makedirs(VIDEO_DIR); return []
    return sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mkv'))])

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚡ Cấu hình & Tối ưu")
    conf_thresh = st.slider("Confidence YOLO", 0.3, 1.0, 0.8) 
    lstm_thresh = st.slider("Ngưỡng LSTM", 0.5, 0.99, 0.7)
    
    st.divider()
    st.subheader("Video")
    for vid in get_video_files():
        if st.button(f"▶ {vid}"):
            st.session_state['selected_video_path'] = os.path.join(VIDEO_DIR, vid)
            clear_history()
            st.rerun()

    uploaded = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mkv'])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        st.session_state['selected_video_path'] = tfile.name

# ================= MAIN UI =================
st.title("⚡ AI Fall Detection (Multi-threaded)")
col_video, col_info = st.columns([3, 1.5])

# Placeholder
with col_video:
    video_ph = st.empty()
with col_info:
    status_ph = st.empty()
    st.divider()
    gallery_ph = st.empty() # Gallery placeholder
    stop_btn = st.button("⏹ DỪNG", type="primary")

# Logic chính
video_path = st.session_state.get('selected_video_path')

if video_path and not stop_btn:
    # Hàng đợi giao tiếp giữa 2 luồng
    frame_queue = queue.Queue(maxsize=1) 
    result_queue = queue.Queue(maxsize=2) # Chỉ giữ tối đa 2 kết quả chờ để đảm bảo realtime

    # Khởi động luồng AI
    processor = VideoProcessor(video_path, conf_thresh, lstm_thresh, frame_queue, result_queue)
    processor.start()

    st.toast(f"Đang khởi động AI Engine...", icon="🚀")
    
    # Biến cache để tránh đọc ổ cứng liên tục
    cached_images = []
    last_update_gallery = 0

    while processor.is_alive():
        try:
            # Chờ lấy kết quả từ luồng AI (timeout 0.1s để không treo UI)
            data = result_queue.get(timeout=0.1)
            
            # 1. Hiển thị Video
            frame_rgb = cv2.cvtColor(data['frame'], cv2.COLOR_BGR2RGB)
            video_ph.image(frame_rgb, channels="RGB", width='content')

            # 2. Hiển thị Trạng thái
            if data['fall_count'] > 0:
                status_ph.error(f"🚨 PHÁT HIỆN: {data['fall_count']} NGƯỜI NGÃ!", icon="⚠️")
            else:
                status_ph.success("✅ Đang giám sát...", icon="🛡️")

            # 3. Cập nhật Gallery (Chỉ update khi có ngã hoặc mỗi 5 giây 1 lần)
            # TỐI ƯU: Không gọi glob.glob mỗi frame!
            current_time = time.time()
            if data['has_new_fall'] and (current_time - last_update_gallery > 1.0):
                last_update_gallery = current_time
                if os.path.exists(SNAPSHOT_DIR):
                    cached_images = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, '*.jpg')), key=os.path.getmtime, reverse=True)
                
                with gallery_ph.container():
                    st.write(f"📸 **Bằng chứng ({len(cached_images)})**")
                    if cached_images:
                        # Chỉ hiện 3 ảnh mới nhất để đỡ lag
                        cols = st.columns(3)
                        for idx, img_path in enumerate(cached_images[:3]):
                            cols[idx].image(img_path, caption=os.path.basename(img_path))

        except queue.Empty:
            continue
    
    processor.stop()
    processor.join()
    st.success("Kết thúc video.")

elif stop_btn:
    st.session_state['stop_thread'] = True
    st.write("Hệ thống đã dừng.")
else:
    st.info("👈 Chọn video để bắt đầu.")