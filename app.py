import streamlit as st
import cv2
import tempfile
import os
from fall_logic import FallDetector

st.set_page_config(page_title="Hệ thống Phát hiện Ngã", layout="wide")
st.title("🚨 Surveillance Fall Detection System")

# --- CSS tùy chỉnh để làm đẹp ---
st.markdown("""
    <style>
        .stImage { border: 2px solid #ddd; border-radius: 5px; }
        div[data-testid="stMetricValue"] { font-size: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- Session State để lưu lịch sử snapshot ---
if 'snapshot_history' not in st.session_state:
    st.session_state['snapshot_history'] = []

st.sidebar.title("⚙️ Cấu hình")
conf_thresh = st.sidebar.slider("Độ tin cậy (Confidence)", 0.0, 1.0, 0.8, 0.05) 
fall_thresh = st.sidebar.slider("Ngưỡng tỷ lệ ngã (W/H Ratio)", 0.5, 2.0, 2.0, 0.1)
uploaded_file = st.sidebar.file_uploader("Chọn video đầu vào", type=['mp4', 'avi', 'mov'])

col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("📋 Nhật ký báo động")
    alert_container = st.container() # Vùng chứa danh sách ảnh
    
    st.divider()
    stop_button = st.button("Dừng hệ thống", type="primary")

if uploaded_file is not None:
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
    except Exception as e:
        st.error(f"Lỗi khi tải video: {e}")
        st.stop()
    
    with st.spinner('Đang khởi tạo mô hình AI...'):
        # Lưu ý: Class FallDetector đã được update bên trên
        detector = FallDetector(conf_threshold=conf_thresh, fall_ratio=fall_thresh)
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = col1.empty()

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret: break

        detector.conf_threshold = conf_thresh
        detector.fall_ratio_threshold = fall_thresh

        # --- NHẬN THÊM BIẾN SNAPSHOT_DIR ---
        processed_frame, fall_count, snapshot_dir = detector.process_frame(frame)

        # Hiển thị Video Main
        st_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")

    # Hiển thị Gallery (Bên phải)
    with alert_container:
        if fall_count > 0:
            st.error(f"⚠️ ĐANG CÓ NGƯỜI NGÃ!", icon="🚨")
        
        # Quét thư mục snapshot để lấy danh sách ảnh
        # Lọc file .jpg
        if os.path.exists(snapshot_dir):
            images = [f for f in os.listdir(snapshot_dir) if f.endswith('.jpg')]
            
            if not images:
                st.info("Chưa có dữ liệu ngã.")
            else:
                st.write("📸 Bằng chứng (Best Score):")
                # Hiển thị các ảnh tìm được
                for img_file in images:
                    img_path = os.path.join(snapshot_dir, img_file)
                    # Dùng time để trick bộ nhớ đệm browser, giúp ảnh update realtime
                    # mỗi khi file bị ghi đè bởi score cao hơn
                    st.image(img_path, caption=img_file, width='stretch')
                    
                    # Nút xóa nhanh nếu muốn reset thủ công
                    if st.button(f"Xóa {img_file}", key=img_file):
                        os.remove(img_path)
                        st.experimental_rerun()

    cap.release()
    st.success("Đã dừng hệ thống phát hiện ngã.")