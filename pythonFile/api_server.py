# api_server.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import uvicorn
import time
import os
import sys

# Import AI logic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import FallDetector

app = FastAPI()

# --- CẤU HÌNH ---
SNAPSHOT_DIR = "../snapshots"
if not os.path.exists(SNAPSHOT_DIR): os.makedirs(SNAPSHOT_DIR)

app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")
templates = Jinja2Templates(directory="pythonFile/templates")

# --- DANH SÁCH CAMERA ---
# Định nghĩa danh sách camera cứng (hoặc load từ DB)
CAMERAS = {
    "cam_coffee": "rtsp://rtsp-server:8554/cam_coffee",
    "cam_home": "rtsp://rtsp-server:8554/cam_home"
}

# --- GLOBAL STATE (Lưu trạng thái từng camera) ---
# Cấu trúc: { "cam_id": { "fall": False, "snapshot": "/path/to/img.jpg" } }
camera_states = {cam_id: {"fall": False, "snapshot": None} for cam_id in CAMERAS}

# Model AI
print("⏳ Loading AI Model...")
detector = FallDetector(
    model_pose='../weights/yolo11n-pose.pt', 
    model_lstm='../weights/lstm_fall_model.pth'
)
print("✅ AI Ready!")

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Truyền danh sách camera vào giao diện để render vòng lặp
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "cameras": CAMERAS
    })

@app.get("/video_feed")
def video_feed(cam_id: str):
    """Luồng Video Streaming riêng cho từng Camera"""
    rtsp_url = CAMERAS.get(cam_id)
    if not rtsp_url: return HTMLResponse("Camera not found", status_code=404)
    
    return StreamingResponse(
        generate_frames(cam_id, rtsp_url), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/updates")
def get_updates():
    """API để JS lấy trạng thái mới nhất của TẤT CẢ camera"""
    return camera_states

# --- XỬ LÝ VIDEO & AI ---
def generate_frames(cam_id, video_path):
    # Setup OpenCV
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Tạo thư mục lưu ảnh riêng cho cam này
    save_path = os.path.join(SNAPSHOT_DIR, cam_id)
    if not os.path.exists(save_path): os.makedirs(save_path)

    max_score_in_session = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            cap.open(video_path, cv2.CAP_FFMPEG)
            continue
        
        frame = cv2.resize(frame, (640, 480))

        # --- AI INFERENCE ---
        processed_frame, fall_count, score = detector.process_frame(frame)
        
        # Cập nhật trạng thái Global
        is_fall = (fall_count > 0)
        camera_states[cam_id]["fall"] = is_fall

        # --- LOGIC LƯU ẢNH SNAPSHOT (Best Score) ---
        if is_fall:
            # Chỉ lưu và cập nhật hiển thị nếu điểm tin cậy cao hơn ảnh trước đó
            # Hoặc cập nhật mỗi 2 giây nếu đang ngã liên tục để user thấy diễn biến
            if score > max_score_in_session or score > 0.8: 
                max_score_in_session = score
                timestamp = int(time.time())
                filename = f"{cam_id}_fall_{int(score*100)}.jpg"
                full_path = os.path.join(save_path, filename)
                
                cv2.imwrite(full_path, processed_frame)
                
                # Cập nhật đường dẫn ảnh để Frontend hiển thị
                camera_states[cam_id]["snapshot"] = f"/snapshots/{cam_id}/{filename}"
        else:
            # Reset score khi hết ngã để chuẩn bị cho lần ngã sau
            if max_score_in_session > 0: max_score_in_session = 0.0

        # Encode JPEG gửi về trình duyệt
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)