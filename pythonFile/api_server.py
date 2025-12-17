# api_server.py
import cv2
import uvicorn
import time
import os
import sys
import threading
from typing import Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- IMPORT MODULE CỦA BẠN ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import FallDetector
from camera_loader import CameraStream  

# --- CONFIG ---
# Lấy đường dẫn tuyệt đối của file hiện tại (api_server.py) -> /app/pythonFile/api_server.py
current_file_path = os.path.abspath(__file__)

# Lấy thư mục cha chứa file này -> /app/pythonFile
current_dir = os.path.dirname(current_file_path)

# Lấy thư mục cha của thư mục chứa file (Project Root) -> /app
project_root = os.path.dirname(current_dir)

# Tạo đường dẫn tuyệt đối tới folder snapshots -> /app/snapshots
SNAPSHOT_DIR = os.path.join(project_root, "snapshots")

if not os.path.exists(SNAPSHOT_DIR): os.makedirs(SNAPSHOT_DIR)

CAMERAS_CONFIG = {
    "cam_1": "rtsp://rtsp-server:8554/cam_1",
    "cam_2": "rtsp://rtsp-server:8554/cam_2"
}

# --- WORKER: KẾT HỢP CAMERA STREAM + AI ---
class SmartCameraWorker:
    def __init__(self, cam_id, rtsp_url):
        self.cam_id = cam_id
        
        # 1. Khởi tạo Stream (Lấy ảnh) - Dùng class bạn vừa tách
        self.stream = CameraStream(rtsp_url, cam_id)
        
        # 2. Khởi tạo AI (Xử lý ảnh)
        current_dir = os.path.dirname(os.path.abspath(__file__)) # /app/pythonFile
        project_root = os.path.dirname(current_dir)              # /app
        weights_dir = os.path.join(project_root, "weights")      # /app/weights
        
        path_pose = os.path.join(weights_dir, "yolo11s-pose.pt")
        path_lstm = os.path.join(weights_dir, "lstm_fall_model.pth")

        print(f"🤖 [{cam_id}] Loading AI from: {weights_dir}")
        
        self.detector = FallDetector(
            model_pose=path_pose, 
            model_lstm=path_lstm
        )
        
        # Trạng thái chia sẻ cho API
        self.output_frame = None
        self.lock = threading.Lock()
        self.state = {"fall": False, "snapshot": None}
        self.stopped = False

    def start(self):
        # Bắt đầu luồng lấy ảnh
        self.stream.start()
        
        # Bắt đầu luồng chạy AI
        self.thread = threading.Thread(target=self.run_ai_loop)
        self.thread.daemon = True
        self.thread.start()

    def run_ai_loop(self):
        """Vòng lặp lấy ảnh từ Stream -> Đưa vào AI -> Lưu kết quả"""
        save_path = os.path.join(SNAPSHOT_DIR, self.cam_id)
        if not os.path.exists(save_path): os.makedirs(save_path)
        max_score_in_session = 0.0

        while not self.stopped:
            # Lấy frame mới nhất từ CameraStream
            status, frame = self.stream.read()
            
            if not status or frame is None:
                time.sleep(0.1)
                continue

            # Resize xử lý cho nhanh
            w = frame.shape[1] // 2
            h = frame.shape[0] // 2
            frame_resized = cv2.resize(frame, (w, h))

            # --- AI INFERENCE ---
            processed_frame, fall_count, score = self.detector.process_frame(frame_resized)
            is_fall = (fall_count > 0)

            # Logic Snapshot (Giữ nguyên)
            snapshot_url = self.state["snapshot"]
            if is_fall:
                if score > max_score_in_session or score > 0.8:
                    max_score_in_session = score
                    filename = f"{self.cam_id}_fall_{int(score*100)}.jpg"
                    cv2.imwrite(os.path.join(save_path, filename), processed_frame)
                    snapshot_url = f"/snapshots/{self.cam_id}/{filename}?t={int(time.time())}"
            else:
                if max_score_in_session > 0: max_score_in_session = 0.0

            # Encode ảnh để API hiển thị
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if ret:
                with self.lock:
                    self.output_frame = buffer.tobytes()
                    self.state = {"fall": is_fall, "snapshot": snapshot_url}
            
            # Giới hạn FPS xử lý AI (không cần chạy quá nhanh gây nóng máy)
            # Nếu Camera 30fps nhưng AI chạy 15fps là đủ dùng
            time.sleep(0.03) 

    def get_frame(self):
        with self.lock:
            return self.output_frame
            
    def get_state(self):
        with self.lock:
            return self.state

    def stop(self):
        self.stopped = True
        self.stream.stop()
        self.thread.join()

# --- SERVER SETUP ---
workers: Dict[str, SmartCameraWorker] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    for cam_id, url in CAMERAS_CONFIG.items():
        worker = SmartCameraWorker(cam_id, url)
        worker.start()
        workers[cam_id] = worker
    yield
    # SHUTDOWN
    for worker in workers.values():
        worker.stop()

app = FastAPI(lifespan=lifespan)
app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")
templates = Jinja2Templates(directory="pythonFile/templates")

# --- API ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "cameras": CAMERAS_CONFIG})

@app.get("/api/updates")
def get_updates():
    return {cid: w.get_state() for cid, w in workers.items()}

def frame_generator(cam_id):
    worker = workers.get(cam_id)
    while True:
        frame = worker.get_frame() if worker else None
        if frame:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05) # 20 FPS view

@app.get("/video_feed")
def video_feed(cam_id: str):
    if cam_id not in workers: return HTMLResponse("Offline", 404)
    return StreamingResponse(frame_generator(cam_id), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)