# api_server.py
import cv2
import uvicorn
import time
import os
import sys
import torch
import multiprocessing as mp # Import thư viện multiprocessing
from queue import Empty, Full # Để xử lý ngoại lệ queue

from typing import Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- IMPORT MODULE ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import FallDetector
from camera_loader import CameraStream  

# --- CONFIG ---
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
SNAPSHOT_DIR = os.path.join(project_root, "snapshots")
if not os.path.exists(SNAPSHOT_DIR): os.makedirs(SNAPSHOT_DIR)

CAMERAS_CONFIG = {
    "cam_1": "rtsp://rtsp-server:8554/cam_1",
    "cam_2": "rtsp://rtsp-server:8554/cam_2",
    # Thêm cam_3, cam_4... thoải mái
}

# --- PROCESS CLASS (Thay thế Thread Class cũ) ---
class CameraProcess(mp.Process):
    def __init__(self, cam_id, rtsp_url, frame_queue, state_queue, command_event):
        super().__init__()
        self.cam_id = cam_id
        self.rtsp_url = rtsp_url
        self.frame_queue = frame_queue   # Queue để gửi ảnh về API (hiển thị)
        self.state_queue = state_queue   # Queue để gửi trạng thái (ngã hay không)
        self.command_event = command_event # Event để báo dừng
        
        # CHÚ Ý: KHÔNG load model ở đây (đây là Process Cha)

    def run(self):
        # --- ĐÂY LÀ PROCESS CON (CHẠY ĐỘC LẬP) ---
        print(f"🚀 [{self.cam_id}] Process Started. PID: {os.getpid()}")
        
        # 1. Load Model (Chỉ load trong process con để mỗi con có CUDA context riêng)
        weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
        path_pose = os.path.join(weights_dir, "yolo11s-pose.onnx") 
        path_onnx = os.path.join(weights_dir, "gru_fall_model.onnx")
        
        try:
            detector = FallDetector(model_pose=path_pose, model_onnx=path_onnx)
        except Exception as e:
            print(f"❌ [{self.cam_id}] AI Init Failed: {e}")
            return

        # 2. Khởi tạo Camera Stream
        stream = CameraStream(self.rtsp_url, self.cam_id)
        stream.start()

        save_path = os.path.join(SNAPSHOT_DIR, self.cam_id)
        if not os.path.exists(save_path): os.makedirs(save_path)
        max_score_in_session = 0.0
        
        # Biến local lưu state để không spam queue
        current_state = {"fall": False, "snapshot": None}

        while not self.command_event.is_set():
            status, frame = stream.read()
            if not status or frame is None:
                time.sleep(0.01)
                continue

            # Resize & Inference
            w = frame.shape[1] // 2
            h = frame.shape[0] // 2
            frame_resized = cv2.resize(frame, (w, h))

            processed_frame, fall_count, score = detector.process_frame(frame_resized)
            is_fall = (fall_count > 0)

            # Logic Snapshot (như cũ)
            snapshot_url = current_state["snapshot"]
            if is_fall:
                if score > max_score_in_session or score > 0.8:
                    max_score_in_session = score
                    filename = f"{self.cam_id}_fall_{int(score*100)}.jpg"
                    cv2.imwrite(os.path.join(save_path, filename), processed_frame)
                    snapshot_url = f"/snapshots/{self.cam_id}/{filename}?t={int(time.time())}"
            else:
                if max_score_in_session > 0: max_score_in_session = 0.0
            
            # --- GỬI DỮ LIỆU VỀ API (QUAN TRỌNG) ---
            
            # 1. Gửi Frame (Dùng put_nowait và try-except để không bị block nếu queue đầy)
            # Encode JPG trước khi gửi để giảm dung lượng qua Queue (quan trọng cho performance)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                try:
                    # Nếu queue đầy, lấy cái cũ ra vứt đi để bỏ cái mới vào (luôn lấy ảnh mới nhất)
                    if self.frame_queue.full():
                        try: self.frame_queue.get_nowait()
                        except Empty: pass 
                    self.frame_queue.put_nowait(frame_bytes)
                except Full:
                    pass # Queue vẫn đầy thì bỏ qua frame này

            # 2. Gửi State (Chỉ gửi khi có thay đổi hoặc định kỳ để tiết kiệm CPU)
            new_state = {"fall": is_fall, "snapshot": snapshot_url}
            if new_state != current_state or time.time() % 1.0 < 0.05: # Gửi mỗi 1s hoặc khi khác biệt
                try:
                    if self.state_queue.full():
                        try: self.state_queue.get_nowait()
                        except: pass
                    self.state_queue.put_nowait(new_state)
                    current_state = new_state
                except: pass

            # Giới hạn FPS AI (tùy chỉnh)
            time.sleep(0.03) 
        
        # Cleanup
        stream.stop()
        print(f"🛑 [{self.cam_id}] Process Stopped.")

# --- QUẢN LÝ CÁC PROCESS ---
processes = {}
queues = {} # Lưu queue của từng cam: { "cam_1": {"frame": Q, "state": Q, "last_state_data": {}} }

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Set start method là 'spawn' để an toàn cho CUDA
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # Đã set rồi thì thôi

    print("🚀 Starting Camera Processes...")
    for cam_id, url in CAMERAS_CONFIG.items():
        # Tạo Queue với maxsize=1 (Chỉ giữ 1 frame/state mới nhất)
        frame_q = mp.Queue(maxsize=1)
        state_q = mp.Queue(maxsize=1)
        stop_event = mp.Event()

        p = CameraProcess(cam_id, url, frame_q, state_q, stop_event)
        p.start()
        
        processes[cam_id] = {"process": p, "stop_event": stop_event}
        queues[cam_id] = {"frame": frame_q, "state": state_q, "last_known_state": {"fall": False, "snapshot": None}}
    
    yield
    
    print("🛑 Shutting down Camera Processes...")
    for cam_id, item in processes.items():
        item["stop_event"].set()
        item["process"].join(timeout=5)
        if item["process"].is_alive():
            item["process"].terminate()

app = FastAPI(lifespan=lifespan)
app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")
templates = Jinja2Templates(directory="pythonFile/templates")

# --- API ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "cameras": CAMERAS_CONFIG})

@app.get("/api/updates")
def get_updates():
    # Lấy state mới nhất từ Queue (Non-blocking)
    results = {}
    for cam_id, item in queues.items():
        q = item["state"]
        try:
            # Lấy data mới nếu có
            while not q.empty():
                item["last_known_state"] = q.get_nowait()
        except Empty:
            pass
        results[cam_id] = item["last_known_state"]
    return results

def frame_generator(cam_id):
    if cam_id not in queues: return
    frame_q = queues[cam_id]["frame"]
    
    while True:
        try:
            # Timeout 1s để tránh vòng lặp chết nếu process chết
            frame_bytes = frame_q.get(timeout=1.0) 
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Empty:
            # Nếu không có frame nào trong 1s (Camera mất kết nối hoặc lỗi)
            # Có thể trả về ảnh placeholder hoặc chờ tiếp
            time.sleep(0.1)

@app.get("/video_feed")
def video_feed(cam_id: str):
    if cam_id not in CAMERAS_CONFIG: return HTMLResponse("Not Found", 404)
    return StreamingResponse(frame_generator(cam_id), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)