# api_server.py
import cv2
import uvicorn
import time
import os
import sys
import torch
import multiprocessing as mp
import numpy as np
from queue import Empty, Full
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
# Import class quản lý bộ nhớ vừa tạo
from shared_memory_utils import SharedFrameManager 

# --- CONFIG ---
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "snapshots")
if not os.path.exists(SNAPSHOT_DIR): os.makedirs(SNAPSHOT_DIR)

CAMERAS_CONFIG = {
    "cam_1": "rtsp://rtsp-server:8554/cam_1",
    "cam_2": "rtsp://rtsp-server:8554/cam_2",
}

# Kích thước cố định cho Shared Memory (Nên để bằng kích thước resize trong logic xử lý)
SHM_WIDTH = 640
SHM_HEIGHT = 480

# --- PROCESS CLASS ---
class CameraProcess(mp.Process):
    def __init__(self, cam_id, rtsp_url, shm_name, state_queue, command_event, lock):
        super().__init__()
        self.cam_id = cam_id
        self.rtsp_url = rtsp_url
        self.shm_name = shm_name
        self.state_queue = state_queue 
        self.command_event = command_event
        self.lock = lock # Lưu cái lock này lại

    def run(self):
        print(f"🚀 [{self.cam_id}] Process Started. PID: {os.getpid()}")
        
        # 1. Kết nối vào Shared Memory đã tạo bởi Process Cha
        shm_manager = SharedFrameManager(self.shm_name, SHM_WIDTH, SHM_HEIGHT, create=False)

        # 2. Load Model
        weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
        path_pose = os.path.join(weights_dir, "yolo11s-pose.onnx") 
        path_onnx = os.path.join(weights_dir, "gru_fall_model.onnx")
        
        try:
            detector = FallDetector(model_pose=path_pose, model_onnx=path_onnx)
        except Exception as e:
            print(f"❌ [{self.cam_id}] AI Init Failed: {e}")
            return

        stream = CameraStream(self.rtsp_url, self.cam_id)
        stream.start()
        
        save_path = os.path.join(SNAPSHOT_DIR, self.cam_id)
        if not os.path.exists(save_path): os.makedirs(save_path)
        
        max_score_in_session = 0.0
        current_state = {"fall": False, "snapshot": None}

        while not self.command_event.is_set():
            status, frame = stream.read()
            if not status or frame is None:
                time.sleep(0.01); continue

            # Resize về đúng kích thước Shared Memory
            frame_resized = cv2.resize(frame, (SHM_WIDTH, SHM_HEIGHT))

            # AI Inference
            processed_frame, fall_count, score = detector.process_frame(frame_resized)
            is_fall = (fall_count > 0)

            # Snapshot Logic (Giữ nguyên)
            snapshot_url = current_state["snapshot"]
            if is_fall:
                if score > max_score_in_session or score > 0.8:
                    max_score_in_session = score
                    filename = f"{self.cam_id}_fall_{int(score*100)}.jpg"
                    cv2.imwrite(os.path.join(save_path, filename), processed_frame)
                    snapshot_url = f"/snapshots/{self.cam_id}/{filename}?t={int(time.time())}"
            else:
                if max_score_in_session > 0: max_score_in_session = 0.0
            
            # --- GHI VÀO SHARED MEMORY ---
            # Thay vì queue.put(), ta ghi thẳng vào RAM
            shm_manager.write(processed_frame)

            # Gửi State (State nhỏ nên dùng Queue vẫn ổn)
            new_state = {"fall": is_fall, "snapshot": snapshot_url}
            if new_state != current_state or time.time() % 1.0 < 0.05:
                try:
                    if self.state_queue.full(): self.state_queue.get_nowait()
                    self.state_queue.put_nowait(new_state)
                    current_state = new_state
                except: pass

            time.sleep(0.03) # ~30 FPS limit
        
        stream.stop()
        shm_manager.close() # Đóng kết nối SHM
        print(f"🛑 [{self.cam_id}] Process Stopped.")

# --- QUẢN LÝ ---
processes = {}
queues = {} 
shm_managers = {} # Lưu các object quản lý bộ nhớ của Cha

@asynccontextmanager
async def lifespan(app: FastAPI):
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    print("🚀 Starting Camera Processes with SHARED MEMORY...")
    
    for cam_id, url in CAMERAS_CONFIG.items():
        state_q = mp.Queue(maxsize=1)
        stop_event = mp.Event()
        
        # --- TẠO LOCK CHUNG TẠI ĐÂY ---
        # Lock này thuộc về Process Cha, nhưng có thể truyền qua Process Con
        shm_lock = mp.Lock() 
        
        shm_name = f"shm_{cam_id}"
        
        # Truyền lock vào Manager của Cha (để hàm frame_generator dùng)
        shm_mgr = SharedFrameManager(shm_name, SHM_WIDTH, SHM_HEIGHT, create=True, lock=shm_lock)
        shm_managers[cam_id] = shm_mgr

        # Truyền ĐÚNG cái lock đó vào Process Con
        p = CameraProcess(cam_id, url, shm_name, state_q, stop_event, lock=shm_lock)
        p.start()
        
        processes[cam_id] = {"process": p, "stop_event": stop_event}
        queues[cam_id] = {"state": state_q, "last_known_state": {"fall": False, "snapshot": None}}
    
    yield
    
    print("🛑 Shutting down...")
    for cam_id, item in processes.items():
        item["stop_event"].set()
        item["process"].join(timeout=5)
        if item["process"].is_alive(): item["process"].terminate()
    
    # Dọn dẹp bộ nhớ chia sẻ
    print("🧹 Cleaning up Shared Memory...")
    for mgr in shm_managers.values():
        mgr.unlink() # Quan trọng: Giải phóng RAM cho OS

app = FastAPI(lifespan=lifespan)
app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")
templates = Jinja2Templates(directory="pythonFile/templates")

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "cameras": CAMERAS_CONFIG})

@app.get("/api/updates")
def get_updates():
    results = {}
    for cam_id, item in queues.items():
        q = item["state"]
        try:
            while not q.empty(): item["last_known_state"] = q.get_nowait()
        except Empty: pass
        results[cam_id] = item["last_known_state"]
    return results

def frame_generator(cam_id):
    """Đọc từ Shared Memory để stream về Browser"""
    if cam_id not in shm_managers: return
    
    mgr = shm_managers[cam_id] # Lấy manager tương ứng
    
    while True:
        # Đọc trực tiếp từ RAM (Cực nhanh)
        frame = mgr.read()
        
        # Nếu frame đen xì (chưa có dữ liệu), chờ chút
        if np.all(frame == 0):
            time.sleep(0.1)
            continue

        # Encode JPEG (Vẫn cần encode để gửi qua mạng, nhưng ta đã tiết kiệm công đoạn serialize qua Queue)
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Limit FPS hiển thị trên Web (không cần thiết phải 30fps nếu chỉ xem giám sát)
        time.sleep(0.04) 

@app.get("/video_feed")
def video_feed(cam_id: str):
    if cam_id not in CAMERAS_CONFIG: return HTMLResponse("Not Found", 404)
    return StreamingResponse(frame_generator(cam_id), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)