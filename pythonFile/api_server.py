# api_server.py
import cv2
import uvicorn
import time
import datetime
import os
import sys
import asyncio
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
from shared_memory_utils import SharedFrameManager 
from noti_services import NotificationService

# --- CONFIG ---
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "snapshots")
if not os.path.exists(SNAPSHOT_DIR): os.makedirs(SNAPSHOT_DIR)

CAMERAS_CONFIG = {
    "cam_1": "rtsp://rtsp-server:8554/cam_1",
    "cam_2": "rtsp://rtsp-server:8554/cam_2",
}

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
        self.lock = lock # Nhận Lock từ cha truyền vào (FIX LỖI SHARED MEMORY)

    def run(self):
        print(f"🚀 [{self.cam_id}] Process Started. PID: {os.getpid()}")
        
        # 1. Kết nối Shared Memory (Không tạo mới Lock, dùng lock được truyền vào)
        # Lưu ý: Cần sửa nhẹ shared_memory_utils để nhận lock từ ngoài, 
        # nhưng tạm thời ta sẽ gán lock thủ công sau khi init.
        shm_manager = SharedFrameManager(self.shm_name, SHM_WIDTH, SHM_HEIGHT, create=False)
        shm_manager.lock = self.lock # GHI ĐÈ LOCK CỤC BỘ BẰNG LOCK ĐỒNG BỘ CỦA OS

        # 2. Load Model
        weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
        path_pose = os.path.join(weights_dir, "yolo11s-pose-fp16.onnx") 
        path_onnx = os.path.join(weights_dir, "gru_fall_model_fp16.onnx") # Đã update sang FP16
        
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

            # Resize
            frame_resized = cv2.resize(frame, (SHM_WIDTH, SHM_HEIGHT))

            # AI Inference
            processed_frame, fall_count, score = detector.process_frame(frame_resized)
            is_fall = (fall_count > 0)

            # Snapshot Logic
            snapshot_url = current_state["snapshot"]
            if is_fall:
                if score > max_score_in_session or score > 0.8:
                    max_score_in_session = score
                    filename = f"{self.cam_id}_fall_{int(score*100)}.jpg"
                    cv2.imwrite(os.path.join(save_path, filename), processed_frame)
                    snapshot_url = f"/snapshots/{self.cam_id}/{filename}?t={int(time.time())}"
            else:
                if max_score_in_session > 0: max_score_in_session = 0.0
            
            # Write to SHM
            shm_manager.write(processed_frame)

            # Update State
            now_str = datetime.datetime.now().strftime("%H:%M:%S %d/%m")
            new_state = {"fall": is_fall, "snapshot": snapshot_url, "score": float(score), "time": now_str}
            if new_state["fall"] != current_state.get("fall", False) or \
               (is_fall and score > current_state.get("score", 0.0)) or \
               time.time() % 1.0 < 0.05:
                try:
                    if self.state_queue.full(): self.state_queue.get_nowait()
                    self.state_queue.put_nowait(new_state)
                    current_state = new_state
                except: pass

            time.sleep(0.01) # Low latency
        
        stream.stop()
        shm_manager.close()
        print(f"🛑 [{self.cam_id}] Process Stopped.")

# --- GLOBAL VARIABLES ---
# Cấu trúc mới: processes[cam_id] = { "process": obj, "stop_event": evt, "queue": q, "shm": mgr, "lock": lk, "url": str }
system_state = {} 
notification_service = None

# ⚠️ CẤU HÌNH TELEGRAM CỦA BẠN Ở ĐÂY ⚠️
TELEGRAM_TOKEN = "8534838449:AAG0pq4a1uXonmnBshUCot4HQdR9FKp0qCg"
TELEGRAM_CHAT_ID = "8564243388"

# --- WATCHDOG SERVICE ---
async def watchdog_loop():
    """Vòng lặp chạy ngầm kiểm tra sức khỏe các process"""
    print("🐶 Watchdog Service Started!")
    while True:
        try:
            for cam_id, data in system_state.items():
                proc = data["process"]
                
                # Kiểm tra nếu process đã chết (exitcode is not None)
                if not proc.is_alive():
                    exit_code = proc.exitcode
                    print(f"⚠️ ALERT: Camera Process [{cam_id}] died (Exit Code: {exit_code}). Restarting...")
                    
                    # 1. Dọn dẹp process cũ
                    proc.join(timeout=1)
                    
                    # 2. Tạo process mới (Tái sử dụng Queue, Lock, SHM cũ)
                    # Lưu ý: Không cần tạo lại SHM vì Process cha vẫn đang giữ liên kết
                    new_stop_event = mp.Event()
                    new_proc = CameraProcess(
                        cam_id=cam_id,
                        rtsp_url=data["url"],
                        shm_name=data["shm_name"],
                        state_queue=data["queue"],
                        command_event=new_stop_event,
                        lock=data["lock"]
                    )
                    
                    new_proc.start()
                    
                    # 3. Cập nhật lại System State
                    system_state[cam_id]["process"] = new_proc
                    system_state[cam_id]["stop_event"] = new_stop_event
                    print(f"✅ Camera [{cam_id}] restarted successfully! New PID: {new_proc.pid}")
            
        except Exception as e:
            print(f"❌ Watchdog Error: {e}")
        
        # Ngủ 5 giây trước khi check lại
        await asyncio.sleep(5)
        
# Hàm chạy ngầm để check sự kiện
async def event_processor():
    print("🔔 Event Processor (Telegram) Started!")
    while True:
        try:
            for cam_id, item in system_state.items():
                # Kiểm tra xem hàng đợi có tin mới không
                q = item["queue"]
                if not q.empty():
                    try:
                        # Lấy trạng thái mới nhất
                        new_state = q.get_nowait()
                        
                        # Cập nhật vào RAM để Web hiển thị
                        item["last_known_state"] = new_state
                        
                        # LOGIC GỬI TELEGRAM
                        if new_state["fall"] and notification_service:
                            score = new_state.get("score", 0.0)
                            snapshot = new_state.get("snapshot", "")
                            event_time = new_state.get("time", "") # Lấy thời gian từ sự kiện
                            
                            # Gọi module notification kèm theo thời gian
                            notification_service.send_alert(cam_id, snapshot, score, event_time)
                            
                    except Empty:
                        pass
        except Exception as e:
            print(f"❌ Event Loop Error: {e}")
        
        await asyncio.sleep(0.05) # Nghỉ chút để đỡ tốn CPU

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global notification_service
    notification_service = NotificationService(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
    print("📲 Telegram Service Ready!")
    
    # Thiết lập multiprocessing
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    print("🚀 Starting Camera System...")
    
    # 1. Khởi tạo tài nguyên
    for cam_id, url in CAMERAS_CONFIG.items():
        state_q = mp.Queue(maxsize=1)
        stop_event = mp.Event()
        
        # Shared Memory & Lock (Tạo tại cha)
        shm_name = f"shm_{cam_id}"
        shm_mgr = SharedFrameManager(shm_name, SHM_WIDTH, SHM_HEIGHT, create=True)
        # Tạo Lock đa tiến trình
        proc_lock = mp.Lock()
        shm_mgr.lock = proc_lock # Gán lock cho cha dùng

        p = CameraProcess(cam_id, url, shm_name, state_q, stop_event, proc_lock)
        p.start()
        
        # Lưu toàn bộ info cần thiết để restart sau này
        system_state[cam_id] = {
            "process": p,
            "stop_event": stop_event,
            "queue": state_q,
            "shm_mgr": shm_mgr,  # Giữ ref để không bị GC
            "shm_name": shm_name,
            "lock": proc_lock,
            "url": url,
            "last_known_state": {"fall": False, "snapshot": None}
        }
    
    # 2. Bắt đầu Watchdog (Background Task)
    event_task = asyncio.create_task(event_processor())
    watchdog_task = asyncio.create_task(watchdog_loop())

    yield # --- Server Running Here ---
    
    print("🛑 Shutting down...")
    event_task.cancel() # Dừng check telegram
    watchdog_task.cancel() # Dừng watchdog
    
    for cam_id, item in system_state.items():
        item["stop_event"].set()
        item["process"].join(timeout=3)
        if item["process"].is_alive(): item["process"].terminate()
        
        # Cleanup SHM
        item["shm_mgr"].unlink()

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
    for cam_id, item in system_state.items():
        results[cam_id] = item["last_known_state"]
    return results

def frame_generator(cam_id):
    if cam_id not in system_state: return
    
    mgr = system_state[cam_id]["shm_mgr"]
    
    while True:
        # Đọc từ RAM (sẽ dùng Lock của cha)
        frame = mgr.read()
        
        if np.all(frame == 0):
            time.sleep(0.1); continue

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.04) 

@app.get("/video_feed")
def video_feed(cam_id: str):
    if cam_id not in CAMERAS_CONFIG: return HTMLResponse("Not Found", 404)
    return StreamingResponse(frame_generator(cam_id), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)