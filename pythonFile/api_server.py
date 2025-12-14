# api_server.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import cv2
import uvicorn
import os
import glob
import time
from pydantic import BaseModel
from inference import FallDetector
from typing import List

app = FastAPI()

# --- CẤU HÌNH ---
SNAPSHOT_DIR = "../snapshots"
if not os.path.exists(SNAPSHOT_DIR): os.makedirs(SNAPSHOT_DIR)

# Mount thư mục ảnh để Dashboard có thể xem qua URL
app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")

# --- GLOBAL STATE ---
current_settings = {
    "conf": 0.7,
    "lstm": 0.7,
    "fall_detected_now": False 
}

# Model dữ liệu cho vùng an toàn
class ZoneConfig(BaseModel):
    zones: List[List[List[int]]]

# Khởi tạo model
print("⏳ Đang tải model AI...")
detector = FallDetector(
    model_pose='../weights/yolo11m-pose.pt',
    model_lstm='../weights/lstm_fall_model.pth'
)
print("✅ AI đã sẵn sàng!")

class Settings(BaseModel):
    conf: float
    lstm: float
    
@app.post("/update_zones")
def update_zones(config: ZoneConfig):
    try:
        new_zones = []
        for polygon in config.zones:
            pts = np.array(polygon, np.int32)
            new_zones.append(pts)
        detector.set_safe_zones(new_zones)
        return {"status": "success", "count": len(new_zones)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/update_settings")
def update_settings(settings: Settings):
    current_settings["conf"] = settings.conf
    current_settings["lstm"] = settings.lstm
    detector.conf_threshold = settings.conf
    detector.lstm_threshold = settings.lstm
    print(f"🔄 Updated: Conf={settings.conf}, LSTM={settings.lstm}")
    return {"status": "updated"}

@app.get("/status")
def get_status():
    return {
        "fall_detected": current_settings["fall_detected_now"]
    }

@app.get("/gallery")
def get_gallery(video_name: str):
    specific_dir = os.path.join(SNAPSHOT_DIR, video_name)
    if not os.path.exists(specific_dir):
        return {"images": []}
    
    # Lấy danh sách ảnh, sắp xếp mới nhất lên đầu (theo thời gian sửa đổi)
    files = sorted(glob.glob(os.path.join(specific_dir, "*.jpg")), key=os.path.getmtime, reverse=True)
    
    rel_paths = [os.path.join(video_name, os.path.basename(f)).replace("\\", "/") for f in files[:6]]
    return {"images": rel_paths}

# --- HÀM LƯU ẢNH ---
def save_evidence(frame, score, folder_path, prefix="fall"):
    if frame is None: return
    timestamp = int(time.time())
    # Tạo tên file bao gồm score để dễ debug
    filename = f"{prefix}_{int(score*100)}conf_{timestamp}.jpg"
    full_path = os.path.join(folder_path, filename)
    cv2.imwrite(full_path, frame)
    print(f"📸 Saved Evidence: {full_path} (Score: {score:.2f})")

# --- LOGIC XỬ LÝ VIDEO ---
def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # 1. Tạo thư mục lưu ảnh riêng cho video này
    video_filename = os.path.basename(video_path)
    video_name_only = os.path.splitext(video_filename)[0]
    save_path = os.path.join(SNAPSHOT_DIR, video_name_only)
    if not os.path.exists(save_path): os.makedirs(save_path)

    best_frame = None       
    max_score = 0.0         
    is_falling_sequence = False 
    
    # Biến để kiểm soát việc lưu (tránh lưu quá nhiều trùng lặp)
    last_saved_time = 0 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break # Video kết thúc
        
        # Resize nhẹ 
        frame = cv2.resize(frame, (640, 480))
        
        # --- GỌI AI ---
        processed_frame, fall_count, score = detector.process_frame(frame)
        current_settings["fall_detected_now"] = (fall_count > 0)

        # --- LOGIC BEST SHOT (ĐÃ SỬA) ---
        if fall_count > 0:
            current_time = time.time()
            
            # Nếu là bắt đầu sequence mới
            if not is_falling_sequence:
                is_falling_sequence = True
                max_score = 0.0
                best_frame = None
                print("⚠️ Fall Started - Tracking best shot...")

            # Cập nhật khung hình tốt nhất nếu điểm cao hơn
            if score >= max_score:
                max_score = score
                best_frame = processed_frame.copy()
                
                # OPTIONAL: Lưu ngay lập tức nếu score rất cao (>0.85) để hiển thị ngay trên Dashboard
                # Thay vì chờ ngã xong mới hiện.
                if max_score > 0.85 and (current_time - last_saved_time > 1.0):
                    save_evidence(best_frame, max_score, save_path)
                    last_saved_time = current_time

        else:
            # Người đã đứng dậy hoặc hết ngã
            if is_falling_sequence:
                print("✅ Fall Sequence Ended. Saving final best shot.")
                # Lưu cái tốt nhất còn lại trong sequence
                save_evidence(best_frame, max_score, save_path)
                
                # Reset
                is_falling_sequence = False
                best_frame = None
                max_score = 0.0

        # Encode frame gửi về Client
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # --- QUAN TRỌNG: XỬ LÝ KHI LOOP KẾT THÚC (Video hết) ---
    # Nếu video hết mà vẫn đang trong trạng thái ngã -> LƯU NGAY
    if is_falling_sequence and best_frame is not None:
        print("⏹️ Video Ended during fall. Saving pending evidence.")
        save_evidence(best_frame, max_score, save_path)

    cap.release()

@app.get("/video_feed")
def video_feed(video_path: str):
    return StreamingResponse(
        generate_frames(video_path), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import numpy as np # Import thêm ở đây nếu chưa có global import
    uvicorn.run(app, host="0.0.0.0", port=8000)