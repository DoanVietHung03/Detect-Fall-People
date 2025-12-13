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

app = FastAPI()

# --- CẤU HÌNH ---
SNAPSHOT_DIR = "snapshots"
if not os.path.exists(SNAPSHOT_DIR): os.makedirs(SNAPSHOT_DIR)

# Mount thư mục ảnh để Dashboard có thể xem qua URL
app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")

# --- GLOBAL STATE ---
current_settings = {
    "conf": 0.7,
    "lstm": 0.7,
    "fall_detected_now": False # Trạng thái tức thời để dashboard cảnh báo
}

# Khởi tạo model
print("⏳ Đang tải model AI...")
detector = FallDetector(
    model_pose='weights/yolo11m-pose.pt',
    model_lstm='weights/lstm_fall_model.pth'
)
print("✅ AI đã sẵn sàng!")

class Settings(BaseModel):
    conf: float
    lstm: float

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
        "fall_detected": current_settings["fall_detected_now"],
        "is_running": True
    }

@app.get("/gallery")
def get_gallery(video_name: str):
    """API trả về ảnh của riêng video đó"""
    specific_dir = os.path.join(SNAPSHOT_DIR, video_name)
    if not os.path.exists(specific_dir):
        return {"images": []}
    
    # Lấy danh sách ảnh, sắp xếp mới nhất lên đầu
    files = sorted(glob.glob(os.path.join(specific_dir, "*.jpg")), key=os.path.getmtime, reverse=True)
    
    # Trả về đường dẫn tương đối: video_name/anh.jpg
    # Chỉ lấy 4 ảnh mới nhất để Dashboard đỡ lag
    rel_paths = [os.path.join(video_name, os.path.basename(f)).replace("\\", "/") for f in files[:4]]
    return {"images": rel_paths}

# --- LOGIC XỬ LÝ VIDEO & BEST SHOT ---
def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # 1. Tạo thư mục lưu ảnh riêng cho video này
    video_filename = os.path.basename(video_path)
    video_name_only = os.path.splitext(video_filename)[0]
    save_path = os.path.join(SNAPSHOT_DIR, video_name_only)
    if not os.path.exists(save_path): os.makedirs(save_path)

    # 2. Biến theo dõi logic "Best Shot"
    best_frame = None       
    max_score = 0.0         
    is_falling_sequence = False 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Resize nhẹ 
        frame = cv2.resize(frame, (640, 480))
        
        # --- GỌI AI ---
        # Nhận về: Frame đã vẽ, số người ngã, và ĐIỂM SỐ (score)
        processed_frame, fall_count, score = detector.process_frame(frame)
        
        # Cập nhật trạng thái global cho Dashboard biết ngay lập tức
        current_settings["fall_detected_now"] = (fall_count > 0)

        # --- LOGIC TÌM ẢNH TỐT NHẤT (BEST SHOT) ---
        if fall_count > 0:
            # A. ĐANG TRONG QUÁ TRÌNH NGÃ
            is_falling_sequence = True
            
            # Nếu khung hình này rõ hơn (score cao hơn) -> Lưu tạm vào RAM
            if score >= max_score:
                max_score = score
                best_frame = processed_frame.copy() 
            
        else:
            # B. HẾT NGÃ (Hoặc người vừa đứng dậy / chuyển cảnh)
            if is_falling_sequence:
                # Kết thúc sự kiện -> LƯU ẢNH TỐT NHẤT xuông ổ cứng
                if best_frame is not None:
                    timestamp = int(time.time())
                    filename = f"fall_{int(max_score*100)}conf_{timestamp}.jpg"
                    full_path = os.path.join(save_path, filename)
                    
                    cv2.imwrite(full_path, best_frame)
                    print(f"📸 Saved Evidence: {full_path} (Score: {max_score:.2f})")
                
                # Reset biến để chờ cú ngã tiếp theo
                best_frame = None
                max_score = 0.0
                is_falling_sequence = False

        # Encode frame gửi về Client
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.get("/video_feed")
def video_feed(video_path: str):
    return StreamingResponse(
        generate_frames(video_path), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)