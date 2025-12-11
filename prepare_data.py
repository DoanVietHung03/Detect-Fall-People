import cv2
import numpy as np
import os
import glob
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

from config import DEVICE

# --- CẤU HÌNH ---
DATASET_ROOT = "Dataset"
CSV_FALLS = "urfall-cam0-falls.csv"
CSV_ADLS = "urfall-cam0-adls.csv"

SEQUENCE_LENGTH = 15
STEP_SIZE = 5  # Giảm step size để lấy được nhiều mẫu ngã hơn
MODEL_PATH = 'yolo11s-pose.pt'

def get_csv_path(filename):
    """Tìm file CSV ở thư mục gốc hoặc trong Dataset"""
    if os.path.exists(filename):
        return filename
    path_in_dataset = os.path.join(DATASET_ROOT, filename)
    if os.path.exists(path_in_dataset):
        return path_in_dataset
    # Nếu không thấy, trả về None để xử lý sau
    return None

def load_labels_from_csv():
    labels_dict = {}
    
    # 1. Đọc file Falls
    csv_path = get_csv_path(CSV_FALLS)
    if csv_path:
        print(f"📖 Đang đọc {csv_path}...")
        df_falls = pd.read_csv(csv_path, header=None)
        for _, row in df_falls.iterrows():
            vid_name = str(row[0]).strip()
            frame_id = int(row[1])
            label = int(row[2])
            labels_dict[(vid_name, frame_id)] = label
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy {CSV_FALLS}")

    # 2. Đọc file ADLs
    csv_path = get_csv_path(CSV_ADLS)
    if csv_path:
        print(f"📖 Đang đọc {csv_path}...")
        df_adls = pd.read_csv(csv_path, header=None)
        for _, row in df_adls.iterrows():
            vid_name = str(row[0]).strip()
            frame_id = int(row[1])
            labels_dict[(vid_name, frame_id)] = 0 
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy {CSV_ADLS}")
        
    return labels_dict

def parse_info_from_filename(filename):
    """Parse tên file: fall-01-cam0-rgb-001.png -> ('fall-01', 1)"""
    try:
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('-')
        if len(parts) >= 2:
            video_name = f"{parts[0]}-{parts[1]}"
            frame_part = parts[-1] 
            if frame_part.isdigit():
                return video_name, int(frame_part)
    except:
        return None, None
    return None, None

def normalize_keypoints(keypoints, box):
    x1, y1, x2, y2 = box
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    normalized = []
    for kp in keypoints:
        kx, ky = kp[0], kp[1]
        if kx == 0 and ky == 0:
            normalized.extend([0.0, 0.0])
        else:
            normalized.extend([(kx - cx) / w, (ky - cy) / h])
    return normalized

def create_dataset_balanced():
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ LỖI: Không tìm thấy thư mục '{DATASET_ROOT}'")
        return

    labels_lookup = load_labels_from_csv()
    model = YOLO(MODEL_PATH).to(DEVICE)
    
    # Tách riêng 2 danh sách để dễ cân bằng sau này
    fall_sequences = []
    normal_sequences = []

    categories = ['fall', 'adl']
    
    for cat in categories:
        cat_path = os.path.join(DATASET_ROOT, cat)
        if not os.path.exists(cat_path): continue
        
        sequence_folders = sorted([f.path for f in os.scandir(cat_path) if f.is_dir()])
        print(f"\n--- Processing Category: {cat} ---")

        for seq_folder in tqdm(sequence_folders):
            image_files = sorted(glob.glob(os.path.join(seq_folder, "*.png")))
            if not image_files:
                image_files = sorted(glob.glob(os.path.join(seq_folder, "*.jpg")))
            
            if len(image_files) < SEQUENCE_LENGTH: continue

            # 1. Trích xuất Pose từng frame
            frame_data_list = []
            for img_path in image_files:
                fname = os.path.basename(img_path)
                video_name, frame_id = parse_info_from_filename(fname)
                if video_name is None: continue

                csv_label = labels_lookup.get((video_name, frame_id), -1)
                
                final_label = -1
                if cat == 'fall':
                    if csv_label == 1: final_label = 1
                    elif csv_label == 0: final_label = 0
                else: # adl
                    final_label = 0 
                
                if final_label != -1:
                    frame = cv2.imread(img_path)
                    if frame is None: continue
                    
                    results = model(frame, verbose=False, classes=[0])[0]
                    kps_vector = [0.0] * 34
                    
                    if results.keypoints and len(results.keypoints) > 0:
                        kps = results.keypoints.data[0].cpu().numpy()
                        box = results.boxes.xyxy[0].cpu().numpy()
                        kps_vector = normalize_keypoints(kps, box)
                    
                    frame_data_list.append({"kps": kps_vector, "label": final_label})

            # 2. Tạo Sequence (Sliding Window)
            if len(frame_data_list) < SEQUENCE_LENGTH: continue
            
            # Fill missing data
            df = pd.DataFrame([x['kps'] for x in frame_data_list])
            df.replace(0.0, np.nan, inplace=True)
            df = df.interpolate(method='linear', limit_direction='both')
            df.fillna(0.0, inplace=True)
            filled_kps = df.values
            full_labels = [x['label'] for x in frame_data_list]
            
            for i in range(0, len(filled_kps) - SEQUENCE_LENGTH, STEP_SIZE):
                window_kps = filled_kps[i : i + SEQUENCE_LENGTH]
                window_labels = full_labels[i : i + SEQUENCE_LENGTH]
                
                fall_count = window_labels.count(1)
                
                # Logic phân loại:
                if fall_count >= 5: # Chỉ cần 5 frame ngã là tính là chuỗi ngã (để bắt nhạy hơn)
                    fall_sequences.append(window_kps)
                elif fall_count == 0: # Chỉ lấy normal sạch (không dính tí ngã nào)
                    normal_sequences.append(window_kps)

    # --- BƯỚC QUAN TRỌNG: CÂN BẰNG DỮ LIỆU (DATA BALANCING) ---
    n_fall = len(fall_sequences)
    n_normal = len(normal_sequences)
    print(f"\n📊 Thống kê trước khi cân bằng: Fall={n_fall}, Normal={n_normal}")

    if n_fall == 0:
        print("❌ Vẫn không có mẫu ngã nào. Kiểm tra lại CSV!")
        # Vẫn lưu Normal để debug
    elif n_normal > n_fall:
        factor = n_normal // n_fall
        print(f"⚖️ Đang nhân bản dữ liệu Fall lên {factor} lần...")
        fall_sequences = fall_sequences * factor
        # Cộng thêm phần dư
        rem = n_normal - len(fall_sequences)
        if rem > 0: fall_sequences.extend(fall_sequences[:rem])

    # Gộp lại
    X = np.array(fall_sequences + normal_sequences)
    # Tạo nhãn: Fall=1, Normal=0
    y = np.array([1] * len(fall_sequences) + [0] * len(normal_sequences))
    
    # Shuffle (Trộn đều)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    print(f"✅ KẾT QUẢ CUỐI CÙNG: {len(X)} mẫu.")
    print(f"   - Normal (0): {np.sum(y==0)}")
    print(f"   - Fall (1):   {np.sum(y==1)}")
        
    dataset_kps_dir = "./data_kps"
    os.makedirs(dataset_kps_dir, exist_ok=True)
    np.save(os.path.join(dataset_kps_dir, "X_data.npy"), X)
    np.save(os.path.join(dataset_kps_dir, "y_data.npy"), y)
    print("💾 Đã lưu file .npy mới")

if __name__ == "__main__":
    create_dataset_balanced()