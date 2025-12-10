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

SEQUENCE_LENGTH = 30
STEP_SIZE = 10 
MODEL_PATH = 'yolo11s-pose.pt'

def get_csv_path(filename):
    """Tìm file CSV ở thư mục gốc hoặc trong Dataset"""
    if os.path.exists(filename):
        return filename
    path_in_dataset = os.path.join(DATASET_ROOT, filename)
    if os.path.exists(path_in_dataset):
        return path_in_dataset
    raise FileNotFoundError(f"❌ Không tìm thấy file {filename} ở cả thư mục gốc và {DATASET_ROOT}")

def load_labels_from_csv():
    labels_dict = {}
    
    # 1. Đọc file Falls
    csv_path = get_csv_path(CSV_FALLS)
    print(f"📖 Đang đọc {csv_path}...")
    df_falls = pd.read_csv(csv_path, header=None)
    for _, row in df_falls.iterrows():
        vid_name = str(row[0]).strip() # fall-01
        frame_id = int(row[1])         # 1, 2, 3...
        label = int(row[2])            # -1, 0, 1
        labels_dict[(vid_name, frame_id)] = label

    # 2. Đọc file ADLs (Mặc định ADL là Normal=0)
    csv_path = get_csv_path(CSV_ADLS)
    print(f"📖 Đang đọc {csv_path}...")
    df_adls = pd.read_csv(csv_path, header=None)
    for _, row in df_adls.iterrows():
        vid_name = str(row[0]).strip()
        frame_id = int(row[1])
        labels_dict[(vid_name, frame_id)] = 0 
        
    return labels_dict

def parse_info_from_filename(filename):
    """
    Phân tích tên file để lấy VideoID và FrameID.
    Input:  "fall-01-cam0-rgb-001.png"
    Output: ("fall-01", 1)
    """
    try:
        # Loại bỏ đuôi file (.png)
        base_name = os.path.splitext(filename)[0] # fall-01-cam0-rgb-001
        parts = base_name.split('-')
        
        # Format chuẩn UR Fall: category-id-cam0-rgb-frame
        # parts: ['fall', '01', 'cam0', 'rgb', '001']
        if len(parts) >= 2:
            # Lấy 2 phần đầu làm VideoID (fall-01, adl-01)
            video_name = f"{parts[0]}-{parts[1]}"
            
            # Lấy phần cuối làm FrameID
            frame_part = parts[-1] 
            if frame_part.isdigit():
                return video_name, int(frame_part)
                
    except Exception as e:
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
            nx = (kx - cx) / w
            ny = (ky - cy) / h
            normalized.extend([nx, ny])
    return normalized

def fill_missing_keypoints(sequence_data):
    df = pd.DataFrame(sequence_data)
    df.replace(0.0, np.nan, inplace=True)
    df = df.interpolate(method='linear', limit_direction='both')
    df.fillna(0.0, inplace=True)
    return df.values

def create_dataset_with_csv():
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ LỖI: Không tìm thấy thư mục '{DATASET_ROOT}'")
        return

    try:
        labels_lookup = load_labels_from_csv()
    except Exception as e:
        print(e)
        return

    model = YOLO(MODEL_PATH).to(DEVICE)
    
    all_sequences = []
    all_labels = []

    categories = ['fall', 'adl']
    
    for cat in categories:
        cat_path = os.path.join(DATASET_ROOT, cat)
        if not os.path.exists(cat_path): continue
        
        sequence_folders = sorted([f.path for f in os.scandir(cat_path) if f.is_dir()])
        print(f"\n--- Processing Category: {cat} ---")
        
        # Biến debug để in ra kiểm tra 1 lần đầu tiên
        debug_first_file = True

        for seq_folder in tqdm(sequence_folders):
            image_files = sorted(glob.glob(os.path.join(seq_folder, "*.png")))
            if not image_files:
                image_files = sorted(glob.glob(os.path.join(seq_folder, "*.jpg")))
            
            if len(image_files) < SEQUENCE_LENGTH: continue

            frame_data_list = []
            
            for img_path in image_files:
                fname = os.path.basename(img_path)
                
                # --- SỬ DỤNG HÀM PARSE TỪ TÊN FILE ---
                video_name, frame_id = parse_info_from_filename(fname)
                
                if video_name is None:
                    continue

                # --- DEBUG: In ra để kiểm tra xem parse đúng không ---
                if debug_first_file and cat == 'fall':
                    print(f"\n🔍 DEBUG CHECK (File đầu tiên):")
                    print(f"   - Filename:  {fname}")
                    print(f"   - Parsed ID: {video_name}, Frame: {frame_id}")
                    label_check = labels_lookup.get((video_name, frame_id), -1)
                    print(f"   - Label CSV: {label_check}")
                    debug_first_file = False

                # Tra cứu nhãn
                csv_label = labels_lookup.get((video_name, frame_id), -1)
                
                final_label = -1
                if cat == 'fall':
                    if csv_label == 1: final_label = 1   # FALL
                    elif csv_label == 0: final_label = 0 # NORMAL
                else: # adl
                    final_label = 0 # Luôn coi ADL là normal
                
                if final_label != -1:
                    frame = cv2.imread(img_path)
                    if frame is None: continue
                    
                    results = model(frame, verbose=False, classes=[0])[0]
                    kps_vector = [0.0] * 34
                    
                    if results.keypoints and len(results.keypoints) > 0:
                        kps = results.keypoints.data[0].cpu().numpy()
                        box = results.boxes.xyxy[0].cpu().numpy()
                        kps_vector = normalize_keypoints(kps, box)
                    
                    frame_data_list.append({
                        "kps": kps_vector,
                        "label": final_label
                    })

            # Sliding Window
            if len(frame_data_list) < SEQUENCE_LENGTH: continue
            
            full_kps_sequence = [item["kps"] for item in frame_data_list]
            full_labels = [item["label"] for item in frame_data_list]
            
            filled_kps = fill_missing_keypoints(full_kps_sequence)
            
            for i in range(0, len(filled_kps) - SEQUENCE_LENGTH, STEP_SIZE):
                window_kps = filled_kps[i : i + SEQUENCE_LENGTH]
                window_labels = full_labels[i : i + SEQUENCE_LENGTH]
                
                fall_count = window_labels.count(1)
                
                # Logic gán nhãn cho chuỗi:
                # Nếu > 10 frame là Fall -> Chuỗi là Fall
                seq_label = 1 if fall_count >= 10 else 0
                
                all_sequences.append(window_kps)
                all_labels.append(seq_label)

    X = np.array(all_sequences)
    y = np.array(all_labels)
    
    print(f"\n📊 KẾT QUẢ MỚI:")
    if len(X) == 0:
        print("❌ Dataset rỗng! Vẫn chưa khớp được tên file và CSV.")
    else:
        print(f"✅ Tạo thành công: {len(X)} mẫu.")
        print(f"   - Normal (0): {np.sum(y==0)}")
        print(f"   - Fall (1):   {np.sum(y==1)}")
        
        dataset_kps_dir = "./data_kps"
        os.makedirs(dataset_kps_dir, exist_ok=True)
        np.save(os.path.join(dataset_kps_dir, "X_data.npy"), X)
        np.save(os.path.join(dataset_kps_dir, "y_data.npy"), y)
        print("💾 Đã lưu file .npy mới")

if __name__ == "__main__":
    create_dataset_with_csv()