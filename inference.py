# 3_inference.py (Thay thế rule_based.py)
import cv2
import numpy as np
import os
from collections import deque
import torch
import torch.nn as nn
from ultralytics import YOLO
from supervision import ByteTrack, Detections, BoxAnnotator, LabelAnnotator, ColorPalette, Color

from config import DEVICE
from train_lstm import FallLSTM

class FallDetector:
    def __init__(self, model_pose='weights/yolo11s-pose.pt', model_lstm='weights/lstm_fall_model.pth', conf_threshold=0.8, lstm_threshold=0.75):
        self.conf_threshold = conf_threshold
        self.lstm_threshold = lstm_threshold
        
        # Load Models
        print("Loading YOLO...")
        self.pose_model = YOLO(model_pose).to(DEVICE)
        print("Loading LSTM...")
        self.lstm_model = FallLSTM().to(DEVICE)
        self.lstm_model.load_state_dict(torch.load(model_lstm, map_location=DEVICE))
        self.lstm_model.eval()
        
        # Tracker
        self.tracker = ByteTrack(track_activation_threshold=0.2, lost_track_buffer=30)
        
        # Visualization Tools
        # 1. Bộ Normal (Màu Xanh)
        self.box_annotator_green = BoxAnnotator(color=ColorPalette([Color.GREEN]), thickness=2)
        self.label_annotator_green = LabelAnnotator(text_color=Color.BLACK, text_scale=0.5)
        
        # 2. Bộ Fall (Màu Đỏ, nét đậm hơn)
        self.box_annotator_red = BoxAnnotator(color=ColorPalette([Color.RED]), thickness=2)
        self.label_annotator_red = LabelAnnotator(text_color=Color.WHITE, text_scale=0.5)

        # QUẢN LÝ TRẠNG THÁI (State Management)
        # Dictionary lưu lịch sử keypoints cho từng ID: { track_id: deque(maxlen=30) }
        self.track_history = {} 
        self.SEQUENCE_LENGTH = 30
        
        # Quản lý việc hiển thị cảnh báo (giữ cảnh báo trong vài giây để dễ nhìn)
        self.alert_buffer = {} 

        self.snapshot_dir = "snapshots"
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Lưu trữ pose cuối cùng hợp lệ của mỗi ID để fill khi mất dấu
        self.last_valid_pose = {}

    def normalize_keypoints(self, keypoints, box):
        """Logic chuẩn hóa GIỐNG HỆT lúc train"""
        x1, y1, x2, y2 = box
        w = max(x2 - x1, 1e-6) # Bảo vệ chia cho 0
        h = max(y2 - y1, 1e-6)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        normalized = []
        for kp in keypoints:
            kx, ky = kp[0], kp[1]
            nx = (kx - cx) / w
            ny = (ky - cy) / h
            normalized.extend([nx, ny])
        return normalized

    def process_frame(self, frame):
        # 1. Detect Pose & Track
        results = self.pose_model(frame, verbose=False, conf=self.conf_threshold, device=DEVICE, classes=[0])[0]
        detections = Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        # Prepare lists for annotation
        fall_indices = []
        normal_indices = []
        
        labels_fall = []
        labels_normal = []

        current_tracked_ids = []

        # --- XỬ LÝ DỮ LIỆU TỪ GPU VỀ CPU (FIX LỖI CRASH) ---
        # Lấy data gốc từ YOLO để khớp keypoints
        yolo_boxes_cpu = []
        yolo_kps_cpu = []
        
        if results.boxes is not None and results.keypoints is not None:
            # Chuyển Tensor -> Numpy ngay tại đây để tính toán khoảng cách
            yolo_boxes_cpu = results.boxes.xyxy.cpu().numpy()
            yolo_kps_cpu = results.keypoints.data.cpu().numpy()

        # 2. DUYỆT QUA TỪNG ID ĐANG ĐƯỢC TRACK
        for i, (track_box, track_id) in enumerate(zip(detections.xyxy, detections.tracker_id)):
            current_tracked_ids.append(track_id)
            
            # Tạo buffer lịch sử nếu là ID mới
            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=self.SEQUENCE_LENGTH)
            
            # --- MATCHING: Tìm keypoints khớp với box tracker ---
            # Tracker box có thể hơi lệch so với detection mới nhất, nên cần tìm box gần nhất
            track_center = ((track_box[0] + track_box[2]) / 2, (track_box[1] + track_box[3]) / 2)
            matched_kps = None
            min_dist = 100 # Pixel
            
            for box_orig, kps_orig in zip(yolo_boxes_cpu, yolo_kps_cpu):
                orig_center = ((box_orig[0] + box_orig[2]) / 2, (box_orig[1] + box_orig[3]) / 2)
                
                # Tính khoảng cách Euclidean
                dist = np.hypot(track_center[0] - orig_center[0], track_center[1] - orig_center[1])
                
                if dist < 50: # Nếu tâm 2 box cách nhau < 50px thì coi là cùng 1 người
                    if dist < min_dist:
                        min_dist = dist
                        matched_kps = kps_orig
            
            # --- XỬ LÝ MẤT DẤU (Missing Joints Handling) ---
            final_pose_vector = None

            if matched_kps is not None:
                # 1. Có keypoints -> Chuẩn hóa & Lưu làm Last Valid
                norm_kps = self.normalize_keypoints(matched_kps, track_box)
                self.last_valid_pose[track_id] = norm_kps # Update backup
                final_pose_vector = norm_kps
            else:
                # 2. Mất keypoints (YOLO fail nhưng Tracker vẫn giữ ID)
                # -> Dùng lại Pose cũ nhất (Forward Fill)
                if track_id in self.last_valid_pose:
                     # Copy pose cũ để tránh tham chiếu bộ nhớ
                    final_pose_vector = list(self.last_valid_pose[track_id]) 
                else:
                    # Trường hợp xấu nhất: Mới xuất hiện đã mất dấu -> Fill 0
                    final_pose_vector = [0.0] * 34

            # Đẩy vào hàng đợi lịch sử
            self.track_history[track_id].append(final_pose_vector)

            # --- LOGIC DỰ ĐOÁN (INFERENCE) ---
            is_fall_detected_now = False
            prob_text = ""

            if matched_kps is not None and len(self.track_history[track_id]) == self.SEQUENCE_LENGTH:
                # 1. Chuẩn bị Input Tensor
                sequence = np.array([self.track_history[track_id]]) # Shape (1, 15, 34)
                input_tensor = torch.tensor(sequence, dtype=torch.float32).to(DEVICE)
                
                # 2. Inference
                with torch.no_grad():
                    output = self.lstm_model(input_tensor) # Logits
                    probs = torch.softmax(output, dim=1)   # Probability
                    fall_prob = probs[0, 1].item()         # Lấy xác suất lớp 1 (Fall)
                
                if fall_prob > self.lstm_threshold:
                    is_fall_detected_now = True
                    prob_text = f"({fall_prob:.2f})"

            # --- QUẢN LÝ TRẠNG THÁI CẢNH BÁO (DEBOUNCE) ---
            # Nếu phát hiện ngã -> Bật bộ đếm ngược 40 frames (giữ màu đỏ khoảng 1.5s)
            if is_fall_detected_now:
                self.alert_buffer[track_id] = 40
            
            # Kiểm tra xem ID này có đang trong trạng thái báo động không
            if self.alert_buffer.get(track_id, 0) > 0:
                self.alert_buffer[track_id] -= 1
                
                # Đây là người đang ngã -> Thêm vào danh sách ĐỎ
                fall_indices.append(i)
                labels_fall.append(f"ID:{track_id} FALL! {prob_text}")
                
                # Chụp ảnh bằng chứng (chỉ chụp ở frame đầu tiên của sự kiện)
                if self.alert_buffer[track_id] == 39:
                    # 1. Tạo bản sao để vẽ (không làm bẩn frame gốc đang xử lý)
                    snapshot_frame = frame.copy()
                    
                    # 2. Lấy tọa độ box (ép kiểu về int)
                    x1, y1, x2, y2 = int(track_box[0]), int(track_box[1]), int(track_box[2]), int(track_box[3])
                    
                    # 3. Vẽ Bounding Box (Màu đỏ)
                    cv2.rectangle(snapshot_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    # 4. Vẽ Nhãn (Nền đỏ chữ trắng)
                    label = f"ID:{track_id} FALL! {prob_text}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(snapshot_frame, (x1, y1 - 30), (x1 + w, y1), (0, 0, 255), -1)
                    cv2.putText(snapshot_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    # 5. Lưu ảnh đã vẽ
                    img_name = f"fall_evidence_ID_{track_id}.jpg"
                    cv2.imwrite(os.path.join(self.snapshot_dir, img_name), snapshot_frame)
            
            else:
                # Người bình thường -> Thêm vào danh sách XANH
                normal_indices.append(i)
                labels_normal.append(f"ID:{track_id}")

        # Dọn dẹp memory cho các ID đã ra khỏi màn hình
        active_keys = list(self.track_history.keys())
        for k in active_keys:
            if k not in current_tracked_ids:
                del self.track_history[k]
                if k in self.alert_buffer: del self.alert_buffer[k]
                if k in self.last_valid_pose: del self.last_valid_pose[k]

        # --- VẼ LÊN HÌNH (ANNOTATION) ---
        annotated_frame = frame.copy()
        
        # 1. Vẽ nhóm người bình thường
        if len(normal_indices) > 0:
            det_normal = detections[np.array(normal_indices)]
            annotated_frame = self.box_annotator_green.annotate(annotated_frame, det_normal)
            annotated_frame = self.label_annotator_green.annotate(annotated_frame, det_normal, labels=labels_normal)

        # 2. Vẽ nhóm người ngã (Vẽ sau để đè lên trên)
        if len(fall_indices) > 0:
            det_fall = detections[np.array(fall_indices)]
            annotated_frame = self.box_annotator_red.annotate(annotated_frame, det_fall)
            annotated_frame = self.label_annotator_red.annotate(annotated_frame, det_fall, labels=labels_fall)

        return annotated_frame, len(fall_indices), self.snapshot_dir