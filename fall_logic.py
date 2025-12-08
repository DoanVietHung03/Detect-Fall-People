import cv2
import torch
import numpy as np
import os
import math
from ultralytics import YOLO
from supervision import ByteTrack, BoxAnnotator, LabelAnnotator, Detections, Color, ColorPalette

class FallDetector:
    def __init__(self, model_path='yolo11s-pose.pt', conf_threshold=0.8, fall_ratio=3.0):
        self.conf_threshold = conf_threshold
        self.fall_ratio_threshold = fall_ratio
        
        # Setup Device
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.device_name = torch.cuda.get_device_name(0) if self.device == 0 else "CPU"
        
        # Load Model
        self.model = YOLO(model_path).to(self.device)
        if self.device == 0:
            self.model.to('cuda')

        # Tracker (Giữ cấu hình ổn định như trước)
        self.tracker = ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=60, 
            frame_rate=30
        )
 
        # Annotators
        self.box_annotator_normal = BoxAnnotator(color=ColorPalette([Color.GREEN]))
        self.label_annotator_normal = LabelAnnotator(text_color=Color.BLACK)
        
        self.box_annotator_fall = BoxAnnotator(color=ColorPalette([Color.RED]), thickness=4)
        self.label_annotator_fall = LabelAnnotator(text_color=Color.WHITE)

        # --- LOGIC MỚI: QUẢN LÝ ĐIỂM SỐ CAO NHẤT ---
        self.snapshot_dir = "snapshots"
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Dictionary lưu trạng thái: { track_id: best_ratio_so_far }
        self.fall_best_scores = {} 

    def calculate_aspect_ratio(self, xyxy):
        width = xyxy[2] - xyxy[0]
        height = xyxy[3] - xyxy[1]
        return width / height

    def process_frame(self, frame):
        # 1. Detect
        results = self.model(frame, verbose=False, conf=self.conf_threshold, device=self.device, classes=[0])[0]
        detections = Detections.from_ultralytics(results)

        # 2. Update Tracker
        detections = self.tracker.update_with_detections(detections)

        # 3. Phân loại
        fall_indices = []
        normal_indices = []
        
        current_fall_ids = [] # Danh sách ID đang ngã trong frame này

        for i, (xyxy, track_id) in enumerate(zip(detections.xyxy, detections.tracker_id)):
            ratio = self.calculate_aspect_ratio(xyxy)
            
            if ratio > self.fall_ratio_threshold:
                fall_indices.append(i)
                current_fall_ids.append(track_id)
                
                # --- LOGIC UPDATE SNAPSHOT ---
                # Kiểm tra xem đây có phải là khoảnh khắc ngã "rõ nhất" từ trước tới giờ không
                old_best_score = self.fall_best_scores.get(track_id, 0.0)
                
                if ratio > old_best_score:
                    # Cập nhật điểm kỷ lục mới
                    self.fall_best_scores[track_id] = ratio
                    
                    # Lưu (Ghi đè) ảnh snapshot
                    # Tên file cố định theo ID -> Luôn chỉ có 1 file cho 1 người
                    filename = f"fall_evidence_ID_{track_id}.jpg"
                    save_path = os.path.join(self.snapshot_dir, filename)
                    
                    # Vẽ riêng frame này để lưu (cho đẹp)
                    evidence_frame = frame.copy()
                    cv2.rectangle(evidence_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 4)
                    cv2.putText(evidence_frame, f"FALL DETECTED - Ratio: {ratio:.2f}", (int(xyxy[0]), int(xyxy[1])-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    
                    cv2.imwrite(save_path, evidence_frame)
            else:
                normal_indices.append(i)

        # 4. Dọn dẹp bộ nhớ (Reset logic)
        # Nếu ID nào từng nằm trong danh sách theo dõi nhưng giờ KHÔNG còn ngã nữa (đã đứng dậy hoặc đi mất)
        # Thì xóa khỏi bộ nhớ để lần sau ngã lại sẽ tính là vụ việc mới.
        active_keys = list(self.fall_best_scores.keys())
        for k in active_keys:
            if k not in current_fall_ids:
                del self.fall_best_scores[k]

        # 5. Vẽ hiển thị (Visual cho Streamlit)
        annotated_frame = frame.copy()
        
        detections_fall = detections[np.array(fall_indices)] if fall_indices else Detections.empty()
        detections_normal = detections[np.array(normal_indices)] if normal_indices else Detections.empty()

        if len(detections_normal) > 0:
            annotated_frame = self.box_annotator_normal.annotate(annotated_frame, detections_normal)
            labels_normal = [f"ID:{t_id}" for t_id in detections_normal.tracker_id]
            annotated_frame = self.label_annotator_normal.annotate(annotated_frame, detections_normal, labels=labels_normal)

        if len(detections_fall) > 0:
            annotated_frame = self.box_annotator_fall.annotate(annotated_frame, detections_fall)
            # Hiển thị luôn ratio lên box để debug
            labels_fall = [f"FALL! Ratio:{self.calculate_aspect_ratio(box):.2f}" for box in detections_fall.xyxy]
            annotated_frame = self.label_annotator_fall.annotate(annotated_frame, detections_fall, labels=labels_fall)

        # Trả về đường dẫn thư mục snapshot để App tự load ảnh mới nhất
        return annotated_frame, len(fall_indices), self.snapshot_dir