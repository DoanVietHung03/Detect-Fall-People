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
        
        # Ngưỡng góc: Dưới 45 độ so với mặt đất được coi là nằm ngang
        self.fall_angle_threshold = 15.0  
        
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        
        self.model = YOLO(model_path).to(self.device)
        if self.device == 0:
            self.model.to('cuda')

        self.tracker = ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=60, 
            frame_rate=30
        )
 
        self.box_annotator_normal = BoxAnnotator(color=ColorPalette([Color.GREEN]))
        self.label_annotator_normal = LabelAnnotator(text_color=Color.BLACK)
        
        self.box_annotator_fall = BoxAnnotator(color=ColorPalette([Color.RED]), thickness=4)
        self.label_annotator_fall = LabelAnnotator(text_color=Color.WHITE)

        self.snapshot_dir = "snapshots"
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.fall_best_scores = {} 

    def calculate_aspect_ratio(self, xyxy):
        width = xyxy[2] - xyxy[0]
        height = xyxy[3] - xyxy[1]
        return width / height

    def calculate_spine_angle(self, kp):
        """Tính góc nghiêng lưng so với mặt đất (0-90 độ)"""
        # 5,6: Vai | 11,12: Hông
        if (kp[5][2] < 0.5 or kp[6][2] < 0.5 or kp[11][2] < 0.5 or kp[12][2] < 0.5):
            return None

        shoulder_x = (kp[5][0] + kp[6][0]) / 2
        shoulder_y = (kp[5][1] + kp[6][1]) / 2
        hip_x = (kp[11][0] + kp[12][0]) / 2
        hip_y = (kp[11][1] + kp[12][1]) / 2

        dx = abs(shoulder_x - hip_x)
        dy = abs(shoulder_y - hip_y)
        
        if dy == 0: return 0.0
        angle_rad = math.atan2(dy, dx) 
        return math.degrees(angle_rad)

    def check_legs_standing(self, kp):
        """
        Kiểm tra xem chân có đang trụ vững (thẳng đứng) hay không.
        Trả về True nếu chân có xu hướng thẳng đứng (Standing/Bending).
        Trả về False nếu chân nằm ngang (Falling).
        """
        # Index: 11-Left Hip, 12-Right Hip, 15-Left Ankle, 16-Right Ankle
        # Ta lấy trung điểm Hông và trung điểm Mắt cá chân (Ankle)
        
        # Kiểm tra độ tin cậy của mắt cá chân
        has_left_leg = (kp[15][2] > 0.5)
        has_right_leg = (kp[16][2] > 0.5)
        
        if not has_left_leg and not has_right_leg:
            return False # Không thấy chân -> Giả định là ngã hoặc bị che khuất -> Coi như không đứng
            
        hip_mid_x = (kp[11][0] + kp[12][0]) / 2
        hip_mid_y = (kp[11][1] + kp[12][1]) / 2
        
        ankle_x, ankle_y = 0, 0
        count = 0
        if has_left_leg:
            ankle_x += kp[15][0]
            ankle_y += kp[15][1]
            count += 1
        if has_right_leg:
            ankle_x += kp[16][0]
            ankle_y += kp[16][1]
            count += 1
            
        ankle_avg_x = ankle_x / count
        ankle_avg_y = ankle_y / count
        
        dx = abs(hip_mid_x - ankle_avg_x)
        dy = abs(hip_mid_y - ankle_avg_y)
        
        # Nếu khoảng cách chiều dọc (dy) lớn hơn chiều ngang (dx) => Chân đang đứng
        # Ta có thể dùng góc: atan2(dy, dx). Nếu > 45 độ là chân đang đứng/trụ.
        angle_rad = math.atan2(dy, dx)
        leg_angle = math.degrees(angle_rad)
        
        # Ngưỡng cho chân lỏng hơn chút, khoảng 45-50 độ
        return leg_angle > 45.0

    def process_frame(self, frame):
        results = self.model(frame, verbose=False, conf=self.conf_threshold, device=self.device, classes=[0])[0]
        detections = Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        fall_indices = []
        normal_indices = []
        current_fall_ids = []
        
        labels_fall = []
        labels_normal = []

        for i, (xyxy, track_id) in enumerate(zip(detections.xyxy, detections.tracker_id)):
            
            # --- TÌM KEYPOINTS GỐC ---
            track_center = ((xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2)
            matched_kps = None
            min_dist = 99999
            
            # (Phần matching này giữ nguyên như cũ để đơn giản)
            for box, kps in zip(results.boxes.xyxy, results.keypoints.data):
                orig_center = ((box[0]+box[2])/2, (box[1]+box[3])/2)
                dist = math.hypot(track_center[0]-orig_center[0], track_center[1]-orig_center[1])
                if dist < 50:
                    if dist < min_dist:
                        min_dist = dist
                        matched_kps = kps.cpu().numpy()

            # --- LOGIC QUYẾT ĐỊNH ---
            is_fall = False
            debug_info = ""
            current_score = 0.0

            ratio = self.calculate_aspect_ratio(xyxy)
            angle = None
            legs_are_standing = False

            if matched_kps is not None:
                angle = self.calculate_spine_angle(matched_kps)
                legs_are_standing = self.check_legs_standing(matched_kps)

            # --- TỔNG HỢP LUẬT THÔNG MINH ---
            if angle is not None:
                # 1. Kiểm tra góc Lưng
                if angle < self.fall_angle_threshold: # Lưng nằm ngang (< 15 độ)
                    
                    # 2. Kiểm tra Chân
                    if legs_are_standing:
                        # Lưng ngang + Chân đứng => Đang CÚI (Bending)
                        is_fall = False 
                        debug_info = f"BENDING (Ang:{angle:.0f})"
                    else:
                        # Lưng ngang + Chân ngang => NGÃ THẬT
                        is_fall = True
                        current_score = 90 - angle
                        debug_info = f"FALL! (Ang:{angle:.0f})"
                else:
                    # Lưng thẳng đứng
                    is_fall = False
                    debug_info = f"OK (Ang:{angle:.0f})"
            else:
                # Fallback: Không thấy xương -> Dùng Ratio
                current_score = ratio
                if ratio > self.fall_ratio_threshold:
                    is_fall = True
                    debug_info = f"FALL (R:{ratio:.1f})"
                else:
                    debug_info = f"OK (R:{ratio:.1f})"

            # --- LƯU TRỮ VÀ VẼ ---
            if is_fall:
                fall_indices.append(i)
                current_fall_ids.append(track_id)
                labels_fall.append(debug_info)

                old_best = self.fall_best_scores.get(track_id, 0.0)
                if current_score > old_best:
                    self.fall_best_scores[track_id] = current_score
                    filename = f"fall_evidence_ID_{track_id}.jpg"
                    save_path = os.path.join(self.snapshot_dir, filename)
                    evidence_frame = frame.copy()
                    cv2.rectangle(evidence_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                    cv2.putText(evidence_frame, debug_info, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imwrite(save_path, evidence_frame)
            else:
                normal_indices.append(i)
                labels_normal.append(f"ID:{track_id} {debug_info}")

        # Clean memory
        active_keys = list(self.fall_best_scores.keys())
        for k in active_keys:
            if k not in current_fall_ids:
                del self.fall_best_scores[k]

        # Annotate
        annotated_frame = frame.copy()
        if len(normal_indices) > 0:
            det_norm = detections[np.array(normal_indices)]
            annotated_frame = self.box_annotator_normal.annotate(annotated_frame, det_norm)
            annotated_frame = self.label_annotator_normal.annotate(annotated_frame, det_norm, labels=labels_normal)

        if len(fall_indices) > 0:
            det_fall = detections[np.array(fall_indices)]
            annotated_frame = self.box_annotator_fall.annotate(annotated_frame, det_fall)
            annotated_frame = self.label_annotator_fall.annotate(annotated_frame, det_fall, labels=labels_fall)

        return annotated_frame, len(fall_indices), self.snapshot_dir