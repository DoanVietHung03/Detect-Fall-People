import cv2
import numpy as np
import os
import math
import time
import torch
from collections import deque
from ultralytics import YOLO
from supervision import ByteTrack, Detections, BoxAnnotator, LabelAnnotator, ColorPalette, Color
import onnxruntime as ort

from config import DEVICE

# --- CLASS HELPER: SOFTMAX (NUMPY) ---
def softmax(x):
    """Tính Softmax trên Numpy Array để ra xác suất %"""
    # Trừ max để tránh tràn số (numerical stability)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# --- CLASS HYBRID DETECTOR (ONNX VERSION) ---
class FallDetector:
    def __init__(self, model_pose='weights/yolo11s-pose.onnx', model_onnx='weights/gru_fall_model.onnx', conf_threshold=0.7):
        self.conf_threshold = conf_threshold
        self.device = torch.device(DEVICE)

        print(f"Loading YOLO ({model_pose})...")
        self.pose_model = YOLO(model_pose, task='pose')
        
        # --- LOAD ONNX MODEL ---
        print(f"🚀 Loading ONNX Model ({model_onnx})...")
        if not os.path.exists(model_onnx):
            print(f"❌ ERROR: Không tìm thấy file ONNX tại: {model_onnx}")
        
        # Tự động chọn Provider (Ưu tiên GPU nếu có)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.ort_session = ort.InferenceSession(model_onnx, providers=providers)
            print(f"✅ ONNX Session loaded with providers: {self.ort_session.get_providers()}")
        except Exception as e:
            print(f"⚠️ GPU Error, falling back to CPU: {e}")
            self.ort_session = ort.InferenceSession(model_onnx, providers=['CPUExecutionProvider'])

        # Lấy tên input đầu vào của model ONNX (thường là 'input')
        self.input_name = self.ort_session.get_inputs()[0].name

        # Tracker
        self.tracker = ByteTrack(track_activation_threshold=0.2, lost_track_buffer=60, frame_rate=30)

        # Annotators (Đã fix lỗi màu - Chia thành 3 bộ riêng biệt)
        # 1. GREEN (Bình thường)
        self.box_annotator_green = BoxAnnotator(color=ColorPalette([Color.GREEN]), thickness=1)
        self.label_annotator_green = LabelAnnotator(color=ColorPalette([Color.GREEN]), text_color=Color.BLACK, text_scale=0.5)

        # 2. YELLOW (Cảnh báo/Chờ)
        self.box_annotator_yellow = BoxAnnotator(color=ColorPalette([Color.YELLOW]), thickness=1)
        self.label_annotator_yellow = LabelAnnotator(color=ColorPalette([Color.YELLOW]), text_color=Color.BLACK, text_scale=0.5)

        # 3. RED (Ngã)
        self.box_annotator_red = BoxAnnotator(color=ColorPalette([Color.RED]), thickness=1)
        self.label_annotator_red = LabelAnnotator(color=ColorPalette([Color.RED]), text_color=Color.WHITE, text_scale=0.5)

        # Memory Management
        self.SEQUENCE_LENGTH = 30
        self.track_history = {}      # Lưu chuỗi pose
        self.last_valid_pose = {}    # Lưu pose cuối cùng hợp lệ
        self.track_last_seen = {}    # Thời gian lần cuối nhìn thấy (để cleanup)
        self.MEMORY_TTL = 3.0        # Thời gian nhớ (giây)

        # Business Logic State
        self.fall_start_times = {}
        self.CONFIRM_DELAY = 2.0
        self.safe_zones = []

    def set_safe_zones(self, zones):
        self.safe_zones = zones

    # ================== HELPER FUNCTIONS ===================
    def calculate_aspect_ratio(self, box):
        w = box[2] - box[0]
        h = box[3] - box[1]
        return w / h if h > 0 else 0

    def calculate_spine_angle(self, kp):
        if len(kp) < 13: return None
        # Kiểm tra độ tin cậy của các điểm vai và hông
        if (kp[5][2] < 0.3 or kp[6][2] < 0.3 or kp[11][2] < 0.3 or kp[12][2] < 0.3): return None 
        
        shoulder_x = (kp[5][0] + kp[6][0]) / 2
        shoulder_y = (kp[5][1] + kp[6][1]) / 2
        hip_x = (kp[11][0] + kp[12][0]) / 2
        hip_y = (kp[11][1] + kp[12][1]) / 2
        
        dx = abs(shoulder_x - hip_x)
        dy = abs(shoulder_y - hip_y)
        if dy == 0: return 0.0
        return math.degrees(math.atan2(dy, dx))

    def check_legs_standing(self, kp):
        has_left = (kp[15][2] > 0.3)
        has_right = (kp[16][2] > 0.3)
        if not has_left and not has_right: return False 
        
        hip_x = (kp[11][0] + kp[12][0]) / 2
        hip_y = (kp[11][1] + kp[12][1]) / 2
        
        ankle_x, ankle_y, c = 0, 0, 0
        if has_left: ankle_x += kp[15][0]; ankle_y += kp[15][1]; c += 1
        if has_right: ankle_x += kp[16][0]; ankle_y += kp[16][1]; c += 1
        
        if c == 0: return False
        dx = abs(hip_x - (ankle_x/c))
        dy = abs(hip_y - (ankle_y/c))
        angle = math.degrees(math.atan2(dy, dx))
        # Nếu góc chân > 45 độ so với trục dọc -> có thể đang nằm
        return angle > 45.0 

    def check_head_high(self, kp, box_ymin, box_ymax):
        head_y = []
        for i in range(5): # 5 điểm đầu tiên là mắt mũi tai
            if kp[i][2] > 0.3: head_y.append(kp[i][1])
        if not head_y: return False
        
        avg_head_y = sum(head_y) / len(head_y)
        box_h = box_ymax - box_ymin
        # Nếu đầu nằm ở 30% phía trên của box -> Bình thường
        return (avg_head_y - box_ymin) / box_h < 0.3

    def normalize_keypoints(self, keypoints, box):
        x1, y1, x2, y2 = box
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        normalized = []
        for kp in keypoints:
            # Chuẩn hóa về khoảng [-0.5, 0.5] so với tâm box
            normalized.extend([(kp[0] - cx)/w, (kp[1] - cy)/h])
        return normalized

    def is_in_safe_zone(self, box):
        if not self.safe_zones: return False
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2) 
        center_point = (cx, cy)
        for zone in self.safe_zones:
            if cv2.pointPolygonTest(zone, center_point, False) >= 0: return True
        return False

    # ================== PROCESS FRAME ==================
    def process_frame(self, frame):
        current_time = time.time()
        
        # 1. Detect & Track (YOLO vẫn chạy trên PyTorch/GPU)
        results = self.pose_model(frame, verbose=False, conf=self.conf_threshold, device=self.device, classes=[0])[0]
        detections = Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        yolo_boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
        yolo_kps = results.keypoints.data.cpu().numpy() if results.keypoints else []

        lstm_batch_input = []
        lstm_batch_ids = []
        analysis_results = {} 
        current_ids = []

        # --- VÒNG LẶP 1: THU THẬP DỮ LIỆU & CẬP NHẬT BUFFER ---
        for i, (track_box, track_id) in enumerate(zip(detections.xyxy, detections.tracker_id)):
            current_ids.append(track_id)
            self.track_last_seen[track_id] = current_time

            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=self.SEQUENCE_LENGTH)

            # Match Keypoints từ YOLO sang Tracker Box
            matched_kps = None
            min_dist = 200
            track_center = ((track_box[0]+track_box[2])/2, (track_box[1]+track_box[3])/2)
            
            if len(yolo_kps) > 0:
                for box_orig, kps_orig in zip(yolo_boxes, yolo_kps):
                    orig_ctr = ((box_orig[0]+box_orig[2])/2, (box_orig[1]+box_orig[3])/2)
                    dist = np.hypot(track_center[0]-orig_ctr[0], track_center[1]-orig_ctr[1])
                    if dist < min_dist:
                        min_dist = dist
                        matched_kps = kps_orig

            # Chuẩn hóa Pose
            norm_kps = [0.0] * 34
            has_pose = False
            
            if matched_kps is not None:
                has_pose = True
                norm_kps = self.normalize_keypoints(matched_kps, track_box)
                self.last_valid_pose[track_id] = norm_kps
            elif track_id in self.last_valid_pose:
                # Mất pose tạm thời -> Dùng lại pose cũ để lấp chỗ trống
                norm_kps = self.last_valid_pose[track_id]

            self.track_history[track_id].append(norm_kps)

            # Nếu đủ 30 frames -> Đưa vào hàng đợi để chạy AI
            if len(self.track_history[track_id]) == self.SEQUENCE_LENGTH:
                lstm_batch_input.append(list(self.track_history[track_id]))
                lstm_batch_ids.append(track_id)

            # Lưu thông tin tạm
            analysis_results[track_id] = {
                "box": track_box,
                "kps": matched_kps,
                "has_pose": has_pose,
                "lstm_prob": 0.0, # Sẽ cập nhật sau khi chạy ONNX
                "status": "NORMAL",
                "label": ""
            }

        # --- VÒNG LẶP 2: BATCH INFERENCE VỚI ONNX (QUAN TRỌNG) ---
        if len(lstm_batch_input) > 0:
            # Chuyển list sang Numpy array (Batch, 30, 34)
            # ONNX Runtime yêu cầu np.float32
            input_data = np.array(lstm_batch_input, dtype=np.float32)
            
            # Chạy Inference
            ort_inputs = {self.input_name: input_data}
            ort_outs = self.ort_session.run(None, ort_inputs)
            
            # Lấy Logits và tính Softmax
            logits = ort_outs[0]
            probs = softmax(logits)
            
            # Lấy xác suất ngã (Class 1)
            fall_probs = probs[:, 1]

            # Map kết quả lại cho từng ID
            for idx, tid in enumerate(lstm_batch_ids):
                analysis_results[tid]["lstm_prob"] = float(fall_probs[idx])

        # --- VÒNG LẶP 3: BUSINESS LOGIC & GROUPING ĐỂ VẼ ---
        final_fall_count = 0
        max_score = 0.0
        
        idx_green = []
        labels_green = []
        idx_yellow = []
        labels_yellow = []
        idx_red = []
        labels_red = []
        
        for i, (track_box, track_id) in enumerate(zip(detections.xyxy, detections.tracker_id)):
            data = analysis_results[track_id]
            ai_prob = data["lstm_prob"]
            kps = data["kps"]
            has_pose = data["has_pose"]
            aspect_ratio = self.calculate_aspect_ratio(track_box)
            
            is_potential_fall = False
            reason = "OK"

            # Logic kết hợp Rule-based + AI
            if has_pose:
                spine_angle = self.calculate_spine_angle(kps) or 90
                legs_standing = self.check_legs_standing(kps)
                head_high = self.check_head_high(kps, track_box[1], track_box[3])
                
                if spine_angle < 45:
                    if not legs_standing:
                        if ai_prob > 0.7:
                            is_potential_fall = True
                            reason = f"AI:{ai_prob:.2f}"
                        elif spine_angle < 15:
                            is_potential_fall = True
                            reason = f"Flat:{int(spine_angle)}"
            else:
                # Fallback khi không có keypoints
                if aspect_ratio > 1.2 and ai_prob > 0.6:
                    is_potential_fall = True
                    reason = f"Box+AI:{ai_prob:.1f}"

            # Kiểm tra vùng an toàn
            if is_potential_fall and self.is_in_safe_zone(track_box):
                is_potential_fall = False
                reason = "Safe"

            label_text = f"ID:{track_id}"

            # Phân loại trạng thái để tô màu
            if is_potential_fall:
                if track_id not in self.fall_start_times:
                    self.fall_start_times[track_id] = current_time
                    # Mới phát hiện -> Vàng
                    idx_yellow.append(i)
                    labels_yellow.append(f"Wait... {reason}")
                else:
                    elapsed = current_time - self.fall_start_times[track_id]
                    if elapsed > self.CONFIRM_DELAY:
                        # Đã nằm đủ lâu -> Đỏ (Ngã thật)
                        final_fall_count += 1
                        if ai_prob > max_score: max_score = ai_prob
                        idx_red.append(i)
                        labels_red.append(f"FALL! {reason}")
                    else:
                        # Vẫn đang chờ -> Vàng
                        idx_yellow.append(i)
                        labels_yellow.append(f"Wait {2.0 - elapsed:.1f}s")
            else:
                # Bình thường -> Xanh
                if track_id in self.fall_start_times: del self.fall_start_times[track_id]
                idx_green.append(i)
                labels_green.append(label_text)

        # --- CLEANUP (Grace Period) ---
        cleanup_ids = []
        for tid, last_seen in self.track_last_seen.items():
            if current_time - last_seen > self.MEMORY_TTL: cleanup_ids.append(tid)
            
        for tid in cleanup_ids:
            if tid in self.track_history: del self.track_history[tid]
            if tid in self.last_valid_pose: del self.last_valid_pose[tid]
            if tid in self.fall_start_times: del self.fall_start_times[tid]
            if tid in self.track_last_seen: del self.track_last_seen[tid]

        # --- DRAWING ---
        ann = frame.copy()
        if self.safe_zones: cv2.polylines(ann, self.safe_zones, True, (255, 200, 0), 2)
        
        # Vẽ từng nhóm màu
        if idx_green:
            det = detections[np.array(idx_green)]
            ann = self.box_annotator_green.annotate(ann, det)
            ann = self.label_annotator_green.annotate(ann, det, labels=labels_green)
        if idx_yellow:
            det = detections[np.array(idx_yellow)]
            ann = self.box_annotator_yellow.annotate(ann, det)
            ann = self.label_annotator_yellow.annotate(ann, det, labels=labels_yellow)
        if idx_red:
            det = detections[np.array(idx_red)]
            ann = self.box_annotator_red.annotate(ann, det)
            ann = self.label_annotator_red.annotate(ann, det, labels=labels_red)

        return ann, final_fall_count, max_score