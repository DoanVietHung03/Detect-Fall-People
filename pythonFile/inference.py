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
from onnxruntime import SessionOptions # Import thêm để config log

from config import DEVICE

# --- CLASS HELPER: SOFTMAX (NUMPY) ---
def softmax(x):
    """Tính Softmax trên Numpy Array để ra xác suất %"""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# --- CLASS HYBRID DETECTOR ---
class FallDetector:
    def __init__(self, model_pose='weights/yolo11s-pose.onnx', model_onnx='weights/gru_fall_model.onnx', conf_threshold=0.7):
        self.conf_threshold = conf_threshold
        self.device = torch.device(DEVICE)

        # 1. LOAD YOLO (POSE)
        print(f"Loading YOLO ({model_pose})...")
        # task='pose' giúp định hình output chuẩn ngay cả khi metadata ONNX thiếu
        self.pose_model = YOLO(model_pose, task='pose') 
        
        # 2. LOAD ONNX (CLASSIFIER)
        print(f"🚀 Loading ONNX Model ({model_onnx})...")
        if not os.path.exists(model_onnx):
            print(f"❌ ERROR: Không tìm thấy file ONNX tại: {model_onnx}")
        
        # Cấu hình để tắt cảnh báo "Memcpy nodes"
        sess_options = SessionOptions()
        sess_options.log_severity_level = 3  # 0:Verbose, 1:Info, 2:Warning, 3:Error

        # Tự động chọn Provider (Ưu tiên GPU)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.ort_session = ort.InferenceSession(model_onnx, sess_options=sess_options, providers=providers)
            print(f"✅ ONNX Session loaded with providers: {self.ort_session.get_providers()}")
        except Exception as e:
            print(f"⚠️ GPU Error, falling back to CPU: {e}")
            self.ort_session = ort.InferenceSession(model_onnx, sess_options=sess_options, providers=['CPUExecutionProvider'])

        self.input_name = self.ort_session.get_inputs()[0].name

        # 3. TRACKER CONFIG (Tối ưu cho việc ngã)
        # Tăng lost_track_buffer lên 60 (2 giây) để giữ ID lâu hơn khi bị khuất/biến dạng
        self.tracker = ByteTrack(track_activation_threshold=0.2, lost_track_buffer=90, frame_rate=30)

        # 4. ANNOTATORS
        self.box_annotator_green = BoxAnnotator(color=ColorPalette([Color.GREEN]), thickness=1)
        self.label_annotator_green = LabelAnnotator(color=ColorPalette([Color.GREEN]), text_color=Color.BLACK, text_scale=0.5)

        self.box_annotator_yellow = BoxAnnotator(color=ColorPalette([Color.YELLOW]), thickness=1)
        self.label_annotator_yellow = LabelAnnotator(color=ColorPalette([Color.YELLOW]), text_color=Color.BLACK, text_scale=0.5)

        self.box_annotator_red = BoxAnnotator(color=ColorPalette([Color.RED]), thickness=1)
        self.label_annotator_red = LabelAnnotator(color=ColorPalette([Color.RED]), text_color=Color.WHITE, text_scale=0.5)

        # 5. MEMORY & STATE
        self.SEQUENCE_LENGTH = 30
        self.MEMORY_TTL = 3.0        
        
        self.track_history = {}      # {id: deque([...])}
        self.last_valid_pose = {}    # {id: normalized_kps}
        self.track_last_seen = {}    # {id: timestamp}
        self.track_last_box = {}     # {id: [x1, y1, x2, y2]} -> Lưu vị trí cuối cùng để Re-ID
        
        # Buffer cho tính năng Merge Track (Re-ID logic)
        self.lost_tracks_buffer = {} # {id: {"box": box, "history": deque, "time": t}}
        self.MERGE_DIST_THRESHOLD = 150 # Pixel (Chấp nhận di chuyển 1 đoạn khi ngã)
        self.MERGE_TIME_THRESHOLD = 1.5 # Giây

        # Business Logic
        self.fall_start_times = {}
        self.CONFIRM_DELAY = 1.0
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
        return angle > 45.0 

    def check_head_high(self, kp, box_ymin, box_ymax):
        head_y = []
        for i in range(5): 
            if kp[i][2] > 0.3: head_y.append(kp[i][1])
        if not head_y: return False
        avg_head_y = sum(head_y) / len(head_y)
        box_h = box_ymax - box_ymin
        return (avg_head_y - box_ymin) / box_h < 0.3

    def normalize_keypoints(self, keypoints, box):
        x1, y1, x2, y2 = box
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        normalized = []
        for kp in keypoints:
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

    def try_merge_tracks(self, new_id, new_box, current_time):
        """Logic tìm track cũ để nối vào track mới"""
        best_match_id = None
        min_dist = float('inf')
        new_center = ((new_box[0]+new_box[2])/2, (new_box[1]+new_box[3])/2)

        # Lọc danh sách hết hạn
        expired = []
        for old_id, data in self.lost_tracks_buffer.items():
            if current_time - data["time"] > self.MERGE_TIME_THRESHOLD:
                expired.append(old_id)
                continue
            
            old_box = data["box"]
            old_center = ((old_box[0]+old_box[2])/2, (old_box[1]+old_box[3])/2)
            dist = np.hypot(new_center[0]-old_center[0], new_center[1]-old_center[1])

            if dist < self.MERGE_DIST_THRESHOLD:
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = old_id
        
        for eid in expired: del self.lost_tracks_buffer[eid]
        return best_match_id
    
    def calculate_visibility(self, kps):
        """Trả về % số điểm khớp nhìn thấy rõ"""
        if kps is None or len(kps) == 0: return 0.0
        visible_count = sum(1 for p in kps if p[2] > 0.4) # Conf > 0.4 coi là thấy
        return visible_count / 17.0

    # ================== PROCESS FRAME ==================
    def process_frame(self, frame):
        current_time = time.time()
        
        # 1. Detect YOLO
        # Lưu ý: Khi dùng multiprocessing, device được tự động handle bởi Ultralytics/ONNX
        results = self.pose_model(frame, verbose=False, conf=self.conf_threshold, classes=[0])[0]
        detections = Detections.from_ultralytics(results)
        
        # 2. Tracking
        detections = self.tracker.update_with_detections(detections)

        yolo_boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
        yolo_kps = results.keypoints.data.cpu().numpy() if results.keypoints else []

        # --- RE-ID LOGIC: QUẢN LÝ TRACK MẤT ---
        # Tìm những ID vừa biến mất trong frame này
        active_ids = set(detections.tracker_id) if detections.tracker_id is not None else set()
        existing_ids = set(self.track_history.keys())
        lost_ids = existing_ids - active_ids
        
        for lid in lost_ids:
            # Lưu vào buffer tạm để chờ hồi sinh
            if lid in self.track_last_box and len(self.track_history[lid]) > 5:
                self.lost_tracks_buffer[lid] = {
                    "box": self.track_last_box[lid],
                    "history": self.track_history[lid],
                    "time": current_time
                }
            # Xóa khỏi bộ nhớ chính
            del self.track_history[lid]
            if lid in self.track_last_seen: del self.track_last_seen[lid]

        # Chuẩn bị Batch Input cho LSTM
        lstm_batch_input = []
        lstm_batch_ids = []
        analysis_results = {} 

        # --- VÒNG LẶP 1: THU THẬP & XỬ LÝ ID ---
        for i, (track_box, track_id) in enumerate(zip(detections.xyxy, detections.tracker_id)):
            self.track_last_seen[track_id] = current_time
            self.track_last_box[track_id] = track_box

            # Xử lý ID mới: Thử tìm lại track cũ (Merge)
            if track_id not in self.track_history:
                merged_old_id = self.try_merge_tracks(track_id, track_box, current_time)
                if merged_old_id:
                    # print(f"Merge: {merged_old_id} -> {track_id}")
                    self.track_history[track_id] = self.lost_tracks_buffer[merged_old_id]["history"]
                    del self.lost_tracks_buffer[merged_old_id] # Xóa khỏi buffer chờ
                    
                    # Nếu track cũ đang đếm giờ ngã -> Chuyển sang track mới
                    if merged_old_id in self.fall_start_times:
                        self.fall_start_times[track_id] = self.fall_start_times[merged_old_id]
                        del self.fall_start_times[merged_old_id]
                else:
                    self.track_history[track_id] = deque(maxlen=self.SEQUENCE_LENGTH)

            # Match Keypoints
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

            # Normalize Pose
            norm_kps = [0.0] * 34
            has_pose = False
            
            if matched_kps is not None:
                has_pose = True
                # Chuẩn hóa hiện tại
                current_norm_kps = self.normalize_keypoints(matched_kps, track_box)
                
                # Logic: Nếu điểm nào có độ tin cậy thấp (bị che), lấy từ quá khứ đắp vào
                if track_id in self.last_valid_pose:
                    last_kps = self.last_valid_pose[track_id]
                    final_kps = []
                    for i in range(17): # 17 điểm khớp
                        # Index trong vector phẳng: x=2*i, y=2*i+1
                        idx_x, idx_y = 2*i, 2*i+1
                        conf = matched_kps[i][2] # Độ tin cậy từ YOLO
                        
                        if conf < 0.3: # Bị che hoặc mờ
                            # Lấy toạ độ cũ
                            final_kps.extend([last_kps[idx_x], last_kps[idx_y]])
                        else:
                            # Lấy toạ độ mới
                            final_kps.extend([current_norm_kps[idx_x], current_norm_kps[idx_y]])
                    norm_kps = final_kps
                else:
                    norm_kps = current_norm_kps

                # Cập nhật lại bộ nhớ (Lưu cái đã fill để dùng cho frame sau)
                self.last_valid_pose[track_id] = norm_kps
            
            elif track_id in self.last_valid_pose:
                # Mất toàn bộ Pose -> Dùng lại toàn bộ Pose cũ
                norm_kps = self.last_valid_pose[track_id]

            self.track_history[track_id].append(norm_kps)

            if len(self.track_history[track_id]) == self.SEQUENCE_LENGTH:
                lstm_batch_input.append(list(self.track_history[track_id]))
                lstm_batch_ids.append(track_id)

            analysis_results[track_id] = {
                "box": track_box,
                "kps": matched_kps,
                "has_pose": has_pose,
                "lstm_prob": 0.0,
                "status": "NORMAL"
            }

        # --- VÒNG LẶP 2: BATCH INFERENCE ONNX ---
        if len(lstm_batch_input) > 0:
            input_data = np.array(lstm_batch_input, dtype=np.float32)
            ort_inputs = {self.input_name: input_data}
            ort_outs = self.ort_session.run(None, ort_inputs)
            
            probs = softmax(ort_outs[0])
            fall_probs = probs[:, 1] # Class 1 = Fall

            for idx, tid in enumerate(lstm_batch_ids):
                analysis_results[tid]["lstm_prob"] = float(fall_probs[idx])

        # --- VÒNG LẶP 3: LOGIC & VISUALIZATION ---
        final_fall_count = 0
        max_score = 0.0
        
        idx_green, labels_green = [], []
        idx_yellow, labels_yellow = [], []
        idx_red, labels_red = [], []
        
        for i, (track_box, track_id) in enumerate(zip(detections.xyxy, detections.tracker_id)):
            data = analysis_results[track_id]
            ai_prob = data["lstm_prob"]
            kps = data["kps"]
            has_pose = data["has_pose"]
            aspect_ratio = self.calculate_aspect_ratio(track_box)
            
            visibility = self.calculate_visibility(kps) if has_pose else 0.0
            is_potential_fall = False
            reason = "OK"

            # --- ADAPTIVE LOGIC ---
            
            # CASE 1: NHÌN THẤY RÕ (> 60% cơ thể) -> Dùng luật chặt chẽ như cũ
            if visibility > 0.6:
                spine_angle = self.calculate_spine_angle(kps) or 90
                legs_standing = self.check_legs_standing(kps)
                
                # Nếu AI cực cao thì báo luôn (bất chấp góc)
                if ai_prob > 0.85:
                    is_potential_fall = True
                    reason = f"Clear_AI:{ai_prob:.2f}"
                # Nếu AI khá + Góc nghiêng
                elif ai_prob > 0.6 and spine_angle < 60:
                     if not legs_standing:
                        is_potential_fall = True
                        reason = f"Clear_Hybrid"
                # Nếu nằm bẹp gí
                elif spine_angle < 20:
                    is_potential_fall = True
                    reason = "Clear_Flat"

            # CASE 2: BỊ CHE KHUẤT (< 60% cơ thể)
            # Khi bị che, YOLO hay bắt sai chân tay -> Góc Spine sai -> Bỏ qua check góc
            elif visibility > 0.2: 
                # Chỉ cần AI nghi ngờ + Hộp dẹt (Aspect Ratio)
                # Aspect Ratio: W/H. Người đứng ~0.5. Người ngã/ngồi > 1.0
                if ai_prob > 0.55: # Giảm ngưỡng AI xuống
                    if aspect_ratio > 0.9: # Hộp bắt đầu bè ra
                        is_potential_fall = True
                        reason = f"Obscured_AI:{ai_prob:.2f}"
            
            # CASE 3: MẤT HẾT POSE HOẶC CHE GẦN HẾT -> Chỉ dùng Box
            else:
                if aspect_ratio > 1.2 and ai_prob > 0.5:
                    is_potential_fall = True
                    reason = "BoxOnly"

            if is_potential_fall and self.is_in_safe_zone(track_box):
                is_potential_fall = False
                reason = "Safe"

            # State Machine
            if is_potential_fall:
                if track_id not in self.fall_start_times:
                    self.fall_start_times[track_id] = current_time
                    idx_yellow.append(i)
                    labels_yellow.append(f"Wait... {reason}")
                else:
                    elapsed = current_time - self.fall_start_times[track_id]
                    if elapsed > self.CONFIRM_DELAY:
                        final_fall_count += 1
                        if ai_prob > max_score: max_score = ai_prob
                        idx_red.append(i)
                        labels_red.append(f"FALL! {reason}")
                    else:
                        idx_yellow.append(i)
                        labels_yellow.append(f"Wait {self.CONFIRM_DELAY - elapsed:.1f}s")
            else:
                if track_id in self.fall_start_times: del self.fall_start_times[track_id]
                idx_green.append(i)
                labels_green.append(f"ID:{track_id}")

        # CLEANUP
        cleanup_ids = []
        for tid, last_seen in self.track_last_seen.items():
            if current_time - last_seen > self.MEMORY_TTL: cleanup_ids.append(tid)
        for tid in cleanup_ids:
            if tid in self.track_history: del self.track_history[tid]
            if tid in self.last_valid_pose: del self.last_valid_pose[tid]
            if tid in self.fall_start_times: del self.fall_start_times[tid]
            if tid in self.track_last_seen: del self.track_last_seen[tid]
            if tid in self.track_last_box: del self.track_last_box[tid]

        # DRAW
        ann = frame.copy()
        if self.safe_zones: cv2.polylines(ann, self.safe_zones, True, (255, 200, 0), 2)
        
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