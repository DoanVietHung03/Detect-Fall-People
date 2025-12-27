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
from onnxruntime import SessionOptions

from config import DEVICE

# --- CLASS HELPER: SOFTMAX (NUMPY) ---
def softmax(x):
    """Calculate Softmax for Numpy array of any size along the last axis."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# --- CLASS HYBRID DETECTOR ---
class FallDetector:
    def __init__(self, model_pose='weights/yolo11s-pose.onnx', model_onnx='weights/gru_fall_model.onnx', conf_threshold=0.7):
        self.conf_threshold = conf_threshold
        self.device = torch.device(DEVICE)

        # 1. LOAD YOLO (POSE)
        print(f"Loading YOLO ({model_pose})...")
        self.pose_model = YOLO(model_pose, task='pose') 
        
        # 2. LOAD ONNX (CLASSIFIER)
        print(f"🚀 Loading ONNX Model ({model_onnx})...")
        if not os.path.exists(model_onnx):
            print(f"❌ ERROR: Cannot find ONNX file at: {model_onnx}")
        
        sess_options = SessionOptions()
        sess_options.log_severity_level = 3

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.ort_session = ort.InferenceSession(model_onnx, sess_options=sess_options, providers=providers)
            print(f"✅ ONNX Session loaded with providers: {self.ort_session.get_providers()}")
        except Exception as e:
            print(f"⚠️ GPU Error, falling back to CPU: {e}")
            self.ort_session = ort.InferenceSession(model_onnx, sess_options=sess_options, providers=['CPUExecutionProvider'])

        self.input_name = self.ort_session.get_inputs()[0].name

        # 3. TRACKER CONFIG
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
        self.track_last_box = {}     # {id: [x1, y1, x2, y2]}
        
        # Buffer cho tính năng Merge Track
        self.lost_tracks_buffer = {} 
        self.MERGE_DIST_THRESHOLD = 150 
        self.MERGE_TIME_THRESHOLD = 1.5 

        # --- NEW: Head History cho tính năng phát hiện ngã dọc ---
        # {id: deque([(timestamp, head_y_pixel, box_height), ...], maxlen=15)}
        self.head_history = {}

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
        """
        Kiểm tra xem đầu có nằm ở phần trên của Box không.
        Trả về True nếu đầu ở cao (bình thường), False nếu đầu ở thấp (bất thường).
        """
        head_y = []
        # Check Mũi, Mắt trái/phải, Tai trái/phải (0-4)
        for i in range(5): 
            if kp[i][2] > 0.3: head_y.append(kp[i][1])
        
        if not head_y: return True # Không thấy đầu -> Giả định là OK để tránh False Positive
        
        avg_head_y = sum(head_y) / len(head_y)
        box_h = box_ymax - box_ymin
        # Nếu đầu nằm dưới 40% chiều cao box (tính từ đỉnh) -> Bất thường
        return (avg_head_y - box_ymin) / box_h < 0.4 

    def get_head_info(self, kp, box):
        """Lấy thông tin Y của đầu để tính vận tốc rơi"""
        head_y_vals = []
        for i in range(5):
            if kp[i][2] > 0.4: head_y_vals.append(kp[i][1])
        
        if not head_y_vals: return None
        avg_y = sum(head_y_vals) / len(head_y_vals)
        box_h = max(box[3] - box[1], 1e-6)
        return avg_y, box_h

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
        best_match_id = None
        min_dist = float('inf')
        new_center = ((new_box[0]+new_box[2])/2, (new_box[1]+new_box[3])/2)

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
        if kps is None or len(kps) == 0: return 0.0
        visible_count = sum(1 for p in kps if p[2] > 0.4) 
        return visible_count / 17.0

    # ================== PROCESS FRAME ==================
    def process_frame(self, frame):
        current_time = time.time()
        
        # 1. Detect YOLO
        results = self.pose_model(frame, verbose=False, conf=self.conf_threshold, classes=[0])[0]
        detections = Detections.from_ultralytics(results)
        
        # 2. Tracking
        detections = self.tracker.update_with_detections(detections)

        yolo_boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
        yolo_kps = results.keypoints.data.cpu().numpy() if results.keypoints else []

        # --- RE-ID LOGIC ---
        active_ids = set(detections.tracker_id) if detections.tracker_id is not None else set()
        existing_ids = set(self.track_history.keys())
        lost_ids = existing_ids - active_ids
        
        for lid in lost_ids:
            if lid in self.track_last_box and len(self.track_history[lid]) > 5:
                self.lost_tracks_buffer[lid] = {
                    "box": self.track_last_box[lid],
                    "history": self.track_history[lid],
                    "time": current_time
                }
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

            if track_id not in self.track_history:
                merged_old_id = self.try_merge_tracks(track_id, track_box, current_time)
                if merged_old_id:
                    self.track_history[track_id] = self.lost_tracks_buffer[merged_old_id]["history"]
                    del self.lost_tracks_buffer[merged_old_id]
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
                current_norm_kps = self.normalize_keypoints(matched_kps, track_box)
                
                if track_id in self.last_valid_pose:
                    last_kps = self.last_valid_pose[track_id]
                    final_kps = []
                    for k in range(17): 
                        idx_x, idx_y = 2*k, 2*k+1
                        conf = matched_kps[k][2]
                        if conf < 0.3:
                            final_kps.extend([last_kps[idx_x], last_kps[idx_y]])
                        else:
                            final_kps.extend([current_norm_kps[idx_x], current_norm_kps[idx_y]])
                    norm_kps = final_kps
                else:
                    norm_kps = current_norm_kps

                self.last_valid_pose[track_id] = norm_kps
            
            elif track_id in self.last_valid_pose:
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
            fall_probs = probs[:, 1] 

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
            
            # 1. Tính toán các chỉ số vật lý
            aspect_ratio = self.calculate_aspect_ratio(track_box)
            visibility = self.calculate_visibility(kps) if has_pose else 0.0
            
            # --- LOGIC MỚI: HEAD DROP VELOCITY ---
            is_head_drop = False
            head_drop_score = 0.0
            
            if has_pose:
                head_info = self.get_head_info(kps, track_box)
                
                if track_id not in self.head_history:
                    self.head_history[track_id] = deque(maxlen=15)

                if head_info:
                    cur_head_y, cur_box_h = head_info
                    self.head_history[track_id].append((current_time, cur_head_y, cur_box_h))
                    
                    if len(self.head_history[track_id]) > 3:
                        prev_t, prev_y, prev_h = self.head_history[track_id][0]
                        dt = current_time - prev_t
                        if dt > 0.1: 
                            drop_pixel = cur_head_y - prev_y 
                            normalized_drop = drop_pixel / cur_box_h
                            if normalized_drop > 0.15: # Tụt 15% chiều cao
                                is_head_drop = True
                                head_drop_score = normalized_drop
            # -------------------------------------

            # Sử dụng hàm check_head_high
            is_head_high = True
            if has_pose:
                is_head_high = self.check_head_high(kps, track_box[1], track_box[3])

            is_potential_fall = False
            reason = "OK"

            # --- ADAPTIVE DECISION MAKING ---
            
            # PRIORITY 1: HEAD DROP (Ngã nhanh / Ngã dọc)
            # Nếu phát hiện rơi nhanh VÀ AI nghi ngờ -> Báo ngay
            if is_head_drop and ai_prob > 0.3:
                is_potential_fall = True
                reason = f"Velocity:{head_drop_score:.2f}"

            # PRIORITY 2: RÕ RÀNG (> 50% cơ thể)
            elif visibility > 0.5:
                spine_angle = self.calculate_spine_angle(kps) or 90
                legs_standing = self.check_legs_standing(kps)
                
                if ai_prob > 0.8:
                    # AI quá chắc chắn -> Fall
                    is_potential_fall = True
                    reason = f"AI:{ai_prob:.2f}"
                elif ai_prob > 0.5:
                    # AI hơi nghi ngờ -> Check thêm tư thế
                    # Nếu góc người thấp HOẶC đầu chúi xuống thấp
                    if spine_angle < 50 or not is_head_high:
                         if not legs_standing:
                            is_potential_fall = True
                            reason = f"Hybrid_Pose"
                elif spine_angle < 20 and not is_head_high:
                    # Nằm bẹp gí
                    is_potential_fall = True
                    reason = "Flat_Pose"

            # PRIORITY 3: BỊ CHE KHUẤT
            elif visibility > 0.2: 
                # Nếu AI > 0.6 và khung hình bẹt ra
                if ai_prob > 0.6 and aspect_ratio > 0.8: 
                    is_potential_fall = True
                    reason = f"Obs_AI:{ai_prob:.2f}"
                # Hoặc nếu AI vừa phải nhưng đầu tụt nhanh (đã bắt ở Priority 1)
            
            # PRIORITY 4: MẤT POSE
            else:
                if is_head_drop: # Trước khi mất pose thấy đầu rơi
                     is_potential_fall = True
                     reason = "Drop_NoPose"
                elif aspect_ratio > 1.2 and ai_prob > 0.5:
                    is_potential_fall = True
                    reason = "BoxOnly"

            # Safe Zone Filter
            if is_potential_fall and self.is_in_safe_zone(track_box):
                is_potential_fall = False
                reason = "Safe"

            # State Machine & Log
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
            if tid in self.head_history: del self.head_history[tid]

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