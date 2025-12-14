import cv2
import numpy as np
import os
import math
import torch
import torch.nn as nn
from collections import deque
from ultralytics import YOLO
from supervision import ByteTrack, Detections, BoxAnnotator, LabelAnnotator, ColorPalette, Color

from config import DEVICE

# --- 1. MODEL LSTM (PyTorch) ---
class FallLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_classes=2):
        super(FallLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        x = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- 2. CLASS HYBRID DETECTOR ---
class FallDetector:
    def __init__(self, model_pose='../weights/yolo11m-pose.pt', model_lstm='../weights/lstm_fall_model.pth', conf_threshold=0.7, lstm_threshold=0.7):
        self.conf_threshold = conf_threshold
        self.lstm_threshold = lstm_threshold
        self.device = torch.device(DEVICE)

        print(f"Loading YOLO ({model_pose})...")
        self.pose_model = YOLO(model_pose).to(self.device)
        
        print(f"Loading LSTM ({model_lstm})...")
        self.lstm_model = FallLSTM().to(self.device)
        if os.path.exists(model_lstm):
            self.lstm_model.load_state_dict(torch.load(model_lstm, map_location=self.device))
        else:
            print("⚠️ Warning: LSTM model file not found!")
        self.lstm_model.eval()

        self.tracker = ByteTrack(track_activation_threshold=0.2, lost_track_buffer=45)

        self.box_annotator_green = BoxAnnotator(color=ColorPalette([Color.GREEN]), thickness=2)
        self.label_annotator_green = LabelAnnotator(text_color=Color.BLACK, text_scale=0.5)
        self.box_annotator_red = BoxAnnotator(color=ColorPalette([Color.RED]), thickness=2)
        self.label_annotator_red = LabelAnnotator(text_color=Color.WHITE, text_scale=0.5)

        self.track_history = {} 
        self.SEQUENCE_LENGTH = 30
        self.alert_buffer = {} 
        
        self.last_valid_pose = {}

    # ================== RULE-BASED FUNCTIONS ===================
    def calculate_aspect_ratio(self, box):
        w = box[2] - box[0]
        h = box[3] - box[1]
        return w / h if h > 0 else 0

    def calculate_spine_angle(self, kp):
        if len(kp) < 13: return None
        # 5,6: Vai | 11,12: Hông
        if (kp[5][2] < 0.3 or kp[6][2] < 0.3 or kp[11][2] < 0.3 or kp[12][2] < 0.3):
            return None 

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
        if has_left:
            ankle_x += kp[15][0]; ankle_y += kp[15][1]; c += 1
        if has_right:
            ankle_x += kp[16][0]; ankle_y += kp[16][1]; c += 1
        
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

    # ================== MAIN PROCESS ==================
    def process_frame(self, frame):
        # 1. Detect
        results = self.pose_model(frame, verbose=False, conf=self.conf_threshold, device=self.device, classes=[0])[0]
        detections = Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        fall_indices = []
        normal_indices = []
        labels_fall = []
        labels_normal = []
        current_ids = []
        
        # Biến lưu điểm số cao nhất trong frame này để trả về server
        max_frame_score = 0.0

        # Data CPU
        yolo_boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
        yolo_kps = results.keypoints.data.cpu().numpy() if results.keypoints else []

        for i, (track_box, track_id) in enumerate(zip(detections.xyxy, detections.tracker_id)):
            current_ids.append(track_id)
            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=self.SEQUENCE_LENGTH)

            # Match Keypoints
            matched_kps = None
            min_dist = 100
            track_center = ((track_box[0]+track_box[2])/2, (track_box[1]+track_box[3])/2)
            
            if len(yolo_kps) > 0:
                for box_orig, kps_orig in zip(yolo_boxes, yolo_kps):
                    orig_ctr = ((box_orig[0]+box_orig[2])/2, (box_orig[1]+box_orig[3])/2)
                    dist = np.hypot(track_center[0]-orig_ctr[0], track_center[1]-orig_ctr[1])
                    if dist < 50 and dist < min_dist:
                        min_dist = dist
                        matched_kps = kps_orig

            # --- A. TÍNH TOÁN CÁC THAM SỐ RULE ---
            is_fall = False
            reason = "OK"
            
            aspect_ratio = self.calculate_aspect_ratio(track_box)
            spine_angle = 90.0
            legs_standing = True 
            head_high = True
            
            has_pose = matched_kps is not None
            
            if has_pose:
                ang = self.calculate_spine_angle(matched_kps)
                if ang is not None: spine_angle = ang
                
                legs_standing = self.check_legs_standing(matched_kps)
                head_high = self.check_head_high(matched_kps, track_box[1], track_box[3])
                
                # Fill LSTM Data
                norm_kps = self.normalize_keypoints(matched_kps, track_box)
                self.last_valid_pose[track_id] = norm_kps
                self.track_history[track_id].append(norm_kps)
            else:
                if track_id in self.last_valid_pose:
                    self.track_history[track_id].append(self.last_valid_pose[track_id])
                else:
                    self.track_history[track_id].append([0.0]*34)

            # --- B. DỰ ĐOÁN LSTM ---
            ai_prob = 0.0
            if len(self.track_history[track_id]) == self.SEQUENCE_LENGTH:
                seq = np.array([self.track_history[track_id]])
                inp = torch.tensor(seq, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    out = self.lstm_model(inp)
                    ai_prob = torch.softmax(out, dim=1)[0, 1].item()

            # --- C. HYBRID LOGIC ---
            
            # 1. RULE RÕ RÀNG
            if has_pose:
                if spine_angle < 15: 
                    if not legs_standing: 
                        is_fall = True
                        reason = f"R:LayFlat({int(spine_angle)})"
                    else:
                        reason = "Bending"
                
                elif spine_angle < 60: 
                    if not legs_standing:
                        if ai_prob > 0.5: 
                            is_fall = True
                            reason = f"Hybrid:Angle+AI({ai_prob:.2f})"
                        else:
                            if not head_high:
                                is_fall = True
                                reason = "R:LowHead"
                    else:
                        reason = "Bending"
                
                else: 
                    if aspect_ratio > 0.9: 
                        if head_high: 
                            reason = "Kneel/Sit" 
                        else:
                            if ai_prob > 0.6:
                                is_fall = True
                                reason = "Hybrid:FrontFall"

            # 2. KHÔNG CÓ POSE (Dựa vào BBox)
            else:
                if aspect_ratio > 1.2:
                    if ai_prob > 0.4:
                        is_fall = True
                        reason = f"LostPose+AI({ai_prob:.1f})"
                    else:
                        is_fall = True
                        reason = f"BoxRatio({aspect_ratio:.1f})"

            # --- D. XỬ LÝ KẾT QUẢ ---
            if is_fall:
                self.alert_buffer[track_id] = 40
                
                # Tính điểm cho Server biết
                # Nếu có điểm AI thì lấy, nếu ngã theo luật (không có AI) thì cho điểm 0.95
                current_score = ai_prob if ai_prob > 0 else 0.95
                if current_score > max_frame_score:
                    max_frame_score = current_score
            
            if self.alert_buffer.get(track_id, 0) > 0:
                self.alert_buffer[track_id] -= 1
                fall_indices.append(i)
                labels_fall.append(f"ID:{track_id} {reason}")
                
                # Server sẽ xử lý việc lưu ảnh dựa trên max_frame_score trả về
            else:
                normal_indices.append(i)
                labels_normal.append(f"ID:{track_id}")

        # Clean
        for k in list(self.track_history.keys()):
            if k not in current_ids:
                del self.track_history[k]
                if k in self.alert_buffer: del self.alert_buffer[k]

        # Draw
        ann = frame.copy()
        if normal_indices:
            det = detections[np.array(normal_indices)]
            ann = self.box_annotator_green.annotate(ann, det)
            ann = self.label_annotator_green.annotate(ann, det, labels_normal)
        if fall_indices:
            det = detections[np.array(fall_indices)]
            ann = self.box_annotator_red.annotate(ann, det)
            ann = self.label_annotator_red.annotate(ann, det, labels_fall)

        # TRẢ VỀ: Frame vẽ rồi, số người ngã, và ĐIỂM SỐ CAO NHẤT
        return ann, len(fall_indices), max_frame_score