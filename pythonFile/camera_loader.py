# camera_loader.py
import cv2
import threading
import time

class CameraStream:
    def __init__(self, rtsp_url, cam_id):
        self.rtsp_url = rtsp_url
        self.cam_id = cam_id
        self.frame = None
        self.status = False
        self.capture = None
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()
        self.reconnect_interval = 5

    def start(self):
        if self.is_running: return
        print(f"✅ [{self.cam_id}] Stream started.")
        self.is_running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.is_running:
            if self.capture is None or not self.capture.isOpened():
                self._connect()
            
            if self.capture and self.capture.isOpened():
                ret, frame = self.capture.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                    self.status = True
                else:
                    self.status = False
                    print(f"⚠️ [{self.cam_id}] Frame lost. Reconnecting...")
                    self.capture.release()
                    time.sleep(self.reconnect_interval)
            else:
                time.sleep(self.reconnect_interval)
        
        # Cleanup khi vòng lặp kết thúc
        if self.capture:
            self.capture.release()

    def _connect(self):
        print(f"🔄 [{self.cam_id}] Connecting to RTSP...")
        try:
            self.capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            if self.capture.isOpened():
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            print(f"❌ [{self.cam_id}] Connect Error: {e}")

    def read(self):
        """Trả về frame mới nhất hiện có"""
        with self.lock:
            # Trả về bản copy để tránh xung đột thread khi AI đang xử lý mà frame bị ghi đè
            return self.status, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.is_running = False
        if self.thread: self.thread.join()