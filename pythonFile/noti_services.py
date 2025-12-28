# notification_service.py
import requests
import time
import os
import threading

class NotificationService:
    def __init__(self, token, chat_id):
        self.TOKEN = token
        self.CHAT_ID = chat_id
        self.API_URL = f"https://api.telegram.org/bot{self.TOKEN}/sendPhoto"
        self.MSG_URL = f"https://api.telegram.org/bot{self.TOKEN}/sendMessage"
        
        # Chống Spam: Chỉ gửi 1 tin mỗi 60 giây cho 1 camera
        self.last_sent = {} 
        self.COOLDOWN = 60 

    def send_alert(self, cam_id, snapshot_url, score, event_time=None):
        """
        Hàm chính được gọi từ api_server.py
        snapshot_url: Đường dẫn web (VD: /snapshots/cam_1/img.jpg?t=123)
        """
        # 1. Check Cooldown (Tránh spam nổ điện thoại)
        now = time.time()
        if cam_id in self.last_sent:
            if now - self.last_sent[cam_id] < self.COOLDOWN:
                print(f"⏳ [Telegram] Đang chờ cooldown cho {cam_id}...")
                return
            
        if event_time is None:
            event_time = time.strftime('%H:%M:%S')

        # 2. Xử lý đường dẫn ảnh
        # API trả về URL web (/snapshots/...), ta cần đường dẫn file thực tế trên ổ cứng
        # Xóa các tham số query (?t=...) và thêm dấu chấm (.) để trỏ về thư mục hiện tại
        clean_path = snapshot_url.split('?')[0] # /snapshots/cam_1/img.jpg
        file_path = f".{clean_path}"           # ./snapshots/cam_1/img.jpg

        # 3. Gửi tin nhắn (Chạy thread để không làm đơ Camera)
        t = threading.Thread(target=self._send_thread, args=(cam_id, file_path, score, event_time))
        t.start()
        
        # Cập nhật thời gian gửi
        self.last_sent[cam_id] = now

    def _send_thread(self, cam_id, file_path, score, event_time):
        caption = f"🚨 **CẢNH BÁO PHÁT HIỆN NGÃ!**\n📹 Cam: `{cam_id}`\n🎯 Độ tin cậy: `{score:.2f}`\n⏰ Lúc: {event_time}"
        
        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    payload = {
                        'chat_id': self.CHAT_ID,
                        'caption': caption,
                        'parse_mode': 'Markdown'
                    }
                    files = {'photo': f}
                    resp = requests.post(self.API_URL, data=payload, files=files)
                    
                    if resp.status_code == 200:
                        print(f"✅ [Telegram] Đã gửi ảnh cảnh báo {cam_id}")
                    else:
                        print(f"❌ [Telegram] Lỗi gửi ảnh: {resp.text}")
            else:
                # Nếu không tìm thấy ảnh, gửi tin nhắn text báo lỗi
                err_msg = caption + "\n⚠️ (Không tìm thấy file ảnh snapshot)"
                requests.post(self.MSG_URL, json={'chat_id': self.CHAT_ID, 'text': err_msg})
                print(f"⚠️ [Telegram] Không tìm thấy file: {file_path}")

        except Exception as e:
            print(f"❌ [Telegram] Exception: {e}")