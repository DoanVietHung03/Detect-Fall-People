# shared_memory_utils.py
import numpy as np
from multiprocessing import shared_memory, Lock
import time

class SharedFrameManager:
    """
    Quản lý Shared Memory để truyền Video Frame giữa các Process
    """
    def __init__(self, name, width, height, channels=3, create=False):
        self.name = name
        self.shape = (height, width, channels)
        # Tính kích thước byte cần thiết: H * W * C * 1 byte (uint8)
        self.nbytes = height * width * channels
        self.lock = Lock() # Lock để tránh việc đang ghi thì bị đọc (gây xé hình)

        if create:
            try:
                # Cố gắng xóa nếu tồn tại rác cũ
                shm = shared_memory.SharedMemory(name=name)
                shm.close()
                shm.unlink()
            except:
                pass
            # Tạo bộ nhớ mới
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=self.nbytes)
        else:
            # Kết nối vào bộ nhớ đã tạo
            self.shm = shared_memory.SharedMemory(name=name)

        # Tạo numpy array trỏ thẳng vào vùng nhớ này (Zero-copy view)
        self.buffer = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)

    def write(self, frame):
        """Ghi frame vào Shared Memory"""
        if frame is None: return
        
        # Resize nếu frame không đúng kích thước đã định
        if frame.shape != self.shape:
            frame = cv2.resize(frame, (self.shape[1], self.shape[0]))
            
        with self.lock:
            # Copy dữ liệu vào buffer (Nhanh hơn Queue rất nhiều)
            self.buffer[:] = frame[:]

    def read(self):
        """Đọc frame từ Shared Memory"""
        with self.lock:
            # Trả về bản copy để xử lý tiếp (để giải phóng lock nhanh)
            return self.buffer.copy()

    def close(self):
        """Dọn dẹp"""
        self.shm.close()

    def unlink(self):
        """Hủy bộ nhớ (chỉ gọi ở Process cha khi tắt app)"""
        self.shm.close()
        self.shm.unlink()