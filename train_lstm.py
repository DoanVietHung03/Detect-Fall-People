# 2_train_lstm.py
import numpy as np
import tensorflow as pd
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from config import DEVICE

# --- LOAD DATA ---
print("🔄 Đang load dữ liệu...")
X = np.load("./data_kps/X_data.npy")
y = np.load("./data_kps/y_data.npy")

# One-hot encoding cho Label (0 -> [1, 0], 1 -> [0, 1])
y = to_categorical(y, num_classes=2)

# Chia Train/Test (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- XÂY DỰNG MODEL LSTM ---
model = Sequential()

# Input Shape: (15 frames, 34 keypoints)
# Bidirectional LSTM giúp học ngữ cảnh 2 chiều (quá khứ <-> tương lai trong window)
# Kiến trúc nhẹ hơn, phù hợp với data ít
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

# Chỉ dùng 1 lớp LSTM nhưng tăng nhẹ unit lên
model.add(Bidirectional(LSTM(64, return_sequences=False))) 
model.add(Dropout(0.4)) # Tăng Dropout để model bớt "học vẹt"

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- TRAIN ---
print("🚀 Bắt đầu training...")
history = model.fit(
    X_train, y_train,
    epochs=50,             # Số lần học
    batch_size=24,
    validation_data=(X_test, y_test)
)

# --- LƯU MODEL ---
model_dir = "weights" # Thư mục chứa model
model.save("weights/lstm_fall_model.h5") # Hoặc .keras
print("✅ Đã lưu model tại: weights/lstm_fall_model.h5")

# Đánh giá nhanh
loss, acc = model.evaluate(X_test, y_test)
print(f"🎯 Độ chính xác trên tập Test: {acc*100:.2f}%")