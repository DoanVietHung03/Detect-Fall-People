import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx  # <--- THÊM THƯ VIỆN NÀY
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os
from config import DEVICE 

# --- 1. ĐỔI MODEL TỪ LSTM SANG GRU (NHẸ HƠN) ---
class FallGRU(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_classes=2):
        super(FallGRU, self).__init__()
        # GRU thay cho LSTM (Bỏ cổng c_n, tính toán ít hơn)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        
        # Linear layer nhận vào hidden_size * 2 (do bidirectional)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # GRU chỉ trả về output và hidden state (h_n), không có cell state (c_n)
        out, h_n = self.gru(x)
        
        # Lấy hidden state cuối cùng của 2 chiều (Forward + Backward)
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        x = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1) 
        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # --- LOAD DATA (GIỮ NGUYÊN) ---
    print("🔄 Đang load dữ liệu...")
    if not os.path.exists("./data_kps/X_data.npy"):
        print("❌ Lỗi: Không thấy file data! Hãy chạy prepare_data.py trước.")
        exit()

    X = np.load("./data_kps/X_data.npy")
    y = np.load("./data_kps/y_data.npy")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Tăng batch size lên xíu cho nhanh
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # --- KHỞI TẠO MODEL MỚI (GRU) ---
    print("🚀 Khởi tạo model GRU...")
    model = FallGRU().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- TRAIN LOOP ---
    epochs = 50
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Lưu checkpoint tốt nhất nếu cần
        
        if (epoch + 1) % 5 == 0: # Log mỗi 5 epoch cho đỡ rối
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Loss: {running_loss/len(train_loader):.4f} | "
                  f"Train Acc: {100*correct/total:.2f}% | "
                  f"Val Acc: {val_acc:.2f}%")

    # --- 2. LƯU MODEL & EXPORT ONNX ---
    os.makedirs("weights", exist_ok=True)
    
    # Save PyTorch Weights (để backup hoặc train tiếp)
    # torch.save(model.state_dict(), "../weights/gru_fall_model.pth")
    # print("\n✅ Đã lưu weights PyTorch: weights/gru_fall_model.pth")

    # --- EXPORT ONNX (QUAN TRỌNG) ---
    print("🔄 Đang export sang ONNX...")
    model.eval()
    
    # Tạo input giả (Dummy input) đúng kích thước để định hình model
    # Batch size = 1, Sequence Length = 30, Input Size = 34
    dummy_input = torch.randn(1, 30, 34).to(DEVICE)
    
    onnx_path = "../weights/gru_fall_model.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,        # Lưu trọng số bên trong file
        opset_version=12,          # Version phổ biến
        do_constant_folding=True,  # Tối ưu hóa các hằng số
        input_names=['input'],     # Tên đầu vào
        output_names=['output'],   # Tên đầu ra
        dynamic_axes={
            'input': {0: 'batch_size'},  # Cho phép batch size thay đổi
            'output': {0: 'batch_size'}
        }
    )
    print(f"🚀 ĐÃ EXPORT THÀNH CÔNG: {onnx_path}")
    print("👉 Hãy cập nhật đường dẫn trong inference.py để dùng file .onnx này!")