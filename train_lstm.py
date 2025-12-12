import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from config import DEVICE 

# --- ĐỊNH NGHĨA MODEL PYTORCH ---
class FallLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_classes=2):
        super(FallLSTM, self).__init__()
        # Bidirectional LSTM: input 34 -> hidden 64*2 = 128
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        # Linear layer: 128 -> 32
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.relu = nn.ReLU()
        # Output layer: 32 -> 2
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        # Chúng ta lấy output của bước thời gian cuối cùng
        out, (h_n, c_n) = self.lstm(x)
        
        # Lấy hidden state cuối cùng của cả 2 chiều (forward & backward)
        # h_n[-2] là chiều xuôi, h_n[-1] là chiều ngược
        x = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1) 
        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # --- LOAD DATA ---
    print("🔄 Đang load dữ liệu...")
    X = np.load("./data_kps/X_data.npy")
    y = np.load("./data_kps/y_data.npy")

    # Chuyển sang Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long) # CrossEntropyLoss cần kiểu Long (index)

    # Chia Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y)

    # Tạo DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

    # --- KHỞI TẠO ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")
    
    model = FallLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- TRAIN LOOP ---
    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
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
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {running_loss/len(train_loader):.4f} | "
              f"Train Acc: {100*correct/total:.2f}% | "
              f"Val Acc: {100*val_correct/val_total:.2f}%")

    # --- SAVE MODEL ---
    import os
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/lstm_fall_model.pth")
    print("✅ Đã lưu model tại: weights/lstm_fall_model.pth")