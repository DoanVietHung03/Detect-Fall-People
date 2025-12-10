import torch

# CẤU HÌNH MÔ HÌNH
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")