import onnx
from onnxconverter_common import float16

def convert_to_fp16(input_model_path, output_model_path):
    print(f"🔄 Đang đọc model: {input_model_path}...")
    try:
        model = onnx.load(input_model_path)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {input_model_path}")
        return

    print("⚙️  Đang chuyển đổi sang FP16...")
    # keep_io_types=True: Giữ nguyên kiểu dữ liệu đầu vào/đầu ra là Float32 
    # để không phải sửa code pre-processing trong inference.py
    fp16_model = float16.convert_float_to_float16(model, keep_io_types=True)

    print(f"💾 Đang lưu model mới: {output_model_path}...")
    onnx.save(fp16_model, output_model_path)
    print("✅ Hoàn tất!")

if __name__ == "__main__":
    # 1. Convert Model Phân loại hành vi (GRU/LSTM)
    convert_to_fp16("../weights/gru_fall_model.onnx", "../weights/gru_fall_model_fp16.onnx")

    # 2. (Tùy chọn) Convert Model YOLO Pose nếu bạn đang dùng file ONNX
    # Lưu ý: Nếu bạn export từ Ultralytics, tốt nhất nên dùng lệnh export của họ:
    # yolo export model=yolo11s-pose.pt format=onnx half=True
    # Nhưng nếu bạn chỉ có file onnx, có thể thử convert bằng script này:
    # convert_to_fp16("weights/yolo11s-pose.onnx", "weights/yolo11s-pose_fp16.onnx")