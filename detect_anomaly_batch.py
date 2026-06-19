import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -------------------------------
# Cấu hình
# -------------------------------
model_path = "model/autoencoder_model.h5"
test_data_dir = "processed/test"
threshold = 0.00108  # Ngưỡng phát hiện bất thường

# -------------------------------
# Load model
# -------------------------------
model = load_model(model_path)
print(" Đã load mô hình:", model_path)

# -------------------------------
# Hàm hiển thị kết quả
# -------------------------------
def display_sequence(sequence_name, threshold):
    sequence_path = os.path.join(test_data_dir, f"{sequence_name}.npy")
    data = np.load(sequence_path)
    print(f"\n Đang xử lý chuỗi: {sequence_name} (số frame: {len(data)})")

    plt.figure(figsize=(5, 5))

    for i, frame in enumerate(data):
        frame_input = np.expand_dims(frame, axis=0)
        reconstructed = model.predict(frame_input, verbose=0)
        loss = np.mean((frame_input - reconstructed) ** 2)

        is_anomaly = loss > threshold

        plt.clf()
        plt.imshow(frame.squeeze(), cmap='gray')
        plt.title(f"{sequence_name} - Frame {i} - {' BẤT THƯỜNG' if is_anomaly else ' Bình thường'}\nMSE: {loss:.6f}")
        plt.axis('off')
        plt.pause(0.1)  # Hiển thị 0.3 giây mỗi frame

# -------------------------------
# Chạy trên 1 chuỗi test cụ thể
# -------------------------------
sequence_to_display = "Test007"  # Bạn có thể đổi sang Test002, Test005,...
display_sequence(sequence_to_display, threshold)
