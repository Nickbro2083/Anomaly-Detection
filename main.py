import os
import numpy as np
from tensorflow.keras.models import load_model

from src.preprocess import preprocess_folder, preprocess_test_folder
from src.train_autoencoder import train_autoencoder
from src.detect_anomaly import detect_frame

# -------------------------------
#  1. Đường dẫn dữ liệu
# -------------------------------
train_data_path = "data/Train"
test_data_path = "data/Test"
ground_truth_path = "data/GroundTruth"  # (nếu bạn dùng để đánh giá sau)

# -------------------------------
#  2. Tạo thư mục nếu chưa có
# -------------------------------
os.makedirs("processed/test", exist_ok=True)
os.makedirs("model", exist_ok=True)

# -------------------------------
#  3. Tiền xử lý dữ liệu Train
# -------------------------------
print(" Đang xử lý ảnh Train...")
train_imgs = preprocess_folder(train_data_path)
print(" Train shape:", train_imgs.shape)
np.save("processed/train.npy", train_imgs)

# -------------------------------
#  4. Tiền xử lý dữ liệu Test
# -------------------------------
print(" Đang xử lý ảnh Test...")
test_imgs = preprocess_test_folder(test_data_path)
for seq, imgs in test_imgs.items():
    np.save(f"processed/test/{seq}.npy", imgs)
print(" Đã lưu toàn bộ ảnh test.")

# -------------------------------
#  5. Huấn luyện AutoEncoder
# -------------------------------
print(" Đang huấn luyện mô hình AutoEncoder...")
train_autoencoder(train_imgs, "model/autoencoder_model.h5")
print(" Đã lưu mô hình vào model/autoencoder_model.h5")

# -------------------------------
# 🔍 6. Test thử một ảnh bất kỳ
# -------------------------------
model = load_model("model/autoencoder_model.h5")
sample_frame = test_imgs["Test001"][10]
loss, is_anomaly = detect_frame(sample_frame, model, threshold=0.01)
print(f"\n Test Sample - MSE Loss: {loss:.6f} → {' Bất thường' if is_anomaly else ' Bình thường'}")
