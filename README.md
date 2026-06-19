# Anomaly Detection
Download dataset here: https://www.kaggle.com/datasets/karthiknm1/ucsd-anomaly-detection-dataset

Tools used: Python, PyTorch, OpenCV, NumPy, Matplotlib

Deep Learning Architecture used: U-Net 

# Summarize
<p align="justify">This project focuses on anomaly detection in surveillance videos using deep learning techniques. The goal is to automatically identify unusual events that deviate from normal patterns in video footage, such as bicycles, vehicles, or other unexpected objects appearing in pedestrian-only areas.</p>

<p align="justify">The model is trained on the UCSD Anomaly Detection Dataset, a widely used benchmark dataset for video anomaly detection research. By learning the characteristics of normal scenes, the system can detect and localize abnormal events in previously unseen video frames.</p>

# About this project
<p align="justify">This project presents a deep learning approach for anomaly detection in surveillance videos using the UCSD Anomaly Detection Dataset. The primary objective is to automatically identify unusual events that differ from normal scene behavior, enabling intelligent monitoring in security and public surveillance applications.</p>

<p align="justify">The proposed system is trained exclusively on normal video sequences, allowing the model to learn typical spatial patterns and scene characteristics. A U-Net-based architecture is employed to reconstruct input frames, and anomalies are detected by analyzing reconstruction errors when the model encounters abnormal objects or activities that were not present during training.</p>

<p align="justify">To evaluate the model's performance, experiments were conducted on unseen test videos containing both normal and anomalous events. Ground-truth anomaly masks were used to assess the system's ability to localize abnormal regions accurately. The results demonstrate that the model can effectively distinguish anomalous events from normal activities and highlight suspicious areas within video frames.</p>

<p align="justify">This project showcases the potential of deep learning techniques for automated video surveillance and provides a foundation for future research on real-time anomaly detection and intelligent security systems.</p>
