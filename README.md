#  Pneumonia Detection System

AI-powered web application for detecting Pneumonia from Chest X-Ray images using Deep Learning and Transfer Learning.

---

##  Project Overview

This project uses Transfer Learning with MobileNetV2 to classify Chest X-Ray images into:

- NORMAL
- PNEUMONIA

The trained deep learning model is integrated into a Flask web application for real-time predictions.

---

##  Technologies Used

- Python 3.10
- TensorFlow / Keras
- Flask
- NumPy
- Pillow (PIL)

---

##  Dataset

Dataset Used:

Chest X-Ray Images (Pneumonia)  
Source: Kaggle

Classes:

- NORMAL
- PNEUMONIA

---

##  Model Architecture

- Base Model: MobileNetV2 (Pretrained on ImageNet)
- Transfer Learning
- GlobalAveragePooling Layer
- Dense Layers
- Sigmoid Output (Binary Classification)

---

##  Features

✔ Upload Chest X-Ray Image  
✔ AI-based Prediction  
✔ Confidence Score  
✔ Modern UI  
✔ Real-time Analysis  

---

## ⚙️ Installation

Clone the repository:

git clone https://github.com/yourusername/pneumonia-detection.git  
cd pneumonia-detection

Create virtual environment:

py -3.10 -m venv venv  
venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

---

## ▶ Run Application

python app.py

Open in browser:

http://127.0.0.1:5000

---

##  Usage

1. Upload Chest X-Ray Image  
2. Click Analyze Image  
3. View Prediction & Confidence Score  

---

## ⚠ Disclaimer

This project is for educational and research purposes only.

It is NOT intended for medical diagnosis.

---

## 👨‍💻 Author

Developed as part of an academic Deep Learning project.
