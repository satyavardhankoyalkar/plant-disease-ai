# AI Plant Disease Detector (V2)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-AI-orange)

Version 2 — Optimized with MobileNet and Streamlit UI.

An AI-powered web application that detects plant diseases from leaf images using deep learning.  
This version uses a lightweight MobileNet model and a clean Streamlit interface for fast, real-time predictions.

<img width="748" height="786" alt="image" src="https://github.com/user-attachments/assets/7ea7e5c8-b9b1-4582-8f31-6ee1a624607d" />

<img width="498" height="833" alt="Screenshot 2026-02-24 141757" src="https://github.com/user-attachments/assets/da3c7d8b-92a0-45de-ae40-158bcf9337ac" />

---

## Features

- Upload leaf image and detect disease instantly




- MobileNet-based deep learning model
- Confidence score with probability breakdown
- Fast and lightweight inference
- Supports multiple crops and diseases

---

## Supported Plants

### Tomato (10 classes)
- Bacterial Spot  
- Early Blight  
- Late Blight  
- Leaf Mold  
- Septoria Leaf Spot  
- Spider Mites  
- Target Spot  
- Yellow Leaf Curl Virus  
- Mosaic Virus  
- Healthy  

### Potato (3 classes)
- Early Blight  
- Late Blight  
- Healthy  

### Pepper Bell (2 classes)
- Bacterial Spot  
- Healthy  

---

## Tech Stack

- Frontend/UI: Streamlit  
- ML Model: TensorFlow / Keras (MobileNetV2)  
- Image Processing: Pillow, NumPy  
- Language: Python  

---

## Demo

Upload a plant leaf image and get instant disease prediction.

Example output:

Prediction: Tomato Late Blight  
Confidence: 97.3%

---

## Project Structure
ai-plant-disease-detector/
│
├── app.py
├── plant_model.h5
├── requirements.txt
└── README.md

---

## Installation and Setup

### 1. Clone the repository


---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/satyavardhankoyalkar/ai-plant-disease-detector.git
cd ai-plant-disease-detector   bash```
```
### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the app
```bash
streamlit run app.py
```
Open in browser:
http://localhost:8501
---
## ⚙️ How It Works
User uploads a leaf image

Image resized to 224×224

MobileNet model predicts disease class

Result and confidence displayed
---
## 🚀 Version 2 Improvements

🔄 Migrated from Flask → Streamlit

⚡ Lightweight MobileNet model

📉 Reduced deployment complexity

🎨 Cleaner UI and UX

📊 Class probability visualization
---
## 🔮 Future Improvements
🌍 Multi-language support

📱 Mobile optimization

☁️ HuggingFace deployment

🌿 Disease treatment suggestions

🔥 Grad-CAM visualizations

---
## ⚠️ Disclaimer

This project is built for educational and portfolio purposes.
Model performance may vary on real-world images.
--
## 👨‍💻 Author
Satya 🌿
AI Builder | Full Stack Learner | Future ML Engineer 🚀
---
## ⭐ Support
If you found this project useful, consider giving it a ⭐ on GitHub.

