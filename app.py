import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown

app = Flask(__name__)

# =============================
# MODEL DOWNLOAD FROM DRIVE
# =============================
MODEL_PATH = "plant_model.h5"
DRIVE_ID = "1ioGCHoPyl3sdw16n7ODtWoQuq94wPtWw"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={DRIVE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

print("📦 Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully")

# =============================
# CLASS NAMES
# =============================
CLASS_NAMES = [
    "Apple Scab","Apple Black Rot","Apple Cedar Rust","Apple Healthy",
    "Blueberry Healthy","Cherry Powdery Mildew","Cherry Healthy",
    "Corn Gray Leaf Spot","Corn Common Rust","Corn Healthy",
    "Grape Black Rot","Grape Esca","Grape Leaf Blight","Grape Healthy",
    "Orange Citrus Greening","Peach Bacterial Spot","Peach Healthy",
    "Pepper Bacterial Spot","Pepper Healthy","Potato Early Blight",
    "Potato Late Blight","Potato Healthy","Strawberry Leaf Scorch",
    "Strawberry Healthy","Tomato Bacterial Spot","Tomato Early Blight",
    "Tomato Late Blight","Tomato Leaf Mold","Tomato Septoria Leaf Spot",
    "Tomato Spider Mites","Tomato Target Spot","Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus","Tomato Healthy"
]

IMG_SIZE = 380

# =============================
# ROUTES
# =============================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    img_path = None
    error = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "No file uploaded"
        else:
            file = request.files["image"]

            if file.filename == "":
                error = "No file selected"
            else:
                upload_folder = "static/uploads"
                os.makedirs(upload_folder, exist_ok=True)

                img_path = os.path.join(upload_folder, file.filename)
                file.save(img_path)

                try:
                    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                    img_array = image.img_to_array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    preds = model.predict(img_array)[0]
                    class_index = np.argmax(preds)
                    confidence = round(float(np.max(preds)) * 100, 2)
                    prediction = CLASS_NAMES[class_index]

                except Exception as e:
                    error = f"Prediction error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        img_path=img_path,
        error=error
    )

# =============================
# RENDER COMPATIBLE RUN
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)