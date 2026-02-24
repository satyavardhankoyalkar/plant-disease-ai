import tensorflow as tf
import numpy as np
from PIL import Image

# Lazy load model (RAM safe)
model = None

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites",
    "Tomato_Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "Tomato_healthy"
]

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model("plant_model.h5", compile=False)
    return model

def predict(image: Image.Image):
    model = load_model()

    image = image.resize((224, 224))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    idx = np.argmax(preds)

    return CLASS_NAMES[idx], float(np.max(preds))