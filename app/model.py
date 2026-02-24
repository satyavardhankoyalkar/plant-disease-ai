import json
import os
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "plant_model.h5"
LABELS_PATH = "labels.json"

model = None
labels = None

def load_once():
    global model, labels

    if model is None:
        print("🌿 Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    if labels is None:
        with open(LABELS_PATH) as f:
            labels = json.load(f)

def predict_image(image_bytes):
    load_once()

    img = Image.open(image_bytes).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds) * 100)

    return labels[idx], confidence