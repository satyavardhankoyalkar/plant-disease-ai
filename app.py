import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

print("Loading model...")
model = tf.saved_model.load("saved_model")
infer = model.signatures["serving_default"]
print("Model loaded!")

model = tf.keras.models.load_model("saved_model", compile=False)

labels = [
  "Pepper Bell Bacterial Spot",
  "Pepper Bell Healthy",
  "Potato Early Blight",
  "Potato Late Blight",
  "Potato Healthy",
  "Tomato Bacterial Spot",
  "Tomato Early Blight",
  "Tomato Late Blight",
  "Tomato Leaf Mold",
  "Tomato Septoria Leaf Spot",
  "Tomato Spider Mites",
  "Tomato Target Spot",
  "Tomato Mosaic Virus",
  "Tomato Yellow Curl Virus",
  "Tomato Healthy"
]

def predict(img):
    try:
        img = img.resize((224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0).astype("float32")

        outputs = infer(tf.constant(arr))

        # 🔥 SAFE extraction
        preds = None
        for v in outputs.values():
            preds = v.numpy()[0]

        idx = np.argmax(preds)
        confidence = preds[idx] * 100

        return f"{labels[idx]} ({confidence:.2f}%)"

    except Exception as e:
        return f"ERROR: {str(e)}"

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🌿 Plant Disease Detector"
).launch()