from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io

from app.model import predict

app = FastAPI()


@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    try:
        # Read image safely
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Resize EXACT model size
        img = img.resize((224, 224))

        # Convert to numpy
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0).astype("float32")

        # Predict
        label, confidence = predict(arr)

        return {
            "prediction": label,
            "confidence": confidence
        }

    except Exception as e:
        print("🔥 ERROR:", e)  # shows in Render logs
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )