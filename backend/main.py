from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Surface Crack Detection API",
    description="Upload an image of a concrete surface; returns crack or no_crack.",
    version="1.0.0",
)

# Enable CORS for frontend calls; restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class labels and classification threshold
class_names = os.environ.get("CLASS_NAMES", "no_crack,crack").split(",")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

# Load Keras model from environment or default filename
model_path = os.environ.get("MODEL_FILENAME", "model/ResNet50V2_model.keras")
if not os.path.exists(model_path):
    logger.error(f"Model file {model_path} not found")
    raise RuntimeError(f"Model file {model_path} not found")
logger.info(f"Loading model from {model_path}…")
model = tf.keras.models.load_model(model_path)
logger.info("Model loaded successfully.")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Resize image to 224×224, convert to float32 and normalize to [0,1].
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.error(f"Error opening image: {e}")
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    img = img.resize((224, 224))
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(x, axis=0)

class PredictionResponse(BaseModel):
    label: str
    prob: float

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Predict whether the uploaded image contains a crack.
    Handles both binary sigmoid output (shape (1,1)) and
    softmax output for 2 classes (shape (1,2)).
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided.")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")

    img_bytes = await file.read()
    x = preprocess_image(img_bytes)

    # run predict in a thread pool to avoid blocking the event loop
    preds = await run_in_threadpool(model.predict, x, verbose=0)

    preds = np.asarray(preds)
    if preds.ndim != 2 or preds.shape[0] != 1:
        raise HTTPException(status_code=500, detail=f"Unexpected prediction shape: {preds.shape}")

    if preds.shape[1] == 1:
        # binary sigmoid: probability of positive class (crack)
        p_positive = float(preds[0, 0])
        if p_positive >= THRESHOLD:
            label = class_names[1]  # crack
            prob = p_positive
        else:
            label = class_names[0]  # no_crack
            prob = 1.0 - p_positive
    elif preds.shape[1] == 2:
        # softmax: choose higher probability between two classes
        idx = int(np.argmax(preds[0]))
        label = class_names[idx]
        prob = float(preds[0, idx])
    else:
        raise HTTPException(status_code=500, detail=f"Unsupported number of classes: {preds.shape[1]}")

    return PredictionResponse(label=label, prob=prob)
