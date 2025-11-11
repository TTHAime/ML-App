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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    ปรับภาพให้สอดคล้องกับ PyTorch transforms:
    Resize(shorter side=256) -> CenterCrop(224) -> Grayscale(3) -> Normalize.
    """
    # โหลดและแปลงภาพเป็น RGB
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize: ย่อขนาดโดยรักษาอัตราส่วนให้ด้านสั้นเป็น 256 พิกเซล
    short_side = 256
    w, h = img.size
    if w < h:
        new_w, new_h = short_side, int(h * short_side / w)
    else:
        new_h, new_w = short_side, int(w * short_side / h)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # CenterCrop 224x224
    left = (img.width - 224) / 2
    top = (img.height - 224) / 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))

    # Grayscale to 3-channel: แปลงเป็นภาพขาวดำหนึ่งช่องแล้วซ้ำเป็น 3 ช่อง
    gray = img.convert("L")
    img = Image.merge("RGB", (gray, gray, gray))

    # ToTensor + Normalize: แปลงเป็นอาร์เรย์ float32 ช่วง [0,1]
    x = np.asarray(img, dtype=np.float32) / 255.0
    # Normalize per-channel ตาม ImageNet:contentReference[oaicite:1]{index=1}
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
   
    # เพิ่มมิติ batch (1, 224, 224, 3)
    return x[None, ...].astype(np.float32)

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
