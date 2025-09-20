from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# ---------- Configuración ----------
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS / "dog_breed_classifier.h5"
LABELS_PATH = ARTIFACTS / "labels.txt"
IMG_SIZE = 224

# ---------- Cargar modelo y clases ----------
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f]

# ---------- Inicializar FastAPI ----------
app = FastAPI()

# Permitir requests desde cualquier origen (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Funciones ----------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------- Endpoint ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file)
    tensor = preprocess_image(img)
    preds = model.predict(tensor)[0]
    top_idx = int(np.argmax(preds))
    top_class = class_names[top_idx]
    confidence = float(preds[top_idx])
    return {
        "breed": top_class,
        "confidence": round(confidence * 100, 2)
    }
