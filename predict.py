import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path
from PIL import Image
import sys

# ----------------------------
# Configuración
# ----------------------------
IMG_SIZE = 224
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS / "dog_breed_classifier.h5"
LABELS_PATH = ARTIFACTS / "labels.txt"

# ----------------------------
# Cargar modelo y labels
# ----------------------------
print("📂 Cargando modelo...")
model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f]

# ----------------------------
# Función para preprocesar imagen
# ----------------------------
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0  # normaliza
    img_array = np.expand_dims(img_array, axis=0)  # batch de 1
    return img_array

# ----------------------------
# Función para predecir
# ----------------------------
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)[0]

    # Índice con mayor probabilidad
    top_idx = np.argmax(predictions)
    top_class = class_names[top_idx]
    top_conf = predictions[top_idx] * 100

    print(f"\n📸 Imagen: {image_path}")
    print(f"✅ Predicción principal: {top_class} ({top_conf:.2f}%)")

    # Top 5 predicciones
    print("\n🔝 Top 5 predicciones:")
    top_5_idx = predictions.argsort()[-5:][::-1]
    for idx in top_5_idx:
        print(f"{class_names[idx]}: {predictions[idx]*100:.2f}%")

# ----------------------------
# Ejecutar
# ----------------------------
if len(sys.argv) < 2:
    print("❌ Uso: py predict.py imagen1.jpg [imagen2.jpg ...]")
else:
    for img_path in sys.argv[1:]:
        predict_image(img_path)
