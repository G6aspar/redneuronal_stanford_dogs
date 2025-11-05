# predict.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path
from PIL import Image
import sys

# ----------------------------
# Configuración
# ----------------------------
IMG_SIZE = 224  # Tamaño esperado por el modelo (el mismo usado en entrenamiento)

# ✅ RUTA CORREGIDA: leer desde app/model/
PROJECT_ROOT = Path(__file__).resolve().parent  # Carpeta donde está predict.py
MODEL_DIR = PROJECT_ROOT / "app" / "model"      # → ./app/model/

MODEL_PATH = MODEL_DIR / "dog_breed_classifier.h5"  # Ruta del modelo entrenado
LABELS_PATH = MODEL_DIR / "labels.txt"              # Archivo con nombres de razas

# ----------------------------
# Cargar modelo y labels
# ----------------------------
print("📂 Cargando modelo desde:", MODEL_DIR)
model = load_model(MODEL_PATH)  # Cargamos el modelo previamente entrenado

# Leemos los nombres de las clases (razas) desde labels.txt y los guardamos en una lista
with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f]

# ----------------------------
# Función para preprocesar imagen
# ----------------------------
def preprocess_image(image_path):
    # Abre la imagen, la convierte a RGB (por si viene en otro formato como PNG con transparencia)
    img = Image.open(image_path).convert("RGB")
    # Redimensiona la imagen al tamaño que espera el modelo (224x224)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    # Convierte la imagen a array NumPy y normaliza los valores (0-255 → 0-1)
    img_array = np.array(img) / 255.0
    # Añade una dimensión extra al inicio para que el modelo lo interprete como batch (shape: (1, 224, 224, 3))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# Función para predecir
# ----------------------------
def predict_image(image_path):
    img_array = preprocess_image(image_path)  # Preprocesamos la imagen
    predictions = model.predict(img_array)[0]  # Realizamos la predicción y tomamos el primer (y único) elemento

    # Obtenemos el índice de la clase con mayor probabilidad
    top_idx = np.argmax(predictions)
    top_class = class_names[top_idx].split('-', 1)[1] if '-' in class_names[top_idx] else class_names[top_idx]  # Nombre de la clase más probable
    top_conf = predictions[top_idx] * 100  # Convertimos a porcentaje

    print(f"\n📸 Imagen: {image_path}")
    print(f"✅ Predicción principal: {top_class} ({top_conf:.2f}%)")

    # Mostramos el Top 5 de clases con mayor probabilidad
    print("\n🔝 Top 5 predicciones:")
    top_5_idx = predictions.argsort()[-5:][::-1]  # Ordenamos índices del más alto al más bajo
    for idx in top_5_idx:
        print(f"{class_names[idx]}: {predictions[idx]*100:.2f}%")

# ----------------------------
# Ejecutar
# ----------------------------
# Si no se pasan imágenes como argumento, mostramos mensaje de uso
if len(sys.argv) < 2:
    print("❌ Uso: py predict.py imagen1.jpg [imagen2.jpg ...]")
else:
    # Iteramos por cada imagen pasada como argumento y mostramos la predicción
    for img_path in sys.argv[1:]:
        predict_image(img_path)