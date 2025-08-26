import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ----------------------------
# Configuración
# ----------------------------
IMG_SIZE = 224
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS / "dog_breed_classifier.h5"
LABELS_PATH = ARTIFACTS / "labels.txt"

# ----------------------------
# Cargar modelo y etiquetas
# ----------------------------
print("📂 Cargando modelo...")
model = load_model(MODEL_PATH)
print("✅ Modelo cargado")

with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ----------------------------
# Función para preprocesar imagen
# ----------------------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# ----------------------------
# Predicción
# ----------------------------
img_path = input("📷 Ingrese la ruta de la imagen: ").strip()
if not Path(img_path).exists():
    raise FileNotFoundError(f"No se encontró la imagen: {img_path}")

img_array = preprocess_image(img_path)

predictions = model.predict(img_array)
top_5_indices = predictions[0].argsort()[-5:][::-1]  # Mejores 5
top_5_labels = [(class_names[i], predictions[0][i]) for i in top_5_indices]

print("\n✅ Resultados de predicción:")
for label, prob in top_5_labels:
    print(f"{label}: {prob:.4f}")

