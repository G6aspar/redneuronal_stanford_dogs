# app/routes.py
from flask import Blueprint, render_template, request, jsonify, current_app
from PIL import Image
import numpy as np
import uuid
from pathlib import Path

bp = Blueprint("main", __name__)

IMG_SIZE = 224

# Ruta para imágenes desconocidas
MODEL_DIR = Path(__file__).resolve().parent / "model"
UNKNOWN_DIR = MODEL_DIR / "unknown"
UNKNOWN_DIR.mkdir(exist_ok=True)


def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def clean_breed_name(name):
    """Convierte 'n02105056-groenendael' en 'Groenendael'."""
    if '-' in name:
        clean = name.split('-', 1)[1].replace('_', ' ')
    else:
        clean = name.replace('_', ' ')
    return clean.title()


@bp.route("/")
def home():
    return render_template("index.html")


@bp.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    file = request.files["file"]
    try:
        img = Image.open(file.stream)
        tensor = preprocess_image(img)
        preds = current_app.model.predict(tensor)[0]
        top_idx = int(np.argmax(preds))
        raw_class = current_app.class_names[top_idx]
        top_class = clean_breed_name(raw_class)
        confidence = float(preds[top_idx])

        return jsonify({
            "breed": top_class,
            "confidence": round(confidence * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": f"Error al procesar la imagen: {str(e)}"}), 500


@bp.route("/class_names")
def get_class_names():
    clean_names = [clean_breed_name(name) for name in current_app.class_names]
    return jsonify(clean_names)


@bp.route("/report_unknown", methods=["POST"])
def report_unknown():
    if "file" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    file = request.files["file"]
    filename = file.filename.lower()
    if not (filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg')):
        return jsonify({"error": "Solo se aceptan imágenes en formato PNG o JPG"}), 400

    ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    img_name = f"{uuid.uuid4().hex}.{ext}"
    img_path = UNKNOWN_DIR / img_name
    file.save(img_path)

    return jsonify({
        "message": "¡Gracias! Tu imagen se guardó para futuras mejoras del modelo."
    }), 200