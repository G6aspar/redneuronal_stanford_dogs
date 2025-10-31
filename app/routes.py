# app/routes.py
from flask import Blueprint, render_template, request, jsonify, current_app
from PIL import Image
import numpy as np
import uuid
import csv
from pathlib import Path
import pandas as pd
import tensorflow as tf

bp = Blueprint("main", __name__)

IMG_SIZE = 224

# Rutas para feedback y desconocidos
MODEL_DIR = Path(__file__).resolve().parent / "model"
FEEDBACK_DIR = MODEL_DIR / "feedback"
UNKNOWN_DIR = MODEL_DIR / "unknown"

FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
UNKNOWN_DIR.mkdir(exist_ok=True)

IMAGES_DIR = FEEDBACK_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

FEEDBACK_CSV = FEEDBACK_DIR / "feedback.csv"
if not FEEDBACK_CSV.exists():
    with open(FEEDBACK_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label"])


def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


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
        top_class = current_app.class_names[top_idx]
        confidence = float(preds[top_idx])

        return jsonify({
            "breed": top_class,
            "confidence": round(confidence * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": f"Error al procesar la imagen: {str(e)}"}), 500


@bp.route("/class_names")
def get_class_names():
    return jsonify(current_app.class_names)


@bp.route("/correct", methods=["POST"])
def correct():
    if "file" not in request.files or "true_label" not in request.form:
        return jsonify({"error": "Falta imagen o etiqueta correcta"}), 400

    file = request.files["file"]
    true_label_input = request.form["true_label"].strip()

    if not true_label_input:
        return jsonify({"error": "La etiqueta no puede estar vacía"}), 400

    # Buscar coincidencia insensible a mayúsculas/espacios
    matching_labels = [
        name for name in current_app.class_names
        if name.lower().strip() == true_label_input.lower().strip()
    ]

    if not matching_labels:
        # Sugerir razas similares
        sugerencias = [
            name for name in current_app.class_names
            if true_label_input.lower().strip() in name.lower()
        ][:5]
        return jsonify({
            "error": f"Raza '{true_label_input}' no encontrada en el conjunto soportado.",
            "sugerencias": sugerencias
        }), 400

    true_label = matching_labels[0]  # Nombre canónico

    # Guardar imagen
    ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    img_name = f"{uuid.uuid4().hex}.{ext}"
    img_path = IMAGES_DIR / img_name
    file.save(img_path)

    # Guardar en CSV
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([str(img_path), true_label])

    # Reentrenar
    retrain_with_feedback(current_app)

    return jsonify({"message": f"✅ Corregido a: {true_label}"}), 200


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


def retrain_with_feedback(app, batch_size=8, epochs=1):
    """Reentrena el modelo con los ejemplos de feedback acumulados."""
    if not FEEDBACK_CSV.exists():
        return

    try:
        df = pd.read_csv(FEEDBACK_CSV)
        if df.empty or len(df) == 0:
            return
    except Exception as e:
        print(f"⚠️ Error leyendo feedback.csv: {e}")
        return

    images = []
    labels = []

    for _, row in df.iterrows():
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img) / 255.0
            images.append(img_array)

            label_idx = app.class_names.index(row["true_label"])
            labels.append(label_idx)
        except Exception as e:
            print(f"⚠️ Error procesando {row['image_path']}: {e}")
            continue

    if not images:
        return

    x_feedback = np.array(images)
    y_feedback = np.array(labels)

    # Compilar con learning rate muy bajo
    app.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Entrenar
    app.model.fit(
        x_feedback,
        y_feedback,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    # Guardar modelo actualizado
    MODEL_PATH = MODEL_DIR / "dog_breed_classifier.h5"
    app.model.save(MODEL_PATH)

    print(f"✅ Modelo reentrenado con {len(images)} ejemplos de feedback.")