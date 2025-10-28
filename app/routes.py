from flask import Blueprint, render_template, request, jsonify, current_app
from PIL import Image
import numpy as np

bp = Blueprint("main", __name__)

IMG_SIZE = 224

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
