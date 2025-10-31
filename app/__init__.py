# _init_.py
from flask import Flask
from tensorflow.keras.models import load_model
from pathlib import Path

def create_app():
    app = Flask(__name__)

    # Paths del modelo
    MODEL_DIR = Path(__file__).resolve().parent / "model"
    MODEL_PATH = MODEL_DIR / "dog_breed_classifier.h5"
    LABELS_PATH = MODEL_DIR / "labels.txt"

    # Cargar modelo y etiquetas
    app.model = load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        app.class_names = [line.strip() for line in f]

    # Registrar rutas
    from .routes import bp
    app.register_blueprint(bp)

    return app
