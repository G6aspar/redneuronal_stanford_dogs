import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path

# ----------------------------
# Configuraciones
# ----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
AUTOTUNE = tf.data.AUTOTUNE

# ✅ RUTA EXPLÍCITA: 'app/model' en la raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent  # Carpeta donde está train.py
MODEL_DIR = PROJECT_ROOT / "app" / "model"      # → ./app/model/

# Crear la carpeta si no existe
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"📁 Modelo y etiquetas se guardarán en: {MODEL_DIR}")

# ----------------------------
# Cargar dataset
# ----------------------------
(ds_train, ds_test), ds_info = tfds.load(
    "stanford_dogs",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

class_names = ds_info.features["label"].names

# Guardar etiquetas en app/model/
with open(MODEL_DIR / "labels.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

# ----------------------------
# Preprocesamiento
# ----------------------------
def preprocess(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = ds_train.map(preprocess, num_parallel_calls=AUTOTUNE)
test_ds = ds_test.map(preprocess, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ----------------------------
# Data Augmentation
# ----------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# ----------------------------
# Modelo: Transfer Learning con MobileNetV2
# ----------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = tf.keras.Sequential([
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------
# Callbacks
# ----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=3)
]

# ----------------------------
# Entrenamiento
# ----------------------------
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ----------------------------
# Guardar modelo EN app/model/
# ----------------------------
model.save(MODEL_DIR / "dog_breed_classifier.h5")
print(f"✅ Modelo guardado correctamente en: {MODEL_DIR}")
