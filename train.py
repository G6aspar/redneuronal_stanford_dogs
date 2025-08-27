import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path

# ----------------------------
# Configuraciones
# ----------------------------
IMG_SIZE = 224  # Tamaño al que se redimensionarán las imágenes (224x224, estándar en modelos preentrenados)
BATCH_SIZE = 32  # Cantidad de imágenes que se procesan en cada batch durante el entrenamiento
EPOCHS = 20  # Número máximo de épocas para entrenar el modelo
AUTOTUNE = tf.data.AUTOTUNE  # Permite que TensorFlow optimice automáticamente el rendimiento del pipeline

# Carpeta donde se guardarán el modelo y las etiquetas
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)  # Crea la carpeta si no existe

# ----------------------------
# Cargar dataset desde TFDS
# ----------------------------
# Descarga y carga el dataset "stanford_dogs" desde TensorFlow Datasets
# split=["train", "test"]: obtenemos los datos de entrenamiento y prueba
# as_supervised=True: entrega los datos como pares (imagen, etiqueta)
# with_info=True: devuelve información adicional del dataset (como nombres de clases)
(ds_train, ds_test), ds_info = tfds.load(
    "stanford_dogs",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

# Obtenemos los nombres de las clases (razas de perros)
class_names = ds_info.features["label"].names

# Guardamos las etiquetas en un archivo de texto para usarlas luego en predicciones
with open(ARTIFACTS / "labels.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

# ----------------------------
# Preprocesamiento
# ----------------------------
# Función para preprocesar las imágenes:
# - Redimensiona a 224x224
# - Convierte a float32
# - Normaliza los valores (0 a 1)
def preprocess(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Aplicamos la función de preprocesamiento a los datasets
train_ds = ds_train.map(preprocess, num_parallel_calls=AUTOTUNE)
test_ds = ds_test.map(preprocess, num_parallel_calls=AUTOTUNE)

# Preparamos el dataset para entrenamiento:
# - Barajamos (shuffle)
# - Agrupamos en batches
# - Prefetch para mejorar la velocidad (carga en paralelo mientras entrena)
train_ds = (train_ds
            .shuffle(1000)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

test_ds = (test_ds
           .batch(BATCH_SIZE)
           .prefetch(AUTOTUNE))

# ----------------------------
# Data Augmentation
# ----------------------------
# Creamos una secuencia de transformaciones aleatorias para aumentar la variabilidad de los datos
# Esto ayuda a reducir el sobreajuste
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),  # Invierte imágenes horizontalmente
    tf.keras.layers.RandomRotation(0.1),  # Rotaciones aleatorias
    tf.keras.layers.RandomZoom(0.1),  # Zoom aleatorio
])

# ----------------------------
# Modelo (Transfer Learning con MobileNetV2)
# ----------------------------
# Cargamos MobileNetV2 preentrenado en ImageNet, sin la capa final (include_top=False)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,  # No incluimos la capa de clasificación original
    weights="imagenet"  # Usamos pesos preentrenados
)
base_model.trainable = False  # Congelamos la base para no entrenar sus pesos (solo entrenamos las capas superiores)

# Definimos el modelo final
model = tf.keras.Sequential([
    data_augmentation,  # Aumento de datos
    base_model,  # Base convolucional preentrenada
    tf.keras.layers.GlobalAveragePooling2D(),  # Reduce el mapa de características a un vector
    tf.keras.layers.Dropout(0.3),  # Dropout para evitar sobreajuste
    tf.keras.layers.Dense(len(class_names), activation="softmax")  # Capa final con tantas salidas como clases
])

# Compilamos el modelo con Adam y pérdida adecuada para clasificación multiclase
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",  # Para etiquetas enteras (no one-hot)
    metrics=["accuracy"]
)

# ----------------------------
# Callbacks
# ----------------------------
# EarlyStopping: detiene el entrenamiento si no mejora en 5 épocas
# ReduceLROnPlateau: reduce el learning rate si no mejora en 3 épocas
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=3)
]

# ----------------------------
# Entrenamiento
# ----------------------------
# Entrenamos el modelo usando train_ds y validamos con test_ds
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ----------------------------
# Guardar modelo
# ----------------------------
# Guardamos el modelo entrenado en formato HDF5 (.h5)
model.save(ARTIFACTS / "dog_breed_classifier.h5")
print("✅ Modelo guardado en:", ARTIFACTS / "dog_breed_classifier.h5")
