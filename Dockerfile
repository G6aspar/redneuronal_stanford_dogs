# Imagen base con TensorFlow y Python 3.10
FROM tensorflow/tensorflow:2.12.0

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos requirements.txt e instalamos dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del proyecto
COPY app/ ./app
COPY run.py .

# Exponemos el puerto 8000
EXPOSE 8000

# Comando para iniciar Flask
CMD ["python", "run.py"]
