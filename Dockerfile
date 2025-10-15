# Usamos la imagen oficial de TensorFlow con Python
FROM tensorflow/tensorflow:2.12.0

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos requirements.txt y lo instalamos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos la carpeta backend
COPY backend/ ./backend

# Copiamos la carpeta artifacts (modelo y labels)
COPY artifacts/ ./artifacts

# Copiamos el frontend si lo necesitas
COPY frontend/ ./frontend

# Exponemos el puerto 8000
EXPOSE 8000

# Comando por defecto para ejecutar la app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]