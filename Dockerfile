# Imagen base con TensorFlow y Python 3.10
FROM tensorflow/tensorflow:2.12.0

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer el puerto
EXPOSE 8000

# Comando por defecto para ejecutar Flask como haces con run.py
CMD ["python", "run.py"]

