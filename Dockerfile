# Base image with Python 3.9
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY model1/train6/weights/best.onnx model1/train6/weights/best.onnx
COPY labels.txt labels.txt
COPY fastapi_detect_face.py fastapi_detect_face.py
COPY utils.py utils.py
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "fastapi_detect_face:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# docker build -t yolov12-api .
# docker run -p 8000:8000 yolov12-api

