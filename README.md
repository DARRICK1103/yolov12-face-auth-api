# RealFace YOLOv12 API

🚀 A high-performance FastAPI backend for real human face detection using YOLOv12, built to distinguish real faces from photos (e.g., printed or digital attacks). Easily deployable via Docker.

## Features

- 🧠 Real vs fake face verification using YOLOv12
- ⚡ Fast inference with ONNX support
- 📦 Dockerized for easy deployment
- 🔁 REST API with FastAPI backend
- 📂 Accepts image uploads and returns detection results
- 🛡️ Suitable for anti-spoofing, face authentication systems

## Use Cases

- 👤 Access control systems (door entry, kiosks)
- 📸 ID photo verification
- 🧾 Online registration anti-fraud
- 🎥 Real-time webcam monitoring (extendable)

## Tech Stack

- YOLOv12 (ONNX format)
- FastAPI
- OpenCV
- Docker

## Quickstart (Docker)

```bash
# Build Docker image
docker build -t realface-api .

# Run the container
docker run -p 8000:8000 realface-api
