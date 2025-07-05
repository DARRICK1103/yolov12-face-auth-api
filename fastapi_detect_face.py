import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import onnxruntime as ort
import numpy as np
from PIL import Image
from io import BytesIO

from utils import preprocess_image, postprocess_yolov12_output, load_class_names

app = FastAPI(title="YOLOv12 ONNX Inference API")

MODEL_PATH = "model1/train6/weights/best.onnx"
LABELS_PATH = "labels.txt"
INPUT_SHAPE = (640, 640)
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45

ort_session = None
class_names = []

@app.on_event("startup")
async def load_model():
    global ort_session, class_names
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    
    if os.path.exists(LABELS_PATH):
        class_names = load_class_names(LABELS_PATH)
    else:
        print("Warning: labels.txt not found.")
        class_names = None

    ort_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print("Model loaded successfully.")

@app.get("/")
def root():
    return {"message": "Welcome to the YOLOv12 Inference API."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    try:
        image_bytes = await file.read()
        original_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        orig_width, orig_height = original_image.size

        input_tensor = preprocess_image(image_bytes, INPUT_SHAPE)

        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name

        raw_output = ort_session.run([output_name], {input_name: input_tensor})[0]

        detections = postprocess_yolov12_output(
            raw_output,
            (orig_height, orig_width),
            conf_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD
        )

        if class_names:
            for det in detections:
                if det['class_id'] < len(class_names):
                    det['class_name'] = class_names[det['class_id']]
                else:
                    det['class_name'] = f"class_{det['class_id']}"

        return JSONResponse(content={"detections": detections})
    
    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
