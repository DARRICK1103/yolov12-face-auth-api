import cv2
import numpy as np
from PIL import Image
from io import BytesIO

def preprocess_image(image_bytes: bytes, input_shape=(640, 640)) -> np.ndarray:
    """
    Preprocesses an image for YOLOv12 ONNX inference.
    Resizes, normalizes, and transposes the image.
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize(input_shape)
    image_np = np.array(image, dtype=np.float32)

    # Normalize to [0, 1]
    image_np /= 255.0

    # Transpose to (C, H, W) for YOLO models
    image_np = np.transpose(image_np, (2, 0, 1))

    # Add batch dimension
    image_np = np.expand_dims(image_np, axis=0)

    return image_np

def postprocess_yolov12_output(output: np.ndarray, original_image_shape: tuple, conf_threshold=0.25, iou_threshold=0.45):
    """
    Post-processes the raw output from a YOLOv8 ONNX model with (1, 6, 8400) shape.

    Args:
        output (np.ndarray): The raw output array from the ONNX model inference.
                             Expected shape: (1, 6, 8400).
        original_image_shape (tuple): The original image dimensions (height, width).
        conf_threshold (float): Minimum confidence score to consider a detection.
        iou_threshold (float): IoU threshold for Non-Maximum Suppression.

    Returns:
        list: A list of dictionaries, where each dictionary represents a final detection
              with 'box' ([x1, y1, x2, y2]), 'score', and 'class_id'.
    """
    # 1. Adapt output shape: (1, 6, 8400) -> (1, 8400, 6)
    # This aligns with a more standard "N_detections, Attributes" layout
    output = np.transpose(output, (0, 2, 1)) # Becomes (1, 8400, 6)

    # Remove batch dimension
    output = output[0] # Becomes (8400, 6)

    img_h, img_w = original_image_shape # e.g., (480, 640)
    input_h, input_w = 640, 640         # Model's expected input size

    final_results = []
    
    # Extract boxes and scores based on the new shape
    # YOLOv8 typically outputs [x_center, y_center, width, height, class_score_0, class_score_1, ...]
    boxes_raw = output[:, :4]        # All rows, first 4 columns for box coords
    class_scores_raw = output[:, 4:] # All rows, remaining columns for class scores

    # No need for softmax if the ONNX model already applies it or uses logits directly
    # For YOLOv8 ONNX exports, class_scores_raw are often already probabilities (or close to it)
    # The `conf_threshold` will filter based on these.

    # Iterate through predictions to filter by confidence and prepare for NMS
    nms_boxes = [] # Will store [x1, y1, w, h] for NMS
    nms_scores = []
    nms_class_ids = []

    num_detections = output.shape[0] # 8400
    num_classes = class_scores_raw.shape[1] # Should be 2 in your case

    for i in range(num_detections):
        x_center, y_center, width, height = boxes_raw[i]
        class_scores = class_scores_raw[i]

        class_id = int(np.argmax(class_scores))
        confidence = class_scores[class_id] # Use the highest class score as confidence

        if confidence < conf_threshold:
            continue

        # Convert (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
        # These are relative to the model's input size (640x640)
        x1_model = x_center - width / 2
        y1_model = y_center - height / 2
        x2_model = x_center + width / 2
        y2_model = y_center + height / 2

        # Scale coordinates back to original image dimensions (img_w x img_h)
        x1_orig = x1_model * img_w / input_w
        y1_orig = y1_model * img_h / input_h
        x2_orig = x2_model * img_w / input_w
        y2_orig = y2_model * img_h / input_h

        # Clip coordinates to image boundaries
        x1_orig = max(0, x1_orig)
        y1_orig = max(0, y1_orig)
        x2_orig = min(img_w - 1, x2_orig)
        y2_orig = min(img_h - 1, y2_orig)

        # Append for NMS (OpenCV NMSBoxes expects [x, y, w, h])
        nms_boxes.append([int(x1_orig), int(y1_orig), int(x2_orig - x1_orig), int(y2_orig - y1_orig)])
        nms_scores.append(float(confidence))
        nms_class_ids.append(class_id)
    
    # Apply Non-Maximum Suppression
    # cv2.dnn.NMSBoxes returns indices of kept boxes
    indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, conf_threshold, iou_threshold)

    if len(indices) > 0:
        for i in indices.flatten(): # flatten if it's a 2D array
            # Reconstruct the final detection dictionary with scaled (x1, y1, x2, y2)
            # and the original score and class_id
            box_xywh = nms_boxes[i]
            x1, y1, w, h = box_xywh
            x2 = x1 + w
            y2 = y1 + h

            final_results.append({
                "box": [x1, y1, x2, y2], # Storing as [x1, y1, x2, y2]
                "score": nms_scores[i],
                "class_id": nms_class_ids[i]
            })

    return final_results


def load_class_names(names_path: str):
    """Loads class names from a text file."""
    with open(names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names