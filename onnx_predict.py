import cv2
import numpy as np
import onnxruntime as ort
import os

# --- YOUR SPECIFIED PATHS ---
MODEL_PATH = "model1/train6/weights/best.onnx"
IMAGE_PATH = "datasets/images/test/494647.png"
OUTPUT_IMAGE_PATH = "output_image_with_detections.jpg" # A descriptive output name

def run_onnx_model_and_draw_rectangles(
    model_path, image_path, output_image_path="output_image.jpg"
):
    """
    Runs an ONNX object detection model, draws bounding boxes on the image,
    and displays/saves the result.

    Args:
        model_path (str): Path to your ONNX model file.
        image_path (str): Path to the input image file.
        output_image_path (str): Path to save the output image.
    """
    # 1. Load the ONNX Model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        # Assuming a single output. If your model has multiple outputs,
        # you might need to inspect session.get_outputs() to find the correct one.
        output_name = session.get_outputs()[0].name
        print(f"ONNX model loaded: {model_path}")
        print(f"Model input name: {input_name}")
        print(f"Model output name: {output_name}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        print("Please ensure the ONNX runtime is correctly installed and the model file is valid.")
        return

    # 2. Prepare Input Image
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}. Check if it's a valid image file.")
        return

    # Get original image dimensions (OpenCV loads as HxWxC)
    original_height, original_width = image.shape[:2]
    print(f"Original image dimensions: {original_width}x{original_height} (WxH)")

    # --- Preprocessing (Crucial: Adapt this based on your model's requirements) ---
    input_shape = session.get_inputs()[0].shape
    print(f"Model expected input shape: {input_shape}")

    model_input_height, model_input_width = 0, 0

    # Determine expected input dimensions from the model's input shape
    # Common formats: [N, C, H, W] or [N, H, W, C]
    if len(input_shape) == 4:
        if input_shape[1] == 3: # NCHW format (e.g., PyTorch models like YOLOv8)
            model_input_height, model_input_width = input_shape[2], input_shape[3]
            print("Model expects NCHW input format.")
        elif input_shape[3] == 3: # NHWC format (e.g., TensorFlow models)
            model_input_height, model_input_width = input_shape[1], input_shape[2]
            print("Model expects NHWC input format.")
        else:
            print(f"Warning: Could not infer model input dimensions from shape {input_shape}. "
                  "Assuming a common object detection input size like 640x640 for resizing.")
            model_input_height, model_input_width = 640, 640
    else:
        print(f"Warning: Model input shape {input_shape} not in expected 4D format. "
              "Assuming a common object detection input size like 640x640 for resizing.")
        model_input_height, model_input_width = 640, 640

    if model_input_height == 0 or model_input_width == 0:
         print("Error: Could not determine model input dimensions. Please check your model's input shape.")
         return

    print(f"Resizing image to model's expected input dimensions: {model_input_width}x{model_input_height}")
    resized_image = cv2.resize(image, (model_input_width, model_input_height))

    # Convert to float32
    input_tensor = resized_image.astype(np.float32)

    # Normalize: Scale pixel values to 0-1 range (common for many models like YOLOv8)
    input_tensor /= 255.0

    # Transpose/Reshape: Change channel order if the model expects NCHW (Channels First)
    if input_shape[1] == 3: # If model expects NCHW (e.g., from PyTorch)
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC to CHW

    # Add batch dimension (batch_size=1)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    print(f"Input tensor shape after preprocessing: {input_tensor.shape}")

    # 3. Run Inference
    try:
        outputs = session.run([output_name], {input_name: input_tensor})
        # The output is a list, so we take the first element
        model_output = outputs[0]
        print(f"Raw model output shape: {model_output.shape}")
        # print(f"Raw model output (first 5x5 section):\n{model_output[0, :5, :5]}") # For debugging
    except Exception as e:
        print(f"Error running inference: {e}")
        return

    # 4. Process Model Output (!!!! THIS IS WHERE YOU NEED TO CUSTOMIZE HEAVILY !!!!)
    # Based on your output: Raw model output shape: (1, 6, 8400)
    # This indicates 2 classes: 4 box coords + 2 class scores.
    # The common output for YOLOv8 is often (1, 84, N_detections), but yours is different.

    detected_boxes = []
    confidence_threshold = 0.25  # Adjust this value (0.0 to 1.0). Lower -> more detections, more false positives.
    iou_threshold = 0.45         # IoU threshold for Non-Maximum Suppression (NMS)

    # Transpose the output to make processing easier: (1, 6, 8400) -> (1, 8400, 6)
    # This brings the individual detections to the second dimension
    if model_output.shape[1] == 6 and len(model_output.shape) == 3: # Confirming (1, 6, 8400) structure
        model_output = np.transpose(model_output, (0, 2, 1))
        print(f"Transposed model output shape for processing: {model_output.shape}") # Should be (1, 8400, 6)
    else:
        print(f"Warning: Unexpected model output shape for transposition. "
              f"Expected (1, 6, 8400), got {model_output.shape}. "
              "Post-processing might be incorrect.")

    # Now, model_output[0] is (8400, 6)
    # The first 4 columns are box coordinates (x_center, y_center, width, height)
    # The last 2 columns are class scores (assuming 2 classes for this model)
    boxes = model_output[0, :, :4]        # Shape: (8400, 4) - contains [x_c, y_c, w, h]
    scores = model_output[0, :, 4:]       # Shape: (8400, Num_Classes) - contains class probabilities

    # Get the maximum score and corresponding class ID for each box
    class_ids = np.argmax(scores, axis=1) # Get the index of the highest score (e.g., 0 or 1)
    confidences = np.max(scores, axis=1)  # Get the value of the highest score

    # Prepare for NMS
    nms_boxes = []
    nms_scores = []
    nms_class_ids = []

    for i in range(model_output.shape[1]): # Iterate through the N_detections (e.g., 8400)
        confidence = confidences[i]
        class_id = class_ids[i]

        if confidence >= confidence_threshold:
            x_center, y_center, width, height = boxes[i]

            # Convert (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
            # These coordinates are normalized (0 to 1) relative to the model's input size (e.g., 640x640)
            x_min_model = (x_center - width / 2)
            y_min_model = (y_center - height / 2)
            x_max_model = (x_center + width / 2)
            y_max_model = (y_center + height / 2)

            # Scale back to original image dimensions (original_width x original_height)
            # original_width/height are from the input image.
            # model_input_width/height are the dimensions the model expects (e.g., 640x640).
            x_min = int(x_min_model * original_width / model_input_width)
            y_min = int(y_min_model * original_height / model_input_height)
            x_max = int(x_max_model * original_width / model_input_width)
            y_max = int(y_max_model * original_height / model_input_height)

            # Ensure coordinates are within image bounds to prevent drawing errors
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(original_width - 1, x_max)
            y_max = min(original_height - 1, y_max)

            nms_boxes.append([x_min, y_min, x_max, y_max])
            nms_scores.append(float(confidence))
            nms_class_ids.append(int(class_id))

    # Define the Non-Maximum Suppression (NMS) function
    def non_max_suppression(boxes, scores, iou_threshold):
        """
        Applies Non-Maximum Suppression to filter overlapping bounding boxes.
        boxes: List of [x_min, y_min, x_max, y_max]
        scores: List of confidence scores
        iou_threshold: Intersection Over Union threshold (0.0 to 1.0)
        Returns: List of indices of the boxes to keep.
        """
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        scores = np.array(scores)

        # Calculate areas of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Sort by scores in descending order
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Calculate intersection coordinates
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # Calculate width and height of intersection
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            # Calculate intersection area
            inter = w * h

            # Calculate IoU
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # Remove boxes with IoU greater than threshold
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1] # +1 because ovr is smaller than order

        return keep

    # Apply NMS
    indices = non_max_suppression(nms_boxes, nms_scores, iou_threshold)

    final_detections = []
    for i in indices:
        final_detections.append({
            'box': nms_boxes[i],
            'confidence': nms_scores[i],
            'class_id': nms_class_ids[i]
        })

    print(f"Total detected objects after NMS: {len(final_detections)}")

    # 5. Draw Rectangles
    output_image = image.copy()
    # IMPORTANT: REPLACE THESE WITH YOUR ACTUAL CLASS NAMES
    # Based on your model's output (2 classes), ensure these match your training.
    class_labels = {
        0: 'Your_Class_0_Name', # e.g., 'person', 'fruit', 'defect'
        1: 'Your_Class_1_Name', # e.g., 'car', 'vegetable', 'good'
    }

    for det in final_detections:
        x_min, y_min, x_max, y_max = det['box']
        confidence = det['confidence']
        class_id = det['class_id']

        color = (0, 255, 0)  # Green color for the rectangle (BGR format)
        thickness = 2
        cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), color, thickness)

        # Get class name, default to 'Class X' if not found in dictionary
        label = f"{class_labels.get(class_id, f'Class {class_id}')}: {confidence:.2f}"
        
        # Put text slightly above the box
        text_y = y_min - 10 if y_min - 10 > 10 else y_min + 10 # Adjust text position if too high
        cv2.putText(output_image, label, (x_min, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    # 6. Display/Save Image
    cv2.imshow("Detected Objects", output_image)
    cv2.imwrite(output_image_path, output_image)
    print(f"Output image with detections saved to: {output_image_path}")
    cv2.waitKey(0) # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows() # Close all OpenCV windows

if __name__ == "__main__":
    # Call the function with your specified paths
    run_onnx_model_and_draw_rectangles(MODEL_PATH, IMAGE_PATH, OUTPUT_IMAGE_PATH)