import requests
import cv2

# --- Config ---
API_URL = "http://localhost:8000/predict"
IMAGE_PATH = "datasets/images/test/494647.png"
LABELS_PATH = "labels.txt"

# --- Load image and labels ---
image = cv2.imread(IMAGE_PATH)
with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# --- Send image to FastAPI ---
with open(IMAGE_PATH, "rb") as f:
    files = {"file": (IMAGE_PATH, f, "image/jpeg")}
    response = requests.post(API_URL, files=files)

if response.status_code != 200:
    print("❌ API Error:", response.status_code, response.text)
    exit()

# --- Parse detections ---
detections = response.json()["detections"]

for det in detections:
    x1, y1, x2, y2 = det["box"]
    score = det["score"]
    class_id = det["class_id"]
    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label
    label = f"{class_name} {score:.2f}"
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# --- Show result ---
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
