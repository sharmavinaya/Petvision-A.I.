from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

# ========================
# Load YOLOv8 Model
# ========================
# First run will auto-download weights
model = YOLO("yolov8n.pt")   # fast & lightweight

# Only detect these classes
TARGET_CLASSES = ["cat", "dog", "person"]


# ========================
# Prediction Route
# ========================
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]
    img_bytes = file.read()

    # Convert bytes → OpenCV image
    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Run YOLO
    results = model(img)

    detections = []

    for r in results:
        for box in r.boxes:

            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[cls_id]

            # 🎯 Filter only your classes
            if class_name not in TARGET_CLASSES:
                continue

            # 🔥 Confidence threshold (60%)
            if confidence < 0.6:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class": class_name,
                "confidence": confidence,
                "box": [x1, y1, x2, y2]
            })

    return jsonify(detections)


# ========================
# Run Server
# ========================
if __name__ == "__main__":
    app.run(debug=True)