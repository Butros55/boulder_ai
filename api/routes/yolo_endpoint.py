from extensions import db
from flask import Blueprint, app, request, jsonify
from flask_jwt_extended import (
    jwt_required, get_jwt_identity
)
import cv2
import numpy as np
import io
from PIL import Image
import base64
from ultralytics import YOLO
from models.models import Analysis


model = YOLO('./weights/best.pt')
yolo_bp = Blueprint('yolo_bp', __name__)

COLOR_RANGES = {
    "gelb":    [(20, 100, 100), (30, 255, 255)],
    "tuerkis": [(80, 100, 100), (95, 255, 255)],
    "lila":    [(140, 100, 100), (160, 255, 255)],
    "rot1":    [(0, 120, 70), (10, 255, 255)],
    "rot2":    [(170, 120, 70), (180, 255, 255)],
    "blau":    [(100, 100, 70), (130, 255, 255)],
    "orange":  [(10, 100, 100), (20, 255, 255)],
    "weiss":   [(0, 0, 220), (180, 40, 255)]
}

CLASS_NAMES = [
    "black",
    "blue",
    "grey",
    "orange",
    "purple",
    "red",
    "turquoise",
    "white",
    "wood",
    "yellow",
]

CLASS_COLORS = [
    (0, 0, 0),       # black
    (255, 0, 0),     # blue
    (128, 128, 128), # grey
    (0, 165, 255),   # orange
    (128, 0, 128),   # purple
    (0, 0, 255),     # red
    (255, 255, 0),   # turquoise
    (255, 255, 255), # white
    (19, 69, 139),   # wood
    (0, 255, 255),   # yellow
]



@yolo_bp.route("/process", methods=["POST"])
@jwt_required(optional=False)
def process_image():
    user_id = get_jwt_identity()

    file = request.files['image']
    in_memory_file = io.BytesIO(file.read())
    img = Image.open(in_memory_file).convert('RGB')
    img = np.array(img)

    if img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img

    _, orig_buffer = cv2.imencode('.jpg', img_bgr)
    orig_base64 = base64.b64encode(orig_buffer).decode('utf-8')

    results = model.predict(source=img_bgr, conf=0.6)
    detections = []
    yolo_output = img_bgr.copy()

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0]

            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"

            cv2.rectangle(yolo_output, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label_text = f"{class_name}: {conf:.2f}"
            cv2.putText(yolo_output, label_text, (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detections.append({
                "class": cls_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })

    _, yolo_buffer = cv2.imencode('.jpg', yolo_output)
    yolo_base64 = base64.b64encode(yolo_buffer).decode('utf-8')

    response = {
        "original_image": orig_base64,
        "detection_image": yolo_base64,
        "detections": detections
    }

    if user_id is not None:
        analysis = Analysis(
            user_id=user_id,
            original_image=orig_base64,
            detection_image=yolo_base64
        )
        db.session.add(analysis)
        db.session.commit()

    return jsonify(response)