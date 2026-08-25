import base64
import io
import os

import cv2
import numpy as np
from flask import Blueprint, json, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required
from PIL import Image

from extensions import db
from models.models import Analysis
from vision.classes import DETECTION_CLASSES
from vision.color_extraction import extract_color
from vision.route_grouping import group_routes

yolo_bp = Blueprint("yolo_bp", __name__)

# Modelle werden beim ersten Request geladen, nicht beim Import: der Server
# startet auch ohne Gewichte und meldet dann einen klaren Fehler statt eines
# Import-Absturzes.
_models = {}


def _get_yolo():
    if "yolo" not in _models:
        from ultralytics import YOLO

        weights = os.getenv(
            "WEIGHTS_PATH",
            os.path.join(os.path.dirname(__file__), "..", "weights", "best.pt"),
        )
        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"Keine Modellgewichte unter {weights}. Training siehe README, "
                "Pfad per WEIGHTS_PATH übersteuerbar."
            )
        _models["yolo"] = YOLO(weights)
    return _models["yolo"]


def _get_sam():
    if os.getenv("USE_SAM", "1") != "1":
        return None
    if "sam" not in _models:
        from ultralytics import SAM

        _models["sam"] = SAM(os.getenv("SAM_WEIGHTS", "sam2.1_b.pt"))
    return _models["sam"]


def _segment(img_rgb, detections):
    """SAM-Masken je Bounding Box als Polygon an die Detektionen hängen."""
    sam = _get_sam()
    if sam is None or not detections:
        return
    bboxes = [det["bbox"] for det in detections]
    results = sam.predict(source=img_rgb, bboxes=bboxes, verbose=False)
    masks = getattr(results[0], "masks", None)
    if masks is None:
        return
    # zip statt Laufindex: liefert SAM weniger Masken als Boxen, bleiben die
    # restlichen Detektionen einfach ohne Segmentierung statt zu crashen.
    for det, mask in zip(detections, masks):
        mask_np = np.asarray(mask.data.cpu().numpy()).squeeze()
        mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
            det["segmentation"] = contour.tolist()


@yolo_bp.route("/process", methods=["POST"])
@jwt_required(optional=False)
def process_image():
    user_id = get_jwt_identity()

    if "image" not in request.files:
        return jsonify({"msg": "Bilddatei fehlt (Feld 'image')"}), 400

    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = np.array(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    _, orig_buffer = cv2.imencode(".jpg", img_bgr)
    orig_base64 = base64.b64encode(orig_buffer).decode("utf-8")

    try:
        model = _get_yolo()
    except FileNotFoundError as e:
        return jsonify({"msg": str(e)}), 503

    conf = float(os.getenv("DETECT_CONF", "0.4"))
    results = model.predict(source=img_bgr, conf=conf, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            detections.append(
                {
                    "det_class": DETECTION_CLASSES[cls_id]
                    if cls_id < len(DETECTION_CLASSES)
                    else "hold",
                    "confidence": float(box.conf[0]),
                    "bbox": [x1, y1, x2, y2],
                }
            )

    _segment(img, detections)

    # Farbe aus den Pixeln bestimmen. class/class_name bleiben als Aliase auf
    # die Farbe erhalten, damit ältere App-Stände weiter funktionieren.
    for det in detections:
        color = extract_color(
            img_bgr, bbox=det["bbox"], polygon=det.get("segmentation")
        )
        det["color"] = color
        det["class"] = color["index"]
        det["class_name"] = color["name"]
        det["color_hex"] = color["hex"]

    height, width = img_bgr.shape[:2]
    routes = group_routes(detections, width, height)

    response = {
        "original_image": orig_base64,
        "detections": detections,
        "routes": routes,
        "image_width": width,
        "image_height": height,
    }

    if user_id is not None:
        analysis = Analysis(
            user_id=user_id,
            original_image=orig_base64,
            image_width=width,
            image_height=height,
            routes=json.dumps(routes),
            detections=detections,
        )
        db.session.add(analysis)
        db.session.commit()

    return jsonify(response)
