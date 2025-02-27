from sklearn.cluster import DBSCAN
import numpy as np
from extensions import db
from flask import Blueprint, app, json, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import cv2
import io
from PIL import Image
import base64
from ultralytics import YOLO, SAM
from models.models import Analysis

sam_model = SAM('sam2_b.pt')
model = YOLO('./weights/best.pt')
yolo_bp = Blueprint('yolo_bp', __name__)

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

def group_detections_into_routes_by_color(
    detections, threshold, min_grip_count=3, ignored_class_ids=None
):
    if ignored_class_ids is None:
        ignored_class_ids = set()

    detections_by_class = {}
    for det in detections:
        cls_id = det["class"]
        if cls_id in ignored_class_ids:
            continue
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        det["_center"] = (center_x, center_y)
        detections_by_class.setdefault(cls_id, []).append(det)

    routes = []
    for cls_id, group in detections_by_class.items():
        n = len(group)
        visited = [False] * n

        for i in range(n):
            if not visited[i]:
                component = []
                stack = [i]
                visited[i] = True
                while stack:
                    cur = stack.pop()
                    component.append(group[cur])
                    cur_center = group[cur]["_center"]
                    for j in range(n):
                        if not visited[j]:
                            dx = cur_center[0] - group[j]["_center"][0]
                            dy = cur_center[1] - group[j]["_center"][1]
                            dist = (dx * dx + dy * dy) ** 0.5
                            if dist < threshold:
                                visited[j] = True
                                stack.append(j)
                if len(component) >= min_grip_count:
                    routes.append(component)
    return routes

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

    results = model.predict(source=img_bgr, conf=0.4)
    detections = []
    yolo_output = img_bgr.copy()

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0]
            detections.append({
                "class": cls_id,
                "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown",
                "confidence": conf,
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })

    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bboxes = [det["bbox"] for det in detections]
    results_sam = sam_model.predict(source=image_rgb, bboxes=bboxes)

    i=0
    for det in detections:
        mask = results_sam[0].masks[i]
        i+=1
        mask_np = np.array(mask.cpu().numpy().data).squeeze()
        mask_uint8 = (mask_np.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            contour = contour.reshape(-1, 2)
            polygon = contour.tolist()
            det["segmentation"] = polygon





    _, yolo_buffer = cv2.imencode('.jpg', yolo_output)
    yolo_base64 = base64.b64encode(yolo_buffer).decode('utf-8')

    height, width = img_bgr.shape[:2]
    routes = group_detections_into_routes_by_color(detections, threshold=min(width, height) * 0.3, min_grip_count=3)

    response = {
        "original_image": orig_base64,
        "detections": detections,
        "routes": routes,
        "image_width": width,
        "image_height": height
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
