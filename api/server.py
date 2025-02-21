from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import io
from PIL import Image
import base64
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

model = YOLO('./boulder_yolo/weights/best.pt')

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

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    in_memory_file = io.BytesIO(file.read())
    img = np.array(Image.open(in_memory_file))

    if img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img

    _, orig_buffer = cv2.imencode('.jpg', img_bgr)
    orig_base64 = base64.b64encode(orig_buffer).decode('utf-8')

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    filtered_output = np.zeros_like(img_bgr)
    for color, ranges in COLOR_RANGES.items():
        if color == "rot1":
            lower1 = np.array(COLOR_RANGES["rot1"][0], dtype=np.uint8)
            upper1 = np.array(COLOR_RANGES["rot1"][1], dtype=np.uint8)
            mask1 = cv2.inRange(hsv, lower1, upper1)
            lower2 = np.array(COLOR_RANGES["rot2"][0], dtype=np.uint8)
            upper2 = np.array(COLOR_RANGES["rot2"][1], dtype=np.uint8)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = mask1 | mask2
        elif color == "rot2":
            continue
        else:
            lower = np.array(ranges[0], dtype=np.uint8)
            upper = np.array(ranges[1], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        color_segment = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        filtered_output = cv2.add(filtered_output, color_segment)
    
    _, filt_buffer = cv2.imencode('.jpg', filtered_output)
    filt_base64 = base64.b64encode(filt_buffer).decode('utf-8')

    results = model.predict(source=img_bgr, conf=0.5)
    detections = []
    yolo_output = img_bgr.copy()
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0]
            detections.append({
                "class": cls_id,
                "confidence": conf,
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })
            cv2.rectangle(yolo_output, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
    _, yolo_buffer = cv2.imencode('.jpg', yolo_output)
    yolo_base64 = base64.b64encode(yolo_buffer).decode('utf-8')

    return jsonify({
        "original_image": orig_base64,
        "filtered_image": filt_base64,
        "detection_image": yolo_base64,
        "detections": detections
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
