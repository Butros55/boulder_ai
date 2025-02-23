import cv2
from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase

import os

class YOLOModelBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(current_dir, "weights", "best.pt")

        model_path = kwargs.get("model_path", default_model_path)
        self.model = YOLO(model_path)


    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            image_url = task["data"].get("image_url")
            if not image_url:
                predictions.append({"result": []})
                continue

            img = cv2.imread(image_url)
            if img is None:
                predictions.append({"result": []})
                continue

            results = self.model.predict(img, conf=0.5)
            boxes = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0]

                    box_result = {
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {
                            "x": x1,  
                            "y": y1,
                            "width": x2 - x1,
                            "height": y2 - y1,
                            "rectanglelabels": [str(cls_id)]
                        },
                        "score": conf
                    }
                    boxes.append(box_result)

            predictions.append({"result": boxes})
        return predictions

if __name__ == "__main__":
    pass
