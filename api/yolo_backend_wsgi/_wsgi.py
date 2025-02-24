import sys
import os
from yolo_backend import YOLOModelBackend

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


app = YOLOModelBackend()
print("YOLOModelBackend loaded successfully. App is ready.")
