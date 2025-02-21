from ultralytics import YOLO

# Laden eines vortrainierten YOLOv8-Modells (z. B. das Nano-Modell)
model = YOLO('yolov8n.pt')

# Starte das Training
model.train(
    data='data.yaml',  # Pfad zur data.yaml
    epochs=50,         # Anzahl der Epochen – anpassen je nach Datensatzgröße
    imgsz=640,         # Bildgröße
    batch=8,           # Batch-Größe
    name='boulder_yolo'
)
