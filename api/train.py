from ultralytics import YOLO

model = YOLO('yolo12x.pt')

model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    name='boulder_yolo_12x'
)
