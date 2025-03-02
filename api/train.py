from ultralytics import YOLO

model = YOLO('yolo12x.pt')

model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    name='boulder_yolo_12x-augmented',
    hsv_h=0.015,    # leichte Farbtonänderung
    hsv_s=0.7,      # verstärkte Sättigung
    hsv_v=0.4,      # veränderte Helligkeit
    degrees=15,     # Rotation bis zu ±15°
    translate=0.1,  # Verschiebung um bis zu 10%
    scale=0.5,      # Skalierung (Werte hier experimentell wählen)
    shear=0.0,      # Scherung, hier erstmal deaktiviert
    mosaic=1.0,     # Mosaic-Augmentation (normalerweise standardmäßig aktiviert)
    mixup=0.0       # Mixup kann je nach Anwendungsfall nützlich sein, hier z. B. deaktiviert
)
