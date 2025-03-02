import albumentations as A
import cv2
import numpy as np
import os
import glob
from albumentations.pytorch import ToTensorV2

# Definiere die Augmentierungs-Pipeline
albumentations_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.GaussianBlur(p=0.2),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
    A.CLAHE(p=0.1),
    A.Resize(640, 640),
    ToTensorV2()
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

# Erstelle (falls nicht vorhanden) die Zielordner
image_folder = "C:\\dev\\boulder_yolo\\datasets\\images\\train"
label_folder = "C:\\dev\\boulder_yolo\\datasets\\labels\\train"
augmented_image_folder = "C:\\dev\\boulder_yolo\\datasets\\images\\train_augmented"
augmented_label_folder = "C:\\dev\\boulder_yolo\\datasets\\labels\\train_augmented"

os.makedirs(augmented_image_folder, exist_ok=True)
os.makedirs(augmented_label_folder, exist_ok=True)

# Hole alle Bilddateien (Passe ggf. das Pattern an, wenn du auch PNGs hast)
image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))

num_augmentations = 5  # Anzahl der Augmentierungen pro Originalbild

for img_path in image_paths:
    # Lade und konvertiere das Bild
    image = cv2.imread(img_path)
    if image is None:
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Lade das zugehörige Label
    label_path = os.path.join(label_folder, os.path.basename(img_path).replace(".jpg", ".txt"))
    if not os.path.exists(label_path):
        continue
    
    with open(label_path, "r") as f:
        labels = f.readlines()
    
    # Labels konvertieren (angenommen YOLO Format: class x_center y_center width height)
    bboxes = []
    class_labels = []
    for label in labels:
        try:
            cls, x_center, y_center, width, height = map(float, label.strip().split())
        except ValueError:
            print(f"Label-Formatfehler in {label_path}: {label}")
            continue
        bboxes.append([x_center, y_center, width, height])
        class_labels.append(int(cls))
    
    # Erstelle mehrere augmentierte Versionen
    for i in range(num_augmentations):
        augmented = albumentations_transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
        # Konvertiere den Tensor zurück in ein NumPy-Array (RGB -> BGR für cv2.imwrite)
        aug_img = augmented["image"].permute(1, 2, 0).cpu().numpy()[..., ::-1]
        augmented_img_path = os.path.join(augmented_image_folder, f"{os.path.splitext(os.path.basename(img_path))[0]}_aug{i}.jpg")
        cv2.imwrite(augmented_img_path, aug_img)
        
        # Speichere die augmentierten Labels
        augmented_label_path = os.path.join(augmented_label_folder, f"{os.path.splitext(os.path.basename(label_path))[0]}_aug{i}.txt")
        with open(augmented_label_path, "w") as f:
            for bbox, cls in zip(augmented["bboxes"], augmented["class_labels"]):
                f.write(f"{cls} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")


print("✅ Augmentierte Bilder und Labels gespeichert!")
