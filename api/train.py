"""YOLO26-Training für die Grifferkennung.

Vorher einmal das Dataset bauen: python scripts/prepare_dataset.py
Dann z. B.:            python train.py --model yolo26s.pt --epochs 120

Läuft sinnvoll nur auf einem Rechner mit GPU (oder Colab). Das Modell lernt
nur hold/volume — Farb-Augmentierung (hsv_*) ist deshalb ausdrücklich
erwünscht und macht die Erkennung robust gegen Gym-Beleuchtung; die Farbe
selbst bestimmt später vision/color_extraction.py aus den Pixeln.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Boulder AI Training (YOLO26)")
    parser.add_argument("--model", default="yolo26s.pt",
                        help="Basisgewichte: yolo26n/s/m/l/x.pt")
    parser.add_argument("--data", default="./datasets/merged/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1,
                        help="-1 = automatisch an den VRAM anpassen")
    parser.add_argument("--name", default="boulder-holds-yolo26")
    parser.add_argument("--device", default=None,
                        help="z. B. 0, cpu — leer lassen für Auto-Wahl")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        # Farbe ist keine Klasse mehr: kräftige Farb-Augmentierung erlaubt
        hsv_h=0.05,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )
    metrics = model.val()
    print(f"Fertig. mAP50-95: {metrics.box.map:.3f}  mAP50: {metrics.box.map50:.3f}")
    print("Beste Gewichte: runs/detect/<name>/weights/best.pt -> nach api/weights/best.pt kopieren")


if __name__ == "__main__":
    main()
