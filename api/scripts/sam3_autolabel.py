"""SAM-3-Detektionen als YOLO-Labels exportieren (Auto-Labeling-Pfad).

Erzeugt aus den eigenen Hallenfotos ein kleines YOLO-Dataset mit den zwei
Zielklassen hold (0) und volume (1). Prompts: "climbing hold" findet Griffe
UND Volumes, "large angular volume" nur Volumes — Überschneidungen werden
zugunsten der Volume-Klasse aufgelöst (IoU > 0.6 = dasselbe Objekt).

Zweck: Smoke-Test für train.py ohne Roboflow-Key und Praxistest des
Auto-Labeler-Ansatzes. Kein Ersatz für prepare_dataset.py.

Aufruf aus api/:  python scripts/sam3_autolabel.py --count 40
Ergebnis:         datasets/sam3_smoke/{train,valid}/... + data.yaml
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from sam3_probe import (
    DEFAULT_IMAGES,
    build_sam31_image_detector,
    list_images,
    load_image,
    pick_sample,
)

API_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = API_DIR / "datasets" / "sam3_smoke"

HOLD_PROMPT = "climbing hold"
VOLUME_PROMPT = "large angular volume"


def iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def main():
    parser = argparse.ArgumentParser(description="SAM-3 Auto-Labeling")
    parser.add_argument("--images", default=str(DEFAULT_IMAGES))
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--valid-every", type=int, default=5,
                        help="jedes n-te Bild wird valid statt train")
    args = parser.parse_args()

    from pillow_heif import register_heif_opener
    register_heif_opener()
    from sam3.model.sam3_image_processor import Sam3Processor

    for split in ("train", "valid"):
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    paths = pick_sample(list_images(args.images), args.count)
    model = build_sam31_image_detector(args.device)
    processor = Sam3Processor(model, device=args.device, confidence_threshold=0.5)

    n_hold = n_vol = 0
    for i, path in enumerate(paths):
        t0 = time.time()
        img = load_image(path)
        w, h = img.size
        state = processor.set_image(img)

        state = processor.set_text_prompt(prompt=HOLD_PROMPT, state=state)
        hold_boxes = state["boxes"].cpu().numpy()
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(prompt=VOLUME_PROMPT, state=state)
        vol_boxes = state["boxes"].cpu().numpy()

        lines = []
        for box in vol_boxes:
            cx = (box[0] + box[2]) / 2 / w; cy = (box[1] + box[3]) / 2 / h
            bw = (box[2] - box[0]) / w; bh = (box[3] - box[1]) / h
            lines.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            n_vol += 1
        for box in hold_boxes:
            if any(iou(box, vb) > 0.6 for vb in vol_boxes):
                continue  # dasselbe Objekt, Volume-Klasse gewinnt
            cx = (box[0] + box[2]) / 2 / w; cy = (box[1] + box[3]) / 2 / h
            bw = (box[2] - box[0]) / w; bh = (box[3] - box[1]) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            n_hold += 1

        split = "valid" if i % args.valid_every == args.valid_every - 1 else "train"
        img.save(OUT_DIR / split / "images" / f"{path.stem}.jpg", quality=92)
        (OUT_DIR / split / "labels" / f"{path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        print(f"[{split}] {path.name}: {len(hold_boxes)} hold, "
              f"{len(vol_boxes)} volume in {time.time() - t0:.1f}s")

    with open(OUT_DIR / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": str(OUT_DIR),
                "train": "train/images",
                "val": "valid/images",
                "nc": 2,
                "names": ["hold", "volume"],
            },
            f,
            sort_keys=False,
        )
    print(f"\n{n_hold} hold- und {n_vol} volume-Labels unter {OUT_DIR}")


if __name__ == "__main__":
    main()
