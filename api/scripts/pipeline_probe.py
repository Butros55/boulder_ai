"""Gesamte Vision-Pipeline auf eigenen Fotos testen und visualisieren.

Läuft produktionsnah: YOLO-Detektion -> SAM-Masken (bbox-Prompts) ->
Farbextraktion -> Routengruppierung. Erzeugt je Bild zwei Overlays:
  *--farben.png   Formen in der extrahierten Grifffarbe + Farbname
  *--routen.png   Routenmitglieder verbunden, je Route eine Nummer

Aufruf aus api/:  python scripts/pipeline_probe.py [--count 5] [--device cpu]
Ergebnis:         datasets/pipeline_probe/
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from sam3_probe import DEFAULT_IMAGES, list_images, load_image, pick_sample

API_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = API_DIR / "datasets" / "pipeline_probe"


def hex_to_bgr(hex_str):
    h = hex_str.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def main():
    parser = argparse.ArgumentParser(description="Pipeline-Probe")
    parser.add_argument("--images", default=str(DEFAULT_IMAGES))
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--device", default="cpu",
                        help="cpu solange die GPU trainiert")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--sam-weights", default="sam2.1_t.pt")
    args = parser.parse_args()

    import os
    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    from pillow_heif import register_heif_opener
    register_heif_opener()
    import sys
    sys.path.insert(0, str(API_DIR))
    from ultralytics import SAM, YOLO
    from vision.color_extraction import extract_color
    from vision.route_grouping import group_routes
    from vision.video_masks import candidate_from_mask, volume_mask_candidates
    from vision.video_tracking import crop_sharpness

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(str(API_DIR / "weights" / "best.pt"))
    sam = SAM(args.sam_weights)

    for path in pick_sample(list_images(args.images), args.count):
        t0 = time.time()
        img = load_image(path)
        w, h = img.size
        rgb = np.array(img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        r = yolo.predict(source=bgr, conf=args.conf, verbose=False)[0]
        detections = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "det_class": "volume" if cls_id == 1 else "hold",
                "confidence": float(box.conf[0]),
                "bbox": [float(v) for v in box.xyxy[0]],
            })

        if detections:
            res = sam.predict(source=rgb,
                              bboxes=[d["bbox"] for d in detections],
                              verbose=False)
            masks_obj = getattr(res[0], "masks", None)
            masks = list(masks_obj) if masks_obj is not None else []
            for index, det in enumerate(detections):
                class_id = 1 if det["det_class"] == "volume" else 0
                sharpness = crop_sharpness(bgr, det["bbox"])
                candidates = []
                if index < len(masks):
                    mask = np.asarray(masks[index].data.cpu().numpy()).squeeze()
                    candidates.append(candidate_from_mask(
                        bgr, det["bbox"], mask > 0.5,
                        class_id=class_id, source="sam",
                        confidence=det["confidence"], sharpness=sharpness,
                    ))
                if class_id == 1:
                    candidates.extend(volume_mask_candidates(
                        bgr, det["bbox"], confidence=det["confidence"],
                        sharpness=sharpness,
                    ))
                accepted = [candidate for candidate in candidates
                            if candidate.accepted]
                if accepted:
                    best = max(accepted, key=lambda candidate: candidate.score)
                    det["segmentation"] = best.polygon.astype(int).tolist()
                    det["segmentation_source"] = best.source

        for det in detections:
            color = extract_color(bgr, bbox=det["bbox"],
                                  polygon=det.get("segmentation"))
            det["color"] = color
            det["class"] = color["index"]
            det["class_name"] = color["name"]
            det["color_hex"] = color["hex"]

        routes = group_routes(detections, w, h)

        # ---- Overlay 1: Formen in Grifffarbe
        canvas = bgr.copy()
        overlay = bgr.copy()
        for det in detections:
            col = hex_to_bgr(det["color_hex"])
            if det.get("segmentation"):
                pts = np.array(det["segmentation"], np.int32)
                cv2.fillPoly(overlay, [pts], col)
                cv2.polylines(canvas, [pts], True, col, 2)
            else:
                x0, y0, x1, y1 = [int(v) for v in det["bbox"]]
                cv2.rectangle(canvas, (x0, y0), (x1, y1), col, 2)
            x0, y0 = int(det["bbox"][0]), int(det["bbox"][1])
            tag = det["class_name"] + ("/VOL" if det["det_class"] == "volume" else "")
            cv2.putText(canvas, tag, (x0, max(y0 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
        cv2.imwrite(str(OUT_DIR / f"{path.stem}--farben.png"), canvas)

        # ---- Overlay 2: Routen
        canvas = bgr.copy()
        for ri, route in enumerate(routes):
            col = hex_to_bgr(route[0]["color_hex"])
            centers = [(int((d["bbox"][0] + d["bbox"][2]) / 2),
                        int((d["bbox"][1] + d["bbox"][3]) / 2)) for d in route]
            order = np.argsort([c[1] for c in centers])[::-1]  # von unten nach oben
            pts = [centers[i] for i in order]
            for a, b in zip(pts, pts[1:]):
                cv2.line(canvas, a, b, col, 3, cv2.LINE_AA)
            for d in route:
                x0, y0, x1, y1 = [int(v) for v in d["bbox"]]
                cv2.rectangle(canvas, (x0, y0), (x1, y1), col, 3)
            cx, cy = pts[0]
            cv2.circle(canvas, (cx, cy), 16, col, -1)
            cv2.putText(canvas, str(ri + 1), (cx - 7, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)
        cv2.imwrite(str(OUT_DIR / f"{path.stem}--routen.png"), canvas)

        # ---- Overlay 3: Beta (Zugvorschlag) der größten Route
        from vision.beta_planner import plan_beta
        if routes:
            route = routes[0]
            steps = plan_beta(route)
            canvas = bgr.copy()
            col = hex_to_bgr(route[0]["color_hex"])
            centers = [(int((d["bbox"][0] + d["bbox"][2]) / 2),
                        int((d["bbox"][1] + d["bbox"][3]) / 2)) for d in route]
            for d in route:
                x0, y0, x1, y1 = [int(v) for v in d["bbox"]]
                cv2.rectangle(canvas, (x0, y0), (x1, y1), col, 2)
            prev_pt = None
            for s in steps:
                cx, cy = centers[s["hold"]]
                if prev_pt is not None and prev_pt != (cx, cy):
                    cv2.arrowedLine(canvas, prev_pt, (cx, cy), col, 3,
                                    cv2.LINE_AA, tipLength=0.12)
                prev_pt = (cx, cy)
                hand = {"left": "L", "right": "R", "both": "L+R"}[s["hand"]]
                tag = f'{s["step"]}{hand}'
                if s["action"] == "start":
                    tag += " S"
                if s["action"] == "top":
                    tag = f'{s["step"]} TOP'
                cv2.rectangle(canvas, (cx + 10, cy - 26), (cx + 30 + 16 * len(tag), cy - 2),
                              (250, 250, 250), -1)
                cv2.putText(canvas, tag, (cx + 14, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2,
                            cv2.LINE_AA)
            cv2.imwrite(str(OUT_DIR / f"{path.stem}--beta.png"), canvas)

        farben = {}
        for d in detections:
            farben[d["class_name"]] = farben.get(d["class_name"], 0) + 1
        print(f"[ok] {path.name}: {len(detections)} Detektionen, "
              f"{len(routes)} Routen, Farben {farben}, {time.time()-t0:.1f}s")

    print("\nErgebnis unter", OUT_DIR)


if __name__ == "__main__":
    main()
