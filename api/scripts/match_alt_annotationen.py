"""Alte Label-Studio-Annotationen den Original-Fotos zuordnen (Inhalts-Match).

Die Annotationen von 02/2025 referenzieren WhatsApp-Dateinamen; die Bilder
dazu sind weg. Die Fotos existieren aber als Originale unter anderem Namen.
Da die Annotationen in Prozentkoordinaten gespeichert sind, reicht die
richtige Bild-Zuordnung — und die findet dieses Skript über den Inhalt:
Die Box-Konstellation einer Annotation wird gegen die YOLO-Detektionen
jedes Kandidatenbilds gematcht (mittlere Best-IoU im Prozentraum).
Jedes Wandfoto hat ein praktisch einzigartiges Griffmuster.

Aufruf aus api/:
    python scripts/match_alt_annotationen.py \
        --export "<scratch>/alt_colors_v2.json" --out datasets/alt_match_v2
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

API_DIR = Path(__file__).resolve().parent.parent


def boxes_from_task(task):
    out = []
    for a in task.get("annotations", []):
        if a.get("was_cancelled"):
            continue
        for res in a.get("result", []):
            v = res.get("value", {})
            if "rectanglelabels" in v and "x" in v:
                out.append((v["x"], v["y"], v["width"], v["height"],
                            v["rectanglelabels"][0]))
    return out


def iou_pct(a, b):
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def constellation_score(ann_boxes, det_boxes):
    """Mittlere Best-IoU der Annotationsboxen gegen die Detektionen."""
    if not ann_boxes or not det_boxes:
        return 0.0
    scores = []
    for ab in ann_boxes:
        scores.append(max(iou_pct(ab[:4], db) for db in det_boxes))
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description="Alt-Annotationen matchen")
    parser.add_argument("--export", required=True,
                        help="Label-Studio-JSON-Export des alten Projekts")
    parser.add_argument("--images",
                        default=str(API_DIR / "datasets" / "eigene_halle_jpg"))
    parser.add_argument("--out", required=True,
                        help="Zielverzeichnis für mapping.json + Kontroll-Overlays")
    parser.add_argument("--weights", default=str(API_DIR / "weights" / "best.pt"))
    parser.add_argument("--min-score", type=float, default=0.30)
    parser.add_argument("--min-margin", type=float, default=0.05)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import cv2
    from ultralytics import YOLO

    tasks = json.loads(Path(args.export).read_text(encoding="utf-8"))
    tasks = [t for t in tasks if boxes_from_task(t)]
    print(f"[alt] {len(tasks)} annotierte Tasks im Export")

    yolo = YOLO(args.weights)
    kandidaten = sorted(Path(args.images).glob("*.jpg"))
    det_cache = {}
    for p in kandidaten:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        r = yolo.predict(source=img, conf=0.30, verbose=False)[0]
        det_cache[p.name] = (
            [(float(b.xyxy[0][0]) / w * 100, float(b.xyxy[0][1]) / h * 100,
              float(b.xyxy[0][2] - b.xyxy[0][0]) / w * 100,
              float(b.xyxy[0][3] - b.xyxy[0][1]) / h * 100)
             for b in r.boxes],
            w / h,
        )
    print(f"[detektor] {len(det_cache)} Kandidatenbilder analysiert")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping, vergeben = {}, set()
    zeilen = []
    for t in tasks:
        ann = boxes_from_task(t)
        ref = t.get("data", {}).get("image_url") or t.get("data", {}).get("image", "")
        alt_name = os.path.basename(ref.replace("%5C", "/").replace("\\", "/"))
        # Seitenverhältnis der alten Datei aus den result-Metadaten
        ow = oh = None
        for a in t.get("annotations", []):
            for res in a.get("result", []):
                ow = res.get("original_width") or ow
                oh = res.get("original_height") or oh
        aspekt = (ow / oh) if ow and oh else None

        wertung = []
        for name, (dets, kand_aspekt) in det_cache.items():
            if aspekt and abs(kand_aspekt - aspekt) / aspekt > 0.03:
                continue
            wertung.append((constellation_score(ann, dets), name))
        wertung.sort(reverse=True)
        if not wertung:
            zeilen.append((alt_name, None, 0.0, "kein Aspekt-Kandidat"))
            continue
        best_score, best_name = wertung[0]
        second = wertung[1][0] if len(wertung) > 1 else 0.0
        if best_score < args.min_score:
            zeilen.append((alt_name, None, best_score, "Score zu niedrig"))
        elif best_score - second < args.min_margin:
            zeilen.append((alt_name, None, best_score,
                           f"nicht eindeutig (2.: {second:.2f} {wertung[1][1]})"))
        elif best_name in vergeben:
            zeilen.append((alt_name, None, best_score,
                           f"{best_name} schon vergeben"))
        else:
            mapping[alt_name] = {"bild": best_name,
                                 "score": round(best_score, 3),
                                 "boxen": len(ann)}
            vergeben.add(best_name)
            zeilen.append((alt_name, best_name, best_score, "ok"))

    for alt, neu, score, status in zeilen:
        mark = "OK " if neu else "-- "
        print(f"{mark}{alt} -> {neu or '?'} ({score:.2f}) {status}")
    ok = sum(1 for _, n, _, _ in zeilen if n)
    print(f"\n{ok}/{len(zeilen)} eindeutig zugeordnet")

    (out_dir / "mapping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")

    # Kontroll-Overlays für die drei besten und den schlechtesten Match
    geordnet = sorted(mapping.items(), key=lambda kv: -kv[1]["score"])
    for alt_name, info in geordnet[:2] + geordnet[-1:]:
        task = next(t for t in tasks if alt_name in
                    (t.get("data", {}).get("image_url", "") +
                     t.get("data", {}).get("image", "")).replace("%5C", "/"))
        img = cv2.imread(str(Path(args.images) / info["bild"]))
        h, w = img.shape[:2]
        for x, y, bw, bh, lab in boxes_from_task(task):
            cv2.rectangle(img, (int(x * w / 100), int(y * h / 100)),
                          (int((x + bw) * w / 100), int((y + bh) * h / 100)),
                          (0, 200, 255), 2)
        cv2.imwrite(str(out_dir / f"kontrolle_{info['bild']}.png"), img)
    print(f"Mapping + Kontroll-Overlays unter {out_dir}")


if __name__ == "__main__":
    main()
