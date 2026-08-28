"""Label-Studio-Annotationen als YOLO-Trainingsdaten exportieren.

Eine Labelrunde = ein frisches Dataset: Das Skript zieht den JSON-Export
eines Projekts über die API, konvertiert Polygone (YOLO-Segmentformat)
und Rechtecke (YOLO-Boxformat) in normalisierte Labeldateien, kopiert die
zugehörigen Bilder aus dem lokalen Quellordner, macht einen deterministischen
train/val-Split und schreibt die data.yaml.

Klassenreihenfolge = Reihenfolge in der Labeling-Config des Projekts, damit
Klassenindizes über Labelrunden stabil bleiben.

Aufruf aus api/:
    python scripts/ls_export.py --project-id 8 --out datasets/griffarten
    python scripts/ls_export.py --project-id 8 --out datasets/griffarten --boxes-only
"""

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent


def login(base, email, password):
    import requests

    s = requests.Session()
    s.get(f"{base}/user/login/")
    s.post(f"{base}/user/login/",
           data={"email": email, "password": password,
                 "csrfmiddlewaretoken": s.cookies.get("csrftoken")},
           headers={"Referer": f"{base}/user/login/"})
    if not s.cookies.get("sessionid"):
        raise SystemExit("Label-Studio-Login fehlgeschlagen")
    return s


def main():
    parser = argparse.ArgumentParser(description="Label Studio -> YOLO")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--images",
                        default=str(API_DIR / "datasets" / "eigene_halle_jpg"),
                        help="Quellordner der Originalbilder (Dateinamen-Match)")
    parser.add_argument("--val-anteil", type=float, default=0.15)
    parser.add_argument("--boxes-only", action="store_true",
                        help="Polygone zu Bounding Boxes reduzieren")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(API_DIR.parent / ".env")
    load_dotenv(API_DIR / ".env")
    base = os.getenv("LS_URL", "http://127.0.0.1:8080")
    s = login(base, os.getenv("LS_EMAIL", ""), os.getenv("LS_PASSWORD", ""))

    projekt = s.get(f"{base}/api/projects/{args.project_id}").json()
    klassen = re.findall(r'<Label value="([^"]+)"', projekt["label_config"])
    klassen = list(dict.fromkeys(klassen))
    idx = {k: i for i, k in enumerate(klassen)}
    print(f"[klassen] {klassen}")

    tasks = s.get(f"{base}/api/projects/{args.project_id}/export",
                  params={"exportType": "JSON",
                          "download_all_tasks": "false"}).json()
    print(f"[export] {len(tasks)} Tasks mit Annotationen")

    out = Path(args.out)
    for split in ("train", "valid"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    quelle = Path(args.images)
    n_img = n_lbl = 0
    for t in tasks:
        ref = t.get("data", {}).get("image", "") or t.get("data", {}).get("image_url", "")
        name = os.path.basename(ref)
        # Upload-Dateien tragen ein Hash-Präfix "abcd1234-name.jpg"
        if "/upload/" in ref and "-" in name:
            name = name.split("-", 1)[1]
        src = quelle / name
        if not src.exists():
            print(f"[warn] Bild fehlt lokal: {name}")
            continue

        zeilen = []
        for a in t.get("annotations", []):
            if a.get("was_cancelled"):
                continue
            for res in a.get("result", []):
                v = res.get("value", {})
                if "polygonlabels" in v:
                    cls = idx.get(v["polygonlabels"][0])
                    pts = [(p[0] / 100.0, p[1] / 100.0) for p in v["points"]]
                    if cls is None or len(pts) < 3:
                        continue
                    if args.boxes_only:
                        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                        cx = (min(xs) + max(xs)) / 2
                        cy = (min(ys) + max(ys)) / 2
                        w = max(xs) - min(xs)
                        h = max(ys) - min(ys)
                        zeilen.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    else:
                        flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
                        zeilen.append(f"{cls} {flat}")
                elif "rectanglelabels" in v:
                    cls = idx.get(v["rectanglelabels"][0])
                    if cls is None:
                        continue
                    cx = (v["x"] + v["width"] / 2) / 100.0
                    cy = (v["y"] + v["height"] / 2) / 100.0
                    zeilen.append(f"{cls} {cx:.6f} {cy:.6f} "
                                  f"{v['width'] / 100.0:.6f} {v['height'] / 100.0:.6f}")
        if not zeilen:
            continue

        # deterministischer Split über den Dateinamen
        anteil = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100 / 100.0
        split = "valid" if anteil < args.val_anteil else "train"
        shutil.copy(src, out / split / "images" / name)
        (out / split / "labels" / f"{Path(name).stem}.txt").write_text(
            "\n".join(zeilen) + "\n", encoding="utf-8")
        n_img += 1
        n_lbl += len(zeilen)

    import yaml
    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": str(out.resolve()), "train": "train/images",
                        "val": "valid/images", "nc": len(klassen),
                        "names": klassen}, f, sort_keys=False)
    print(f"[fertig] {n_img} Bilder, {n_lbl} Labels -> {out} "
          f"({'Boxen' if args.boxes_only else 'Segmente+Boxen'})")


if __name__ == "__main__":
    main()
