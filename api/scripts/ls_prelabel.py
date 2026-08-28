"""Bilder mit YOLO+SAM vorlabeln und als Predictions in Label Studio laden.

Der schnellste Labeling-Workflow (recherchiert 25.08.2026, siehe Vault
[[Beta-Regelwerk Bouldern]]-Session): Predictions werden VORAB erzeugt und
per API importiert — kein ML-Backend nötig, keine Latenz beim Taskwechsel,
jeder Task öffnet fertig vorbefüllt. Die YOLO-Confidence wird als score
mitgegeben, damit im Data Manager nach unsicheren Bildern sortiert werden
kann. Formen: SAM2.1-tiny macht aus jeder Box ein Polygon (punktweise
korrigierbar, exportierbar als YOLO-Segmentierung); ohne --shapes gibt es
Rechtecke.

Zugang: LS_URL/LS_EMAIL/LS_PASSWORD aus api/.env (lokale Instanz).

Aufruf aus api/ (Beispiele):
    python scripts/ls_prelabel.py --project-id 7
    python scripts/ls_prelabel.py --create-project "Griffarten eigene Halle" \
        --images datasets/eigene_halle_jpg --model-version yolo26-run-01
"""

import argparse
import json
import os
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent

LABEL_CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <PolygonLabels name="label" toName="image" strokeWidth="2" pointSize="small">
    <Label value="hold" background="#9E9E9E" hotkey="1"/>
    <Label value="volume" background="#F57C00" hotkey="2"/>
    <Label value="jug" background="#4CAF50" hotkey="3"/>
    <Label value="crimp" background="#D32F2F" hotkey="4"/>
    <Label value="sloper" background="#1E63D0" hotkey="5"/>
    <Label value="pocket" background="#7B2FBE" hotkey="6"/>
    <Label value="pinch" background="#EC407A" hotkey="7"/>
    <Label value="foothold" background="#FBC02D" hotkey="8"/>
    <Label value="start_marker" background="#00BCD4" hotkey="9"/>
  </PolygonLabels>
  <RectangleLabels name="bbox" toName="image">
    <Label value="hold" background="#9E9E9E"/>
    <Label value="volume" background="#F57C00"/>
    <Label value="jug" background="#4CAF50"/>
    <Label value="crimp" background="#D32F2F"/>
    <Label value="sloper" background="#1E63D0"/>
    <Label value="pocket" background="#7B2FBE"/>
    <Label value="pinch" background="#EC407A"/>
    <Label value="foothold" background="#FBC02D"/>
    <Label value="start_marker" background="#00BCD4"/>
  </RectangleLabels>
</View>"""


def login(base, email, password):
    import requests

    s = requests.Session()
    s.get(f"{base}/user/login/")
    s.post(
        f"{base}/user/login/",
        data={"email": email, "password": password,
              "csrfmiddlewaretoken": s.cookies.get("csrftoken")},
        headers={"Referer": f"{base}/user/login/"},
    )
    if not s.cookies.get("sessionid"):
        raise SystemExit("Label-Studio-Login fehlgeschlagen (LS_EMAIL/LS_PASSWORD prüfen)")
    return s


def hdr(s, base):
    return {"X-CSRFToken": s.cookies.get("csrftoken"), "Referer": base}


def mask_to_polygon_pct(mask, w, h, max_points=24):
    import cv2
    import numpy as np

    m = (np.asarray(mask).squeeze() > 0.5).astype("uint8") * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 9:
        return None
    eps = 0.008 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    while len(approx) > max_points:
        eps *= 1.4
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    return [[float(x) / w * 100.0, float(y) / h * 100.0] for x, y in approx]


def main():
    parser = argparse.ArgumentParser(description="Label-Studio-Vorlabeler")
    parser.add_argument("--images", default=str(API_DIR / "datasets" / "eigene_halle_jpg"))
    parser.add_argument("--project-id", type=int)
    parser.add_argument("--create-project", metavar="TITEL")
    parser.add_argument("--weights", default=str(API_DIR / "weights" / "best.pt"))
    parser.add_argument("--model-version", default="yolo26-run-01")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--no-shapes", action="store_true",
                        help="nur Rechtecke, kein SAM")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    from dotenv import load_dotenv
    load_dotenv(API_DIR.parent / ".env")
    load_dotenv(API_DIR / ".env")
    base = os.getenv("LS_URL", "http://127.0.0.1:8080")
    email = os.getenv("LS_EMAIL", "geret@local.dev")
    password = os.getenv("LS_PASSWORD", "")
    if not password:
        raise SystemExit("LS_PASSWORD fehlt in api/.env")

    s = login(base, email, password)

    if args.create_project:
        r = s.post(f"{base}/api/projects",
                   json={"title": args.create_project,
                         "label_config": LABEL_CONFIG,
                         "show_collab_predictions": True},
                   headers=hdr(s, base))
        r.raise_for_status()
        project_id = r.json()["id"]
        print(f"[projekt] neu angelegt: id={project_id} {args.create_project!r}")
    elif args.project_id:
        project_id = args.project_id
    else:
        raise SystemExit("--project-id oder --create-project angeben")

    # vorhandene Tasks nach Original-Dateinamen indexieren
    def task_index():
        idx, page = {}, 1
        while True:
            r = s.get(f"{base}/api/tasks",
                      params={"project": project_id, "page": page,
                              "page_size": 100})
            if r.status_code != 200:
                break
            chunk = r.json().get("tasks", [])
            if not chunk:
                break
            for t in chunk:
                ref = t.get("data", {}).get("image", "")
                name = os.path.basename(ref).split("-", 1)[-1] \
                    if "/upload/" in ref else os.path.basename(ref)
                idx[name] = t["id"]
            page += 1
        return idx

    idx = task_index()
    bilder = sorted(Path(args.images).glob("*.jpg"))
    neue = [p for p in bilder if p.name not in idx]
    for p in neue:
        with open(p, "rb") as fh:
            s.post(f"{base}/api/projects/{project_id}/import",
                   files={"file": (p.name, fh, "image/jpeg")},
                   headers=hdr(s, base))
    if neue:
        print(f"[import] {len(neue)} Bilder hochgeladen")
        idx = task_index()

    from PIL import Image
    from ultralytics import YOLO
    yolo = YOLO(args.weights)
    sam = None
    if not args.no_shapes:
        from ultralytics import SAM
        sam = SAM(os.getenv("SAM_WEIGHTS", "sam2.1_t.pt"))

    import numpy as np
    n_pred = 0
    for p in bilder:
        task_id = idx.get(p.name)
        if task_id is None:
            print(f"[warn] kein Task für {p.name}")
            continue
        img = Image.open(p)
        w, h = img.size
        arr = np.array(img.convert("RGB"))
        r0 = yolo.predict(source=arr[:, :, ::-1], conf=args.conf, verbose=False)[0]
        boxes = [([float(v) for v in b.xyxy[0]], int(b.cls[0]), float(b.conf[0]))
                 for b in r0.boxes]
        masks = None
        if sam is not None and boxes:
            res = sam.predict(source=arr, bboxes=[b[0] for b in boxes],
                              verbose=False)
            masks = getattr(res[0], "masks", None)

        results = []
        for i, (bb, cls, conf) in enumerate(boxes):
            label = "volume" if cls == 1 else "hold"
            poly = None
            if masks is not None and i < len(masks):
                poly = mask_to_polygon_pct(masks[i].data.cpu().numpy(), w, h)
            if poly:
                results.append({
                    "id": f"p{i}", "from_name": "label", "to_name": "image",
                    "type": "polygonlabels",
                    "original_width": w, "original_height": h,
                    "score": conf,
                    "value": {"points": poly, "polygonlabels": [label]},
                })
            else:
                x0, y0, x1, y1 = bb
                results.append({
                    "id": f"b{i}", "from_name": "bbox", "to_name": "image",
                    "type": "rectanglelabels",
                    "original_width": w, "original_height": h,
                    "score": conf,
                    "value": {"x": x0 / w * 100, "y": y0 / h * 100,
                              "width": (x1 - x0) / w * 100,
                              "height": (y1 - y0) / h * 100,
                              "rectanglelabels": [label]},
                })

        score = float(np.mean([b[2] for b in boxes])) if boxes else 0.0
        r = s.post(f"{base}/api/predictions",
                   json={"task": task_id, "model_version": args.model_version,
                         "score": round(score, 4), "result": results},
                   headers=hdr(s, base))
        if r.status_code in (200, 201):
            n_pred += 1
            print(f"[ok] {p.name}: {len(results)} Vorlabels (score {score:.2f})")
        else:
            print(f"[fehler] {p.name}: {r.status_code} {r.text[:120]}")

    print(f"\n{n_pred} Tasks vorgelabelt (model_version={args.model_version}).")
    print("In Label Studio: Projekt-Settings -> Annotation -> Prelabeling auf "
          "diese model_version stellen, dann öffnet jeder Task vorbefüllt.")


if __name__ == "__main__":
    main()
