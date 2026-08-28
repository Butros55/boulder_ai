# Boulder AI

[![Flutter](https://img.shields.io/badge/Flutter-Framework-02569B?logo=flutter&logoColor=white)](https://flutter.dev/)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![YOLO26](https://img.shields.io/badge/YOLO26-Ultralytics-111111?logo=python&logoColor=white)](https://docs.ultralytics.com/models/yolo26/)
[![SAM](https://img.shields.io/badge/SAM_2.1-Segment%20Anything-6E40C9)](https://docs.ultralytics.com/models/sam-2/)

**Boulder AI** analysiert Fotos und Videos von Boulder-/Kletterwänden: Griffe erkennen, segmentieren, nach Farbe zu Routen gruppieren. Flutter-App (`app/`) + Flask-Backend (`api/`), Ergebnisse pro Nutzer (JWT) in SQLite.

## Pipeline (Stand 2026)

1. **Detektion — YOLO26**, zwei Klassen: `hold` und `volume`. Die Farbe ist **keine** Modellklasse mehr.
2. **Segmentierung — SAM 2.1 + Qualitäts-Gates** (optional, `USE_SAM=0` schaltet ab): mehrere Maskenkandidaten je Bounding Box; Wandanteil, Flaeche, Randkontakt und zeitliche Konsistenz entscheiden, ob eine Kontur sichtbar wird.
3. **Farbbestimmung — klassisch statt gelernt** (`api/vision/color_extraction.py`): k-means über die Pixel in der Maske im Lab-Raum, größtes Cluster = Grifffarbe, Mapping auf eine benannte Palette (`api/vision/classes.py`). Neue Grifffarbe = eine Zeile Palette, kein Retraining.
4. **Routengruppierung** (`api/vision/route_grouping.py`): DBSCAN je Farbe über die Griffzentren.

Warum der Umbau: Das alte Modell lernte 12 Farbklassen direkt — fragil bei Gym-Beleuchtung, und jede neue Farbe brauchte Retraining. Mit `hold`/`volume` als einzigen Klassen sind außerdem alle öffentlichen Kletter-Datasets als Trainingsdaten nutzbar, egal welche Farbklassen sie haben.

## Projektstruktur

```text
.
├─ api/
│  ├─ routes/            # Flask-Blueprints (auth, analysis, yolo_endpoint)
│  ├─ vision/            # Klassen/Palette, Farbextraktion, Routengruppierung
│  ├─ scripts/           # prepare_dataset.py (Roboflow Universe -> merged Dataset)
│  ├─ tests/             # pytest für die vision-Module
│  ├─ train.py           # YOLO26-Training
│  └─ weights/           # best.pt (nicht im Repo)
├─ app/                  # Flutter App
├─ docker-compose.yml
└─ .env.example
```

## Quickstart Backend

```bash
cp .env.example .env     # JWT_SECRET_KEY setzen!
docker compose up --build
```

Ohne Docker: `cd api && pip install -r requirements.txt && python server.py`

Ohne `api/weights/best.pt` startet der Server trotzdem; `/process` antwortet dann mit 503 und einem Hinweis. Die SAM-Gewichte lädt Ultralytics beim ersten Aufruf selbst herunter.

## Trainingsdaten besorgen

Es gibt keine versionierten Trainingsbilder — das externe Dataset wird reproduzierbar aus der kanonischen Roboflow-Universe-Quelle `rock-climbing-hold-detection` gebaut. Weitere gefundene Exporte derselben Originalbilder werden bewusst nicht noch einmal gemischt:

```bash
cd api
pip install -r requirements-train.txt
# ROBOFLOW_API_KEY in .env eintragen (kostenloser Account -> Settings -> API Key)
python scripts/prepare_dataset.py
```

Ergebnis: `api/datasets/merged_clean/` mit `data.yaml` (Klassen `hold`, `volume`). Der Builder nutzt genau eine kanonische Roboflow-Quelle, gruppiert alle Augmentierungen nach Ursprungsbild und verbindet zusätzlich byte-identische Dateien, bevor Train/Valid/Test geschrieben werden. Dadurch kann dasselbe Motiv nicht über Re-Uploads oder Augmentierungs-Geschwister in mehreren Splits landen. `scripts/audit_dataset_splits.py` ist das harte Leakage-Gate. Quellen stehen unter eigenen Lizenzen (i. d. R. CC BY 4.0) — bei Veröffentlichung nennen.

Eigene Hallenbilder werden separat mit `scripts/baue_halle_dataset.py` nach `datasets/eigene_halle_clean/` gebaut. Dessen Validierung enthält vorerst nur menschliche hold-Labels; Pseudo-Volumes landen ausschließlich im Training. `scripts/validate_weights.py` meldet deshalb zwei getrennte Gates: extern hold+volume und eigene Halle hold-only. Der externe Testsplit wird erst mit `--test` für das final ausgewählte Modell geöffnet.

## Training

Braucht eine GPU (lokal oder Colab):

```bash
cd api
python train.py --model yolo26s.pt --epochs 120
# danach:
cp runs/detect/boulder-holds-yolo26/weights/best.pt weights/best.pt
```

`--model yolo26n.pt` für schnelle Experimente, `yolo26m/l/x.pt` für mehr Genauigkeit.

## App starten

```bash
cd app
flutter pub get
flutter run
```

Backend-URL: Android-Emulator `http://10.0.2.2:5000`, iOS-Simulator `http://127.0.0.1:5000`, echtes Gerät → LAN-IP des Rechners.

## API (Auszug)

- `POST /register`, `POST /login` → JWT
- `POST /process` (multipart `image`, JWT) →

```json
{
  "detections": [{
    "det_class": "hold",
    "confidence": 0.91,
    "bbox": [x1, y1, x2, y2],
    "segmentation": [[x, y], ...],
    "color": {"name": "red", "hex": "#D32F2F", "index": 5, "confidence": 0.74},
    "class": 5, "class_name": "red", "color_hex": "#D32F2F"
  }],
  "routes": [[<detections>], ...],
  "original_image": "<base64>", "image_width": 4032, "image_height": 3024
}
```

`class`/`class_name` sind Kompatibilitäts-Aliase auf die Farbe (alte App-Stände funktionieren weiter).

### Videoverarbeitung

`POST /process_video` legt einen Hintergrundjob an. Der Standardrenderer arbeitet offline in zwei Durchlaeufen:

1. YOLO-Detektionen werden klassenbewusst ueber die Kamerabewegung zu Tracks verbunden. Die Klasse entsteht aus mehreren Beobachtungen; Boxen werden bei Rotation ueber alle vier Ecken transformiert.
2. Pro Track werden bis zu drei zeitlich getrennte, scharfe Frames segmentiert. SAM und die spezielle Volume-Hintergrundsubtraktion konkurrieren ueber dieselben Qualitaets-Gates. Die zeitlich konsistenteste Maske wird ohne anisotropes Polygon-Strecken in alle Frames projiziert.

Ein-/Ausblend-Hysterese verhindert harte Pop-ins. Der bisherige Ein-Pass-Renderer bleibt mit `VIDEO_PIPELINE=legacy` fuer reproduzierbare A/B-Vergleiche verfuegbar. Jeder neue API-Job schreibt `video_debug.json`; der Eigentuemer kann ihn unter `GET /process_video/<job_id>/debug` abrufen.

```bash
cd api
# lokaler Zwei-Pass-Render samt Diagnose
python scripts/video_probe.py --in video.mp4 --out annotiert.mp4 \
  --shapes --debug video_debug.json

# Ursachen getrennt ansehen: frische Masken, Trackboxen, fertige Zwei-Pass-Ausgabe
python scripts/video_ablation.py --in video.mp4 --out-dir datasets/video_ablation

# 59 feste Frames aus drei Hallenclips anlegen; vorhandene Annotationen bleiben erhalten
python scripts/build_video_holdout.py

# nach manueller Korrektur gegen einen Render messen
python scripts/evaluate_video_holdout.py \
  --manifest datasets/video_holdout/manifest.json \
  --debug video_debug.json --clip-id mov890
```

Solange die 59 Holdout-Frames noch `status: pending` tragen, gibt die Auswertung bewusst keine erfundenen Precision-/Recall-Werte aus.

## Konfiguration (.env)

| Variable | Zweck | Default |
| --- | --- | --- |
| `SERVER_PORT` | API-Port | `5000` |
| `SQLALCHEMY_DATABASE_URI` | DB | `sqlite:///boulder-ai.db` |
| `JWT_SECRET_KEY` | Token-Signierung | — (setzen!) |
| `WEIGHTS_PATH` | YOLO-Gewichte | `./weights/best.pt` |
| `DETECT_CONF` | Konfidenzschwelle | `0.4` |
| `USE_SAM` | Segmentierung an/aus | `1` |
| `SAM_WEIGHTS` | SAM-Variante | `sam2.1_b.pt` |
| `VIDEO_PIPELINE` | `two_pass` oder A/B-Fallback `legacy` | `two_pass` |
| `VIDEO_STRIDE` | YOLO nur jeden n-ten Frame ausfuehren | `2` |
| `VIDEO_KEYFRAME` | Mindestabstand der Best-Frame-Maskenkandidaten | `20` |
| `VIDEO_MASK_CANDIDATES` | maximal segmentierte Frames je Track | `3` |
| `VIDEO_TRACK_PATIENCE` | Trackleben ohne neue Detektion in Frames | `30` |
| `VIDEO_TEMPORAL_IOU` | Mindestkonsistenz mehrerer Maskenkandidaten | `0.12` |
| `VIDEO_FADE_FRAMES` | weiches Ein-/Ausblenden | `5` |
| `VIDEO_DEVICE` | optionales Ultralytics-Device, z. B. `cpu` | automatisch |
| `ROBOFLOW_API_KEY` | nur Datendownload | — |

## Tests

```bash
cd api && python -m pytest tests/ -v
```

## Ideen / nächste Schritte

- **SAM 3 mit Textprompt** („climbing hold") als Detektor+Segmentierer in einem Schritt testen — könnte YOLO+SAM ersetzen oder als Auto-Labeler fürs eigene Feintuning dienen.
- YOLO26n als TFLite direkt in die App exportieren → Inferenz ohne Server.
- Eigene Gym-Fotos nachannotieren (Roboflow/CVAT haben SAM-Auto-Labeling eingebaut) und aufs merged Dataset drauftrainieren.
