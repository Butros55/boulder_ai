# Boulder AI

[![Flutter](https://img.shields.io/badge/Flutter-Framework-02569B?logo=flutter&logoColor=white)](https://flutter.dev/)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-111111?logo=python&logoColor=white)](https://docs.ultralytics.com/)
[![SAM](https://img.shields.io/badge/SAM-Segment%20Anything-6E40C9)](https://github.com/facebookresearch/segment-anything)
[![SQLite](https://img.shields.io/badge/SQLite-DB-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)

**Boulder AI** ist ein Prototyp aus **Flutter App** (`app/`) + **Python/Flask Backend** (`api/`) für die Analyse von Boulder-/Kletterwand-Fotos:  
Griffe werden per **YOLO** erkannt, per **SAM** segmentiert und anschließend (vereinfacht) zu Routen gruppiert. Ergebnisse werden pro Nutzer (JWT) in einer **SQLite**-DB gespeichert.

---

## Inhalt

- [Überblick](#überblick)
- [Features](#features)
- [Projektstruktur](#projektstruktur)
- [Voraussetzungen](#voraussetzungen)
- [Quickstart](#quickstart)
- [Konfiguration](#konfiguration)
- [Modelle](#modelle)
- [Backend starten (Docker)](#backend-starten-docker)
- [App starten (Flutter)](#app-starten-flutter)
- [Backend-URL für Mobile/Emulator](#backend-url-für-mobileemulator)
- [Nutzung (Kurz)](#nutzung-kurz)
- [API (Auszug)](#api-auszug)
- [Training (optional)](#training-optional)
- [Troubleshooting](#troubleshooting)

---

## Überblick

- **Frontend:** Flutter App mit Login/Registrierung, Kamera/Galerie, Ergebnis-Overlay, Analyse-Historie  
- **Backend:** Flask REST API mit JWT Auth und SQLite Persistence  
- **ML:** YOLO für Detektion, SAM für Segmentierung (Weights nicht im Repo)

## Features

- Auth (Register/Login) mit JWT
- Bildanalyse + Visualisierung der Griffe
- Speichern der letzten Analysen pro Nutzer
- Docker-ready Backend

## Projektstruktur

```text
.
├─ api/                # Flask API + ML-Inferenz
├─ app/                # Flutter App
├─ docker-compose.yml  # Backend-Services
├─ .env.example        # Beispiel-Konfiguration
└─ start_backend.py    # Optionales Startskript
```

## Voraussetzungen

**Backend (empfohlen über Docker):**
- Docker Desktop (mit Compose)

**Frontend:**
- Flutter SDK (passend zu `app/pubspec.yaml`)

## Quickstart

1) `.env` anlegen  
2) Modelldateien hinzufügen  
3) Backend starten  
4) App starten  

Details siehe die folgenden Abschnitte.

## Konfiguration

Erstelle eine `.env` im Repo-Root:

- macOS/Linux:
  ```bash
  cp .env.example .env
  ```
- Windows (PowerShell):
  ```powershell
  Copy-Item .env.example .env
  ```

Wichtige Variablen in `.env`:

- `SERVER_PORT` (Standard: `5000`)
- `ML_BACKEND_PORT` (Standard: `9090`)
- `SQLALCHEMY_DATABASE_URI` (Standard: `sqlite:///boulder-ai.db`)
- `JWT_SECRET_KEY` (**ändern**, sobald mehr als lokal genutzt)

## Modelle

Das Backend erwartet folgende Dateien (nicht im Repo):

- YOLO Gewichte: `api/weights/best.pt`
- SAM Checkpoint: `api/sam2_b.pt`

Hinweis: `*.pt` ist in `.gitignore` ausgeschlossen – die Dateien musst du lokal hinzufügen (oder per Volume mounten).

## Backend starten (Docker)

Im Repo-Root:

```bash
docker compose up --build
```

**Services:**
- `server`: Flask API auf `http://localhost:5000`  
  _Hinweis:_ Port ist aktuell in `api/server.py` auf `5000` fixiert.  
  `SERVER_PORT` sollte daher `5000` sein oder du passt `api/server.py` an.
- `ml_backend`: Label-Studio-ML Backend (Port über `ML_BACKEND_PORT`, optional/experimentell)

Stoppen:

```bash
docker compose down
```

Optionales Wrapper-Skript:

```bash
python start_backend.py
```

## App starten (Flutter)

```bash
cd app
flutter pub get
flutter run
```

## Backend-URL für Mobile/Emulator

Die App nutzt aktuell **fest** `http://127.0.0.1:5000` (z. B. in `app/lib/util/image_processor.dart` sowie den Login/Register-Screens).

- Flutter Web/Desktop lokal: `127.0.0.1` passt meist.
- Android Emulator: oft `http://10.0.2.2:5000` statt `127.0.0.1`.
- Physisches Gerät: Host-IP im WLAN nutzen (Firewall/Portfreigabe beachten).

## Nutzung (Kurz)

1) Backend starten  
2) App starten  
3) In der App registrieren → einloggen  
4) Bild aufnehmen oder auswählen → Analyse ansehen  
5) Unter „Recent Analyses“ kannst du die letzten Analysen erneut öffnen  

## API (Auszug)

Basis: `http://localhost:5000`

- `POST /register` – `{ "username": "...", "password": "..." }`
- `POST /login` – `{ "username": "...", "password": "..." }` → `access_token`
- `POST /process` – `multipart/form-data` (`image=@file`) + `Authorization: Bearer <token>`
- `GET /analyses` – `Authorization: Bearer <token>` → letzte Analysen + Gesamtanzahl

Beispiel (cURL):

```bash
curl -s -X POST http://localhost:5000/register \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo"}'

curl -s -X POST http://localhost:5000/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo"}'
# -> {"access_token":"<TOKEN>"}

curl -X POST http://localhost:5000/process \
  -H "Authorization: Bearer <TOKEN>" \
  -F "image=@./path/to/image.jpg"
```

## Training (optional)

- Dataset-Konfiguration: `api/data.yaml`
- Training-Skript: `api/train.py` (Ultralytics)

Das Repo enthält keine Trainingsdaten/Weights – du musst Pfade/Dateinamen ggf. an deine Umgebung anpassen.

## Troubleshooting

**401 Unauthorized**  
→ Token fehlt/abgelaufen. Bitte erneut einloggen.

**Keine Verbindung vom Emulator**  
→ Verwende `http://10.0.2.2:5000` statt `127.0.0.1`.

**FileNotFoundError für Weights**  
→ Stelle sicher, dass `api/weights/best.pt` und `api/sam2_b.pt` existieren.
