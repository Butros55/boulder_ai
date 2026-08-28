"""Videoverarbeitung: Upload, Hintergrund-Rendering, Abruf.

Ablauf für die App:
    POST /process_video            (multipart, Feld "video") -> {"job_id": ...}
    GET  /process_video/<job_id>   -> {"status": "queued|running|done|error",
                                       "progress": 0.0-1.0, ...}
    GET  /process_video/<job_id>/result -> das annotierte MP4

Das Rendering läuft in einem Hintergrund-Thread (ein Job zur Zeit, weitere
warten in der Reihenfolge des Eingangs). Dateien liegen unter
instance/videos/<job_id>/ (gitignored) und gehören dem hochladenden Nutzer.
"""

import os
import threading
import uuid
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file
from flask_jwt_extended import get_jwt_identity, jwt_required

video_bp = Blueprint("video_bp", __name__)

VIDEOS_DIR = Path(__file__).resolve().parent.parent / "instance" / "videos"
ERLAUBT = {".mp4", ".mov", ".m4v", ".avi", ".webm"}
MAX_SEKUNDEN = float(os.getenv("VIDEO_MAX_SECONDS", "60"))

_jobs = {}
_queue_lock = threading.Lock()
_worker_busy = threading.Event()


def _render_job(job_id):
    job = _jobs[job_id]
    try:
        job["status"] = "running"
        from vision.video_overlay import render_video, render_video_legacy

        def cb(p):
            job["progress"] = round(p, 3)

        pipeline = os.getenv("VIDEO_PIPELINE", "two_pass").strip().lower()
        if pipeline not in {"two_pass", "legacy"}:
            raise ValueError(
                "VIDEO_PIPELINE muss 'two_pass' oder 'legacy' sein"
            )
        renderer = render_video_legacy if pipeline == "legacy" else render_video
        render_kwargs = {}
        if renderer is render_video:
            render_kwargs.update(
                debug_path=job["debug"],
                device=os.getenv("VIDEO_DEVICE") or None,
            )
        frames, dauer = renderer(
            job["src"], job["dst"],
            conf=float(os.getenv("DETECT_CONF", "0.4")),
            shapes=os.getenv("VIDEO_SHAPES", "1") == "1",
            stride=int(os.getenv("VIDEO_STRIDE", "2")),
            keyframe=int(os.getenv("VIDEO_KEYFRAME", "20")),
            max_seconds=MAX_SEKUNDEN, progress_cb=cb,
            **render_kwargs,
        )
        job.update(status="done", progress=1.0, frames=frames,
                   dauer=round(dauer, 1), pipeline=pipeline)
    except Exception as e:  # Fehler landet beim Client, nicht im Serverlog-Nirwana
        job.update(status="error", fehler=str(e)[:300])
    finally:
        _worker_busy.clear()
        _naechsten_starten()


def _naechsten_starten():
    with _queue_lock:
        if _worker_busy.is_set():
            return
        wartend = [j for j in _jobs.values() if j["status"] == "queued"]
        if not wartend:
            return
        job = sorted(wartend, key=lambda j: j["reihe"])[0]
        _worker_busy.set()
        threading.Thread(target=_render_job, args=(job["id"],),
                         daemon=True).start()


@video_bp.route("/process_video", methods=["POST"])
@jwt_required()
def upload_video():
    if "video" not in request.files:
        return jsonify({"msg": "Videodatei fehlt (Feld 'video')"}), 400
    f = request.files["video"]
    endung = Path(f.filename or "video.mp4").suffix.lower()
    if endung not in ERLAUBT:
        return jsonify({"msg": f"Format {endung} nicht unterstützt"}), 400

    job_id = uuid.uuid4().hex[:12]
    ordner = VIDEOS_DIR / job_id
    ordner.mkdir(parents=True, exist_ok=True)
    src = ordner / f"eingang{endung}"
    f.save(src)

    _jobs[job_id] = {
        "id": job_id,
        "user_id": get_jwt_identity(),
        "src": str(src),
        "dst": str(ordner / "annotiert.mp4"),
        "debug": str(ordner / "video_debug.json"),
        "status": "queued",
        "progress": 0.0,
        "reihe": len(_jobs),
        "max_sekunden": MAX_SEKUNDEN,
    }
    _naechsten_starten()
    return jsonify({"job_id": job_id, "status": "queued",
                    "hinweis": f"maximal die ersten {MAX_SEKUNDEN:.0f}s "
                               "werden verarbeitet"}), 201


def _job_oder_404(job_id):
    job = _jobs.get(job_id)
    if job is None or str(job["user_id"]) != str(get_jwt_identity()):
        return None
    return job


@video_bp.route("/process_video/<job_id>", methods=["GET"])
@jwt_required()
def job_status(job_id):
    job = _job_oder_404(job_id)
    if job is None:
        return jsonify({"msg": "Job nicht gefunden"}), 404
    antwort = {k: job[k] for k in ("status", "progress")}
    antwort["job_id"] = job_id
    for k in ("frames", "dauer", "fehler", "pipeline"):
        if k in job:
            antwort[k] = job[k]
    return jsonify(antwort)


@video_bp.route("/process_video/<job_id>/result", methods=["GET"])
@jwt_required()
def job_result(job_id):
    job = _job_oder_404(job_id)
    if job is None:
        return jsonify({"msg": "Job nicht gefunden"}), 404
    if job["status"] != "done":
        return jsonify({"msg": "noch nicht fertig", "status": job["status"]}), 409
    return send_file(job["dst"], mimetype="video/mp4",
                     as_attachment=False, download_name="annotiert.mp4",
                     conditional=True)


@video_bp.route("/process_video/<job_id>/debug", methods=["GET"])
@jwt_required()
def job_debug(job_id):
    """Messdaten fuer A/B-Vergleiche; nur fuer den Job-Eigentuemer."""
    job = _job_oder_404(job_id)
    if job is None:
        return jsonify({"msg": "Job nicht gefunden"}), 404
    if job["status"] != "done":
        return jsonify({"msg": "noch nicht fertig", "status": job["status"]}), 409
    if not Path(job["debug"]).exists():
        return jsonify({"msg": "Keine Debugdaten fuer diese Pipeline"}), 404
    return send_file(job["debug"], mimetype="application/json",
                     as_attachment=True, download_name="video_debug.json",
                     conditional=True)
