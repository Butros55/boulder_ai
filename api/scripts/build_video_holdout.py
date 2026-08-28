"""Reproduzierbare Keyframes fuer den Video-Holdout extrahieren.

Der Builder erzeugt bewusst noch keine Ground-Truth. Bestehende manuelle
Annotationen im Manifest werden bei erneutem Lauf anhand der Bild-ID erhalten.

Aufruf aus ``api``::

    python scripts/build_video_holdout.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

API_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = API_DIR.parent


def sample_indices(total: int, count: int, included=()) -> list[int]:
    if total <= 0:
        return []
    count = max(1, min(int(count), total))
    evenly = np.linspace(0, total - 1, count, dtype=int).tolist()
    valid_included = [int(index) for index in included if 0 <= int(index) < total]
    return sorted(set(evenly + valid_included))


def _existing_annotations(manifest_path: Path) -> dict[str, dict]:
    if not manifest_path.exists():
        return {}
    try:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(item["id"]): item for item in old.get("frames", [])}


def _resolve_video(path_string: str) -> Path:
    path = Path(path_string)
    return path if path.is_absolute() else REPO_DIR / path


def build(config_path: Path, output_dir: Path, *, force: bool = False) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    previous = _existing_annotations(manifest_path)
    frames = []
    clips = []

    for clip in config["clips"]:
        source = _resolve_video(clip["path"])
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise FileNotFoundError(f"Video nicht lesbar: {source}")
        try:
            capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        except cv2.error:
            pass
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        selected = sample_indices(
            total,
            int(clip.get("sample_count", 18)),
            clip.get("include_frames", []),
        )
        clip_written = 0
        for frame_index in selected:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            frame_id = f"{clip['id']}_f{frame_index:05d}"
            filename = f"{frame_id}.jpg"
            target = images_dir / filename
            if force or not target.exists():
                if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"Frame konnte nicht geschrieben werden: {target}")
            old = previous.get(frame_id, {})
            frames.append(
                {
                    "id": frame_id,
                    "clip_id": clip["id"],
                    "frame": frame_index,
                    "seconds": round(frame_index / fps, 3) if fps else None,
                    "image": f"images/{filename}",
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "status": old.get("status", "pending"),
                    "annotations": old.get("annotations", []),
                }
            )
            clip_written += 1
        capture.release()
        clips.append(
            {
                "id": clip["id"],
                "source": str(source),
                "fps": fps,
                "total_frames": total,
                "sampled_frames": clip_written,
            }
        )

    result = {
        "version": 1,
        "schema": {
            "annotation": {
                "class_id": "0=hold, 1=volume",
                "bbox": "[x1,y1,x2,y2]",
                "polygon": "[[x,y], ...]",
            }
        },
        "clips": clips,
        "frames": frames,
        "summary": {
            "clips": len(clips),
            "frames": len(frames),
            "annotated": sum(item["status"] == "done" for item in frames),
            "pending": sum(item["status"] != "done" for item in frames),
        },
    }
    manifest_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Video-Holdout-Keyframes extrahieren")
    parser.add_argument(
        "--config",
        type=Path,
        default=API_DIR / "configs" / "video_holdout.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=API_DIR / "datasets" / "video_holdout",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = build(args.config.resolve(), args.out.resolve(), force=args.force)
    print(json.dumps(result["summary"], ensure_ascii=False))
    for clip in result["clips"]:
        print(
            f"{clip['id']}: {clip['sampled_frames']} Frames aus "
            f"{clip['total_frames']} ({clip['fps']:.2f} FPS)"
        )


if __name__ == "__main__":
    main()
