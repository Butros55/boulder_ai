"""Drei Ursachenebenen derselben Videosequenz getrennt rendern.

``fresh`` zeigt unabhaengige YOLO+SAM-Ergebnisse ohne Tracking, ``tracks``
zeigt nur die zeitliche Boxzuordnung und ``two_pass`` das Produktionsoverlay.
Jede Variante laeuft in einem eigenen Prozess, damit GPU-Speicher garantiert
freigegeben wird.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))


def _writer(path: Path, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), meta.fps,
        (meta.width, meta.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Ausgabe nicht schreibbar: {path}")
    return writer


def _models(weights, *, sam_enabled: bool):
    from ultralytics import SAM, YOLO

    model = YOLO(weights or str(API_DIR / "weights" / "best.pt"))
    sam = None
    if sam_enabled:
        import torch

        default = "sam2.1_b.pt" if torch.cuda.is_available() else "sam2.1_t.pt"
        sam = SAM(os.getenv("SAM_WEIGHTS", default))
    return model, sam


def render_fresh(args, destination: Path):
    """Frische Einzelbildmasken: absichtlich keinerlei Zustand zwischen Frames."""
    from vision.color_extraction import extract_color
    from vision.video_masks import candidate_from_mask, volume_mask_candidates
    from vision.video_overlay_v2 import (
        _open_capture,
        _oriented,
        _predict_detections,
        _sam_masks,
        _video_meta,
    )
    from vision.video_tracking import crop_sharpness

    meta = _video_meta(args.src, args.max_seconds)
    model, sam = _models(args.weights, sam_enabled=True)
    capture = _open_capture(args.src)
    writer = _writer(destination, meta)
    total = meta.frames if meta.frames > 0 else int(1e12)
    for frame_index in range(total):
        ok, raw = capture.read()
        if not ok:
            break
        frame = _oriented(raw, meta)
        if frame_index % args.stride == 0:
            detections = _predict_detections(
                model, frame, args.conf, device=args.device
            )
            masks = _sam_masks(
                sam, frame, [list(item.bbox) for item in detections],
                device=args.device,
            )
            overlay = frame.copy()
            for index, detection in enumerate(detections):
                sharpness = crop_sharpness(frame, detection.bbox)
                candidates = []
                if index < len(masks):
                    candidates.append(
                        candidate_from_mask(
                            frame,
                            detection.bbox,
                            masks[index],
                            class_id=detection.class_id,
                            source="sam",
                            confidence=detection.confidence,
                            sharpness=sharpness,
                        )
                    )
                if detection.class_id == 1:
                    candidates.extend(
                        volume_mask_candidates(
                            frame,
                            detection.bbox,
                            confidence=detection.confidence,
                            sharpness=sharpness,
                        )
                    )
                accepted = [candidate for candidate in candidates if candidate.accepted]
                if not accepted:
                    continue
                best = max(accepted, key=lambda candidate: candidate.score)
                color = extract_color(
                    frame,
                    bbox=detection.bbox,
                    polygon=best.polygon.astype(int).tolist(),
                )
                value = color["hex"].lstrip("#")
                bgr = (int(value[4:6], 16), int(value[2:4], 16), int(value[:2], 16))
                points = best.polygon.astype(np.int32)
                cv2.fillPoly(overlay, [points], bgr)
                cv2.polylines(overlay, [points], True, bgr, 2, cv2.LINE_AA)
            frame = cv2.addWeighted(overlay, 0.30, frame, 0.70, 0.0)
        writer.write(frame)
    capture.release()
    writer.release()


def _world_box(track, poses):
    polygons = []
    for observation in track.observations:
        x0, y0, x1, y1 = observation.bbox
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        from vision.video_tracking import transform_points

        polygons.append(
            transform_points(corners, np.linalg.inv(poses[observation.frame]))
        )
    return np.median(np.asarray(polygons), axis=0) if polygons else None


def render_tracks(args, destination: Path):
    """Nur Track-Geometrie; trennt Detektor/Zuordnung von Maskenfehlern."""
    from vision.video_overlay_v2 import (
        _analyse_video,
        _open_capture,
        _oriented,
        _video_meta,
    )
    from vision.video_tracking import transform_points

    meta = _video_meta(args.src, args.max_seconds)
    model, _ = _models(args.weights, sam_enabled=False)
    tracks, poses, segments, _, processed = _analyse_video(
        args.src,
        meta,
        model,
        confidence=args.conf,
        stride=args.stride,
        device=args.device,
        progress_cb=None,
    )
    world_boxes = {track.id: _world_box(track, poses) for track in tracks}
    capture = _open_capture(args.src)
    writer = _writer(destination, meta)
    colors = {0: (208, 99, 30), 1: (0, 174, 255)}
    for frame_index in range(processed):
        ok, raw = capture.read()
        if not ok:
            break
        frame = _oriented(raw, meta)
        for track in tracks:
            end = track.ended_frame if track.ended_frame is not None else track.last_frame
            world = world_boxes[track.id]
            if world is None or track.segment_id != segments[frame_index] \
                    or not track.first_frame <= frame_index <= end:
                continue
            points = transform_points(world, poses[frame_index]).astype(np.int32)
            color = colors.get(track.class_id, (180, 180, 180))
            cv2.polylines(frame, [points], True, color, 2, cv2.LINE_AA)
            anchor = tuple(points[np.argmin(points[:, 1])])
            cv2.putText(
                frame, f"T{track.id}", anchor, cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, cv2.LINE_AA,
            )
        writer.write(frame)
    capture.release()
    writer.release()


def render_existing(args, destination: Path, *, legacy: bool):
    from vision.video_overlay import render_video, render_video_legacy

    renderer = render_video_legacy if legacy else render_video
    kwargs = {}
    if not legacy:
        kwargs.update(
            debug_path=str(destination.with_suffix(".debug.json")),
            device=args.device,
        )
    renderer(
        args.src,
        destination,
        weights=args.weights,
        conf=args.conf,
        stride=args.stride,
        shapes=True,
        keyframe=args.keyframe,
        max_seconds=args.max_seconds,
        **kwargs,
    )


def _single(args, variant: str, destination: Path):
    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if variant == "fresh":
        render_fresh(args, destination)
    elif variant == "tracks":
        render_tracks(args, destination)
    elif variant == "legacy":
        render_existing(args, destination, legacy=True)
    elif variant == "two_pass":
        render_existing(args, destination, legacy=False)
    else:
        raise ValueError(variant)


def main():
    parser = argparse.ArgumentParser(description="Video-Ablation rendern")
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--weights")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--keyframe", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=90.0)
    parser.add_argument("--device")
    parser.add_argument(
        "--variants", default="fresh,tracks,two_pass",
        help="Kommagetrennt: fresh,tracks,legacy,two_pass",
    )
    parser.add_argument("--single", choices=["fresh", "tracks", "legacy", "two_pass"])
    parser.add_argument("--single-out", type=Path)
    args = parser.parse_args()
    args.stride = max(1, args.stride)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.single:
        if args.single_out is None:
            parser.error("--single-out fehlt")
        _single(args, args.single, args.single_out)
        return

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    stem = Path(args.src).stem
    for variant in variants:
        destination = args.out_dir / f"{stem}_{variant}.mp4"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--in", args.src,
            "--out-dir", str(args.out_dir),
            "--conf", str(args.conf),
            "--stride", str(args.stride),
            "--keyframe", str(args.keyframe),
            "--max-seconds", str(args.max_seconds),
            "--single", variant,
            "--single-out", str(destination),
        ]
        if args.weights:
            command.extend(["--weights", args.weights])
        if args.device:
            command.extend(["--device", args.device])
        print(f"[{variant}] -> {destination}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
