"""Stabiles Zwei-Pass-Overlay fuer Hallenvideos.

Pass 1 erkennt Objekte und Kamerabewegung ueber den gesamten Clip. Pass 2
segmentiert jeden bestaetigten Track auf seinen schaerfsten Frames und waehlt
die zeitlich konsistenteste Maske. Erst danach wird gerendert. Dadurch kann
eine unscharfe Einzelaufnahme keine falsche Kontur mehr 45 Frames lang
konservieren und neue Objekte poppen nicht zufaellig pro Frame auf.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vision.video_masks import (
    MaskCandidate,
    candidate_from_mask,
    polygon_iou,
    volume_mask_candidates,
)
from vision.video_tracking import (
    Detection,
    IDENTITY_AFFINE,
    MotionEstimate,
    Observation,
    Track,
    TrackManager,
    affine_to_homogeneous,
    bbox_center,
    estimate_camera_motion,
    relative_pose,
    transform_points,
)

API_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class VideoMeta:
    fps: float
    width: int
    height: int
    frames: int
    rotate: int | None


@dataclass
class _EvaluatedCandidate:
    track: Track
    observation: Observation
    mask: MaskCandidate
    adjusted_score: float = -100.0
    temporal_iou: float = 0.0


def _hex_to_bgr(hex_string: str) -> tuple[int, int, int]:
    value = hex_string.lstrip("#")
    return (int(value[4:6], 16), int(value[2:4], 16), int(value[0:2], 16))


def _video_meta(source: str | Path, max_seconds: float) -> VideoMeta:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Video nicht lesbar: {source}")
    try:
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except cv2.error:
        pass
    rotation_meta = int(capture.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
    ok, probe = capture.read()
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if not ok or probe is None:
        raise ValueError(f"Erster Videoframe nicht lesbar: {source}")

    rotate = None
    if rotation_meta in (90, 270) and probe.shape[1] > probe.shape[0]:
        rotate = (
            cv2.ROTATE_90_CLOCKWISE
            if rotation_meta == 90
            else cv2.ROTATE_90_COUNTERCLOCKWISE
        )
    elif rotation_meta == 180:
        rotate = cv2.ROTATE_180
    height, width = probe.shape[:2]
    if rotate in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        height, width = width, height
    if max_seconds and total > 0:
        total = min(total, max(1, int(round(max_seconds * fps))))
    return VideoMeta(fps=fps, width=width, height=height, frames=total, rotate=rotate)


def _open_capture(source: str | Path):
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Video nicht lesbar: {source}")
    try:
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except cv2.error:
        pass
    return capture


def _oriented(frame: np.ndarray, meta: VideoMeta) -> np.ndarray:
    if meta.rotate is not None and frame.shape[1] != meta.width:
        return cv2.rotate(frame, meta.rotate)
    return frame


def _small_gray(frame: np.ndarray, width: int = 480) -> np.ndarray:
    height = max(1, int(round(width * frame.shape[0] / frame.shape[1])))
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def _predict_detections(model, frame: np.ndarray, confidence: float, device=None):
    kwargs = {"source": frame, "conf": confidence, "verbose": False}
    if device is not None:
        kwargs["device"] = device
    result = model.predict(**kwargs)[0]
    detections: list[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections
    for box in boxes:
        bbox = tuple(float(value) for value in box.xyxy[0])
        detections.append(
            Detection(
                bbox=bbox,
                class_id=int(box.cls[0]),
                confidence=float(box.conf[0]),
            )
        )
    return detections


def _progress(callback, value: float) -> None:
    if callback is not None:
        callback(float(np.clip(value, 0.0, 1.0)))


def _analyse_video(
    source: str | Path,
    meta: VideoMeta,
    model,
    *,
    confidence: float,
    stride: int,
    device,
    progress_cb,
) -> tuple[list[Track], list[np.ndarray], list[int], list[dict], int]:
    patience_frames = max(4, int(os.getenv("VIDEO_TRACK_PATIENCE", "30")))
    manager = TrackManager(
        confirmations=max(2, int(os.getenv("VIDEO_CONFIRMATIONS", "2"))),
        patience=max(2, math.ceil(patience_frames / max(1, stride))),
        match_iou=float(os.getenv("VIDEO_MATCH_IOU", "0.28")),
        merge_iou=float(os.getenv("VIDEO_MERGE_IOU", "0.72")),
    )
    capture = _open_capture(source)
    previous_gray = None
    current_pose = np.eye(3, dtype=np.float64)
    poses: list[np.ndarray] = []
    segments: list[int] = []
    debug_frames: list[dict] = []
    low_motion_streak = 0
    processed = 0

    limit = meta.frames if meta.frames > 0 else int(1e12)
    for frame_index in range(limit):
        ok, raw = capture.read()
        if not ok:
            break
        frame = _oriented(raw, meta)
        gray = _small_gray(frame)
        if previous_gray is None:
            motion = MotionEstimate(
                matrix=IDENTITY_AFFINE.copy(), method="initial", quality=1.0
            )
        else:
            motion = estimate_camera_motion(previous_gray, gray, meta.width)

        events = manager.move(motion, frame_index)
        if motion.scene_cut:
            current_pose = np.eye(3, dtype=np.float64)
            low_motion_streak = 0
        else:
            current_pose = affine_to_homogeneous(motion.matrix) @ current_pose
            low_motion_streak = low_motion_streak + 1 if motion.quality < 0.08 else 0
        poses.append(current_pose.copy())
        segments.append(manager.segment_id)

        detection_count = 0
        if frame_index % stride == 0:
            detections = _predict_detections(model, frame, confidence, device=device)
            detection_count = len(detections)
            # Bei mehreren unsicheren Motion-Frames duerfen unbestaetigte
            # Alttracks nicht noch eine Sekunde als Geistermaske ueberleben.
            if low_motion_streak >= max(3, stride):
                for track in manager.active:
                    track.missed += 1
            events.extend(manager.update(frame_index, frame, detections))

        debug_frames.append(
            {
                "frame": frame_index,
                "detections": detection_count,
                "active_tracks": len(manager.active),
                "segment": manager.segment_id,
                "motion": motion.as_debug(),
                "events": events,
            }
        )
        previous_gray = gray
        processed = frame_index + 1
        if frame_index % 10 == 0:
            _progress(progress_cb, 0.45 * processed / max(1, meta.frames))

    capture.release()
    tracks = manager.finish(max(0, processed - 1))
    tracks = [track for track in tracks if track.hits >= manager.confirmations]
    return tracks, poses, segments, debug_frames, processed


def _sam_masks(sam, frame: np.ndarray, boxes: list[list[float]], device=None):
    if sam is None or not boxes:
        return []
    kwargs = {
        "source": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        "bboxes": boxes,
        "verbose": False,
    }
    if device is not None:
        kwargs["device"] = device
    results = sam.predict(**kwargs)
    masks = getattr(results[0], "masks", None) if results else None
    if masks is None:
        return []
    data = np.asarray(masks.data.cpu().numpy())
    if data.ndim == 2:
        data = data[None, ...]
    output = []
    for mask in data:
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.float32),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        output.append(mask > 0.5)
    return output


def _candidate_plan(tracks: list[Track], keyframe: int) -> dict[int, list[tuple[Track, Observation]]]:
    """Waehlt gute Maskenframes und buendelt moeglichst viele Tracks.

    Ein rein trackweises Top-N erzeugt bei 80 Griffen leicht 150 verschiedene
    SAM-Aufrufe. Hier konkurrieren stattdessen ganze Frames anhand der Summe
    ihrer Beobachtungsqualitaet. Ein scharfer Frame mit vielen Detektionen
    wird dadurch fuer alle passenden Tracks gemeinsam genutzt.
    """
    per_frame: dict[int, list[tuple[Track, Observation]]] = defaultdict(list)
    maximum = max(1, int(os.getenv("VIDEO_MASK_CANDIDATES", "3")))
    minimum_distance = max(2, keyframe // 2)
    wanted: dict[int, int] = {}
    observations_by_frame: dict[int, list[tuple[Track, Observation]]] = defaultdict(list)
    for track in tracks:
        duration = max(1, track.last_frame - track.first_frame + 1)
        wanted[track.id] = min(
            maximum, max(2, math.ceil(duration / max(1, keyframe)))
        )
        for observation in track.observations:
            observations_by_frame[observation.frame].append((track, observation))

    selected: dict[int, list[Observation]] = defaultdict(list)
    frame_order = sorted(
        observations_by_frame,
        key=lambda frame: (
            sum(observation.quality for _, observation in observations_by_frame[frame]),
            len(observations_by_frame[frame]),
        ),
        reverse=True,
    )
    for frame in frame_order:
        additions = []
        for track, observation in sorted(
            observations_by_frame[frame],
            key=lambda item: item[1].quality,
            reverse=True,
        ):
            current = selected[track.id]
            if len(current) >= wanted[track.id]:
                continue
            if any(
                abs(observation.frame - other.frame) < minimum_distance
                for other in current
            ):
                continue
            additions.append((track, observation))
            current.append(observation)
        if additions:
            per_frame[frame].extend(additions)
        if all(len(selected[track.id]) >= wanted[track.id] for track in tracks):
            break

    # Sehr kurze oder oft verpasste Tracks koennen in den gemeinsamen Frames
    # leer ausgehen. Ihr bester Einzelbeleg bleibt als sicherer Fallback.
    for track in tracks:
        if selected[track.id]:
            continue
        fallback = track.candidate_observations(count=1)
        if fallback:
            per_frame[fallback[0].frame].append((track, fallback[0]))
    return per_frame


def _measure_candidate_color(candidate: MaskCandidate, frame, bbox, extract_color):
    if not candidate.accepted or candidate.polygon is None:
        return
    candidate.color = extract_color(
        frame, bbox=bbox, polygon=candidate.polygon.astype(int).tolist()
    )


def _segment_candidates(
    source: str | Path,
    meta: VideoMeta,
    tracks: list[Track],
    sam,
    *,
    keyframe: int,
    device,
    progress_cb,
) -> tuple[dict[int, list[_EvaluatedCandidate]], list[dict]]:
    from vision.color_extraction import extract_color

    plan = _candidate_plan(tracks, keyframe)
    needed = sorted(plan)
    candidates: dict[int, list[_EvaluatedCandidate]] = defaultdict(list)
    debug: list[dict] = []
    if not needed:
        return candidates, debug

    needed_set = set(needed)
    capture = _open_capture(source)
    handled = 0
    last_needed = needed[-1]
    for frame_index in range(last_needed + 1):
        ok, raw = capture.read()
        if not ok:
            break
        if frame_index not in needed_set:
            continue
        frame = _oriented(raw, meta)
        items = plan[frame_index]
        boxes = [list(observation.bbox) for _, observation in items]
        masks = _sam_masks(sam, frame, boxes, device=device)

        for item_index, (track, observation) in enumerate(items):
            generated: list[MaskCandidate] = []
            if item_index < len(masks):
                generated.append(
                    candidate_from_mask(
                        frame,
                        observation.bbox,
                        masks[item_index],
                        class_id=track.class_id,
                        source="sam",
                        confidence=observation.confidence,
                        sharpness=observation.sharpness,
                    )
                )
            if track.class_id == 1:
                generated.extend(
                    volume_mask_candidates(
                        frame,
                        observation.bbox,
                        confidence=observation.confidence,
                        sharpness=observation.sharpness,
                    )
                )
            for candidate in generated:
                _measure_candidate_color(
                    candidate, frame, observation.bbox, extract_color
                )
                evaluated = _EvaluatedCandidate(track, observation, candidate)
                candidates[track.id].append(evaluated)
                debug.append(
                    {
                        "track_id": track.id,
                        "frame": frame_index,
                        **candidate.as_debug(),
                    }
                )
        handled += 1
        _progress(progress_cb, 0.45 + 0.30 * handled / max(1, len(needed)))
    capture.release()
    return candidates, debug


def _apply_track_fallback(
    track: Track, poses: list[np.ndarray], segments: list[int]
) -> None:
    """Ohne bestandene Maske bleibt die Detektion als Box-Umriss sichtbar.

    Die Anti-Segel-Gates verwerfen nur die Form; die Detektion selbst ist
    durch Bestaetigungen belegt und darf nicht komplett verschwinden.
    """
    corner_lists = []
    centers = []
    for observation in track.observations:
        if observation.frame >= len(poses) or segments[observation.frame] != track.segment_id:
            continue
        inverse = np.linalg.inv(poses[observation.frame])
        x0, y0, x1, y1 = observation.bbox
        corners = np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64
        )
        corner_lists.append(transform_points(corners, inverse))
        centers.append(
            transform_points(
                np.array([bbox_center(observation.bbox)], dtype=np.float64), inverse
            )[0]
        )
    if not corner_lists:
        return
    world_corners = np.median(np.asarray(corner_lists), axis=0)
    target_center = np.median(np.asarray(centers), axis=0)
    box_center = (world_corners.min(axis=0) + world_corners.max(axis=0)) / 2.0
    world_corners += target_center - box_center

    residuals = []
    for observation in track.observations:
        if observation.frame >= len(poses) or segments[observation.frame] != track.segment_id:
            continue
        predicted = transform_points(
            np.array([target_center]), poses[observation.frame]
        )[0]
        actual = np.asarray(bbox_center(observation.bbox))
        residuals.append(
            (
                observation.frame,
                float(actual[0] - predicted[0]),
                float(actual[1] - predicted[1]),
            )
        )
    track.fallback_box_world = world_corners
    track.alignment_residuals = _smooth_residuals(residuals)


def _select_track_masks(
    tracks: list[Track],
    candidates: dict[int, list[_EvaluatedCandidate]],
    poses: list[np.ndarray],
    segments: list[int],
) -> None:
    temporal_minimum = float(os.getenv("VIDEO_TEMPORAL_IOU", "0.12"))
    for track in tracks:
        available = [item for item in candidates.get(track.id, []) if item.mask.accepted]
        if not available:
            _apply_track_fallback(track, poses, segments)
            continue

        distinct_frames = {item.observation.frame for item in available}
        for item in available:
            agreements = []
            for other in available:
                if other.observation.frame == item.observation.frame:
                    continue
                first_frame = item.observation.frame
                second_frame = other.observation.frame
                if segments[first_frame] != segments[second_frame]:
                    continue
                transform = relative_pose(poses[first_frame], poses[second_frame])
                # Globale Kamerapose plus kleine lokale Detektionskorrektur.
                # So misst die zeitliche IoU die Maskenform und nicht den
                # unvermeidbaren Restfehler des globalen Wandmodells.
                first_center = np.array(
                    [bbox_center(item.observation.bbox)], dtype=np.float64
                )
                predicted_center = transform_points(first_center, transform)[0]
                target_center = np.asarray(bbox_center(other.observation.bbox))
                transform = transform.copy()
                transform[:2, 2] += target_center - predicted_center
                agreements.append(
                    polygon_iou(
                        item.mask.polygon,
                        other.mask.polygon,
                        first_to_second=transform,
                    )
                )
            item.temporal_iou = max(agreements, default=0.0)
            item.mask.metrics["temporal_iou"] = round(item.temporal_iou, 4)
            item.adjusted_score = item.mask.score + 0.90 * item.temporal_iou

        stable = available
        if len(distinct_frames) >= 2:
            stable = [item for item in available if item.temporal_iou >= temporal_minimum]
        if not stable:
            # Lieber gar keine Form als eine zeitlich klar widerspruechliche —
            # die Detektion bleibt aber als Box-Umriss sichtbar.
            _apply_track_fallback(track, poses, segments)
            continue
        selected = max(stable, key=lambda item: item.adjusted_score)
        polygon = np.asarray(selected.mask.polygon, dtype=np.float64)
        reference_frame = selected.observation.frame

        world_polygon = transform_points(polygon, np.linalg.inv(poses[reference_frame]))
        world_centers = []
        for observation in track.observations:
            if observation.frame >= len(poses) or segments[observation.frame] != track.segment_id:
                continue
            center = np.array([bbox_center(observation.bbox)], dtype=np.float64)
            world_centers.append(
                transform_points(center, np.linalg.inv(poses[observation.frame]))[0]
            )
        if world_centers:
            target_center = np.median(np.asarray(world_centers), axis=0)
            polygon_center = (
                world_polygon.min(axis=0) + world_polygon.max(axis=0)
            ) / 2.0
            world_polygon += target_center - polygon_center
        else:
            target_center = (
                world_polygon.min(axis=0) + world_polygon.max(axis=0)
            ) / 2.0

        residuals = []
        for observation in track.observations:
            if observation.frame >= len(poses) or segments[observation.frame] != track.segment_id:
                continue
            predicted = transform_points(
                np.array([target_center]), poses[observation.frame]
            )[0]
            actual = np.asarray(bbox_center(observation.bbox))
            residuals.append(
                (
                    observation.frame,
                    float(actual[0] - predicted[0]),
                    float(actual[1] - predicted[1]),
                )
            )

        # Farbe ueber alle stabilen Maskenkandidaten abstimmen, nicht aus
        # einer fruehen wanddominierten Trackbox einfrieren.
        color_scores: dict[str, float] = defaultdict(float)
        colors: dict[str, dict] = {}
        for item in stable:
            color = item.mask.color
            if color is None:
                continue
            name = str(color["name"])
            weight = max(0.05, item.adjusted_score) * (
                0.25 + float(color.get("confidence", 0.0))
            )
            color_scores[name] += weight
            colors[name] = color
        chosen_color = selected.mask.color
        if color_scores:
            chosen_color = colors[max(color_scores, key=color_scores.get)]
        if chosen_color is None:
            continue

        track.mask_polygon = polygon
        track.mask_frame = reference_frame
        track.mask_source = selected.mask.source
        track.mask_score = selected.adjusted_score
        track.world_polygon = world_polygon
        track.alignment_residuals = _smooth_residuals(residuals)
        track.color = chosen_color
        track.color_bgr = _hex_to_bgr(chosen_color["hex"])


def _smooth_residuals(
    residuals: list[tuple[int, float, float]], window: int = 5
) -> list[tuple[int, float, float]]:
    """Offline-Medianfilter: Kameradrift behalten, YOLO-Boxzitterer entfernen."""
    if len(residuals) < 3:
        return sorted(residuals)
    ordered = sorted(residuals)
    frames = [item[0] for item in ordered]
    values = np.asarray([item[1:] for item in ordered], dtype=np.float64)
    radius = max(1, window // 2)
    filtered = np.vstack(
        [
            np.median(
                values[max(0, index - radius):min(len(values), index + radius + 1)],
                axis=0,
            )
            for index in range(len(values))
        ]
    )
    # Kleine zweite Dreiecksglaettung vermeidet Stufen des Medianfilters.
    if len(filtered) >= 3:
        padded = np.pad(filtered, ((1, 1), (0, 0)), mode="edge")
        filtered = (
            0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]
        )
    return [
        (frame, float(value[0]), float(value[1]))
        for frame, value in zip(frames, filtered)
    ]


def _residual_at(track: Track, frame_index: int) -> np.ndarray:
    values = track.alignment_residuals
    if not values:
        return np.zeros(2, dtype=np.float64)
    if frame_index <= values[0][0]:
        return np.array(values[0][1:], dtype=np.float64)
    if frame_index >= values[-1][0]:
        return np.array(values[-1][1:], dtype=np.float64)
    for left, right in zip(values, values[1:]):
        if left[0] <= frame_index <= right[0]:
            span = max(1, right[0] - left[0])
            factor = (frame_index - left[0]) / span
            return (
                np.array(left[1:], dtype=np.float64) * (1.0 - factor)
                + np.array(right[1:], dtype=np.float64) * factor
            )
    return np.zeros(2, dtype=np.float64)


def _fade(track: Track, frame_index: int, fade_frames: int) -> float:
    end = track.ended_frame if track.ended_frame is not None else track.last_frame
    if frame_index < track.first_frame or frame_index > end:
        return 0.0
    if fade_frames <= 0:
        return 1.0
    fade_in = (frame_index - track.first_frame + 1) / fade_frames
    fade_out = (end - frame_index + 1) / fade_frames
    return float(np.clip(min(fade_in, fade_out, 1.0), 0.0, 1.0))


def _render_pass(
    source: str | Path,
    destination: str | Path,
    meta: VideoMeta,
    tracks: list[Track],
    poses: list[np.ndarray],
    segments: list[int],
    *,
    processed_frames: int,
    progress_cb,
) -> int:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        meta.fps,
        (meta.width, meta.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Videoausgabe konnte nicht geoeffnet werden: {destination}")
    capture = _open_capture(source)
    alpha = float(os.getenv("VIDEO_MASK_ALPHA", "0.30"))
    fade_frames = max(0, int(os.getenv("VIDEO_FADE_FRAMES", "5")))
    thickness = max(2, int(round(meta.width / 540.0)))
    visible_tracks = [
        track
        for track in tracks
        if track.world_polygon is not None or track.fallback_box_world is not None
    ]
    written = 0

    for frame_index in range(processed_frames):
        ok, raw = capture.read()
        if not ok:
            break
        frame = _oriented(raw, meta)
        for track in visible_tracks:
            if track.segment_id != segments[frame_index]:
                continue
            visibility = _fade(track, frame_index, fade_frames)
            if visibility <= 0.0:
                continue
            has_mask = track.world_polygon is not None
            shape = track.world_polygon if has_mask else track.fallback_box_world
            polygon = transform_points(shape, poses[frame_index])
            polygon += _residual_at(track, frame_index)
            if not np.isfinite(polygon).all():
                continue
            if (
                polygon[:, 0].max() < 0
                or polygon[:, 1].max() < 0
                or polygon[:, 0].min() >= meta.width
                or polygon[:, 1].min() >= meta.height
            ):
                continue
            points = polygon.astype(np.int32)
            if track.color_bgr is None:
                _measure_fallback_color(track, frame, polygon, meta)
                if track.color_bgr is None:
                    continue
            layer = frame.copy()
            if has_mask:
                cv2.fillPoly(layer, [points], track.color_bgr)
                cv2.polylines(
                    layer, [points], True, track.color_bgr, thickness, cv2.LINE_AA
                )
                blend = alpha * visibility
            else:
                # Nur Umriss: Detektion belegt, exakte Form nicht bestaetigt.
                cv2.polylines(
                    layer, [points], True, track.color_bgr, thickness, cv2.LINE_AA
                )
                blend = min(1.0, 3.0 * alpha) * visibility
            frame = cv2.addWeighted(layer, blend, frame, 1.0 - blend, 0.0)
        writer.write(frame)
        written += 1
        if frame_index % 10 == 0 or frame_index == processed_frames - 1:
            _progress(progress_cb, 0.75 + 0.25 * written / max(1, processed_frames))

    capture.release()
    writer.release()
    return written


def _measure_fallback_color(
    track: Track, frame: np.ndarray, polygon: np.ndarray, meta
) -> None:
    """Farbe fuer Fallback-Boxen einmalig aus den Boxpixeln messen."""
    from vision.color_extraction import extract_color

    x0 = int(max(0.0, float(polygon[:, 0].min())))
    y0 = int(max(0.0, float(polygon[:, 1].min())))
    x1 = int(min(float(meta.width), float(polygon[:, 0].max())))
    y1 = int(min(float(meta.height), float(polygon[:, 1].max())))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    color = extract_color(frame, bbox=[x0, y0, x1, y1])
    if not color:
        return
    track.color = color
    track.color_bgr = _hex_to_bgr(color["hex"])


def _debug_payload(
    source,
    destination,
    meta,
    tracks,
    debug_frames,
    mask_debug,
    poses,
    segments,
    *,
    confidence,
    stride,
    keyframe,
    elapsed,
):
    for frame_record in debug_frames:
        frame_index = frame_record["frame"]
        predictions = []
        for track in tracks:
            end = track.ended_frame if track.ended_frame is not None else track.last_frame
            shape = (
                track.world_polygon
                if track.world_polygon is not None
                else track.fallback_box_world
            )
            if not (
                shape is not None
                and track.segment_id == segments[frame_index]
                and track.first_frame <= frame_index <= end
            ):
                continue
            polygon = transform_points(shape, poses[frame_index])
            polygon += _residual_at(track, frame_index)
            predictions.append(
                {
                    "track_id": track.id,
                    "class_id": track.class_id,
                    "confidence": round(track.best_confidence, 4),
                    "bbox": [
                        round(float(polygon[:, 0].min()), 2),
                        round(float(polygon[:, 1].min()), 2),
                        round(float(polygon[:, 0].max()), 2),
                        round(float(polygon[:, 1].max()), 2),
                    ],
                    "polygon": polygon.round(2).tolist(),
                    "mask_source": (
                        track.mask_source
                        if track.world_polygon is not None
                        else "bbox_fallback"
                    ),
                    "mask_age": (
                        abs(frame_index - int(track.mask_frame))
                        if track.mask_frame is not None
                        else None
                    ),
                    "mask_score": round(track.mask_score, 4),
                    "color": track.color,
                }
            )
        frame_record["predictions"] = predictions

    return {
        "version": 2,
        "pipeline": "two_pass_best_frame",
        "source": str(source),
        "destination": str(destination),
        "video": {
            "fps": meta.fps,
            "width": meta.width,
            "height": meta.height,
            "frames": len(debug_frames),
        },
        "config": {
            "confidence": confidence,
            "stride": stride,
            "keyframe": keyframe,
            "track_patience_frames": int(os.getenv("VIDEO_TRACK_PATIENCE", "30")),
            "mask_candidates": int(os.getenv("VIDEO_MASK_CANDIDATES", "3")),
            "temporal_iou": float(os.getenv("VIDEO_TEMPORAL_IOU", "0.12")),
        },
        "summary": {
            "tracks_confirmed": len(tracks),
            "tracks_with_mask": sum(track.world_polygon is not None for track in tracks),
            "mask_inference_frames": len(
                {item["frame"] for item in mask_debug}
            ),
            "mask_candidates_accepted": sum(
                bool(item["accepted"]) for item in mask_debug
            ),
            "mask_candidates_rejected": sum(
                not bool(item["accepted"]) for item in mask_debug
            ),
            "elapsed_seconds": round(float(elapsed), 3),
        },
        "tracks": [
            {
                "id": track.id,
                "segment": track.segment_id,
                "class_id": track.class_id,
                "class_votes": dict(track.class_scores),
                "hits": track.hits,
                "first_frame": track.first_frame,
                "last_detection": track.last_frame,
                "ended_frame": track.ended_frame,
                "best_confidence": round(track.best_confidence, 4),
                "mask_source": track.mask_source,
                "mask_frame": track.mask_frame,
                "mask_score": round(track.mask_score, 4),
                "color": track.color,
                "observations": [
                    {
                        "frame": observation.frame,
                        "bbox": [round(value, 2) for value in observation.bbox],
                        "class_id": observation.class_id,
                        "confidence": round(observation.confidence, 4),
                        "sharpness": round(observation.sharpness, 2),
                    }
                    for observation in track.observations
                ],
            }
            for track in tracks
        ],
        "frames": debug_frames,
        "mask_candidates": mask_debug,
    }


def render_video(
    src,
    dst,
    weights=None,
    conf=0.4,
    stride=2,
    shapes=True,
    keyframe=20,
    max_seconds=90.0,
    progress_cb=None,
    debug_path=None,
    device=None,
):
    """Rendert ``src`` mit stabilen Konturen nach ``dst``.

    Die Rueckgabe bleibt kompatibel zur bisherigen API: ``(frames, sekunden)``.
    ``keyframe`` steuert in der Zwei-Pass-Pipeline den zeitlichen Abstand der
    Maskenkandidaten; es ist damit kein wirkungsloser Parameter mehr.
    """
    from ultralytics import YOLO

    stride = max(1, int(stride))
    keyframe = max(stride, int(keyframe))
    started = time.time()
    meta = _video_meta(src, max_seconds)
    model = YOLO(weights or str(API_DIR / "weights" / "best.pt"))

    sam = None
    if shapes:
        import torch
        from ultralytics import SAM

        default_weights = "sam2.1_b.pt" if torch.cuda.is_available() else "sam2.1_t.pt"
        sam = SAM(os.getenv("SAM_WEIGHTS", default_weights))

    tracks, poses, segments, debug_frames, processed = _analyse_video(
        src,
        meta,
        model,
        confidence=float(conf),
        stride=stride,
        device=device,
        progress_cb=progress_cb,
    )
    mask_debug: list[dict] = []
    if shapes and tracks:
        candidates, mask_debug = _segment_candidates(
            src,
            meta,
            tracks,
            sam,
            keyframe=keyframe,
            device=device,
            progress_cb=progress_cb,
        )
        _select_track_masks(tracks, candidates, poses, segments)
    else:
        _progress(progress_cb, 0.75)

    written = _render_pass(
        src,
        dst,
        meta,
        tracks,
        poses,
        segments,
        processed_frames=processed,
        progress_cb=progress_cb,
    )
    elapsed = time.time() - started

    if debug_path is None and os.getenv("VIDEO_DEBUG", "0") == "1":
        debug_path = str(Path(dst).with_suffix(".debug.json"))
    if debug_path:
        target = Path(debug_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                _debug_payload(
                    src,
                    dst,
                    meta,
                    tracks,
                    debug_frames,
                    mask_debug,
                    poses,
                    segments,
                    confidence=float(conf),
                    stride=stride,
                    keyframe=keyframe,
                    elapsed=elapsed,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return written, elapsed
