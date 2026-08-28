"""Zeitlich stabiles Tracking fuer statische Objekte an einer Kletterwand.

Die Kamera bewegt sich, die Griffe nicht. Deshalb wird zuerst die globale
Kamerabewegung zwischen zwei Frames geschaetzt und auf alle aktiven Tracks
angewendet. Erst danach werden neue Detektionen zugeordnet. Das Modul kennt
keine YOLO- oder SAM-Objekte und bleibt dadurch mit synthetischen Daten
testbar.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np


IDENTITY_AFFINE = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def bbox_area(bbox: Iterable[float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, float(x1) - float(x0)) * max(0.0, float(y1) - float(y0))


def bbox_iou(a: Iterable[float], b: Iterable[float]) -> float:
    a = list(a)
    b = list(b)
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def bbox_center(bbox: Iterable[float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (float(x0 + x1) / 2.0, float(y0 + y1) / 2.0)


def affine_to_homogeneous(matrix: np.ndarray) -> np.ndarray:
    result = np.eye(3, dtype=np.float64)
    result[:2] = np.asarray(matrix, dtype=np.float64)
    return result


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Transformiert Nx2-Punkte mit einer 2x3- oder 3x3-Matrix."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (2, 3):
        return points @ matrix[:, :2].T + matrix[:, 2]
    if matrix.shape == (3, 3):
        hom = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
        moved = hom @ matrix.T
        return moved[:, :2] / np.maximum(moved[:, 2:3], 1e-12)
    raise ValueError(f"Affine Matrix muss 2x3 oder 3x3 sein, nicht {matrix.shape}")


def transform_bbox(bbox: Iterable[float], matrix: np.ndarray) -> list[float]:
    """Transformiert alle vier Ecken einer Box.

    Nur zwei diagonale Ecken zu transformieren ist bei Rotation falsch und
    war die Ursache fuer schrumpfende oder abgeschnittene Track-Boxen.
    """
    x0, y0, x1, y1 = [float(v) for v in bbox]
    corners = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64
    )
    moved = transform_points(corners, matrix)
    return [
        float(moved[:, 0].min()),
        float(moved[:, 1].min()),
        float(moved[:, 0].max()),
        float(moved[:, 1].max()),
    ]


def relative_pose(source_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    """Matrix, die Koordinaten aus source nach target ueberfuehrt."""
    return np.asarray(target_pose) @ np.linalg.inv(np.asarray(source_pose))


@dataclass
class MotionEstimate:
    matrix: np.ndarray = field(default_factory=lambda: IDENTITY_AFFINE.copy())
    method: str = "identity"
    quality: float = 0.0
    inliers: int = 0
    points: int = 0
    scene_cut: bool = False

    def as_debug(self) -> dict:
        return {
            "method": self.method,
            "quality": round(float(self.quality), 4),
            "inliers": int(self.inliers),
            "points": int(self.points),
            "scene_cut": bool(self.scene_cut),
            "matrix": np.asarray(self.matrix).round(6).tolist(),
        }


def _histogram_similarity(previous: np.ndarray, current: np.ndarray) -> float:
    h0 = cv2.calcHist([previous], [0], None, [32], [0, 256])
    h1 = cv2.calcHist([current], [0], None, [32], [0, 256])
    cv2.normalize(h0, h0)
    cv2.normalize(h1, h1)
    return float(cv2.compareHist(h0, h1, cv2.HISTCMP_CORREL))


def estimate_camera_motion(
    previous: np.ndarray,
    current: np.ndarray,
    full_width: int,
    *,
    min_points: int = 12,
) -> MotionEstimate:
    """Robuste affine Bewegung mit Forward/Backward-Check und Fallback.

    ``previous`` und ``current`` sind verkleinerte Graubilder. Translationen
    werden auf ``full_width`` skaliert, Rotation und Zoom bleiben erhalten.
    Bei einem echten Szenenschnitt wird identity geliefert und der Aufrufer
    muss die aktiven Tracks beenden.
    """
    if previous.shape != current.shape:
        raise ValueError("Bewegungsframes muessen dieselbe Groesse haben")

    hist_similarity = _histogram_similarity(previous, current)
    points0 = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=350,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    if points0 is not None and len(points0) >= min_points:
        points1, status1, _ = cv2.calcOpticalFlowPyrLK(
            previous, current, points0, None
        )
        if points1 is not None and status1 is not None:
            points0_back, status0, _ = cv2.calcOpticalFlowPyrLK(
                current, previous, points1, None
            )
            if points0_back is not None and status0 is not None:
                forward_ok = status1.reshape(-1) == 1
                backward_ok = status0.reshape(-1) == 1
                fb_error = np.linalg.norm(
                    points0.reshape(-1, 2) - points0_back.reshape(-1, 2), axis=1
                )
                good = forward_ok & backward_ok & (fb_error < 1.5)
                if int(good.sum()) >= min_points:
                    src = points0.reshape(-1, 2)[good]
                    dst = points1.reshape(-1, 2)[good]
                    matrix, inlier_mask = cv2.estimateAffinePartial2D(
                        src,
                        dst,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=2.5,
                        maxIters=2000,
                        confidence=0.99,
                        refineIters=10,
                    )
                    if matrix is not None and inlier_mask is not None:
                        inliers = int(inlier_mask.sum())
                        ratio = inliers / max(1, len(src))
                        scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
                        angle = abs(float(np.degrees(np.arctan2(
                            matrix[1, 0], matrix[0, 0]
                        ))))
                        # Ein einzelner Handy-Frame kann sich bewegen, aber
                        # 20 % Zoom oder 20 Grad Rotation zwischen Frames sind
                        # praktisch immer ein Fehlfit oder Schnitt.
                        plausible = 0.82 <= scale <= 1.22 and angle <= 20.0
                        result = np.asarray(matrix, dtype=np.float64)
                        result[:, 2] *= full_width / previous.shape[1]
                        full_height = previous.shape[0] * full_width / previous.shape[1]
                        translation = float(np.hypot(*result[:, 2]))
                        plausible = plausible and translation <= 0.35 * float(
                            np.hypot(full_width, full_height)
                        )
                        if plausible and ratio >= 0.35:
                            return MotionEstimate(
                                matrix=result,
                                method="affine",
                                quality=min(1.0, ratio),
                                inliers=inliers,
                                points=len(src),
                            )

    # Phase correlation ist als Translationsfallback stabil, liefert aber
    # keine Rotation. Der response-Wert verhindert grosse Zufallsspruenge.
    (dx, dy), response = cv2.phaseCorrelate(
        previous.astype(np.float32), current.astype(np.float32)
    )
    response = float(response)
    scale = full_width / previous.shape[1]
    full_height = previous.shape[0] * scale
    translation = float(np.hypot(dx * scale, dy * scale))
    if response >= 0.08 and translation <= 0.35 * float(
        np.hypot(full_width, full_height)
    ):
        return MotionEstimate(
            matrix=np.array(
                [[1.0, 0.0, dx * scale], [0.0, 1.0, dy * scale]],
                dtype=np.float64,
            ),
            method="translation",
            quality=min(0.6, max(0.0, response)),
        )

    scene_cut = hist_similarity < 0.15 and (
        response < 0.03
        or translation > 0.35 * float(np.hypot(full_width, full_height))
    )
    return MotionEstimate(
        matrix=IDENTITY_AFFINE.copy(),
        method="scene_cut" if scene_cut else "identity",
        quality=0.0,
        scene_cut=scene_cut,
    )


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    class_id: int
    confidence: float


@dataclass(frozen=True)
class Observation:
    frame: int
    bbox: tuple[float, float, float, float]
    class_id: int
    confidence: float
    sharpness: float

    @property
    def quality(self) -> float:
        # Laplace-Varianz ist nicht linear; log1p verhindert, dass ein sehr
        # kontrastreicher Frame die Detektionskonfidenz vollstaendig ueberrollt.
        sharp = min(1.0, np.log1p(max(0.0, self.sharpness)) / np.log(501.0))
        return 0.70 * float(self.confidence) + 0.30 * float(sharp)


@dataclass
class Track:
    id: int
    bbox: list[float]
    segment_id: int
    first_frame: int
    last_frame: int
    hits: int = 0
    missed: int = 0
    ended_frame: int | None = None
    observations: list[Observation] = field(default_factory=list)
    class_scores: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    class_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    mask_polygon: np.ndarray | None = None
    mask_frame: int | None = None
    mask_source: str | None = None
    mask_score: float = 0.0
    world_polygon: np.ndarray | None = None
    fallback_box_world: np.ndarray | None = None
    alignment_residuals: list[tuple[int, float, float]] = field(default_factory=list)
    color_bgr: tuple[int, int, int] | None = None
    color: dict | None = None

    @property
    def class_id(self) -> int:
        if not self.class_scores:
            return 0
        # Konfidenzsumme entscheidet; Count bricht sehr knappe Gleichstaende.
        return max(
            self.class_scores,
            key=lambda cls: (self.class_scores[cls], self.class_counts[cls]),
        )

    @property
    def confirmed(self) -> bool:
        return self.hits >= 2

    @property
    def best_confidence(self) -> float:
        return max((o.confidence for o in self.observations), default=0.0)

    def observe(self, observation: Observation, *, ema_alpha: float = 0.72) -> None:
        previous = list(self.bbox)
        incoming = list(observation.bbox)
        overlap = bbox_iou(previous, incoming)
        # Bei deutlicher Abweichung ist die frische Detektion verlaesslicher
        # als eine bereits weggedriftete Trackbox. Das trifft insbesondere
        # den dokumentierten Halbmond-Fehler.
        alpha = 1.0 if overlap < 0.25 else ema_alpha
        self.bbox = [
            previous[i] * (1.0 - alpha) + incoming[i] * alpha for i in range(4)
        ]
        self.observations.append(observation)
        self.class_scores[observation.class_id] += max(0.05, observation.confidence)
        self.class_counts[observation.class_id] += 1
        self.hits += 1
        self.missed = 0
        self.last_frame = observation.frame

    def move(self, matrix: np.ndarray) -> None:
        self.bbox = transform_bbox(self.bbox, matrix)

    def candidate_observations(
        self, count: int = 2, min_frame_distance: int = 6
    ) -> list[Observation]:
        selected: list[Observation] = []
        for observation in sorted(
            self.observations, key=lambda item: item.quality, reverse=True
        ):
            if all(
                abs(observation.frame - other.frame) >= min_frame_distance
                for other in selected
            ):
                selected.append(observation)
            if len(selected) >= count:
                break
        return selected or self.observations[:1]


def crop_sharpness(frame: np.ndarray, bbox: Iterable[float]) -> float:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def deduplicate_detections(
    detections: Iterable[Detection], iou_threshold: float = 0.60
) -> list[Detection]:
    """Unterdrueckt echte Doppelboxen, ohne Nachbargriffe zu verschmelzen."""
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: -item.confidence):
        duplicate = False
        for existing in kept:
            overlap = bbox_iou(detection.bbox, existing.bbox)
            if overlap >= iou_threshold:
                # Bei Klassenwiderspruch entscheidet nur eine sehr starke
                # Ueberdeckung. Damit entstehen nicht zwei Tracks fuer genau
                # denselben Gegenstand, benachbarte Hold/Volume-Objekte bleiben.
                same_class = detection.class_id == existing.class_id
                if same_class or overlap >= 0.82:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(detection)
    return kept


class TrackManager:
    """Klassen- und qualitaetsbewusste Zuordnung von Detektionen."""

    def __init__(
        self,
        *,
        confirmations: int = 2,
        patience: int = 30,
        match_iou: float = 0.28,
        merge_iou: float = 0.72,
    ) -> None:
        self.confirmations = confirmations
        self.patience = patience
        self.match_iou = match_iou
        self.merge_iou = merge_iou
        self.active: list[Track] = []
        self.finished: list[Track] = []
        self.next_id = 0
        self.segment_id = 0

    def move(self, motion: MotionEstimate, frame: int) -> list[dict]:
        events: list[dict] = []
        if motion.scene_cut:
            for track in self.active:
                track.ended_frame = max(track.last_frame, frame - 1)
                self.finished.append(track)
                events.append({"event": "scene_end", "track_id": track.id})
            self.active = []
            self.segment_id += 1
            return events
        if motion.quality >= 0.08:
            for track in self.active:
                track.move(motion.matrix)
        return events

    @staticmethod
    def _compatible(track: Track, detection: Detection, overlap: float) -> bool:
        if not track.class_scores or detection.class_id == track.class_id:
            return True
        # Eine einzelne fruehe Fehlklasse darf korrigiert werden, ein stabiler
        # Track aber nur bei nahezu identischer Box ueberstimmt werden.
        return track.hits < 3 or overlap >= 0.72

    def update(
        self, frame_index: int, frame: np.ndarray, detections: Iterable[Detection]
    ) -> list[dict]:
        detections = deduplicate_detections(detections)
        events: list[dict] = []

        # Global greedy ueber alle Paare statt detektionsreihenfolgenabhaengig.
        candidates: list[tuple[float, float, int, int]] = []
        for ti, track in enumerate(self.active):
            for di, detection in enumerate(detections):
                overlap = bbox_iou(track.bbox, detection.bbox)
                if overlap < self.match_iou:
                    continue
                if not self._compatible(track, detection, overlap):
                    continue
                class_bonus = 0.12 if detection.class_id == track.class_id else -0.10
                candidates.append(
                    (overlap + class_bonus, detection.confidence, ti, di)
                )
        candidates.sort(reverse=True)

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, _, ti, di in candidates:
            if ti in used_tracks or di in used_detections:
                continue
            track = self.active[ti]
            detection = detections[di]
            observation = Observation(
                frame=frame_index,
                bbox=detection.bbox,
                class_id=detection.class_id,
                confidence=detection.confidence,
                sharpness=crop_sharpness(frame, detection.bbox),
            )
            previous_class = track.class_id
            track.observe(observation)
            used_tracks.add(ti)
            used_detections.add(di)
            events.append(
                {
                    "event": "match",
                    "track_id": track.id,
                    "iou": round(bbox_iou(track.bbox, detection.bbox), 4),
                    "class_changed": previous_class != track.class_id,
                }
            )

        for ti, track in enumerate(self.active):
            if ti not in used_tracks:
                track.missed += 1

        for di, detection in enumerate(detections):
            if di in used_detections:
                continue
            track = Track(
                id=self.next_id,
                bbox=list(detection.bbox),
                segment_id=self.segment_id,
                first_frame=frame_index,
                last_frame=frame_index,
            )
            self.next_id += 1
            track.observe(
                Observation(
                    frame=frame_index,
                    bbox=detection.bbox,
                    class_id=detection.class_id,
                    confidence=detection.confidence,
                    sharpness=crop_sharpness(frame, detection.bbox),
                )
            )
            self.active.append(track)
            events.append({"event": "birth", "track_id": track.id})

        survivors: list[Track] = []
        for track in self.active:
            early_false_positive = track.hits < self.confirmations and track.missed > 2
            expired = track.missed > self.patience
            if early_false_positive or expired:
                track.ended_frame = max(track.last_frame, frame_index - 1)
                self.finished.append(track)
                events.append(
                    {
                        "event": "death",
                        "track_id": track.id,
                        "reason": "unconfirmed" if early_false_positive else "expired",
                    }
                )
            else:
                survivors.append(track)
        self.active = self._merge_tracks(survivors, events)
        return events

    def _merge_tracks(self, tracks: list[Track], events: list[dict]) -> list[Track]:
        ordered = sorted(
            tracks,
            key=lambda item: (item.hits, item.best_confidence, -item.id),
            reverse=True,
        )
        kept: list[Track] = []
        for track in ordered:
            duplicate = next(
                (
                    existing
                    for existing in kept
                    if existing.class_id == track.class_id
                    and bbox_iou(existing.bbox, track.bbox) >= self.merge_iou
                ),
                None,
            )
            if duplicate is None:
                kept.append(track)
                continue
            duplicate.observations.extend(track.observations)
            duplicate.observations.sort(key=lambda item: item.frame)
            for cls, score in track.class_scores.items():
                duplicate.class_scores[cls] += score
            for cls, count in track.class_counts.items():
                duplicate.class_counts[cls] += count
            duplicate.hits += track.hits
            duplicate.first_frame = min(duplicate.first_frame, track.first_frame)
            duplicate.last_frame = max(duplicate.last_frame, track.last_frame)
            events.append(
                {"event": "merge", "track_id": duplicate.id, "merged_id": track.id}
            )
        return kept

    def finish(self, final_frame: int) -> list[Track]:
        for track in self.active:
            track.ended_frame = final_frame
            self.finished.append(track)
        self.active = []
        return sorted(self.finished, key=lambda track: track.id)
