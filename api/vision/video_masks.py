"""Maskenaufbereitung und Qualitaets-Gates fuer die Videopipeline.

Eine falsche Maske ist im Video auffaelliger als eine kurz fehlende Maske.
Darum liefert dieses Modul neben einer Kontur immer messbare Qualitaetswerte
und lehnt rechteckige Wandflaechen, Randsegel und winzige Komponenten hart ab.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from vision.video_tracking import bbox_area, transform_points


@dataclass
class MaskCandidate:
    polygon: np.ndarray | None
    source: str
    score: float
    accepted: bool
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    color: dict | None = None

    def as_debug(self) -> dict:
        return {
            "source": self.source,
            "score": round(float(self.score), 4),
            "accepted": bool(self.accepted),
            "metrics": self.metrics,
        }


def _clip_bbox(
    bbox: Iterable[float], width: int, height: int, padding: float = 0.0
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    pad_x = (x1 - x0) * padding
    pad_y = (y1 - y0) * padding
    return (
        max(0, int(np.floor(x0 - pad_x))),
        max(0, int(np.floor(y0 - pad_y))),
        min(width, int(np.ceil(x1 + pad_x))),
        min(height, int(np.ceil(y1 + pad_y))),
    )


def _scaled_kernel(bbox: Iterable[float]) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    extent = max(1.0, min(x1 - x0, y1 - y0))
    size = int(np.clip(round(extent * 0.018), 1, 7))
    size = size if size % 2 == 1 else size + 1
    return np.ones((size, size), np.uint8)


def _clean_mask(mask: np.ndarray, bbox: Iterable[float]) -> np.ndarray:
    binary = (np.asarray(mask).squeeze() > 0).astype(np.uint8) * 255
    kernel = _scaled_kernel(bbox)
    if kernel.shape[0] > 1:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def _best_component(mask: np.ndarray, bbox: Iterable[float]) -> np.ndarray:
    """Waehlt die Komponente, die am plausibelsten zur Detektionsbox gehoert."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(mask)
    x0, y0, x1, y1 = [float(v) for v in bbox]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    diagonal = max(1.0, float(np.hypot(x1 - x0, y1 - y0)))
    best_label = None
    best_score = -np.inf
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 6:
            continue
        lx = float(stats[label, cv2.CC_STAT_LEFT])
        ly = float(stats[label, cv2.CC_STAT_TOP])
        lw = float(stats[label, cv2.CC_STAT_WIDTH])
        lh = float(stats[label, cv2.CC_STAT_HEIGHT])
        ix = max(0.0, min(x1, lx + lw) - max(x0, lx))
        iy = max(0.0, min(y1, ly + lh) - max(y0, ly))
        overlap = ix * iy / max(area, 1.0)
        distance = float(np.hypot(centroids[label][0] - cx, centroids[label][1] - cy))
        score = np.log1p(area) + 2.0 * overlap - distance / diagonal
        if score > best_score:
            best_label = label
            best_score = score
    if best_label is None:
        return np.zeros_like(mask)
    return (labels == best_label).astype(np.uint8) * 255


def _wall_reference(frame: np.ndarray, bbox: Iterable[float]) -> np.ndarray | None:
    """Medianfarbe eines Rings ausserhalb der Detektionsbox."""
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = _clip_bbox(bbox, width, height)
    ex0, ey0, ex1, ey1 = _clip_bbox(bbox, width, height, padding=0.14)
    if ex1 <= ex0 or ey1 <= ey0:
        return None
    ring = np.ones((ey1 - ey0, ex1 - ex0), dtype=bool)
    ring[max(0, y0 - ey0):max(0, y1 - ey0),
         max(0, x0 - ex0):max(0, x1 - ex0)] = False
    pixels = frame[ey0:ey1, ex0:ex1][ring]
    if len(pixels) < 12:
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        pixels = np.concatenate(
            [crop[0], crop[-1], crop[:, 0], crop[:, -1]], axis=0
        )
    return np.median(pixels.astype(np.float32), axis=0)


def _mask_metrics(
    frame: np.ndarray, mask: np.ndarray, bbox: Iterable[float]
) -> dict[str, float | int]:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = _clip_bbox(bbox, width, height)
    px0, py0, px1, py1 = _clip_bbox(bbox, width, height, padding=0.10)
    area = int((mask > 0).sum())
    if area == 0:
        return {
            "area": 0,
            "area_box_ratio": 0.0,
            "containment": 0.0,
            "border_sides": 0,
            "border_fraction": 0.0,
            "wall_fraction": 1.0,
        }

    inside = int((mask[py0:py1, px0:px1] > 0).sum())
    containment = inside / area
    ratio = area / max(1.0, bbox_area(bbox))

    border_sides = 0
    border_fraction = 0.0
    margin = max(1, int(round(min(max(1, x1 - x0), max(1, y1 - y0)) * 0.025)))
    crop = mask[y0:y1, x0:x1] > 0
    if crop.size:
        border_sides += int(crop[:margin].any())
        border_sides += int(crop[-margin:].any())
        border_sides += int(crop[:, :margin].any())
        border_sides += int(crop[:, -margin:].any())
        border_band = np.zeros_like(crop)
        border_band[:margin] = True
        border_band[-margin:] = True
        border_band[:, :margin] = True
        border_band[:, -margin:] = True
        border_fraction = float(np.logical_and(crop, border_band).sum()) / max(
            1, int(crop.sum())
        )

    wall = _wall_reference(frame, bbox)
    wall_fraction = 0.0
    if wall is not None:
        pixels = frame[mask > 0].astype(np.float32)
        if len(pixels) > 5000:
            pixels = pixels[:: max(1, len(pixels) // 5000)]
        # BGR-Abstand. Bewusst tolerant: Schatten derselben Wandfarbe soll
        # nicht sofort als Objekt gelten.
        wall_fraction = float((np.linalg.norm(pixels - wall, axis=1) < 52.0).mean())

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    compactness = 0.0
    if perimeter > 0:
        compactness = float(4.0 * np.pi * area / (perimeter * perimeter))
    return {
        "area": area,
        "area_box_ratio": round(float(ratio), 4),
        "containment": round(float(containment), 4),
        "border_sides": int(border_sides),
        "border_fraction": round(float(border_fraction), 4),
        "wall_fraction": round(float(wall_fraction), 4),
        "compactness": round(float(compactness), 4),
    }


def candidate_from_mask(
    frame: np.ndarray,
    bbox: Iterable[float],
    mask: np.ndarray,
    *,
    class_id: int,
    source: str,
    confidence: float,
    sharpness: float,
) -> MaskCandidate:
    """Bereinigt, bewertet und vereinfacht eine binare Vollbildmaske."""
    if np.asarray(mask).squeeze().shape != frame.shape[:2]:
        return MaskCandidate(
            None, source, -10.0, False, {"reason": "wrong_shape"}
        )
    cleaned = _clean_mask(mask, bbox)
    cleaned = _best_component(cleaned, bbox)
    metrics = _mask_metrics(frame, cleaned, bbox)
    ratio = float(metrics["area_box_ratio"])
    containment = float(metrics["containment"])
    wall_fraction = float(metrics["wall_fraction"])
    border_sides = int(metrics["border_sides"])
    border_fraction = float(metrics["border_fraction"])

    reason = None
    if int(metrics["area"]) < 9:
        reason = "too_small"
    elif ratio < 0.035:
        reason = "too_small_for_box"
    elif ratio > (0.92 if class_id == 1 else 1.08):
        reason = "too_large_for_box"
    elif containment < 0.90:
        reason = "outside_prompt"
    elif border_sides >= 3 and border_fraction > 0.08 and ratio > 0.42:
        reason = "box_shaped"
    elif wall_fraction > (
        0.36 if class_id == 1 and source == "sam" else 0.52
    ):
        reason = "wall_dominated"

    sharp_score = min(1.0, np.log1p(max(0.0, sharpness)) / np.log(501.0))
    target_ratio = 0.48 if class_id == 1 else 0.42
    ratio_score = max(0.0, 1.0 - abs(ratio - target_ratio) / 0.55)
    score = (
        1.30 * float(confidence)
        + 0.45 * sharp_score
        + 0.55 * ratio_score
        + 0.40 * containment
        + 0.40 * (1.0 - wall_fraction)
        - 0.10 * border_sides
    )
    if reason is not None:
        metrics["reason"] = reason
        return MaskCandidate(None, source, score - 4.0, False, metrics)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        metrics["reason"] = "no_contour"
        return MaskCandidate(None, source, score - 4.0, False, metrics)
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(0.75, 0.004 * cv2.arcLength(contour, True))
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(polygon) < 3:
        metrics["reason"] = "degenerate_contour"
        return MaskCandidate(None, source, score - 4.0, False, metrics)
    return MaskCandidate(
        polygon.astype(np.float64), source, score, True, metrics
    )


def _expanded_crop(frame: np.ndarray, bbox: Iterable[float], padding: float = 0.14):
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = _clip_bbox(bbox, width, height, padding=padding)
    return frame[y0:y1, x0:x1], (x0, y0, x1, y1)


def volume_mask_candidates(
    frame: np.ndarray,
    bbox: Iterable[float],
    *,
    confidence: float,
    sharpness: float,
) -> list[MaskCandidate]:
    """Erzeugt robuste Hintergrund-Subtraktionskandidaten fuer Volumes.

    Anders als die alte Heuristik wird die Wand aus einem *erweiterten* Ring
    ausserhalb der Detektionsbox bestimmt. So gilt ein Objekt, das die enge
    YOLO-Box beruehrt, nicht automatisch als Hintergrund und ein Wandcluster
    kann nicht als grosses Rechteck konserviert werden.
    """
    crop, (x0, y0, x1, y1) = _expanded_crop(frame, bbox)
    if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
        return []

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    samples = lab.reshape(-1, 3).astype(np.float32)
    if len(samples) < 40:
        return []
    sample_for_fit = samples
    if len(samples) > 12000:
        indices = np.random.default_rng(0).choice(len(samples), 12000, replace=False)
        sample_for_fit = samples[indices]
    clusters = min(4, max(2, len(sample_for_fit) // 200))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.8)
    cv2.setRNGSeed(0)
    _, _, centers = cv2.kmeans(
        sample_for_fit,
        clusters,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    distances = np.linalg.norm(
        samples[:, None, :] - centers[None, :, :], axis=2
    )
    labels = distances.argmin(axis=1).reshape(crop.shape[:2])

    bx0, by0, bx1, by1 = _clip_bbox(
        bbox, frame.shape[1], frame.shape[0]
    )
    local_box = (
        max(0, bx0 - x0), max(0, by0 - y0),
        min(crop.shape[1], bx1 - x0), min(crop.shape[0], by1 - y0),
    )
    ring = np.ones(crop.shape[:2], dtype=bool)
    lx0, ly0, lx1, ly1 = local_box
    ring[ly0:ly1, lx0:lx1] = False
    if int(ring.sum()) < 20:
        thickness = max(2, int(round(min(crop.shape[:2]) * 0.08)))
        ring[:] = False
        ring[:thickness] = True
        ring[-thickness:] = True
        ring[:, :thickness] = True
        ring[:, -thickness:] = True

    counts = np.array([(labels[ring] == idx).mean() for idx in range(clusters)])
    wall_labels: set[int] = set()
    covered = 0.0
    for idx in np.argsort(counts)[::-1]:
        if counts[idx] < 0.08 and wall_labels:
            break
        wall_labels.add(int(idx))
        covered += float(counts[idx])
        if covered >= 0.78:
            break

    raw_masks: list[tuple[str, np.ndarray]] = []
    non_wall = [idx for idx in range(clusters) if idx not in wall_labels]
    if non_wall:
        raw_masks.append(("volume_kmeans_union", np.isin(labels, non_wall)))
        for idx in non_wall:
            raw_masks.append((f"volume_kmeans_{idx}", labels == idx))

    wall_pixels = lab[ring].astype(np.float32)
    if len(wall_pixels):
        wall_center = np.median(wall_pixels, axis=0)
        ring_distances = np.linalg.norm(wall_pixels - wall_center, axis=1)
        threshold = max(12.0, float(np.percentile(ring_distances, 95)) + 5.0)
        delta = np.linalg.norm(lab.astype(np.float32) - wall_center, axis=2)
        raw_masks.append(("volume_wall_delta", delta > threshold))

    results: list[MaskCandidate] = []
    seen: set[bytes] = set()
    for source, local_mask in raw_masks:
        full = np.zeros(frame.shape[:2], dtype=np.uint8)
        full[y0:y1, x0:x1] = local_mask.astype(np.uint8) * 255
        # Kandidat auf die Detektionsbox plus kleinen Rand begrenzen.
        px0, py0, px1, py1 = _clip_bbox(
            bbox, frame.shape[1], frame.shape[0], padding=0.06
        )
        clipped = np.zeros_like(full)
        clipped[py0:py1, px0:px1] = full[py0:py1, px0:px1]
        digest = cv2.resize(clipped, (32, 32), interpolation=cv2.INTER_NEAREST).tobytes()
        if digest in seen:
            continue
        seen.add(digest)
        results.append(
            candidate_from_mask(
                frame,
                bbox,
                clipped,
                class_id=1,
                source=source,
                confidence=confidence,
                sharpness=sharpness,
            )
        )
    return results


def polygon_iou(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_to_second: np.ndarray | None = None,
) -> float:
    """IoU zweier Konturen; optional erste Kontur ins zweite Frame bewegen."""
    first = np.asarray(first, dtype=np.float64).reshape(-1, 2)
    second = np.asarray(second, dtype=np.float64).reshape(-1, 2)
    if first_to_second is not None:
        first = transform_points(first, first_to_second)
    all_points = np.vstack([first, second])
    x0 = int(np.floor(all_points[:, 0].min())) - 2
    y0 = int(np.floor(all_points[:, 1].min())) - 2
    x1 = int(np.ceil(all_points[:, 0].max())) + 3
    y1 = int(np.ceil(all_points[:, 1].max())) + 3
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0 or width * height > 20_000_000:
        return 0.0
    mask0 = np.zeros((height, width), dtype=np.uint8)
    mask1 = np.zeros_like(mask0)
    offset = np.array([x0, y0])
    cv2.fillPoly(mask0, [(first - offset).astype(np.int32)], 1)
    cv2.fillPoly(mask1, [(second - offset).astype(np.int32)], 1)
    intersection = int(np.logical_and(mask0, mask1).sum())
    union = int(np.logical_or(mask0, mask1).sum())
    return intersection / union if union else 0.0
