"""Dominante Grifffarbe aus den Bildpixeln bestimmen.

Ablauf je Detektion:
1. Pixel einsammeln — bevorzugt innerhalb der Segmentierungsmaske, sonst aus
   dem inneren Bereich der Bounding Box (Ränder enthalten meist Wand).
2. k-means im Lab-Raum, größtes Cluster = Grifffarbe. Schrauben, Schatten
   und Wandreste landen in den kleineren Clustern.
3. Cluster-Zentrum auf die Palette mappen: bunte Farben über den nächsten
   Lab-Anker, unbunte (Chroma unter Schwelle) über die Helligkeit.
"""

import cv2
import numpy as np

from vision.classes import ACHROMATIC, COLOR_INDEX, COLOR_PALETTE

# Unterhalb dieser Buntheit (Abstand vom Neutralpunkt in a/b) gilt eine Farbe
# als schwarz/grau/weiß. Holz liegt bei ~20-30 und bleibt damit "bunt".
CHROMA_THRESHOLD = 14.0
# L-Abstand zählt schwächer als der Farbstich: Beleuchtung ändert vor allem L.
LIGHTNESS_WEIGHT = 0.5
MIN_PIXELS = 20

_anchors_cache = None


def _hex_to_bgr(hex_str):
    h = hex_str.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def anchors():
    """Lab-Anker (OpenCV-8-Bit-Skalierung) je Palettenfarbe, aus hex berechnet."""
    global _anchors_cache
    if _anchors_cache is None:
        px = np.array([[_hex_to_bgr(c["hex"]) for c in COLOR_PALETTE]], dtype=np.uint8)
        lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0].astype(np.float32)
        _anchors_cache = [
            {"name": c["name"], "lab": lab[i]} for i, c in enumerate(COLOR_PALETTE)
        ]
    return _anchors_cache


def _pixels_from_polygon(img_bgr, polygon):
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
    cv2.fillPoly(mask, [pts], 255)
    # Rand der Maske abtragen, damit Kantenpixel (Wand/Schatten) rausfallen
    erode_px = max(1, int(0.03 * max(img_bgr.shape[:2])))
    mask = cv2.erode(mask, np.ones((erode_px, erode_px), np.uint8))
    return img_bgr[mask > 0]


def _pixels_from_bbox(img_bgr, bbox, inner=0.6):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 3), dtype=np.uint8)
    # nur der innere Bereich: die Box umschließt den Griff, die Ecken sind Wand
    bw, bh = x2 - x1, y2 - y1
    dx = int(bw * (1 - inner) / 2)
    dy = int(bh * (1 - inner) / 2)
    crop = img_bgr[y1 + dy : y2 - dy, x1 + dx : x2 - dx]
    return crop.reshape(-1, 3)


def _dominant_lab(pixels_bgr):
    """Lab-Zentrum des größten k-means-Clusters + dessen Pixelanteil."""
    lab = cv2.cvtColor(pixels_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2LAB)
    samples = lab.reshape(-1, 3).astype(np.float32)
    if len(samples) > 4000:
        idx = np.random.default_rng(0).choice(len(samples), 4000, replace=False)
        samples = samples[idx]
    k = 3 if len(samples) >= 60 else 1
    if k == 1:
        return samples.mean(axis=0), 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.flatten(), minlength=k)
    best = int(np.argmax(counts))
    return centers[best], float(counts[best]) / float(len(samples))


def _match_palette(lab_center):
    l, a, b = [float(v) for v in lab_center]
    chroma = ((a - 128.0) ** 2 + (b - 128.0) ** 2) ** 0.5
    candidates = []
    for anchor in anchors():
        is_achromatic = anchor["name"] in ACHROMATIC
        if chroma < CHROMA_THRESHOLD and not is_achromatic:
            continue
        if chroma >= CHROMA_THRESHOLD and is_achromatic:
            continue
        al, aa, ab = [float(v) for v in anchor["lab"]]
        dist = (
            (LIGHTNESS_WEIGHT * (l - al)) ** 2 + (a - aa) ** 2 + (b - ab) ** 2
        ) ** 0.5
        candidates.append((dist, anchor["name"]))
    candidates.sort()
    best_dist, best_name = candidates[0]
    margin = candidates[1][0] - best_dist if len(candidates) > 1 else best_dist
    return best_name, best_dist, margin


def extract_color(img_bgr, bbox=None, polygon=None):
    """Grifffarbe bestimmen. Liefert Palettenfarbe + Konfidenz.

    polygon: Liste [[x, y], ...] in Pixelkoordinaten (bevorzugt).
    bbox: [x1, y1, x2, y2] als Rückfallebene.
    """
    pixels = np.empty((0, 3), dtype=np.uint8)
    if polygon is not None and len(polygon) >= 3:
        pixels = _pixels_from_polygon(img_bgr, polygon)
    if len(pixels) < MIN_PIXELS and bbox is not None:
        pixels = _pixels_from_bbox(img_bgr, bbox)
    if len(pixels) < MIN_PIXELS and bbox is not None:
        pixels = _pixels_from_bbox(img_bgr, bbox, inner=1.0)
    if len(pixels) < 3:
        entry = COLOR_PALETTE[COLOR_INDEX["grey"]]
        return {
            "name": entry["name"], "hex": entry["hex"],
            "index": COLOR_INDEX["grey"], "confidence": 0.0,
        }

    center, share = _dominant_lab(np.ascontiguousarray(pixels, dtype=np.uint8))
    name, dist, margin = _match_palette(center)
    # Konfidenz: wie klar dominiert das Cluster x wie eindeutig der Anker.
    confidence = share * float(np.clip(margin / 40.0, 0.05, 1.0))
    entry = COLOR_PALETTE[COLOR_INDEX[name]]
    return {
        "name": name,
        "hex": entry["hex"],
        "index": COLOR_INDEX[name],
        "confidence": round(float(confidence), 3),
    }
