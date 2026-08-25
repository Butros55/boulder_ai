import cv2
import numpy as np
import pytest

from vision.classes import COLOR_PALETTE
from vision.color_extraction import _hex_to_bgr, extract_color

WALL = (180, 175, 170)  # helle Wand als Hintergrund


def make_image(color_bgr, bg=WALL, size=200, radius=60):
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, color_bgr, -1)
    return img


def circle_polygon(size=200, radius=60, n=24):
    cx = cy = size // 2
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [[cx + radius * np.cos(a), cy + radius * np.sin(a)] for a in angles]


@pytest.mark.parametrize("entry", COLOR_PALETTE, ids=[c["name"] for c in COLOR_PALETTE])
def test_palette_color_from_polygon(entry):
    img = make_image(_hex_to_bgr(entry["hex"]))
    result = extract_color(img, polygon=circle_polygon())
    assert result["name"] == entry["name"]
    assert result["confidence"] > 0.1


@pytest.mark.parametrize("entry", COLOR_PALETTE, ids=[c["name"] for c in COLOR_PALETTE])
def test_palette_color_from_bbox_with_background(entry):
    # Bbox umschließt den Kreis samt Wandecken — die Farbe muss trotzdem stimmen
    img = make_image(_hex_to_bgr(entry["hex"]))
    result = extract_color(img, bbox=[35, 35, 165, 165])
    assert result["name"] == entry["name"]


def test_lighting_shift_keeps_hue():
    # abgedunkeltes Rot bleibt Rot (L zählt schwächer als der Farbstich)
    base = np.array(_hex_to_bgr(COLOR_PALETTE[5]["hex"]), dtype=np.float32)  # red
    dark = tuple(int(v) for v in (base * 0.6))
    img = make_image(dark)
    result = extract_color(img, polygon=circle_polygon())
    assert result["name"] == "red"


def test_screw_holes_do_not_flip_color():
    # kleine schwarze Punkte (Schrauben) im blauen Griff: Cluster-Mehrheit gewinnt
    img = make_image(_hex_to_bgr(COLOR_PALETTE[1]["hex"]))  # blue
    for pos in [(80, 80), (120, 120), (100, 90)]:
        cv2.circle(img, pos, 6, (10, 10, 10), -1)
    result = extract_color(img, polygon=circle_polygon())
    assert result["name"] == "blue"


def test_degenerate_bbox_falls_back_to_grey():
    img = make_image(_hex_to_bgr(COLOR_PALETTE[5]["hex"]))
    result = extract_color(img, bbox=[5, 5, 5, 5])
    assert result["confidence"] == 0.0
    assert result["name"] == "grey"
