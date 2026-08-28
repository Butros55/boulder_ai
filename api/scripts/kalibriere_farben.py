"""Farbanker an den eigenen Hallenfotos kalibrieren.

Nutzt die wiederhergestellten Alt-Annotationen (Farbklasse je Box, Ground
Truth) als Messlatte: extract_color wird auf jeder Box ausgewertet, dann
werden die Anker **aus den Daten** neu bestimmt (Median von L/Chroma/Hue je
Klasse) und nach vision/farb_anker.json geschrieben, das anchors() bei
Vorhandensein lädt. Vorher/Nachher-Genauigkeit und Konfusionen werden
ausgegeben.

Aufruf aus api/:  python scripts/kalibriere_farben.py
"""

import json
import sys
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))

SCRATCH = Path(r"C:\Users\geret\AppData\Local\Temp\claude\C--dev-Vault"
               r"\023a5e3f-9087-49f8-a7c0-d8f4b1329992\scratchpad")
LS_MEDIA = Path.home() / "AppData" / "Local" / "label-studio" / "label-studio" / "media"
ANKER_DATEI = API_DIR / "vision" / "farb_anker.json"

# alte Labelnamen -> aktuelle Palette (dark-green gab es 47x, black-white 9x)
NAME_MAP = {
    "red": "red", "blue": "blue", "white": "white", "yellow": "yellow",
    "turquoise": "mint", "purple": "purple", "orange": "orange",
    "black": "black", "grey": "grey", "wood": "wood",
    "dark-green": "green",
}


def boxen_sammeln():
    """(bild_pfad, x0,y0,x1,y1, zielklasse) aus beiden Alt-Projekten."""
    eintraege = []

    def aus_export(export_pfad, bild_aufloeser):
        data = json.loads(Path(export_pfad).read_text(encoding="utf-8"))
        for t in data:
            ref = t.get("data", {}).get("image_url") or t.get("data", {}).get("image", "")
            pfad = bild_aufloeser(ref)
            if pfad is None or not pfad.exists():
                continue
            for a in t.get("annotations", []):
                if a.get("was_cancelled"):
                    continue
                for res in a.get("result", []):
                    v = res.get("value", {})
                    if "rectanglelabels" not in v:
                        continue
                    ziel = NAME_MAP.get(v["rectanglelabels"][0])
                    if ziel is None:
                        continue
                    eintraege.append((pfad, v["x"], v["y"],
                                      v["width"], v["height"], ziel))

    mapping_v2 = json.loads(
        (API_DIR / "datasets" / "alt_match_v2" / "mapping.json").read_text())
    jpg_dir = API_DIR / "datasets" / "eigene_halle_jpg"

    def aufloeser_v2(ref):
        name = ref.replace("%5C", "/").replace("\\", "/").rsplit("/", 1)[-1]
        info = mapping_v2.get(name)
        return (jpg_dir / info["bild"]) if info else None

    def aufloeser_v1(ref):
        rel = urllib.parse.unquote(ref.replace("/data/", ""))
        return LS_MEDIA / rel

    aus_export(SCRATCH / "alt_colors_v2.json", aufloeser_v2)
    aus_export(SCRATCH / "alt_colors_v1.json", aufloeser_v1)
    return eintraege


def main():
    from vision.color_extraction import _dominant_lab, _pixels_from_bbox

    eintraege = boxen_sammeln()
    print(f"[daten] {len(eintraege)} gelabelte Boxen")

    # Lab-Zentren je Box messen (Cache je Bild)
    cache = {}
    messungen = []  # (lab_center, zielklasse)
    for pfad, x, y, w, h, ziel in eintraege:
        if pfad not in cache:
            cache[pfad] = cv2.imread(str(pfad))
        img = cache[pfad]
        if img is None:
            continue
        H, W = img.shape[:2]
        bbox = [x * W / 100, y * H / 100, (x + w) * W / 100, (y + h) * H / 100]
        px = _pixels_from_bbox(img, bbox)
        if len(px) < 12:
            continue
        center, _ = _dominant_lab(np.ascontiguousarray(px, dtype=np.uint8))
        messungen.append((center, ziel))
    print(f"[messung] {len(messungen)} Lab-Zentren")

    def bewerten(tag):
        from vision.color_extraction import _match_palette
        richtig = 0
        konfusion = Counter()
        for center, ziel in messungen:
            name, _, _ = _match_palette(center)
            if name == ziel:
                richtig += 1
            else:
                konfusion[(ziel, name)] += 1
        acc = richtig / max(1, len(messungen))
        print(f"[{tag}] Genauigkeit {acc:.1%}")
        for (soll, ist), n in konfusion.most_common(8):
            print(f"    {soll} -> {ist}: {n}x")
        return acc

    acc_vorher = bewerten("vorher")

    # datengetriebene Anker: Median über L, zirkulärer Mittelwert über Hue
    je_klasse = defaultdict(list)
    for center, ziel in messungen:
        je_klasse[ziel].append(center)
    anker = {}
    for name, zentren in je_klasse.items():
        arr = np.array(zentren, dtype=float)
        L = float(np.median(arr[:, 0]))
        a = arr[:, 1] - 128.0
        b = arr[:, 2] - 128.0
        chroma = float(np.median(np.sqrt(a ** 2 + b ** 2)))
        hue = float(np.degrees(np.arctan2(np.median(b), np.median(a))))
        anker[name] = {"l": round(L, 1), "chroma": round(chroma, 1),
                       "hue": round(hue, 1), "n": len(zentren)}
    ANKER_DATEI.write_text(json.dumps(anker, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"[anker] {len(anker)} kalibrierte Anker -> {ANKER_DATEI.name}")

    # Cache leeren und neu bewerten
    import vision.color_extraction as ce
    ce._anchors_cache = None
    acc_nachher = bewerten("nachher")
    print(f"\nGenauigkeit: {acc_vorher:.1%} -> {acc_nachher:.1%}")


if __name__ == "__main__":
    main()
