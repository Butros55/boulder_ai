"""Detektionen zu Routen gruppieren.

Eine Route = Griffe gleicher Farbe, die räumlich zusammenhängen. Statt
strikt über den Palettennamen zu gruppieren, vergleicht die Gruppierung —
wo vorhanden — die gemessenen Lab-Werte der Griffe: zwei kräftig bunte
Griffe gehören zusammen, wenn ihr Farbton nah beieinander liegt, auch wenn
die Palettenquantisierung sie auf verschiedene Namen gerundet hat
(ocker/gelb, rot/pink). Das war in der Praxis der häufigste Grund für
zerfallene Routen. Unbunte oder alte Payloads fallen auf Namensgleichheit
zurück.

Clustering: Zusammenhangskomponenten über einem eps-Nachbarschaftsgraphen
(DBSCAN, min_samples=2, vorberechnete Distanzmatrix — unverträgliche
Farben stehen auf unendlich). Der Radius wird je Bild aus dem Median des
Nächste-Nachbar-Abstands farbverträglicher Griffe geschätzt; horizontale
Abstände zählen leicht stärker als vertikale, weil Routen nach oben laufen.

Volumes werden nicht zu Routen gezählt: Sie sind in der Regel neutral
(von allen Routen nutzbar) und würden sonst getrennte Routen verkleben.

Rückgabeformat wie bisher: Liste von Routen, jede Route eine Liste der
Detection-Dicts (die App verlässt sich darauf).
"""

import numpy as np
from sklearn.cluster import DBSCAN

# Lücke im Farbton (Grad), an der zwischen zwei Routenfarben getrennt wird
HUE_GAP = 18.0
# breiter darf ein einzelnes Farbband nicht werden — sonst an der größten
# inneren Lücke nachteilen (fängt schleichende Ketten ab)
HUE_BAND_MAX = 45.0
# darunter gilt ein Griff als unbunt und wird über den Palettennamen verglichen
CHROMA_MIN = 8.0
# Routen laufen nach oben: seitlicher Versatz wiegt schwerer als vertikaler
X_WEIGHT = 1.3


def _color_features(det):
    name = det.get("color", {}).get("name") or det.get("class_name")
    lab = det.get("color", {}).get("lab")
    if lab is None:
        return name, None, None
    l, a, b = [float(v) for v in lab]
    chroma = ((a - 128.0) ** 2 + (b - 128.0) ** 2) ** 0.5
    hue = float(np.degrees(np.arctan2(b - 128.0, a - 128.0)))
    return name, chroma, hue


def _hue_bands(hues):
    """Farbtöne (Grad, zirkulär) in Bänder teilen: Schnitt an Lücken >= HUE_GAP.

    Eine Route ist EINE Farbe. Paarweise Toleranz reichte nicht, weil sie
    transitiv kettet (rot~orange~ocker~gelb wird eine Mega-Route). Bänder
    trennen dort, wo im Bild tatsächlich eine Farblücke ist, und tolerieren
    zugleich die Streuung innerhalb einer Farbe.
    Rückgabe: Bandindex je Eingabewert.
    """
    n = len(hues)
    if n == 0:
        return []
    reihenfolge = sorted(range(n), key=lambda i: hues[i] % 360.0)
    sortiert = [hues[i] % 360.0 for i in reihenfolge]
    # zirkulär: die größte Lücke bestimmt, wo der Kreis aufgeschnitten wird
    luecken = [(sortiert[(k + 1) % n] - sortiert[k]) % 360.0 for k in range(n)]
    start = (int(np.argmax(luecken)) + 1) % n

    band = [0] * n
    aktuelles_band = 0
    band_start_wert = sortiert[start]
    for schritt in range(n):
        k = (start + schritt) % n
        if schritt > 0:
            vorher = (start + schritt - 1) % n
            luecke = (sortiert[k] - sortiert[vorher]) % 360.0
            spannweite = (sortiert[k] - band_start_wert) % 360.0
            if luecke >= HUE_GAP or spannweite > HUE_BAND_MAX:
                aktuelles_band += 1
                band_start_wert = sortiert[k]
        band[reihenfolge[k]] = aktuelles_band
    return band


def _color_groups(holds, features):
    """Griffe in Farbgruppen teilen: unbunte nach Palettenname, bunte
    nach Farbton-Band. Rückgabe: Gruppenschlüssel je Griff."""
    chrom_idx = [i for i, f in enumerate(features)
                 if f[1] is not None and f[1] >= CHROMA_MIN]
    baender = _hue_bands([features[i][2] for i in chrom_idx])
    schluessel = {}
    for pos, i in enumerate(chrom_idx):
        schluessel[i] = f"hue{baender[pos]}"
    for i, f in enumerate(features):
        if i not in schluessel:
            schluessel[i] = f"name:{f[0]}"
    return schluessel


def group_routes(detections, image_width, image_height,
                 neighbor_factor=2.6, min_grip_count=3):
    holds = [d for d in detections if d.get("det_class", "hold") != "volume"]
    holds = [d for d in holds if _color_features(d)[0] is not None]
    n = len(holds)
    if n < min_grip_count:
        return []

    centers = np.array(
        [
            [
                (d["bbox"][0] + d["bbox"][2]) / 2.0,
                (d["bbox"][1] + d["bbox"][3]) / 2.0,
            ]
            for d in holds
        ],
        dtype=np.float64,
    )
    features = [_color_features(d) for d in holds]
    schluessel = _color_groups(holds, features)

    short_side = float(min(image_width, image_height))
    routes = []
    for gruppe in sorted(set(schluessel.values())):
        idx = [i for i in range(n) if schluessel[i] == gruppe]
        if len(idx) < min_grip_count:
            continue
        pts = centers[idx] * np.array([X_WEIGHT, 1.0])
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        typical_spacing = float(np.median(dist.min(axis=1)))
        eps = float(np.clip(neighbor_factor * typical_spacing,
                            0.04 * short_side, 0.25 * short_side))
        labels = DBSCAN(eps=eps, min_samples=2).fit_predict(pts)
        for label in sorted(set(labels)):
            if label == -1:
                continue
            members = [holds[i] for k, i in enumerate(idx) if labels[k] == label]
            if len(members) >= min_grip_count:
                routes.append(members)

    # größte Routen zuerst — die App zeigt sie in dieser Reihenfolge
    routes.sort(key=len, reverse=True)
    return routes
