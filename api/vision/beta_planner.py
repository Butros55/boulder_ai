"""Zugvorschläge (Beta) für eine erkannte Route ableiten — Version 1.

Regelbasiert, ohne Trainingsdaten: Die Route wird von unten nach oben als
Grifffolge geordnet, dann werden Hände zugewiesen. Grundregeln aus der
Kletterpraxis, bewusst einfach gehalten:

- Start am tiefsten Griff; liegen zwei Griffe fast gleich tief nebeneinander,
  ist es ein Startgriffpaar (beide Hände).
- Es greift bevorzugt die Hand, auf deren Seite der nächste Griff liegt
  (rechts vom Körper -> rechte Hand), sonst wird alterniert.
- Sehr weite Züge (über ~1,6x des typischen Zugabstands) werden gedoppelt:
  erst beide Hände auf den aktuellen Griff, dann weiter.
- Der letzte Griff ist das Top und wird gedoppelt.

Das ist eine plausible Standard-Beta, keine Biomechanik-Simulation:
Körpergröße, Kraft und Technik verschieben die echte Lösung. Die Ausgabe
ist dafür gedacht, in der App als nummerierte Hand-Schritte angezeigt und
später durch gelernte oder physikbasierte Modelle ersetzt zu werden.

Regelbasis: [[Beta-Regelwerk Bouldern]] im Vault (destilliert aus
Technik-Guides, Regelwerken und der BetaMove-/Strange-Beta-Literatur).
Umgesetzt sind die rein aus 2D-Positionen und -größen ableitbaren Regeln:
Bump, Kreuzer, Deadpoint/Dyno-Staffelung mit Zielgriffgröße als Gate,
Barn-Door-/Flag-Hinweis, Sitzstart-Erkennung. Orientierungsabhängige
Techniken (Sidepull, Gaston, Undercling) und Fußarbeit folgen in v3.

Ausgabe je Route: Liste von Schritten
    {"step": 1, "hold": <Index in der Routen-Liste>, "hand": "left"|"right"|"both",
     "action": "start"|"move"|"match"|"top", "hinweis": <optional, deutsch>}
"""

import numpy as np

# ab diesem Vielfachen des typischen Zugabstands wird vor dem Zug gedoppelt
LONG_MOVE_FACTOR = 1.6
# knapp über der Komfortreichweite: kontrolliert-dynamisch (Deadpoint)
DEADPOINT_FACTOR = 2.0
# deutlich außer Reichweite ohne Zwischengriff: Sprungzug (Dyno)
DYNO_FACTOR = 2.6
# Bump: Zwischengriff derselben Seite näher als dieses Vielfache
BUMP_FACTOR = 0.7
# maximaler Höhenunterschied (relativ zum typischen Zugabstand), bei dem zwei
# tiefe Griffe als Startpaar gelten
START_PAIR_DY = 0.5


def _center(det):
    b = det["bbox"]
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _order_upward(route):
    """Griffe streng von unten nach oben ordnen; bei ähnlicher Höhe
    entscheidet die Nähe zum Vorgänger. Eine Greedy-Pfadsuche klang
    schlauer, lief aber in Sackgassen und musste wieder nach unten —
    monotone Höhenordnung ist für eine Standard-Beta die ehrlichere Basis."""
    centers = [_center(d) for d in route]
    ys = sorted((c[1] for c in centers), reverse=True)
    band = np.median([abs(a - b) for a, b in zip(ys, ys[1:])]) if len(ys) > 1 else 0.0

    remaining = sorted(range(len(route)), key=lambda i: -centers[i][1])
    order = [remaining.pop(0)]
    while remaining:
        cx, cy = centers[order[-1]]
        # Kandidaten: alle im untersten Höhenband der verbleibenden Griffe
        base_y = centers[remaining[0]][1]
        candidates = [i for i in remaining
                      if centers[i][1] >= base_y - 0.5 * band]
        best = min(candidates, key=lambda i: abs(centers[i][0] - cx))
        order.append(best)
        remaining.remove(best)
    return order


def _size(det):
    b = det["bbox"]
    return max(1.0, min(b[2] - b[0], b[3] - b[1]))


def plan_beta(route, image_height=None):
    """Standard-Beta für eine Route (Liste von Detection-Dicts)."""
    if len(route) < 2:
        return []
    centers = [_center(d) for d in route]
    order = _order_upward(route)
    median_size = float(np.median([_size(d) for d in route]))

    dists = [
        float(np.hypot(centers[a][0] - centers[b][0], centers[a][1] - centers[b][1]))
        for a, b in zip(order, order[1:])
    ]
    typical = float(np.median(dists)) if dists else 0.0

    steps = []
    step_no = 1

    # Startgriff(e): zwei fast gleich tiefe Griffe am Anfang = Startpaar
    first, second = order[0], order[1] if len(order) > 1 else None
    start_pair = (
        second is not None
        and typical > 0
        and abs(centers[first][1] - centers[second][1]) <= START_PAIR_DY * typical
    )
    # Sitzstart-Hinweis: Startgriffe extrem tief im Bild
    sit_start = (
        image_height is not None
        and centers[first][1] > 0.85 * float(image_height)
    )

    if start_pair:
        left_first = centers[first][0] <= centers[second][0]
        start_note = "evtl. Sitzstart" if sit_start else None
        steps.append({"step": step_no, "hold": first,
                      "hand": "left" if left_first else "right",
                      "action": "start",
                      **({"hinweis": start_note} if start_note else {})})
        steps.append({"step": step_no, "hold": second,
                      "hand": "right" if left_first else "left",
                      "action": "start"})
        step_no += 1
        last_hand = "right" if left_first else "left"
        used = 2
    else:
        step = {"step": step_no, "hold": first, "hand": "both",
                "action": "start"}
        if sit_start:
            step["hinweis"] = "evtl. Sitzstart"
        steps.append(step)
        step_no += 1
        last_hand = "right"
        used = 1

    for k in range(used, len(order)):
        idx = order[k]
        prev = order[k - 1]
        dx = centers[idx][0] - centers[prev][0]
        dist = float(np.hypot(dx, centers[idx][1] - centers[prev][1]))
        is_top = k == len(order) - 1
        target_small = _size(route[idx]) < 0.75 * median_size
        note = None

        # weite Züge: erst doppeln, dann Staffelung statisch/Deadpoint/Dyno
        if typical > 0 and dist > LONG_MOVE_FACTOR * typical and not is_top:
            steps.append({"step": step_no, "hold": prev, "hand": "both",
                          "action": "match",
                          "hinweis": "Doppeln vor weitem Zug"})
            step_no += 1
            if dist > DYNO_FACTOR * typical and not target_small:
                note = "weiter Zug — evtl. Dyno"
            elif dist > DEADPOINT_FACTOR * typical:
                # kleines Ziel schließt den Sprung aus (Präzision nötig)
                note = "Deadpoint: kontrolliert dynamisch"

        same_side = (dx > 0 and last_hand == "right") or \
                    (dx < 0 and last_hand == "left")
        if is_top:
            hand = "both"
        elif same_side and abs(dx) > 0.25 * max(typical, 1.0) \
                and dist < BUMP_FACTOR * max(typical, 1.0):
            # naher Zwischengriff auf der Zugseite: dieselbe Hand zieht nach
            hand = last_hand
            note = note or "Bump: gleiche Hand zieht nach"
        elif abs(dx) > 0.25 * max(typical, 1.0):
            # nächster Griff klar seitlich: die Hand dieser Seite greift
            hand = "right" if dx > 0 else "left"
            if typical > 0 and abs(dx) > 1.2 * typical:
                # Ziel weit jenseits der Körperlinie: Kreuzer oder Umgreifen
                note = note or "weit seitlich — Kreuzer oder umgreifen"
        else:
            hand = "left" if last_hand == "right" else "right"

        # Diagonalregel: zieht die Hand der Zugrichtungsseite, droht das
        # Aufdrehen (Barn Door) — Flag-Hinweis
        if note is None and not is_top and hand != "both" \
                and abs(dx) > 0.6 * max(typical, 1.0) \
                and ((dx > 0) == (hand == "right")):
            note = "Flaggen gegen Aufdrehen"

        step = {
            "step": step_no,
            "hold": idx,
            "hand": hand,
            "action": "top" if is_top else "move",
        }
        if note:
            step["hinweis"] = note
        steps.append(step)
        step_no += 1
        if hand != "both":
            last_hand = hand

    return steps
