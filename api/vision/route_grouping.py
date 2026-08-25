"""Detektionen zu Routen gruppieren.

Eine Route = Griffe derselben (extrahierten) Farbe, die räumlich
zusammenhängen. Clustering per DBSCAN mit min_samples=2 — das entspricht
Zusammenhangskomponenten über dem eps-Nachbarschaftsgraphen, Ketten von
Griffen bleiben also eine Route. Zu kleine Gruppen fallen raus.

Rückgabeformat wie bisher: Liste von Routen, jede Route eine Liste der
Detection-Dicts (die App verlässt sich darauf).
"""

import numpy as np
from sklearn.cluster import DBSCAN


def group_routes(detections, image_width, image_height,
                 eps_fraction=0.18, min_grip_count=3):
    by_color = {}
    for det in detections:
        color = det.get("color", {}).get("name") or det.get("class_name")
        if not color:
            continue
        by_color.setdefault(color, []).append(det)

    eps = max(1.0, eps_fraction * min(image_width, image_height))
    routes = []
    for color, group in by_color.items():
        if len(group) < min_grip_count:
            continue
        centers = np.array(
            [
                [
                    (d["bbox"][0] + d["bbox"][2]) / 2.0,
                    (d["bbox"][1] + d["bbox"][3]) / 2.0,
                ]
                for d in group
            ],
            dtype=np.float64,
        )
        labels = DBSCAN(eps=eps, min_samples=2).fit_predict(centers)
        for label in sorted(set(labels)):
            if label == -1:
                continue
            members = [group[i] for i in range(len(group)) if labels[i] == label]
            if len(members) >= min_grip_count:
                routes.append(members)

    # größte Routen zuerst — die App zeigt sie in dieser Reihenfolge
    routes.sort(key=len, reverse=True)
    return routes
