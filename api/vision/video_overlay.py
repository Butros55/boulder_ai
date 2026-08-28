"""Hallenvideo mit Grifferkennungs-Overlay rendern (Kern für CLI und API).

Kein Einzelbild-Malen: Das Modul führt **persistente Tracks** — Griffe
bewegen sich nicht, nur die Kamera. Detektionen (stride-ausgedünnt)
bestätigen und glätten die Tracks, die globale Kamerabewegung (per
Phasenkorrelation geschätzt) verschiebt alle Tracks in jedem Frame.
Dadurch bleibt ein Griff auch dann stehen, wenn die Detektion ein paar
Frames aussetzt, die **Farbe wird einmal je Track bestimmt** (Median der
ersten Messungen) statt pro Frame neu, und SAM-Formen hängen am Track
und überleben Neu-Detektionen. Das beseitigt das Flackern von Boxen,
Formen und Farben, das eine reine Pro-Frame-Pipeline zeigt.
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np

API_DIR = Path(__file__).resolve().parent.parent

# Track-Parameter
POLY_MAX_ALTER = 45       # Frames, nach denen eine Form neu berechnet wird
EMA_ALPHA = 0.55          # Glättung der Boxpositionen bei Bestätigung
BESTAETIGUNGEN = 2        # so oft muss ein neuer Track gesehen werden, bevor er malt
GEDULD_FRAMES = 45        # so lange überlebt ein Track ohne Bestätigung
FARB_MESSUNGEN = 3        # Farbmessungen, aus denen der Median-Farbwert entsteht
IOU_MATCH = 0.35          # Mindest-IoU Detektion<->Track


def _hex_to_bgr(hex_str):
    h = hex_str.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def _volumen_maske(frame, bbox):
    """Volume-Form per Hintergrund-Subtraktion im Box-Ausschnitt.

    SAM segmentiert konkave, glänzende Volumes (Halbmonde) per Box-Prompt
    notorisch falsch — es füllt die Aushöhlung. Robuster für unifarbene
    Hallenobjekte: k-means im Ausschnitt, die Cluster, die den Boxrand
    dominieren, sind Wand/Hintergrund, der Rest ist das Volume.
    Rückgabe: Kontur in Vollbild-Koordinaten oder None.
    """
    h, w = frame.shape[:2]
    x0, y0 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x1, y1 = min(w, int(bbox[2])), min(h, int(bbox[3]))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    crop = frame[y0:y1, x0:x1]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    k = 3
    kriterium = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _, labels, _ = cv2.kmeans(lab, k, None, kriterium, 2, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(crop.shape[:2])
    rand = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    maske = np.ones(crop.shape[:2], np.uint8)
    for c in range(k):
        # Cluster, die >35 % des Rands stellen, sind Hintergrund
        if (rand == c).mean() > 0.35:
            maske[labels == c] = 0
    kern = np.ones((5, 5), np.uint8)
    maske = cv2.morphologyEx(maske * 255, cv2.MORPH_OPEN, kern)
    maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, kern)
    cnts, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.10 * (x1 - x0) * (y1 - y0):
        return None
    return (c.reshape(-1, 2) + np.array([x0, y0])).astype(np.float64)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


class _Track:
    def __init__(self, bbox, volume, track_id):
        self.id = track_id
        self.bbox = list(bbox)
        self.volume = volume
        self.gesehen = 1
        self.verpasst = 0
        self.farben = []      # (name, hex) der ersten Messungen
        self.farbe_bgr = None
        self.poly = None
        self.poly_alter = 0   # Frames seit der letzten SAM-Neuberechnung

    def bestaetigen(self, bbox):
        a = EMA_ALPHA
        alt = list(self.bbox)
        self.bbox = [self.bbox[i] * (1 - a) + bbox[i] * a for i in range(4)]
        if self.poly is not None:
            # Form an der neuen Box verankern: verschieben UND skalieren,
            # sonst läuft sie bei Zoom aus der Position
            alt_cx, alt_cy = (alt[0] + alt[2]) / 2, (alt[1] + alt[3]) / 2
            neu_cx = (self.bbox[0] + self.bbox[2]) / 2
            neu_cy = (self.bbox[1] + self.bbox[3]) / 2
            sw = (self.bbox[2] - self.bbox[0]) / max(alt[2] - alt[0], 1e-6)
            sh = (self.bbox[3] - self.bbox[1]) / max(alt[3] - alt[1], 1e-6)
            p = self.poly - np.array([alt_cx, alt_cy])
            p = p * np.array([sw, sh])
            self.poly = p + np.array([neu_cx, neu_cy])
            # Qualitätsprüfung: eine Form, deren Fläche nicht mehr zur Box
            # passt (z. B. SAM-Fehlmaske von einem bewegungsunscharfen
            # Frame), wird verworfen und eifrig neu berechnet
            box_flaeche = ((self.bbox[2] - self.bbox[0])
                           * (self.bbox[3] - self.bbox[1]))
            poly_flaeche = float(cv2.contourArea(
                self.poly.astype(np.float32)))
            if box_flaeche > 0:
                verhaeltnis = poly_flaeche / box_flaeche
                if verhaeltnis > 1.25 or verhaeltnis < 0.30:
                    self.poly = None
        self.gesehen += 1
        self.verpasst = 0

    def transformieren(self, M):
        """Affine Kamerabewegung (2x3-Matrix in Vollbild-Koordinaten)."""
        ecken = np.array([[self.bbox[0], self.bbox[1]],
                          [self.bbox[2], self.bbox[3]]])
        neu = ecken @ M[:, :2].T + M[:, 2]
        self.bbox = [float(min(neu[0, 0], neu[1, 0])),
                     float(min(neu[0, 1], neu[1, 1])),
                     float(max(neu[0, 0], neu[1, 0])),
                     float(max(neu[0, 1], neu[1, 1]))]
        if self.poly is not None:
            self.poly = self.poly @ M[:, :2].T + M[:, 2]

    def farbe_messen(self, frame, extract_color):
        if len(self.farben) >= FARB_MESSUNGEN:
            return
        c = extract_color(frame, bbox=self.bbox)
        self.farben.append(c["hex"])
        # Median über die bisherigen Messungen: häufigster Hex gewinnt
        werte, zaehl = np.unique(self.farben, return_counts=True)
        self.farbe_bgr = _hex_to_bgr(str(werte[np.argmax(zaehl)]))


def render_video_legacy(src, dst, weights=None, conf=0.4, stride=2, shapes=True,
                        keyframe=20, max_seconds=90.0, progress_cb=None):
    """Rendert src nach dst (mp4). Gibt (frames, sekunden) zurück."""
    from ultralytics import YOLO

    from vision.color_extraction import extract_color

    yolo = YOLO(weights or str(API_DIR / "weights" / "best.pt"))
    sam = None
    if shapes:
        from ultralytics import SAM
        import torch
        default_w = "sam2.1_b.pt" if torch.cuda.is_available() else "sam2.1_t.pt"
        sam = SAM(os.getenv("SAM_WEIGHTS", default_w))

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"Video nicht lesbar: {src}")
    # iPhone-Videos tragen die Hochkant-Lage als Rotations-Metadatum;
    # OpenCV liefert die Frames sonst quer. Erst Auto-Orientierung
    # versuchen, sonst von Hand drehen.
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except cv2.error:
        pass
    meta = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
    ok_probe, probe = cap.read()
    rotate = None
    if ok_probe and meta in (90, 270) and probe.shape[1] > probe.shape[0]:
        rotate = (cv2.ROTATE_90_CLOCKWISE if meta == 90
                  else cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif ok_probe and meta == 180:
        rotate = cv2.ROTATE_180
    cap.release()
    cap = cv2.VideoCapture(str(src))
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except cv2.error:
        pass

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if ok_probe:
        ph, pw = probe.shape[:2]
        if rotate in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            ph, pw = pw, ph
        w, h = pw, ph
    else:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_seconds:
        n_frames = min(n_frames, int(max_seconds * fps))
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracks = []
    naechste_id = 0
    prev_small = None
    t0 = time.time()
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if rotate is not None and frame.shape[1] != w:
            frame = cv2.rotate(frame, rotate)
        small = cv2.cvtColor(cv2.resize(frame, (480, max(1, 480 * h // w))),
                             cv2.COLOR_BGR2GRAY)

        # 1) Kamerabewegung: affines Modell (Rotation/Zoom/Verschiebung) aus
        # Feature-Tracking — reine Translation (Phasenkorrelation) ließ die
        # Formen bei Handy-Zoom und -Drehung von den Griffen laufen
        if prev_small is not None and tracks:
            M_full = None
            pts0 = cv2.goodFeaturesToTrack(prev_small, maxCorners=250,
                                           qualityLevel=0.01, minDistance=8)
            if pts0 is not None and len(pts0) >= 12:
                pts1, st, _ = cv2.calcOpticalFlowPyrLK(prev_small, small,
                                                       pts0, None)
                gut = st.reshape(-1) == 1
                if gut.sum() >= 12:
                    M, _ = cv2.estimateAffinePartial2D(
                        pts0[gut], pts1[gut], method=cv2.RANSAC,
                        ransacReprojThreshold=3.0)
                    if M is not None:
                        s = w / 480.0
                        M_full = M.copy()
                        M_full[:, 2] *= s  # Translation auf Vollbild skalieren
            if M_full is None:
                (dx, dy), _ = cv2.phaseCorrelate(prev_small.astype(np.float32),
                                                 small.astype(np.float32))
                M_full = np.array([[1, 0, dx * w / 480.0],
                                   [0, 1, dy * w / 480.0]], dtype=np.float64)
            for t in tracks:
                t.transformieren(M_full)
        prev_small = small

        # 2) Detektionen bestätigen/erzeugen Tracks
        if i % stride == 0:
            r = yolo.predict(source=frame, conf=conf, verbose=False)[0]
            roh = sorted(
                ([[float(v) for v in b.xyxy[0]], int(b.cls[0]) == 1,
                  float(b.conf[0])] for b in r.boxes),
                key=lambda d: -d[2])
            # Detektions-Dedup: überlappende Roh-Boxen (Doppel-Detektionen
            # desselben Griffs) auf die konfidenteste eindampfen
            dets = []
            for bb, ist_vol, c in roh:
                if all(_iou(bb, d[0]) < 0.55 for d in dets):
                    dets.append((bb, ist_vol, c))

            frei = set(range(len(tracks)))
            for bb, ist_vol, c in dets:
                best, best_iou = None, IOU_MATCH
                for ti in frei:
                    iou = _iou(bb, tracks[ti].bbox)
                    if iou > best_iou:
                        best, best_iou = ti, iou
                if best is None:
                    # Zentrums-Fallback: leicht weggedriftete Tracks wieder
                    # einfangen statt einen Duplikat-Track anzulegen
                    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
                    for ti in frei:
                        tb = tracks[ti].bbox
                        if tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3] \
                                and _iou(bb, tb) > 0.1:
                            best = ti
                            break
                if best is not None:
                    tracks[best].bestaetigen(bb)
                    tracks[best].farbe_messen(frame, extract_color)
                    frei.discard(best)
                else:
                    t = _Track(bb, ist_vol, naechste_id)
                    naechste_id += 1
                    t.farbe_messen(frame, extract_color)
                    tracks.append(t)
            for ti in frei:
                tracks[ti].verpasst += stride
            # abgelaufene oder nie bestätigte Tracks entsorgen
            tracks = [t for t in tracks
                      if t.verpasst < GEDULD_FRAMES
                      and not (t.gesehen < BESTAETIGUNGEN
                               and t.verpasst > 3 * stride)]
            # Track-Verschmelzung: zwei Tracks auf demselben Griff -> der
            # ältere/öfter gesehene gewinnt
            tracks.sort(key=lambda t: -t.gesehen)
            behalten = []
            for t in tracks:
                if all(_iou(t.bbox, b.bbox) < 0.5 for b in behalten):
                    behalten.append(t)
            tracks = behalten

        for t in tracks:
            if t.poly is not None:
                t.poly_alter += 1

        # 3) SAM-Formen: eifrig für formlose Tracks, und mit Verfallsdatum —
        # ein ewig weitertransformiertes Polygon sammelt Verzerrung an,
        # deshalb wird jede Form nach POLY_MAX_ALTER Frames neu berechnet
        if sam is not None and i % stride == 0:
            faellig = [t for t in tracks
                       if t.gesehen >= BESTAETIGUNGEN and t.verpasst == 0
                       and (t.poly is None or t.poly_alter > POLY_MAX_ALTER)]
            # Volumes: Hintergrund-Subtraktion statt SAM (Halbmond-Problem)
            offene = []
            for t in faellig:
                if t.volume:
                    poly = _volumen_maske(frame, t.bbox)
                    if poly is not None:
                        t.poly = poly
                        t.poly_alter = 0
                else:
                    offene.append(t)
            if offene:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = sam.predict(source=rgb,
                                  bboxes=[t.bbox for t in offene],
                                  verbose=False)
                masks = getattr(res[0], "masks", None)
                if masks is not None:
                    for t, m in zip(offene, masks):
                        mm = (np.asarray(m.data.cpu().numpy()).squeeze() > 0.5)
                        # Maske auf die Prompt-Box (+10 % Rand) stutzen:
                        # kein Übersegeln auf die Wand mehr möglich
                        rx = 0.10 * (t.bbox[2] - t.bbox[0])
                        ry = 0.10 * (t.bbox[3] - t.bbox[1])
                        gx0 = max(0, int(t.bbox[0] - rx))
                        gy0 = max(0, int(t.bbox[1] - ry))
                        gx1 = min(w, int(t.bbox[2] + rx))
                        gy1 = min(h, int(t.bbox[3] + ry))
                        beschnitten = np.zeros_like(mm)
                        beschnitten[gy0:gy1, gx0:gx1] = mm[gy0:gy1, gx0:gx1]
                        mm = beschnitten
                        cnts, _ = cv2.findContours(
                            mm.astype(np.uint8) * 255, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
                        if not cnts:
                            continue
                        c = max(cnts, key=cv2.contourArea)
                        flaeche = cv2.contourArea(c)
                        if flaeche < 9:
                            continue
                        # Konkav-Verdacht (Halbmond-Problem): enthält die
                        # SAM-Maske einen nennenswerten Anteil WANDFARBE,
                        # hat sie die Aushöhlung mitgenommen. Die Wandfarbe
                        # kommt aus dem Boxrand-Ring — unabhängig von der
                        # (dort oft verschmutzten) Track-Farbmessung.
                        bx0, by0 = max(0, int(t.bbox[0])), max(0, int(t.bbox[1]))
                        bx1, by1 = min(w, int(t.bbox[2])), min(h, int(t.bbox[3]))
                        rand_px = np.concatenate([
                            frame[by0:by1, max(0, bx0 - 4):bx0].reshape(-1, 3),
                            frame[by0:by1, bx1:min(w, bx1 + 4)].reshape(-1, 3),
                            frame[max(0, by0 - 4):by0, bx0:bx1].reshape(-1, 3),
                            frame[by1:min(h, by1 + 4), bx0:bx1].reshape(-1, 3),
                        ])
                        if len(rand_px):
                            wand = np.median(rand_px.astype(float), axis=0)
                            innen = frame[mm].reshape(-1, 3).astype(float)
                            if len(innen) > 4000:
                                innen = innen[:: len(innen) // 4000]
                            wand_anteil = float((np.linalg.norm(
                                innen - wand, axis=1) < 60.0).mean())
                            if wand_anteil > 0.30:
                                alt = _volumen_maske(frame, t.bbox)
                                if alt is not None:
                                    t.poly = alt
                                    t.poly_alter = 0
                                    continue
                        # Containment-Gate: eine Maske, die deutlich über
                        # ihre Prompt-Box hinausschwappt (SAM-Fehlgriff auf
                        # unscharfen Frames), wird verworfen und später
                        # neu versucht
                        x, y, bw, bh = cv2.boundingRect(c)
                        rand_x = 0.12 * (t.bbox[2] - t.bbox[0])
                        rand_y = 0.12 * (t.bbox[3] - t.bbox[1])
                        schnitt_x = (max(x, t.bbox[0] - rand_x),
                                     min(x + bw, t.bbox[2] + rand_x))
                        schnitt_y = (max(y, t.bbox[1] - rand_y),
                                     min(y + bh, t.bbox[3] + rand_y))
                        schnitt = (max(0.0, schnitt_x[1] - schnitt_x[0])
                                   * max(0.0, schnitt_y[1] - schnitt_y[0]))
                        if schnitt / max(bw * bh, 1e-6) < 0.80:
                            continue
                        # KEIN Vergleich mit der Track-Farbe mehr: die ist
                        # bei konkaven Volumes selbst wanddominiert und hat
                        # nachweislich gute Masken abgelehnt (Halbmond-Bug).
                        # Der Wand-Test oben misst denselben Fehler direkt.
                        t.poly = c.reshape(-1, 2).astype(np.float64)
                        t.poly_alter = 0
                        # akzeptierte Maske korrigiert die Trackfarbe: Pixel
                        # der Form sind die verlässlichste Farbquelle
                        farbe = extract_color(
                            frame, bbox=t.bbox,
                            polygon=t.poly.astype(int).tolist())
                        t.farbe_bgr = _hex_to_bgr(farbe["hex"])

        # 4) Malen: NUR Formen. Ein Track ohne Form bleibt unsichtbar, bis
        # SAM geliefert hat (Boxen empfand der Nutzer als störendes Springen)
        overlay = frame.copy()
        for t in tracks:
            if t.gesehen < BESTAETIGUNGEN or t.farbe_bgr is None \
                    or t.poly is None:
                continue
            pts = t.poly.astype(np.int32)
            cv2.fillPoly(overlay, [pts], t.farbe_bgr)
            cv2.polylines(frame, [pts], True, t.farbe_bgr, 2, cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, 0.30, frame, 0.70, 0)
        out.write(frame)

        if progress_cb and (i % 15 == 0 or i == n_frames - 1):
            progress_cb((i + 1) / max(1, n_frames))

    cap.release()
    out.release()
    return n_frames, time.time() - t0


def render_video(src, dst, weights=None, conf=0.4, stride=2, shapes=True,
                 keyframe=20, max_seconds=90.0, progress_cb=None,
                 debug_path=None, device=None):
    """Oeffentlicher Einstieg fuer die stabile Zwei-Pass-Pipeline.

    Die bisherige Implementierung bleibt als ``render_video_legacy`` fuer
    reproduzierbare A/B-Vergleiche erhalten.
    """
    from vision.video_overlay_v2 import render_video as render_video_v2

    return render_video_v2(
        src,
        dst,
        weights=weights,
        conf=conf,
        stride=stride,
        shapes=shapes,
        keyframe=keyframe,
        max_seconds=max_seconds,
        progress_cb=progress_cb,
        debug_path=debug_path,
        device=device,
    )
