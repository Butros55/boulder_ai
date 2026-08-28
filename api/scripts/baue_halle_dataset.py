"""Feintuning-Dataset "eigene Halle" aus den wiederhergestellten Annotationen.

Quellen:
- alt_colors_v2 (101 Bilder, per Konstellations-Matching den Originalfotos
  zugeordnet) und alt_colors_v1 (22 Bilder aus dem Label-Studio-Upload):
  jede Farb-Box wird zur Klasse **hold** (Farben sind keine Klassen mehr).
- Volumes waren damals nicht gelabelt. Damit das Feintuning die Volume-
  Klasse nicht verlernt (sichtbare, aber ungelabelte Volumes wirken wie
  Negativbeispiele), ergänzt --pseudo-volumes Volume-Boxen aus dem
  aktuellen Modell (hohe Konfidenz, ohne Überlappung mit hold-Labels).

Aufruf aus api/:
    python scripts/baue_halle_dataset.py --pseudo-volumes
Ergebnis: datasets/eigene_halle_clean/{train,valid}/... + data.yaml

Die beiden Alt-Exports werden ueber ihre Matching-Dateien zuerst auf das
kanonische Originalfoto abgebildet. Valid enthaelt nur menschliche hold-Labels;
Pseudo-Volumes werden ausschliesslich dem Training hinzugefuegt.
"""

import argparse
import json
import shutil
import urllib.parse
from pathlib import Path
from hashlib import sha256

API_DIR = Path(__file__).resolve().parent.parent
# Dauerhafte Ablage der wiederhergestellten Label-Studio-Exports; die
# urspruengliche Quelle war ein Session-Scratchpad und kann verschwinden.
EXPORTS = API_DIR / "datasets" / "alt_exports"
OUT = API_DIR / "datasets" / "eigene_halle_clean"


def _ref_name(ref):
    decoded = urllib.parse.unquote(ref).replace("%5C", "/").replace("\\", "/")
    return decoded.rsplit("/", 1)[-1]


def _saubere_box(cx, cy, w, h, min_seite=0.002):
    """Alt-Boxen ans Bild klemmen; ausserhalb liegende Splitter verwerfen.

    Die Prozentkoordinaten der Alt-Annotationen koennen ueber den Bildrand
    hinausragen (belegt: IMG_0768 mit x-Zentrum 1,67) - YOLO wirft solche
    Bilder sonst komplett aus der Validierung.
    """
    x0, y0 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x1, y1 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    if x1 - x0 < min_seite or y1 - y0 < min_seite:
        return None
    return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def _source_boxes(export_name, mapping_name):
    """Kanonischer Originalpfad -> menschlich annotierte hold-Boxen."""
    mapping = json.loads(
        (API_DIR / "datasets" / mapping_name / "mapping.json").read_text(
            encoding="utf-8"
        )
    )
    jpg_dir = API_DIR / "datasets" / "eigene_halle_jpg"
    data = json.loads((EXPORTS / export_name).read_text(encoding="utf-8"))
    bilder = {}
    for task in data:
        ref = (task.get("data", {}).get("image_url")
               or task.get("data", {}).get("image", ""))
        match = mapping.get(_ref_name(ref))
        if not match:
            continue
        pfad = jpg_dir / match["bild"]
        if not pfad.exists():
            continue
        holds = bilder.setdefault(pfad, [])
        for annotation in task.get("annotations", []):
            if annotation.get("was_cancelled"):
                continue
            for result in annotation.get("result", []):
                value = result.get("value", {})
                if "rectanglelabels" not in value:
                    continue
                x, y = value["x"], value["y"]
                w, h = value["width"], value["height"]
                holds.append(((x + w / 2) / 100.0,
                              (y + h / 2) / 100.0,
                              w / 100.0, h / 100.0))
    return bilder


def sammle():
    """Annotationsversionen auf kanonische Originalfotos zusammenfuehren.

    v2 ist die neuere, vollstaendigere Annotation und gewinnt beim einzigen
    Ueberlappungsbild (IMG_0775). Dadurch kann dessen komprimierte v1-Kopie
    nicht mehr in einem anderen Split landen.
    """
    v2 = _source_boxes("alt_colors_v2.json", "alt_match_v2")
    v1 = _source_boxes("alt_colors_v1.json", "alt_match_v1")
    bilder = dict(v2)
    overlaps = 0
    for pfad, holds in v1.items():
        if pfad in bilder:
            overlaps += 1
            continue
        bilder[pfad] = holds
    return bilder, overlaps


def validation_names(paths, fraction):
    """Exakt reproduzierbare Anzahl kanonischer Bilder reservieren."""
    ranked = sorted(
        (p.name for p in paths),
        key=lambda name: sha256(f"boulder-halle-v1:{name}".encode()).hexdigest(),
    )
    count = max(1, min(len(ranked) - 1, round(len(ranked) * fraction)))
    return set(ranked[:count])


def reused_pseudo_volumes(dataset_dir):
    """Vorhandene class-1-Pseudo-Labels auf kanonische Bildnamen abbilden."""
    inverse = {}
    for mapping_name in ("alt_match_v2", "alt_match_v1"):
        mapping = json.loads(
            (API_DIR / "datasets" / mapping_name / "mapping.json").read_text(
                encoding="utf-8"
            )
        )
        for old_name, match in mapping.items():
            inverse[old_name] = match["bild"]

    result = {}
    for split in ("train", "valid"):
        image_dir = dataset_dir / split / "images"
        label_dir = dataset_dir / split / "labels"
        if not image_dir.exists():
            continue
        for image in image_dir.iterdir():
            canonical = inverse.get(image.name, image.name)
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                continue
            lines = [line for line in label.read_text(encoding="utf-8").splitlines()
                     if line.split() and line.split()[0] == "1"]
            if lines:
                # v2/originale Dateinamen sind die bevorzugte Quelle.
                if canonical not in result or image.name == canonical:
                    result[canonical] = lines
    return result


def main():
    parser = argparse.ArgumentParser(description="Eigene-Halle-Dataset")
    parser.add_argument("--pseudo-volumes", action="store_true")
    parser.add_argument(
        "--reuse-pseudo-from", type=Path,
        help="Vorhandene class-1-Pseudo-Labels wiederverwenden (kein Modelllauf)",
    )
    parser.add_argument("--weights", default=str(API_DIR / "weights" / "best.pt"))
    parser.add_argument("--device", default="cpu",
                        help="Device fuer optionale Pseudo-Volumes (Standard: cpu)")
    parser.add_argument("--val-anteil", type=float, default=0.18)
    args = parser.parse_args()
    if args.pseudo_volumes and args.reuse_pseudo_from:
        parser.error("--pseudo-volumes und --reuse-pseudo-from schliessen sich aus")

    bilder, overlaps = sammle()
    val_names = validation_names(bilder, args.val_anteil)
    print(f"[daten] {len(bilder)} kanonische Bilder mit hold-Labels, "
          f"{len(val_names)} valid, {overlaps} Alt-Duplikat(e) entfernt")

    model = None
    reused = {}
    if args.pseudo_volumes:
        from ultralytics import YOLO
        model = YOLO(args.weights)
    elif args.reuse_pseudo_from:
        reused = reused_pseudo_volumes(args.reuse_pseudo_from.resolve())
        print(f"[pseudo] vorhandene Labels fuer {len(reused)} Bilder geladen")

    import yaml
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "valid"):
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {"train": {"images": 0, "hold": 0, "pseudo_volume": 0},
             "valid": {"images": 0, "hold": 0, "pseudo_volume": 0}}
    verworfene_boxen = 0
    for pfad, holds in sorted(bilder.items()):
        split = "valid" if pfad.name in val_names else "train"
        bereinigt = [box for box in (_saubere_box(*roh) for roh in holds) if box]
        verworfene_boxen += len(holds) - len(bereinigt)
        holds = bereinigt
        zeilen = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                  for cx, cy, w, h in holds]
        stats[split]["hold"] += len(holds)

        # Pseudo-Labels duerfen nur Trainingsdaten ergaenzen. In der
        # Validierung waere das lediglich ein Vergleich des Modells mit
        # seinen eigenen Vorhersagen und keine echte Ground Truth.
        if model is not None and split == "train":
            r = model.predict(source=str(pfad), conf=0.6, verbose=False,
                              device=args.device)[0]
            H, W = r.orig_shape
            for b in r.boxes:
                if int(b.cls[0]) != 1:
                    continue
                x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
                cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
                bw, bh = (x1 - x0) / W, (y1 - y0) / H
                # nicht über gelabelte holds stülpen
                if any(abs(cx - hx) < (bw + hw) / 2 * 0.5
                       and abs(cy - hy) < (bh + hh) / 2 * 0.5
                       and hw * hh > 0.4 * bw * bh
                       for hx, hy, hw, hh in holds):
                    continue
                zeilen.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                stats[split]["pseudo_volume"] += 1
        elif split == "train" and pfad.name in reused:
            zeilen.extend(reused[pfad.name])
            stats[split]["pseudo_volume"] += len(reused[pfad.name])

        ziel = OUT / split / "images" / pfad.name
        shutil.copy2(pfad, ziel)
        (OUT / split / "labels" / f"{pfad.stem}.txt").write_text(
            "\n".join(zeilen) + "\n", encoding="utf-8")
        stats[split]["images"] += 1

    if verworfene_boxen:
        print(f"[label] {verworfene_boxen} Alt-Box(en) ohne Bildueberlappung verworfen")
    with open(OUT / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": str(OUT.resolve()), "train": "train/images",
                        "val": "valid/images", "nc": 2,
                        "names": ["hold", "volume"]}, f, sort_keys=False)
    manifest = {
        "schema": 1,
        "canonical_images": len(bilder),
        "removed_cross_version_duplicates": overlaps,
        "split_seed": "boulder-halle-v1",
        "validation_fraction": args.val_anteil,
        "validation_images": sorted(val_names),
        "validation_scope": "human hold labels only; volume ground truth pending",
        "pseudo_volumes": (
            "train only, model generated" if model is not None
            else "train only, reused" if reused else "disabled"
        ),
        "stats": stats,
    }
    (OUT / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[fertig] {json.dumps(stats, ensure_ascii=False)} -> {OUT}")


if __name__ == "__main__":
    main()
