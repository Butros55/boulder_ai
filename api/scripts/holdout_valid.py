"""Diverse Trainingsbilder als festes Validierungs-Holdout reservieren.

Verschiebt ausgewählte Bilder samt Labels von train/ nach valid/ im gemergten
Dataset, damit das Modell sie nie im Training sieht und sie als konstante
Testfälle dienen. Die Auswahl wird beim ersten Lauf getroffen (gestreut über
Quelle, Instanzdichte und Volume-Anteil) und in holdout_stems.txt neben diesem
Skript festgeschrieben — nach jedem neuen prepare_dataset.py-Lauf stellt ein
erneuter Aufruf dieselbe Aufteilung wieder her.

Aufruf aus api/:  python scripts/holdout_valid.py [--count 20]
"""

import argparse
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent
MERGED = API_DIR / "datasets" / "merged"
STEMS_FILE = Path(__file__).resolve().parent / "holdout_stems.txt"


def instance_stats(lbl_path):
    n_hold = n_vol = 0
    if lbl_path.exists():
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                if parts[0] == "1":
                    n_vol += 1
                else:
                    n_hold += 1
    return n_hold, n_vol


def select_stems(count):
    """Streuung: je Quelle über Instanzdichte-Quantile, Volume-Bilder bevorzugt.

    Bilder mit weniger als 8 Instanzen fallen raus — das sind in den Quellen
    fast immer Katalog-/Produktfotos (Griff-Sets auf weißem Grund), keine
    echten Wandszenen, und als Testfälle wenig aussagekräftig.
    """
    train_imgs = sorted((MERGED / "train" / "images").iterdir())
    by_source = {}
    for img in train_imgs:
        source = img.stem.split("--")[0]
        n_hold, n_vol = instance_stats(MERGED / "train" / "labels" / f"{img.stem}.txt")
        if n_hold + n_vol < 8:
            continue
        by_source.setdefault(source, []).append((img.stem, n_hold, n_vol))

    per_source = max(1, count // len(by_source))
    chosen = []
    for source, entries in sorted(by_source.items()):
        entries.sort(key=lambda e: e[1] + e[2])
        with_vol = [e for e in entries if e[2] > 0]
        without_vol = [e for e in entries if e[2] == 0]
        picks = []
        # Hälfte mit Volumes (über deren Dichte-Spektrum verteilt), Hälfte ohne
        for pool, share in ((with_vol, per_source - per_source // 2),
                            (without_vol, per_source // 2)):
            if not pool:
                continue
            step = max(1, len(pool) // max(1, share))
            picks.extend(pool[i] for i in range(0, len(pool), step)[:share])
        chosen.extend(p[0] for p in picks[:per_source])
    return chosen[:count]


def main():
    parser = argparse.ArgumentParser(description="Holdout train -> valid")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    if STEMS_FILE.exists():
        stems = [s for s in STEMS_FILE.read_text(encoding="utf-8").split() if s]
        print(f"[holdout] {len(stems)} Bilder aus {STEMS_FILE.name}")
    else:
        stems = select_stems(args.count)
        STEMS_FILE.write_text("\n".join(stems) + "\n", encoding="utf-8")
        print(f"[holdout] {len(stems)} Bilder neu ausgewählt und in "
              f"{STEMS_FILE.name} festgeschrieben")

    moved = missing = 0
    for stem in stems:
        img = next((MERGED / "train" / "images").glob(stem + ".*"), None)
        if img is None:
            if next((MERGED / "valid" / "images").glob(stem + ".*"), None):
                continue  # schon verschoben
            missing += 1
            print(f"[warn] {stem}: weder in train noch valid gefunden")
            continue
        lbl = MERGED / "train" / "labels" / f"{stem}.txt"
        img.rename(MERGED / "valid" / "images" / img.name)
        if lbl.exists():
            lbl.rename(MERGED / "valid" / "labels" / lbl.name)
        moved += 1

    n_train = len(list((MERGED / "train" / "images").iterdir()))
    n_valid = len(list((MERGED / "valid" / "images").iterdir()))
    print(f"[holdout] {moved} verschoben, {missing} fehlend — "
          f"train: {n_train}, valid: {n_valid}")


if __name__ == "__main__":
    main()
