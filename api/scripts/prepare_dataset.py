"""Trainingsdaten von Roboflow Universe ziehen und zu einem Dataset mergen.

Quellen sind öffentliche Kletter-Datasets. Alle Quellklassen werden auf zwei
Zielklassen abgebildet: hold (0) und volume (1) — die Farbe ist keine Klasse
mehr, sie wird zur Laufzeit aus den Pixeln bestimmt.

Voraussetzungen:
    pip install roboflow          (nur zum Herunterladen, nicht für die API)
    ROBOFLOW_API_KEY in .env oder Umgebung (kostenloser Account reicht:
    https://app.roboflow.com -> Settings -> API Key)

Aufruf aus api/:  python scripts/prepare_dataset.py
Ergebnis:         datasets/merged/{train,valid,test}/... + data.yaml
"""

import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

# Öffentliche Datasets auf Roboflow Universe (workspace, project).
# version=None -> neueste Version wird automatisch aufgelöst.
SOURCES = [
    {"workspace": "gdscmoonless", "project": "rock-climbing-hold-detection", "version": None},
    {"workspace": "robworkspace-xfv3r", "project": "climbing-holds-and-volumes-c91sa", "version": None},
    {"workspace": "blackcreed-xpgxh", "project": "climbing-holds-and-volumes", "version": None},
]

TARGET_CLASSES = ["hold", "volume"]
API_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = API_DIR / "datasets" / "raw"
MERGED_DIR = API_DIR / "datasets" / "merged"
SPLITS = ["train", "valid", "test"]


def target_class(source_name):
    """Quellklasse -> Zielklasse. Alles mit 'volume' wird volume, Rest hold."""
    return 1 if "volume" in source_name.lower() else 0


def download_sources(api_key):
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    locations = []
    for src in SOURCES:
        slug = f"{src['workspace']}--{src['project']}"
        dest = RAW_DIR / slug
        if dest.exists() and any(dest.iterdir()):
            print(f"[skip] {slug} liegt schon unter {dest}")
            locations.append((slug, dest))
            continue
        project = rf.workspace(src["workspace"]).project(src["project"])
        version_num = src["version"]
        if version_num is None:
            versions = project.versions()
            if not versions:
                print(f"[warn] {slug}: keine Versionen gefunden, übersprungen")
                continue
            version_num = max(int(v.version.split("/")[-1]) for v in versions)
        print(f"[laden] {slug} v{version_num} ...")
        project.version(version_num).download("yolov8", location=str(dest), overwrite=True)
        locations.append((slug, dest))
    return locations


def merge(locations):
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    for split in SPLITS:
        (MERGED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = Counter()
    for slug, dest in locations:
        data_yaml = dest / "data.yaml"
        if not data_yaml.exists():
            print(f"[warn] {slug}: kein data.yaml, übersprungen")
            continue
        with open(data_yaml, encoding="utf-8") as f:
            names = yaml.safe_load(f)["names"]
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        remap = {i: target_class(n) for i, n in enumerate(names)}
        print(f"[merge] {slug}: {dict(zip(names, [TARGET_CLASSES[remap[i]] for i in range(len(names))]))}")

        for split in SPLITS:
            img_dir = dest / split / "images"
            lbl_dir = dest / split / "labels"
            if not img_dir.exists():
                continue
            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                new_stem = f"{slug}--{img_path.stem}"
                shutil.copy(img_path, MERGED_DIR / split / "images" / f"{new_stem}{img_path.suffix}")
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                lines_out = []
                if lbl_path.exists():
                    for line in lbl_path.read_text(encoding="utf-8").splitlines():
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        cls = remap.get(int(parts[0]), 0)
                        lines_out.append(" ".join([str(cls)] + parts[1:]))
                        stats[f"{split}/{TARGET_CLASSES[cls]}"] += 1
                (MERGED_DIR / split / "labels" / f"{new_stem}.txt").write_text(
                    "\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8"
                )
                stats[f"{split}/bilder"] += 1

    with open(MERGED_DIR / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": str(MERGED_DIR),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "nc": len(TARGET_CLASSES),
                "names": TARGET_CLASSES,
            },
            f,
            sort_keys=False,
        )
    return stats


def main():
    try:
        from dotenv import load_dotenv

        load_dotenv(API_DIR.parent / ".env")
        load_dotenv(API_DIR / ".env")
    except ImportError:
        pass

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit(
            "ROBOFLOW_API_KEY fehlt. Kostenlosen Account anlegen "
            "(https://app.roboflow.com), Key unter Settings -> API Key holen "
            "und in die .env eintragen."
        )
    try:
        import roboflow  # noqa: F401
    except ImportError:
        sys.exit("Das Paket 'roboflow' fehlt: pip install roboflow")

    locations = download_sources(api_key)
    if not locations:
        sys.exit("Keine Quelle ließ sich laden.")
    stats = merge(locations)

    print("\nErgebnis unter", MERGED_DIR)
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print("\nHinweis: Die Quelldatasets stehen unter eigenen Lizenzen auf "
          "Roboflow Universe (i. d. R. CC BY 4.0) — bei Veröffentlichung "
          "des Modells die Quellen nennen.")


if __name__ == "__main__":
    main()
