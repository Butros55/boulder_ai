"""Trainingsdaten von Roboflow Universe ziehen und zu einem Dataset mergen.

Quellen sind öffentliche Kletter-Datasets. Alle Quellklassen werden auf zwei
Zielklassen abgebildet: hold (0) und volume (1) — die Farbe ist keine Klasse
mehr, sie wird zur Laufzeit aus den Pixeln bestimmt.

Voraussetzungen:
    pip install roboflow          (nur zum Herunterladen, nicht für die API)
    ROBOFLOW_API_KEY in .env oder Umgebung (kostenloser Account reicht:
    https://app.roboflow.com -> Settings -> API Key)

Aufruf aus api/:  python scripts/prepare_dataset.py
Ergebnis:         datasets/merged_clean/{train,valid,test}/... + data.yaml

Der Merge gruppiert alle Roboflow-Augmentierungen eines Ursprungsbilds und
byte-identische Dateien, bevor er sie einem Split zuordnet. So kann kein
Original gleichzeitig in Training und Validierung/Test landen.
"""

import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

# Öffentliche Datasets auf Roboflow Universe (workspace, project).
# version=None -> neueste Version wird automatisch aufgelöst.
# class_map: explizites Mapping Quellklassen-Index -> Zielklasse, für Quellen,
# deren data.yaml nur numerische Klassennamen trägt ('0', '1') — dort greift
# die namensbasierte Zuordnung nicht. Beide "holds and volumes"-Quellen sind
# geprüft (25.08.2026): Klasse 1 ist ~8x größer und ~13x seltener als Klasse 0
# und visuell eindeutig das Volume.
# Audit vom 26.08.2026: robworkspace v1 und blackcreed v14 enthalten exakt
# dieselben 1.716 Ursprungsbilder. blackcreed bringt zusaetzlich die
# Trainings-Augmentierungen mit. Beide zu mergen verdoppelte den Datensatz und
# erzeugte Train/Valid-Leakage. gdscmoonless hat keine herunterladbare Version
# und ist ebenfalls ein Re-Upload derselben Daten. Deshalb bleibt genau eine
# kanonische Quelle aktiv.
SOURCES = [
    {"workspace": "blackcreed-xpgxh", "project": "climbing-holds-and-volumes",
     "version": 14, "class_map": {0: 0, 1: 1}},
]

TARGET_CLASSES = ["hold", "volume"]
API_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = API_DIR / "datasets" / "raw"
MERGED_DIR = API_DIR / "datasets" / "merged_clean"
SPLITS = ["train", "valid", "test"]
SPLIT_PRIORITY = {"valid": 0, "test": 1}


def logical_image_key(path):
    """Roboflow-Augmentierungen auf das gemeinsame Ursprungsbild abbilden."""
    return re.sub(r"\.rf\.[0-9a-f]+$", "", Path(path).stem)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def select_leakage_free(records):
    """Gruppiert Ursprungsbilder und byte-identische Dateien splitfest.

    Train behaelt unterschiedliche Augmentierungen eines Ursprungsbilds.
    Valid/Test behalten je Ursprungsgruppe genau ein Bild, damit die Metrik
    nicht durch Augmentierungs-Geschwister kuenstlich gewichtet wird. Taucht
    eine Gruppe historisch im Training auf, bleibt sie dort und wird aus den
    Evaluationssplits entfernt; sonst gewinnt test vor valid. Dadurch sind
    auch bereits trainierte Altgewichte gegen den neuen Eval-Split pruefbar.
    """
    uf = _UnionFind(len(records))
    by_key = {}
    by_sha = {}
    for i, record in enumerate(records):
        for lookup, value in ((by_key, record["logical_key"]),
                              (by_sha, record["sha256"])):
            if value in lookup:
                uf.union(i, lookup[value])
            else:
                lookup[value] = i

    groups = {}
    for i, record in enumerate(records):
        groups.setdefault(uf.find(i), []).append(record)

    selected = []
    group_stats = Counter()
    for members in groups.values():
        member_splits = {m["split"] for m in members}
        target = ("train" if "train" in member_splits
                  else max(member_splits, key=lambda s: SPLIT_PRIORITY[s]))
        candidates = sorted(
            (m for m in members if m["split"] == target),
            key=lambda m: str(m["image"]),
        )
        if target == "train":
            # Augmentierungen duerfen im Training bleiben, exakte Kopien nicht.
            seen_hashes = set()
            selected_in_group = 0
            for candidate in candidates:
                if candidate["sha256"] not in seen_hashes:
                    candidate["target_split"] = target
                    selected.append(candidate)
                    selected_in_group += 1
                    seen_hashes.add(candidate["sha256"])
        else:
            candidate = candidates[0]
            candidate["target_split"] = target
            selected.append(candidate)
            selected_in_group = 1
        group_stats[f"groups/{target}"] += 1
        group_stats["discarded_copies"] += len(members) - selected_in_group
    return selected, groups, group_stats


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
            locations.append((slug, dest, src.get("class_map")))
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
        locations.append((slug, dest, src.get("class_map")))
    return locations


def merge(locations):
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    for split in SPLITS:
        (MERGED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    records = []
    source_descriptions = []
    for slug, dest, class_map in locations:
        data_yaml = dest / "data.yaml"
        if not data_yaml.exists():
            print(f"[warn] {slug}: kein data.yaml, übersprungen")
            continue
        with open(data_yaml, encoding="utf-8") as f:
            names = yaml.safe_load(f)["names"]
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        if class_map is not None:
            remap = {i: class_map.get(i, 0) for i in range(len(names))}
        else:
            remap = {i: target_class(n) for i, n in enumerate(names)}
        print(f"[merge] {slug}: {dict(zip(names, [TARGET_CLASSES[remap[i]] for i in range(len(names))]))}")
        source_descriptions.append({"slug": slug, "path": str(dest),
                                    "class_map": remap})

        for split in SPLITS:
            img_dir = dest / split / "images"
            lbl_dir = dest / split / "labels"
            if not img_dir.exists():
                continue
            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                records.append({
                    "slug": slug,
                    "image": img_path,
                    "label": lbl_path,
                    "remap": remap,
                    "split": split,
                    "logical_key": logical_image_key(img_path),
                    "sha256": file_sha256(img_path),
                })

    selected, groups, group_stats = select_leakage_free(records)
    stats = Counter(group_stats)
    manifest_entries = []
    for record in selected:
        split = record["target_split"]
        img_path = record["image"]
        new_stem = f"{record['slug']}--{img_path.stem}"
        shutil.copy2(
            img_path,
            MERGED_DIR / split / "images" / f"{new_stem}{img_path.suffix.lower()}",
        )

        lines_out = []
        if record["label"].exists():
            for line in record["label"].read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = record["remap"].get(int(parts[0]), 0)
                lines_out.append(" ".join([str(cls)] + parts[1:]))
                stats[f"{split}/{TARGET_CLASSES[cls]}"] += 1
        (MERGED_DIR / split / "labels" / f"{new_stem}.txt").write_text(
            "\n".join(lines_out) + ("\n" if lines_out else ""),
            encoding="utf-8",
        )
        stats[f"{split}/bilder"] += 1
        manifest_entries.append({
            "split": split,
            "logical_key": record["logical_key"],
            "sha256": record["sha256"],
            "source": str(img_path),
            "output": f"{split}/images/{new_stem}{img_path.suffix.lower()}",
        })

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
    manifest = {
        "schema": 1,
        "policy": {
            "group_by": ["roboflow_original_name", "sha256"],
            "split_conflict": (
                "historical train wins over eval; otherwise test wins over valid"
            ),
            "train": "unique augmentations of one original stay together",
            "valid_test": "one representative per original group",
        },
        "sources": source_descriptions,
        "raw_images": len(records),
        "source_groups": len(groups),
        "selected_images": len(selected),
        "stats": dict(stats),
        "entries": manifest_entries,
    }
    (MERGED_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
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
