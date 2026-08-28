"""Train/Valid/Test auf identische oder logisch gleiche Bilder pruefen.

Aufruf aus api/:
    python scripts/audit_dataset_splits.py

Ein Exitcode 1 bedeutet echte Split-Leakage oder fehlende Labeldateien.
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASETS = [
    API_DIR / "datasets" / "merged_clean",
    API_DIR / "datasets" / "eigene_halle_clean",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_key(path, external):
    stem = path.stem
    if external:
        parts = stem.split("--")
        if len(parts) >= 3:
            stem = "--".join(parts[2:])
        stem = re.sub(r"\.rf\.[0-9a-f]+$", "", stem)
    return stem


def audit(dataset):
    external = dataset.name.startswith("merged")
    records = []
    missing_labels = []
    for split in ("train", "valid", "test"):
        image_dir = dataset / split / "images"
        label_dir = dataset / split / "labels"
        if not image_dir.exists():
            continue
        for image in sorted(image_dir.iterdir()):
            if not image.is_file():
                continue
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                missing_labels.append(f"{split}/{image.name}")
            records.append({
                "split": split,
                "name": image.name,
                "logical": logical_key(image, external),
                "sha256": sha256(image),
            })

    leaks = []
    for field in ("logical", "sha256"):
        groups = defaultdict(list)
        for record in records:
            groups[record[field]].append(record)
        for key, members in groups.items():
            splits = sorted({m["split"] for m in members})
            if len(splits) > 1:
                leaks.append({
                    "kind": field,
                    "key": key,
                    "splits": splits,
                    "files": [f"{m['split']}/{m['name']}" for m in members],
                })

    return {
        "dataset": str(dataset),
        "images": len(records),
        "splits": {
            split: sum(r["split"] == split for r in records)
            for split in ("train", "valid", "test")
        },
        "missing_labels": missing_labels,
        "leaks": leaks,
        "ok": not missing_labels and not leaks,
    }


def main():
    parser = argparse.ArgumentParser(description="Dataset-Split-Audit")
    parser.add_argument("datasets", nargs="*", type=Path,
                        default=DEFAULT_DATASETS)
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args()

    reports = [audit(path.resolve()) for path in args.datasets]
    for report in reports:
        state = "OK" if report["ok"] else "FEHLER"
        print(f"[{state}] {report['dataset']}: {report['splits']}, "
              f"leaks={len(report['leaks'])}, "
              f"missing_labels={len(report['missing_labels'])}")
        for leak in report["leaks"][:10]:
            print(f"  {leak['kind']} {leak['splits']}: "
                  + " | ".join(leak["files"]))

    if args.json_path:
        args.json_path.write_text(
            json.dumps(reports, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    raise SystemExit(0 if all(r["ok"] for r in reports) else 1)


if __name__ == "__main__":
    main()
