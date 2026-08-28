"""Gewichte auf getrennten, nicht vermischten Validierungs-Gates messen.

External valid misst hold+volume auf menschlich gelabelten Fremddaten.
Halle valid misst vorerst nur hold (Klasse 0), weil die Hallen-Volumes noch
nicht menschlich korrigiert sind. External test wird erst nach der
Modellauswahl mit --test geoeffnet.
"""

import argparse
import json
import re
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent


def _label(path: Path) -> str:
    parent = path.parent.parent.name if path.parent.name == "weights" else path.parent.name
    value = f"{parent}_{path.stem}" if parent else path.stem
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")


def _summary(metrics) -> dict:
    box = getattr(metrics, "box", None)
    result = {
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
        "per_class": {},
    }
    if box is None:
        return result
    mean_results = list(box.mean_results()) if hasattr(box, "mean_results") else []
    if len(mean_results) >= 4:
        result.update(
            precision=round(float(mean_results[0]), 6),
            recall=round(float(mean_results[1]), 6),
            map50=round(float(mean_results[2]), 6),
            map50_95=round(float(mean_results[3]), 6),
        )
    names = getattr(metrics, "names", {}) or {}
    if hasattr(box, "class_result"):
        for index, name in names.items():
            try:
                values = list(box.class_result(int(index)))
            except (IndexError, KeyError, TypeError):
                continue
            if len(values) >= 4:
                result["per_class"][str(name)] = {
                    "precision": round(float(values[0]), 6),
                    "recall": round(float(values[1]), 6),
                    "map50": round(float(values[2]), 6),
                    "map50_95": round(float(values[3]), 6),
                }
    return result


def main():
    parser = argparse.ArgumentParser(description="Saubere Modellvalidierung")
    parser.add_argument("--weights", required=True, nargs="+")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out", type=Path,
                        help="Vergleich zusaetzlich als JSON schreiben")
    parser.add_argument("--test", action="store_true",
                        help="External test erst fuer das final gewaehlte Modell")
    args = parser.parse_args()

    from ultralytics import YOLO

    common = {
        "device": args.device,
        "batch": args.batch,
        "project": str(API_DIR / "runs" / "validation"),
        "plots": False,
    }
    runs = [
        ("external-valid", API_DIR / "datasets" / "merged_clean" / "data.yaml",
         "val", None),
        ("halle-valid-hold", API_DIR / "datasets" / "eigene_halle_clean" / "data.yaml",
         "val", [0]),
    ]
    if args.test:
        runs.append(("external-test",
                     API_DIR / "datasets" / "merged_clean" / "data.yaml",
                     "test", None))

    comparison = {"test_opened": bool(args.test), "models": []}
    for weight_string in args.weights:
        weight_path = Path(weight_string).resolve()
        weight_label = _label(weight_path)
        model = YOLO(str(weight_path))
        model_result = {"label": weight_label, "weights": str(weight_path), "gates": {}}
        for name, data, split, classes in runs:
            print(f"\n=== {weight_label}: {name} ===")
            run_name = f"{weight_label}_{name}"
            kwargs = dict(common, data=str(data), split=split, name=run_name)
            if classes is not None:
                kwargs["classes"] = classes
            metrics = model.val(**kwargs)
            model_result["gates"][name] = _summary(metrics)
        comparison["models"].append(model_result)

    encoded = json.dumps(comparison, ensure_ascii=False, indent=2)
    print("\n=== Vergleich ===")
    print(encoded)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
