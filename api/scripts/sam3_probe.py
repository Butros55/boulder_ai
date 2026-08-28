"""SAM-3-Schnelltest: Trennt SAM 3 per Textprompt Griffe und Volumes sauber genug,
um YOLO als Detektor zu ersetzen oder als Auto-Labeler zu dienen?

Läuft gegen das lokale sam3-Repo (C:/dev/sam3) mit dem SAM-3.1-Checkpoint.
Pro Bild werden zwei Prompts gefahren ("climbing hold", "climbing volume"),
die Ergebnisse als Overlay-PNG gespeichert und in einer summary.json gezählt.

Aufruf aus api/:
    python scripts/sam3_probe.py --count 3
    python scripts/sam3_probe.py --count 18 --device cpu
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

API_DIR = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES = API_DIR.parent / "app" / "Boulder data"
DEFAULT_OUT = API_DIR / "datasets" / "sam3_probe"
CHECKPOINT = Path("C:/dev/sam3/checkpoints/sam3.1/sam3.1_multiplex.pt")

PROMPTS = ["climbing hold", "climbing volume"]
MAX_SIDE = 1500  # Originale sind 4000er-Kanten; Masken werden auf Bildgröße
                 # interpoliert, ohne Downscale läuft der VRAM voll.


def list_images(img_dir):
    exts = {".heic", ".png", ".jpg", ".jpeg"}
    return sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in exts)


def pick_sample(paths, count):
    """Gleichmäßig über die sortierte Liste verteilen, damit verschiedene
    Wände/Situationen dabei sind, nicht 18 Fotos derselben Route."""
    if count >= len(paths):
        return paths
    idx = np.linspace(0, len(paths) - 1, count).round().astype(int)
    return [paths[i] for i in sorted(set(idx.tolist()))]


def load_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    w, h = img.size
    scale = MAX_SIDE / max(w, h)
    if scale < 1:
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return img


def color_for(i):
    hsv = np.uint8([[[int(i * 47) % 180, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_overlay(img_rgb, masks, boxes, scores):
    """masks: (N,1,H,W) bool, boxes: (N,4) xyxy Pixel, scores: (N,)"""
    canvas = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    overlay = canvas.copy()
    for i in range(len(scores)):
        m = masks[i, 0]
        overlay[m] = color_for(i)
    canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)
    for i in range(len(scores)):
        x0, y0, x1, y1 = boxes[i].round().astype(int)
        c = color_for(i)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), c, 2)
        cv2.putText(canvas, f"{scores[i]:.2f}", (x0, max(y0 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return canvas


def build_sam31_image_detector(device):
    """Bild-Detektor passend zum sam3.1-Multiplex-Checkpoint bauen.

    build_sam3_image_model baut den Dual-Neck mit 4 Leveln (convs.0-3), der
    sam3.1-Detektor hat aber einen TriHead-Neck mit 3 Leveln — beim Laden
    fehlten sonst die convs.3-Gewichte. Aufbau deshalb wie in
    build_sam3_multiplex_video_predictor (model_builder.py), nur ohne Tracker.
    """
    import pkg_resources
    import torch
    from sam3.model_builder import (
        _create_dot_product_scoring,
        _create_geometry_encoder,
        _create_multiplex_tri_backbone,
        _create_sam3_transformer,
        _create_segmentation_head,
        _create_text_encoder,
    )
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.vl_combiner import SAM3VLBackboneTri

    bpe_path = pkg_resources.resource_filename(
        "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
    )
    tri_neck = _create_multiplex_tri_backbone(
        compile_mode=None, use_fa3=False, use_rope_real=False
    )
    backbone = SAM3VLBackboneTri(
        scalp=0, visual=tri_neck, text=_create_text_encoder(bpe_path)
    )
    detector = Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=backbone,
        transformer=_create_sam3_transformer(use_fa3=False),
        segmentation_head=_create_segmentation_head(use_fa3=False),
        semantic_segmentation_head=None,
        input_geometry_encoder=_create_geometry_encoder(),
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=_create_dot_product_scoring(),
        supervise_joint_box_scores=True,
        is_multiplex=True,
    )
    ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    det_ckpt = {
        k[len("detector."):]: v for k, v in ckpt.items()
        if k.startswith("detector.")
    }
    missing, unexpected = detector.load_state_dict(det_ckpt, strict=False)
    print(f"[setup] state_dict: {len(missing)} fehlend, {len(unexpected)} unerwartet")
    if missing:
        print("  fehlend:", missing[:8])
    if unexpected:
        print("  unerwartet:", unexpected[:8])
    return detector.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description="SAM-3-Schnelltest auf Hallenfotos")
    parser.add_argument("--images", default=str(DEFAULT_IMAGES))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--device", default=None, help="cuda, cpu — leer für Auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    from pillow_heif import register_heif_opener
    register_heif_opener()

    import torch
    from sam3.model.sam3_image_processor import Sam3Processor

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = pick_sample(list_images(args.images), args.count)
    print(f"[setup] {len(paths)} Bilder, device={device}, threshold={args.threshold}")

    t0 = time.time()
    model = build_sam31_image_detector(device)
    processor = Sam3Processor(model, device=device,
                              confidence_threshold=args.threshold)
    print(f"[setup] Modell geladen in {time.time() - t0:.0f}s")

    results = []
    for path in paths:
        img = load_image(path)
        entry = {"bild": path.name, "groesse": list(img.size), "device": device}
        try:
            t0 = time.time()
            state = processor.set_image(img)
            for prompt in PROMPTS:
                processor.reset_all_prompts(state)
                state = processor.set_text_prompt(prompt=prompt, state=state)
                masks = state["masks"].cpu().numpy()
                boxes = state["boxes"].cpu().numpy()
                scores = state["scores"].cpu().numpy()
                canvas = draw_overlay(img, masks, boxes, scores)
                tag = prompt.replace(" ", "_")
                out_png = out_dir / f"{path.stem}--{tag}.png"
                cv2.imwrite(str(out_png), canvas,
                            [cv2.IMWRITE_PNG_COMPRESSION, 6])
                entry[prompt] = {
                    "anzahl": int(len(scores)),
                    "scores_min": float(scores.min()) if len(scores) else None,
                    "scores_max": float(scores.max()) if len(scores) else None,
                    "scores_mean": float(scores.mean()) if len(scores) else None,
                }
            entry["sekunden"] = round(time.time() - t0, 1)
            print(f"[ok] {path.name}: "
                  + ", ".join(f"{p}={entry[p]['anzahl']}" for p in PROMPTS)
                  + f" in {entry['sekunden']}s")
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            entry["fehler"] = f"CUDA OOM: {e}"
            print(f"[oom] {path.name}: {e}")
        except Exception as e:  # noqa: BLE001 — Probe-Skript, weiterlaufen und dokumentieren
            entry["fehler"] = f"{type(e).__name__}: {e}"
            print(f"[fehler] {path.name}: {type(e).__name__}: {e}")
        results.append(entry)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nErgebnisse unter {out_dir}")


if __name__ == "__main__":
    main()
