"""Video-Debugexport gegen den manuell korrigierten Holdout auswerten.

Ohne als ``done`` markierte Frames werden nur Pipeline-Diagnosen ausgegeben;
es werden niemals vermeintliche Ground-Truth-Zahlen erfunden.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from vision.video_masks import polygon_iou
from vision.video_tracking import bbox_iou


def _safe_div(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def infer_clip_id(debug: dict) -> str | None:
    stem = Path(debug.get("source", "")).stem.lower()
    match = re.search(r"(\d{3,4})$", stem)
    return f"mov{int(match.group(1))}" if match else None


def _match_frame(ground_truth: list[dict], predictions: list[dict], threshold=0.5):
    pairs = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, prediction in enumerate(predictions):
            if int(gt["class_id"]) != int(prediction["class_id"]):
                continue
            overlap = bbox_iou(gt["bbox"], prediction["bbox"])
            if overlap >= threshold:
                pairs.append((overlap, gt_index, pred_index))
    pairs.sort(reverse=True)
    used_gt, used_pred, matches = set(), set(), []
    for overlap, gt_index, pred_index in pairs:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_index, pred_index, overlap))
    return matches, used_gt, used_pred


def evaluate(manifest: dict, debug: dict, clip_id: str | None = None) -> dict:
    clip_id = clip_id or infer_clip_id(debug)
    selected = [
        frame
        for frame in manifest.get("frames", [])
        if frame.get("clip_id") == clip_id and frame.get("status") == "done"
    ]
    debug_frames = {int(item["frame"]): item for item in debug.get("frames", [])}
    class_counts = defaultdict(lambda: Counter(tp=0, fp=0, fn=0))
    mask_ious = []
    object_tracks: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for frame in selected:
        ground_truth = frame.get("annotations", [])
        predictions = debug_frames.get(int(frame["frame"]), {}).get("predictions", [])
        matches, used_gt, used_pred = _match_frame(ground_truth, predictions)
        for gt_index, pred_index, _ in matches:
            gt = ground_truth[gt_index]
            prediction = predictions[pred_index]
            class_counts[int(gt["class_id"])]["tp"] += 1
            if gt.get("polygon") and prediction.get("polygon"):
                mask_ious.append(
                    polygon_iou(
                        np.asarray(gt["polygon"], dtype=float),
                        np.asarray(prediction["polygon"], dtype=float),
                    )
                )
            if gt.get("object_id") is not None:
                object_tracks[str(gt["object_id"])].append(
                    (int(frame["frame"]), int(prediction["track_id"]))
                )
        for index, gt in enumerate(ground_truth):
            if index not in used_gt:
                class_counts[int(gt["class_id"])]["fn"] += 1
        for index, prediction in enumerate(predictions):
            if index not in used_pred:
                class_counts[int(prediction["class_id"])]["fp"] += 1

    per_class = {}
    for class_id in sorted(class_counts):
        counts = class_counts[class_id]
        per_class[str(class_id)] = {
            **dict(counts),
            "precision": _safe_div(counts["tp"], counts["tp"] + counts["fp"]),
            "recall": _safe_div(counts["tp"], counts["tp"] + counts["fn"]),
        }

    fragments = 0
    switches = 0
    for assignments in object_tracks.values():
        ordered = [track for _, track in sorted(assignments)]
        fragments += max(0, len(set(ordered)) - 1)
        switches += sum(a != b for a, b in zip(ordered, ordered[1:]))

    candidate_metrics = [
        item.get("metrics", {}) for item in debug.get("mask_candidates", [])
    ]
    temporal = [
        float(item["temporal_iou"])
        for item in candidate_metrics
        if "temporal_iou" in item
    ]
    reject_reasons = Counter(
        str(item["reason"]) for item in candidate_metrics if item.get("reason")
    )
    event_counts = Counter(
        event["event"]
        for frame in debug.get("frames", [])
        for event in frame.get("events", [])
    )
    total_frames = max(1, int(debug.get("video", {}).get("frames", 0)))

    return {
        "clip_id": clip_id,
        "status": "evaluated" if selected else "ground_truth_pending",
        "annotated_frames": len(selected),
        "detection": per_class,
        "mask": {
            "matched_polygons": len(mask_ious),
            "mean_iou": round(float(np.mean(mask_ious)), 4) if mask_ious else None,
            "median_temporal_iou": round(float(np.median(temporal)), 4)
            if temporal
            else None,
            "rejections": dict(reject_reasons),
        },
        "tracking": {
            "objects_with_ids": len(object_tracks),
            "fragments": fragments,
            "id_switches": switches,
            "births_per_100_frames": round(
                100.0 * event_counts["birth"] / total_frames, 3
            ),
            "deaths_per_100_frames": round(
                100.0 * event_counts["death"] / total_frames, 3
            ),
        },
        "pipeline_summary": debug.get("summary", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="Video-Holdout auswerten")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--debug", type=Path, required=True)
    parser.add_argument("--clip-id")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    debug = json.loads(args.debug.read_text(encoding="utf-8"))
    result = evaluate(manifest, debug, args.clip_id)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
