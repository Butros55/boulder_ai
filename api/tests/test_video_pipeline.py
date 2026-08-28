import cv2
import numpy as np

from scripts.build_video_holdout import sample_indices
from scripts.evaluate_video_holdout import evaluate
from vision.video_masks import (
    candidate_from_mask,
    polygon_iou,
    volume_mask_candidates,
)
from vision.video_overlay_v2 import (
    _candidate_plan,
    _select_track_masks,
    _smooth_residuals,
)
from vision.video_tracking import (
    Detection,
    MotionEstimate,
    Observation,
    Track,
    TrackManager,
    estimate_camera_motion,
    transform_bbox,
)


def test_bbox_transform_uses_all_four_corners():
    angle = np.radians(30)
    matrix = np.array(
        [[np.cos(angle), -np.sin(angle), 7.0],
         [np.sin(angle), np.cos(angle), -3.0]],
        dtype=np.float64,
    )
    result = transform_bbox([10, 20, 50, 70], matrix)
    corners = np.array([[10, 20], [50, 20], [50, 70], [10, 70]])
    moved = corners @ matrix[:, :2].T + matrix[:, 2]
    expected = [
        moved[:, 0].min(), moved[:, 1].min(),
        moved[:, 0].max(), moved[:, 1].max(),
    ]
    assert np.allclose(result, expected)


def test_motion_estimator_recovers_translation():
    previous = np.zeros((180, 240), dtype=np.uint8)
    rng = np.random.default_rng(4)
    for x, y in rng.integers([15, 15], [225, 165], size=(80, 2)):
        cv2.circle(previous, (int(x), int(y)), 2, 255, -1)
    expected = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]])
    current = cv2.warpAffine(previous, expected, (240, 180))
    motion = estimate_camera_motion(previous, current, full_width=240)
    assert motion.method == "affine"
    assert motion.quality > 0.7
    assert np.allclose(motion.matrix[:, 2], [5.0, -3.0], atol=0.7)


def test_motion_estimator_marks_clear_scene_cut():
    previous = np.zeros((120, 160), dtype=np.uint8)
    current = np.full((120, 160), 255, dtype=np.uint8)
    motion = estimate_camera_motion(previous, current, full_width=160)
    assert motion.scene_cut
    assert motion.method == "scene_cut"


def test_track_class_is_voted_over_multiple_observations():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    manager = TrackManager(patience=5)
    manager.update(0, frame, [Detection((20, 20, 60, 60), 1, 0.55)])
    manager.update(1, frame, [Detection((21, 20, 61, 60), 0, 0.90)])
    manager.update(2, frame, [Detection((20, 21, 60, 61), 0, 0.88)])
    assert len(manager.active) == 1
    assert manager.active[0].class_id == 0
    assert manager.active[0].hits == 3


def test_scene_cut_finishes_active_tracks():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    manager = TrackManager(patience=5)
    manager.update(0, frame, [Detection((20, 20, 60, 60), 0, 0.9)])
    manager.update(1, frame, [Detection((20, 20, 60, 60), 0, 0.9)])
    events = manager.move(MotionEstimate(scene_cut=True), frame=2)
    assert not manager.active
    assert len(manager.finished) == 1
    assert manager.finished[0].ended_frame == 1
    assert any(event["event"] == "scene_end" for event in events)


def test_box_shaped_wall_mask_is_rejected():
    frame = np.full((120, 120, 3), 230, dtype=np.uint8)
    bbox = [25, 25, 95, 95]
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[25:95, 25:95] = 255
    candidate = candidate_from_mask(
        frame,
        bbox,
        mask,
        class_id=1,
        source="synthetic",
        confidence=0.9,
        sharpness=100.0,
    )
    assert not candidate.accepted
    assert candidate.metrics["reason"] in {"too_large_for_box", "box_shaped", "wall_dominated"}


def test_concave_volume_mask_is_accepted():
    frame = np.full((140, 140, 3), 235, dtype=np.uint8)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (70, 70), (38, 45), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (82, 70), (28, 34), 0, 0, 360, 0, -1)
    frame[mask > 0] = (25, 25, 25)
    candidate = candidate_from_mask(
        frame,
        [30, 20, 108, 120],
        mask,
        class_id=1,
        source="synthetic_crescent",
        confidence=0.9,
        sharpness=180.0,
    )
    assert candidate.accepted, candidate.metrics
    assert candidate.polygon is not None
    assert candidate.metrics["area_box_ratio"] < 0.92


def test_sam_mask_that_fills_crescent_hole_is_rejected():
    frame = np.full((140, 140, 3), 235, dtype=np.uint8)
    object_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    filled_mask = np.zeros_like(object_mask)
    cv2.ellipse(object_mask, (70, 70), (38, 45), 0, 0, 360, 255, -1)
    cv2.ellipse(filled_mask, (70, 70), (38, 45), 0, 0, 360, 255, -1)
    cv2.ellipse(object_mask, (82, 70), (28, 34), 0, 0, 360, 0, -1)
    frame[object_mask > 0] = (25, 25, 25)
    candidate = candidate_from_mask(
        frame,
        [30, 20, 108, 120],
        filled_mask,
        class_id=1,
        source="sam",
        confidence=0.9,
        sharpness=180.0,
    )
    assert not candidate.accepted
    assert candidate.metrics["reason"] == "wall_dominated"


def test_volume_background_candidates_find_dark_crescent():
    frame = np.full((160, 180, 3), (230, 232, 228), dtype=np.uint8)
    object_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.ellipse(object_mask, (90, 80), (44, 52), 0, 0, 360, 255, -1)
    cv2.ellipse(object_mask, (104, 80), (33, 40), 0, 0, 360, 0, -1)
    frame[object_mask > 0] = (18, 20, 22)
    candidates = volume_mask_candidates(
        frame, [42, 24, 135, 137], confidence=0.92, sharpness=160.0
    )
    accepted = [candidate for candidate in candidates if candidate.accepted]
    assert accepted
    assert max(candidate.metrics["area_box_ratio"] for candidate in accepted) < 0.92


def test_temporal_polygon_iou_accounts_for_camera_motion():
    polygon = np.array([[10, 10], [40, 10], [40, 35], [10, 35]], dtype=float)
    moved = polygon + np.array([8.0, -4.0])
    transform = np.array([[1, 0, 8], [0, 1, -4]], dtype=float)
    assert polygon_iou(polygon, moved, first_to_second=transform) > 0.99


def test_keyframe_controls_mask_candidate_spacing(monkeypatch):
    monkeypatch.setenv("VIDEO_MASK_CANDIDATES", "3")
    track = Track(1, [0, 0, 20, 20], 0, 0, 60)
    for frame in range(0, 61, 2):
        track.observations.append(
            Observation(frame, (0, 0, 20, 20), 0, 0.8, 100 + frame)
        )
    plan = _candidate_plan([track], keyframe=20)
    selected = sorted(plan)
    assert len(selected) == 3
    assert all(right - left >= 10 for left, right in zip(selected, selected[1:]))


def test_track_without_accepted_mask_keeps_bbox_fallback():
    track = Track(7, [10, 10, 30, 30], 0, 0, 20)
    track.hits = 5
    for frame in range(0, 21, 2):
        track.observations.append(
            Observation(frame, (10, 10, 30, 30), 0, 0.8, 120.0)
        )
    poses = [np.eye(3) for _ in range(21)]
    segments = [0] * 21
    _select_track_masks([track], {}, poses, segments)
    assert track.world_polygon is None
    assert track.fallback_box_world is not None
    assert np.allclose(track.fallback_box_world.min(axis=0), [10, 10], atol=1e-6)
    assert np.allclose(track.fallback_box_world.max(axis=0), [30, 30], atol=1e-6)
    assert track.alignment_residuals


def test_local_alignment_filters_detector_jitter():
    raw = [(index, 8.0 if index % 2 else -8.0, 4.0 if index % 2 else -4.0)
           for index in range(11)]
    smoothed = _smooth_residuals(raw)
    raw_x = np.asarray([item[1] for item in raw])
    smooth_x = np.asarray([item[1] for item in smoothed])
    assert smooth_x.std() < raw_x.std() * 0.65


def test_holdout_sampling_keeps_hardcases_and_is_reproducible():
    first = sample_indices(376, 18, [100, 192, 210, 234, 240])
    second = sample_indices(376, 18, [100, 192, 210, 234, 240])
    assert first == second
    assert {100, 192, 210, 234, 240}.issubset(first)
    assert first[0] == 0
    assert first[-1] == 375


def test_holdout_evaluator_reports_real_precision_and_recall():
    manifest = {
        "frames": [
            {
                "clip_id": "mov890",
                "frame": 100,
                "status": "done",
                "annotations": [
                    {
                        "object_id": "halfmoon",
                        "class_id": 1,
                        "bbox": [10, 10, 40, 40],
                        "polygon": [[10, 10], [40, 10], [40, 40], [10, 40]],
                    }
                ],
            }
        ]
    }
    debug = {
        "source": "IMG_0890.MOV",
        "video": {"frames": 120},
        "summary": {"tracks_with_mask": 1},
        "mask_candidates": [],
        "frames": [
            {
                "frame": 100,
                "events": [],
                "predictions": [
                    {
                        "track_id": 7,
                        "class_id": 1,
                        "bbox": [10, 10, 40, 40],
                        "polygon": [[10, 10], [40, 10], [40, 40], [10, 40]],
                    }
                ],
            }
        ],
    }
    result = evaluate(manifest, debug)
    assert result["status"] == "evaluated"
    assert result["detection"]["1"]["precision"] == 1.0
    assert result["detection"]["1"]["recall"] == 1.0
    assert result["mask"]["mean_iou"] == 1.0
    assert result["tracking"]["id_switches"] == 0
