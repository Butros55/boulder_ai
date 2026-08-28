from pathlib import Path

from scripts.baue_halle_dataset import validation_names
from scripts.prepare_dataset import logical_image_key, select_leakage_free


def record(name, split, logical, digest):
    return {
        "image": Path(name),
        "split": split,
        "logical_key": logical,
        "sha256": digest,
    }


def test_logical_image_key_groups_roboflow_augmentations():
    assert logical_image_key("wall_jpg.rf.1234abcd.jpg") == "wall_jpg"


def test_historical_train_group_is_removed_from_validation():
    records = [
        record("train-a.jpg", "train", "wall", "a"),
        record("train-b.jpg", "train", "wall", "b"),
        record("valid.jpg", "valid", "wall", "c"),
    ]
    selected, _, _ = select_leakage_free(records)
    assert {(r["target_split"], r["image"].name) for r in selected} == {
        ("train", "train-a.jpg"),
        ("train", "train-b.jpg"),
    }


def test_exact_duplicate_seen_in_train_is_removed_from_test():
    records = [
        record("train.jpg", "train", "name-a", "same"),
        record("test.jpg", "test", "name-b", "same"),
    ]
    selected, _, _ = select_leakage_free(records)
    assert len(selected) == 1
    assert selected[0]["target_split"] == "train"


def test_train_keeps_unique_augmentations_but_not_exact_copies():
    records = [
        record("a.jpg", "train", "wall", "a"),
        record("b.jpg", "train", "wall", "b"),
        record("a-copy.jpg", "train", "wall", "a"),
    ]
    selected, _, _ = select_leakage_free(records)
    assert {r["sha256"] for r in selected} == {"a", "b"}


def test_hall_validation_selection_is_stable_and_exact_size():
    paths = [Path(f"IMG_{i:04d}.jpg") for i in range(100)]
    first = validation_names(paths, 0.18)
    second = validation_names(reversed(paths), 0.18)
    assert first == second
    assert len(first) == 18
