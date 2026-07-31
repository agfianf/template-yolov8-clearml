"""The train/valid/test split is a partition -- no pair is silently discarded.

Every test here is a regression for issue #19. Before it, `split_folder_yolo`
built three independent `int()` slices, so pairs could fall outside all of them
and be deleted with the staging directories, and `test_ratio` was a truthiness
switch whose numeric value was never read as a size. The worst case combined a
`test_ratio` with a dedicated CVAT test set: the ratio withheld images from
training and the dedicated set then overwrote the directory holding them.

Pairs are written directly rather than converted from COCO -- these assert
arithmetic on file counts, and a real annotation tree would only slow that down.
"""

import logging

from pathlib import Path

import pytest

from src.data.setup import setup_dataset, split_folder_yolo


def _make_pairs(root: Path, n: int, prefix: str) -> None:
    """Write `n` image/label pairs under `root/images` and `root/labels`."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (root / "images" / f"{prefix}_{i:04d}.jpg").write_text("x")
        (root / "labels" / f"{prefix}_{i:04d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")


def _counts(dataset_dir: Path) -> dict[str, int]:
    out = {}
    for split in ("train", "valid", "test"):
        images = dataset_dir / split / "images"
        out[split] = len(list(images.iterdir())) if images.is_dir() else 0
    return out


# --------------------------------------------------------------------------
# Nothing is lost
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "ratios"),
    [
        # 1 - 0.8 is 0.19999999999999996, so the old int(n * (1 - train_ratio))
        # dropped the last pair of every dataset whose size ended in 0.
        (100, (0.8, 0.2, None)),
        (10, (0.8, 0.2, None)),
        (100, (0.7, 0.2, 0.1)),
        (13, (0.7, 0.2, 0.1)),
        (7, (0.6, 0.2, 0.2)),
    ],
)
def test_split_is_a_partition(tmp_path: Path, n: int, ratios: tuple) -> None:
    dataset_dir = tmp_path / "dataset-yolov8"
    _make_pairs(dataset_dir, n, "pool")
    train_ratio, valid_ratio, test_ratio = ratios

    split_folder_yolo(
        source_dir=str(dataset_dir),
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )

    counts = _counts(dataset_dir)
    assert sum(counts.values()) == n, counts
    # Labels travel with their image, so the label tree partitions identically.
    for split, count in counts.items():
        labels = dataset_dir / split / "labels"
        assert (len(list(labels.iterdir())) if labels.is_dir() else 0) == count


def test_remainder_goes_to_test_when_there_is_one(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset-yolov8"
    _make_pairs(dataset_dir, 100, "pool")

    split_folder_yolo(
        source_dir=str(dataset_dir),
        train_ratio=0.7,
        valid_ratio=0.2,
        test_ratio=0.1,
    )

    assert _counts(dataset_dir) == {"train": 70, "valid": 20, "test": 10}


def test_remainder_goes_to_valid_without_a_test_split(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset-yolov8"
    _make_pairs(dataset_dir, 100, "pool")

    split_folder_yolo(source_dir=str(dataset_dir), train_ratio=0.8, valid_ratio=0.2)

    assert _counts(dataset_dir) == {"train": 80, "valid": 20, "test": 0}


# --------------------------------------------------------------------------
# test_ratio is a size, not a switch
# --------------------------------------------------------------------------


def test_test_ratio_sizes_the_test_split(tmp_path: Path) -> None:
    """0.6/0.2/0.2 and 0.6/0.2/0.1 used to produce the same 60/20/20 split."""
    for test_ratio, expected in ((0.2, 20), (0.3, 30)):
        dataset_dir = tmp_path / f"dataset-{test_ratio}"
        _make_pairs(dataset_dir, 100, "pool")
        split_folder_yolo(
            source_dir=str(dataset_dir),
            train_ratio=0.6,
            valid_ratio=1.0 - 0.6 - test_ratio,
            test_ratio=test_ratio,
        )
        assert _counts(dataset_dir)["test"] == expected


@pytest.mark.parametrize("ratios", [(0.8, 0.2, 0.1), (0.8, 0.1, 0.9), (0.5, 0.2, 0.1)])
def test_ratios_that_do_not_sum_to_one_are_rejected(
    tmp_path: Path, ratios: tuple
) -> None:
    """(0.8, 0.2, 0.1) used to yield an empty test/ that was silently dropped."""
    dataset_dir = tmp_path / "dataset-yolov8"
    _make_pairs(dataset_dir, 100, "pool")
    train_ratio, valid_ratio, test_ratio = ratios

    with pytest.raises(Exception, match="expected 1.0"):
        split_folder_yolo(
            source_dir=str(dataset_dir),
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
        )

    # Refused before anything moved: the staging tree is still intact.
    assert len(list((dataset_dir / "images").iterdir())) == 100


def test_valid_ratio_ignored_without_test_ratio_is_warned_about(
    tmp_path: Path, caplog
) -> None:
    dataset_dir = tmp_path / "dataset-yolov8"
    _make_pairs(dataset_dir, 100, "pool")

    with caplog.at_level(logging.WARNING, logger="src.data.setup"):
        split_folder_yolo(source_dir=str(dataset_dir), train_ratio=0.6, valid_ratio=0.1)

    assert _counts(dataset_dir) == {"train": 60, "valid": 40, "test": 0}
    assert any("valid_ratio=0.1 ignored" in r.getMessage() for r in caplog.records)


def test_matching_valid_ratio_is_not_warned_about(tmp_path: Path, caplog) -> None:
    dataset_dir = tmp_path / "dataset-yolov8"
    _make_pairs(dataset_dir, 100, "pool")

    with caplog.at_level(logging.WARNING, logger="src.data.setup"):
        split_folder_yolo(source_dir=str(dataset_dir), train_ratio=0.8, valid_ratio=0.2)

    assert not caplog.records


# --------------------------------------------------------------------------
# test_ratio vs a dedicated CVAT test set -- issue #19 proper
# --------------------------------------------------------------------------


def test_test_ratio_with_a_dedicated_test_set_is_refused(tmp_path: Path) -> None:
    """The combination that used to delete `1 - train - valid` of the pool."""
    dataset_dir = tmp_path / "dataset-yolov8"
    dataset_test = tmp_path / "dataset-yolov8-test"
    _make_pairs(dataset_dir, 100, "pool")
    _make_pairs(dataset_test, 20, "cvattest")

    with pytest.raises(Exception, match="mutually exclusive"):
        setup_dataset(
            dataset_dir=str(dataset_dir),
            label_names=["fruit", "stalk"],
            train_ratio=0.8,
            valid_ratio=0.1,
            test_ratio=0.1,
            dataset_test=str(dataset_test),
        )

    # Refused before the split ran, so no pair has been moved or deleted yet.
    assert len(list((dataset_dir / "images").iterdir())) == 100
    assert len(list((dataset_test / "images").iterdir())) == 20


def test_dedicated_test_set_alone_loses_nothing(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset-yolov8"
    dataset_test = tmp_path / "dataset-yolov8-test"
    _make_pairs(dataset_dir, 100, "pool")
    _make_pairs(dataset_test, 20, "cvattest")

    setup_dataset(
        dataset_dir=str(dataset_dir),
        label_names=["fruit", "stalk"],
        train_ratio=0.8,
        valid_ratio=0.2,
        dataset_test=str(dataset_test),
    )

    assert _counts(dataset_dir) == {"train": 80, "valid": 20, "test": 20}
    assert "test: test/images" in (dataset_dir / "data.yaml").read_text()


def test_empty_dedicated_test_set_warns_instead_of_raising(
    tmp_path: Path, caplog
) -> None:
    """A test task with no annotations yet should not stop a training run."""
    dataset_dir = tmp_path / "dataset-yolov8"
    dataset_test = tmp_path / "dataset-yolov8-test"
    _make_pairs(dataset_dir, 100, "pool")
    _make_pairs(dataset_test, 0, "cvattest")

    with caplog.at_level(logging.WARNING, logger="src.data.setup"):
        setup_dataset(
            dataset_dir=str(dataset_dir),
            label_names=["fruit", "stalk"],
            train_ratio=0.8,
            valid_ratio=0.2,
            dataset_test=str(dataset_test),
        )

    assert _counts(dataset_dir) == {"train": 80, "valid": 20, "test": 0}
    assert any('split "test" invalid' in r.getMessage() for r in caplog.records)
    assert "test: test/images" not in (dataset_dir / "data.yaml").read_text()
