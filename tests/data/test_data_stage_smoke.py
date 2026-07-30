"""Fast smoke + log-volume tests for the data preparation stage.

The stage that produced the most console noise had no test at all: reaching it
for real needs CVAT credentials and a few minutes. These build a miniature COCO
tree in tmp_path instead and run the same code, in milliseconds. Not marked
`slow` -- nothing here downloads or trains.

The volume assertions are written as *scale invariance*, not as fixed line
counts: what matters is that a dataset ten times the size does not produce ten
times the log.
"""

import json
import logging

from pathlib import Path

import pytest

from src.data.converter.coco2yolo import Coco2Yolo
from src.data.setup import setup_dataset, split_folder_yolo
from src.schema.coco import Coco as CocoSchema


CATEGORIES = [
    {"id": 1, "name": "fruit", "supercategory": ""},
    {"id": 2, "name": "stalk", "supercategory": ""},
]


def _annotation(
    ann_id: int, image_id: int, category_id: int, area: float = 100.0
) -> dict:
    # Geometry has to differ per annotation: the converter drops duplicate boxes
    # and duplicate segments, so identical shapes would collapse into one line.
    x = float(ann_id % 5) * 10.0
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [[x, 0.0, x + 10, 0.0, x + 10, 10.0, x, 10.0]],
        "area": area,
        "bbox": [x, 0.0, 10.0, 10.0],
        "iscrowd": 0,
        "attributes": {"maturity_truth": "ripe"},
    }


def _coco_dict(n_images: int, anns_per_image: int) -> dict:
    images = [
        {
            "id": i,
            "width": 64,
            "height": 64,
            "file_name": f"img_{i}.jpg",
            "license": 0,
            "flickr_url": None,
            "coco_url": None,
            "date_captured": 0,
        }
        for i in range(1, n_images + 1)
    ]
    annotations = []
    ann_id = 1
    for img in images:
        for k in range(anns_per_image):
            # Alternate the classes so the excluded one is always represented.
            annotations.append(_annotation(ann_id, img["id"], 1 if k % 2 else 2))
            ann_id += 1
    return {
        "licenses": [],
        "info": {
            "year": "2026",
            "version": "1",
            "description": "",
            "contributor": "",
            "url": "",
            "date_created": "",
        },
        "categories": CATEGORIES,
        "images": images,
        "annotations": annotations,
    }


def _write_project(root: Path, n_images: int, anns_per_image: int) -> Path:
    """Write a COCO project the way CVAT lays it out: images/ + annotations/."""
    from PIL import Image

    project = root / "project"
    (project / "images").mkdir(parents=True)
    (project / "annotations").mkdir(parents=True)

    for i in range(1, n_images + 1):
        Image.new("RGB", (64, 64)).save(project / "images" / f"img_{i}.jpg")

    (project / "annotations" / "instances_default.json").write_text(
        json.dumps(_coco_dict(n_images, anns_per_image))
    )
    return project


# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------


def test_convert_produces_a_yolo_tree(tmp_path: Path) -> None:
    project = _write_project(tmp_path, n_images=5, anns_per_image=4)
    out_dir = tmp_path / "dataset-yolov8"

    output, categories, count = Coco2Yolo(
        src_dir=str(project), output_dir=str(out_dir)
    ).convert(use_segments=True, exclude_class=["stalk"])

    assert Path(output) == out_dir
    assert categories == ["fruit", "stalk"]
    assert count == 5

    images = sorted((out_dir / "images").iterdir())
    labels = sorted((out_dir / "labels").iterdir())
    assert len(images) == 5
    assert len(labels) == 5

    # Two of the four annotations per image were `stalk`, and stalk is excluded.
    first_label = labels[0].read_text().strip().splitlines()
    assert len(first_label) == 2
    assert {line.split()[0] for line in first_label} == {"0"}


def test_excluded_class_is_reported_not_written() -> None:
    coco = CocoSchema(**_coco_dict(n_images=5, anns_per_image=4))
    _anns, report = coco.get_imageid_to_annotations(exclude_class=["stalk"])

    assert report.total == 20
    assert report.kept == 10
    assert report.dropped == 10
    assert report.dropped_class["stalk"] == 10


def test_area_filter_is_reported() -> None:
    payload = _coco_dict(n_images=2, anns_per_image=2)
    payload["annotations"][0]["area"] = 1.0
    coco = CocoSchema(**payload)

    _anns, report = coco.get_imageid_to_annotations(area_segment_min=50.0)

    assert report.dropped_area == 1
    assert report.kept == 3


def test_setup_dataset_splits_and_writes_yaml(tmp_path: Path) -> None:
    project = _write_project(tmp_path, n_images=10, anns_per_image=2)
    out_dir = tmp_path / "dataset-yolov8"
    _output, categories, _count = Coco2Yolo(
        src_dir=str(project), output_dir=str(out_dir)
    ).convert(use_segments=False)

    setup_dataset(
        dataset_dir=str(out_dir),
        label_names=categories,
        train_ratio=0.8,
        valid_ratio=0.2,
    )

    # 8 / 1, not 8 / 2: num_valid is int(10 * (1 - 0.8)) and 1 - 0.8 is
    # 0.19999999999999996 in binary floating point, so the last pair is dropped.
    # Asserted as-is rather than fixed -- changing it would move every existing
    # experiment's split.
    assert len(list((out_dir / "train" / "images").iterdir())) == 8
    assert len(list((out_dir / "valid" / "images").iterdir())) == 1
    yaml_text = (out_dir / "data.yaml").read_text()
    assert "nc: 2" in yaml_text
    assert "train: train/images" in yaml_text


# --------------------------------------------------------------------------
# Log volume -- constant in the size of the dataset
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_annotations", [100, 10_000])
def test_coco_filtering_volume_is_constant(capture_src_logs, n_annotations: int) -> None:
    coco = CocoSchema(**_coco_dict(n_images=n_annotations // 4, anns_per_image=4))

    with capture_src_logs(logging.INFO) as records:
        coco.get_imageid_to_annotations(exclude_class=["stalk"])

    assert len(records) == 1


@pytest.mark.parametrize("n_images", [10, 200])
def test_convert_volume_is_constant(
    capture_src_logs, tmp_path: Path, n_images: int
) -> None:
    project = _write_project(tmp_path, n_images=n_images, anns_per_image=2)

    with capture_src_logs(logging.INFO) as records:
        Coco2Yolo(src_dir=str(project), output_dir=str(tmp_path / "out")).convert(
            use_segments=False, exclude_class=["stalk"]
        )

    # Budget for one CVAT task: annotation filtering, label conversion, and the
    # file layout summary.
    assert len(records) == 3, [r.getMessage() for r in records]


@pytest.mark.parametrize("n_images", [10, 200])
def test_split_volume_is_constant(
    capture_src_logs, tmp_path: Path, n_images: int
) -> None:
    project = _write_project(tmp_path, n_images=n_images, anns_per_image=2)
    out_dir = tmp_path / "out"
    Coco2Yolo(src_dir=str(project), output_dir=str(out_dir)).convert(use_segments=False)

    with capture_src_logs(logging.INFO) as records:
        split_folder_yolo(source_dir=str(out_dir), train_ratio=0.8, valid_ratio=0.2)

    # The progress summary and the split summary.
    assert len(records) == 2, [r.getMessage() for r in records]


# 10 images minimum: with 5, num_valid is int(5 * (1 - 0.8)) == 0 and
# creating_yaml_file then rejects the empty valid split. Pre-existing floating
# point behaviour, out of scope for a logging change.
@pytest.mark.parametrize("n_images", [10, 200])
def test_setup_dataset_volume_is_constant(
    capture_src_logs, tmp_path: Path, n_images: int
) -> None:
    project = _write_project(tmp_path, n_images=n_images, anns_per_image=2)
    out_dir = tmp_path / "out"
    _output, categories, _count = Coco2Yolo(
        src_dir=str(project), output_dir=str(out_dir)
    ).convert(use_segments=False)

    with capture_src_logs(logging.INFO) as records:
        setup_dataset(dataset_dir=str(out_dir), label_names=categories)

    # split progress, split summary, data.yaml -- the plan's budget of 3 for
    # "Split + data.yaml".
    assert len(records) == 3, [r.getMessage() for r in records]
