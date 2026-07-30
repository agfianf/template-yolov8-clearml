"""Builders for miniature CVAT exports.

Shared as a fixture rather than as an importable helper because `tests/` is not
a package -- `from tests.data.x import y` does not resolve under pytest's
rootdir-based sys.path handling.
"""

import json

from collections.abc import Callable
from pathlib import Path

import pytest


def _coco_payload(categories: list[str], n_images: int) -> dict:
    """One annotation of every category, in every image."""
    cats = [
        {"id": i, "name": name, "supercategory": ""}
        for i, name in enumerate(categories, start=1)
    ]
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
        for cat in cats:
            # Distinct geometry per annotation: the converter drops duplicate
            # boxes and duplicate segments, so identical shapes collapse to one.
            x = float(ann_id % 5) * 10.0
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img["id"],
                    "category_id": cat["id"],
                    "segmentation": [[x, 0.0, x + 10, 0.0, x + 10, 10.0, x, 10.0]],
                    "area": 100.0,
                    "bbox": [x, 0.0, 10.0, 10.0],
                    "iscrowd": 0,
                    "attributes": {},
                }
            )
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
        "categories": cats,
        "images": images,
        "annotations": annotations,
    }


@pytest.fixture
def write_project(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory writing a CVAT export tree: images/ + annotations/.

    The category *order* is the whole subject of these tests, so it is the
    caller's first argument and is preserved as given.
    """
    from PIL import Image

    def _write(name: str, categories: list[str], n_images: int = 2) -> Path:
        project = tmp_path / name
        (project / "images").mkdir(parents=True)
        (project / "annotations").mkdir(parents=True)
        for i in range(1, n_images + 1):
            Image.new("RGB", (64, 64)).save(project / "images" / f"img_{i}.jpg")
        (project / "annotations" / "instances_default.json").write_text(
            json.dumps(_coco_payload(categories, n_images))
        )
        return project

    return _write


@pytest.fixture
def annotation_path() -> Callable[[Path], str]:
    """Return the path of the COCO file inside a project directory."""
    return lambda project: str(project / "annotations" / "instances_default.json")
