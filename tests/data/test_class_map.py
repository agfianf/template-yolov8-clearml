"""The class-order fix: one name -> index map for every source in a run.

The bug these cover is silent and expensive. Two CVAT tasks annotated with the
same labels export them in a different order, or one omits a label entirely;
`category_id - 1` therefore means a different class in each, and the converter
merges both into one dataset. Nothing errors -- the model just trains against
scrambled targets, and no chart in the task says so.

Every source here is built in tmp_path by the `write_project` fixture. No CVAT,
no network.
"""

import logging

from pathlib import Path

import pytest

from src.data.class_map import (
    ClassMap,
    UnknownClassError,
    build_class_map,
    read_category_names,
    warn_if_orders_disagree,
)
from src.data.converter.coco2yolo import Coco2Yolo


def _labels_by_class(label_dir: Path) -> dict[str, int]:
    """Map class index -> number of label lines written with it."""
    counts: dict[str, int] = {}
    for txt in label_dir.iterdir():
        for line in txt.read_text().strip().splitlines():
            index = line.split()[0]
            counts[index] = counts.get(index, 0) + 1
    return counts


# --------------------------------------------------------------------------
# Building the map
# --------------------------------------------------------------------------


def test_derived_order_is_the_sorted_union(write_project, annotation_path) -> None:
    a = write_project("a", ["car", "speed_limit", "person"])
    b = write_project("b", ["person", "car"])  # no speed_limit, and reordered

    class_map = build_class_map([annotation_path(a), annotation_path(b)])

    assert class_map.names == ["car", "person", "speed_limit"]
    assert class_map.index_of("speed_limit") == 2


def test_derived_order_ignores_source_order(write_project, annotation_path) -> None:
    a = write_project("a", ["car", "speed_limit"])
    b = write_project("b", ["speed_limit", "car"])
    paths = [annotation_path(a), annotation_path(b)]

    assert build_class_map(paths).names == build_class_map(paths[::-1]).names


def test_pinned_order_survives_a_missing_class(write_project, annotation_path) -> None:
    b = write_project("b", ["person", "car"])

    class_map = build_class_map(
        [annotation_path(b)], explicit_names=["car", "person", "speed_limit"]
    )

    # speed_limit keeps index 2 even though no source has it. That is the point
    # of pinning: the indices still match the checkpoint being fine-tuned.
    assert class_map.names == ["car", "person", "speed_limit"]


def test_excluded_class_reserves_no_index(write_project, annotation_path) -> None:
    a = write_project("a", ["car", "stalk", "person"])

    class_map = build_class_map([annotation_path(a)], exclude_class=["stalk"])

    assert class_map.names == ["car", "person"]


def test_matching_ignores_case_and_whitespace(write_project, annotation_path) -> None:
    a = write_project("a", ["Speed_Limit", "car"])
    b = write_project("b", ["speed_limit ", "car"])

    class_map = build_class_map([annotation_path(a), annotation_path(b)])

    assert len(class_map) == 2
    assert class_map.index_of("SPEED_LIMIT") == class_map.index_of("speed_limit")


def test_the_displayed_spelling_does_not_depend_on_source_order(
    write_project, annotation_path
) -> None:
    """Which spelling reaches data.yaml is a function of the data, not of order.

    Two sources spell it `speed_limit` and one spells it `Speed_Limit`. Taking
    whichever came first would make the class *name* depend on the order the
    task ids were typed in -- the exact property sorting the union removes.
    """
    a = write_project("a", ["Speed_Limit", "car"])
    b = write_project("b", ["speed_limit", "car"])
    c = write_project("c", ["speed_limit", "car"])
    paths = [annotation_path(p) for p in (a, b, c)]

    forwards = build_class_map(paths)
    backwards = build_class_map(paths[::-1])

    assert forwards.names == backwards.names == ["car", "speed_limit"]


def test_a_tie_in_spelling_is_broken_alphabetically(
    write_project, annotation_path
) -> None:
    a = write_project("a", ["Speed_Limit"])
    b = write_project("b", ["speed_limit"])
    paths = [annotation_path(a), annotation_path(b)]

    # One each: alphabetical, so the answer is still order-independent.
    assert build_class_map(paths).names == build_class_map(paths[::-1]).names


def test_empty_map_is_an_error(write_project, annotation_path) -> None:
    a = write_project("a", ["stalk"])

    with pytest.raises(ValueError, match="class map is empty"):
        build_class_map([annotation_path(a)], exclude_class=["stalk"])


def test_read_category_names_keeps_file_order(write_project, annotation_path) -> None:
    a = write_project("a", ["zebra", "car"])

    assert read_category_names(annotation_path(a)) == ["zebra", "car"]


# --------------------------------------------------------------------------
# Converting through the map
# --------------------------------------------------------------------------


def test_same_label_gets_the_same_index_everywhere(
    write_project, annotation_path, tmp_path: Path
) -> None:
    """The regression itself: two orders, one merged dataset, one meaning."""
    a = write_project("a", ["car", "speed_limit"])
    b = write_project("b", ["speed_limit", "car"])
    class_map = build_class_map([annotation_path(a), annotation_path(b)])

    for project in (a, b):
        _out, names, _count = Coco2Yolo(
            src_dir=str(project), output_dir=str(tmp_path / "dataset")
        ).convert(use_segments=False, class_map=class_map)
        assert names == ["car", "speed_limit"]

    # "car" is category_id 1 in `a` and category_id 2 in `b`, and index 0 in
    # both -- the old `category_id - 1` made it 0 and 1.
    for project in (a, b):
        car_lines = [
            line
            for txt in (project / "labels").iterdir()
            for line in txt.read_text().strip().splitlines()
            if line.startswith("0 ")
        ]
        assert len(car_lines) == 2, project
        assert _labels_by_class(project / "labels") == {"0": 2, "1": 2}, project


def test_without_a_class_map_the_indices_disagree(write_project, tmp_path: Path) -> None:
    """Legacy behaviour, still reachable by the flag -- and demonstrably wrong."""
    a = write_project("a", ["car", "speed_limit"])
    b = write_project("b", ["speed_limit", "car"])

    names = []
    for project in (a, b):
        _out, project_names, _count = Coco2Yolo(
            src_dir=str(project), output_dir=str(tmp_path / "dataset")
        ).convert(use_segments=False)
        names.append(project_names)

    # data.yaml took whichever converted last, so index 0 means "car" in one set
    # of label files and "speed_limit" in the other.
    assert names == [["car", "speed_limit"], ["speed_limit", "car"]]


def test_unknown_class_raises_by_default(write_project, tmp_path: Path) -> None:
    a = write_project("a", ["car", "speed_limit"])

    with pytest.raises(UnknownClassError, match="speed_limit"):
        Coco2Yolo(src_dir=str(a), output_dir=str(tmp_path / "out")).convert(
            use_segments=False, class_map=ClassMap(names=["car"])
        )


def test_unknown_class_can_be_dropped(write_project, tmp_path: Path) -> None:
    a = write_project("a", ["car", "speed_limit"])

    Coco2Yolo(src_dir=str(a), output_dir=str(tmp_path / "out")).convert(
        use_segments=False, class_map=ClassMap(names=["car"]), on_unknown_class="drop"
    )

    assert _labels_by_class(a / "labels") == {"0": 2}


def test_dropping_every_class_leaves_background_images(
    write_project, tmp_path: Path
) -> None:
    """An emptied image is kept, as a background image -- deliberately."""
    a = write_project("a", ["speed_limit"])

    _out, _names, count = Coco2Yolo(
        src_dir=str(a), output_dir=str(tmp_path / "out")
    ).convert(
        use_segments=False, class_map=ClassMap(names=["car"]), on_unknown_class="drop"
    )

    assert count == 2
    assert all(not txt.read_text().strip() for txt in (a / "labels").iterdir())


def test_segments_use_the_mapped_index_too(write_project, tmp_path: Path) -> None:
    a = write_project("a", ["speed_limit", "car"])

    Coco2Yolo(src_dir=str(a), output_dir=str(tmp_path / "out")).convert(
        use_segments=True, class_map=ClassMap(names=["car", "speed_limit"])
    )

    assert _labels_by_class(a / "labels") == {"0": 2, "1": 2}
    # A segment line is a class plus >= 6 coordinates; the box path writes 5
    # fields, so this also proves the segment branch was the one taken.
    first = next((a / "labels").iterdir()).read_text().strip().splitlines()[0]
    assert len(first.split()) > 5


def test_disagreeing_orders_are_warned_about(write_project, annotation_path) -> None:
    a = write_project("a", ["car", "speed_limit"])
    b = write_project("b", ["speed_limit", "car"])
    c = write_project("c", ["car", "speed_limit"])

    assert warn_if_orders_disagree([annotation_path(p) for p in (a, b, c)]) is True
    assert warn_if_orders_disagree([annotation_path(a), annotation_path(c)]) is False


# --------------------------------------------------------------------------
# Log volume -- constant in the size of the dataset
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_images", [10, 200])
def test_convert_with_class_map_volume_is_constant(
    capture_src_logs, write_project, tmp_path: Path, n_images: int
) -> None:
    a = write_project("a", ["car", "speed_limit"], n_images)

    with capture_src_logs(logging.INFO) as records:
        Coco2Yolo(src_dir=str(a), output_dir=str(tmp_path / "out")).convert(
            use_segments=False,
            class_map=ClassMap(names=["car"]),
            on_unknown_class="drop",
        )

    # Annotation filtering, the file layout summary, and one warning naming the
    # dropped class -- however many instances of it there were.
    assert len(records) == 3, [r.getMessage() for r in records]


@pytest.mark.parametrize("n_images", [10, 200])
def test_build_class_map_volume_is_constant(
    capture_src_logs, write_project, annotation_path, n_images: int
) -> None:
    a = write_project("a", ["car", "speed_limit"], n_images)

    with capture_src_logs(logging.INFO) as records:
        build_class_map([annotation_path(a)])

    assert len(records) == 1, [r.getMessage() for r in records]
