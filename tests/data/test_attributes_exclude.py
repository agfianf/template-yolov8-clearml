"""Regression tests for `attributes_exclude` (issue #20).

Two defects shipped together and hid each other. Only the first key of the dict
was evaluated, because every path out of the loop body was a `break`; and an
annotation that did not carry the attribute crashed on
`ann.attributes.get(name).replace(...)`. A bogus key placed second therefore
never crashed -- it was never reached -- while the same key placed first killed
the run, so behaviour depended on dict insertion order, which nobody controls
when the value comes back from the ClearML UI.

The feature had exactly one appearance in the suite before this file, as `None`.
"""

import logging

import pytest

from src.schema.coco import Coco as CocoSchema
from src.yolov8.dataset_report import DatasetStats, warn_unmatched_attribute_rules


def _payload(annotations: list[dict]) -> dict:
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
        "categories": [{"id": 1, "name": "fruit", "supercategory": ""}],
        "images": [
            {
                "id": 1,
                "width": 64,
                "height": 64,
                "file_name": "img_1.jpg",
                "license": 0,
                "flickr_url": None,
                "coco_url": None,
                "date_captured": 0,
            }
        ],
        "annotations": annotations,
    }


def _ann(ann_id: int, attributes: dict) -> dict:
    x = float(ann_id % 5) * 10.0
    return {
        "id": ann_id,
        "image_id": 1,
        "category_id": 1,
        "segmentation": [[x, 0.0, x + 10, 0.0, x + 10, 10.0, x, 10.0]],
        "area": 100.0,
        "bbox": [x, 0.0, 10.0, 10.0],
        "iscrowd": 0,
        "attributes": attributes,
    }


def _filter(annotations: list[dict], config: dict | None):
    """Run the filter and return (kept annotation ids, report)."""
    kept, report = CocoSchema(**_payload(annotations)).get_imageid_to_annotations(
        attributes_excluded=config
    )
    return sorted(ann.id for anns in kept.values() for ann in anns), report


# Three annotations, and each of the two rules below matches exactly one of them.
THREE = [
    _ann(1, {"maturity_truth": "background", "occluded_truth": "no"}),
    _ann(2, {"maturity_truth": "ripe", "occluded_truth": "yes"}),
    _ann(3, {"maturity_truth": "ripe", "occluded_truth": "no"}),
]
TWO_RULES = {"maturity_truth": "background", "occluded_truth": "yes"}


# --------------------------------------------------------------------------
# Defect 1: only the first key was evaluated
# --------------------------------------------------------------------------


def test_every_key_is_evaluated() -> None:
    kept, report = _filter(THREE, TWO_RULES)

    assert kept == [3]
    # Both rules are credited, so the breakdown cannot pass off a skipped rule as
    # one that legitimately matched nothing.
    assert report.dropped_attr["maturity_truth"] == 1
    assert report.dropped_attr["occluded_truth"] == 1


def test_key_order_does_not_change_the_dataset() -> None:
    forward, _ = _filter(THREE, TWO_RULES)
    reversed_cfg = dict(reversed(list(TWO_RULES.items())))
    backward, _ = _filter(THREE, reversed_cfg)

    assert forward == backward == [3]


def test_a_rule_that_matches_nothing_keeps_everything() -> None:
    kept, report = _filter(THREE, {"maturity_truth": "unripe"})

    assert kept == [1, 2, 3]
    assert not report.dropped_attr


# --------------------------------------------------------------------------
# Defect 2: an annotation without the attribute
# --------------------------------------------------------------------------


def test_annotation_without_the_attribute_is_kept_not_crashed() -> None:
    annotations = [_ann(10, {"maturity_truth": "ripe"}), _ann(11, {})]

    kept, _ = _filter(annotations, {"maturity_truth": "background"})

    assert kept == [10, 11]


def test_a_later_rule_still_applies_when_an_earlier_key_is_absent() -> None:
    """The combination neither fix reaches alone.

    Rule 1's key is missing on this annotation, so the old code died before rule
    2 -- which does match -- was ever consulted.
    """
    annotations = [_ann(20, {"occluded_truth": "yes"})]

    kept, report = _filter(annotations, TWO_RULES)

    assert kept == []
    assert report.dropped_attr["occluded_truth"] == 1


def test_a_missing_key_does_not_count_as_a_match() -> None:
    kept, _ = _filter([_ann(30, {})], {"maturity_truth": "background"})

    assert kept == [30]


# --------------------------------------------------------------------------
# Value comparison
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["Background", "BACKGROUND", " background "])
def test_values_compare_case_and_space_insensitively(value: str) -> None:
    """Matched to the class filter, which lowercases for the same reason."""
    annotations = [_ann(40, {"maturity_truth": value})]

    kept, _ = _filter(annotations, {"maturity_truth": "background"})

    assert kept == []


def test_non_string_attribute_values_are_coerced() -> None:
    """A CVAT checkbox arrives as a bool and used to die on `.replace`."""
    annotations = [_ann(50, {"is_blurry": True}), _ann(51, {"is_blurry": False})]

    kept, _ = _filter(annotations, {"is_blurry": "true"})

    assert kept == [51]


def test_commas_list_alternatives_on_both_sides() -> None:
    annotations = [
        _ann(60, {"flags": "occluded, truncated"}),
        _ann(61, {"flags": "clean"}),
    ]

    kept, _ = _filter(annotations, {"flags": "truncated,unknown"})

    assert kept == [61]


def test_an_empty_rule_value_drops_nothing() -> None:
    kept, _ = _filter([_ann(70, {"maturity_truth": ""})], {"maturity_truth": ""})

    assert kept == [70]


def test_no_config_is_a_no_op() -> None:
    for config in (None, {}):
        kept, report = _filter(THREE, config)
        assert kept == [1, 2, 3]
        assert not report.attr_rules


# --------------------------------------------------------------------------
# The run-level warning that replaces the crash
# --------------------------------------------------------------------------


def test_an_attribute_absent_everywhere_is_warned_about_once() -> None:
    _kept, report = _filter(THREE, {"has_longstalk": "yes"})
    stats = DatasetStats()
    stats.note_filter_report(report)

    assert stats.unmatched_attribute_rules() == ["has_longstalk"]


def test_an_attribute_present_in_only_one_source_is_not_warned_about() -> None:
    """The false alarm the per-source version of this warning would have raised.

    CVAT writes an attribute only onto labels that declare it, so a task without
    the label carries the rule's key nowhere -- while the config is correct.
    """
    _kept_a, report_a = _filter(THREE, TWO_RULES)
    _kept_b, report_b = _filter([_ann(80, {"maturity_truth": "ripe"})], TWO_RULES)

    stats = DatasetStats()
    stats.note_filter_report(report_a)
    stats.note_filter_report(report_b)

    assert stats.unmatched_attribute_rules() == []


def test_the_warning_names_the_attribute(caplog: pytest.LogCaptureFixture) -> None:
    _kept, report = _filter(THREE, {"has_longstalk": "yes"})
    stats = DatasetStats()
    stats.note_filter_report(report)

    with caplog.at_level(logging.WARNING):
        unmatched = warn_unmatched_attribute_rules(stats)

    assert unmatched == ["has_longstalk"]
    assert "has_longstalk" in caplog.text


def test_nothing_is_warned_about_without_a_config(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _kept, report = _filter(THREE, None)
    stats = DatasetStats()
    stats.note_filter_report(report)

    with caplog.at_level(logging.WARNING):
        caplog.clear()  # the filter itself logs its INFO summary during setup
        assert warn_unmatched_attribute_rules(stats) == []
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
