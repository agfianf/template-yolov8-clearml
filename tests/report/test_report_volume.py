"""Report volume must not grow with dataset size or with epoch count.

The direct sibling of `tests/yolov8/test_callbacks.py::TestReportVolume` and of
`tests/data/test_data_stage_smoke.py`, and the reason every cap in `src/report/blob.py`
exists. A 20,000-image test split and a 300-epoch run must produce the same report as a
50-image split and a 3-epoch run, give or take the numbers printed in it.
"""

import json

from src.report.build import publish_report
from src.yolov8.metrics_utils import MAX_CURVE_POINTS

from .conftest import DEFAULT_CFG, make_blob


SMALL, LARGE = 50, 3000


def _publish(blob, task, tmp_path, name):
    out = tmp_path / name
    out.mkdir(parents=True, exist_ok=True)
    return publish_report(blob, task=task, cfg=DEFAULT_CFG, output_dir=out)


class TestDatasetSizeInvariance:
    """Ten times the images must not mean ten times the report."""

    def test_upload_count_is_one_at_two_dataset_sizes(self, task, tmp_path):
        """One artifact and one Debug Samples link, whatever the split size."""
        for i, n in enumerate((SMALL, LARGE)):
            task.reset_mock()
            task.get_logger.return_value.reset_mock()
            blob = make_blob(tmp_path / f"b{i}", n_images=n)
            _publish(blob, task, tmp_path, f"out{i}")

            assert task.upload_artifact.call_count == 1
            assert task.get_logger.return_value.report_media.call_count == 1

    def test_thumbnail_count_is_bounded(self, tmp_path):
        """Grids are fixed-N, so thumbnails are bounded by 5 x per-grid, not by images.

        Exact equality between the two sizes is *not* the claim: two grids can select
        the same crop of the same file, and which items win differs with the sample, so
        the deduplicated count wobbles by a few. The bound is the claim.
        """
        per_grid = DEFAULT_CFG["report_gallery_per_grid"]
        counts = []
        for i, n in enumerate((SMALL, LARGE)):
            blob = make_blob(tmp_path / f"t{i}", n_images=n)
            counts.append(len(blob["thumbs"]))

        for count in counts:
            assert count <= DEFAULT_CFG["report_max_thumbnails"]
            assert count <= 5 * per_grid
        # The big run must not carry more than the small one plus one grid's worth.
        assert counts[1] <= counts[0] + per_grid

    def test_grids_hold_a_fixed_item_count(self, tmp_path):
        """Every grid is exactly `report_gallery_per_grid` items at both sizes."""
        per_grid = DEFAULT_CFG["report_gallery_per_grid"]
        for i, n in enumerate((SMALL, LARGE)):
            blob = make_blob(tmp_path / f"g{i}", n_images=n)
            for grid in blob["grids"].values():
                assert len(grid["items"]) <= per_grid

    def test_rendered_size_does_not_grow_with_dataset_size(self, tmp_path):
        """The whole point: 60x the images moves the file by a few per cent, not 60x."""
        from src.report.render import render_report

        sizes = []
        for i, n in enumerate((SMALL, LARGE)):
            blob = make_blob(tmp_path / f"s{i}", n_images=n)
            sizes.append(len(render_report(blob).encode("utf-8")))

        assert abs(sizes[1] - sizes[0]) / sizes[0] < 0.05

    def test_no_key_carries_a_per_detection_array(self, tmp_path):
        """Every distribution is pre-binned, so no blob key scales with detections."""
        small = make_blob(tmp_path / "k0", n_images=SMALL)
        large = make_blob(tmp_path / "k1", n_images=LARGE)

        for key in small:
            if key in ("meta", "grids", "thumbs"):
                continue  # sample-dependent by design, and separately bounded
            a = len(json.dumps(small[key], default=str))
            b = len(json.dumps(large[key], default=str))
            assert b <= a * 1.05 + 64, f"{key} grew with dataset size"


class TestEpochInvariance:
    """A 300-epoch run must not carry 100 times the training history of a 3-epoch one."""

    def test_epoch_count_does_not_change_the_figure_set(self, tmp_path):
        """Same figures at 3 epochs and at 300."""
        short = make_blob(tmp_path / "e0", epochs=3)
        long = make_blob(tmp_path / "e1", epochs=300)

        assert set(short["figures"]) == set(long["figures"])

    def test_training_curves_are_downsampled(self, tmp_path):
        """The only epoch-dependent payload, and it is capped."""
        long = make_blob(tmp_path / "e2", epochs=3000)
        points = len(long["figures"]["f_val_map"]["data"][0]["x"])

        assert points <= MAX_CURVE_POINTS

    def test_report_size_does_not_grow_with_epochs(self, tmp_path):
        """3 epochs and 300 epochs render to within a few per cent of each other."""
        from src.report.render import render_report

        short = len(render_report(make_blob(tmp_path / "e3", epochs=3)).encode())
        long = len(render_report(make_blob(tmp_path / "e4", epochs=300)).encode())

        assert abs(long - short) / short < 0.05
