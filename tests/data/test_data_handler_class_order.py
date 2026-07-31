"""DataHandler end to end over a fake CVAT, for the class-order feature flag.

The map is only worth anything if it reaches every source, the test split
included -- a test split numbered differently from the training split scores
against the wrong classes and reads as a bad model. These run the real
`_handle_cvat` with the downloader replaced, so the wiring is covered.
"""

from pathlib import Path

import pytest

from src.data.class_map import UnknownClassError
from src.yolov8 import data as data_module
from src.yolov8.data import DataHandler


TRAIN_A, TRAIN_B, TEST = 1, 2, 3
# All three tasks live in one CVAT project, which is the case the project-id
# option has to get right: the test task is *inside* the training project.
CVAT_PROJECT = 7


@pytest.fixture
def fake_cvat(write_project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Replace both CVAT downloaders with one that yields prepared directories.

    10 images per task: `split_folder_yolo` computes `int(n * (1 - 0.8))` for
    the valid split and `creating_yaml_file` rejects an empty one.
    """
    projects = {
        TRAIN_A: write_project("train_a", ["car", "speed_limit"], 10),
        TRAIN_B: write_project("train_b", ["speed_limit", "car"], 10),
        TEST: write_project("test", ["car"], 10),
    }

    class FakeDownloader:
        def get_about_server(self):
            return True, {}

        def get_local_dataset_coco(self, task_ids, annotations_only=False):  # noqa: ARG002
            return [str(projects[t]) for t in task_ids]

        def list_tasks_of_project(self, project_id):
            if project_id != CVAT_PROJECT:
                return []
            return [
                {"id": t, "name": f"task-{t}", "size": 10, "status": "completed"}
                for t in (TRAIN_A, TRAIN_B, TEST)
            ]

    monkeypatch.setattr(data_module, "CVATHTTPDownloaderV1", FakeDownloader)
    monkeypatch.setattr(data_module, "CVATHTTPDownloaderV2", FakeDownloader)
    # DataHandler puts its output under os.getcwd().
    monkeypatch.chdir(tmp_path)
    return projects


def _config(**overrides) -> dict:
    config = {
        "cvat": {"task_ids_train": [TRAIN_A, TRAIN_B], "task_ids_test": [TEST]},
        "label_studio": {"project_id_train": None, "project_id_test": None},
        "s3": {"s3_uri_dir_train": None, "s3_uri_dir_test": None},
        "params": {"train_ratio": 0.8, "val_ratio": 0.2, "test_ratio": None},
        "class_exclude": "",
        "unify_class_order": True,
        "class_names": "",
        "on_unknown_class": "error",
        "attributes_exclude": None,
        "area_segment_min": 0,
    }
    config.update(overrides)
    return config


def _yaml(dataset_dir: Path) -> str:
    return (dataset_dir / "data.yaml").read_text()


@pytest.mark.usefixtures("fake_cvat")
def test_every_task_shares_one_class_order() -> None:
    handler = DataHandler(args_data=_config(), task_model="detect")

    dataset_dir = Path(handler.export())

    assert "nc: 2" in _yaml(dataset_dir)
    assert "names: ['car', 'speed_limit']" in _yaml(dataset_dir)

    # The test split went through the same map: its only class is car, at index
    # 0 -- not at index 0 because it happened to come first in that task.
    test_labels = list((dataset_dir / "test" / "labels").iterdir())
    assert test_labels
    indices = {
        line.split()[0]
        for txt in test_labels
        for line in txt.read_text().strip().splitlines()
    }
    assert indices == {"0"}


@pytest.mark.usefixtures("fake_cvat")
def test_pinned_names_are_used_verbatim() -> None:
    handler = DataHandler(
        args_data=_config(class_names="speed_limit, car, pedestrian"),
        task_model="detect",
    )

    dataset_dir = Path(handler.export())

    # Including a class no task has: the checkpoint being fine-tuned has it,
    # which is the reason to pin in the first place.
    assert "names: ['speed_limit', 'car', 'pedestrian']" in _yaml(dataset_dir)
    assert "nc: 3" in _yaml(dataset_dir)


@pytest.mark.usefixtures("fake_cvat")
def test_excluded_class_leaves_no_gap() -> None:
    handler = DataHandler(
        args_data=_config(class_exclude="speed_limit"), task_model="detect"
    )

    dataset_dir = Path(handler.export())

    assert "nc: 1" in _yaml(dataset_dir)
    assert "names: ['car']" in _yaml(dataset_dir)


@pytest.mark.usefixtures("fake_cvat")
def test_flag_off_restores_the_legacy_numbering() -> None:
    handler = DataHandler(args_data=_config(unify_class_order=False), task_model="detect")

    dataset_dir = Path(handler.export())

    # Asserted so that changing it is a deliberate act: data.yaml is whatever
    # the last converted task called its categories.
    assert "names: ['speed_limit', 'car']" in _yaml(dataset_dir)


@pytest.mark.usefixtures("fake_cvat")
def test_unknown_class_fails_the_run() -> None:
    handler = DataHandler(args_data=_config(class_names="car"), task_model="detect")

    with pytest.raises(UnknownClassError, match="speed_limit"):
        handler.export()


@pytest.mark.usefixtures("fake_cvat")
def test_unknown_class_can_be_dropped() -> None:
    handler = DataHandler(
        args_data=_config(class_names="car", on_unknown_class="drop"),
        task_model="detect",
    )

    dataset_dir = Path(handler.export())

    assert "names: ['car']" in _yaml(dataset_dir)


# --------------------------------------------------------------------------
# Config parsing
# --------------------------------------------------------------------------


def test_class_names_accepts_a_list_or_a_string() -> None:
    from_string = DataHandler(args_data=_config(class_names="a, b"))
    from_list = DataHandler(args_data=_config(class_names=["a", "b"]))

    assert from_string.class_names == from_list.class_names == ["a", "b"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("false", False),
        ("False", False),
        ("0", False),
        ("true", True),
        (False, False),
        (True, True),
        # Unrecognised keeps the default rather than guessing.
        ("maybe", True),
        ("", True),
    ],
)
def test_unify_class_order_survives_the_ui_sending_a_string(value, expected) -> None:
    """`bool("false")` is True -- the UI edit would otherwise be a no-op."""
    handler = DataHandler(args_data=_config(unify_class_order=value))

    assert handler.unify_class_order is expected


def test_settings_are_not_mistaken_for_data_sources() -> None:
    """`_check_source` walks args_data; a scalar setting used to crash it."""
    handler = DataHandler(args_data=_config(on_unknown_class="drop"))

    assert handler.source_type == "cvat"


def test_bad_on_unknown_class_is_rejected_up_front() -> None:
    with pytest.raises(ValueError, match="on_unknown_class"):
        DataHandler(args_data=_config(on_unknown_class="ignore"))


# --------------------------------------------------------------------------
# Naming sources by project instead of by task
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("fake_cvat")
def test_a_training_project_excludes_its_own_test_task() -> None:
    """The leak this option exists to prevent, end to end."""
    handler = DataHandler(
        args_data=_config(
            cvat={
                "task_ids_train": [],
                "task_ids_test": [TEST],
                "project_ids_train": [CVAT_PROJECT],
                "project_ids_test": [],
            }
        ),
        task_model="detect",
    )

    dataset_dir = Path(handler.export())

    assert "names: ['car', 'speed_limit']" in _yaml(dataset_dir)
    n_train = len(list((dataset_dir / "train" / "images").iterdir()))
    n_valid = len(list((dataset_dir / "valid" / "images").iterdir()))
    n_test = len(list((dataset_dir / "test" / "images").iterdir()))

    # Two training tasks = 20 pairs, which `split_folder_yolo` turns into 16 + 3
    # (the documented int(20 * (1 - 0.8)) rounding). Had TEST been trained on as
    # well it would be 30 pairs -> 24 + 5, so these numbers *are* the assertion
    # that the test task was excluded.
    assert (n_train, n_valid) == (16, 3)
    assert n_test == 10


@pytest.mark.usefixtures("fake_cvat")
def test_project_ids_accept_ui_text() -> None:
    handler = DataHandler(
        args_data=_config(
            cvat={
                "task_ids_train": "",
                "task_ids_test": f"[{TEST}]",
                "project_ids_train": str(CVAT_PROJECT),
                "project_ids_test": "",
            }
        ),
        task_model="detect",
    )

    dataset_dir = Path(handler.export())

    assert len(list((dataset_dir / "test" / "images").iterdir())) == 10
