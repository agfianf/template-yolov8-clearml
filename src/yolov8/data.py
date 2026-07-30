"""data.py - Data handling utilities for YOLOv8 training pipeline.

This module provides the DataHandler class for managing dataset download, conversion,
and preparation for training, validation, and testing. It supports multiple data sources
(CVAT, S3, Label Studio), converts COCO to YOLO format, and sets up the dataset structure
for YOLO Series from Ultralytics.
"""

import os
import shutil

from typing import Any

from src.data.class_map import ClassMap, build_class_map, warn_if_orders_disagree
from src.data.converter.coco2yolo import (
    Coco2Yolo,
    count_files_in_directory,
    image_extensions,
)
from src.data.downloader.method.cvat import CVATHTTPDownloaderV1, CVATHTTPDownloaderV2
from src.data.setup import setup_dataset
from src.schema.coco import Coco as CocoSchema
from src.utils.general import read_json
from src.utils.logging import get_logger
from src.yolov8.dataset_report import report_dataset_composition


logger = get_logger(__name__)


def _as_name_list(value: Any) -> list[str]:
    """Accept either a list or the comma-separated string the ClearML UI sends.

    `class_exclude` already had to survive both -- params.py writes a string,
    config_clearml() splits it, and the `__main__` blocks pass a list -- and the
    new class-order keys are edited in the same UI panel.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def _as_bool(value: Any, default: bool) -> bool:
    """Read a flag that may arrive as a string from the ClearML UI.

    `bool("false")` is True, so a plain `bool(...)` here would make turning the
    flag off in the UI a no-op -- and the failure mode of *that* is a run that
    silently keeps the behaviour you just switched off. Anything unrecognised
    keeps the default rather than guessing.
    """
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
        if text:
            logger.warning(
                "unify_class_order=%r is not a boolean, using %s", value, default
            )
        return default
    if value is None:
        return default
    return bool(value)


# Keys of args_data that configure the run rather than name a data source.
_NON_SOURCE_KEYS = frozenset(
    {
        "params",
        "class_exclude",
        "attributes_exclude",
        "area_segment_min",
        "unify_class_order",
        "class_names",
        "on_unknown_class",
    }
)


def _annotation_path(project_dir: str) -> str:
    """Return the path of the COCO file inside a CVAT export directory."""
    return os.path.join(project_dir, "annotations", "instances_default.json")


class DataHandler:
    """Handles dataset preparation for YOLOv8 training pipeline.

    Parameters
    ----------
    args_data : dict
        Configuration dictionary for data sources and parameters.
    task_model : str, optional
        Task type (e.g., 'detect', 'segment', 'classify').

    """

    def __init__(self, args_data: dict[str, Any], task_model: str | None = None):
        self.config = args_data
        self.source_type = self._check_source()
        self.dataset_dir = os.path.join(os.getcwd(), "dataset-yolov8")
        self.dataset_test_dir = f"{self.dataset_dir}-test"
        self.exclude_cls = _as_name_list(self.config.get("class_exclude"))
        self.attributes_exclude = self.config.get("attributes_exclude", None)
        self.area_segment_min = self.config.get("area_segment_min", None)
        self.task_model = task_model
        # See src/data/class_map.py for what these three are for. The flag is the
        # escape hatch back to per-source `category_id - 1`, kept only so an old
        # task can be re-run and reproduce the indices it was trained with.
        self.unify_class_order = _as_bool(self.config.get("unify_class_order"), True)
        self.class_names = _as_name_list(self.config.get("class_names"))
        self.on_unknown_class = (
            (self.config.get("on_unknown_class") or "error").strip().lower()
        )
        if self.on_unknown_class not in {"error", "drop"}:
            raise ValueError(
                f"on_unknown_class must be 'error' or 'drop', got"
                f" {self.on_unknown_class!r}"
            )

    def _check_source(self) -> str:
        """Determine the data source type from the configuration.

        Returns
        -------
        str
            The source type ('cvat', 's3', or 'label_studio').

        Raises
        ------
        ValueError
            If more than one source type is specified.

        """
        source_type = set()
        for source, d in self.config.items():
            # A source is a dict of locations. The isinstance check is what keeps
            # a *scalar* setting out without being named here -- it would
            # otherwise reach `.items()` below and raise AttributeError. The set
            # is still needed for the settings that are themselves dicts.
            if source in _NON_SOURCE_KEYS or not isinstance(d, dict):
                continue
            for v in d.values():
                if v not in (None, "", []):
                    source_type.add(source)
        if len(source_type) == 1:
            return list(source_type)[0]
        raise ValueError("source must be just 1")

    def _cleanup_dirs(self):
        """Remove existing dataset directories to ensure a clean state."""
        for d in [self.dataset_dir, self.dataset_test_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)

    def _process_coco_project(
        self,
        project_dir: str,
        output_dir: str,
        use_segments: bool,
        class_map: ClassMap | None = None,
    ) -> tuple:
        """Convert a COCO-format project to YOLO format.

        Parameters
        ----------
        project_dir : str
            Path to the COCO project directory.
        output_dir : str
            Output directory for YOLO-formatted data.
        use_segments : bool
            Whether to use segmentation masks.
        class_map : ClassMap, optional
            Run-wide name -> index map. None means the legacy per-source
            `category_id - 1`.

        Returns
        -------
        tuple
            (output_path, label_names, count_files)

        """
        converter = Coco2Yolo(src_dir=project_dir, output_dir=output_dir)
        return converter.convert(
            use_segments=use_segments,
            exclude_class=self.exclude_cls,
            attributes_excluded=self.attributes_exclude,
            area_segment_min=self.area_segment_min,
            class_map=class_map,
            on_unknown_class=self.on_unknown_class,
        )

    def _build_class_map(self, project_dirs: list[str]) -> ClassMap | None:
        """Return the run's class order, or None when the flag is off."""
        annotation_paths = [_annotation_path(d) for d in project_dirs]
        if not self.unify_class_order:
            warn_if_orders_disagree(annotation_paths)
            return None
        return build_class_map(
            annotation_paths=annotation_paths,
            explicit_names=self.class_names,
            exclude_class=self.exclude_cls,
        )

    def _get_annotation_type(self, ann_path: str) -> list[str]:
        """Read annotation file and return annotation types."""
        d_anns = read_json(ann_path)
        coco = CocoSchema(**d_anns)
        return coco.checking_task()

    def _handle_cvat(self):
        """Handle dataset download and conversion from CVAT source."""
        self._cleanup_dirs()
        total_count_files = 0
        label_names = []
        task_id_train = self.config["cvat"]["task_ids_train"]
        task_id_test = self.config["cvat"]["task_ids_test"]

        is_server1, _ = CVATHTTPDownloaderV1().get_about_server()
        is_server2, _ = CVATHTTPDownloaderV2().get_about_server()
        if is_server1:
            logger.info("cvat: server V1 detected, %d train task(s)", len(task_id_train))
            cvat_http = CVATHTTPDownloaderV1()
        elif is_server2:
            logger.info("cvat: server V2 detected, %d train task(s)", len(task_id_train))
            cvat_http = CVATHTTPDownloaderV2()
        else:
            raise ValueError("CVAT Server not found")

        # Download everything first, train and test alike. The class map has to be
        # built from all of them before the first label file is written, and the
        # downloader returns a list rather than a generator, so nothing is lost by
        # pulling the test tasks early -- a broken test task now fails before an
        # hour of conversion instead of after it.
        train_dirs = cvat_http.get_local_dataset_coco(
            task_ids=task_id_train,
            annotations_only=False,
        )
        test_dirs = (
            cvat_http.get_local_dataset_coco(
                task_ids=task_id_test, annotations_only=False
            )
            if task_id_test
            else []
        )

        class_map = self._build_class_map(train_dirs + test_dirs)
        if class_map is not None:
            label_names = list(class_map.names)

        # Process training projects
        for project_dir in train_dirs:
            annotation_type = self._get_annotation_type(_annotation_path(project_dir))
            logger.debug(
                "%s: annotation_type=%s, task_model=%s",
                project_dir,
                annotation_type,
                self.task_model,
            )
            use_segments = (
                "segmentation" in annotation_type and self.task_model != "detect"
            )
            _output, names, countfiles = self._process_coco_project(
                project_dir=project_dir,
                output_dir=self.dataset_dir,
                use_segments=use_segments,
                class_map=class_map,
            )
            # Only meaningful without a class map, and only then because there is
            # nothing better: whichever task converted last names every class.
            if class_map is None:
                label_names = names
            total_count_files += countfiles

        # Process test projects if provided
        if test_dirs:
            for project_dir in test_dirs:
                annotation_type = self._get_annotation_type(_annotation_path(project_dir))
                use_segments = "segmentation" in annotation_type
                _, _, countfiles = self._process_coco_project(
                    project_dir,
                    self.dataset_test_dir,
                    use_segments,
                    class_map=class_map,
                )
                total_count_files += countfiles
        else:
            self.dataset_test_dir = None

        logger.info(
            "cvat: %s images collected across %d task(s), %d classes",
            f"{total_count_files:,}",
            len(task_id_train) + len(task_id_test or []),
            len(label_names),
        )
        logger.debug("label_names: %s", label_names)
        self._finalize_dataset(label_names)

    def _finalize_dataset(self, label_names: list[str]):
        """Finalize dataset setup and print summary statistics.

        Parameters
        ----------
        label_names : list of str
            List of class names.

        """
        setup_dataset(
            dataset_dir=self.dataset_dir,
            dataset_test=self.dataset_test_dir,
            label_names=label_names,
            train_ratio=self.config["params"]["train_ratio"],
            valid_ratio=self.config["params"]["val_ratio"],
            test_ratio=self.config["params"].get("test_ratio"),
        )
        counts = []
        for split in ["train", "valid", "test"]:
            dir_path = os.path.join(self.dataset_dir, split)
            img_count = count_files_in_directory(dir_path, extensions=image_extensions)
            lbl_count = count_files_in_directory(dir_path, extensions=["txt"])
            if img_count or lbl_count:
                counts.append(f"{split} {img_count:,} img / {lbl_count:,} lbl")
        logger.info("dataset ready: %s", ", ".join(counts))

        # Once per data stage, after every source has been converted -- class balance and
        # object geometry are properties of the whole dataset, not of one CVAT task.
        report_dataset_composition()

    def export(self) -> str:
        """Prepare and export the dataset for YOLOv8 training.

        Returns
        -------
        str
            Path to the prepared dataset directory.

        """
        if self.source_type == "s3":
            logger.warning("S3 source not implemented yet")
        elif self.source_type == "cvat":
            self._handle_cvat()
        elif self.source_type == "label_studio":
            logger.warning("Label Studio source not implemented yet")
        else:
            raise ValueError(
                "Cek config datanya pak. source must be s3, cvat or label_studio"
            )
        return self.dataset_dir


if __name__ == "__main__":
    from schema.params import (  # noqa: F401
        args_augment,
        args_data,
        args_export,
        args_logging,
        args_task,
        args_train,
        args_val,
    )
    from src.utils.general import (  # noqa: F401
        get_task_yolo_name,
        model_name_handler,
        yaml_loader,
    )
    from utils.clearml_settings import init_clearml

    task = init_clearml()
    task_yolo = get_task_yolo_name(args_task["model_name"])
    handler = DataHandler(args_data=args_data, task_model=task_yolo)
    dataset_folder = handler.export(task_model=task_yolo)
