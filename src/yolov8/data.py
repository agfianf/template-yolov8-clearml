"""data.py - Data handling utilities for YOLOv8 training pipeline.

This module provides the DataHandler class for managing dataset download, conversion,
and preparation for training, validation, and testing. It supports multiple data sources
(CVAT, S3, Label Studio), converts COCO to YOLO format, and sets up the dataset structure
for YOLO Series from Ultralytics.
"""

import os
import shutil

from typing import Any

from rich import print

from src.data.converter.coco2yolo import (
    Coco2Yolo,
    count_files_in_directory,
    image_extensions,
)
from src.data.downloader.method.cvat import CVATHTTPDownloaderV1, CVATHTTPDownloaderV2
from src.data.setup import setup_dataset
from src.schema.coco import Coco as CocoSchema
from src.utils.general import read_json


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
        self.exclude_cls = self.config.get("class_exclude", []) or []
        self.attributes_exclude = self.config.get("attributes_exclude", None)
        self.area_segment_min = self.config.get("area_segment_min", None)
        self.task_model = task_model

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
            if source in {
                "params",
                "class_exclude",
                "attributes_exclude",
                "area_segment_min",
            }:
                continue
            for _, v in (d or {}).items():
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
        self, project_dir: str, output_dir: str, use_segments: bool
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
            print("[bold green]CVAT Server V1 detected[/bold green]")
            cvat_http = CVATHTTPDownloaderV1()
        elif is_server2:
            print("[bold green]CVAT Server V2 detected[/bold green]")
            cvat_http = CVATHTTPDownloaderV2()
        else:
            raise ValueError("CVAT Server not found")

        # Process training projects
        for project_dir in cvat_http.get_local_dataset_coco(
            task_ids=task_id_train,
            annotations_only=False,
        ):
            print(f"\n📁 [cyan]Dataset[/cyan] {project_dir} 📁")
            ann_train_val = os.path.join(
                project_dir, "annotations", "instances_default.json"
            )
            annotation_type = self._get_annotation_type(ann_train_val)
            print(f"annotation_type: {annotation_type}, task_model: {self.task_model}")
            use_segments = (
                "segmentation" in annotation_type and self.task_model != "detect"
            )
            output_train, label_names, countfiles = self._process_coco_project(
                project_dir=project_dir,
                output_dir=self.dataset_dir,
                use_segments=use_segments,
            )
            total_count_files += countfiles

        # Process test projects if provided
        if task_id_test:
            for project_dir in cvat_http.get_local_dataset_coco(
                task_ids=task_id_test, annotations_only=False
            ):
                ann_test = os.path.join(
                    project_dir, "annotations", "instances_default.json"
                )
                annotation_type = self._get_annotation_type(ann_test)
                use_segments = "segmentation" in annotation_type
                _, _, countfiles = self._process_coco_project(
                    project_dir, self.dataset_test_dir, use_segments
                )
                total_count_files += countfiles
        else:
            self.dataset_test_dir = None

        print(f"\n🧮 [yellow]TOTAL COUNT IMAGES[/yellow]: {total_count_files}")
        print(f"label_names: {label_names}")
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
        for split in ["train", "valid", "test"]:
            dir_path = os.path.join(self.dataset_dir, split)
            img_count = count_files_in_directory(dir_path, extensions=image_extensions)
            lbl_count = count_files_in_directory(dir_path, extensions=["txt"])
            print(
                f"🧮 [{split.capitalize()}] TOTAL COUNT IMAGES: {img_count}"
                f" | TOTAL COUNT LABELS: {lbl_count}"
            )

    def export(self) -> str:
        """Prepare and export the dataset for YOLOv8 training.

        Returns
        -------
        str
            Path to the prepared dataset directory.

        """
        if self.source_type == "s3":
            print("[yellow]S3 source not implemented yet.[/yellow]")
        elif self.source_type == "cvat":
            self._handle_cvat()
        elif self.source_type == "label_studio":
            print("[yellow]Label Studio source not implemented yet.[/yellow]")
        else:
            raise ValueError(
                "Cek config datanya pak. source must be s3, cvat or label_studio"
            )
        return self.dataset_dir


if __name__ == "__main__":
    from rich import print

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
