import glob
import os
import random

import numpy as np
import torch
import ultralytics

from clearml import InputModel, Task  # noqa: F401
from ultralytics import YOLO

from src.config import settings  # noqa: F401
from src.data.setup import cleanup_cache
from src.initizalization import init_ultralytics_settings
from src.utils.clearml_settings import config_clearml, init_clearml
from src.utils.general import get_task_yolo_name, model_name_handler, yaml_loader
from src.utils.logging import get_logger
from src.yolov8.callbacks import callbacks
from src.yolov8.clearml_logger import YOLOClearMLLogger
from src.yolov8.data import DataHandler
from src.yolov8.exporter import export_handler
from src.yolov8.metrics_utils import collect_prediction_confidences


logger = get_logger(__name__)


def _tagging_handler(task: Task, task_yolo: str, model_name: str, handler: DataHandler):
    task.add_tags(f"ul-{ultralytics.__version__}")
    task.add_tags(task_yolo)
    task.add_tags(os.path.basename(model_name).replace(".pt", ""))
    task.add_tags(handler.source_type.upper())


def _generate_data_yaml(args_train: dict, task_yolo: str, dataset_folder: str) -> str:
    data_yaml_file = os.path.join(dataset_folder, "data.yaml")
    if task_yolo == "classify":
        data_yaml_file = dataset_folder
    if task_yolo == "segment":
        args_train["augment"] = False
    return data_yaml_file


def _set_task_name_on_experiment(args_task: dict, args_train: dict) -> tuple[str, str]:
    task_yolo = get_task_yolo_name(args_task["model_name"])
    if not args_train["resume"]:
        model_name = model_name_handler(args_task["model_name"])
    else:
        Task.current_task().add_tags("resume")
        model_name = args_task["model_name"]
    logger.info("TASK_YOLO: %s", task_yolo)
    return task_yolo, model_name


def _predicting_result(
    args_val: dict,
    dataset_folder: str,
    datadotyaml: dict,
    model_yolo: YOLO,
    args_predict: dict,
    args_visualization: dict | None = None,
) -> None:
    logger.debug("args_predict: %s", args_predict)
    path_test = (
        datadotyaml.get("val")
        if datadotyaml.get("test") is None
        else datadotyaml.get("test")
    )
    dir_valid_images = os.path.join(dataset_folder, path_test)

    # Extract and randomize list of image file paths
    # Collect all image files (add more extensions if needed)
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(
            glob.glob(os.path.join(dir_valid_images, "**", ext), recursive=True)
        )

    random.shuffle(image_paths)
    # Now image_paths is a randomized list of full image paths
    # If you want to enumerate:
    max_images = args_predict.get("max_images", 40)
    image_paths = image_paths[:max_images]

    if not image_paths:
        logger.warning("No validation images found")
        return

    logger.info(
        "image_paths: %s, exists: %s", image_paths[0:5], os.path.exists(image_paths[0])
    )

    # Get class names for confidence logging
    class_names = datadotyaml.get("names", [])
    if isinstance(class_names, dict):
        class_names = list(class_names.values())

    model_yolo.model.eval()  # switch ke eval mode
    # with torch.no_grad():
    result = model_yolo.predict(
        source=image_paths,
        imgsz=args_val["imgsz"],
        device="0" if torch.cuda.is_available() else "cpu",
        **args_predict.get("model"),
    )

    # Collect results into a list for confidence extraction
    result_list = list(result)

    task: Task = Task.current_task()
    clearml_logger = YOLOClearMLLogger(task)

    images_group = []
    for i, r in enumerate(result_list):
        im_bgr = r.plot(
            **args_predict.get("plot"),
        )  # BGR-order numpy array
        im_rgb = im_bgr[..., ::-1]  # RGB-order RGB numpy array
        images_group.append(im_rgb)

        # When we have 4 images or at the end, create a grid and report
        if len(images_group) == 4 or (i == len(image_paths) - 1 and images_group):
            # Determine max height and width for padding
            max_h = max(img.shape[0] for img in images_group)
            max_w = max(img.shape[1] for img in images_group)
            # Pad images to same size
            padded_imgs = []
            for img in images_group:
                h, w = img.shape[:2]
                pad_h = max_h - h
                pad_w = max_w - w
                padded = np.pad(
                    img,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                padded_imgs.append(padded)
            # Fill up to 4 images with black if needed
            while len(padded_imgs) < 4:
                padded_imgs.append(np.zeros((max_h, max_w, 3), dtype=np.uint8))
            # Arrange in 2x2 grid
            top = np.concatenate(padded_imgs[:2], axis=1)
            bottom = np.concatenate(padded_imgs[2:], axis=1)
            grid = np.concatenate([top, bottom], axis=0)
            task.get_logger().report_image(
                title="Prediction",
                series=f"image-{i}",
                iteration=1,
                image=grid,
            )
            images_group = []

    # Log confidence histograms if enabled
    if args_visualization is None:
        args_visualization = {}
    if args_visualization.get("log_confidence_histograms", True):
        all_confidences, per_class_confidences = collect_prediction_confidences(
            result_list, class_names
        )

        # Log overall confidence distribution
        if all_confidences:
            clearml_logger.log_confidence_histogram(
                all_confidences,
                iteration=1,
                title="Distributions",
                series="All Classes - Confidence",
            )

        # Log per-class confidence distributions
        for cls_name, confidences in per_class_confidences.items():
            if confidences:
                clearml_logger.log_confidence_histogram(
                    confidences,
                    iteration=1,
                    title="Distributions/Per-Class",
                    series=f"{cls_name} - Confidence",
                )


def main():
    logger.info("ultralytics version: %s", ultralytics.__version__)
    # initialialization
    init_ultralytics_settings()
    task: Task = init_clearml()

    # configuration params
    (
        args_task,
        args_data,
        args_augment,
        args_train,
        args_val,
        args_export,
        args_predict,
        args_visualization,
    ) = config_clearml()

    task.execute_remotely()

    # Set Task Name
    task_yolo, model_name = _set_task_name_on_experiment(args_task, args_train)

    # Download Data
    logger.info("[Downloading Data]")
    handler = DataHandler(args_data=args_data, task_model=task_yolo)
    dataset_folder = handler.export()

    data_yaml_file = _generate_data_yaml(args_train, task_yolo, dataset_folder)

    # Tagging
    _tagging_handler(task, task_yolo, model_name, handler)

    # Utils
    datadotyaml = yaml_loader(data_yaml_file)
    class_2_idx = {cls_name: idx for idx, cls_name in enumerate(datadotyaml["names"])}
    task.set_model_label_enumeration(class_2_idx)
    logger.info("datadotyaml: %s, class_2_idx: %s", datadotyaml, class_2_idx)

    logger.info("[Training]")
    logger.info("LOAD MODEL: %s, task: %s", model_name, task_yolo)

    # if args_train["resume"] or (args_task["model_latest_id"] != ""):
    #     print("Resume training from", args_task["model_latest_id"])  # noqa: ERA001
    #     model_id = args_task["model_latest_id"] # noqa: ERA001
    #     model_input = InputModel(model_id=model_id) # noqa: ERA001
    #     path_model = model_input.get_weights() # noqa: ERA001
    #     model_name = model_input.config_dict["net"] # noqa: ERA001
    #     task_yolo = model_input.config_dict["task"] # noqa: ERA001
    #     args_train["imgsz"] = model_input.config_dict["imgsz"] # noqa: ERA001
    #     model_yolo = YOLO(model=model_name, task=task_yolo, verbose=True) # noqa: ERA001
    #     model_yolo.load(path_model) # noqa: ERA001
    # else: # noqa: ERA001
    logger.debug(
        "model_latest_id: %s, type: %s",
        args_task["model_latest_id"],
        type(args_task["model_latest_id"]),
    )
    model_yolo = YOLO(model=model_name, task=task_yolo, verbose=True)

    logger.info("Override Callbacks")
    for event, func in callbacks.items():
        model_yolo.clear_callback(event)
        model_yolo.add_callback(event, func)

    args_val["imgsz"] = args_train["imgsz"]

    model_yolo.train(
        data=data_yaml_file,
        **args_train,
    )

    cleanup_cache(dataset_folder)

    if datadotyaml.get("test"):
        args_val["split"] = "test"

    try:
        args_val["batch"] = 32
        model_yolo.val(data=data_yaml_file, **args_val)
    except Exception as e:
        logger.exception("Error during validation: %s", e)

    # Export ============================
    logger.info("[Exporting Model]")

    export_handler(
        yolo=model_yolo,
        task_yolo=task_yolo,
        dataset_folder=dataset_folder,
        args_export=args_export,
        args_training=args_train,
        args_task=args_task,
    )

    # Predict ============================
    logger.info("[Predicting]")

    try:
        model_yolo = YOLO(model_yolo.trainer.best)
        _predicting_result(
            args_val,
            dataset_folder,
            datadotyaml,
            model_yolo,
            args_predict,
            args_visualization,
        )
    except Exception as e:
        logger.exception("Error during prediction: %s", e)

    logger.info("Done Experiment")


if __name__ == "__main__":
    main()
