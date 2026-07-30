import os
import time

from typing import Any, Literal

import torch
import yaml

from ultralytics import YOLO
from yaml.loader import SafeLoader

from src.utils.logging import get_logger
from src.utils.register_model import register_model_to_clearml


logger = get_logger(__name__)


# Constants for format-specific parameters
FORMAT_PARAMETERS = {
    "onnx": ["imgsz", "half", "dynamic", "simplify", "opset", "nms", "batch", "device"],
    "torchscript": ["imgsz", "half", "optimize", "nms", "batch", "device"],
    "openvino": [
        "imgsz",
        "half",
        "dynamic",
        "int8",
        "nms",
        "batch",
        "data",
        "fraction",
        "device",
    ],
    "engine": None,  # None indicates that all parameters should be used
}


def _size_mb(path: str) -> float:
    """Size of an exported artifact -- openvino produces a directory, not a file."""
    if os.path.isfile(path):
        return os.path.getsize(path) / 1024**2
    if os.path.isdir(path):
        return (
            sum(
                os.path.getsize(os.path.join(root, name))
                for root, _, files in os.walk(path)
                for name in files
            )
            / 1024**2
        )
    return 0.0


def load_data_yaml(dataset_folder: str) -> dict:
    """Load the data YAML file from the dataset folder."""
    yaml_path = os.path.join(dataset_folder, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"data.yaml not found in {dataset_folder}")
    with open(yaml_path) as f:
        return yaml.load(f, Loader=SafeLoader)


def filter_export_parameters(format_model: str, params: dict[str, Any]) -> dict[str, Any]:
    """Filter export parameters based on the format requirements."""
    if format_model == "engine":
        return params

    allowed_params = FORMAT_PARAMETERS.get(format_model, [])
    if allowed_params is None:
        return params.copy()

    return {k: v for k, v in params.items() if k in allowed_params}


def export_model_format(
    yolo: YOLO, format_model: str, imgsz: int, export_params: dict[str, Any]
) -> str:
    """Export model in the specified format with appropriate parameters."""
    logger.debug("exporting %s (imgsz=%s)", format_model, imgsz)
    if format_model == "engine":
        logger.debug("torch.cuda.is_available(): %s", torch.cuda.is_available())
        return yolo.export(
            format=format_model,
            imgsz=imgsz,
            device="0",
            **export_params,
        )

    filtered_params = filter_export_parameters(format_model, export_params)
    logger.debug("parameters for %s: %s", format_model, filtered_params)

    if "fraction" in yolo.overrides:
        yolo.overrides.pop("fraction")

    if torch.cuda.is_available() and format_model != "openvino":
        filtered_params["device"] = "0"

    return yolo.export(
        format=format_model,
        imgsz=imgsz,
        **filtered_params,
    )


def export_handler(
    yolo: YOLO,
    task_yolo: Literal["segment", "detect", "classify"],
    dataset_folder: str,
    args_export: dict,
    args_training: dict,
    args_task: dict,
) -> None:
    """Handle the export of YOLO models to various formats.

    Parameters
    ----------
    yolo : YOLO
        The YOLO model instance.
    task_yolo : Literal["segment", "detect", "classify"]
        Type of YOLO task to perform (segment, detect, classify).
    dataset_folder : str
        Path to the dataset folder.
    args_export : dict
        Dictionary containing export configuration, including formats and parameters.
    args_training : dict
        Dictionary containing training parameters such as image size.
    args_task : dict
        Dictionary containing task-specific parameters, e.g., model name.

    Raises
    ------
    Exception
        If an error occurs during model export, it is caught and logged, and the process
        continues for other formats.

    """
    # No banner here: src/train.py already logs "[Exporting Model]".
    data_yaml = load_data_yaml(dataset_folder)

    requested = [fmt for fmt, is_use in args_export["format"].items() if is_use]
    succeeded = 0

    for format_model in requested:
        started = time.monotonic()
        try:
            path_model = export_model_format(
                yolo=yolo,
                format_model=format_model,
                imgsz=args_training["imgsz"],
                export_params=args_export["params"],
            )

            # One implementation of registration, in src/utils/register_model.py.
            # The copy that used to live here logged the same two lines with the
            # same wording -- that is where the "duplicate" messages came from --
            # and had already drifted: no `format` metadata, no lifecycle tags,
            # and imgsz sent as an int to a field declared "int" as a string.
            register_model_to_clearml(
                path_model=path_model,
                format_model=format_model,
                model_name=args_task["model_name"],
                data_yaml=data_yaml,
                task_yolo=task_yolo,
                imgsz=args_training["imgsz"],
            )
            succeeded += 1
            logger.info(
                "export %s: ok in %.1fs -> %s (%.1f MB)",
                format_model,
                time.monotonic() - started,
                os.path.basename(str(path_model).rstrip("/")),
                _size_mb(str(path_model)),
            )

        except Exception as e:
            # TensorRT without a GPU is an absent capability, not a defect --
            # warn rather than raising the alarm with a full traceback.
            if format_model == "engine" and not torch.cuda.is_available():
                logger.warning("export engine: skipped, no CUDA device available (%s)", e)
            else:
                logger.exception("export %s: failed", format_model)
            continue

    logger.info("export: %d/%d formats ok", succeeded, len(requested))
