import os

from typing import Any, Literal

import torch
import yaml

from clearml import OutputModel, Task
from rich import print
from ultralytics import YOLO
from yaml.loader import SafeLoader


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


def load_data_yaml(dataset_folder: str) -> dict:
    """Load the data YAML file from the dataset folder."""
    with open(os.path.join(dataset_folder, "data.yaml")) as f:
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
    print(f"Exporting {format_model.upper()}...")
    if format_model == "engine":
        print("torch.cuda.is_available():", torch.cuda.is_available())
        return yolo.export(
            format=format_model,
            imgsz=imgsz,
            device="0",
            **export_params,
        )

    filtered_params = filter_export_parameters(format_model, export_params)
    print(f"Using parameters for `{format_model.upper()}`: {filtered_params}")

    if "fraction" in yolo.overrides:
        yolo.overrides.pop("fraction")

    if torch.cuda.is_available() and format_model != "openvino":
        filtered_params["device"] = "0"

    return yolo.export(
        format=format_model,
        imgsz=imgsz,
        **filtered_params,
    )


def register_model_with_clearml(
    path_model: str,
    format_model: str,
    model_name: str,
    data_yaml: dict,
    task_yolo: str,
    imgsz: int,
) -> None:
    """Register the exported model with ClearML."""
    name_model = f"{format_model}-{model_name}"
    target_filename = f"{name_model}.{format_model}"
    task: Task = Task.current_task()
    output_model = OutputModel(
        task=task,
        name=name_model,
        comment=str(data_yaml["names"])
        + "\n this CONVERTED BY USING BEST VERSION of this Experiment",
        label_enumeration={lbl: idx for idx, lbl in enumerate(data_yaml["names"])},
        tags=[task_yolo, task.id],
    )

    url_model = output_model.update_weights(
        weights_filename=path_model,
        target_filename=target_filename,
        auto_delete_file=False,
    )
    output_model.wait_for_uploads()

    config_dict = {
        "net": model_name.replace(".pt", ""),
        "imgsz": imgsz,
        "task": task_yolo,
    }
    print(f"Config dict: {config_dict}")
    output_model.update_design(config_dict=config_dict)
    output_model.set_metadata("imgsz", imgsz, "int")
    output_model.set_metadata("task", task_yolo, "str")

    print(
        f"Model registered with ClearML: {name_model} | {output_model.id} | {url_model}"
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
        If an error occurs during model export, it is caught and printed, and the process
        continues for other formats.

    """
    print("\n[Export Model]")

    data_yaml = load_data_yaml(dataset_folder)

    for format_model, is_use in args_export["format"].items():
        if not is_use:
            continue

        try:
            path_model = export_model_format(
                yolo=yolo,
                format_model=format_model,
                imgsz=args_training["imgsz"],
                export_params=args_export["params"],
            )
            print(f"Exported to: {path_model}")

            register_model_with_clearml(
                path_model=path_model,
                format_model=format_model,
                model_name=args_task["model_name"],
                data_yaml=data_yaml,
                task_yolo=task_yolo,
                imgsz=args_training["imgsz"],
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error exporting to {format_model}: {e}")
            continue
