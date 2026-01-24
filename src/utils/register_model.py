from clearml import OutputModel, Task

from src.utils.logging import get_logger


logger = get_logger(__name__)


def register_model_to_clearml(
    path_model: str,
    format_model: str,
    model_name: str,
    data_yaml: dict,
    task_yolo: str,
    imgsz: int,
    suffix: str | None = None,
    metrics: dict | None = None,
) -> None:
    """Register the exported model with ClearML.

    Args:
        path_model: Path to the model weights file.
        format_model: Model format (e.g., 'pytorch', 'onnx', 'engine').
        model_name: Name of the model architecture.
        data_yaml: Dataset configuration with class names.
        task_yolo: YOLO task type (detect, segment, classify).
        imgsz: Input image size.
        suffix: Optional suffix for model name (e.g., 'best', 'last').
        metrics: Optional dict with model metrics for tagging (e.g., {'map50': 0.85}).

    """
    name_model = f"{format_model.lower()}-{model_name}"
    if suffix:
        name_model = f"{name_model}-{suffix}"

    if format_model.lower() == "pytorch":
        format_model = "pt"
    target_filename = f"{name_model}.{format_model}"

    task: Task = Task.current_task()

    # Build model tags with lifecycle and metric info
    model_tags = [task_yolo, task.id, "candidate"]
    if suffix == "best":
        model_tags.append("best")
    if metrics:
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                model_tags.append(f"{metric_name}:{metric_value:.3f}")

    output_model = OutputModel(
        task=task,
        name=name_model,
        comment=str(data_yaml["names"]),
        label_enumeration={lbl: idx for idx, lbl in enumerate(data_yaml["names"])},
        tags=model_tags,
    )

    url_model = output_model.update_weights(
        weights_filename=path_model,
        target_filename=target_filename,
        auto_delete_file=False,
    )
    output_model.wait_for_uploads()

    config_dict = {
        "net": model_name,
        "imgsz": imgsz,
        "task": task_yolo,
    }
    logger.info("Config dict: %s", config_dict)
    output_model.update_design(config_dict=config_dict)
    output_model.set_metadata("imgsz", str(imgsz), "int")
    output_model.set_metadata("task", task_yolo, "str")
    output_model.set_metadata("format", format_model, "str")

    logger.info(
        "Model registered with ClearML: %s | %s | %s", name_model, output_model.id, url_model
    )
