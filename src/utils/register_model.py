from clearml import OutputModel, Task
from rich import print


def register_model_to_clearml(
    path_model: str,
    format_model: str,
    model_name: str,
    data_yaml: dict,
    task_yolo: str,
    imgsz: int,
    suffix: str | None = None,
) -> None:
    """Register the exported model with ClearML."""
    name_model = f"{format_model.lower()}-{model_name}"
    if suffix:
        name_model = f"{name_model}-{suffix}"

    if format_model.lower() == "pytorch":
        format_model = "pt"
    target_filename = f"{name_model}.{format_model}"

    task: Task = Task.current_task()
    output_model = OutputModel(
        task=task,
        name=name_model,
        comment=str(data_yaml["names"]),
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
        "net": model_name,
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
