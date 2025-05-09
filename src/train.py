import os

import ultralytics

from clearml import Task
from rich import print
from ultralytics import YOLO

from src.config import settings  # noqa: F401
from src.data.setup import cleanup_cache
from src.initizalization import init_ultralytics_settings
from src.utils.general import get_task_yolo_name, model_name_handler, yaml_loader
from src.yolov8.callbacks import callbacks
from src.yolov8.data import DataHandler
from src.yolov8.exporter import export_handler
from utils.clearml_settings import config_clearml, init_clearml


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
    print("TASK_YOLO", task_yolo)
    return task_yolo, model_name


def main():
    print("ultralytics: version", ultralytics.__version__)
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
    ) = config_clearml()

    task.execute_remotely()

    # Set Task Name
    task_yolo, model_name = _set_task_name_on_experiment(args_task, args_train)

    # Download Data
    print("\n[Downloading Data]")
    handler = DataHandler(args_data=args_data, task_model=task_yolo)
    dataset_folder = handler.export()

    data_yaml_file = _generate_data_yaml(args_train, task_yolo, dataset_folder)

    # Tagging
    _tagging_handler(task, task_yolo, model_name, handler)

    # Utils
    datadotyaml = yaml_loader(data_yaml_file)
    class_2_idx = {cls_name: idx for idx, cls_name in enumerate(datadotyaml["names"])}
    task.set_model_label_enumeration(class_2_idx)
    print("datadotyaml", datadotyaml, "class_2_idx", class_2_idx)

    print("\n[Training]")
    print("LOAD MODEL", model_name, task_yolo)
    model_yolo = YOLO(model=model_name, task=task_yolo, verbose=True)

    print("Override Callbacks")
    for event, func in callbacks.items():
        model_yolo.clear_callback(event)
        model_yolo.add_callback(event, func)

    args_val["imgsz"] = args_train["imgsz"]

    if args_train["resume"]:
        print(">>> RESUME TRAINING <<<")
        model_yolo.resume = True
        model_yolo.train(
            data=data_yaml_file,
            epochs=args_train["epochs"],
            batch=args_train["batch"],
            patience=args_train["patience"],
        )
    else:
        model_yolo.train(
            data=data_yaml_file,
            **args_train,
        )

    cleanup_cache(dataset_folder)

    if datadotyaml.get("test"):
        args_val["split"] = "test"

    try:
        model_yolo.val(data=data_yaml_file, **args_val)
    except Exception as e:
        print("Error Validation", e)

    print("\n[Exporting Model]")

    export_handler(
        yolo=model_yolo,
        task_yolo=task_yolo,
        dataset_folder=dataset_folder,
        args_export=args_export,
        args_training=args_train,
        args_task=args_task,
    )


if __name__ == "__main__":
    main()
