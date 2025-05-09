import os

import yaml

from clearml import OutputModel, Task
from rich import print
from yaml.loader import SafeLoader


def export_handler(
    yolo, task_yolo, dataset_folder, args_export, args_training, args_task
):
    print("\n[Export Model]")

    with open(os.path.join(dataset_folder, "data.yaml")) as f:
        data_yaml = yaml.load(f, Loader=SafeLoader)

    for format_model, is_use in args_export["format"].items():
        try:
            if not is_use:
                continue
            print(f"Exporting {format_model.upper()}...")
            if format_model == "engine":
                import torch

                print("torch.cuda.is_available():", torch.cuda.is_available())
                path_model = yolo.export(
                    format=format_model,
                    imgsz=args_training["imgsz"],
                    device=0,
                    **args_export["params"],
                )
            else:
                path_model = yolo.export(
                    format=format_model,
                    imgsz=args_training["imgsz"],
                    **args_export["params"],
                )
            print(path_model)
            output_model_last = OutputModel(
                task=Task.current_task(),
                name=format_model + "-" + args_task["model_name"],
                comment=str(data_yaml["names"]),
                label_enumeration={
                    lbl: idx for idx, lbl in enumerate(data_yaml["names"])
                },
            )
            output_model_last.update_weights(
                weights_filename=path_model,
                target_filename=format_model
                + "-"
                + args_task["model_name"]
                + "."
                + format_model,
                auto_delete_file=False,
            )

            output_model_last.update_design(
                config_dict={
                    "net": args_task["model_name"].replace(".pt", ""),
                    "imgsz": args_training["imgsz"],
                    "task": task_yolo,
                }
            )
            output_model_last.set_metadata("imgsz", args_training["imgsz"], "int")
            output_model_last.set_metadata("task", task_yolo, "str")

        except Exception as e:
            print(e)
