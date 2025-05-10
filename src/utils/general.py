import json
import os
import shutil

from typing import Literal

import yaml

from clearml import Task
from yaml.loader import SafeLoader


def get_task_yolo_name(arg_model_name) -> Literal["segment", "detect", "classify"]:
    task_yolo = "None"
    arg_model_name = arg_model_name.replace(".pt", "")
    if "-seg" in arg_model_name:
        task_yolo = "segment"
    if "-cls" in arg_model_name:
        task_yolo = "classify"
    if "-cls" not in arg_model_name and "-seg" not in arg_model_name:
        task_yolo = "detect"
    return task_yolo


def model_name_handler(arg_model_name):
    """Return str: model_pt or path_model_yml."""
    default_path_model_yaml = os.path.join(os.getcwd(), "src/yolov8/yolov8.yaml")
    default_path_model_yaml = Task.current_task().connect_configuration(
        default_path_model_yaml, name="Model YAML"
    )

    if "yaml" in arg_model_name:
        # rename
        new_path_model_yaml = os.path.join(os.getcwd(), f"src/yolov8/{arg_model_name}")
        shutil.copy(default_path_model_yaml, new_path_model_yaml)

        print("config_file", default_path_model_yaml, new_path_model_yaml)
        return new_path_model_yaml
    new_model_name = arg_model_name + ".pt"
    return new_model_name


def yaml_loader(filepath):
    with open(filepath) as file_descriptor:
        data = yaml.load(file_descriptor, Loader=SafeLoader)
    return data


def read_json(ann_path):
    with open(ann_path) as f:
        ann_d = json.load(f)
    return ann_d


def get_label_enumeration_from_data_yaml(path_data_yaml: str):
    datadotyaml = yaml_loader(path_data_yaml)
    class_2_idx = {cls_name: idx for idx, cls_name in enumerate(datadotyaml["names"])}
    return class_2_idx
