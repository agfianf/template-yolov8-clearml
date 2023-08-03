import os
from clearml import Task
from src.config import (
    args_augment,
    args_export,
    args_logging,
    args_task,
    args_data,
    args_train,
    args_val
)

def init_clearml():
    Task.add_requirements("/workspace/requirements.txt")
    task = Task.init(
        project_name="Debugging/Yolov8",
        task_name="yolov8-train-debugging",
        reuse_last_task_id=False,
        auto_connect_frameworks={"pytorch": False, "matplotlib": False},
        output_uri="s3://10.8.0.66:9000/clearml/mlops-artifacs"
    )

    task.set_script(
        repository="https://github.com/muhammadAgfian96/template-yolov8-clearml.git",
        branch="public",
        working_dir=".",
        entry_point="src/train.py",
    )

    task.set_base_docker(
        docker_image="yolov8-custom:gpu-py3.10.11",
        docker_arguments=["-e PYTHONPATH=/workspacet", "--gpus all", "--ipc=host", "--shm-size=8gb"],
    )
    tags = ['🏷️ v2.4', '🐞 debug']
    task.set_tags(tags)
    return Task.current_task()

def config_clearml():
    Task.current_task().connect(args_task, name="1_Task")
    Task.current_task().connect(args_data, name="2_Data")
    Task.current_task().connect(args_augment, name="3_Augment")
    Task.current_task().connect(args_train, name="4_Training")
    Task.current_task().connect(args_val, name="5_Testing")
    Task.current_task().connect(args_export, name="6_Export")

    ls_exclude = args_data["exclude"].replace(", ", ",").replace(" ,", ",").split(",")

    args_train.update(args_logging)
    args_train.update(args_augment)
    args_data.update({"exclude":ls_exclude})
    return  args_task, args_data, args_augment, args_train, args_val, args_export