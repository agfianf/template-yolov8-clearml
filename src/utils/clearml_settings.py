import os

from clearml import InputModel, Task

from src.params import (
    args_augment,
    args_data,
    args_export,
    args_logging,
    args_task,
    args_train,
    args_val,
)


def init_clearml() -> Task:
    curr_task = Task.current_task()
    print("init clearml, Task.current_task=", curr_task)

    if curr_task is None:
        curr_dir = os.getcwd()
        req_path = os.path.join(curr_dir, "requirements.txt")
        Task.add_requirements(req_path)
        curr_task = Task.init(
            project_name="Template/Yolov11",
            task_name="yolov11-train",
            reuse_last_task_id=False,
            auto_connect_frameworks={"pytorch": False, "matplotlib": False},
        )

        curr_task.set_script(
            repository="https://github.com/agfianf/template-yolov8-clearml.git",
            branch="feat/upgrade-ultralytics",  # noqa: ERA001
            working_dir=".",
            entry_point="src/train.py",
        )

    curr_task.set_base_docker(
        docker_image="yolov11-binsho:py3.12",
        docker_arguments=[
            "-e CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1",
            "-e CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=/workspace/.venv/bin/python",
            "-e PYTHONPATH=/workspace",
            "--gpus all",
            "--ipc=host",
            "--shm-size=50gb",
        ],
    )

    tags = ["🏷️ v2.7", "DEBUG"]
    curr_task.set_tags(tags)

    return Task.current_task()


def config_clearml():
    """Overwrite `args_task`, `args_data`, `args_augment`, `args_train`, `args_val`,
    `args_export` from ClearML UI using.

    `Task.connect()` method.

    This function will be called in the main function of train.py
    """  # noqa: D205
    curr_task: Task = Task.current_task()
    curr_task.connect(args_task, name="1_Task")
    curr_task.connect(args_data, name="2_Data")
    curr_task.connect(args_augment, name="3_Augment")
    curr_task.connect(args_train, name="4_Training")
    curr_task.connect(args_val, name="5_Testing")
    curr_task.connect(args_export, name="6_Export")

    exclude_data = args_data.get("class_exclude", "")
    if exclude_data is None:
        exclude_data = ""
    ls_exclude = exclude_data.replace(", ", ",").replace(" ,", ",").split(",")

    if args_task["model_latest_id"] != "":
        print("Downloading latest model")
        latest_model = InputModel(model_id=args_task["model_latest_id"])
        path_latest_model = latest_model.get_weights()
        args_train["resume"] = True
        args_task["model_name"] = path_latest_model
        print("▶️ Resume training from", latest_model)

    args_train.update(args_logging)
    args_train.update(args_augment)
    args_data.update({"class_exclude": ls_exclude})
    args_data.update({"attributes_exclude": args_data.get("attributes_exclude", {})})
    args_data.update({"area_segment_min": args_data.get("area_segment_min", None)})

    return args_task, args_data, args_augment, args_train, args_val, args_export
