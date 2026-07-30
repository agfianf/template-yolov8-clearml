import os

from clearml import InputModel, Task

from src.params import (
    DOCKER_ARGUMENTS,
    DOCKER_IMAGE,
    args_augment,
    args_console,
    args_data,
    args_export,
    args_logging,
    args_predict,
    args_task,
    args_train,
    args_val,
    args_visualization,
)
from src.utils.logging import get_logger


logger = get_logger(__name__)


def init_clearml() -> Task:
    curr_task: Task = Task.current_task()
    logger.info("init clearml, Task.current_task=%s", curr_task)

    if curr_task is None:
        curr_dir = os.getcwd()
        req_path = os.path.join(curr_dir, "requirements.txt")
        Task.add_requirements(req_path)
        curr_task = Task.init(
            project_name=args_task["clearml_project"],
            task_name=args_task["clearml_task_name"],
            reuse_last_task_id=False,
            auto_connect_frameworks={"pytorch": False, "matplotlib": False},
        )

        curr_task.set_script(
            repository="https://github.com/agfianf/template-yolov8-clearml.git",
            branch="main",
            working_dir=".",
            entry_point="src/train.py",
        )

    # Both arguments must be passed together. set_base_docker() replaces the whole
    # container section, so naming only docker_image silently drops the arguments --
    # the agent then ignores the baked venv and tries to build one with pip.
    curr_task.set_base_docker(
        docker_image=DOCKER_IMAGE,
        docker_arguments=DOCKER_ARGUMENTS,
    )

    # add_tags, not set_tags: set_tags replaces the list, so it wiped any tag added
    # from the ClearML UI on the next run. The image tag is recorded because a task
    # outlives the image it ran on and the UI shows no other trace of which one.
    curr_task.add_tags([f"image:{DOCKER_IMAGE.rsplit(':', 1)[-1]}"])

    return Task.current_task()


def config_clearml():
    """Overwrite `args_task`, `args_data`, `args_augment`, `args_train`, `args_val`,
    `args_export` from ClearML UI using.

    `Task.connect()` method.

    This function will be called in the main function of train.py
    """  # noqa: D205
    curr_task: Task = Task.current_task()
    curr_task.connect(args_console, name="0_Console")
    curr_task.connect(args_task, name="1_Task")
    curr_task.connect(args_data, name="2_Data")
    curr_task.connect(args_augment, name="3_Augment")
    curr_task.connect(args_train, name="4_Training")
    curr_task.connect(args_val, name="5_Testing")
    curr_task.connect(args_predict, name="6_Predict")
    curr_task.connect(args_export, name="7_Export")
    curr_task.connect(args_visualization, name="8_Visualization")

    exclude_data = args_data.get("class_exclude", "")
    if exclude_data is None:
        exclude_data = ""
    ls_exclude = exclude_data.replace(", ", ",").replace(" ,", ",").split(",")

    if args_task["model_latest_id"] != "":
        logger.info("Downloading latest model")
        latest_model = InputModel(model_id=args_task["model_latest_id"])
        path_latest_model = latest_model.get_weights()
        args_train["resume"] = True
        args_task["model_name"] = path_latest_model
        logger.info("Resume training from %s", latest_model)

    args_train.update(args_logging)
    args_train.update(args_augment)
    args_data.update({"class_exclude": ls_exclude})
    args_data.update({"attributes_exclude": args_data.get("attributes_exclude", {})})
    args_data.update({"area_segment_min": args_data.get("area_segment_min", None)})

    return (
        args_task,
        args_data,
        args_augment,
        args_train,
        args_val,
        args_export,
        args_predict,
        args_visualization,
    )
