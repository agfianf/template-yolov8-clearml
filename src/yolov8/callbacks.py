# Ultralytics YOLO 🚀, AGPL-3.0 license
# https://docs.ultralytics.com/usage/callbacks/#all-callbacks

import re

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import LOGGER, TESTS_RUNNING
from ultralytics.utils.torch_utils import model_info_for_loggers

from src.utils.general import yaml_loader
from src.utils.logging import get_logger
from src.utils.register_model import register_model_to_clearml


logger = get_logger(__name__)


try:
    import clearml

    from clearml import Task
    from clearml.binding.frameworks.pytorch_bind import PatchPyTorchModelIO
    from clearml.binding.matplotlib_bind import PatchedMatplotlib

    assert hasattr(clearml, "__version__")  # verify package is not directory
    assert not TESTS_RUNNING  # do not log pytest
except (ImportError, AssertionError):
    clearml = None


def _log_debug_samples(files, title="Debug Samples") -> None:
    """Log files (images) as debug samples in the ClearML task.

    Args:
        files (list): A list of file paths in PosixPath format.
        title (str): A title that groups together images with the same values.

    """
    if task := Task.current_task():
        for f in files:
            if f.exists():
                it = re.search(r"_batch(\d+)", f.name)
                iteration = int(it.groups()[0]) if it else 0
                task.get_logger().report_image(
                    title=title,
                    series=f.name.replace(it.group(), ""),
                    local_path=str(f),
                    iteration=iteration,
                )


def _log_plot(title, plot_path) -> None:
    """Log an image as a plot in the plot section of ClearML.

    Args:
        title (str): The title of the plot.
        plot_path (str): The path to the saved image file.

    """
    img = mpimg.imread(plot_path)
    fig = plt.figure()
    ax = fig.add_axes(
        [0, 0, 1, 1], frameon=False, aspect="auto", xticks=[], yticks=[]
    )  # no ticks
    ax.imshow(img)

    series = ""
    if "confusion_matrix" in title:
        series = title
        title = "Confusion Matrix"
    if "Mask" in title:
        series = title
        title = "Mask"
    if "Box" in title:
        series = title
        title = "Box"
    if "labels" in title:
        series = title
        title = "Labels"

    task: Task = Task.current_task()
    task.get_logger().report_matplotlib_figure(
        title=title,
        series=series,
        figure=fig,
        report_interactive=False,
    )


def on_pretrain_routine_start(trainer: BaseTrainer):
    """Run at start of pretraining routine; initialize and connect/log task to ClearML."""
    try:
        task: Task | None = Task.current_task()
        logger.info("override on_pretrain_routine_start")
        if task:
            # Make sure the automatic pytorch and matplotlib bindings are disabled!
            # We are logging these plots and model files manually in the integration
            PatchPyTorchModelIO.update_current_task(None)
            PatchedMatplotlib.update_current_task(None)
        else:
            task: Task = Task.init(
                project_name=trainer.args.project or "YOLOv8",
                task_name=trainer.args.name,
                tags=["YOLOv8"],
                output_uri=True,
                reuse_last_task_id=False,
                auto_connect_frameworks={"pytorch": False, "matplotlib": False},
            )
            LOGGER.warning(
                "ClearML Initialized a new task. If you want to run remotely, "
                "please add clearml-init and connect your arguments"
                " before initializing YOLO."
            )
    except Exception as e:
        LOGGER.warning(
            "WARNING ⚠️ ClearML installed but not initialized correctly,"
            f" not logging this run. {e}"
        )


def on_train_epoch_end(trainer: BaseTrainer):
    task: Task = Task.current_task()

    if task:
        """Logs debug samples for the first epoch of YOLO training."""
        if trainer.epoch == 1:
            _log_debug_samples(
                sorted(trainer.save_dir.glob("train_batch*.jpg")), "Mosaic"
            )
        """Report the current training progress."""
        # print("trainer", trainer.metrics)  # noqa: ERA001
        # print("trainer.validator.metrics",trainer.validator.metrics.results_dict)  # noqa: E501, ERA001
        for k, v in trainer.validator.metrics.results_dict.items():
            if k == "fitness":
                task.get_logger().report_scalar("fitness", k, v, iteration=trainer.epoch)
                continue
            if "precision" in k:
                task.get_logger().report_scalar(
                    "precision",
                    k,
                    v,
                    iteration=trainer.epoch,
                )
                continue
            if "recall" in k:
                task.get_logger().report_scalar("recall", k, v, iteration=trainer.epoch)
                continue
            if "mAP50" in k:
                task.get_logger().report_scalar("mAP50", k, v, iteration=trainer.epoch)
                continue

            task.get_logger().report_scalar("train", k, v, iteration=trainer.epoch)

        for k, v in trainer.metrics.items():
            if "val/" in k:
                task.get_logger().report_scalar("val/loss", k, v, iteration=trainer.epoch)


def on_fit_epoch_end(trainer: BaseTrainer):
    """Report model information to logger at the end of an epoch."""
    if task := Task.current_task():
        # You should have access to the validation bboxes under jdict
        task.get_logger().report_scalar(
            title="Epoch Time",
            series="Epoch Time",
            value=trainer.epoch_time,
            iteration=trainer.epoch,
        )

        if trainer.epoch == 0:
            for k, v in model_info_for_loggers(trainer).items():
                task.get_logger().report_single_value(k, v)


def on_val_end(validator: BaseTrainer):
    """Log validation results including labels and predictions."""
    if Task.current_task():
        # Log val_labels and val_pred
        _log_debug_samples(sorted(validator.save_dir.glob("val*.jpg")), "Validation")


def on_train_end(trainer: BaseTrainer):
    """Log final model and its name on training completion.

    Parameters
    ----------
    trainer : BaseTrainer
        The YOLO trainer instance with training information.

    """
    task: Task = Task.current_task()
    if task := Task.current_task():
        _log_debug_samples(
            sorted(trainer.validator.save_dir.glob("val*.jpg")), "Validation"
        )
        # Log final results, CM matrix + PR plots
        files = [
            "results.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "labels_correlogram.jpg",
            "labels.jpg",
            *(
                f"{x}_curve.png"
                for x in (
                    "BoxF1",
                    "BoxPR",
                    "BoxP",
                    "BoxR",
                    "MaskF1",
                    "MaskPR",
                    "MaskP",
                    "MaskR",
                )
            ),
        ]
        files = [
            (trainer.save_dir / f) for f in files if (trainer.save_dir / f).exists()
        ]  # filter

        for f in files:
            _log_plot(title=f.stem, plot_path=f)

        # Report final metrics
        for k, v in trainer.validator.metrics.results_dict.items():
            task.get_logger().report_single_value(k, v)

        # Log the final models
        # trainer.args.task  # segment, detect, classify
        # trainer.args.imgsz  # 640
        # trainer.args.model  # yolo11n-seg.pt
        data_yaml = yaml_loader(trainer.args.data)
        config_data = {
            "model_name": trainer.args.model.replace(".pt", ""),
            "imgsz": trainer.args.imgsz,
            "task_yolo": trainer.args.task,
            "format_model": "PyTorch",
            "data_yaml": data_yaml,
        }
        logger.info("config_data: %s", config_data)

        register_model_to_clearml(
            path_model=trainer.best,
            suffix="best",
            **config_data,
        )
        register_model_to_clearml(
            path_model=trainer.last,
            **config_data,
            suffix="last",
        )


# Define available callbacks based on ClearML availability
callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_val_end": on_val_end,
        "on_train_end": on_train_end,
    }
    if clearml
    else {}
)
