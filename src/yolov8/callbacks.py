# Ultralytics YOLO 🚀, AGPL-3.0 license
# https://docs.ultralytics.com/usage/callbacks/#all-callbacks

import re

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, TESTS_RUNNING
from ultralytics.utils.torch_utils import model_info_for_loggers

from src.params import args_visualization
from src.utils.general import yaml_loader
from src.utils.logging import get_logger
from src.utils.register_model import register_model_to_clearml
from src.yolov8.clearml_logger import YOLOClearMLLogger
from src.yolov8.metrics_utils import (
    ValStatsAccumulator,
    extract_calibration_bins,
    extract_confusion_matrix_data,
    extract_curve_data,
    extract_learning_rate,
    extract_loss_components,
    extract_mask_box_gap,
    extract_optimal_confidence,
    extract_per_class_metrics,
    extract_speed_metrics,
    extract_worst_images,
)


logger = get_logger(__name__)

# Module-level because the data has to be gathered across many `on_val_batch_end` calls
# and consumed later in `on_train_end`. Callbacks only ever run in the main process --
# dataloader workers do not fire them -- so the forkserver re-import that Python 3.14
# performs per worker cannot duplicate or race this.
val_stats = ValStatsAccumulator()


class _ValidatorRef:
    """Holds the most recent validator so a standalone `val()` can be reported.

    `Model.val()` constructs its validator locally and returns only `validator.metrics`
    (engine/model.py:625-627) -- it never stores the validator, so there is no
    `model.validator` to read afterwards, and `model.validator` resolves through
    `Model.__getattr__` to the nn.Module and raises. `on_val_end` is handed the validator
    directly, which makes it the one reliable way to get a reference.
    """

    current: BaseValidator | None = None


validator_ref = _ValidatorRef()


try:
    import clearml

    from clearml import Task
    from clearml.binding.frameworks.pytorch_bind import PatchPyTorchModelIO
    from clearml.binding.matplotlib_bind import PatchedMatplotlib

    assert hasattr(clearml, "__version__")  # verify package is not directory
    assert not TESTS_RUNNING  # do not log pytest
except ImportError, AssertionError:
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
                # Both the iteration and the series have to tolerate a missing match.
                # The upstream version guards the iteration but then calls it.group()
                # unconditionally, so any file without a _batchN suffix raises inside a
                # callback -- which aborts training, not just the report.
                iteration = int(it.group(1)) if it else 0
                series = f.name.replace(it.group(), "") if it else f.name
                task.get_logger().report_image(
                    title=title,
                    series=series,
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
    try:
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
    finally:
        # pyplot keeps every figure alive until it is closed; this runs once per
        # plot at the end of training.
        plt.close(fig)


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


def _report_metric_scalars(task: Task, trainer: BaseTrainer) -> None:
    """Route each results_dict entry into a titled scalar group."""
    for k, v in trainer.validator.metrics.results_dict.items():
        if k == "fitness":
            # For a segmentation model this is seg.fitness() + box.fitness(), i.e. the
            # sum of two mAP50-95 values and therefore 0-2, not 0-1
            # (SegmentMetrics.fitness, ultralytics/utils/metrics.py:1371). For detection
            # it is exactly mAP50-95. Same series name, two different scales.
            task.get_logger().report_scalar(
                "Metrics/Summary", k, v, iteration=trainer.epoch
            )
            continue
        if "precision" in k:
            task.get_logger().report_scalar(
                "Metrics/Precision",
                k,
                v,
                iteration=trainer.epoch,
            )
            continue
        if "recall" in k:
            task.get_logger().report_scalar(
                "Metrics/Recall", k, v, iteration=trainer.epoch
            )
            continue
        if "mAP50" in k:
            task.get_logger().report_scalar("Metrics/mAP", k, v, iteration=trainer.epoch)
            continue

        task.get_logger().report_scalar("Metrics/Other", k, v, iteration=trainer.epoch)

    # Validation losses
    for k, v in trainer.metrics.items():
        if "val/" in k:
            task.get_logger().report_scalar(
                "Losses/Validation", k, v, iteration=trainer.epoch
            )


def on_train_epoch_end(trainer: BaseTrainer):
    """Report per-epoch training-side values.

    Deliberately does NOT report validation metrics. This hook fires at
    ultralytics/engine/trainer.py:569 and `validate()` does not run until :577, so
    `trainer.validator.metrics` here still holds the *previous* epoch's numbers -- zeros
    on epoch 0. Those moved to `on_fit_epoch_end`, which is where upstream puts them.
    Train losses and the LR are read from the trainer itself and are correct here.
    """
    task: Task = Task.current_task()

    if task:
        clearml_logger = YOLOClearMLLogger(task)

        # Mosaic samples are written during the first epochs; accept either so that a
        # 1-epoch run still gets one (epoch == 1 alone logs nothing for such a run).
        if trainer.epoch in (0, 1):
            _log_debug_samples(
                sorted(trainer.save_dir.glob("train_batch*.jpg")), "Mosaic"
            )

        # Log learning rate if enabled
        if args_visualization.get("log_learning_rate", True):
            learning_rates = extract_learning_rate(trainer)
            if learning_rates:
                clearml_logger.log_learning_rates(learning_rates, trainer.epoch)

        # Log individual loss components if enabled
        if args_visualization.get("log_loss_components", True):
            losses = extract_loss_components(trainer)
            if losses:
                clearml_logger.log_loss_components(losses, trainer.epoch)
                _report_loss_ratios(clearml_logger, losses, trainer.epoch)


def _report_loss_ratios(
    clearml_logger: YOLOClearMLLogger, losses: dict, iteration: int
) -> None:
    """Report loss components relative to box loss.

    Absolute loss values are not comparable between runs with different `box`/`cls`/`dfl`
    gains, but their ratios are. A seg/box ratio drifting upward says the mask branch is
    losing ground against the detector even while the total falls.
    """
    box = next(
        (v for k, v in losses.items() if "box" in k and v not in (0, None)),
        None,
    )
    if not box:
        return
    ratios = {
        f"{name}_over_box": float(value) / float(box)
        for name, value in losses.items()
        if "box" not in name and name != "total_loss" and value is not None
    }
    if ratios:
        clearml_logger.log_scalar_group("Losses/Balance", ratios, iteration)


def on_fit_epoch_end(trainer: BaseTrainer):
    """Report validation metrics, speed and model info once validation has run.

    This hook fires at ultralytics/engine/trainer.py:605, after `validate()` at :577, so
    `trainer.validator.metrics` is current here. That ordering is the whole reason the
    metric reporting lives in this hook rather than `on_train_epoch_end`.
    """
    if task := Task.current_task():
        clearml_logger = YOLOClearMLLogger(task)

        # Validation metrics and losses -- correct as of this hook, stale in the previous.
        _report_metric_scalars(task, trainer)

        # Log epoch time
        task.get_logger().report_scalar(
            title="Speed/Training",
            series="epoch_time_seconds",
            value=trainer.epoch_time,
            iteration=trainer.epoch,
        )

        # Log model info on first epoch
        if trainer.epoch == 0:
            for k, v in model_info_for_loggers(trainer).items():
                task.get_logger().report_single_value(k, v)

        should_log_speed = args_visualization.get("log_speed_metrics", True)
        has_validator = hasattr(trainer, "validator") and trainer.validator is not None
        if should_log_speed and has_validator:
            speeds = extract_speed_metrics(trainer.validator)
            if speeds:
                clearml_logger.log_speed_metrics(speeds, trainer.epoch)

        # Mask-vs-box gap: cheap, and the clearest per-epoch signal that the mask head
        # rather than the detector is the limiting factor on a segmentation run.
        if has_validator and args_visualization.get("log_mask_box_gap", True):
            gap = extract_mask_box_gap(trainer.validator)
            if gap:
                clearml_logger.log_mask_box_gap(gap, trainer.epoch)


def on_val_batch_start(validator: BaseValidator):
    """Pin the confusion matrix's confidence threshold, decoupling it from `args.conf`.

    Ultralytics passes the *same* `args.conf` to NMS and to the confusion matrix
    (detect/val.py:195), but the two want opposite values. Metrics need a floor near zero
    so the PR curve keeps its high-recall tail -- anything filtered by NMS never reaches
    the metrics at all, so a floor of 0.25 silently understates mAP. A readable confusion
    matrix needs the opposite: a threshold around 0.25, or it fills with low-confidence
    detections and a background column that swamps everything else.

    So `args_val["conf"]` is 0.001 (the ultralytics val default, correct for metrics) and
    this wraps `process_batch` to hold the matrix at its own display threshold.

    Fires once per batch, so it is written to be idempotent; `init_metrics` (which creates
    the matrix) runs at engine/validator.py:242, before the first `on_val_batch_start` at
    :245 and before any `update_metrics` at :266.
    """
    matrix = getattr(validator, "confusion_matrix", None)
    if matrix is None or getattr(matrix, "_conf_pinned", False):
        return

    threshold = args_visualization.get("confusion_matrix_conf", 0.25)
    original = matrix.process_batch

    def process_batch_at_display_conf(*args, **kwargs):
        kwargs["conf"] = threshold
        return original(*args, **kwargs)

    matrix.process_batch = process_batch_at_display_conf
    matrix._conf_pinned = True
    logger.debug("confusion matrix pinned to conf=%s", threshold)


def on_val_batch_end(validator: BaseTrainer):
    """Capture per-detection confidence/match pairs before ultralytics discards them.

    `get_stats()` calls `metrics.process()` then `metrics.clear_stats()` on the next line
    (models/yolo/detect/val.py:277-278), and this hook is the last one to fire before that
    (engine/validator.py:271 vs :276). Without this capture there is no way to build a
    TP-vs-FP confidence split or a calibration curve after the fact.

    Logs nothing: it runs once per batch, and per-item loops must not emit output.
    """
    if not args_visualization.get("log_calibration", True):
        return
    val_stats.update(validator)


def on_val_end(validator: BaseValidator):
    """Log validation results, and keep a reference to the validator.

    The reference is what lets src/train.py report the standalone post-training `val()`:
    that call returns only metrics, so `save_dir`, `confusion_matrix` and
    `image_metrics` would otherwise be unreachable. Stored unconditionally, including
    when there is no ClearML task, so the two concerns stay independent.
    """
    validator_ref.current = validator

    if Task.current_task():
        # Log val_labels and val_pred
        _log_debug_samples(sorted(validator.save_dir.glob("val*.jpg")), "Validation")


def report_validation_analysis(
    validator: BaseValidator,
    *,
    iteration: int = 0,
    gallery_dir: Path | None = None,
    title_prefix: str = "",
) -> None:
    """Report the heavy per-class / error-analysis set for a finished validation pass.

    Called once per validation *run*, never per epoch: everything here is a table, a
    multi-series plot or a batch of images, so report volume must not scale with epochs
    any more than log volume scales with dataset size.

    Shared by `on_train_end` and the standalone `val()` in src/train.py. `title_prefix`
    keeps the test split from overwriting the validation split's plots.

    Args:
        validator: A validator whose `metrics` have been processed.
        iteration: Iteration to file the reports under.
        gallery_dir: Directory containing ultralytics' `visualizations/` output, if
            `visualize=True` was set for this run.
        title_prefix: Prefix applied to every ClearML title, e.g. "Test ".

    """
    task: Task | None = Task.current_task()
    if task is None:
        return
    clearml_logger = YOLOClearMLLogger(task)

    _report_class_breakdown(clearml_logger, validator, iteration, title_prefix)
    _report_error_analysis(
        clearml_logger, validator, iteration, gallery_dir, title_prefix
    )
    _report_calibration(clearml_logger, iteration, title_prefix)


def _report_class_breakdown(
    clearml_logger: YOLOClearMLLogger,
    validator: BaseValidator,
    iteration: int,
    title_prefix: str,
) -> None:
    """Per-class table, worst-first bar chart, curve families and operating point."""
    want_table = args_visualization.get("log_per_class_table", True)
    want_bar = args_visualization.get("log_per_class_scatter", True)

    need_metrics = want_table or want_bar
    metrics_df = extract_per_class_metrics(validator) if need_metrics else None

    if metrics_df is not None and want_table:
        clearml_logger.log_per_class_table(
            metrics_df,
            iteration=iteration,
            title=f"{title_prefix}Per-Class Metrics",
            series="Detailed",
        )

    if metrics_df is not None and want_bar:
        clearml_logger.log_per_class_bar(
            metrics_df,
            iteration=iteration,
            title=f"{title_prefix}Per-Class Performance",
        )

    if args_visualization.get("log_interactive_pr_curves", True):
        for curve in extract_curve_data(validator) or []:
            clearml_logger.log_curve_family(
                curve, iteration=iteration, title_prefix=f"{title_prefix}Curves"
            )

    if args_visualization.get("log_optimal_confidence", True):
        optimal = extract_optimal_confidence(validator)
        if optimal:
            clearml_logger.log_optimal_confidence(
                optimal, iteration=iteration, title=f"{title_prefix}Operating Point"
            )


def _report_error_analysis(
    clearml_logger: YOLOClearMLLogger,
    validator: BaseValidator,
    iteration: int,
    gallery_dir: Path | None,
    title_prefix: str,
) -> None:
    """Worst-image table plus the matching GT/FP/TP/FN panels."""
    if not args_visualization.get("log_worst_images", True):
        return

    worst = extract_worst_images(
        validator, limit=args_visualization.get("worst_images_limit", 16)
    )
    if worst is None:
        return

    clearml_logger.log_worst_images_table(
        worst, iteration=iteration, title=f"{title_prefix}Error Analysis"
    )
    _log_error_gallery(clearml_logger, worst, gallery_dir, iteration, title_prefix)


def _report_calibration(
    clearml_logger: YOLOClearMLLogger,
    iteration: int,
    title_prefix: str,
) -> None:
    """TP-vs-FP confidence split and the reliability diagram."""
    if not args_visualization.get("log_calibration", True):
        return

    split = val_stats.confidence_split()
    if not split:
        return

    clearml_logger.log_confidence_split(
        split, iteration=iteration, title=f"{title_prefix}Distributions"
    )
    calibration = extract_calibration_bins(split)
    if calibration:
        clearml_logger.log_reliability_diagram(
            calibration, iteration=iteration, title=f"{title_prefix}Calibration"
        )


def _log_error_gallery(
    clearml_logger: YOLOClearMLLogger,
    worst,
    gallery_dir: Path | None,
    iteration: int,
    title_prefix: str,
) -> None:
    """Upload the GT/FP/TP/FN panels for the worst-ranked images, if they exist.

    Ultralytics writes these into `<save_dir>/visualizations/<image name>` and only when
    `plots=True and visualize=True` (models/yolo/detect/val.py:196). It writes one file
    per validated image with no cap, which is why we enable it for the final val only and
    upload just the worst N here.

    The panels are box-IoU based even on a segmentation run -- `SegmentationValidator`
    inherits `update_metrics`, so the confusion matrix that feeds them never sees masks.
    The title says "Box" so nobody reads them as mask errors.
    """
    if gallery_dir is None:
        return
    vis_dir = Path(gallery_dir) / "visualizations"
    if not vis_dir.is_dir():
        return

    by_name = {p.name: p for p in vis_dir.iterdir() if p.is_file()}
    ordered = [by_name[n] for n in worst["Image"].tolist() if n in by_name]
    if not ordered:
        logger.debug("no visualization panels matched the worst-image list")
        return

    sent = clearml_logger.log_image_files(
        ordered,
        title=f"{title_prefix}Error Analysis",
        series_prefix="worst-box-",
        iteration=iteration,
    )
    logger.info("error gallery: reported %d/%d worst-image panels", sent, len(ordered))


def on_train_end(trainer: BaseTrainer):
    """Log final model and its name on training completion.

    Parameters
    ----------
    trainer : BaseTrainer
        The YOLO trainer instance with training information.

    """
    task: Task = Task.current_task()
    if task := Task.current_task():
        clearml_logger = YOLOClearMLLogger(task)

        _log_debug_samples(
            sorted(trainer.validator.save_dir.glob("val*.jpg")), "Validation"
        )

        # Get class names from data yaml
        data_yaml = yaml_loader(trainer.args.data)
        class_names = data_yaml.get("names", [])
        if isinstance(class_names, dict):
            class_names = list(class_names.values())

        # Log final results, CM matrix + PR plots (static - kept as fallback)
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

        # Log interactive confusion matrix if enabled
        if args_visualization.get("log_interactive_confusion_matrix", True):
            normalized_matrix, labels, counts_matrix = extract_confusion_matrix_data(
                trainer.validator, class_names
            )
            if normalized_matrix is not None:
                clearml_logger.log_interactive_confusion_matrix(
                    normalized_matrix,
                    labels,
                    iteration=trainer.epoch,
                    title="Confusion Matrix",
                    series="Normalized",
                )
            if counts_matrix is not None:
                clearml_logger.log_interactive_confusion_matrix(
                    counts_matrix,
                    labels,
                    iteration=trainer.epoch,
                    title="Confusion Matrix",
                    series="Counts",
                )

        # Per-class tables, curves, thresholds, error analysis and calibration. Shared
        # with the standalone post-training val() in src/train.py so both report the
        # same set under different title prefixes.
        report_validation_analysis(
            trainer.validator,
            iteration=trainer.epoch,
            gallery_dir=trainer.validator.save_dir,
        )

        # Log the final models
        config_data = {
            "model_name": trainer.args.model.replace(".pt", ""),
            "imgsz": trainer.args.imgsz,
            "task_yolo": trainer.args.task,
            "format_model": "PyTorch",
            "data_yaml": data_yaml,
        }
        # data_yaml holds the full class list; keep it out of the INFO line.
        logger.info(
            "registering %s (%s, imgsz=%s) as best + last",
            config_data["model_name"],
            config_data["task_yolo"],
            config_data["imgsz"],
        )
        logger.debug("config_data: %s", config_data)

        # Stamp quality onto the registered model. register_model_to_clearml() has always
        # accepted `metrics` and turned it into `map50:0.873`-style tags, but no caller
        # ever passed it -- so every model in the registry was untagged and could not be
        # ranked or filtered by how good it actually was.
        model_metrics = _registration_metrics(trainer.validator)
        logger.debug("model metrics for registration: %s", model_metrics)

        register_model_to_clearml(
            path_model=trainer.best,
            suffix="best",
            metrics=model_metrics,
            **config_data,
        )
        register_model_to_clearml(
            path_model=trainer.last,
            **config_data,
            suffix="last",
            metrics=model_metrics,
        )


def _registration_metrics(validator: BaseValidator) -> dict[str, float]:
    """Build the metric dict used to tag a registered model.

    Keys become tag prefixes, so they are kept short and lowercase: `map50`, `map`, and
    the mask equivalents on a segmentation run.
    """
    metrics = getattr(validator, "metrics", None)
    box = getattr(metrics, "box", None)
    if box is None:
        return {}

    out: dict[str, float] = {}
    try:
        out["map50"] = float(box.map50)
        out["map"] = float(box.map)
        seg = getattr(metrics, "seg", None)
        if seg is not None and len(getattr(seg, "p", [])) > 0:
            out["mask_map50"] = float(seg.map50)
            out["mask_map"] = float(seg.map)
    except Exception as e:
        logger.debug("could not read metrics for model tags: %s", e)
    return out


# Define available callbacks based on ClearML availability
callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_val_batch_start": on_val_batch_start,
        "on_val_batch_end": on_val_batch_end,
        "on_val_end": on_val_end,
        "on_train_end": on_train_end,
    }
    if clearml
    else {}
)
