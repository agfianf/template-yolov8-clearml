# https://docs.ultralytics.com/usage/cfg/#train
# https://docs.ultralytics.com/usage/cfg/#augmentation

import os

from pathlib import Path


# Stdlib only, deliberately: the Makefile imports this module with bare python3 to
# read DOCKER_IMAGE back out, so it must work with no virtualenv.

# ./VERSION is the one place the version is written. It is deliberately not in
# pyproject.toml: that file is bind-mounted into the Dockerfile's `uv sync` layer,
# so a version string there makes every bump re-install ~8GB of dependencies.
VERSION_FILE = Path(__file__).resolve().parents[1] / "VERSION"


def _read_version() -> str:
    """Read ./VERSION, or "unknown" if it is missing or empty.

    "unknown" is chosen to fail loudly: `yolo-trainer:unknown` is not a tag that
    exists anywhere, so the agent's pull fails with an obvious message instead of
    a task quietly running on whatever image happened to be lying around.
    """
    try:
        return VERSION_FILE.read_text().strip() or "unknown"
    except OSError:
        return "unknown"


VERSION = _read_version()

# Single source of truth for the training image. The Makefile reads this constant
# rather than declaring its own copy, and set_base_docker() stamps it onto every
# task, so `make build` and what the agents pull can never drift apart.
#
# Never re-tag a version that agents have already pulled: a ClearML task stores the
# tag it was created with and will keep requesting it forever. Bump ./VERSION
# (`make bump PART=patch`) before any build that changes the image contents.
#
# TRAINER_IMAGE overrides it -- point it at a registry path to run agents off a
# pushed image, e.g. TRAINER_IMAGE=ghcr.io/acme/yolo-trainer:0.2.2.
DOCKER_IMAGE = os.getenv("TRAINER_IMAGE", f"yolo-trainer:{VERSION}")

# Passed to every containerised run. The two SKIP_* flags are what make the image's
# baked venv authoritative: without them the agent ignores /workspace/.venv and
# rebuilds an environment with pip from the task's recorded packages.
DOCKER_ARGUMENTS = [
    "-e CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1",
    "-e CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=/workspace/.venv/bin/python",
    "-e PYTHONPATH=/workspace",
    "--gpus all",
    # Dataloader workers exchange batches through shared memory; Docker's 64MB
    # default is far too small and shows up as a silent worker crash mid-epoch.
    "--ipc=host",
    "--shm-size=50gb",
]

args_augment = {
    "hsv_h": 0.015,  # image HSV-Hue augmentation (fraction)
    "hsv_s": 0.7,  # image HSV-Saturation augmentation (fraction)
    "hsv_v": 0.4,  # image HSV-Value augmentation (fraction)
    "degrees": 25.0,  # image rotation (+/- deg)
    "translate": 0.1,  # image translation (+/- fraction)
    "scale": 0.5,  # image scale (+/- gain)
    "shear": 0.0,  # image shear (+/- deg)
    "perspective": 0.0,  # image perspective (+/- fraction), range 0-0.001
    "flipud": 0.2,  # image flip up-down (probability)
    "fliplr": 0.5,  # image flip left-right (probability)
    "mosaic": 1.0,  # image mosaic (probability)
    "mixup": 0.0,  # image mixup (probability)
    "copy_paste": 0.0,  # segment copy-paste (probability)
}

args_export = {
    "format": {
        "torchscript": 0,  # TorchScript
        "onnx": 0,  # ONNX
        "openvino": 0,  # OpenVINO
        "engine": 0,  # TensorRT
        "coreml": 0,  # CoreML
        "saved_model": 0,  # TensorFlow SavedModel
        "pb": 0,  # TensorFlow GraphDef
        "tflite": 0,  # TensorFlow Lite
        "edgetpu": 0,  # TensorFlow Edge TPU
        "tfjs": 0,  # TensorFlow.js
        "paddle": 0,  # PaddlePaddle
    },
    "params": {
        "keras": False,
        "optimize": False,
        "half": True,
        "int8": False,
        "dynamic": False,
        "simplify": False,
        "opset": None,
        "workspace": 4,
        "nms": False,
        "fraction": 1.0,
    },
}

args_logging = {
    "project": "YOLO/Training",
    "name": "training-yolo",
}

# Console verbosity. A dict of its own on purpose: config_clearml() does
# `args_train.update(args_logging)` and passes args_train straight to
# model_yolo.train(), where ultralytics rejects unknown keys --
# "SyntaxError: 'log_level' is not a valid YOLO argument" would break every run.
#
# Only takes effect from the point train.py applies it, so anything logged
# before that (init_clearml, imports) still follows $LOG_LEVEL / the default.
args_console = {
    "log_level": "",  # empty = follow $LOG_LEVEL, which defaults to INFO
    "progress": "auto",  # auto (bars on a TTY only) | on | off
}

args_task = {
    "model_name": "yolo11n-seg",
    "model_latest_id": "",
    # Where a *locally launched* run creates its task. Read by init_clearml() before
    # Task.connect(), so editing these two in the ClearML UI renames nothing -- a task
    # that already exists keeps the project it was created in, and a clone inherits it.
    # Move an existing task with the UI's own project field instead.
    "clearml_project": "YOLO/Training",
    "clearml_task_name": "yolo-train",
    # "pretrained": "(model_id)"
}

args_data = {
    "cvat": {
        # Two ways to name sources, and they add up: list tasks directly, or name
        # whole projects and let every task in them be fetched. Projects are the
        # simpler knob -- a task added in CVAT later is then picked up on the next
        # run instead of being silently left out of training.
        "task_ids_train": [741, 733, 731, 728],
        "task_ids_test": [730],
        # Every task of these projects. Whatever ends up in the test set is
        # subtracted from training, so setting one task of a training project as
        # `task_ids_test` does the obvious thing rather than leaking it into both.
        "project_ids_train": [],
        "project_ids_test": [],
    },
    "label_studio": {
        "project_id_train": None,
        "project_id_test": None,
    },
    "s3": {"s3_uri_dir_train": None, "s3_uri_dir_test": None},
    # How the training pool is split. The three are a partition: every downloaded
    # pair lands in exactly one split. `test_ratio` is the other way to get a
    # `test/` directory, so it is mutually exclusive with `task_ids_test` /
    # `project_ids_test` above and refused if both are set. Leave it None and the
    # ratios are a two-way split: `valid_ratio` is then ignored, because valid/
    # takes everything training does not. Set it and all three must sum to 1.
    "params": {
        "train_ratio": 0.8,
        "val_ratio": 0.2,
        "test_ratio": None,
    },
    "class_exclude": "stalk, foreign_object",
    # --- class order --------------------------------------------------------------
    # CVAT numbers categories per project, so the same label sits at a different
    # position in every task, and a task that never saw a class simply omits it.
    # Every task is merged into one dataset directory, so the old per-source
    # `category_id - 1` wrote "car" and "speed_limit" to the same index. With this
    # on, one name -> index map is built from the union of all sources -- train and
    # test -- and every task is converted through it. Off restores the per-source
    # numbering and exists only to reproduce an older run. See src/data/class_map.py.
    "unify_class_order": True,
    # Pinned order, comma-separated. Empty = the union of every source, sorted:
    # reproducible across runs, but *not* across a change to the class list -- a new
    # name sorts into the middle and shifts every index after it. Pin this before
    # fine-tuning on top of an existing checkpoint, matching that model's `names`.
    "class_names": "",
    # A class the pinned list does not mention: `error` or `drop`. Unreachable when
    # class_names is empty, since a derived map contains every class by construction.
    "on_unknown_class": "error",
    # "attributes_exclude": {"maturity_truth": "background"},  # noqa: ERA001
    "attributes_exclude": None,
    "area_segment_min": 0,
}


args_train = {
    "augment": True,
    "epochs": 20,  # number of epochs to train for
    "patience": 0,  # epochs to wait for no observable improvement for early stopping of training # noqa: E501
    "batch": 64,  # number of images per batch (-1 for AutoBatch)
    "imgsz": 640,  # size of input images as integer or w,h
    "save": True,  # save train checkpoints and predict results
    "save_period": -1,  # Save checkpoint every x epochs (disabled if < 1)
    "cache": True,  # True/ram, disk or False. Use cache for data loading
    "device": None,  # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu # noqa: E501
    "workers": 8,  # number of worker threads for data loading (per RANK if DDP)
    "project": None,  # project name
    "name": None,  # experiment name
    "exist_ok": True,  # whether to overwrite existing experiment
    "pretrained": True,  # whether to use a pretrained model
    "optimizer": "auto",  # optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]  # noqa: E501
    "verbose": True,  # whether to print verbose output
    "seed": 0,  # random seed for reproducibility
    "deterministic": True,  # whether to enable deterministic mode
    "single_cls": False,  # train multi-class data as single-class
    "rect": False,  # rectangular training with each batch collated for minimum padding
    "cos_lr": False,  # use cosine learning rate scheduler
    "close_mosaic": 0,  # (int) disable mosaic augmentation for final epochs
    "resume": False,  # resume training from last checkpoint
    "amp": True,  # Automatic Mixed Precision (AMP) training, choices=[True, False]
    "fraction": 0.9,  # dataset fraction to train on (default is 1.0, all images in train set) # noqa: E501
    "profile": False,  # profile ONNX and TensorRT speeds during training for loggers
    "lr0": 0.001,  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    "lrf": 0.0001,  # final learning rate (lr0 * lrf)
    "momentum": 0.937,  # SGD momentum/Adam beta1
    "weight_decay": 0.0005,  # optimizer weight decay 5e-4
    "warmup_epochs": 3.0,  # warmup epochs (fractions ok)
    "warmup_momentum": 0.8,  # warmup initial momentum
    "warmup_bias_lr": 0.1,  # warmup initial bias lr
    "box": 7.5,  # box loss gain
    "cls": 0.5,  # cls loss gain (scale with pixels)
    "dfl": 1.5,  # dfl loss gain
    "pose": 12.0,  # pose loss gain (pose-only)
    "kobj": 2.0,  # keypoint obj loss gain (pose-only)
    "label_smoothing": 0.0,  # label smoothing (fraction)
    "nbs": 64,  # nominal batch size
    "overlap_mask": True,  # masks should overlap during training (segment train only)
    "mask_ratio": 4,  # mask downsample ratio (segment train only)
    "dropout": 0.0,  # use dropout regularization (classify train only)
    "val": True,  # validate/test during training
}

args_val = {
    "batch": 16,  # number of images per batch (-1 for AutoBatch)
    "save_json": False,  # save results to JSON file
    "save_hybrid": False,  # save hybrid version of labels (labels + additional predictions)  # noqa: E501
    # 0.001 is the ultralytics *val* default (cfg/default.yaml:55), and it is not a
    # cosmetic choice: args_val["conf"] is passed straight into NMS
    # (detect/val.py:118), so anything below it is discarded before the metrics ever
    # see it. The 0.25 that used to be here truncated the high-recall tail of every PR
    # curve and understated mAP by ~12% relative on a realistic confidence
    # distribution. It also blinded the operating-point and calibration reports below
    # 0.25 -- they could not recommend or measure a threshold they never saw.
    # The confusion matrix wants the opposite value; it is pinned separately by
    # on_val_batch_start. See args_visualization["confusion_matrix_conf"].
    "conf": 0.001,
    "iou": 0.7,  # intersection over union (IoU) threshold for NMS
    # 300 is the ultralytics default. 100 caps recall on crowded images and so lowers
    # mAP independently of `conf`. Check "Instances per image" in the Dataset report
    # before lowering it again.
    "max_det": 300,
    "half": True,  # use half precision (FP16)
    "device": 0,  # device to run on, i.e. cuda device=0/1/2/3 or device=cpu
    "dnn": False,  # use OpenCV DNN for ONNX inference
    "plots": True,  # show plots during training
    "rect": False,  # rectangular val with each batch collated for minimum padding
    "save_crop": True,  # save cropped images
    "split": "val",  # dataset split to use for validation, i.e. 'val', 'test' or 'train'
    # One console line per class from models/yolo/detect/val.py; the same numbers
    # are logged to ClearML as a per-class table. Re-enabled at LOG_LEVEL=DEBUG.
    "verbose": False,
}


args_predict = {
    "max_images": 40,  # maximum number of images to predict
    "plot": {
        "conf": True,
        "kpt_radius": 5,
        "kpt_line": True,
        "labels": False,
        "boxes": True,
        "masks": True,
        "probs": True,
        "color_mode": "instance",  # class or instance
        "txt_color": (255, 255, 255),
    },
    "model": {
        "batch": 16,
        "conf": 0.25,  # object confidence threshold for detection
        "iou": 0.7,  # intersection over union (IoU) threshold for NMS
        "max_det": 1000,  # maximum number of detections per image
        "stream": True,  # stream mode for real-time inference
        # One line per image from ultralytics/engine/predictor.py -- 40 of them,
        # the single biggest block of noise in the run. Re-enabled at DEBUG.
        # Note this is per-call verbose, not YOLO_VERBOSE=0: the env var drops
        # every ultralytics logger to ERROR and would take the per-epoch metric
        # table and the AMP/dataset warnings with it.
        "verbose": False,
    },
}

args_visualization = {
    "log_interactive_confusion_matrix": True,  # Use interactive report_confusion_matrix()
    "log_per_class_table": True,  # Log per-class metrics as DataFrame table
    "log_interactive_pr_curves": True,  # Plotly PR/F1/P/R-vs-conf curves, box and mask
    "log_confidence_histograms": True,  # Log confidence score distributions
    "log_learning_rate": True,  # Track learning rate per epoch
    "log_loss_components": True,  # Separate box, cls, dfl loss logging
    "log_speed_metrics": True,  # Log preprocess, inference, postprocess times
    "log_per_class_scatter": True,  # Log per-class metric scatter/bar plots
    # Display threshold for the confusion matrix only, held independent of
    # args_val["conf"]. Ultralytics feeds one value to both NMS and the matrix, but they
    # want opposite things: metrics need a floor near zero, a readable matrix needs
    # ~0.25 or it fills with low-confidence detections. Pinned in on_val_batch_start.
    "confusion_matrix_conf": 0.25,
    # Re-upload ultralytics' own PNGs (results, labels, labels_correlogram) as plots.
    # The curve and confusion-matrix PNGs are no longer sent at all: each duplicated an
    # interactive plot we report ourselves. Turn this off entirely if the ClearML
    # fileserver cannot serve images, since a bad image URL renders as a blank panel
    # with no error anywhere.
    "log_static_plots": True,
    # --- per-epoch, cheap ---------------------------------------------------------
    # mask mAP minus box mAP. Segmentation runs only; a no-op on a detect model.
    "log_mask_box_gap": True,
    # --- once per validation run, heavier -----------------------------------------
    # F1-optimal confidence, global and per class. This is the threshold to deploy at,
    # and it is none of the other three thresholds visible in the task (mAP uses 0.001,
    # the confusion matrix 0.25/IoU-0.45, args_val["conf"] only filters validation).
    "log_optimal_confidence": True,
    # Worst-scoring validation images as a table, plus their GT/FP/TP/FN panels when
    # args_val["visualize"] produced them. Ranked by mask F1 on a segmentation run.
    "log_worst_images": True,
    "worst_images_limit": 16,
    # TP-vs-FP confidence split and the reliability diagram. Requires the
    # on_val_batch_end capture, because ultralytics clears metrics.stats immediately
    # after processing them.
    "log_calibration": True,
}
