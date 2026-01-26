"""Utility functions for extracting metrics from Ultralytics trainer/validator.

This module provides helper functions to extract various metrics data in formats
suitable for ClearML logging, including per-class metrics, confusion matrix data,
loss components, learning rates, and speed metrics.
"""

from typing import Any

import numpy as np
import pandas as pd

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.engine.validator import BaseValidator


def extract_per_class_metrics(
    validator: BaseValidator,
    class_names: list[str] | None = None,
) -> pd.DataFrame | None:
    """Extract per-class precision, recall, and mAP metrics from validator.

    Args:
        validator: The Ultralytics validator instance after validation.
        class_names: Optional list of class names. If None, uses indices.

    Returns:
        DataFrame with columns: Class, P, R, mAP50, mAP50-95 or None if unavailable.

    """
    metrics = validator.metrics
    if not hasattr(metrics, "box") or metrics.box is None:
        return None

    box_metrics = metrics.box
    n_classes = len(box_metrics.p) if hasattr(box_metrics, "p") else 0

    if n_classes == 0:
        return None

    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    zeros = [0] * n_classes
    data = {
        "Class": class_names[:n_classes],
        "Precision": box_metrics.p.tolist() if hasattr(box_metrics, "p") else zeros,
        "Recall": box_metrics.r.tolist() if hasattr(box_metrics, "r") else zeros,
        "mAP50": box_metrics.ap50.tolist() if hasattr(box_metrics, "ap50") else zeros,
        "mAP50-95": box_metrics.ap.tolist() if hasattr(box_metrics, "ap") else zeros,
    }

    return pd.DataFrame(data)


def extract_confusion_matrix_data(
    validator: BaseValidator,
    class_names: list[str] | None = None,
) -> tuple[np.ndarray | None, list[str], np.ndarray | None]:
    """Extract confusion matrix data from validator.

    Args:
        validator: The Ultralytics validator instance after validation.
        class_names: Optional list of class names.

    Returns:
        Tuple of (normalized_matrix, labels, counts_matrix) or (None, [], None).

    """
    if not hasattr(validator, "confusion_matrix") or validator.confusion_matrix is None:
        return None, [], None

    cm = validator.confusion_matrix
    matrix = cm.matrix if hasattr(cm, "matrix") else None

    if matrix is None:
        return None, [], None

    # Get labels
    if class_names is not None:
        labels = list(class_names)
    elif hasattr(cm, "names") and cm.names:
        labels = list(cm.names.values()) if isinstance(cm.names, dict) else list(cm.names)
    else:
        labels = [f"class_{i}" for i in range(matrix.shape[0])]

    # Add background class if matrix is larger
    if len(labels) < matrix.shape[0]:
        labels.append("background")

    # Calculate normalized matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized_matrix = np.where(row_sums > 0, matrix / row_sums, 0)

    return normalized_matrix, labels, matrix


def extract_loss_components(trainer: BaseTrainer) -> dict[str, float]:
    """Extract individual loss components from trainer.

    Args:
        trainer: The Ultralytics trainer instance during training.

    Returns:
        Dictionary with loss component names and values.

    """
    losses = {}

    if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
        loss_names = getattr(trainer, "loss_names", ["box_loss", "cls_loss", "dfl_loss"])
        for name, value in zip(loss_names, trainer.loss_items, strict=False):
            losses[name] = float(value)

    # Also try to get from tloss
    if hasattr(trainer, "tloss") and trainer.tloss is not None:
        losses["total_loss"] = float(trainer.tloss)

    return losses


def extract_learning_rate(trainer: BaseTrainer) -> dict[str, float]:
    """Extract learning rate from trainer optimizer.

    Args:
        trainer: The Ultralytics trainer instance during training.

    Returns:
        Dictionary with parameter group indices and their learning rates.

    """
    learning_rates = {}

    if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
        for i, param_group in enumerate(trainer.optimizer.param_groups):
            lr = param_group.get("lr", 0.0)
            learning_rates[f"param_group_{i}"] = float(lr)

    return learning_rates


def extract_speed_metrics(validator: BaseValidator) -> dict[str, float]:
    """Extract inference speed metrics from validator.

    Args:
        validator: The Ultralytics validator instance after validation.

    Returns:
        Dictionary with speed metrics (preprocess, inference, postprocess) in ms.

    """
    speeds = {}

    if hasattr(validator, "speed") and validator.speed is not None:
        speed_dict = validator.speed
        if isinstance(speed_dict, dict):
            speeds["preprocess_ms"] = float(speed_dict.get("preprocess", 0))
            speeds["inference_ms"] = float(speed_dict.get("inference", 0))
            speeds["postprocess_ms"] = float(speed_dict.get("postprocess", 0))
            speeds["total_ms"] = sum(speeds.values())

    return speeds


def extract_pr_curve_data(
    validator: BaseValidator,
    class_names: list[str] | None = None,
) -> list[dict[str, Any]] | None:
    """Extract precision-recall curve data from validator.

    Args:
        validator: The Ultralytics validator instance after validation.
        class_names: Optional list of class names.

    Returns:
        List of dicts with keys: class_name, precision, recall, ap or None if unavailable.

    """
    metrics = validator.metrics
    if not hasattr(metrics, "box") or metrics.box is None:
        return None

    box_metrics = metrics.box

    # Check for PR curve data
    if not hasattr(box_metrics, "px") or not hasattr(box_metrics, "py"):
        return None

    px = box_metrics.px  # recall values (x-axis)
    py = box_metrics.py  # precision values per class (y-axis)

    if px is None or py is None:
        return None

    n_classes = py.shape[0] if len(py.shape) > 1 else 1

    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    pr_curves = []
    for i in range(min(n_classes, len(class_names))):
        precision = py[i] if len(py.shape) > 1 else py
        recall = px

        prec_list = (
            precision.tolist() if hasattr(precision, "tolist") else list(precision)
        )
        recall_list = recall.tolist() if hasattr(recall, "tolist") else list(recall)
        ap50_val = float(box_metrics.ap50[i]) if hasattr(box_metrics, "ap50") else 0.0

        pr_curves.append({
            "class_name": class_names[i],
            "precision": prec_list,
            "recall": recall_list,
            "ap50": ap50_val,
        })

    return pr_curves


def collect_prediction_confidences(
    results: list,
    class_names: list[str] | None = None,
) -> tuple[list[float], dict[str, list[float]]]:
    """Collect confidence scores from prediction results.

    Args:
        results: List of Ultralytics prediction results.
        class_names: Optional list of class names.

    Returns:
        Tuple of (all_confidences, per_class_confidences).

    """
    all_confidences = []
    per_class_confidences: dict[str, list[float]] = {}

    for r in results:
        boxes = r.boxes if hasattr(r, "boxes") else None
        if boxes is None:
            continue

        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        classes = boxes.cls.cpu().numpy() if boxes.cls is not None else []

        for conf, cls_id in zip(confs, classes, strict=False):
            conf_value = float(conf)
            all_confidences.append(conf_value)

            cls_idx = int(cls_id)
            if class_names and cls_idx < len(class_names):
                cls_name = class_names[cls_idx]
            else:
                cls_name = f"class_{cls_idx}"

            if cls_name not in per_class_confidences:
                per_class_confidences[cls_name] = []
            per_class_confidences[cls_name].append(conf_value)

    return all_confidences, per_class_confidences
