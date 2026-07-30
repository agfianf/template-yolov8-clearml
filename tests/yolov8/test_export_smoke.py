"""Fast smoke tests for the export stage.

Export needs nothing but a model file, so these run without CVAT, without
training and without downloading weights -- the model is built from its yaml
config. A few seconds each, versus ~15 minutes to reach the export stage
through a full pipeline run.

Marked `slow` because they still invoke real ONNX/OpenVINO exporters. Run them
with `pytest -m slow`, or skip with `pytest -m "not slow"`.
"""

from pathlib import Path

import pytest

from ultralytics import YOLO

from src.params import args_export
from src.yolov8.exporter import export_model_format, filter_export_parameters


# Small model + small imgsz keeps each export to a couple of seconds. Built from
# yaml rather than a .pt so no network access is needed; random weights are fine
# because we assert on the exporter, not on predictions.
MODEL_CONFIG = "yolov8n.yaml"
IMGSZ = 320


@pytest.fixture(scope="module")
def model() -> YOLO:
    """Untrained model built from config -- no weights download."""
    return YOLO(MODEL_CONFIG)


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run each export in its own directory; ultralytics writes next to cwd."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.slow
@pytest.mark.parametrize("format_model", ["onnx", "openvino"])
def test_export_with_project_params(
    model: YOLO,
    workdir: Path,
    format_model: str,
) -> None:
    """Export succeeds using the parameters the pipeline actually sends.

    Doubles as a deprecation guard: args_export carries `half` and `int8`, both
    of which ultralytics has replaced with `quantize` and dropped from
    default.yaml. They are still tolerated, so this passes -- when a future
    release stops accepting them it fails here rather than mid-training.
    """
    exported = Path(
        str(export_model_format(model, format_model, IMGSZ, dict(args_export["params"])))
    ).resolve()

    assert exported.exists(), f"{format_model} produced no artifact"
    assert exported.is_relative_to(workdir), f"{format_model} wrote outside the tmp dir"


@pytest.mark.parametrize("format_model", ["onnx", "openvino", "torchscript"])
def test_filter_export_parameters_drops_unsupported(format_model: str) -> None:
    """Only parameters the format understands survive filtering.

    No export involved, so this stays fast and always runs.
    """
    filtered = filter_export_parameters(format_model, dict(args_export["params"]))

    assert filtered, f"{format_model} got no parameters at all"
    assert set(filtered) <= set(args_export["params"]), "filter invented a parameter"
    assert "keras" not in filtered, "keras is not valid for these formats"


def test_engine_format_bypasses_filtering() -> None:
    """TensorRT is special-cased in the exporter and keeps every parameter."""
    params = dict(args_export["params"])

    assert filter_export_parameters("engine", params) == params
