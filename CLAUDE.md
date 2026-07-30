# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A template for training Ultralytics YOLOv8/YOLO11 models with ClearML integration for experiment tracking, dataset management, model registration, and remote execution. Supports multiple data sources (CVAT, S3/MinIO) with COCO-to-YOLO conversion.

## Commands PYthon
always using uv not python to running python
```
PYTHONPATH=. uv run {path}.py
```


## Common Commands

```bash
# Run training locally
make run                    # PYTHONPATH=. uv run src/train.py

# Run tests
make test_code              # pytest tests -v
make test_fast              # skip tests that invoke real exporters

# Test the export stage on its own (~7s, no CVAT, no training, no weights download)
PYTHONPATH=. uv run pytest tests/yolov8/test_export_smoke.py

# Linting and formatting (via pre-commit or directly)
uv run ruff check --fix     # Lint with auto-fix
uv run ruff format          # Format code

# Docker
make image-name             # Print the resolved image reference
make build                  # Build the training image (yolo-trainer:<version>)
make run-docker             # Run in Docker with GPU
make push                   # Only meaningful with a registry path (see below)

# Build/push under a registry instead of the bare local name
make build push TRAINER_IMAGE=ghcr.io/acme/yolo-trainer:0.2.0

# Export requirements
make get-req                # Generate requirements.txt from uv
```

## Architecture

### Pipeline Flow (src/train.py)

```
1. init_clearml()           → Create/connect ClearML task
2. config_clearml()         → Connect parameters from UI
3. DataHandler.export()     → Download & convert datasets
4. YOLO.train()             → Train with custom callbacks
5. YOLO.val()               → Validate model
6. export_handler()         → Export to multiple formats (ONNX, TensorRT, etc.)
7. _predicting_result()     → Run predictions & log grids to ClearML
```

### Data Flow

```
CVAT/S3 (COCO format)
    ↓
DownloaderFactory (src/data/downloader/)
    ↓
Coco2Yolo Converter (src/data/converter/)
    ↓
Dataset Split & YAML (src/data/setup.py)
    ↓
YOLO Training
```

### Key Directories

- **src/data/**: Data pipeline (downloaders, converters, dataset setup)
  - `downloader/method/`: CVAT, S3, MinIO implementations
  - `converter/coco2yolo.py`: COCO→YOLO format conversion
- **src/yolov8/**: YOLO-specific code (callbacks, exporter)
- **src/schema/**: Pydantic models for parameter validation
- **src/utils/**: ClearML settings, model registration helpers

### Configuration

- **src/params.py**: Default parameters for all pipeline stages, plus `DOCKER_IMAGE` /
  `DOCKER_ARGUMENTS` — the single source of truth for the training image. The Makefile
  reads `DOCKER_IMAGE` back out, so `make build` and `set_base_docker()` cannot drift.
  Override with `TRAINER_IMAGE=...`; keep the tag in step with `version` in
  `pyproject.toml`.
- **src/config.py**: Environment variables via Pydantic Settings (CVAT credentials, etc.)
- Parameters can be overridden in ClearML UI after first run — except
  `clearml_project` / `clearml_task_name`, which are read before `Task.connect()` and
  therefore only apply to a locally launched first run.

## Code Style

- Python 3.14 required
- Line length: 90 characters
- Double quotes for strings
- Ruff for linting/formatting with rules: E, F, I, UP, B, W, C90, N, D, PYI, PT, RET, SIM, ARG, ERA
- Pre-commit hooks configured

## Gotchas When Running on an Agent

- **The image shadows `src/`.** The container sets `PYTHONPATH=/workspace` and bakes a
  copy of `src/` in, so `src.yolov8.*` resolves to the **image's** copy while
  `src/train.py` runs from the git checkout. Editing anything under `src/yolov8/` has no
  effect on a remote run until the image is rebuilt (`make build`). The two can drift
  silently.
- **The container runs as root on purpose.** clearml-agent's docker-mode bootstrap writes
  to `/etc/apt`, `/root/.cache/pip` and `/root/.ssh` and runs `apt-get`. Adding a
  `USER` directive to the Dockerfile makes every remote task hang before reaching
  `src/train.py`.
- **`torch` is pinned to a CUDA 12.x wheel window** (`>=2.9,<2.10`). Newer torch resolves
  to a CUDA 13 wheel that needs a much newer driver and, if unmet, silently falls back to
  CPU instead of erroring. Check every agent host's driver before widening it.
- **A task pins its own commit.** `version_num` on an existing task wins over its branch
  name, so editing the branch alone does not move it to newer code; clear the commit to
  follow the branch. Agents also cache clones per repo, and a stale cache can write an old
  branch name back onto the task after execution.
- **Deprecated export args.** `args_export["params"]` still uses `half` and `int8`, which
  ultralytics has replaced with `quantize`. Tolerated today; `tests/yolov8/test_export_smoke.py`
  fails when that stops being true.
- **Never re-tag a published image version.** A ClearML task stores the image tag it was
  created with and keeps requesting it, so overwriting `yolo-trainer:0.2.0` silently
  changes what every past task reruns on. Bump `version` in `pyproject.toml` and the tag
  in `DOCKER_IMAGE` together instead.
- **Python 3.14 defaults `multiprocessing` to `forkserver` on Linux**, not `fork`.
  Forkserver re-imports the `__main__` module in every dataloader worker, so anything
  that runs at module scope in `src/train.py` would re-run per worker — re-initialising
  ClearML and re-downloading datasets. The `if __name__ == "__main__":` guard is what
  prevents that; keep all work inside `main()`.
- **`set_base_docker()` replaces the whole container section.** Passing `docker_image`
  without `docker_arguments` drops the `CLEARML_AGENT_SKIP_*` env vars, and the agent
  then ignores the image's baked venv and rebuilds one with pip. Always pass both.

## ClearML Integration Points

- **src/utils/clearml_settings.py**: Task initialization and parameter connection
- **src/yolov8/callbacks.py**: Custom training callbacks for metric logging, debug samples, model registration
- **src/utils/register_model.py**: Model registration with metadata
