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

# Linting and formatting (via pre-commit or directly)
uv run ruff check --fix     # Lint with auto-fix
uv run ruff format          # Format code

# Docker
make build-v2               # Build Python 3.12 image
make run-docker-v2          # Run in Docker with GPU

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

- **src/params.py**: Default parameters for all pipeline stages
- **src/config.py**: Environment variables via Pydantic Settings (CVAT credentials, etc.)
- Parameters can be overridden in ClearML UI after first run

## Code Style

- Python 3.12 required
- Line length: 90 characters
- Double quotes for strings
- Ruff for linting/formatting with rules: E, F, I, UP, B, W, C90, N, D, PYI, PT, RET, SIM, ARG, ERA
- Pre-commit hooks configured

## ClearML Integration Points

- **src/utils/clearml_settings.py**: Task initialization and parameter connection
- **src/yolov8/callbacks.py**: Custom training callbacks for metric logging, debug samples, model registration
- **src/utils/register_model.py**: Model registration with metadata
