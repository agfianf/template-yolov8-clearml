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

# Test the data stage on its own (<1s, builds a mini COCO tree in tmp_path).
# Also asserts log volume does not grow with dataset size.
PYTHONPATH=. uv run pytest tests/data/test_data_stage_smoke.py

# Linting and formatting (via pre-commit or directly)
uv run ruff check --fix     # Lint with auto-fix
uv run ruff format          # Format code

# Docker
make image-name             # Print the resolved image reference
make bump                   # ./VERSION 0.2.2 -> 0.2.3 (PART=minor|major for the rest)
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
  Override with `TRAINER_IMAGE=...`.
- **VERSION**: the image tag, and the only place the version is written.
  `make bump PART=patch|minor|major` before any build whose image contents changed.
  It is deliberately *not* in `pyproject.toml` — see the Versioning note below.
- **src/config.py**: Environment variables via Pydantic Settings (CVAT credentials, etc.)
- Parameters can be overridden in ClearML UI after first run — except
  `clearml_project` / `clearml_task_name`, which are read before `Task.connect()` and
  therefore only apply to a locally launched first run.

## Logging

Everything goes through `src/utils/logging.py`. `print()` is banned — ruff `T20`
enforces it and `tests/utils/test_no_print.py` covers the `from rich import print`
alias that `T20` cannot see.

| Level | Meaning |
|---|---|
| ERROR | the run is affected: an export failed, validation crashed |
| WARNING | degraded but continuing: CVAT export timed out, unpaired labels |
| INFO | stage boundaries, and **one summary line per unit of work** |
| DEBUG | per-item detail: each filter decision, raw response bodies |

**The one rule: no INFO line may be emitted from inside a loop over dataset items.**
Tally with `src.utils.logging.Tally` and log one summary. Log volume must not grow
with dataset size; `tests/data/test_data_stage_smoke.py` asserts exactly that by
running each stage at two input sizes and expecting the same line count.

Turning the volume up:

- `LOG_LEVEL=debug` (or `10`) as an env var. An unreadable value falls back to INFO
  and says so.
- The `0_Console` parameter group in the ClearML UI: `log_level` (empty = follow
  `$LOG_LEVEL`) and `progress` (`auto` | `on` | `off`). At DEBUG, ultralytics' own
  per-class validation output and per-image predict output come back too.

Two caveats worth knowing before chasing a missing line:

- **`0_Console` only applies from `src/train.py`'s call to `set_log_level()`.**
  Anything logged earlier — `init_clearml()`, import-time lines in `src/config.py` —
  follows the env var or the default, whatever the UI says.
- **`task.execute_remotely()` kills the local process on the next line.** A value set
  in the UI therefore only ever takes effect on the agent — which is the one console
  we actually read.

Measuring against the baseline (334 lines of our own output on a 3-epoch, 4-CVAT-task
run): `grep -cE '\| src\.' console.txt`. The `%(name)s` field in the format string is
what makes that a one-command check.

## Reporting to ClearML

Three modules, one direction of dependency:
`metrics_utils.py` (read ultralytics) → `clearml_logger.py` (report) →
`callbacks.py` (decide when). `dataset_report.py` does the same for the data stage.
Every report is gated by a flag in `args_visualization` (`src/params.py`), so it is
switchable from the `8_Visualization` group in the ClearML UI.

**Cadence rule, the sibling of the logging rule: report volume must not grow with
epoch count.** Scalars are series and are meant to grow one point per epoch; plots,
tables and image galleries are not. Anything heavy belongs in
`report_validation_analysis()`, which runs once per validation pass.
`tests/yolov8/test_callbacks.py::TestReportVolume` asserts this by running the
per-epoch hooks at 3 and 30 epochs and requiring an identical heavy-report count.

### Ultralytics facts that are easy to get wrong

Read these before touching `metrics_utils.py`; each one is a bug we shipped or nearly
shipped, and each has a regression test that fails if it is undone.

- **Per-class arrays are indexed by `ap_class_index`, not by class id.** It contains
  only classes with ground truth in the split, so slicing a `names` list against
  `box.p` mislabels every row after an absent class. Use `metrics.summary()`, which
  resolves names via `names[ap_class_index[i]]` — and on `SegmentMetrics` also yields
  `Mask-P/R/F1`. Per-class mask *mAP* is not in `summary()`; it is `seg.ap50` / `seg.ap`.
- **The PR-curve attribute is `prec_values`, not `py`.** Reading `py` returns nothing
  and the plot silently never appears. Prefer `metrics.curves_results`, which returns
  `[x, y, xlabel, ylabel]` per family with the axis labels included, paired positionally
  with `metrics.curves` for the `(B)` / `(M)` name.
- **`SegmentMetrics.fitness` is `seg.fitness() + box.fitness()`** — a 0–2 sum of two
  mAP50-95 values, not a 0–1 metric. For detection it is exactly mAP50-95.
- **`SegmentMetrics.maps` is an element-wise sum** of box and mask per class, not a
  concatenation. Any per-class chart built from it is meaningless; use `summary()`.
- **Validation metrics are stale in `on_train_epoch_end`.** That hook fires at
  `engine/trainer.py:569`; `validate()` runs at `:577`. Report them from
  `on_fit_epoch_end` (`:605`) or epoch 0 shows zeros and epoch N shows N−1.
- **`metrics.stats` is cleared one line after it is processed** (`get_stats()`,
  `detect/val.py:277-278`). Anything needing raw per-detection data — a TP-vs-FP
  confidence split, a calibration curve — must capture it in `on_val_batch_end`, the
  last hook before that. `ValStatsAccumulator` exists for this.
- **The confusion matrix is `matrix[predicted, ground_truth]`, normalized by column**
  (`sum(0)`). Normalizing by row gives precision-like numbers that contradict the
  `confusion_matrix_normalized.png` uploaded under the same title.
- **The confusion matrix and the TP/FP/FN gallery are box-IoU based even for
  segmentation** — `SegmentationValidator` inherits `update_metrics`, so
  `process_batch` never sees masks. Label those plots "Box". Mask-aware ranking comes
  from `metrics.seg.image_metrics`, which *is* fed by `mask_iou`.
- **`args_val["conf"]` goes into NMS, so it silently changes mAP.** It is passed to
  `non_max_suppression` (`detect/val.py:118`), and anything filtered there never reaches
  the metrics. It must stay at `0.001` — the ultralytics val default. The `0.25` that was
  here truncated the high-recall tail of every PR curve, understated mAP by ~12% relative,
  and blinded the operating-point and calibration reports below 0.25. `max_det` is the
  same kind of trap: 100 caps recall on crowded images, so it is 300 (the default).
- **The confusion matrix wants the opposite threshold, and is pinned separately.**
  Ultralytics feeds one `args.conf` to both NMS and the matrix (`detect/val.py:195`), but
  metrics need a floor near zero while a readable matrix needs ~0.25 or it fills with
  low-confidence detections. `on_val_batch_start` wraps `process_batch` to hold it at
  `args_visualization["confusion_matrix_conf"]`. Its `iou_thres=0.45` is hard-coded
  upstream and is *not* `args_val["iou"]`.
- **Three different thresholds therefore coexist** and read as contradictions if
  unlabelled: mAP and the curves at 0.001, the matrix at 0.25, and the P/R/F1 scalars
  ultralytics prints at each class's max-F1 confidence. The threshold to actually deploy
  at is reported separately under `Operating Point`.
- **Mask mAP is computed at quarter resolution unless `save_json` or `save_txt` is set.**
  `segment/val.py:74` picks `process_mask` (prototype resolution, 160×160 at
  `imgsz=640`) versus `process_mask_native`, and `:128` downsamples the ground-truth
  masks to match. Both are self-consistent, but mask mAP from the two settings is **not
  comparable** — do not read a jump in mask mAP as a model improvement if `save_json`
  changed. This also means the mask-vs-box gap and per-class mask mAP inherit that
  resolution dependence.
- **`args_val["visualize"]` is uncapped**: it writes one GT/FP/TP/FN panel per
  validated image into `<save_dir>/visualizations`. It is enabled for the single final
  `val()` only, and only the worst N panels are uploaded.
- **ClearML image retention is per title/series.** Put the rank in the series
  (`worst-box-00`) and the epoch in `iteration`, so each rank slot keeps its own
  history. Folding the filename into the series mints a new series whenever a different
  image becomes the worst.

### Deliberately not implemented

No new dependencies were added, which rules out three things worth knowing about:

- **AP small/medium/large and AR@k.** Ultralytics has this in `coco_evaluate`
  (`detect/val.py:475`) but gates it on `save_json and (is_coco or is_lvis)` — never
  true for CVAT/S3 data — and it needs `faster-coco-eval`, which has no cp314 wheel and
  would be source-built inside the agent container. Without it there is no principled
  answer to "is this a small-object problem?".
- **TIDE's six-way error typology** (Cls/Loc/Both/Dup/Bkg/Miss) — needs `tidecv`. It
  does *not* need COCO JSON; the barrier is only the dependency. The worst-N TP/FP/FN
  galleries are the qualitative stand-in.
- **Boundary IoU / boundary F-score.** Mask AP under-penalizes boundary error, so mask
  sharpness is genuinely unmeasured here.

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
  changes what every past task reruns on. `make bump PART=patch` instead — it edits
  `./VERSION`, which `src/params.py` turns into the tag.
- **Anything that varies per release goes at the *bottom* of the Dockerfile.** A changed
  `--build-arg` invalidates every instruction below the one that consumes it, and
  `IMAGE_VERSION` feeds a `LABEL`. With that block at the top, a version bump alone
  rebuilt `apt-get` (71s) and `uv sync` (352s): **495s for a five-byte change**. Moving
  the `ARG`/`LABEL` below the `COPY` steps took the same bump to **4s**. Both numbers
  measured on this repo; the control was holding `IMAGE_VERSION` fixed and changing only
  the image name, which built in 2.5s fully cached.
- **Keep the version out of `pyproject.toml` and `uv.lock`.** Those two are bind-mounted
  into the `uv sync` layer, and BuildKit keys that layer on their contents — a one-line
  change to a bind-mounted file does invalidate the `RUN` (verified in isolation). This
  was masked by the `ARG` problem above rather than being the cause of it, but it is a
  live trap once that is fixed. `pyproject.toml` declares `dynamic = ["version"]`, which
  uv accepts for a virtual project — which this is, nothing ever builds or publishes it —
  and `tests/utils/test_version.py` fails if a static version returns to either file.
- **Python 3.14 defaults `multiprocessing` to `forkserver` on Linux**, not `fork`.
  Forkserver re-imports the `__main__` module in every dataloader worker, so anything
  that runs at module scope in `src/train.py` would re-run per worker — re-initialising
  ClearML and re-downloading datasets. The `if __name__ == "__main__":` guard is what
  prevents that; keep all work inside `main()`.
- **Quieten ultralytics per call, never with `YOLO_VERBOSE=0`.** The env var is read
  once in `ultralytics/utils/__init__.py` and drops every ultralytics logger to
  ERROR — the per-epoch metrics table and the AMP/dataset warnings go with it.
  `args_val["verbose"]` and `args_predict["model"]["verbose"]` are the right knobs,
  and `src/train.py` turns them back on at `LOG_LEVEL=DEBUG`.
- **Console parameters live in `args_console`, not `args_logging`.**
  `config_clearml()` does `args_train.update(args_logging)` and `args_train` is
  splatted into `model_yolo.train()`, which rejects unknown keys outright
  (`SyntaxError: 'log_level' is not a valid YOLO argument`).
- **`set_base_docker()` replaces the whole container section.** Passing `docker_image`
  without `docker_arguments` drops the `CLEARML_AGENT_SKIP_*` env vars, and the agent
  then ignores the image's baked venv and rebuilds one with pip. Always pass both.

## ClearML Integration Points

- **src/utils/clearml_settings.py**: Task initialization and parameter connection
- **src/yolov8/callbacks.py**: Custom training callbacks for metric logging, debug samples, model registration
- **src/utils/register_model.py**: Model registration with metadata
