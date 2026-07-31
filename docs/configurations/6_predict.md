# `6_Predict` — `args_predict`

The `args_predict` dict in `src/params.py` is published to the ClearML UI by `curr_task.connect(args_predict, name="6_Predict")` in `src/utils/clearml_settings.py:78`, and appears in the experiment's **Configuration → Hyperparameters → 6_Predict** section. It controls the last stage of a run: `_predicting_result()` in `src/train.py:74-200`, which samples images from the evaluation split, runs the trained model over them, draws the detections, and uploads them to ClearML as 2×2 image grids under the title `Prediction`. It also, as a side effect, produces the predict-stage confidence histograms under `Distributions`. Nothing in this group affects any metric — mAP, the curves, the confusion matrix and the operating point are all fixed by the time this stage runs. It is a qualitative eyeball check, and every value in it should be read as a display setting.

## Quick reference

| Key | Default | Type | Meaning |
|---|---|---|---|
| `max_images` | `40` | int | How many images are sampled from the split and uploaded. |
| `plot.conf` | `True` | bool | Draw the confidence number on each detection. |
| `plot.kpt_radius` | `5` | int | Keypoint dot radius. Pose models only. |
| `plot.kpt_line` | `True` | bool | Draw skeleton lines between keypoints. Pose models only. |
| `plot.labels` | `False` | bool | Draw the class name on each box. Off by default — see the note below. |
| `plot.boxes` | `True` | bool | Draw bounding boxes. |
| `plot.masks` | `True` | bool | Draw segmentation masks. Segment models only. |
| `plot.probs` | `True` | bool | Draw the top-5 class probability block. Classify models only. |
| `plot.color_mode` | `"instance"` | str | `"instance"` colours each detection separately, `"class"` colours by class id. |
| `plot.txt_color` | `(255, 255, 255)` | tuple | Text colour for the classification probability block. Classify models only. |
| `model.batch` | `16` | int | Images per forward pass in the prediction loop. |
| `model.conf` | `0.25` | float | **Display** confidence threshold for the gallery. Not a metric threshold. |
| `model.iou` | `0.7` | float | NMS IoU threshold for the prediction pass. |
| `model.max_det` | `1000` | int | Maximum detections drawn per image. |
| `model.stream` | `True` | bool | Return a generator rather than a list from `predict()`. |
| `model.verbose` | `False` | bool | Ultralytics' one-line-per-image console output. Re-enabled at `LOG_LEVEL=DEBUG`. |

## Where the images come from

`_predicting_result()` picks the split the same way the final validation does, but independently of it. It reads the generated `data.yaml` and takes `test` if it exists, otherwise `val` (`src/train.py:83-87`), then recursively globs that directory for `*.jpg`, `*.jpeg`, `*.png`, `*.bmp`, `*.tif` and `*.tiff` (`:92-97`), shuffles the result with `random.shuffle`, and keeps the first `max_images` (`:99-103`). If nothing matches, it logs a warning and returns — the run still succeeds.

Two consequences worth knowing. First, the selection is random and there is no `seed` key in this group: which images end up in the gallery is not controlled by anything you can set here, so two otherwise identical runs will generally show different images. Second, the model used is **not** the one that was just trained in memory — `src/train.py:367` reloads `YOLO(model_yolo.trainer.best)` first, so the gallery always reflects the best checkpoint, which is the same weights that get registered to ClearML. The whole stage is wrapped in a `try`/`except` (`:366-377`), so a failure here logs a traceback and never costs you the trained model or the exports.

The images are then grouped four at a time, padded to a common size with black at the bottom and right edges, filled out with black tiles if the last group is short, tiled into a 2×2 grid, and reported with `title="Prediction"`, `series=f"image-{i}"` and `iteration=1` (`src/train.py:142-172`). With the default `max_images=40` that is ten grids, in series `image-3` through `image-39`. Each grid is its own ClearML series, which is what keeps all ten visible instead of the last one replacing the rest — ClearML image retention is per title/series.

## `max_images`

**What it does.** Caps how many images are sampled, predicted on, drawn and uploaded.

**Valid values.** Any positive integer. Values that are not multiples of four leave a partly black final grid, which is cosmetic only.

**Why the default.** Forty images is ten grids: enough to spot a systematic failure mode, small enough that the upload and the render are a rounding error against training time. It is also the number that makes `model.verbose` expensive — see below.

**What goes wrong if set wrong.** Raising it scales three things linearly: inference time, upload volume, and — if `model.verbose` is on — console lines. Setting it very high on a large split turns a five-second stage into a second validation pass with no metrics to show for it. Setting it to `0` empties the list before the emptiness check at `src/train.py:105`, so the stage logs `No validation images found` and returns — the same warning you would get from a genuinely empty split directory, which makes a zero here indistinguishable from a data-stage failure when reading the console.

## The `plot` sub-dict

Every key here is forwarded verbatim to `Results.plot()` (`src/train.py:136-138`, `ultralytics/engine/results.py:476`). They change only the pixels in the uploaded grid; none of them re-runs inference or changes what was detected.

### `conf` and `labels` — read these two together

This is the least obvious pair in the group, because ultralytics builds one label string from both. From `engine/results.py`, the label for each box is:

```python
label = (f"{name} {d_conf:.2f}" if conf else name) if labels else (f"{d_conf:.2f}" if conf else None)
```

So the four combinations are: `labels=True, conf=True` gives `person 0.87`; `labels=True, conf=False` gives `person`; `labels=False, conf=True` gives `0.87` — **the bare number with no class name**; and `labels=False, conf=False` gives no text at all.

The default is `labels=False, conf=True`, which means the shipped gallery shows confidence numbers and no class names. On a single-class dataset that is exactly right and keeps dense scenes readable. On a multi-class dataset it is usually the wrong choice — you cannot tell a misclassification from a correct detection — and `labels: True` is the first thing to change. Colour is the only remaining class signal in that configuration, and only if `color_mode` is `"class"`.

### `boxes`

**What it does.** Draws the bounding-box rectangles and their labels.

**Valid values.** `True` | `False`.

**Why the default.** `True` — boxes are the primary output of a detect model, and on a segment model they are what carries the label text.

**What goes wrong if set wrong.** Setting `False` on a detect model produces an empty image. On a segment model it produces masks with no outlines and no labels, which is a good way to judge mask quality and a bad way to judge classification.

### `masks`

**What it does.** Overlays the predicted segmentation masks, coloured according to `color_mode`.

**Valid values.** `True` | `False`.

**Why the default.** `True`. It is inert on a detect model, which has no masks to draw, so it costs nothing to leave on.

**What goes wrong if set wrong.** Setting `False` on a segmentation run hides the thing you are trying to inspect. Keep it on unless masks are so dense the underlying image is invisible — in which case turn `boxes` off instead and look at them separately.

### `probs`

**What it does.** Draws the top-5 class probability block in the corner. Only classification models populate `Results.probs`, so this is inert for detect, segment and pose.

**Valid values.** `True` | `False`.

**Why the default.** `True`, harmlessly.

**What goes wrong if set wrong.** Nothing, unless you are training a classifier, in which case `False` removes the only output there is to look at.

### `kpt_radius` and `kpt_line`

**What they do.** The radius of each drawn keypoint and whether the skeleton edges between keypoints are drawn. Pose models only.

**Valid values.** A positive integer, and a bool.

**Why the defaults.** `5` and `True` are the ultralytics defaults.

**What goes wrong if set wrong.** Nothing for a detect or segment model — these are read but there are no keypoints to draw. On a pose model a large radius on small subjects covers the person entirely.

### `color_mode`

**What it does.** Chooses the palette index for boxes and masks. `"class"` uses the class id, so every instance of a class shares a colour; `"instance"` uses the detection index within the image, so adjacent objects of the same class get different colours.

**Valid values.** `"class"` | `"instance"`.

**Why the default.** `"instance"`. On a dense single-class dataset — the case this template was built around — class colouring makes every object the same colour and adjacent instances merge into one blob; instance colouring is what lets you see that the model split or merged two objects.

**What goes wrong if set wrong.** On a multi-class dataset `"instance"` throws away the only colour signal that told you what class something is, and combined with the default `labels=False` the gallery carries no class information whatsoever. If you set `labels: True`, `"instance"` is fine again.

### `txt_color`

**What it does.** The colour of the classification probability text block, passed to `Annotator.text` (`engine/results.py`, the "Plot Classify results" branch). It is documented upstream as BGR. Box labels do **not** use it — their text colour is chosen automatically for contrast against the box colour.

**Valid values.** A 3-tuple of 0–255 integers.

**Why the default.** White on the dark box the annotator draws behind the text.

**What goes wrong if set wrong.** Nothing on a detect or segment model, where it is never read. Note that this is the only non-scalar value in the group, so it is also the one most likely to be mangled by a round-trip through the ClearML UI's text field; if the gallery starts failing to render on a remote run only, check this value first.

## The `model` sub-dict

These keys are splatted into `model_yolo.predict()` (`src/train.py:121-126`), alongside a `source` of the sampled paths, an `imgsz` taken from `args_val["imgsz"]` — which itself was set from `args_train["imgsz"]` at `src/train.py:307` — and a hard-coded `device="0" if torch.cuda.is_available() else "cpu"`. There is no `device` key in this group and `args_val["device"]` is not consulted here.

> **`model.conf` is not the same `conf` as in `5_Testing`.**
>
> `args_predict["model"]["conf"] = 0.25` is a **display** threshold. It decides which detections are drawn in the `Prediction` grid, and nothing else. `args_val["conf"] = 0.001` is a **metric** threshold: it goes into NMS during validation, and everything below it is discarded before mAP, the PR curves, the calibration diagram and the operating point are computed. The two numbers are different on purpose, they cannot be unified, and confusing them is the most common misreading of a finished task. If you want a cleaner gallery, change this one. If you change `args_val["conf"]` instead, you will silently lower the reported mAP — see [`5_testing.md`](5_testing.md).
>
> One knock-on effect is easy to miss. The predict-stage confidence histograms are computed from these same filtered results by `collect_prediction_confidences` (`src/train.py:178-200`), so they contain **only detections above 0.25** and are reported under the title `Distributions` with series `All Classes - Confidence` and `Distributions/Per-Class`. The validation-stage TP-vs-FP confidence split is also reported under `Distributions` — but it comes from `ValStatsAccumulator`, captured during validation at `conf=0.001`, so it covers the entire range. Two panels, one title group, two different populations. Read the series name, not just the title.

### `conf`

**What it does.** The confidence threshold for the prediction pass. It is the ultralytics predict-mode default, and `Model.predict()` would apply `0.25` even if this key were absent (`engine/model.py:535`).

**Valid values.** `0.0`–`1.0`.

**Why the default.** A gallery drawn at 0.001 is a wall of overlapping low-confidence boxes with no information in it.

**What goes wrong if set wrong.** Too high and the gallery looks flawless while recall is poor — you see only the detections the model was already sure about. Too low and the grid becomes unreadable and the `Distributions` histogram grows a huge spike near zero that is not comparable to anything from previous runs. If you want to know what threshold to actually deploy at, do not tune this by eye: read the `Operating Point` report, which is the F1-optimal confidence computed from the validation curves.

### `iou`

**What it does.** NMS IoU threshold for the prediction pass.

**Valid values.** `0.0`–`1.0`. `0.7` matches `args_val["iou"]`, which is deliberate — the gallery should show roughly what validation measured.

**What goes wrong if set wrong.** Diverging from `args_val["iou"]` means the gallery shows a different duplicate-suppression behaviour from the one the metrics describe, so an apparent double-detection problem in the images may not exist in the numbers, or vice versa.

### `max_det`

**What it does.** Caps detections per image.

**Valid values.** Any positive integer. `1000` here versus `300` in `5_Testing`.

**Why the default.** At `conf=0.25` almost nothing survives NMS in the first place, so a cap of 1000 essentially never binds and the gallery is never silently truncated. The `300` in `5_Testing` exists because that pass runs at `conf=0.001`, where the cap does bind.

**What goes wrong if set wrong.** Lowering it below the real object count on crowded images makes the gallery look like the model is missing objects it actually found, which is a false alarm the metrics will not corroborate.

### `batch`

**What it does.** How many images are loaded and inferred per forward pass. It is handed to the inference source loader (`engine/predictor.py:267`).

**Valid values.** A positive integer. `Model.predict()` defaults it to `1` and caller kwargs win (`engine/model.py:535-536`), so `16` is an explicit speedup over the upstream default.

**What goes wrong if set wrong.** Only CUDA OOM, and only if it is raised well above 16 at a large `imgsz`. It has no effect on what is detected.

### `stream`

**What it does.** `stream=True` makes `predict()` return a generator instead of materialising a list of `Results` (`engine/predictor.py:216-234`).

**Valid values.** `True` | `False`.

**Why the default.** `True` is the correct habit for prediction over many images, because `Results` objects retain the original image array and accumulate in RAM otherwise.

**What goes wrong if set wrong.** Nothing, in either direction, in this pipeline — `src/train.py:129` immediately does `result_list = list(result)`, which consumes the generator and holds all `max_images` results in memory anyway, because the confidence-histogram step needs a second pass over them. At 40 images that is fine. If `max_images` is ever raised into the thousands, this is the line that will fail, not the `stream` setting.

### `verbose`

**What it does.** Ultralytics prints one line per image from `engine/predictor.py` (around `:299` and `:370`) describing the source, the shape and what was found.

**Valid values.** `True` | `False`.

**Why the default.** With `max_images=40` this is 40 lines, the single biggest contiguous block of console noise in a run — and every one of those detections is already in ClearML as an image and as a histogram. Against a baseline of 334 lines of the project's own output for a 3-epoch run, it is a 12% increase for information you can see in the gallery.

**What goes wrong if set wrong.** Nothing breaks. It returns automatically at DEBUG: `_apply_console_verbosity` (`src/train.py:35-44`) sets `args_predict["model"]["verbose"] = True` — together with `args_val["verbose"]` — when `logger.isEnabledFor(logging.DEBUG)`. Set `log_level` in [`0_console.md`](0_console.md) rather than flipping this key by hand, so both come back together.

> **Do not use `YOLO_VERBOSE=0` to quieten ultralytics.**
>
> The env var is read once at import time in `ultralytics/utils/__init__.py` and drops *every* ultralytics logger to `ERROR`. That removes the per-epoch metrics table, the AMP check result, and the dataset scan warnings about missing or corrupt labels — the three pieces of ultralytics output actually worth reading. The per-call `verbose` arguments in this group and in [`5_testing.md`](5_testing.md) are the targeted knobs: they silence the two per-item loops and leave everything else intact. This is why `args_predict["model"]["verbose"]` exists as a parameter at all instead of being an environment variable.

## Scenarios

### Scenario 1 — a multi-class gallery you can actually read

The defaults were tuned for a dense single-class dataset. On an eight-class dataset they hide exactly the information you need:

```python
args_predict = {
    "max_images": 40,
    "plot": {
        "conf": True,
        "labels": True,          # was False: without this you get a bare "0.87"
        "boxes": True,
        "masks": True,
        "probs": True,
        "color_mode": "class",   # was "instance": one colour per class
        "kpt_radius": 5,
        "kpt_line": True,
        "txt_color": (255, 255, 255),
    },
    "model": {"batch": 16, "conf": 0.25, "iou": 0.7, "max_det": 1000, "stream": True, "verbose": False},
}
```

Resulting behaviour: each box is drawn in its class's palette colour and labelled `speed_limit 0.91`, so a misclassification is visible at a glance instead of being indistinguishable from a correct detection. Nothing about the model, the metrics or the uploaded reports changes — only the ten `Prediction/image-*` grids. On a crowded image the extra label text costs readability, which is the trade `color_mode: "instance"` and `labels: False` were making in the first place.

### Scenario 2 — "the gallery looks perfect but recall is bad"

The `Test Error Analysis` table shows high `FN` counts, but every grid in `Prediction` looks clean. Lower the display threshold for one diagnostic run:

```python
args_predict = {"model": {"conf": 0.05, ...}}   # 6_Predict only
args_val     = {"conf": 0.001, ...}             # 5_Testing: leave alone
```

Resulting behaviour: the grids now show the low-confidence detections the model *is* producing. If the missed objects appear at 0.05–0.2, the model is finding them and the problem is threshold selection — check `Operating Point`, which will be reporting an F1-optimal confidence well below 0.25. If they do not appear at all, the model genuinely is not detecting them and no threshold will help. Critically, `args_val["conf"]` was not touched, so mAP, the curves and the calibration diagram in this run remain directly comparable to every previous run. Doing the same experiment by lowering `args_val["conf"]` would have told you nothing new and would have changed the metrics; doing it by *raising* `args_val["conf"]` would have understated mAP by roughly 12% relative — see [`5_testing.md`](5_testing.md).

### Scenario 3 — chasing a prediction-stage failure

The `Prediction` section is empty and the console says nothing useful. Turn the whole run's verbosity up rather than flipping keys individually:

```
LOG_LEVEL=debug            # env var for a local run
```

or, for a remote run, set `log_level: debug` in the `0_Console` group — the UI value only ever takes effect on the agent, because `task.execute_remotely()` kills the local process immediately after `set_log_level()` runs.

Resulting behaviour: `_apply_console_verbosity` flips both `args_val["verbose"]` and `args_predict["model"]["verbose"]` to `True`, so you get ultralytics' per-class validation table and one line per predicted image; `src/train.py:82` logs the full resolved `args_predict` dict; `:112` logs the first five sampled image paths, which is how you confirm the glob found the right directory; and `src/train.py:110` reports how many images were selected and from which split. The two most common causes both become obvious here: an empty split directory (the `No validation images found` warning fires) and an image extension outside the six the glob matches.

## Related groups

- [`0_console.md`](0_console.md) — `log_level` and `progress`; DEBUG is what re-enables `model.verbose`.
- [`1_task.md`](1_task.md) — model selection; determines whether the pose- and classify-only plot keys mean anything.
- [`2_data.md`](2_data.md) — the test/val split whose directory is sampled for the gallery.
- [`3_augment.md`](3_augment.md) — augmentation; not applied at prediction time.
- [`4_training.md`](4_training.md) — `imgsz`, which reaches this stage via `args_val["imgsz"]`.
- [`5_testing.md`](5_testing.md) — the metric thresholds, and why `conf` there is not `conf` here.
- [`7_export.md`](7_export.md) — export formats; runs immediately before this stage.
- [`8_visualization.md`](8_visualization.md) — `log_confidence_histograms`, which gates the `Distributions` output of this stage.
