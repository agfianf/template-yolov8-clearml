# `5_Testing` — `args_val`

The `args_val` dict in `src/params.py` is published to the ClearML UI by `curr_task.connect(args_val, name="5_Testing")` in `src/utils/clearml_settings.py:77`, and appears in the experiment's **Configuration → Hyperparameters → 5_Testing** section. It configures the single standalone `model_yolo.val()` call that `src/train.py` runs after training finishes. That call is the one that produces the headline numbers you read off a finished task: mAP, the PR/F1/P/R curve families, the per-class table, the confusion matrix, the operating point, the calibration diagram and the worst-image error gallery. Every one of those is downstream of the values in this group, and one of them — `conf` — can move mAP by double-digit percentages without anything in the console looking wrong.

## Quick reference

| Key | Default | Type | Meaning |
|---|---|---|---|
| `batch` | `16` | int | Images per validation batch. Overridden to `32` by `src/train.py` for the final pass. |
| `save_json` | `False` | bool | Write COCO-style JSON. On a segmentation model it also switches mask mAP to full resolution. |
| `save_hybrid` | `False` | bool | Removed upstream in ultralytics 8.4. Inert; emits a deprecation warning. |
| `conf` | `0.001` | float | NMS confidence floor. **Do not change.** Everything below it is discarded before any metric sees it. |
| `iou` | `0.7` | float | IoU threshold for NMS. Not the confusion matrix's IoU. |
| `max_det` | `300` | int | Maximum detections kept per image after NMS. Caps recall if lowered. |
| `half` | `True` | bool | FP16 inference. Forwarded to the newer `quantize=16` with a deprecation warning. |
| `device` | `0` | int/str | CUDA device index, or `cpu`. |
| `dnn` | `False` | bool | OpenCV DNN backend for ONNX. Inert when validating a `.pt` checkpoint, which is always the case here. |
| `plots` | `True` | bool | Gates the confusion matrix, the `val*.jpg` debug samples and the error gallery. Not the interactive curves. |
| `rect` | `False` | bool | Rectangular batching. `Model.val()` defaults this to `True`; we explicitly turn it off. |
| `save_crop` | `True` | bool | A predict-mode argument. Nothing in the validation path reads it — effectively inert. |
| `split` | `"val"` | str | Which dataset split to validate. Overridden to `"test"` when `data.yaml` declares one. |
| `verbose` | `False` | bool | Ultralytics' own one-line-per-class console table. Re-enabled at `LOG_LEVEL=DEBUG`. |

## Where this group actually applies

This is the first thing to understand, and it is not obvious from the group's name: **`args_val` does not configure the per-epoch validation that runs during training.** `src/train.py` passes `args_train` to `model_yolo.train()` and `args_val` to `model_yolo.val()`, and those are two disjoint dicts. During training, ultralytics builds its validator from a copy of the *trainer's* args (`models/yolo/detect/train.py:205-209`), so the per-epoch numbers you watch in the `Metrics/*` scalar series are computed with ultralytics' own defaults — which happen to be the same `conf`, `iou` and `max_det` values this group holds, but only by coincidence, not because this group was read.

The practical consequence: editing `conf` or `max_det` in the ClearML UI changes the final `Test `/`Final ` numbers and every heavy report, and leaves the per-epoch curve untouched. If those two disagree in a task, this is why.

Three values in this group are also written by `src/train.py` after `Task.connect()` has already run, so what you set in the UI is either ignored or supplemented:

- `imgsz` is injected at `src/train.py:307` from `args_train["imgsz"]`. It is not a key in `args_val` at all, so it never appears in the `5_Testing` group — change the image size in [`4_training.md`](4_training.md).
- `batch` is forced to `32` at `src/train.py:320`, overwriting whatever the group says. The `16` in `src/params.py` is effectively dead.
- `visualize` is injected at `src/train.py:325` from `args_visualization["log_worst_images"]` — see [`8_visualization.md`](8_visualization.md). It makes ultralytics write one GT/FP/TP/FN panel per validated image into `<save_dir>/visualizations` (`models/yolo/detect/val.py:196`), uncapped, which is why it is enabled for this single final pass and never during training. Only the worst N panels are then uploaded.
- `split` is forced to `"test"` at `src/train.py:317` whenever the generated `data.yaml` has a test split.

## The three thresholds

A finished ClearML task shows at least three different confidence thresholds, and they contradict each other if you assume they are the same number. They are not. This table is the one to keep open while reading a task.

| Threshold | Value | Set by | Governs | Why that value |
|---|---|---|---|---|
| NMS confidence floor | `0.001` | `args_val["conf"]` (this group) | mAP50, mAP50-95, every PR/F1/P/R curve, the calibration diagram, the confidence split, the operating-point search | A floor near zero is required to keep the high-recall tail of the curves; anything filtered here never reaches the metrics at all |
| Confusion-matrix display confidence | `0.25` | `args_visualization["confusion_matrix_conf"]`, pinned by `on_val_batch_start` in `src/yolov8/callbacks.py` | `Confusion Matrix / Counts` and `/ Normalized`, and the GT/FP/TP/FN error gallery | A matrix built at 0.001 fills with low-confidence detections and a background column that swamps everything else |
| Confusion-matrix IoU | `0.45` | Hard-coded upstream as the `iou_thres` default of `ConfusionMatrix.process_batch` (`utils/metrics.py:402-407`) | The same matrix and gallery | Upstream default, and the pinning wrapper does not touch it. It is **not** `args_val["iou"]`, which is the NMS threshold |
| Per-class max-F1 confidence | Derived per class | Ultralytics | The `Metrics/Precision`, `Metrics/Recall` and F1 scalars ultralytics prints and reports | Each class is summarised at its own best-F1 operating point, so no single number describes them |
| Deployment threshold | Derived | Reported as `Operating Point` by `extract_optimal_confidence` | Nothing at validation time — it is an output, not an input | This is the number to actually ship with |
| Prediction-gallery display confidence | `0.25` | `args_predict["model"]["conf"]` — see [`6_predict.md`](6_predict.md) | The uploaded `Prediction` image grid and the predict-stage `Distributions` histograms | Purely cosmetic; a gallery at 0.001 is unreadable |

Ultralytics itself only has one `conf` and feeds it to both NMS and the confusion matrix (`models/yolo/detect/val.py:118` and `:195`). Splitting the two is the entire job of `on_val_batch_start`: it wraps `confusion_matrix.process_batch` so that whatever `conf` ultralytics passes in, the matrix uses the display threshold instead. `tests/yolov8/test_callbacks.py::TestConfusionMatrixConfPinning` asserts this — including that the pin survives whatever the caller passes, that it reads the value from `args_visualization` rather than a constant, and that it is idempotent across batches.

## `conf` — the one knob that must not move

**What it does.** It is passed straight into `non_max_suppression` (`models/yolo/detect/val.py:118`). NMS discards every detection below it before `update_metrics` runs, so the metrics, the curves, the calibration capture and the operating-point search are all computed over a truncated set.

**Valid values.** `0.0` to `1.0`. The only correct value here is `0.001`, which is the ultralytics val-mode default (`cfg/default.yaml`: `conf:` is unset and resolves to `predict=0.25, val=0.001`).

**Why the default.** Average precision is an integral over the whole recall range. Raising the floor removes the low-confidence detections that make up the high-recall end of the curve, so the curve simply stops early and the area under it shrinks. The number that comes out is still a valid PR curve — it is just a PR curve for a different, artificially truncated detector, reported under the same name.

**What goes wrong if set wrong.** This group used to hold `0.25`. The measured damage: the high-recall tail of every PR curve was cut off, mAP was understated by roughly 12% relative on a realistic confidence distribution, and the operating-point and calibration reports were blinded below 0.25 — they could not recommend or measure a threshold they had never observed. Nothing in the console indicates any of this; the run completes normally and reports a lower number. If you want a gallery or a matrix at a higher threshold, that is what `args_visualization["confusion_matrix_conf"]` and `args_predict["model"]["conf"]` are for.

## `max_det`

**What it does.** The cap on how many detections NMS returns per image (`models/yolo/detect/val.py:123`), applied after sorting by confidence. Whatever exceeds it is dropped, exactly as if `conf` had been raised for that image.

**Valid values.** Any positive integer. `300` is the ultralytics default.

**Why the default.** With `conf` at `0.001` almost every image produces a long low-confidence tail, and `max_det` is what actually bounds it. A cap of 100 truncates that tail on crowded images specifically, which lowers recall and therefore mAP — independently of `conf`, and only on the images that needed it most.

**What goes wrong if set wrong.** Too low and mAP falls in a way that looks like a model regression rather than a measurement artefact, and only on dense scenes. Before lowering it, read "Instances per image" in the Dataset report produced by the data stage ([`2_data.md`](2_data.md)). Raising it above 300 is harmless but slows NMS and postprocessing for no metric gain unless your images genuinely carry more than 300 objects.

## `iou`

**What it does.** The IoU threshold used by NMS to decide that two boxes are the same object (`models/yolo/detect/val.py:119`). Lower means more aggressive suppression.

**Valid values.** `0.0`–`1.0`. `0.7` is the ultralytics default.

**Why the default.** It is the value the mAP numbers everyone quotes are computed at. Changing it makes your mAP incomparable to any published figure and to your own earlier runs.

**What goes wrong if set wrong.** Lowering it suppresses legitimately overlapping objects and costs recall; raising it lets duplicate detections through, which cost precision. It is also frequently confused with two other IoUs: the confusion matrix's matching threshold, which is hard-coded at `0.45` upstream and does not read this value, and the mAP50/mAP50-95 IoU sweep, which is fixed by the metric definition and is not configurable at all.

## `split`

**What it does.** Selects which split from `data.yaml` the final `val()` runs against: `"val"`, `"test"` or `"train"`.

**Valid values.** `"val"` | `"test"` | `"train"`.

**Why the default.** `"val"` is the safe fallback for a dataset with no dedicated test split. But `src/train.py:316-317` overrides it to `"test"` whenever the generated `data.yaml` declares a test split — which happens when `2_Data` was given `task_ids_test` / `project_ids_test` / `s3_uri_dir_test`, or a non-null `test_ratio`. See [`2_data.md`](2_data.md).

**How to tell which one ran.** The report titles. `src/train.py:331` sets a `split_label` of `"Test "` when the split is `test` and `"Final "` otherwise, and that prefix is stamped onto every heavy report title (`Test Per-Class Metrics`, `Final Operating Point`, …) and onto the single-value scalars (`test-metrics/mAP50(B)` vs `final-metrics/mAP50(B)`). The prefix exists specifically so the test-split reports do not overwrite the validation-split ones.

**What goes wrong if set wrong.** Setting `"train"` reports training-set performance, which is not a generalisation estimate and will look implausibly good. Setting `"test"` on a dataset that has no test split makes `val()` fail; the failure is caught and logged by the `try` block in `src/train.py`, so the run continues to export and predict with no final metrics at all.

## `save_json`

**What it does.** Two things, and the second one is the trap. It writes prediction JSON for external evaluation, and — on a segmentation model only — it changes how masks are computed. `models/yolo/segment/val.py:74` selects `ops.process_mask_native` when `save_json or save_txt` is set and `ops.process_mask` otherwise, and `:128` downsamples the ground-truth masks to match (`s // 4` in the non-native case).

**Valid values.** `True` | `False`.

**Why the default.** `False` keeps the fast path. `process_mask` works at prototype resolution — 160×160 at `imgsz=640` — which is cheaper and is what ultralytics does by default.

**What goes wrong if set wrong.** Both settings are internally self-consistent, so neither is "wrong" in isolation. What is wrong is comparing across them: **mask mAP from `save_json=False` and mask mAP from `save_json=True` are not comparable numbers.** Do not read a jump in mask mAP as a model improvement if this flag changed between the two runs. The same dependence propagates into the mask-vs-box gap reported per epoch and into per-class mask mAP in the per-class table. Note also that ultralytics may turn this on by itself at `models/yolo/detect/val.py:91`, but only for COCO or LVIS datasets — never for the CVAT/S3 data this template handles.

## `plots`

**What it does.** More than the name suggests. It gates the confusion matrix being populated at all (`models/yolo/detect/val.py:194-195`), the `visualizations/` panel directory (`:100`, `save_matches=self.args.plots and self.args.visualize`), the `val*.jpg` labelled/predicted batch samples that `on_val_end` uploads (`engine/validator.py:267`), and the static PNGs ultralytics renders into `save_dir`.

**Valid values.** `True` | `False`.

**Why the default.** Turning it off silently empties four ClearML report sections.

**What goes wrong if set wrong.** With `plots=False` you lose `Confusion Matrix / Counts` and `/ Normalized` (the matrix stays all zeros, so `extract_confusion_matrix_data` reports an empty plot), the `Validation` debug-sample images, and the worst-image error gallery — `_log_error_gallery` finds no `visualizations/` directory and skips silently, which by design is not an error. What you do **not** lose is the interactive curve family: `prec_values` and the per-class curves are computed inside `ap_per_class` regardless of `plot`, which only controls PNG rendering. During training this flag is additionally forced off by ultralytics for every epoch except the last (`engine/validator.py:167`), which is why the confusion matrix only ever appears once.

## `batch`

**What it does.** Validation batch size.

**Valid values.** A positive integer, or `-1` for AutoBatch.

**Why the default — and why it does not matter.** `src/train.py:320` sets `args_val["batch"] = 32` immediately before the final `val()`, unconditionally. The `16` in `src/params.py` and anything you type into the UI are both overwritten. Treat this key as read-only until that line changes.

**What goes wrong if set wrong.** Nothing, currently, because it has no effect. If the override is ever removed, the only failure mode is CUDA OOM on the final pass — validation batch size does not affect metrics.

## `half`

**What it does.** FP16 inference. Ultralytics 8.4 replaced `half`/`int8` with a unified `quantize` argument; `half=True` is forwarded to `quantize=16` by the deprecation shim in `cfg/__init__.py` (around `:563-571`) and emits a deprecation warning on every run.

**Valid values.** `True` | `False`.

**Why the default.** FP16 roughly halves validation time on any modern CUDA GPU with a negligible effect on mAP.

**What goes wrong if set wrong.** On `device: cpu` the setting is ignored — the validator resolves the actual precision from the backend and records it (`engine/validator.py:188-191`). The real risk is the deprecation itself: when ultralytics finally removes `half`, it will join `save_hybrid` in `removed_keys` and stop having any effect, silently reverting the final validation to FP32. The same deprecation lives in `args_export["params"]` and is covered by `tests/yolov8/test_export_smoke.py`.

## `device`

**What it does.** Which device the final validation runs on. Passed to `select_device` (`engine/validator.py:183`).

**Valid values.** An int GPU index (`0`), a list (`[0,1]`), `"cpu"`, `"mps"`, or `-1` to auto-select an idle GPU.

**Why the default.** `0` matches the single-GPU agent containers this template targets (`--gpus all` in `DOCKER_ARGUMENTS`).

**What goes wrong if set wrong.** Naming a device index that does not exist on the agent host fails the final `val()`; the exception is caught and logged, and the run proceeds to export with no final metrics. Choosing `cpu` additionally forces `rect = False` upstream (`engine/validator.py:222-225`) regardless of what you set below, and turns a two-minute validation into a very long one. Note that the *prediction* stage ignores this key entirely — `src/train.py:124` hard-codes `device="0" if torch.cuda.is_available() else "cpu"`.

## `dnn`

**What it does.** Selects OpenCV's DNN module as the ONNX inference backend inside `AutoBackend` (`engine/validator.py:186`).

**Valid values.** `True` | `False`.

**Why the default.** It only has meaning when the thing being validated is an ONNX file. `src/train.py` always validates the in-memory PyTorch model, so this key is inert in this pipeline.

**What goes wrong if set wrong.** Nothing today. Documented here so nobody spends time toggling it looking for a speedup that cannot happen.

## `rect`

**What it does.** Rectangular batching — images are grouped by aspect ratio and letterboxed to the minimum common size instead of to a square, so less of each batch is padding.

**Valid values.** `True` | `False`.

**Why the default.** This is a deliberate override, not an inherited default. `Model.val()` sets `rect: True` as its own method default (`engine/model.py:622`) and merges caller kwargs on top (`:623`), so `args_val["rect"] = False` actively switches it off. Square letterboxing is what the prediction stage and every exported model see, so keeping validation square means the reported metrics describe the preprocessing that actually ships.

**What goes wrong if set wrong.** Setting `True` makes validation faster and slightly changes the numbers, because a different amount of padding means different effective object scales. The change is small but real, and it makes the metrics marginally optimistic relative to square-input deployment. It is also silently forced back to `False` on CPU and MPS devices (`engine/validator.py:222-225`).

## `save_crop`

**What it does.** In predict mode it writes each detection's cropped region to disk. Nothing in the validation code path reads it — it does not appear anywhere under `models/yolo/` for validation, nor in `engine/validator.py`.

**Valid values.** `True` | `False`.

**Why the default.** `True` here is historical. It is inert.

**What goes wrong if set wrong.** Nothing. Listed for completeness so its presence in the UI is not mistaken for a feature that is switched on.

## `save_hybrid`

**What it does.** Nothing. It was removed in ultralytics 8.4 — `cfg/__init__.py:561` lists it in `removed_keys`, and the deprecation handler pops it from the config after warning.

**Valid values.** Irrelevant.

**What goes wrong if set wrong.** Nothing beyond one extra deprecation warning per run. It is kept in `src/params.py` only so removing it is a deliberate act rather than an accident.

## `verbose`

**What it does.** Ultralytics prints one console line per class at the end of validation (`models/yolo/detect/val.py:290`, gated on `self.args.verbose and not self.training and self.nc > 1`).

**Valid values.** `True` | `False`.

**Why the default.** The same numbers are already in ClearML as the per-class table (`Per-Class Metrics / Detailed`) and the per-class bar chart, in a form you can sort. On a 20-class dataset the console version is 20 lines of duplicated information, which works directly against the project rule that log volume must not grow with dataset size.

**What goes wrong if set wrong.** Nothing breaks; the console just gets longer. It comes back automatically when the effective log level is DEBUG — `_apply_console_verbosity` in `src/train.py:35-44` sets `args_val["verbose"] = True` if `logger.isEnabledFor(logging.DEBUG)`. Set `log_level` in [`0_console.md`](0_console.md) rather than editing this key, so the predict-side verbosity comes back with it. Never reach for `YOLO_VERBOSE=0` to go the other way; see [`6_predict.md`](6_predict.md) for why.

## Scenarios

### Scenario 1 — "mAP dropped 12% and I did not change the model"

Somebody set a confidence floor in the `5_Testing` group because the confusion matrix was unreadable:

```python
args_val = {
    "conf": 0.25,   # WRONG: this is the NMS floor, not a display setting
    "iou": 0.7,
    "max_det": 300,
    ...
}
```

Resulting behaviour: every detection scoring below 0.25 is dropped inside `non_max_suppression` before `update_metrics` runs. `Curves/Precision-Recall (B)` stops at whatever recall the 0.25-and-above detections reach and never approaches 1.0; mAP50-95 falls by roughly 12% relative; `Operating Point` cannot report an F1-optimal confidence below 0.25 because no such detection was ever observed; the `Calibration` reliability diagram has empty bins for the bottom quarter of the range and its ECE is meaningless. The per-epoch `Metrics/mAP` series is unaffected, because per-epoch validation does not read this group — so the task shows an epoch-20 mAP that is visibly higher than the final `Test mAP50-95(B)` single value, with no explanation anywhere in the console.

The correct fix, which produces the readable matrix without touching any metric:

```python
args_val = {"conf": 0.001, ...}                      # 5_Testing:      metrics
args_visualization = {"confusion_matrix_conf": 0.4}  # 8_Visualization: display only
```

### Scenario 2 — a dedicated test split, on a segmentation model, with the full-resolution mask metric

`2_Data` names a CVAT task as test-only, so `data.yaml` gets a `test:` entry and `src/train.py:317` flips the split for you. You additionally want mask mAP at native resolution because a downstream consumer evaluates that way:

```python
args_data = {"cvat": {"task_ids_train": [741, 733], "task_ids_test": [730]}}   # 2_Data
args_val = {
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "save_json": True,     # process_mask_native instead of process_mask
    "split": "val",        # overwritten to "test" by src/train.py
    "plots": True,
    "verbose": False,
}
```

Resulting behaviour: the final pass runs against the test split, and every heavy report is titled with a `Test ` prefix — `Test Per-Class Metrics`, `Test Curves/*`, `Test Operating Point`, `Test Error Analysis`, `Test Calibration` — leaving the untitled versions from `on_train_end` (which describe the validation split) intact side by side. Single values arrive as `test-metrics/mAP50(M)` and friends. Because `save_json=True`, `segment/val.py:74` picks `process_mask_native` and `:128` stops downsampling the ground-truth masks, so `Test mAP50-95(M)` and the `Mask-mAP50-95` column of the per-class table are computed at full resolution. Those mask numbers are **not** comparable to the per-epoch `Metrics/mAP` series from the same run, which used `process_mask` at 160×160 — the box numbers are comparable, the mask numbers are not.

### Scenario 3 — crowded imagery

An inspection dataset with 400+ instances on the worst images:

```python
args_val = {"conf": 0.001, "max_det": 300, ...}
```

Resulting behaviour: on those images NMS returns the 300 highest-confidence detections and discards the rest, so the ground truth beyond 300 can never be matched and recall is capped below 1.0 for those images alone. This shows up as a PR curve that plateaus short of full recall and as those images dominating the `Error Analysis` worst-image table with a high `FN` count and a normal `FP` count. Raise `max_det` to 600 and re-run; if the plateau moves, `max_det` was the binding constraint rather than the model. Check "Instances per image" in the data-stage report first — see [`2_data.md`](2_data.md).

## Related groups

- [`0_console.md`](0_console.md) — `log_level` and `progress`; DEBUG re-enables `verbose` here and in `6_Predict`.
- [`1_task.md`](1_task.md) — model selection and ClearML project/task naming.
- [`2_data.md`](2_data.md) — where the test split comes from, and the instances-per-image figure `max_det` should be checked against.
- [`3_augment.md`](3_augment.md) — augmentation; not applied at validation time.
- [`4_training.md`](4_training.md) — `imgsz` (injected into this group), and the per-epoch validation that this group does *not* configure.
- [`6_predict.md`](6_predict.md) — the prediction gallery and its separate, display-only `conf`.
- [`7_export.md`](7_export.md) — export formats; shares the deprecated `half` argument.
- [`8_visualization.md`](8_visualization.md) — `confusion_matrix_conf`, `log_worst_images` (which injects `visualize` here), and every report switch.
