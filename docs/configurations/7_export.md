# 7_Export

`args_export` in [`src/params.py`](../../src/params.py) is connected to the ClearML UI as the hyperparameter group **`7_Export`** (see `config_clearml()` in `src/utils/clearml_settings.py`). It controls one stage only: the `export_handler()` call in `src/train.py`, which runs *after* training and after the final standalone `val()`, and *before* the prediction stage. Nothing in this group affects training, validation or metrics — it decides which deployment artifacts are produced from the trained weights and, because every successful export is registered as a ClearML `OutputModel`, what ends up in the task's **Models** tab. The group has two sub-dicts: `format`, a set of on/off switches, and `params`, the arguments handed to `ultralytics`' exporter.

## Quick reference

### `args_export["format"]` — which artifacts to produce

| Key | Default | Type | Meaning |
|---|---|---|---|
| `torchscript` | `0` | int (0/1) | TorchScript `.torchscript` — a self-contained traced PyTorch graph |
| `onnx` | `0` | int (0/1) | ONNX `.onnx` — the portable interchange format most runtimes read |
| `openvino` | `0` | int (0/1) | OpenVINO IR — a `_openvino_model/` **directory**, for Intel CPU/iGPU |
| `engine` | `0` | int (0/1) | TensorRT `.engine` — NVIDIA GPU only, and machine-specific |
| `coreml` | `0` | int (0/1) | CoreML `.mlpackage` — Apple platforms |
| `saved_model` | `0` | int (0/1) | TensorFlow SavedModel directory |
| `pb` | `0` | int (0/1) | TensorFlow GraphDef `.pb` (produced via SavedModel) |
| `tflite` | `0` | int (0/1) | **Stale name.** ultralytics 8.4.110 renamed this format to `litert` |
| `edgetpu` | `0` | int (0/1) | Edge TPU `_edgetpu.tflite` — needs the Coral compiler, a system package |
| `tfjs` | `0` | int (0/1) | **Dead switch.** Not a valid format in ultralytics 8.4.110 at all |
| `paddle` | `0` | int (0/1) | PaddlePaddle `_paddle_model/` directory |

### `args_export["params"]` — how to export

| Key | Default | Type | Meaning |
|---|---|---|---|
| `keras` | `False` | bool | TensorFlow SavedModel only: emit Keras layers |
| `optimize` | `False` | bool | Historically TorchScript mobile optimisation; in 8.4.110 it is documented as DEEPX-only and is a no-op for every format this group can enable |
| `half` | `True` | bool | **Deprecated upstream.** Forwarded to `quantize=16`, i.e. FP16 weights |
| `int8` | `False` | bool | **Deprecated upstream.** Forwarded to `quantize=8`; beats `half` when both are on |
| `dynamic` | `False` | bool | Variable input shape; applies to torchscript, onnx, openvino, engine, coreml |
| `simplify` | `False` | bool | Simplify the intermediate ONNX graph; pulls in `onnxruntime` + `onnxslim` |
| `opset` | `None` | int / None | ONNX opset version; `None` lets ultralytics pick the best one for the runtime |
| `workspace` | `4` | int (GiB) | TensorRT builder workspace; ignored by every other format |
| `nms` | `False` | bool | Fuse NMS into the exported graph so the artifact returns final detections |
| `fraction` | `1.0` | float | Fraction of the calibration dataset used for INT8 quantisation |

## A default run exports nothing

This is the single most surprising thing about this group, so it is worth stating plainly: **every switch in `format` is `0`, so a run with default parameters produces no artifacts at all.** `export_handler()` builds its worklist as `[fmt for fmt, is_use in args_export["format"].items() if is_use]`, which on defaults is empty, so the loop body never runs, no `OutputModel` is created from an export, and the console prints exactly one line:

```
export: 0/0 formats ok
```

`0/0` is not an error and not a silent failure — it is "you asked for zero formats and zero succeeded". If you expected an ONNX file to appear in the Models tab and it did not, check `7_Export/format/onnx` before checking anything else. The two PyTorch models (`best.pt` and `last.pt`) *are* still registered, but they come from `on_train_end` in `src/yolov8/callbacks.py`, not from this group, which is why the Models tab is never completely empty even on a default run.

The counterpart line for a run that did request formats is `export: N/M formats ok`, where `M` is how many switches were on and `N` how many exports both produced an artifact and registered it. A format that raised is counted in `M` and not in `N`, and its exception is logged with a traceback — with one exception: TensorRT on a host with no CUDA device is logged as a `WARNING`, not an error, because an absent GPU is a missing capability rather than a defect.

Enable a format by setting its value to `1` in the ClearML UI (or `True`; ClearML casts the UI string back to the type of the default, and both are truthy). Any non-zero value works, but `1` is the convention the file uses.

## Which formats realistically work in this container

The training image is `python:3.14-slim` with `torch 2.9.1+cu128`, `ultralytics 8.4.110`, `onnx 1.22.0` and `openvino 2026.2.1`. Not installed: `onnxruntime`, `onnxslim`, `tensorrt`, `coremltools`, `tensorflow`, `paddlepaddle`, `ncnn`. That list is what decides the answer below, together with one behaviour of ultralytics that is easy to miss: **when an export dependency is missing, ultralytics does not fail — it runs `pip install` for it at export time.** So a format that "needs extra dependencies" does not necessarily fail; it may instead spend several minutes installing a large package into the agent's container on every single run, or fail at the install step with a resolver error that has nothing obviously to do with exporting.

**Verified working.** `torchscript`, `onnx` and `openvino`. `onnx` and `openvino` are the two formats covered by `tests/yolov8/test_export_smoke.py`, which runs the real exporters with the exact `args_export["params"]` dict this project ships and asserts an artifact appears. `torchscript` needs nothing beyond `torch`. These three are the safe choices.

**Works only with a GPU, and only after a live install.** `engine` (TensorRT). The exporter special-cases it: it forces `device="0"`, and `ultralytics` asserts the model is not on the CPU. `tensorrt` is not in the image, so `check_tensorrt()` will attempt `pip install tensorrt-cu12>=7.0.0` inside the container at export time — that needs outbound network access from the agent, and whether a Python 3.14 wheel resolves has not been verified here. Two further properties matter more than the install: a TensorRT engine is built for the *specific* GPU architecture and TensorRT version present at build time and will not load elsewhere, and the build itself commonly takes minutes. If it fails for lack of a GPU you get `export engine: skipped, no CUDA device available` and a `skipped (no CUDA)` row in the export table rather than a stack trace.

**Unverified, and unlikely to work as-is.** `coreml`, `saved_model`, `pb`, `edgetpu`, `paddle`. None of their dependencies are in the image, so each triggers an auto-install of a heavyweight package (`coremltools`, `tensorflow`, `paddlepaddle`/`x2paddle`). ultralytics' own export environment matrix in `engine/exporter.py` pins CoreML export to Python 3.13 and the TensorFlow chain to Python 3.12; this image is Python 3.14, which is a strong hint that those exports are not expected to resolve here. `edgetpu` additionally needs the Coral `edgetpu_compiler` binary, which is an apt package and is not in the Dockerfile — no `pip install` can supply it. I have not run any of these, so treat "unlikely" as exactly that: if you need one, try it once in a throwaway task and read the export table, rather than assuming either outcome.

**Broken by a name change.** `tflite` is no longer a valid ultralytics format name — it was renamed `litert` in 8.4.110. Enabling it does not fail outright: ultralytics fuzzy-matches the string, logs `Invalid export format='tflite', updating to format='litert'`, and proceeds with a LiteRT export (whose dependencies are also absent). The artifact suffix is still `.tflite`.

**Genuinely dead.** `tfjs` has no counterpart in 8.4.110 and is not close enough to any surviving name to be fuzzy-matched, so it raises `ValueError: Invalid export format='tfjs'`. The exporter catches it, logs `export tfjs: failed`, records a `failed: ValueError` row, and carries on with the other formats. It costs you a red row and nothing else, but it will never produce an artifact.

## The `params` sub-dict

### `half` and `int8` — deprecated, tolerated, and guarded by a test

ultralytics has replaced both flags with a single `quantize` argument. They still work: `_handle_deprecation()` in `ultralytics/cfg/__init__.py` pops both and rewrites them as `quantize = 8 if int8 else 16 if half else None`, emitting a deprecation warning as it does so. With this project's defaults (`half: True`, `int8: False`) that resolves to `quantize=16`, so **every export here is FP16 by default**, and a deprecation warning is printed on every export call. Note the precedence: `int8` wins over `half`, so setting both to `True` gives you INT8, not FP16.

This is deliberate rather than an oversight, and it is pinned by a test. `tests/yolov8/test_export_smoke.py::test_export_with_project_params` runs a real ONNX and OpenVINO export with the shipped params dict; its docstring says so explicitly — "when a future release stops accepting them it fails here rather than mid-training". If that test starts failing after an ultralytics bump, the fix is to rename these two keys to `quantize` in `src/params.py`, not to pin ultralytics.

What goes wrong if you leave `half: True` without thinking about it: an FP16 ONNX graph is the wrong artifact for a CPU deployment. Most CPU ONNX Runtime builds either refuse FP16 or fall back to an emulated path that is slower than the FP32 model would have been. If the target is a CPU, set `half: False` and accept the larger file. For OpenVINO an FP16 IR is the normal and correct choice, and for TensorRT FP16 is usually a straight win on any modern NVIDIA GPU.

Turning on `int8` has a trap of its own that this group cannot fix: INT8 export needs a **calibration dataset**, and `args_export["params"]` has no `data` key, so nothing passes one. ultralytics then logs `INT8 export requires a missing 'data' arg for calibration. Using default 'data=...'` and calibrates on its own tiny sample dataset — not on your data. The resulting quantised model is calibrated against the wrong activation distribution and can lose a great deal of accuracy for no visible reason. Do not enable `int8` here without first adding a `data` entry to the params dict and confirming it reaches the exporter.

### `dynamic`

Exports a graph that accepts variable input height and width instead of being frozen at `args_train["imgsz"]`. Applies to `torchscript`, `onnx`, `openvino`, `engine` and `coreml`. The default `False` is right for the common case: a fixed shape lets every runtime pre-plan its memory and kernel selection, which is typically measurably faster, and this pipeline exports at exactly one size anyway (`export_handler()` passes `imgsz=args_training["imgsz"]`). Turn it on only when the consumer genuinely feeds varying sizes — a batched server that letterboxes to different aspect ratios, say. On TensorRT, `dynamic` changes the engine build into a profile-based build, which is slower to build and usually slower to run.

### `simplify`

Runs a graph simplification pass over the intermediate ONNX before it is written or handed to TensorRT. Upstream's default is `True`; this project's default is `False`, and that is not an accident: enabling it makes ultralytics require `onnxslim` *and* `onnxruntime` (or `onnxruntime-gpu`), neither of which is in the image, so a run with `simplify: True` triggers a pip install of both inside the container before the export can start. The simplified graph is usually a little smaller and occasionally avoids an unsupported-op error in a downstream runtime, which is a real benefit — but pay for it deliberately, and ideally by adding the packages to `pyproject.toml` and rebuilding the image rather than by installing them at run time on every task.

### `opset`

The ONNX opset version. `None` means "let ultralytics choose", which it does via `best_onnx_opset()` based on the installed `onnx` version and whether the export is CUDA-bound — that is nearly always the right answer, and it is why the default is `None`. Pin it only when a specific downstream runtime refuses the auto-chosen version.

There is a caveat specific to setting this **from the ClearML UI**. ClearML casts a UI value back to the type of the code default, and the code default here is `None`; its casting logic for a `None` default is `str(param) if param else None`, which means a UI value of `17` comes back as the *string* `"17"`, not the integer `17`. That string is truthy, so it is used instead of the auto-selected opset, and it is then handed to the ONNX exporter as an opset version. This was read out of the ClearML casting code rather than observed end-to-end, so treat it as a strong warning rather than a certainty: if you need a pinned opset, prefer editing `src/params.py` to a real integer (and rebuilding the image) over typing one into the UI, and check the export table if you do it from the UI anyway.

### `workspace`

TensorRT builder workspace in GiB, default `4`. Ignored entirely by every other format — it is passed to them (see the filtering note below) but has no effect. Too small and the TensorRT builder cannot try its faster tactics and silently produces a slower engine, or fails outright on a large model; too large and it can push the build into swapping on a busy GPU. `4` is a reasonable middle for the model sizes this template trains. Upstream's default is unset.

### `nms`

Fuses non-maximum suppression into the exported graph, so the artifact outputs final detections instead of raw boxes that the consumer must post-process. Default `False`, which matches what most serving stacks expect — they have their own NMS, and a fused one is harder to tune. Turn it on when the consumer is a runtime with no post-processing layer of its own. Two things to know: when `nms=True` the model's `conf` and `iou` thresholds are baked into the graph, and `args_export["params"]` exposes neither, so you get ultralytics' defaults rather than the operating point reported under `Operating Point` (see [`8_visualization.md`](8_visualization.md)); and a fused-NMS graph exports as a fixed-topology model, which interacts badly with `dynamic`.

### `fraction`

The fraction of the calibration dataset used during INT8 quantisation. Only meaningful when `int8` is on, and given the calibration-data problem above, only meaningful once you have supplied a `data` argument. `1.0` means "use all of it". Note that `export_model_format()` explicitly pops `fraction` out of `yolo.overrides` before exporting: after training, the model carries `fraction=0.9` from `args_train`, and leaving it there would let the *training* subsample setting silently become the *calibration* subsample setting.

### `keras` and `optimize`

`keras` applies to `saved_model` only and is filtered out of every other export — `test_filter_export_parameters_drops_unsupported` asserts exactly that. `optimize` used to mean TorchScript mobile optimisation; in ultralytics 8.4.110's `default.yaml` it is documented as DEEPX-only, so for every format this group can enable it is accepted and ignored. Leave both at `False`.

### How parameters are filtered per format — and the trap in it

`filter_export_parameters()` in `src/yolov8/exporter.py` holds a `FORMAT_PARAMETERS` table with explicit allow-lists for `onnx`, `torchscript` and `openvino`, and a special case for `engine` that passes everything through. The lookup is `FORMAT_PARAMETERS.get(format_model, [])` — note the default is an **empty list, not `None`**. So any format not named in that table (`coreml`, `saved_model`, `pb`, `tflite`, `edgetpu`, `tfjs`, `paddle`) receives **no parameters at all**: no `half`, no `dynamic`, no `nms`. Those exports run at FP32 with every default, and the `params` group you carefully set has no effect on them. If you enable one of those formats and are surprised the artifact is not FP16, this is why. The `if allowed_params is None: return params.copy()` branch that the docstring describes is unreachable, because the only entry whose value is `None` is `engine`, which returns from an earlier branch in both functions.

## What registration puts in the Models tab

Every export that produces an artifact is immediately handed to `register_model_to_clearml()` in [`src/utils/register_model.py`](../../src/utils/register_model.py). The same function is called twice more from `on_train_end` for `best.pt` and `last.pt`, which is why PyTorch weights appear even with every format switch off.

For each registered model you get:

- **Name** — `{format}-{model_name}`, e.g. `onnx-yolo11n-seg`, `engine-yolo11n-seg`. The two PyTorch registrations add a suffix: `pytorch-yolo11n-seg-best` and `pytorch-yolo11n-seg-last`.
- **Uploaded filename** — `{name}.{format}`, e.g. `onnx-yolo11n-seg.onnx`. For `PyTorch` the extension is normalised to `.pt`.
- **Tags** — the YOLO task (`segment` / `detect` / `classify`), the ClearML task id, and `candidate`; `best` is added for `best.pt`. Metric tags such as `map50:0.873`, `map:0.612` and, on a segmentation run, `mask_map50:` / `mask_map:` are also attached — but **only to the two PyTorch registrations**, because `on_train_end` is the only caller that passes a `metrics` dict. Exported-format models are registered without quality tags, so you cannot rank ONNX or TensorRT artifacts by mAP in the Models list. Rank the PyTorch entry from the same task instead.
- **Comment** — the full class-name list, so a model found later in the registry carries its own label vocabulary.
- **Label enumeration** — name-to-index mapping, taken from the dataset's `data.yaml`.
- **Design / metadata** — `net`, `imgsz` and `task` as the design config, plus `imgsz`, `task` and `format` as typed metadata fields.

Artifacts that are directories rather than files — OpenVINO's `_openvino_model/`, and PaddlePaddle's — are handled by ClearML itself: `update_weights()` detects a directory and delegates to `update_weights_package()`, so they land as a zipped package. The exporter's `_size_mb()` walks directories for the same reason, so the size reported for an OpenVINO export is the total of the directory contents.

Alongside the models, `export_handler()` reports a single ClearML table under **Plots → `Export` / `Formats`**, one row per requested format, with `Format`, `Status`, `Size (MB)`, `Export time (s)` and `Artifact`. Failures are rows, not omissions — an absent row would be indistinguishable from a format that was never requested, whereas `failed: ValueError` or `skipped (no CUDA)` tells you what actually happened. This table is the fastest way to compare deployment targets, since it puts artifact size next to build time.

## Scenarios

### Exporting ONNX for a CPU deployment

Set `format/onnx = 1`, and set `params/half = False`. That second change is the one people skip: with `half` left at its default the artifact is an FP16 graph, which is the wrong precision for CPU inference. Leave `simplify` at `False` unless the target runtime rejects an op in the graph — turning it on installs `onnxruntime` and `onnxslim` in the container on every run. Leave `opset` at `None` and `dynamic` at `False` unless you know the consumer needs otherwise. Expect one row in the `Export`/`Formats` table and a new `onnx-<model>` entry in the Models tab, and expect the console to read `export: 1/1 formats ok`.

### Exporting TensorRT for the GPU that will serve it

Set `format/engine = 1` and keep `half: True` — FP16 is the point of TensorRT on modern NVIDIA hardware. Raise `workspace` if the builder complains it cannot find a tactic. Two constraints dominate: the task must run on an agent with a GPU (the exporter forces `device="0"` and ultralytics asserts a non-CPU device), and `tensorrt` is not baked into the image, so the first export attempts a live install — if your agents have no outbound network, this will fail at the install step, and the fix is to add TensorRT to `pyproject.toml` and `make build` rather than to retry. Remember the engine is only valid on the same GPU architecture and TensorRT version it was built on; registering it to ClearML does not make it portable. It is usually worth enabling `onnx` alongside `engine` so the task also carries a portable fallback — and the ONNX export happens anyway, since TensorRT builds from it.

### Checking whether an exotic format works before depending on it

Enable exactly one format on a cheap task — one epoch, a small `fraction` — and read the `Export`/`Formats` table. `ok` with a plausible size means it works in this image. `failed: ValueError` for `tfjs` means the format name does not exist upstream. A row that never appears means the loop never reached that format because a previous one hung, most likely inside an auto-install. Do this once and record the answer; do not discover it at the end of a fifteen-hour training run.

## Gotchas

- **Editing the exporter has no effect on a remote run until the image is rebuilt.** The container sets `PYTHONPATH=/workspace` and bakes a copy of `src/`, so `src.yolov8.exporter` resolves to the image's copy while `src/train.py` runs from the git checkout. Run `make bump PART=patch && make build` after changing `src/yolov8/exporter.py`. The same applies to the defaults in `src/params.py` — though for an existing task the ClearML UI values win over the code defaults anyway.
- **Never re-tag a published image version.** A ClearML task stores the image tag it was created with and keeps requesting it, so overwriting an existing tag silently changes what every past task reruns on.
- **Export runs after the final `val()`, in a `try` that is not shared with it.** If validation raised, the pipeline logs the exception and still proceeds to export — so a task can contain healthy artifacts and no final metrics. Read the console for `Error during validation` before trusting an export from a task whose plots look thin.
- **A failed export never stops the run.** Each format is attempted inside its own `try`, and the loop continues. That is deliberate — one missing dependency should not cost you the other artifacts — but it means "the run finished" is not evidence that the artifact you wanted exists. Check `export: N/M formats ok`.

## Related groups

[`0_console.md`](0_console.md) · [`1_task.md`](1_task.md) · [`2_data.md`](2_data.md) · [`3_augment.md`](3_augment.md) · [`4_training.md`](4_training.md) · [`5_testing.md`](5_testing.md) · [`6_predict.md`](6_predict.md) · **7_export** · [`8_visualization.md`](8_visualization.md)

`args_train["imgsz"]` from [`4_training.md`](4_training.md) is the image size every export is frozen at, and `args_task["model_name"]` from [`1_task.md`](1_task.md) is what every registered model is named after. The `Operating Point` threshold that a fused-NMS export should ideally use is produced by [`8_visualization.md`](8_visualization.md).
