# `4_Training` — `args_train`

This group is the training call itself: how long the run is, which optimiser and learning-rate schedule it uses, how the loss terms are weighted, how much hardware it consumes, what it writes to disk, and which task-specific switches are active. It is defined as `args_train` in `src/params.py`, published to the ClearML UI by `curr_task.connect(args_train, name="4_Training")` in `src/utils/clearml_settings.py`, and appears under **Configuration → Hyperparameters → 4_Training**. It is also the dict that everything else is folded into: `args_train` is what `src/train.py` finally passes to `model_yolo.train()`, so keys from `3_Augment` and the internal logging dict travel to ultralytics through this group.

## How these values reach the model

```python
# src/utils/clearml_settings.py, config_clearml()
curr_task.connect(args_train, name="4_Training")
...
args_train.update(args_logging)     # {"project": "YOLO/Training", "name": "training-yolo"}
args_train.update(args_augment)     # the whole 3_Augment group
```

```python
# src/train.py, main()
args_val["imgsz"] = args_train["imgsz"]
model_yolo.train(data=data_yaml_file, **args_train)
```

Four consequences worth internalising before you change anything:

- **Ultralytics rejects unknown keys outright.** A key added to `args_train` that is not a valid YOLO argument fails the run with `SyntaxError: '<key>' is not a valid YOLO argument`. This is why console verbosity lives in its own `args_console` dict — see [`0_console.md`](0_console.md).
- **`project` and `name` in this group are always overwritten.** `args_logging` is merged *after* the UI connect and is not itself connected to ClearML, so the values you see in the `4_Training` group (`None`/`None`) are replaced by `YOLO/Training` and `training-yolo` on every run, and editing them in the UI has no effect. The run's output directory is therefore always `YOLO/Training/training-yolo`.
- **`imgsz` is propagated to validation** by `src/train.py` before the train call, and to export by `src/yolov8/exporter.py`, which reads `args_training["imgsz"]`. It is one value for the whole pipeline.
- **`src/schema/params.py::TrainParams` documents ranges but does not enforce them here.** It is a Pydantic model exercised by `tests/test_critical_fixes.py`; nothing in `src/train.py` validates the UI values through it. The real validation is ultralytics' own: `check_imgsz()` rounds `imgsz` up to a multiple of the model stride with a warning, and `check_dict_alignment()` rejects unknown keys. Some schema defaults (`epochs=1000`, `batch=2`) deliberately differ from the runtime defaults in `src/params.py`; the runtime dict is the source of truth.

## Quick reference

| Key | Default | Type / range | One-line meaning |
|---|---|---|---|
| `epochs` | `20` | int ≥ 1 | Number of passes over the training split. |
| `patience` | `0` | int ≥ 0 | Early-stopping patience in epochs. **`0` means never stop early.** |
| `close_mosaic` | `0` | int ≥ 0 | Disable mosaic/mixup/copy-paste/cutmix for the final N epochs. `0` = never. |
| `batch` | `64` | int ≥ 1, or `-1`, or float 0–1 | Images per optimiser step. Any value < 1 enables AutoBatch. |
| `imgsz` | `640` | int, multiple of 32 | Training (and val, and export) image size. |
| `fraction` | `0.9` | float, 0 < f ≤ 1 | **Trains on only this fraction of the training split.** |
| `workers` | `8` | int ≥ 0 | Dataloader worker processes (capped by CPU count and batch count). |
| `cache` | `True` | `True`/`"ram"`/`"disk"`/`False` | Cache decoded images in RAM or as `.npy` on disk. |
| `device` | `None` | `None`, int, `"0,1"`, `"cpu"` | CUDA device selection; `None` auto-selects. |
| `amp` | `True` | bool | Automatic mixed precision. |
| `rect` | `False` | bool | Rectangular batching. **Silently forces `mosaic`/`mixup`/`cutmix` to 0.** |
| `save` | `True` | bool | Write `best.pt` / `last.pt`. |
| `save_period` | `-1` | int | Save `epoch<N>.pt` every N epochs. **`< 1` disables it.** |
| `project` | `None` | str/None | Output dir root. **Overwritten with `YOLO/Training`.** |
| `name` | `None` | str/None | Output dir name. **Overwritten with `training-yolo`.** |
| `exist_ok` | `True` | bool | Reuse the output dir instead of appending a suffix. |
| `pretrained` | `True` | bool | Start from pretrained weights rather than random init. |
| `resume` | `False` | bool | Resume an interrupted run. Set to `True` automatically by `model_latest_id`. |
| `optimizer` | `"auto"` | `SGD`/`Adam`/`Adamax`/`AdamW`/`NAdam`/`RAdam`/`RMSProp`/`auto` | Optimiser. **`auto` overrides `lr0` and `momentum`.** |
| `lr0` | `0.001` | float > 0 | Initial learning rate. Ignored when `optimizer: auto`. |
| `lrf` | `0.0001` | float > 0 | Final LR as a fraction of `lr0`. |
| `momentum` | `0.937` | float 0–1 | SGD momentum / Adam beta1. Ignored when `optimizer: auto`. |
| `weight_decay` | `0.0005` | float ≥ 0 | L2 penalty, rescaled by `batch * accumulate / nbs`. |
| `warmup_epochs` | `3.0` | float ≥ 0 | Linear warmup length; fractions allowed. |
| `warmup_momentum` | `0.8` | float 0–1 | Momentum at the start of warmup. |
| `warmup_bias_lr` | `0.1` | float ≥ 0 | Bias-group LR at the start of warmup. Forced to `0.0` by `optimizer: auto`. |
| `cos_lr` | `False` | bool | Cosine LR decay instead of linear. |
| `nbs` | `64` | int ≥ 1 | Nominal batch size; sets gradient accumulation and rescales `weight_decay`. |
| `box` | `7.5` | float > 0 | Box-regression loss gain. |
| `cls` | `0.5` | float > 0 | Classification loss gain. |
| `dfl` | `1.5` | float > 0 | Distribution-focal-loss gain. |
| `pose` | `12.0` | float > 0 | Pose loss gain. **Pose task only — inert here.** |
| `kobj` | `2.0` | float > 0 | Keypoint objectness gain. **Pose task only — inert here.** |
| `dropout` | `0.0` | float 0–1 | **Classification task only — inert here.** |
| `label_smoothing` | `0.0` | float 0–1 | **Removed from ultralytics 8.4.x — accepted with a deprecation warning, then dropped.** |
| `overlap_mask` | `True` | bool | Segment only: allow masks to overlap in the training target. |
| `mask_ratio` | `4` | int ≥ 1 | Segment only: mask downsample ratio. |
| `single_cls` | `False` | bool | Collapse every class to one during training. |
| `val` | `True` | bool | Validate each epoch. The final epoch validates regardless. |
| `augment` | `True` | bool | Test-time augmentation flag. **Inert during training**; forced to `False` for segment runs. |
| `seed` | `0` | int ≥ 0 | RNG seed. |
| `deterministic` | `True` | bool | Deterministic cuDNN/torch algorithms. |
| `verbose` | `True` | bool | Ultralytics' own training-side verbosity. |
| `profile` | `False` | bool | Profile ONNX/TensorRT speeds during training. |

## Run length and stopping

### `epochs` (default `20`)

The number of full passes over the (fractioned) training split. `20` is a "get an answer today" default, not a converged-model default; a pretrained YOLO backbone on a few thousand images typically keeps improving for 100-300 epochs. The trainer's LR schedule is built from `epochs`, so changing it changes the shape of the whole run rather than just its length — a 20-epoch run and the first 20 epochs of a 200-epoch run are different runs.

**Reporting cadence, and why `epochs` is safe to raise.** This is the sibling of the logging rule that governs the data stage — log volume must not grow with dataset size — applied to the training stage instead: *report volume must not grow with epoch count*. Concretely, in `src/yolov8/callbacks.py`:

- `on_train_epoch_end` reports learning rates, individual loss components and loss ratios — all **scalars**, which are series and are meant to grow one point per epoch.
- `on_fit_epoch_end` reports validation metrics, epoch time, speed metrics and the mask-vs-box gap — again all scalars.
- Everything heavy — per-class tables, PR/F1 curve families, the operating-point report, the worst-image gallery, the calibration and reliability plots — lives in `report_validation_analysis()`, which runs **once per validation run**, not per epoch. `tests/yolov8/test_callbacks.py::TestReportVolume` enforces this by running the per-epoch hooks at 3 and 30 epochs and requiring an identical heavy-report count.

So a 300-epoch run costs 300 points per scalar series and the same number of tables and galleries as a 3-epoch run. Two exceptions are worth knowing: the `Mosaic` debug images are uploaded only at epochs 0 and 1, and `save_period` (below) is the one knob that *does* make artifact volume scale with epochs.

### `patience` (default `0`)

Epochs to wait after the best fitness before stopping early. **`0` disables early stopping entirely** — ultralytics does `self.patience = patience or float("inf")` in `EarlyStopping.__init__`, so a falsy value becomes infinity, not "stop immediately". With the default, a run always trains all `epochs`.

That is a reasonable pairing with `epochs: 20` (there is nothing to save) and a poor one with `epochs: 300` (you burn GPU hours after convergence). If you raise `epochs`, raise `patience` with it — roughly 20-30% of `epochs` is a common choice. Note that fitness for a segmentation model is `seg.fitness() + box.fitness()`, a 0-2 sum, so early-stopping deltas are on a different scale than for a detection model.

`best.pt` is written whenever fitness improves regardless of `patience`, so a long run without early stopping still yields the right weights — it just wastes time getting there.

### `close_mosaic` (default `0`)

Turns off mosaic, mixup, copy-paste and cutmix for the final N epochs. `0` means never. It is documented in full, together with `mosaic`, in [`3_augment.md`](3_augment.md#mosaic-and-close_mosaic-one-mechanism-two-ui-groups) — the short version is that this repository's `0` deviates from ultralytics' default of `10`, and means the model never trains on an un-collaged image.

### `val` (default `True`)

Whether to run validation after each epoch. Setting it to `False` does not skip validation entirely: `engine/trainer.py:575` gates on `if self.args.val or final_epoch or self.stopper.possible_stop or self.stop`, so the last epoch validates regardless. What you lose is the per-epoch `Metrics/*` and `Losses/Validation` scalar series, and with them any ability to see overfitting as it happens or to select a meaningful `best.pt`. Turning it off buys back the validation time per epoch; on a small val split that is a few percent of the run, which is rarely worth blinding yourself for.

## Optimiser and learning rate

### `optimizer` (default `"auto"`) — read this before tuning `lr0`

With `optimizer: "auto"` ultralytics **ignores `lr0` and `momentum` and picks all three itself** (`engine/trainer.py::build_optimizer`):

```python
if name == "auto":
    LOGGER.info("'optimizer=auto' found, ignoring 'lr0=...' and 'momentum=...' ...")
    nc = self.data.get("nc", 10)
    lr_fit = round(0.002 * 5 / (4 + nc), 6)
    name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
    self.args.warmup_bias_lr = 0.0
```

So on a default run the `lr0: 0.001` and `momentum: 0.937` visible in the ClearML UI are **not the values used**, and `warmup_bias_lr: 0.1` is silently replaced by `0.0`. The chosen values are printed to the console at the start of training, and the learning rate actually applied is reported per epoch as the `LR` scalar group by `on_train_epoch_end` — that scalar, not the UI parameter, is the truth. Which branch is taken depends on total iterations (`epochs × batches_per_epoch`), so a longer run can silently switch you from AdamW to MuSGD.

If you want your `lr0` respected, name an optimiser explicitly: `optimizer: AdamW` or `optimizer: SGD`. Rule of thumb for the LR that goes with it — Adam-family around `1e-3` and below, SGD around `1e-2`.

### `lr0` (default `0.001`) and `lrf` (default `0.0001`)

`lr0` is the initial LR; the schedule decays it to `lr0 * lrf` at the last epoch, linearly by default or on a cosine if `cos_lr` is set. This repository's `lrf: 0.0001` is a **100× smaller final LR** than ultralytics' own `0.01` default, i.e. the LR is driven to essentially zero by the end of the run. That is a good fit for short fine-tuning runs (the tail epochs act as a fine polish) and a poor one for long runs from scratch, where the model spends its last third barely moving. If you lengthen a run substantially, consider raising `lrf` towards `0.01`.

Both are ignored in the `lr0` sense when `optimizer: auto` — the decay *shape* still applies, but to the auto-selected initial LR.

### `momentum` (`0.937`), `warmup_epochs` (`3.0`), `warmup_momentum` (`0.8`), `warmup_bias_lr` (`0.1`)

Warmup ramps the LR from near zero to `lr0`, and the momentum from `warmup_momentum` to `momentum`, over the first `warmup_epochs` epochs (fractional values are allowed and are converted to iterations). The bias parameter group starts at the much higher `warmup_bias_lr` and decays into the normal schedule, which stabilises the detection head early.

`warmup_epochs: 3.0` out of `epochs: 20` means **15% of the default run is warmup**. On very short runs that is a large fraction; on a 2-epoch smoke run it means the model never reaches its nominal LR at all. Lower it to `0.5-1.0` for short runs. Note again that `optimizer: auto` zeroes `warmup_bias_lr`.

### `weight_decay` (`0.0005`) and `nbs` (`64`)

`nbs` ("nominal batch size") is the batch size the hyperparameters are calibrated for, and it drives two things (`engine/trainer.py:287-288`):

```python
self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
```

With `batch: 64` and `nbs: 64`, `accumulate` is 1 and `weight_decay` is used as written. Drop `batch` to 16 to fit a smaller GPU and `accumulate` becomes 4 — the optimiser steps every 4 batches, so the *effective* batch stays 64 and the decay stays correctly scaled. This is the mechanism that makes lowering `batch` for memory reasons roughly loss-neutral, and it is a good reason to leave `nbs` alone. During warmup, `accumulate` is additionally interpolated up from 1.

### `cos_lr` (default `False`)

Cosine annealing instead of the linear ramp-down. Generally worth a try on runs long enough for the schedule shape to matter (100+ epochs); irrelevant on a 20-epoch run.

## Loss weighting

`box` (`7.5`), `cls` (`0.5`) and `dfl` (`1.5`) are the ultralytics defaults and scale the three detection loss terms. They are the last thing to tune, not the first — the defaults were fit across COCO-scale training and are rarely the reason a model underperforms.

The one honest signal for touching them is reported by `_report_loss_ratios` in `src/yolov8/callbacks.py`, which logs each component **relative to box loss** under `Losses/Balance`. Absolute losses are not comparable between runs with different gains; the ratios are. A `cls_over_box` ratio that climbs while validation mAP stalls says the classifier is the bottleneck and `cls` may deserve raising; the reverse says the same for localisation.

**Inert keys in this group.** Three loss-related keys exist in `args_train` but do nothing for the detection and segmentation models this template trains:

- `pose: 12.0` and `kobj: 2.0` are the pose-estimation loss gains. They only apply to `-pose` models, and nothing in this repository trains one. Changing them has no effect.
- `dropout: 0.0` applies to classification training only.
- `label_smoothing: 0.0` is worse than inert — it has been **removed** from ultralytics. `cfg/__init__.py` lists it in `removed_keys = {"label_smoothing", "save_hybrid", "crop_fraction"}` and pops it with a deprecation warning, so the value never reaches any loss function in the pinned version (8.4.110). It survives in `src/params.py` only because ultralytics removes it politely instead of erroring.

## Segmentation-specific

`overlap_mask: True` and `mask_ratio: 4` only apply when the model is a segmentation model (`model_name: yolo11n-seg`, the default — see [`1_task.md`](1_task.md)). `overlap_mask` packs overlapping instance masks into a single sorted target map rather than one plane per instance, which is faster and is what the standard recipe assumes. `mask_ratio: 4` trains the mask head at `imgsz/4` and is the standard trade of mask sharpness against memory; lowering it to `1` sharpens masks and increases memory use significantly.

Beware of a related trap that lives in `5_Testing`, not here: mask mAP is computed at prototype resolution unless `save_json` or `save_txt` is set, so mask mAP from two runs with different `save_json` settings is not comparable. See [`5_testing.md`](5_testing.md).

`single_cls: False` collapses every class to class 0 when true. It is a legitimate diagnostic — if `single_cls: True` produces a far better mAP than the multi-class run, the problem is classification, not detection — but it must never be left on for a model you intend to deploy.

## Hardware, throughput, and the shared-memory story

### `batch` (default `64`)

Images per optimiser step, and the main GPU-memory knob. Any value below 1 enables AutoBatch (`engine/trainer.py:382`): `-1` targets 60% of free CUDA memory, and a float in `(0.0, 1.0)` is used as that memory fraction directly (`utils/autobatch.py::check_train_batch_size`). AutoBatch profiles batch sizes `[1, 2, 4, 8, 16]`, extended to 64 on GPUs with ≥16 GB total. It is disabled on CPU/MPS and when `torch.backends.cudnn.benchmark` is on, falling back to the default batch size with a warning in both cases.

Ultralytics 8.4.110 also has an OOM safety net: if a CUDA OOM occurs during the **first** epoch on a single GPU, the trainer halves `batch`, rebuilds the dataloaders/optimiser/scheduler and restarts the epoch, up to 3 times (`engine/trainer.py:504-522`). An OOM in epoch 2 or later still crashes the run — which is why an OOM that appears mid-run is usually caused by something other than batch size (a validation pass at a larger effective batch, cached images filling RAM, or another job landing on the same GPU).

Because `nbs` gradient accumulation compensates, lowering `batch` to fit memory is close to free in terms of final quality; it costs wall-clock time, not accuracy.

### `workers` (default `8`) and shared memory

The number of dataloader worker *processes*. The effective count is `min(os.cpu_count() // max(num_cuda_devices, 1), workers, batches_per_epoch)` (`data/build.py:359`), so 8 is an upper bound rather than a promise, and it is forced to `0` on a CPU/MPS device.

**This is the key that makes `--ipc=host --shm-size=50gb` necessary.** Those two flags are in `DOCKER_ARGUMENTS` in `src/params.py`, with the comment that says why:

```python
# Dataloader workers exchange batches through shared memory; Docker's 64MB
# default is far too small and shows up as a silent worker crash mid-epoch.
"--ipc=host",
"--shm-size=50gb",
```

PyTorch dataloader workers return tensors to the main process through `/dev/shm`. A container started without those flags gets Docker's 64 MB default, which one batch of `64 × 3 × 640 × 640` float tensors exceeds many times over. The failure is not a clean error: workers die and the run reports `DataLoader worker (pid N) is killed by signal: Bus error` — or, worse, simply hangs. If you see that, the container arguments are wrong, not `workers`. `init_clearml()` stamps both onto every task via `set_base_docker()`, and both must be passed together with `docker_image` because `set_base_docker()` replaces the whole container section.

Two more forces act on this number. Python 3.14 defaults `multiprocessing` to `forkserver` on Linux, which re-imports `__main__` in every worker — this is why all work in `src/train.py` lives inside `main()` behind an `if __name__ == "__main__":` guard, and why raising `workers` does not re-initialise ClearML or re-download datasets. And callbacks fire only in the main process, so no amount of workers can duplicate or race the module-level `val_stats` accumulator in `src/yolov8/callbacks.py`.

### `cache` (default `True`)

`True` is normalised to `"ram"` by `data/base.py:136`. The dataset then decodes every image once and holds it in memory, which removes JPEG decoding from the per-epoch cost and is usually the single biggest throughput win available. Ultralytics checks first: `check_cache_ram()` estimates the requirement with a 0.5 safety margin, and if it does not fit it warns and silently falls back to no caching — so a run that mysteriously got slower after the dataset grew is likely one where the cache stopped fitting.

`"disk"` writes decoded `.npy` files next to the images instead, trading disk space for RAM. `False` decodes every image every epoch.

Note that `cache` interacts with the shared-memory story: RAM-cached images live in the parent process and are copied into workers on access, so a large cache plus many workers is the combination most likely to exhaust container memory. Separately, `src/train.py` calls `cleanup_cache(dataset_folder)` after training, which deletes the `labels.cache` files (label metadata, not image data) from each split.

### `device` (default `None`)

`None` lets ultralytics pick — the first available CUDA device, else CPU. Accepts an int, a comma-separated string for DDP (`"0,1"`), or `"cpu"`. On a ClearML agent the visible GPUs are decided by the agent's queue configuration and `--gpus all` in `DOCKER_ARGUMENTS`, so leaving this `None` is usually correct; a hard-coded `device: 1` on a single-GPU agent fails. Validation has its own `device` in [`5_testing.md`](5_testing.md), defaulted to `0`.

Also relevant: `torch` is pinned to a CUDA 12.x wheel window (`>=2.9,<2.10`) because newer torch resolves to a CUDA 13 wheel that, if the host driver is too old, **silently falls back to CPU instead of erroring**. A run that is inexplicably 50× slow with `device: None` is worth checking against that.

### `amp` (default `True`)

Mixed precision. Roughly halves activation memory and speeds up training substantially on any modern NVIDIA GPU. Ultralytics runs `check_amp()` first and disables it if the check fails, so `True` is safe. Turn it off only when debugging a suspected numerical problem — the trainer has its own NaN-recovery path that reloads the last checkpoint, so NaN losses are not automatically an AMP bug.

### `rect` (default `False`)

Rectangular batching sorts images by aspect ratio and letterboxes each batch to the minimum common shape, cutting padding and therefore compute. The cost is hidden in `data/dataset.py:310-312`: when `rect` is true, ultralytics forces `mosaic`, `mixup` and `cutmix` to `0.0`, because those transforms need a fixed square canvas. So `rect: True` is not a throughput tweak, it is a decision to train without multi-image augmentation. Leave it `False` unless you have measured that the augmentation is not helping.

### `seed` (`0`), `deterministic` (`True`)

`init_seeds(seed, deterministic)` seeds Python, NumPy and torch, and with `deterministic` also sets `torch.use_deterministic_algorithms(True, warn_only=True)`, `cudnn.deterministic = True`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PYTHONHASHSEED`. `warn_only=True` means an operation with no deterministic implementation warns rather than raising, so this cannot break a run — but it does cost some throughput, and it does not make a run bit-reproducible across different GPUs or worker counts. Keep it on for comparable experiments; turn it off if you are chasing maximum throughput and do not need reproducibility.

### `profile` (default `False`)

Profiles ONNX and TensorRT speeds during training for loggers. It adds startup cost and produces numbers that the standalone export step in [`7_export.md`](7_export.md) covers better. Leave it off.

## Data selection

### `fraction` (default `0.9`) — the one that surprises people

`fraction` is the proportion of the training split actually used. **The default in this repository is `0.9`, not `1.0`, so every default run silently discards 10% of the training images.** Ultralytics' own default is `1.0`.

Worse, the discard is not random. `data/base.py:182-183`:

```python
if self.fraction < 1:
    im_files = im_files[: round(len(im_files) * self.fraction)]  # retain a fraction of the dataset
```

and `im_files` at that point is the **sorted** file list. So `fraction: 0.9` keeps the alphabetically first 90% of filenames and throws away the alphabetical tail. On this template's datasets, which merge several CVAT tasks into one directory, that tail is not a random 10% of the data — depending on the naming convention it can be a whole source, a whole camera, or a whole recording session. Two runs that differ only in which CVAT tasks were added can therefore drop entirely different content.

Set `fraction: 1.0` for any run whose numbers you intend to trust or compare. Use a low value deliberately for smoke runs (`0.05`) where the point is that the pipeline executes, not what it learns. Note that `fraction` applies to the training split only — validation and test are always complete — so it does not corrupt metrics, it just trains on less than you think.

### `imgsz` (default `640`)

Training resolution, and — because `src/train.py` copies it into `args_val` and `src/yolov8/exporter.py` reads it — validation and export resolution too. Must be a multiple of the model stride (32); ultralytics' `check_imgsz()` rounds up with a warning rather than failing, while the Pydantic schema in `src/schema/params.py` rejects non-multiples outright in tests.

Compute scales roughly with `imgsz²`, so `1280` is about 4× the cost of `640` at the same batch size and usually forces `batch` down as well. It is nonetheless the highest-leverage knob for small objects: an object under about 20 px at `640` has very little signal, and doubling `imgsz` is often more effective than any amount of augmentation or schedule tuning. The `Dataset` report from the data stage (see [`2_data.md`](2_data.md)) is where to check object sizes before deciding.

## Checkpointing, resuming and run identity

### `save` (`True`) and `save_period` (`-1`)

`save: True` writes `best.pt` (highest fitness so far) and `last.pt` (most recent epoch) into the run directory. These are the two files `on_train_end` registers into the ClearML model registry, tagged with `map50`/`map` (and the mask equivalents on a segmentation run) by `_registration_metrics`. Turning `save` off means no registered model and nothing for the export and prediction stages to use, so it is only appropriate for a throwaway experiment.

`save_period: -1` **disables periodic checkpoints**; the gate is `if (self.save_period > 0) and (self.epoch % self.save_period == 0)` (`engine/trainer.py:749`), so any value below 1 means "never". Set it to e.g. `10` when you want to be able to roll back to a mid-run state or inspect how a model evolved. This is the one setting in this group whose artifact volume *does* grow with `epochs` — `epochs: 300` with `save_period: 1` produces 300 checkpoint files — so pick a period, not `1`.

### `pretrained` (default `True`)

Start from pretrained weights instead of random initialisation. Which weights is decided in [`1_task.md`](1_task.md) by `model_name`. Setting `False` trains from scratch, which for any dataset smaller than tens of thousands of images is strictly worse, and needs several times the epochs to approach the pretrained result.

### `resume` (default `False`) — and its interaction with `model_latest_id`

`resume` is designed for one thing: continuing an **interrupted run of the same training job**, picking up the optimiser state, EMA, epoch counter and best fitness from `last.pt`. It is not a fine-tuning mechanism, and the difference matters because of what ultralytics does with the rest of your configuration (`engine/trainer.py::check_resume`):

```python
exists = isinstance(resume, (str, Path)) and Path(resume).exists()
last = Path(check_file(resume) if exists else get_latest_run())
ckpt_args = load_checkpoint(last)[0].args
...
self.args = get_cfg(ckpt_args)                       # <- your whole config is replaced
self.args.model = self.args.resume = str(last)
for k in ("imgsz", "batch", "device", "close_mosaic", "augmentations", "save_period",
          "workers", "cache", "patience", "time", "freeze", "val", "plots",
          "distill_model", "save_dir"):                # <- only these survive from overrides
    if k in overrides:
        setattr(self.args, k, overrides[k])
```

**On resume, everything you set in the ClearML UI is discarded except that whitelist.** `epochs`, `lr0`, `optimizer`, every loss gain and the entire `3_Augment` group are restored from the checkpoint's saved args instead. A resumed run therefore reproduces the original run's hyperparameters by design, and any change you made in the UI silently does not apply.

`config_clearml()` couples this to `model_latest_id` from [`1_task.md`](1_task.md):

```python
if args_task["model_latest_id"] != "":
    latest_model = InputModel(model_id=args_task["model_latest_id"])
    path_latest_model = latest_model.get_weights()
    args_train["resume"] = True
    args_task["model_name"] = path_latest_model
```

Setting `model_latest_id` downloads the registered weights, points `model_name` at them, and flips `resume` to `True`. Be aware of what that combination means for the trainer: `resume` is the boolean `True`, not a path, so `check_resume` takes the `get_latest_run()` branch, which globs `./**/last*.pt` relative to the working directory rather than using the weights that were just downloaded. On a fresh agent container with no previous run directory there is nothing to find, and the trainer raises `FileNotFoundError: Resume checkpoint not found`. Treat `model_latest_id` as usable only in a working directory that already contains the run being resumed, and prefer plain fine-tuning — point `model_name` at the weights and leave `resume: False` — when what you actually want is to continue training on new data with new hyperparameters.

Also note `resume_training`'s assertion: if the checkpoint's epoch is already at or beyond `epochs`, the run aborts with "training to N epochs is finished, nothing to resume."

### `project`, `name`, `exist_ok`

`project` and `name` form the output directory (`YOLO/Training/training-yolo`), and as noted at the top of this page **their UI values are overwritten by `args_logging` and cannot be changed from the ClearML UI**. `exist_ok: True` makes ultralytics reuse that directory instead of appending `2`, `3`, … to the name — which keeps paths predictable on an agent, at the cost of overwriting the previous local run's plots and weights. The ClearML task, not the local directory, is the durable record of a run.

The ClearML project and task *name* are a different thing entirely and live in `args_task` — see [`1_task.md`](1_task.md).

### `verbose` (default `True`)

Ultralytics' training-side verbosity: the per-epoch metrics table, the AMP and dataset warnings, the model summary. Keep it on — this is the output you read when a run behaves oddly. Do **not** try to quieten ultralytics with `YOLO_VERBOSE=0`; that env var is read once at import and drops every ultralytics logger to ERROR, taking the per-epoch table with it. Per-call verbosity for validation and prediction is handled separately in [`5_testing.md`](5_testing.md) and [`6_predict.md`](6_predict.md), and both are turned back on automatically at `LOG_LEVEL=DEBUG`.

### `augment` (default `True`)

Despite the name, this is not the training-augmentation switch. In ultralytics' config `augment` means test-time augmentation during prediction (`cfg/default.yaml:68`), and the validator reads `augment = self.args.augment and (not self.training)`, so it is inert during a training run. `src/train.py::_generate_data_yaml()` additionally forces it to `False` for segmentation tasks, which with the default `yolo11n-seg` model is every default run. Training augmentation is configured entirely in [`3_augment.md`](3_augment.md).

## Scenarios

### Scenario 1 — Fast smoke run: does the pipeline work at all?

Goal: exercise CVAT export, class mapping, training, validation, model registration, export and prediction in a few minutes, without caring about the model.

```
# 4_Training
epochs: 2
fraction: 0.05
batch: 16
imgsz: 320
cache: False
workers: 4
warmup_epochs: 0.5
save_period: -1
val: True
```

Expected consequences: with `fraction: 0.05` the dataset is a small alphabetical prefix and metrics are meaningless — that is fine, you are testing plumbing. `warmup_epochs: 0.5` matters here: leaving it at `3.0` on a 2-epoch run means the LR never leaves warmup. `cache: False` avoids spending a minute decoding images you will use twice. `val: True` is kept because the validation path is one of the things being smoke-tested, and the once-per-validation heavy reports (per-class table, curves, operating point, worst-image gallery) are what confirm the reporting chain works. Pair with the all-zero augmentation config in [`3_augment.md`](3_augment.md).

### Scenario 2 — Small dataset, want the best model you can get

Goal: a few hundred to a few thousand images, one GPU, overnight is acceptable.

```
# 4_Training
epochs: 200
patience: 40
fraction: 1.0
batch: 32
imgsz: 640
close_mosaic: 25
optimizer: AdamW
lr0: 0.001
lrf: 0.01
warmup_epochs: 3.0
cos_lr: True
cache: True
workers: 8
save_period: 25
```

Expected consequences: `fraction: 1.0` recovers the 10% of training images the default silently drops — on a small dataset that is the change with the best effort-to-effect ratio on this page. `patience: 40` makes the long schedule safe: if the model converges at epoch 90 the run stops around 130 instead of burning to 200. Naming `AdamW` explicitly means `lr0` and `momentum` are actually used rather than being overridden by `optimizer: auto`; `lrf: 0.01` avoids the near-zero tail that the repository default `0.0001` produces over 200 epochs. `close_mosaic: 25` gives the model 25 un-collaged epochs to finish on. `save_period: 25` yields 8 intermediate checkpoints, which is a reasonable artifact count. Scalar series grow to 200 points each; the heavy reports still run once.

### Scenario 3 — CUDA out of memory

Symptom: `torch.cuda.OutOfMemoryError` at startup or during the first epoch, or the first-epoch auto-reduce warning `CUDA out of memory with batch=64. Reducing to batch=32 and retrying (1/3).`

```
# 4_Training
batch: 16          # or -1 to let AutoBatch pick
imgsz: 640
amp: True
cache: disk        # or False
workers: 4
```

Expected consequences: `nbs: 64` means dropping `batch` from 64 to 16 sets `accumulate: 4`, so the effective batch and the scaled `weight_decay` are unchanged and the run should reach a similar result, just slower per epoch. `batch: -1` is the hands-off alternative — AutoBatch targets ~60% of free CUDA memory, but note it profiles only up to 16 on GPUs with under 16 GB total. Keep `amp: True`; turning it off roughly doubles activation memory and makes the problem worse. If the OOM happens after epoch 1, batch size is probably not the cause — the auto-reduce only guards the first epoch, and a later OOM usually means the validation pass (which has its own `batch` in [`5_testing.md`](5_testing.md)) or another process on the same GPU. Moving `cache` from RAM to `disk` helps host memory, not GPU memory; do it if the symptom is the process being OOM-killed rather than a CUDA error.

### Scenario 4 — Dataloader workers crash mid-epoch

Symptom: training runs for part of an epoch, then `DataLoader worker (pid 1234) is killed by signal: Bus error`, or the run simply hangs with no progress and no error.

This is almost always shared memory, not `workers`. The fix is in the container arguments, and this repository already ships it:

```python
# src/params.py, DOCKER_ARGUMENTS
"--ipc=host",
"--shm-size=50gb",
```

Checklist, in order: confirm the task's container arguments in the ClearML UI actually contain both flags — `set_base_docker()` replaces the whole container section, so a task created before those flags existed, or by code that passed `docker_image` without `docker_arguments`, will not have them; if they are missing, the task needs re-creating or its container section editing. If they are present and workers still die, reduce `workers` (to 4, then 2) and `batch`, since the shared-memory footprint is roughly proportional to `workers × batch × imgsz²`. `workers: 0` loads data in the main process and is a definitive test: if that makes the crash disappear, it was shared memory. Note that `cache: True` (RAM) increases parent-process memory rather than shared memory, so it is a separate axis — if the *main* process is being OOM-killed, `cache: disk` is the lever.

### Scenario 5 — Long production run, and you need the numbers to be trustworthy

```
# 4_Training
epochs: 300
patience: 60
fraction: 1.0
batch: 64
imgsz: 640
close_mosaic: 40
optimizer: AdamW
lr0: 0.001
lrf: 0.01
cos_lr: True
deterministic: True
seed: 0
cache: True
workers: 8
save: True
save_period: 50
val: True
single_cls: False
```

Expected consequences: 300 points per scalar series, one set of heavy reports, and 6 periodic checkpoints — report volume stays flat with `epochs` exactly as `tests/yolov8/test_callbacks.py::TestReportVolume` requires. `deterministic: True` plus a fixed `seed` makes the run comparable against its siblings on the same hardware. `fraction: 1.0` is non-negotiable for a number you are going to publish. Watch three ClearML panels: the `LR` scalars (to confirm the optimiser actually used the LR you set, rather than `auto` having overridden it), `Metrics/mAP` around epoch 260 (where `close_mosaic` fires and mAP should step up), and `Losses/Balance` (a component ratio drifting while mAP is flat is the only evidence-based reason to touch `box`/`cls`/`dfl`).

### Scenario 6 — Continuing from a previously registered model

You have a registered model in ClearML and want to keep training it on more data.

Recommended: set `model_name` in [`1_task.md`](1_task.md) to the weights and leave `resume: False`. That is plain fine-tuning — a fresh optimiser, a fresh schedule, and **your** hyperparameters, including anything you changed in the UI.

Not recommended unless you are literally restarting an interrupted run in its original working directory: setting `model_latest_id`, which flips `resume: True` for you. As described under [`resume`](#resume-default-false--and-its-interaction-with-model_latest_id), that path replaces your entire configuration with the checkpoint's saved args except a fifteen-key whitelist, and resolves the checkpoint by globbing `./**/last*.pt` rather than using the downloaded weights — which on a fresh agent container fails with `Resume checkpoint not found`. When it does work, the run is tagged `resume` by `_set_task_name_on_experiment`, which is the fastest way to tell from the ClearML UI which path a task took.

## See also

- [`0_console.md`](0_console.md) — `log_level` and `progress`; console keys live there, not here, because ultralytics rejects unknown train arguments.
- [`1_task.md`](1_task.md) — `model_name` (which decides detect vs segment, and therefore which keys here are inert) and `model_latest_id` (which flips `resume`).
- [`2_data.md`](2_data.md) — the dataset and split that `fraction` then trims.
- [`3_augment.md`](3_augment.md) — merged into this group at run time; `close_mosaic` and `rect` are the two keys that cross the boundary.
- [`5_testing.md`](5_testing.md) — validation has its own `batch`, `conf`, `iou`, `max_det` and `device`; `imgsz` is inherited from here.
- [`6_predict.md`](6_predict.md) — the post-training prediction gallery.
- [`7_export.md`](7_export.md) — export reads `imgsz` from this group.
- [`8_visualization.md`](8_visualization.md) — which reports are produced per epoch and which once per validation run.
