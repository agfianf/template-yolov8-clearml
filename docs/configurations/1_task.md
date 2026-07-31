# `1_Task` — which model, and which ClearML task

The `1_Task` group decides two separate things that happen to be adjacent: **what model architecture is trained** (`model_name`, `model_latest_id`) and **where a locally launched run creates its ClearML task** (`clearml_project`, `clearml_task_name`). It maps to the `args_task` dict in `src/params.py` and is connected by `config_clearml()` (`src/utils/clearml_settings.py`) as the parameter group `1_Task`, visible in the ClearML UI under **Configuration → Hyperparameters → 1_Task**. `model_name` is the highest-leverage parameter in the whole configuration — it selects the ultralytics task type, which in turn changes how labels are converted, which augmentations run, what the validator computes, and what the exporter registers. The two `clearml_*` keys, by contrast, are read before `Task.connect()` and are inert once a task exists; that is explained in full below. This file also documents `args_logging`, which has no UI group of its own and is too small to deserve a file.

## Quick reference

| Key | Default | Type | Meaning |
|---|---|---|---|
| `model_name` | `"yolo11n-seg"` | str | Ultralytics model to train. The `-seg` / `-cls` suffix is what selects the task type. A name containing `yaml` trains an architecture from scratch. |
| `model_latest_id` | `""` | str | ClearML model id to resume from. Empty disables it. **Half-wired — see below; leave empty.** |
| `clearml_project` | `"YOLO/Training"` | str | Project a *locally launched* run creates its task in. Read before `Task.connect()`, so editing it in the UI renames nothing. |
| `clearml_task_name` | `"yolo-train"` | str | Task name for that same locally launched run. Same caveat. |

And `args_logging`, which is **not** connected to any UI group:

| Key | Default | Type | Meaning |
|---|---|---|---|
| `project` | `"YOLO/Training"` | str | Ultralytics `project` — a directory name, not a ClearML project. |
| `name` | `"training-yolo"` | str | Ultralytics `name` — the run subdirectory. |

## `model_name`

### How the task type is derived

`src/utils/general.py::get_task_yolo_name()` is the whole of it — a substring test on the string you type:

```python
if "-seg" in arg_model_name:   task_yolo = "segment"
if "-cls" in arg_model_name:   task_yolo = "classify"
if "-cls" not in ... and "-seg" not in ...: task_yolo = "detect"
```

So `yolo11n-seg` → `segment`, `yolo11s-cls` → `classify`, and **everything else** → `detect`. That last clause is broader than it looks: `yolo11n-pose` and `yolo11n-obb` both resolve to `detect`, and the pipeline would then build a detection dataset and a detection validator around a pose model. Pose and OBB are not supported here despite `args_train` carrying ultralytics' `pose` and `kobj` loss gains and `args_predict["plot"]` carrying `kpt_radius` / `kpt_line`; those keys are inherited from ultralytics' defaults, not evidence of a working pose path. If you need pose, `get_task_yolo_name` is the function to extend, and `src/yolov8/data.py` is where the converter would need a keypoint branch.

`src/utils/general.py::model_name_handler()` then turns the name into something `YOLO(model=...)` accepts:

- A plain name gets `.pt` appended — `yolo11n-seg` becomes `yolo11n-seg.pt`, which ultralytics downloads from the Ultralytics release assets on first use and caches thereafter.
- A name containing the substring `yaml` is treated as an architecture request: the file `src/yolov8/yolov8.yaml` is registered with `Task.connect_configuration(name="Model YAML")`, copied to `src/yolov8/<model_name>`, and that path is returned. Two consequences worth knowing. First, the **connected** copy is what gets written, so you can edit the architecture in the ClearML UI's Configuration Objects tab and a cloned task will train the edited version. Second, the source is always `yolov8.yaml` regardless of the name you ask for — `model_name: "yolov8s.yaml"` copies `yolov8.yaml`'s content over `src/yolov8/yolov8s.yaml` and returns that, which is how ultralytics' scale-from-filename convention (`n`/`s`/`m`/`l`/`x`) is exploited. It also means the repo's own `yolov8s.yaml` is overwritten in the checkout on such a run.
- A `.yaml` model is built with **random weights**. `pretrained: True` in [`4_training.md`](4_training.md) does not change that: ultralytics only loads weights from `pretrained` when it is a string path (`engine/trainer.py:798`), and a `.yaml` model carries no checkpoint. Use a `.pt` name unless you specifically want to train from scratch.

### Choosing `yolo11n` versus `yolo11n-seg`

Pick the one that matches your annotations, not the one that sounds more capable.

**A segmentation model on bbox-only annotations is a real mistake, not a graceful degradation.** The converter decides what to write from the annotation content and the task type together (`src/yolov8/data.py:299`): `use_segments = "segmentation" in annotation_type and self.task_model != "detect"`. If the CVAT annotations are rectangles, `annotation_type` contains no `segmentation`, so `use_segments` is `False` and the label files are five-column boxes — regardless of the `-seg` in your model name. A `-seg` model then trains with a mask loss and no mask targets. Mask mAP is meaningless, and depending on the ultralytics version you either get a dataset-verification error or a silent run whose `(M)` metrics are all zero. Either way you have burned the GPU time.

**A detection model on polygon annotations is safe and often correct.** `use_segments` becomes `False` because of the `task_model != "detect"` clause, and the converter derives boxes from the polygons. You lose masks; you gain a faster, smaller model. This is the right choice when the deployment consumes boxes.

One asymmetry to be aware of before trusting a mixed setup: the guard above is applied to the **training** sources only. The test-split loop (`src/yolov8/data.py:318`) reads `use_segments = "segmentation" in annotation_type` with no task-type clause, so a detect run over polygon-annotated data writes box labels for train/valid and polygon labels for the test split. Ultralytics' loader tolerates it — it derives boxes from segments when the task is detect — but the two splits are not written the same way, and any tooling that reads the label files directly will see two formats in one dataset directory.

### Side effects of the task type

Choosing `model_name` changes four things elsewhere, none of which is in the `1_Task` group:

- **`args_train["augment"]` is forced to `False` on a segmentation run.** `_generate_data_yaml()` in `src/train.py` does this unconditionally, so the value you set in `4_Training/augment` is overridden for any `-seg` model. The `3_Augment` values are still passed through and still apply — this is ultralytics' separate `augment` flag, not the augmentation hyperparameters. See [`3_augment.md`](3_augment.md).
- **Classification changes what is passed as `data`.** For `classify`, `data_yaml_file` is the dataset *directory* rather than `data.yaml`, matching ultralytics' folder-per-class convention. The CVAT/COCO path in this repo is built around detect and segment; treat classify as untested.
- **The task type is stamped on the ClearML task as a tag.** `_tagging_handler()` adds `ul-<ultralytics version>`, the task type, `os.path.basename(model_name)` without `.pt`, and the uppercased data source — so a normal run carries tags like `ul-8.3.x`, `segment`, `yolo11n-seg`, `CVAT`.
- **It reaches the registered model.** `register_model_to_clearml()` writes `task` into both the metadata and the design `config_dict`, and every model is tagged with the task type. See [`7_export.md`](7_export.md).

## `model_latest_id`

### What it is meant to do

Set it to the id of a ClearML model (the id shown on the Models page, or in the `registered <name> with ClearML: <id>` console line) and the run should continue training from those weights instead of from a fresh checkpoint.

### What it actually does today

`config_clearml()` (`src/utils/clearml_settings.py:87-93`) does exactly three things:

```python
if args_task["model_latest_id"] != "":
    logger.info("Downloading latest model")
    latest_model = InputModel(model_id=args_task["model_latest_id"])
    path_latest_model = latest_model.get_weights()
    args_train["resume"] = True
    args_task["model_name"] = path_latest_model
    logger.info("Resume training from %s", latest_model)
```

It downloads the weights into the ClearML cache, sets `args_train["resume"] = True`, and replaces `model_name` with the cache path. `_set_task_name_on_experiment()` in `src/train.py` then takes the `resume` branch: it adds a `resume` tag to the task, and — because `args_train["resume"]` is truthy — skips `model_name_handler()` entirely, so no `.pt` is appended and no Model YAML is connected. `get_task_yolo_name()` is still called on the new value, meaning **the task type is now inferred from the cache path string**. It happens to work, because `register_model_to_clearml()` names artifacts `pytorch-<model>-best.pt`, so a model registered from a `yolo11n-seg` run is cached under a filename that still contains `-seg`. It is a filename-substring accident, not a lookup, and the model's own `config_dict` — which records `net`, `imgsz` and `task` precisely, and which `register_model_to_clearml()` writes on every registration — is never read.

### Why it does not work as written

Three independent problems, each sufficient on its own. This is why the README still lists "Resume training from registered model" as an open TODO, and why `src/train.py` carries the intended implementation commented out at lines 284-294.

1. **`resume` is set to `True`, not to the checkpoint path.** Ultralytics `check_resume()` (`engine/trainer.py:935-948`) tests `isinstance(resume, (str, Path))`; a bare `True` fails that test and it falls back to `get_latest_run()`, which globs `./**/last*.pt` relative to the working directory. The downloaded weights live in `~/.clearml/cache/`, not under the checkout, so they are never found. On a fresh agent container the glob is empty and the run dies at the start of training with ultralytics' own message: `FileNotFoundError: Resume checkpoint not found. Please pass a valid checkpoint to resume from, i.e. 'yolo train resume model=path/to/last.pt'`. On a host that has trained before, it is worse than an error — it silently resumes the wrong run, whichever `last.pt` happens to be newest on disk.

2. **The registered checkpoints have had their training state stripped.** `final_eval()` (`engine/trainer.py:916-924`) calls `strip_optimizer()` on both `last.pt` and `best.pt`, which sets `optimizer`, `ema`, `updates`, `scaler` and `best_fitness` to `None` and `epoch` to `-1`. Registration happens in `on_train_end`, which fires *after* `final_eval` (`:621` then `:625`), so every model in the registry is a stripped inference checkpoint. `resume_training()` computes `start_epoch = ckpt["epoch"] + 1 = 0` and trips `assert 0 < start_epoch < self.epochs` with "training to N epochs is finished, nothing to resume."

3. **Resuming discards your configuration.** `check_resume()` does `self.args = get_cfg(ckpt_args)` — the checkpoint's arguments replace the current ones wholesale, and only a whitelist (`imgsz`, `batch`, `device`, `close_mosaic`, `save_period`, `workers`, `cache`, `patience`, `val`, `plots`, and a few more) is re-applied from the overrides. `epochs`, `lr0`, every augmentation value, `fraction` — everything you set in `3_Augment` and `4_Training` — is silently replaced by whatever the original run used. That is correct behaviour for a genuine resume-after-crash and almost never what someone cloning a task in the UI expects.

### What to do instead

Leave `model_latest_id` empty. To continue from previously trained weights, do a **warm start** rather than a resume: point `model_name` at the checkpoint file, minus the `.pt` that `model_name_handler()` appends.

```
1_Task/model_name       = /data/checkpoints/pytorch-yolo11n-seg-best
1_Task/model_latest_id  =
4_Training/epochs       = 40
```

Weights are loaded, the optimizer starts fresh, and every parameter in the UI is honoured. Three conditions: the file must exist **inside the agent container** (mount it, or bake it in — a path from your laptop means nothing on the agent); the filename must keep its `-seg` / `-cls` marker so `get_task_yolo_name()` still infers the right task; and the class order must match the checkpoint's, which means pinning `2_Data/class_names` to the model's own `names` — a derived order shifts every index after any newly added class. See [`2_data.md`](2_data.md).

## `clearml_project` and `clearml_task_name`

**These two are read before `Task.connect()` and therefore cannot be edited from the UI.** `init_clearml()` runs first in `main()` and passes them straight to `Task.init()`:

```python
curr_task = Task.init(
    project_name=args_task["clearml_project"],
    task_name=args_task["clearml_task_name"],
    reuse_last_task_id=False,
    auto_connect_frameworks={"pytorch": False, "matplotlib": False},
)
```

Only `config_clearml()` — which runs afterwards — connects `args_task` to the UI. So the values are visible in the `1_Task` group, they are editable, they are stored on the task, and **nothing reads them again**. Editing `clearml_project` on an existing task moves nothing, and a clone of that task inherits the project it was cloned from, not the string in the field. Treat both as a record of how the task was originally created.

Note also that the `Task.init()` block is guarded by `if curr_task is None`. On an agent, ClearML has already created the current task before `src/train.py` starts, so the block is skipped entirely — another reason the two values only ever matter on a locally launched run. `reuse_last_task_id=False` means each such launch creates a brand-new task rather than overwriting the previous one.

Two other things happen in that same locally-launched-only block and are worth knowing when a task ends up somewhere unexpected: `set_script()` hard-codes the repository URL, `branch="main"` and `entry_point="src/train.py"`, and `set_base_docker()` stamps `DOCKER_IMAGE` / `DOCKER_ARGUMENTS` from `src/params.py` onto the task. A task also pins the commit it was created at, so pointing it at a branch is not enough to move it to newer code — clear `version_num` if you need it to follow the branch.

**To actually move or rename an existing task**, use ClearML's own UI: the task context menu has Rename and Move to Project. To change the default for everything created from now on, edit `src/params.py` and launch a new local run.

**Naming trap.** `clearml_project` and `args_logging["project"]` share both a default value (`"YOLO/Training"`) and a name, and they are unrelated: one is a ClearML project, the other is a directory on disk.

## `args_logging` — `project` and `name`

`args_logging` is a two-key dict in `src/params.py` that is **never connected to ClearML**. `config_clearml()` connects nine groups and `args_logging` is not among them; instead it is merged into the training arguments:

```python
args_train.update(args_logging)
```

Two things follow, and the second is the one that bites.

**These are ultralytics' `project` and `name`, i.e. the output directory.** `get_save_dir()` (`ultralytics/cfg/__init__.py:518-532`) nests a relative `project` under the ultralytics runs directory and the task type, so the defaults produce `<runs_dir>/segment/YOLO/Training/training-yolo`. That is where `weights/best.pt`, `weights/last.pt`, the ultralytics PNGs and the `visualizations/` panels land, and it is the `save_dir` that the worst-image gallery in [`8_visualization.md`](8_visualization.md) reads from. Because `args_train["exist_ok"]` is `True`, a second run in the same container reuses and overwrites that directory instead of incrementing to `training-yolo2` — fine on an agent, where each task gets a fresh container, and a real hazard locally.

**Setting `project` or `name` in the `4_Training` UI group is inert.** `args_train` does declare both keys (as `None`), and they are connected as part of group `4_Training` — but `args_train.update(args_logging)` runs at the *end* of `config_clearml()`, after every `connect()` call, and overwrites them unconditionally with the literals from `src/params.py`. Whatever you type into `4_Training/project` is discarded before training starts. Changing the output directory today therefore means editing `src/params.py` — and on an agent, `src.*` imports resolve to the image's baked copy of `src/` rather than the git checkout, so it also means `make build`. If this needs to become a UI knob, the fix is to connect `args_logging` as its own group and merge it *before* the connects, not after.

Why the keys are split out at all: the same reason `args_console` is split out. Anything merged into `args_train` reaches `model_yolo.train(**args_train)`, where ultralytics rejects unknown keyword arguments outright — `project` and `name` are valid ultralytics arguments and may live there; `log_level` is not and may not. `tests/utils/test_console_params.py` asserts that separation. See [`0_console.md`](0_console.md).

## Scenarios

### Scenario 1 — detection on box-only CVAT annotations

The annotations are rectangles and the deployment consumes boxes.

```python
args_task = {
    "model_name": "yolo11n",
    "model_latest_id": "",
    "clearml_project": "YOLO/Training",
    "clearml_task_name": "signs-detect",
}
```

`get_task_yolo_name("yolo11n")` finds neither `-seg` nor `-cls`, so the task type is `detect`; `model_name_handler` appends `.pt`. The console reads:

```
2026-07-31 09:13:43 | INFO  | src.train | TASK_YOLO: detect
2026-07-31 09:13:43 | INFO  | src.train | [Downloading Data]
2026-07-31 09:14:29 | INFO  | src.yolov8.data | dataset ready: train 7,520 img / 7,520 lbl, valid 1,880 img / 1,880 lbl
2026-07-31 09:14:30 | INFO  | src.train | [Training]
2026-07-31 09:14:30 | INFO  | src.train | LOAD MODEL: yolo11n.pt, task: detect
2026-07-31 09:14:30 | INFO  | src.train | Override Callbacks
```

`args_train["augment"]` keeps whatever `4_Training` says, since the segment override does not apply. Weights are written under `<runs_dir>/detect/YOLO/Training/training-yolo/weights/`, and at the end of training:

```
2026-07-31 10:41:02 | INFO  | src.yolov8.callbacks | registering yolo11n (detect, imgsz=640) as best + last
2026-07-31 10:41:14 | INFO  | src.utils.register_model | registered pytorch-yolo11n-best with ClearML: 3f1c...e9
```

The task carries the tags `ul-8.3.x`, `detect`, `yolo11n`, `CVAT`.

### Scenario 2 — instance segmentation on polygon annotations

Same data source, but the CVAT tasks are annotated with polygons and masks are wanted.

```python
args_task = {
    "model_name": "yolo11n-seg",
    "model_latest_id": "",
    "clearml_project": "YOLO/Training",
    "clearml_task_name": "yolo-train",
}
```

Task type is `segment`. Three things change relative to Scenario 1 and none of them is visible in the `1_Task` group: the converter writes polygon label files (`use_segments` is `True` because the annotations carry `segmentation` *and* the task type is not `detect`); `_generate_data_yaml()` forces `args_train["augment"] = False`; and the validator produces a second family of metrics, so every `(B)` scalar and curve gains an `(M)` sibling and the mask-versus-box gap is reported per epoch. If those `(M)` numbers come back as zeros, the annotations were rectangles after all — check the `annotation_type=...` DEBUG line from `src/yolov8/data.py`, which needs `0_Console/log_level = debug` to appear.

### Scenario 3 — trying to continue from a registered model

The tempting configuration, and what it actually does:

```
1_Task/model_name      = yolo11n-seg
1_Task/model_latest_id = 8b41f0c0d1e34c8fa1b7e6f2a9c50d33
```

`config_clearml()` logs:

```
2026-07-31 09:13:42 | INFO  | src.utils.clearml_settings | Downloading latest model
2026-07-31 09:13:47 | INFO  | src.utils.clearml_settings | Resume training from <clearml.model.InputModel object at 0x...>
```

`model_name` is now `/root/.clearml/cache/storage_manager/models/.../pytorch-yolo11n-seg-best.pt`, `args_train["resume"]` is `True`, the task gains a `resume` tag, and `TASK_YOLO: segment` is still correct — because the cached filename contains `-seg`. Then training starts and ultralytics ignores that path entirely, globs `./**/last*.pt`, finds nothing in a fresh container, and raises:

```
FileNotFoundError: Resume checkpoint not found. Please pass a valid checkpoint to resume from, i.e. 'yolo train resume model=path/to/last.pt'
```

Do this instead — a warm start, which is what "continue from the last model" almost always means in practice:

```
1_Task/model_name      = /workspace/checkpoints/pytorch-yolo11n-seg-best
1_Task/model_latest_id =
2_Data/class_names     = car, person, speed_limit
4_Training/epochs      = 40
```

giving `LOAD MODEL: /workspace/checkpoints/pytorch-yolo11n-seg-best.pt, task: segment`, a fresh optimizer, all 40 epochs, and every UI parameter respected. Pin `class_names` to the checkpoint's own `names`: without it a newly added class sorts into the middle of the derived order and shifts every index after it, so the loaded head is being fine-tuned against relabelled targets.

## Related groups

- [`0_console.md`](0_console.md) — console verbosity and progress bars
- [`2_data.md`](2_data.md) — data sources, class order, filtering
- [`3_augment.md`](3_augment.md) — augmentation hyperparameters
- [`4_training.md`](4_training.md) — training hyperparameters, including the inert `project` / `name`
- [`5_testing.md`](5_testing.md) — validation
- [`6_predict.md`](6_predict.md) — post-training prediction
- [`7_export.md`](7_export.md) — export formats and model registration
- [`8_visualization.md`](8_visualization.md) — ClearML reporting toggles
