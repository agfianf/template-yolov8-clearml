# `0_Console` — how much the run says, and whether it draws bars

The `0_Console` group is the pipeline's own console verbosity: how much of `src/`'s logging reaches stdout, and whether the long loops in the data stage draw a tqdm progress bar. It maps to the `args_console` dict in `src/params.py` and is connected to ClearML by `config_clearml()` (`src/utils/clearml_settings.py`) as the parameter group `0_Console`, so it appears in the ClearML UI under **Configuration → Hyperparameters → 0_Console** on any task. It controls nothing about *what* is computed and nothing about what is reported to ClearML — for report toggles see [`8_visualization.md`](8_visualization.md) — it only changes what a human reading the console sees.

## Quick reference

| Key | Default | Type | Meaning |
|---|---|---|---|
| `log_level` | `""` | str | Level for the `src` logger tree. Empty means "no opinion": `$LOG_LEVEL` decides, and if that is unset too, INFO. |
| `progress` | `"auto"` | str | tqdm bars in the data stage: `auto` (only when stdout is a TTY), `on` (always), `off` (never). |

Both values are applied in `main()` (`src/train.py`):

```python
set_log_level(resolve_level(args_console["log_level"] or None))
set_progress_mode(args_console["progress"])
_apply_console_verbosity(args_val, args_predict)

task.execute_remotely()
```

## `log_level`

**What it does.** It sets the level of the `src` logger, which is the single logger tree every module in this repo writes to (`src/utils/logging.py` attaches exactly one `StreamHandler` to the logger named `src` and turns `propagate` off — ClearML calls `logging.basicConfig()` on the root logger, and propagating would print every one of our lines twice). Lines are formatted as `%(asctime)s | %(levelname)s | %(name)s | %(message)s` with `%Y-%m-%d %H:%M:%S` dates, which is why the module name is visible on every line and why `grep -cE '\| src\.' console.txt` is a one-command way to measure how much output the pipeline produced.

**Valid values.** A level name in any case (`debug`, `DEBUG`, `Warning`, `error`) or a numeric string (`10`, `20`, `30`) — `resolve_level()` accepts both, via `logging.getLevelNamesMapping()` for names and `str.isdigit()` for numbers. Empty or whitespace-only means "not set" and falls through to `$LOG_LEVEL`, which itself falls through to INFO. This is asserted by `tests/utils/test_console_params.py::test_console_defaults_defer_to_the_environment` and by the parametrised cases in `tests/utils/test_logging.py::test_resolve_level_from_env`.

**Why the default is empty.** An empty string is the only value that means "defer". If the default were `"INFO"` it would silently beat `LOG_LEVEL=debug` set on the container or on the shell, and the environment variable — the only lever available before the ClearML parameters are connected at all — would become useless.

**What goes wrong if it is set wrong.** Nothing silent. An unreadable value falls back to INFO *and* warns, once per distinct bad value rather than once per module that asks for a logger:

```
2026-07-31 09:14:02 | WARNING | src.utils.logging | unreadable log level from explicit value: 'verbose' -- falling back to INFO
```

The same line appears with `$LOG_LEVEL` in place of `explicit value` when the environment variable is the unreadable one. `tests/utils/test_logging.py::test_unreadable_level_warns_rather_than_failing_silently` and `::test_unreadable_level_warns_once_not_once_per_module` pin both halves of that behaviour. The one value to actually avoid is `0`: it parses as a number, and level 0 is `NOTSET`, which on a logger with `propagate=False` means nothing is emitted at all. Use `10` if you want DEBUG numerically.

## `progress`

**What it does.** It decides whether `src.utils.logging.progress()` draws a tqdm bar. Two loops use it today — `splitting files` in `src/data/setup.py` and `coco -> yolo labels` in `src/data/converter/coco2yolo.py` — and both are per-item loops over the dataset, which is exactly where a bar is worth having and a log line is not.

**Valid values.** `auto`, `on`, `off`. They map to tqdm's `disable` argument as `{"auto": None, "on": False, "off": True}`; `disable=None` is tqdm's own "draw only on a TTY" mode. Anything else warns and is treated as `auto`:

```
2026-07-31 09:14:02 | WARNING | src.utils.logging | unknown progress mode 'yes' -- using 'auto' (one of auto, on, off)
```

**Why the default is `auto`.** A ClearML agent's console is a redirected pipe, not a terminal, and a tqdm bar written to a pipe becomes thousands of lines of carriage-return spam in the task log. `auto` gives bars to a human at a terminal and nothing to the agent, with no per-environment configuration. `is_tty()` is evaluated per call and never cached at import, so a run that redirects its own stdout mid-flight still behaves correctly (`tests/utils/test_logging.py::test_is_tty_is_evaluated_lazily`).

**What goes wrong if it is set wrong.** `on` inside an agent produces an unreadable task log. `off` on a terminal costs you nothing but the feedback that a long conversion is still moving. Neither affects log volume: bars go to stderr through tqdm and are not `src` log records, so the `grep -cE '\| src\.'` count is unchanged either way.

**Note that the bar is not the summary.** Whether or not a bar is drawn, `progress()` logs one line per loop at DEBUG and never at INFO:

```
2026-07-31 09:14:11 | DEBUG | src.data.converter.coco2yolo | coco -> yolo labels: 87 items in 0.1s
```

It is DEBUG on purpose: every caller already emits its own INFO summary for the same unit of work, and at INFO the two lines appeared as a redundant pair per CVAT task. `tests/utils/test_logging.py::test_progress_stays_silent_at_info` fails if that ever regresses, and `::test_progress_emits_exactly_one_summary_line_at_debug` fails if the summary turns into one line per item.

## Caveat 1: these apply from `set_log_level()` onwards, not from process start

`config_clearml()` has to run before either value is known, and `config_clearml()` is itself preceded by module imports and `init_clearml()`. Everything logged before the two calls above follows `$LOG_LEVEL` (or INFO) no matter what the UI says. Concretely, that is:

- the import-time lines in `src/config.py` — `path_to_env %s exists=%s` and `environment variables loaded successfully`, both DEBUG;
- `init_ultralytics_settings()`'s `ultralytics clearml integration: %s`, DEBUG;
- `init_clearml()`'s `init clearml, Task.current_task=%s`, INFO;
- `main()`'s own first line, `ultralytics version: %s`, INFO.

So if you set `log_level: debug` in the UI to chase a problem in `src/config.py`, you will not get it. Use the `LOG_LEVEL` environment variable for anything that early: it is read straight from `os.environ` by design — `src/utils/logging.py` deliberately imports nothing from the project, because `src/config.py` raises at import time when any of its six required CVAT settings is missing, and reading the log level through it would make logging unusable in exactly the runs that most need it.

## Caveat 2: a value set in the UI only ever takes effect on the agent

`task.execute_remotely()` is called on the very next lines after the console setup, and it terminates the local process. The sequence in `main()` is: connect parameters → apply `0_Console` → `execute_remotely()` → everything else. That means:

- On the **local** launch that creates the task, the values applied are the ones in `src/params.py` (plus `$LOG_LEVEL`), because the UI has never been touched yet and the process dies immediately after.
- On the **agent**, the task's stored parameters are injected by ClearML before `main()` runs, so whatever you typed in the UI is what `config_clearml()` returns and what the console setup applies.

This is not a limitation so much as the point: the agent's console is the one that gets read after the fact, and it is the one the UI controls. To change verbosity for a local run, export `LOG_LEVEL=debug` before `make run`.

## Why these two keys live in `args_console` and not in `args_logging`

`config_clearml()` does `args_train.update(args_logging)` and `args_train.update(args_augment)`, and `args_train` is then splatted straight into `model_yolo.train(**args_train)`. Ultralytics validates its keyword arguments and rejects unknown ones outright, so a `log_level` key reaching that call breaks the run with:

```
SyntaxError: 'log_level' is not a valid YOLO argument
```

`args_console` is therefore a dict of its own, connected under its own group, and never merged into anything. Two tests guard it: `tests/utils/test_console_params.py::test_console_params_are_not_ultralytics_arguments` asserts the key sets do not intersect `args_train` or `args_logging`, and `::test_console_params_stay_out_of_the_dicts_merged_into_train` asserts neither key appears in the merged dict. If you add a third console knob, add it to `args_console`.

## What changes at DEBUG

Two things, and they are different in kind.

**Our own DEBUG lines become visible.** Raw response bodies, per-filter decisions, the resolved `data.yaml` class map, the export parameter dicts, the `progress()` summaries, and the `model_latest_id` diagnostics in `src/train.py`.

**Ultralytics' own per-item output is handed back.** `_apply_console_verbosity()` in `src/train.py` is the only place this happens:

```python
if not logger.isEnabledFor(logging.DEBUG):
    return
args_val["verbose"] = True
args_predict["model"]["verbose"] = True
```

`args_val["verbose"]` restores the per-class table that `models/yolo/detect/val.py` prints — one line per class, per validation pass. `args_predict["model"]["verbose"]` restores the one-line-per-image output of `ultralytics/engine/predictor.py`, which at the default `max_images: 40` is forty lines and is the single largest block of noise in a run. Both default to `False` in [`5_testing.md`](5_testing.md) and [`6_predict.md`](6_predict.md) because the same numbers are already in ClearML as a table and as image grids.

Note what is deliberately *not* used to achieve this: `YOLO_VERBOSE=0`. That environment variable is read once in `ultralytics/utils/__init__.py` and drops every ultralytics logger to ERROR, taking the per-epoch metrics table and the AMP and dataset warnings with it. The per-call `verbose` arguments are the right knob.

## The discipline these knobs sit on top of

`log_level` is a volume control on a codebase that is already meant to be quiet. The rule the pipeline is written to is: **no INFO line may be emitted from inside a loop over dataset items.** Per-item loops tally with `src.utils.logging.Tally` and the caller emits one summary; per-item detail is DEBUG. The consequence is that log volume does not grow with dataset size, and `tests/data/test_data_stage_smoke.py` asserts exactly that by running each stage at two input sizes and requiring the same line count.

`Tally` is what makes it practical: it counts occurrences per key and keeps the first few examples of each, and `summary()` renders `stalk 300, foreign_object 212`, biggest key first. A class missing from the class map costs one warning line at the end of the conversion regardless of how many instances it has.

The reference point for "is this run unusually noisy" is 334 lines of our own output on a 3-epoch, 4-CVAT-task run at INFO. `grep -cE '\| src\.' console.txt` measures it.

## Scenarios

### Scenario 1 — a remote run converted fewer images than expected

Ten thousand images went into CVAT, the dataset report says 9,400. At INFO the conversion is one line per source, which tells you the counts but not which images were dropped or why. Clone the task, set:

```
0_Console/log_level  = debug
0_Console/progress   = auto
```

and enqueue. On the agent, stdout is a pipe, so `auto` draws no bars — you get the DEBUG summaries instead. The console now shows, among much else:

```
2026-07-31 09:13:44 | INFO  | src.data.task_scope | cvat scope: train 7 task(s), test 1 task(s); 1 excluded from train as test: [1181]
2026-07-31 09:13:58 | INFO  | src.data.downloader.method.cvat | cvat task 741 "batch-03": ready after 4 polls / 12.6s, 118.4 MB
2026-07-31 09:14:07 | INFO  | src.data.class_map | class map (derived): 3 classes -> 0:car, 1:person, 2:speed_limit
2026-07-31 09:14:11 | DEBUG | src.data.converter.coco2yolo | coco -> yolo labels: 87 items in 0.1s
2026-07-31 09:14:11 | INFO  | src.data.converter.coco2yolo | batch-03: 2,431 images -> 2,388 copied, 43 without labels (labels on disk 2,388, match=True)
2026-07-31 09:14:29 | INFO  | src.data.setup | split: 9,400 pairs -> train 7,520 / valid 1,880
2026-07-31 09:14:29 | INFO  | src.yolov8.data | dataset ready: train 7,520 img / 7,520 lbl, valid 1,880 img / 1,880 lbl
```

The `43 without labels` per source is the answer, and the DEBUG level is what adds the per-decision lines that say *which* filter dropped them. Set `log_level` back to empty afterwards — DEBUG also turns on the ultralytics per-class and per-image output described above, and a 30-epoch run at DEBUG produces a task log nobody will read.

### Scenario 2 — running locally at a terminal, and from a cron job

Locally, leave the group alone and use the environment variable, because the local process dies at `execute_remotely()` before the UI could ever matter:

```bash
LOG_LEVEL=debug PYTHONPATH=. uv run src/train.py
```

stdout is a TTY, `progress` is `auto`, so the two data-stage loops draw bars *and* log their DEBUG summaries. The run starts with:

```
2026-07-31 09:13:40 | INFO  | src.train | ultralytics version: 8.3.xx
2026-07-31 09:13:41 | INFO  | src.utils.clearml_settings | init clearml, Task.current_task=None
2026-07-31 09:13:43 | INFO  | src.train | TASK_YOLO: segment
2026-07-31 09:13:43 | INFO  | src.train | [Downloading Data]
```

The first two of those lines are before `set_log_level()` and are showing at their env-var level; the `TASK_YOLO` line is after `execute_remotely()` and so only appears at all on a run that is not being shipped to an agent.

If the same command runs from cron with output redirected to a file, `auto` already suppresses the bars — there is nothing to configure. Set `progress: off` only if some intermediate tool makes `isatty()` lie, for example a pty-allocating CI runner.

### Scenario 3 — a value the parser cannot read

```
0_Console/log_level = verbose
0_Console/progress  = yes
```

Neither is valid, neither is fatal, and both say so exactly once:

```
2026-07-31 09:13:43 | WARNING | src.utils.logging | unreadable log level from explicit value: 'verbose' -- falling back to INFO
2026-07-31 09:13:43 | WARNING | src.utils.logging | unknown progress mode 'yes' -- using 'auto' (one of auto, on, off)
```

The run continues at INFO with TTY-only bars. The reason both warn rather than failing silently is that the alternative is debugging a run that quietly ignored the flag you set — which is strictly worse than a run that is slightly too quiet.

## Related groups

- [`1_task.md`](1_task.md) — model selection and where the ClearML task lives
- [`2_data.md`](2_data.md) — data sources, class order and filtering
- [`3_augment.md`](3_augment.md) — augmentation
- [`4_training.md`](4_training.md) — training hyperparameters
- [`5_testing.md`](5_testing.md) — validation, including `verbose` which DEBUG overrides
- [`6_predict.md`](6_predict.md) — post-training prediction, including `model.verbose` which DEBUG overrides
- [`7_export.md`](7_export.md) — export formats
- [`8_visualization.md`](8_visualization.md) — what gets reported to ClearML, as opposed to printed
