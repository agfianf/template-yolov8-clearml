# Configuration reference

Every parameter of this pipeline lives in `src/params.py` and is connected to ClearML under a numbered group, so the same key can be edited in two places: in the file before a run is created, and in the ClearML UI on an existing task. One file here per group, in the order the pipeline uses them.

| Group | Dict in `src/params.py` | Covers |
|---|---|---|
| [`0_Console`](0_console.md) | `args_console` | Log level and progress bars, and why a value set in the UI only ever affects the agent |
| [`1_Task`](1_task.md) | `args_task`, `args_logging` | Which model architecture, which ClearML project and task name, resuming from a registered model |
| [`2_Data`](2_data.md) | `args_data` | Which CVAT projects and tasks, how they are split, which classes are kept, and what decides class index order |
| [`3_Augment`](3_augment.md) | `args_augment` | Every augmentation, its unit, and which ones hurt on fixed-camera plate footage |
| [`4_Training`](4_training.md) | `args_train` | Schedule, optimizer, loss gains, throughput, and the container memory settings they depend on |
| [`5_Testing`](5_testing.md) | `args_val` | The final validation pass, and the three different confidence thresholds that coexist in one task |
| [`6_Predict`](6_predict.md) | `args_predict` | The prediction gallery uploaded at the end of a run |
| [`7_Export`](7_export.md) | `args_export` | Export formats and their parameters, and why a default run produces no artifacts |
| [`8_Visualization`](8_visualization.md) | `args_visualization` | Every report, where it lands in the ClearML UI, and what it costs |

## Reading these

Each file opens with a quick-reference table of every key with its default, then explains the keys in groups, then gives worked scenarios with the config snippet and the log lines the pipeline actually emits. Where a parameter is inert, overwritten by code after `Task.connect()`, or deprecated upstream, the file says so rather than describing what the name suggests it does — several keys in `5_Testing` and `7_Export` fall into that category.

## Two rules that cut across every group

**A value set in the ClearML UI only takes effect on the agent.** `task.execute_remotely()` ends the local process, so a locally launched run never reads what you typed in the UI. The exceptions are `clearml_project` and `clearml_task_name`, which are read before `Task.connect()` and therefore only apply to a locally launched first run.

**The training image shadows `src/`.** The container sets `PYTHONPATH=/workspace` and bakes a copy of `src/` in, so on an agent `src.params` resolves to the image's copy while `src/train.py` runs from the git checkout. Editing a default in `src/params.py` has no effect on a remote run until `make build`; editing the same value in the UI does, because connected parameters are stored on the task itself.
