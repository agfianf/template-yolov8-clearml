# 8_Visualization

`args_visualization` in [`src/params.py`](../../src/params.py) is connected to the ClearML UI as the hyperparameter group **`8_Visualization`** (see `config_clearml()` in `src/utils/clearml_settings.py`). It is a set of switches over what the run *reports*, not over what it computes: every flag here gates a call in `src/yolov8/clearml_logger.py`, fired from a hook in `src/yolov8/callbacks.py`, and turning one off removes a plot or a scalar series from the task without changing a single training or validation number. The one exception is `confusion_matrix_conf`, which is a display threshold rather than a switch, and the near-exception is `log_worst_images`, which also decides whether ultralytics writes per-image diagnostic panels to disk during the final validation pass. Nothing here is expensive relative to training — the point of the group is to let you trade UI richness against upload volume and clutter when you are running many tasks rather than one.

## Quick reference

| Key | Default | Type | Meaning |
|---|---|---|---|
| `log_interactive_confusion_matrix` | `True` | bool | Plotly confusion matrix, `Normalized` and `Counts`, once at end of training |
| `log_per_class_table` | `True` | bool | Per-class P/R/F1/mAP table, once per validation pass |
| `log_interactive_pr_curves` | `True` | bool | PR, F1-conf, P-conf and R-conf curve families, box and mask |
| `log_confidence_histograms` | `True` | bool | Confidence histograms from the prediction stage, overall and per class |
| `log_learning_rate` | `True` | bool | Learning rate per optimizer param group, one point per epoch |
| `log_loss_components` | `True` | bool | box/cls/dfl/seg train losses and their ratios, one point per epoch |
| `log_speed_metrics` | `True` | bool | preprocess / inference / postprocess / total ms, one point per epoch |
| `log_per_class_scatter` | `True` | bool | Per-class mAP50-95 bar chart, sorted worst-first (name says scatter; it is a bar chart) |
| `confusion_matrix_conf` | `0.25` | float | Display-only confidence threshold for the confusion matrix, independent of `args_val["conf"]` |
| `log_static_plots` | `True` | bool | Re-upload ultralytics' own `results.png`, `labels.jpg`, `labels_correlogram.jpg` |
| `log_mask_box_gap` | `True` | bool | mask mAP minus box mAP, one point per epoch; segmentation runs only |
| `log_optimal_confidence` | `True` | bool | F1-optimal threshold, global scalars plus a per-class table |
| `log_worst_images` | `True` | bool | Worst-N image table, the GT/FP/TP/FN gallery, and `args_val["visualize"]` |
| `worst_images_limit` | `16` | int | How many rows and panels the worst-N reports contain |
| `log_calibration` | `True` | bool | TP-vs-FP confidence split, reliability diagram, and the ECE scalar |
| `html_report` | `True` | bool | Build and upload the interactive HTML evaluation report |
| `report_intended_use` | `""` | str | One line for the model card: what this model is for |
| `report_out_of_scope` | `""` | str | One line for the model card: what it must not be used for |
| `report_gallery_per_grid` | `24` | int | Items per gallery grid, fixed regardless of split size |
| `report_max_thumbnails` | `200` | int | Hard cap on unique 192px thumbnails in the whole report |
| `report_thumbnail_px` | `192` | int | Thumbnail edge in px; JPEG q80, 4:2:0 |
| `report_high_conf_fp_threshold` | `0.7` | float | Score above which a false positive joins the "missing annotation" grid |
| `report_low_support_threshold` | `30` | int | Below this instance count a class is dimmed as unreliable |
| `report_tide` | `True` | bool | Compute the ΔAP oracles; off falls back to error counts |
| `report_tide_max_images` | `4000` | int | Reservoir size for the error decomposition; bounds capture memory |
| `report_scan_labels` | `True` | bool | Read the on-disk label files for the split-aware dataset section |
| `report_cm_max_classes` | `60` | int | Classes kept in the report's confusion heatmap |
| `report_split_bytes` | `5_000_000` | int | Above this the galleries move into their own artifact |
| `report_max_bytes` | `15_000_000` | int | Hard ceiling; above it the galleries are dropped, never the report |

## Where each report lands in the ClearML UI

ClearML routes reports by the API call used, not by the name, so it is worth having the mapping in one place. `report_scalar` and `report_single_value` land in **Scalars**. `report_plotly`, `report_table`, `report_histogram`, `report_confusion_matrix` and `report_matplotlib_figure` all land in **Plots** — note that tables are plots as far as the UI is concerned. `report_image` lands in **Debug Samples**.

| Flag | ClearML title | Tab | Cadence | Cost |
|---|---|---|---|---|
| `log_learning_rate` | `Learning Rate` | Scalars | per epoch | negligible |
| `log_loss_components` | `Losses/Train`, `Losses/Balance` | Scalars | per epoch | negligible |
| `log_speed_metrics` | `Speed/Inference` | Scalars | per epoch | negligible |
| `log_mask_box_gap` | `Metrics/Mask-vs-Box` | Scalars | per epoch | negligible |
| `log_interactive_confusion_matrix` | `Confusion Matrix` / `Normalized`, `Counts` | Plots | once, end of training | one plot per series, size grows with class count squared |
| `log_per_class_table` | `Per-Class Metrics` / `Detailed` | Plots | once per validation pass | one table, one row per class |
| `log_per_class_scatter` | `Per-Class Performance` / `mAP50-95 (sorted)` | Plots | once per validation pass | one grouped bar chart |
| `log_interactive_pr_curves` | `Curves/<family>` / `Box`, `Mask` | Plots | once per validation pass | four plots on a detect run, eight on a segmentation run |
| `log_optimal_confidence` | `Operating Point` | Scalars **and** Plots | once per validation pass | two scalars plus one table |
| `log_worst_images` | `Error Analysis` | Plots **and** Debug Samples | once per validation pass | one table plus up to `worst_images_limit` image uploads |
| `log_calibration` | `Distributions` / `TP vs FP Confidence`, `Calibration` / `Reliability Diagram` | Plots and Scalars | once per validation pass | two plots plus one scalar; also a per-batch memory cost during validation |
| `log_confidence_histograms` | `Distributions`, `Distributions/Per-Class` | Plots | once, prediction stage | one plot plus **one per class** |
| `log_static_plots` | `results`, `Labels` | Plots | once, end of training | three PNG uploads |
| `html_report` | `evaluation_report` | **Artifacts** and Debug Samples (`Evaluation`/`report`) | once, after the final `val()` | one file, ~2-3 MB, of which 1.42 MB is the vendored plotly bundle |

Several reports are **not** gated by anything in this group and will keep appearing with every flag turned off: the `Metrics/*` and `Losses/Validation` scalars (`_report_metric_scalars`), `Speed/Training/epoch_time_seconds`, the model-info single values on epoch 0, the `Mosaic` and `Validation` debug-sample galleries, the `Prediction` 2×2 image grids from the prediction stage, the `Export`/`Formats` table from [`7_export.md`](7_export.md), and the dataset composition report from the data stage. This group is a volume control on the analysis layer, not a mute button for the task.

## Per-epoch cheap versus once-per-validation heavy

The code splits into two tiers, and the comments in `src/params.py` say so with a `--- per-epoch, cheap ---` / `--- once per validation run, heavier ---` divider.

**Cheap, per epoch.** `log_learning_rate` and `log_loss_components` fire from `on_train_epoch_end`; `log_speed_metrics` and `log_mask_box_gap` fire from `on_fit_epoch_end`. All four report only scalars. They are read straight off the trainer or validator with no extra computation, and a scalar point is a few bytes.

The split between those two hooks is not stylistic. `on_train_epoch_end` fires at `engine/trainer.py:569` and `validate()` does not run until `:577`, so anything read from `trainer.validator.metrics` in the earlier hook is the *previous* epoch's numbers — zeros on epoch 0, and epoch N showing epoch N−1 forever after. That is why validation metrics, speed and the mask/box gap all live in `on_fit_epoch_end` (`:605`) and only the train-side losses and LR live in `on_train_epoch_end`. `tests/yolov8/test_callbacks.py::TestEpochLagFix` asserts the routing in both directions.

**Heavy, once per validation pass.** Everything else runs through `report_validation_analysis()` in `src/yolov8/callbacks.py`, which is called from exactly two places: `on_train_end`, for the validation pass that closes training, and `src/train.py`, for the standalone `val()` that runs afterwards. It never runs per epoch. It covers the per-class table, the worst-first bar chart, the curve families, the operating point, the worst-image table and gallery, and the calibration pair.

Because it runs twice, most heavy plots exist twice in a finished task, distinguished by a title prefix: the end-of-training set has no prefix, and the standalone set is prefixed `Test ` when a test split exists and `Final ` when it does not. That prefix is what stops the second pass from overwriting the first. One report breaks the symmetry: the confusion matrix is reported from `on_train_end` directly and is **not** part of `report_validation_analysis()`, so there is no `Test Confusion Matrix`. If you are comparing the two passes, that absence is expected, not a failure.

## The cadence rule

**Report volume must not grow with epoch count.** This is the direct sibling of the project's logging rule ("no INFO line may be emitted from inside a loop over dataset items") and it exists for the same reason: a 300-epoch run must not upload 300 confusion matrices, 300 per-class tables and 4,800 error-gallery images.

Scalars are exempt, and deliberately so — a scalar *is* a series, and one point per epoch is the entire point of it. Plots, tables and image galleries are not series and must be reported once per validation pass instead.

`tests/yolov8/test_callbacks.py::TestReportVolume` enforces this the same way `tests/data/test_data_stage_smoke.py` enforces the logging rule: it runs the per-epoch hooks at 3 epochs and at 30 epochs, counts `report_plotly + report_table + report_confusion_matrix + report_matplotlib_figure` calls in both cases, and asserts the two counts are identical — and, since the per-epoch hooks emit nothing heavy at all, that both are zero. `test_scalars_do_grow_with_epochs` is the control: without it the assertion would pass trivially if reporting were broken entirely. A third test, `test_image_gallery_is_bounded_by_the_limit`, checks the other axis: given a directory of 50 panels and a 4-row worst-image table, exactly 4 images are uploaded — the gallery must not re-scan the directory and send everything it finds.

If you add a report, the rule tells you where to put it: anything that produces a plot, a table or images belongs inside `report_validation_analysis()`, and anything that produces a single number per epoch belongs in `on_fit_epoch_end`. Adding a plot to a per-epoch hook fails `TestReportVolume`.

## Flag by flag

### `log_interactive_confusion_matrix`

Reports the confusion matrix as an interactive ClearML matrix under `Confusion Matrix`, in two series: `Normalized` and `Counts`. Built by `extract_confusion_matrix_data()` in `src/yolov8/metrics_utils.py`.

Two details are load-bearing. Ultralytics stores the matrix as `matrix[predicted, ground_truth]` and normalizes it by **column** (`matrix.sum(0)`), so each column sums to 1 and the diagonal reads as per-class recall; normalizing by row instead gives precision-like numbers wearing the same title, which is the bug this replaced. And the matrix is **box-IoU based even on a segmentation run** — `SegmentationValidator` inherits `update_metrics`, so `process_batch` never sees masks. Do not read a segmentation task's confusion matrix as a statement about mask quality.

Cost grows with the square of the class count, since the payload is a full n×n matrix (plus a background row and column). At a few dozen classes it is fine; at several hundred it is worth turning off.

### `confusion_matrix_conf`

Not a switch — a threshold, and the most subtle parameter in the group. Ultralytics feeds a single `args.conf` to both NMS and the confusion matrix (`detect/val.py:195`), but the two want opposite values.

Metrics want a floor near zero. `args_val["conf"]` is passed straight into `non_max_suppression`, so anything below it is discarded *before the metrics ever see it* — a threshold of 0.25 truncates the high-recall tail of every PR curve and understates mAP. That is why `args_val["conf"]` is pinned at `0.001`, the ultralytics validation default; see [`5_testing.md`](5_testing.md) for the full argument and the measured impact.

A readable confusion matrix wants the opposite. At `conf=0.001` the matrix fills with low-confidence detections and a background column that swamps every real cell, and it stops being usable for the thing a confusion matrix is for — spotting which two classes get mixed up.

`on_val_batch_start` resolves this by wrapping the matrix's `process_batch` so it always receives `confusion_matrix_conf` regardless of what NMS used. The wrapper is idempotent (it sets a `_conf_pinned` marker) because the hook fires once per batch. Note that the matrix's IoU threshold is `0.45`, hard-coded upstream, and is **not** `args_val["iou"]`.

The consequence is that a finished task shows **three different confidence thresholds at once**, and they read as contradictions if you do not know which is which: mAP and the curves at 0.001, the confusion matrix at 0.25, and the P/R/F1 scalars that ultralytics prints at each class's own max-F1 confidence. The threshold you would actually deploy at is none of those three and is reported separately under `Operating Point`.

Raise `confusion_matrix_conf` if the matrix is still noisy; lower it if classes you know are being detected show up as all-background. Changing it changes only the matrix — mAP, the curves and the operating point are untouched.

### `log_per_class_table`

A `Per-Class Metrics` / `Detailed` table built from `metrics.summary()`. Using `summary()` rather than zipping a `names` list against `box.p` is not a convenience: **per-class arrays are indexed by `ap_class_index`**, which contains only the classes that had ground truth in this split, so a naive slice mislabels every row after an absent class. `summary()` resolves names via `names[ap_class_index[i]]` and is the only index-safe source.

On a segmentation run the table gains `Mask-P`/`Mask-R`/`Mask-F1` from `summary()`, plus `Mask-mAP50` and `Mask-mAP50-95` read from `seg.ap50` / `seg.ap` (per-class mask mAP is not in `summary()`), plus a `Mask-Box-mAP50-95-delta` column that isolates classes whose masks lag their boxes. Do not try to get per-class mask numbers out of `SegmentMetrics.maps` — that array is an element-wise *sum* of box and mask per class, not a concatenation, and any chart built from it is meaningless.

One row per class, once per validation pass. Cheap unless the class list is enormous.

### `log_per_class_scatter`

Despite the name, this gates a **bar chart**, not a scatter plot: `Per-Class Performance` / `mAP50-95 (sorted)`, drawn worst-first, with mask bars beside box bars on a segmentation run. The sort is the feature — it puts the classes you have to fix on the left, which class-index or alphabetical ordering buries. `YOLOClearMLLogger.log_per_class_scatter()` does exist as a method but has no callers; the flag name is a leftover from it.

Turning this off while leaving `log_per_class_table` on gives you the same numbers without the plot, which is a reasonable trade when the class list is long enough that the bar chart is unreadable anyway.

### `log_interactive_pr_curves`

Reports every curve family ultralytics computed, as Plotly plots under `Curves/<family>` with `Box` and `Mask` as selectable series of the same plot rather than as separate plots or, worse, overlaid without labels. On a detect run that is four plots (Precision-Recall, F1-Confidence, Precision-Confidence, Recall-Confidence); on a segmentation run, eight.

Two upstream facts to know before touching this. The PR-curve attribute is `prec_values`, **not** `py` — reading `py` returns nothing and the plot silently never appears, which is exactly what happened before `extract_curve_data()` was rewritten. And `metrics.curves_results` is the right source because it returns `[x, y, xlabel, ylabel]` per family with the axis labels included, paired positionally with `metrics.curves` for the `(B)`/`(M)` suffix.

Curves are downsampled to `MAX_CURVE_POINTS = 250` before reporting. Ultralytics computes 1,000 x-points per class per family; at 80 classes and eight families the raw payload runs to megabytes, and the ClearML UI renders an oversized plot as a **blank panel with no error**. 250 points is visually indistinguishable for a monotone-ish curve. If you ever see an empty curve panel, payload size is the first thing to suspect.

### `log_confidence_histograms`

The only flag in this group consumed outside `callbacks.py`: `_predicting_result()` in `src/train.py` reads it after the prediction stage. It reports one overall histogram (`Distributions` / `All Classes - Confidence`) and then **one histogram per class** under `Distributions/Per-Class`.

That per-class fan-out is the cost. It is bounded by class count, not by epoch count, so it does not violate the cadence rule — but on a 60-class dataset it is 61 plots, and it is measured over at most `args_predict["max_images"]` images (40 by default), which is a small enough sample that the per-class histograms are often too sparse to say anything. Turning this off is the cheapest single reduction in plot count for a many-class project. Note this is a *prediction-stage* distribution at `args_predict["model"]["conf"]` (0.25 by default), which is a different population from the validation-stage confidences that `log_calibration` reports — do not compare the two.

### `log_learning_rate`

One scalar per optimizer param group per epoch, under `Learning Rate`, read from `trainer.optimizer.param_groups`. Reported from `on_train_epoch_end`, where it is correct. Its value is mostly in confirming that warmup and the scheduler did what `args_train` said they would — with `optimizer: "auto"` the actual LR can differ from `lr0`, and this is where you see it.

### `log_loss_components`

Reports each train loss component (`box_loss`, `cls_loss`, `dfl_loss`, and `seg_loss` on a segmentation run) plus `total_loss` under `Losses/Train`, and additionally reports every non-box component as a ratio of box loss under `Losses/Balance`.

The ratios are the interesting half. Absolute loss values are not comparable between runs with different `box`/`cls`/`dfl` gains in [`4_training.md`](4_training.md), but their ratios are. A `seg_loss_over_box` that drifts upward says the mask branch is losing ground against the detector even while the total falls — which the total alone hides completely.

`extract_loss_components()` handles both the pre-8.4 shape (a tensor zipped against `loss_names`) and the current one (a dict), because iterating a dict yields its keys and the old zip silently paired names with names.

### `log_speed_metrics`

`preprocess_ms`, `inference_ms`, `postprocess_ms` and their total under `Speed/Inference`, per epoch, from `validator.speed`. Useful as a regression signal — a postprocess time that climbs across epochs usually means detection counts are climbing, which is worth knowing before you read it as a model improvement. Note this is validation-time speed on the training GPU and is not a deployment benchmark; use the `Export`/`Formats` table in [`7_export.md`](7_export.md) for artifact-level comparisons.

### `log_mask_box_gap`

Segmentation only; a no-op on a detect model, where `extract_mask_box_gap()` returns an empty dict and nothing is reported. Two scalars under `Metrics/Mask-vs-Box`: `mask_minus_box_mAP50-95` and `mask_minus_box_mAP50`, normally negative because masks usually lag boxes.

This is the clearest single per-epoch signal that the mask head rather than the detector is the limiting factor: a gap that *widens* over training means the boxes are being found but the masks fitted inside them are getting relatively worse.

One caveat that applies to any mask mAP number in the task: **mask mAP is computed at quarter resolution unless `save_json` or `save_txt` is set.** `segment/val.py:74` picks `process_mask` (prototype resolution, 160×160 at `imgsz=640`) versus `process_mask_native`. Both settings are internally consistent, but mask mAP from the two is **not comparable** — so if `args_val["save_json"]` changed between two runs, do not read the jump in this gap as a model change.

### `log_optimal_confidence`

Reports the F1-optimal confidence threshold — the number you would actually deploy at — as two scalars (`f1_optimal_confidence`, `f1_at_optimal_confidence`) and a `Per-Class Threshold` table, all under `Operating Point`.

It exists because none of the three thresholds already visible in the task is the one to ship: mAP is computed at 0.001, the confusion matrix at 0.25, and the P/R/F1 scalars ultralytics prints are each at that class's own max-F1 point. `extract_optimal_confidence()` reuses ultralytics' own `smooth(f1_curve.mean(0), 0.1).argmax()` so the global number matches what ultralytics itself prints rather than being a second, subtly different answer.

The per-class table is where the value is on an imbalanced dataset: a rare class often peaks at a much lower confidence than the global optimum, and shipping one threshold for everything quietly discards it. This report is also one of the things `args_val["conf"] = 0.001` buys you — at 0.25 the optimizer could not see, let alone recommend, any threshold below 0.25.

### `log_worst_images` and `worst_images_limit`

The error-analysis pair, and the flag with the widest blast radius: `src/train.py` reads it to set `args_val["visualize"]` for the final validation pass. That is the only place `visualize` is ever set, and it does not appear in the `5_Testing` UI group at all.

**Ranking.** `extract_worst_images()` reads `Metric.image_metrics`, a per-image dict of `{precision, recall, f1, tp, fp, fn}` at IoU 0.50, sorts F1 ascending, and breaks ties by FN then FP so the noisiest images float to the top. On a segmentation run it prefers `metrics.seg`, whose true positives come from `mask_iou`, so the ranking is genuinely mask-aware rather than a box ranking with a mask label. The table gets a `Rank` column, which is what ties it to the gallery.

**The `1.0` gotcha.** Ultralytics deliberately scores an image with no ground truth *and* no prediction as F1 = 1.0, a trivially correct call. Those sort to the bottom and never reach a worst-N list, which is what we want — but it means any *mean* over `image_metrics` is inflated by empty images. Do not compute one from this table.

**The gallery.** When `args_val["visualize"]` was on, ultralytics writes one GT/FP/TP/FN panel per validated image into `<save_dir>/visualizations`, uncapped. That is why it is enabled for the single final `val()` only and never during training — during training you get the worst-image table but no panels, because the directory does not exist. Only the worst `worst_images_limit` panels are uploaded, and `_log_error_gallery()` looks each one up by filename from the already-truncated table rather than scanning the directory. These panels are **box-IoU based even on a segmentation run**, which is why the series is named `worst-box-`.

**ClearML image retention.** Retention (`sdk.metrics.file_history_size`) is applied **per title/series**, so the series name is the rank and nothing else — `worst-box-00`, `worst-box-01` — with the epoch in `iteration`. Each rank slot then keeps its own history and can be scrubbed across iterations. Folding the filename into the series would mint a brand-new series every time a different image became the worst, and every slot would keep exactly one frame. This is the single most important implementation detail in this section; if you change the series naming, you silently destroy the history.

`worst_images_limit` at 16 is a compromise: enough to see a pattern, few enough that the Debug Samples tab stays navigable and the upload cost stays flat. Raising it raises upload volume linearly and is bounded by validation-set size, not epoch count, so it does not break the cadence rule.

### `log_calibration`

Two plots and a scalar: a TP-vs-FP confidence split under `Distributions` / `TP vs FP Confidence`, a reliability diagram under `Calibration` / `Reliability Diagram`, and `expected_calibration_error` as a scalar.

The split answers a question the undifferentiated confidence histogram cannot: is there *any* threshold that separates true from false detections? Two overlapping humps mean no threshold will fix precision, and the model — or the labels — need work rather than tuning.

**This flag also switches on a per-batch capture, which is why it is not purely cosmetic.** `DetectionValidator.get_stats()` calls `metrics.process()` and then `metrics.clear_stats()` on the very next line (`detect/val.py:277-278`), so by the time `on_val_end` or `on_train_end` runs, the raw per-detection data is gone. `on_val_batch_end` is the last hook that fires before that clear (`engine/validator.py:271` vs `:276`), and it is where `ValStatsAccumulator.update()` grabs the confidence and true-positive arrays. Turning `log_calibration` off skips that capture entirely — which is the correct way to save the memory it holds, since there is nothing to report without it. The accumulator is a module-level singleton, reset explicitly in `src/train.py` before the standalone `val()` so the two passes do not pool their statistics; callbacks only ever run in the main process, so the Python 3.14 forkserver re-import in dataloader workers cannot duplicate it.

The histograms are **binned in Python and shipped as bar charts**, not handed to `go.Histogram`. Plotly bins in the browser, which means the raw samples are serialised into the plot payload — a single TP-vs-FP histogram measured 1.0 MB on a real validation set, and the ClearML UI renders a plot that size as a blank panel with no error. Binning first takes it to a few KB.

Read the ECE with a caveat that is printed in the plot title: it is computed over *predictions only*, so it is blind to false negatives. A model that confidently finds one easy object per image and misses everything else scores beautifully here. Always read it next to recall.

### `log_static_plots`

Re-uploads three of ultralytics' own PNGs as ClearML plots at the end of training: `results.png`, `labels.jpg` and `labels_correlogram.jpg`. The curve PNGs and the two confusion-matrix PNGs used to be uploaded here as well and no longer are — each of them duplicated an interactive Plotly plot this pipeline reports itself, at the cost of a matplotlib render and an upload per file, and put a second non-interactive copy of the same information in the UI.

The failure mode is worth knowing because it produces no error anywhere. `report_matplotlib_figure` uploads a PNG and references it by URL; if the ClearML **fileserver** cannot serve that URL, the panel renders **blank** — no exception in the console, no warning in the task log, no red anything. If your Plots tab has empty panels titled `results` or `Labels` while the Plotly plots beside them render fine, the fileserver is the suspect, and turning this flag off is the correct response until it is fixed. The interactive plots survive that failure precisely because they carry their data inline rather than by URL.

## The HTML evaluation report

`html_report` builds one self-contained interactive HTML file at the end of the run and uploads it as the `evaluation_report` artifact, then reports its URL as media so it also appears under **Debug Samples** → `Evaluation` / `report`. It is the only report in this group that is not a ClearML plot: it is a whole page, with a sticky nav, sortable tables, lazily-rendered Plotly charts and five thumbnail galleries with a click-to-zoom overlay. Everything else here answers one question each; this answers "what happened in this run" in one place you can send to somebody.

It is built **once, from the final `val()` only** — the standalone pass `src/train.py` runs after training, on the test split when there is one. It never runs per epoch, and the training-curve appendix is the only part of it that reads anything epoch-shaped. `src/report/build.py` is called between `_report_final_scalars()` and `export_handler()`, inside its own `try/except`, so a report that fails costs the report and nothing else — the export and prediction stages after it are untouched.

**Open it in a new tab.** The Debug Samples panel shows the page in an iframe whose sandbox policy is undocumented and has tightened before, so the first element on the page is a banner linking to the report's own URL with `target="_blank"`. That link is `href=""`, which the browser resolves to the current document — no JavaScript, and the generator never has to know its own address, so the escape hatch works even when the iframe blocks scripts entirely.

**Nothing is fetched from the network.** The page inlines its CSS, its JavaScript, a vendored `plotly.js-cartesian-dist-min@3.7.0` bundle (1.42 MB, committed under `src/report/assets/` and re-fetchable with `make fetch-plotlyjs`, which verifies a pinned sha256) and every thumbnail as base64. `plotly.py`'s own `include_plotlyjs=True` is 4.85 MB and is banned. One CDN reference would turn half the page blank on any deployment without outbound access, with no error anywhere, so `tests/report/test_report_html.py` fails the build on any `src=`/`href=`/`url()` that is not an anchor, a `data:` URI or the deliberate outbound link to the ClearML task.

### What is in it

Twelve sections, in this order: header and KPI tiles; model card; dataset and split composition; the per-class table; the confusion matrix and the most-confused pairs; the operating-threshold panel; the TIDE-style error decomposition; size and shape strata; box against mask (segmentation runs only); five galleries; the training-curve appendix; and a caveats footer.

Three things about it are deliberate and worth knowing before you read a number off it.

**Every KPI tile prints the confidence and IoU it was computed at.** A finished task shows at least three coexisting confidence thresholds — the 0.001 NMS floor that mAP and every curve are computed at, the 0.25 display threshold the confusion matrix and every gallery are drawn at, and the F1-optimal threshold you would actually deploy at — and a metric with no stated basis is a rumour. The model card repeats the three with the run's actual numbers.

**`fitness` never gets a meter or a percentage on a segmentation run.** `SegmentMetrics.fitness` is `seg.fitness() + box.fitness()`, i.e. the sum of two mAP50-95 values and therefore 0–2. The tile carries `scale: null` in the JSON blob and the renderer draws a bare number with the range spelled out.

**The dataset section reads the label files on disk**, under `<dataset_dir>/{train,valid,test}/labels`, rather than the pre-split `DatasetStats` accumulated during conversion. That is what makes the split-composition table possible, and that table sorts by validation count ascending — so a class with no ground truth in the measured split, which is silently absent from mAP entirely, is the first row. Object size buckets come from **this dataset's own terciles**, printed in px, not COCO's 32²/96² absolutes, and the captions say so. Set `report_scan_labels = False` to skip the scan; the section then collapses to a banner.

### Caps, and why they are not tunable upward for free

Report volume must not grow with dataset size or with epoch count — the same rule as the rest of this group, applied to a file rather than to a plot count. Every distribution in the page is binned in Python before it is serialised, every gallery holds a fixed item count, and the training curves are downsampled to `MAX_CURVE_POINTS`. `tests/report/test_report_volume.py` builds the report at 50 images and at 3,000 and asserts the rendered file differs by under 5%; measured, it differs by about 0.2%.

`report_gallery_per_grid` (24) × five grids is the item count, and `report_max_thumbnails` (200) is the hard ceiling on unique 192px JPEGs after deduplication — two grids referencing the same crop of the same file share one base64 string. At roughly 9 KB each, 200 thumbnails is 1.8 MB, which is most of what this project actually controls in the file. Raising either raises the file size linearly. A second, larger thumbnail tier for the lightbox was measured at ~28 KB each and is deliberately absent: the lightbox enlarges the same bytes and its caption says "192 px thumbnail, enlarged" so nobody reads the softness as a model artefact.

`report_cm_max_classes` (60) truncates the confusion heatmap to the classes with the most ground truth, because that payload is O(n²); the caption states the truncation and the per-class table still carries every row up to its own 120-row cap.

Above `report_split_bytes` (5 MB) the galleries move into a second artifact, `evaluation_report_galleries`, uploaded **first** so the index can link to its absolute URL — relative links between artifacts break, because each artifact name is its own directory on the fileserver. Only the index is reported as media. Above `report_max_bytes` (15 MB), reachable only by raising the caps, the builder drops the galleries entirely and adds a caveat: a degraded report at the end of a GPU run beats no report.

### The per-image capture

The error decomposition, the size/shape strata and the galleries all need per-image detections and ground truth, which nothing in ultralytics exposes after the fact. `on_val_batch_start` installs a wrapper on `validator._process_batch` (`src/report/capture.py`) next to the confusion-matrix pinning — a different method for a different job. That call runs once per image, receives the full post-NMS predictions at the NMS floor plus `pbatch`'s `im_file`, returns the `(n, 10)` IoU sweep, and runs even with `plots=False`. Installing it on the *instance* shadows the class method, so `SegmentationValidator._process_batch` still runs intact and `tp_m` is visible.

Memory is bounded **before** accumulation, not trimmed afterwards: a seeded reservoir of at most `report_tide_max_images` (4,000) whole image records for the decomposition, five fixed-size heaps for the galleries, and fixed-size histograms for everything else — about 23 MB in total, flat in dataset size. Whole records rather than individual detections, because the ΔAP oracles re-match detections against ground truth and a half-sampled image would invent both false positives and misses the model never made. **No pixels are held during validation**; thumbnails are produced afterwards by re-reading at most the cap's worth of files from the dataset directory, which is still on disk (`cleanup_cache` deletes only `labels.cache`).

The wrapper cannot raise into validation. `note()` is wrapped, failures are tallied, and after 50 of them the wrapper uninstalls itself and hands `_process_batch` back — a capture failing on every image is worth nothing and its `try` is not free at 20,000 images. Nothing is logged from inside that loop; one summary line is emitted at report time.

`report_tide` chooses between two honest modes and never a silent third. On, the six error types (Cls / Loc / Both / Dupe / Bkg / Miss, at `t_f = 0.5` and `t_b = 0.1`) each get a ΔAP50 measured by an independent oracle that fixes only that error and re-runs ultralytics' own `compute_ap`, plus two ceilings for removing every false positive and every miss. Off — or with no capture — only the counts are shown and the section title says so. The counts table is rendered in both modes, so the section never changes shape between runs. Two caveats are printed in the section and are not optional reading: the six ΔAP values are **independent oracles and do not sum** to the total headroom, and the matching is **box IoU only even on a segmentation run**, because ultralytics computes and then discards the pairwise mask IoU matrix inside its segmentation `_process_batch`. The box-vs-mask section carries the mask story instead, using the quantised `tp_m` levels.

### When it is not there

Every input is allowed to be missing, and each absence becomes a card naming what is gone rather than an exception. No capture: no error decomposition, no strata, no galleries. `log_calibration = False`: no reliability diagram and no TP-vs-FP split. `plots = False`: no confusion matrix. No `results.csv`: no training appendix. Unreadable images: empty grids with a reason. A cleaned-up dataset directory: no split composition. Every one of them is also listed in the caveats footer, and `tests/report/test_report_degradation.py` asserts the artifact still uploads in all of them.

### Gotchas specific to this report

- **The header never reads `args_val`.** `src/train.py` overwrites `batch`, `split` and `visualize` *after* the ClearML connect, so the connected `5_Testing` group is not what the pass ran with. Every displayed validation setting comes from `validator.args`.
- **`src/report/` is baked into the image like the rest of `src/`.** Editing the report has no effect on an agent until `make bump PART=patch && make build`.
- **The whole package is new code on the validation hot path.** If a run behaves oddly during validation and you want it out of the way entirely, `html_report = False` skips both the report and the `_process_batch` wrapper.

## Scenarios

### A long hyperparameter sweep

You are launching thirty short tasks and will read only the headline scalars. Set `html_report = False` first — it is the single largest artifact in the group and it also switches off the per-image `_process_batch` capture that feeds it. Then set `log_worst_images = False` (this also stops ultralytics writing a panel per validated image to disk in the final pass, which is the largest single I/O cost here), `log_calibration = False` (which also skips the per-batch stats capture), `log_confidence_histograms = False` (which removes one plot per class), `log_static_plots = False`, and `log_interactive_pr_curves = False`. Keep every per-epoch scalar flag on — they cost effectively nothing and they are what you will actually compare across the sweep — and keep `log_per_class_table` on if any of your classes are rare, since a sweep that improves mAP by losing a rare class entirely looks like an improvement in the scalars alone. Then turn everything back on for the single full run you launch from the winning configuration.

Note what this does *not* do: `Metrics/*`, `Losses/Validation`, the mosaic and validation debug samples, the prediction grids and the dataset report are ungated and will still appear. If a sweep needs to be quieter than that, the lever is `args_predict["max_images"]` in [`6_predict.md`](6_predict.md) and `args_val["plots"]` in [`5_testing.md`](5_testing.md), not this group.

### Two classes are being confused and the matrix will not show it

mAP looks acceptable but you suspect two visually similar classes are being swapped. Keep `log_interactive_confusion_matrix` and `log_per_class_table` on, and treat `confusion_matrix_conf` as the knob. If the matrix is dominated by the background column and the real cells are all pale, the threshold is too low for this model's confidence distribution — raise it toward 0.4. If classes you know are detected show up as almost entirely background, it is too high — lower it toward 0.1. Cross-check against `Operating Point`: if the F1-optimal confidence for those two classes is far from `confusion_matrix_conf`, the matrix is describing a threshold nobody would deploy at, and moving `confusion_matrix_conf` to roughly the reported optimum makes it describe the model you would actually ship. Changing it costs one re-validation and affects no metric — see [`5_testing.md`](5_testing.md) for why the metric threshold must stay at 0.001 regardless.

### Half the Plots tab is blank

Blank panels have two distinct causes and this group distinguishes them. If the blank ones are `results` / `Labels` and the Plotly plots render, it is the fileserver failing to serve uploaded PNGs — set `log_static_plots = False`. If a *Plotly* panel is blank, it is payload size: the curve downsampling (`MAX_CURVE_POINTS`) and the pre-binned confidence histograms exist to prevent exactly this, so a blank Plotly panel means something new has been added that ships raw per-detection or per-x-point data. Check the class count first; the confusion matrix and the curve families are the two reports whose payloads grow with it.

## Deliberately not implemented

Three things a reader of this group might reasonably expect are absent, all because adding them would add a dependency. They are listed here so nobody spends an afternoon looking for a flag that does not exist.

- **AP small / medium / large, and AR@k.** Ultralytics has this in `coco_evaluate` (`detect/val.py:475`) but gates it on `save_json and (is_coco or is_lvis)` — never true for CVAT or S3 data — and it needs `faster-coco-eval`, which has no cp314 wheel and would be source-built inside the agent container. Without it there is no principled answer to "is this a small-object problem?"; the worst-image gallery is the qualitative substitute.
- ~~**TIDE's six-way error typology**~~ — **implemented**, in the HTML evaluation report rather than as ClearML plots, and without `tidecv`: `src/report/tide.py` types every false positive from the per-image capture and measures each type's ΔAP50 with an independent oracle over ultralytics' own `compute_ap`. It is box-IoU only and runs on a seeded subsample; both are stated in the section. See `report_tide` above.
- **Boundary IoU and boundary F-score.** Mask AP under-penalizes boundary error, so mask *sharpness* is genuinely unmeasured in this template. The mask-vs-box gap tells you the mask head is weak; it cannot tell you whether the weakness is in the edges.

## Gotchas

- **Editing the reporting code has no effect on a remote run until the image is rebuilt.** `callbacks.py`, `clearml_logger.py` and `metrics_utils.py` all live under `src/yolov8/`, which the container bakes in and `PYTHONPATH=/workspace` makes authoritative — while `src/train.py` runs from the git checkout. The two drift silently. Run `make bump PART=patch && make build` after changing any of them.
- **Flags are read at call time, from the module-level `args_visualization` dict.** `config_clearml()` mutates that dict in place with the UI's values, so the callbacks see whatever the UI said — but only for a task that has already been created. `clearml_project` / `clearml_task_name` aside, all of `8_Visualization` is editable in the UI and applies on the next run.
- **Every reporting call is wrapped in a `try`.** A failure logs a warning and returns `False`; it never aborts training. That is correct, but it means a missing plot is a warning in the console rather than a crash — grep the console before concluding the flag was off.
- **`_log_debug_samples` tolerates a missing `_batchN` suffix on purpose.** The upstream version guards the iteration lookup and then calls `.group()` unconditionally, so any file without that suffix raises *inside a callback*, which aborts training rather than just the report.

## Related groups

[`0_console.md`](0_console.md) · [`1_task.md`](1_task.md) · [`2_data.md`](2_data.md) · [`3_augment.md`](3_augment.md) · [`4_training.md`](4_training.md) · [`5_testing.md`](5_testing.md) · [`6_predict.md`](6_predict.md) · [`7_export.md`](7_export.md) · **8_visualization**

[`5_testing.md`](5_testing.md) is the essential companion: `args_val["conf"]`, `args_val["iou"]`, `args_val["max_det"]` and `args_val["save_json"]` decide what the numbers on these plots actually mean, and `confusion_matrix_conf` only makes sense read against `args_val["conf"]`. [`0_console.md`](0_console.md) controls the console-side equivalent of this group — `log_level` and `progress` — and at `LOG_LEVEL=DEBUG` ultralytics' own per-class validation output comes back alongside these reports.
