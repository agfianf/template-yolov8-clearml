# `2_Data` — the data stage

Everything about *which* images and *which* labels the run trains on lives in `args_data` (`src/params.py`), connected to ClearML as the parameter group **`2_Data`** by `config_clearml()` (`src/utils/clearml_settings.py:74`). It decides the CVAT tasks that are downloaded, the train/valid/test split, which annotations are filtered out before a single YOLO label line is written, and — most consequentially — which integer each class name is written as. Everything downstream inherits those decisions: `nc` and `names` in `data.yaml`, the class enumeration registered on the ClearML model, the per-class metric tables, and whether the final validation pass runs on `val` or on a held-out `test` split. ClearML flattens the nested dicts, so a nested key appears in the UI with its parent as a prefix (`cvat/task_ids_train`, `params/train_ratio`); the scalar keys appear under their own name. Values edited in the UI take effect on the next execution of the task; a run launched locally reads `src/params.py` first and then lets the UI override it.

## Quick reference

| Key | Default | Type | Meaning |
|---|---|---|---|
| `cvat.task_ids_train` | `[741, 733, 731, 728]` | list of int (or UI text) | Individual CVAT task ids to train on |
| `cvat.task_ids_test` | `[730]` | list of int (or UI text) | Individual CVAT task ids to hold out as the test split |
| `cvat.project_ids_train` | `[]` | list of int (or UI text) | Every task in these CVAT projects, added to training |
| `cvat.project_ids_test` | `[]` | list of int (or UI text) | Every task in these CVAT projects, added to the test split |
| `label_studio.project_id_train` | `None` | — | **Not implemented.** Setting it only selects the Label Studio branch, which logs a warning and does nothing |
| `label_studio.project_id_test` | `None` | — | **Not implemented**, as above |
| `s3.s3_uri_dir_train` | `None` | — | **Not implemented.** Setting it only selects the S3 branch, which logs a warning and does nothing |
| `s3.s3_uri_dir_test` | `None` | — | **Not implemented**, as above |
| `params.train_ratio` | `0.8` | float | Fraction of the merged training images that go to `train/` |
| `params.val_ratio` | `0.2` | float | Fraction that goes to `valid/`. **Only read when `test_ratio` is set** — without a test split, `valid/` takes everything that is not training, and a `val_ratio` that disagrees is warned about |
| `params.test_ratio` | `None` | float or `None` | Fraction that goes to `test/`, or `None` for no test split. When set, the three ratios must sum to 1, and it is mutually exclusive with `task_ids_test` / `project_ids_test` |
| `class_exclude` | `"stalk, foreign_object"` | comma-separated string or list | Class names whose annotations are dropped, and which reserve no class index |
| `attributes_exclude` | `None` | `dict[str, str]` or `None` | Drop annotations whose CVAT attribute value matches. Every key is evaluated, OR'd together — see below |
| `area_segment_min` | `0` | number | Drop annotations whose COCO `area` is strictly below this. `0` is a no-op |
| `unify_class_order` | `True` | bool (or UI string) | Build one name → index map for every source in the run. `False` restores the legacy per-source `category_id - 1` |
| `class_names` | `""` | comma-separated string or list | Pinned class order. Empty = derive it as the sorted union of every source |
| `on_unknown_class` | `"error"` | `"error"` \| `"drop"` | What to do with a class the pinned list does not mention. Unreachable when `class_names` is empty |

Type tolerance is deliberate and is worth knowing about, because the ClearML UI hands values back as text. `src/yolov8/data.py` normalises three ways: `_as_int_list` accepts `[741, 733]`, `"741, 733"` and `"[741, 733]"` alike and drops a non-numeric entry with `ignoring non-numeric id '73a'` rather than failing at download time; `_as_name_list` accepts a list or `"a, b"`; `_as_bool` accepts `true/1/yes/on` and `false/0/no/off` (case-insensitive) because plain `bool("false")` is `True` and would have made switching `unify_class_order` off in the UI a silent no-op. `attributes_exclude` has **no** such normalisation — it is read straight through with `self.config.get("attributes_exclude", None)` — so it is realistically only settable by editing `src/params.py`.

---

## `cvat.*` — choosing sources

### What it does

Four keys, and they **add up** rather than replace each other. `resolve_task_scope()` (`src/data/task_scope.py:88`) turns them into exactly two lists of task ids:

```
project_ids_train ──► list_tasks_of_project() ──► [task, task, ...] ─┐
                                                                     ├─► dedupe ─► train_all ─┐
task_ids_train ──────────────────────────────────────────────────────┘                        │
                                                                                              ├─► train = train_all - test
project_ids_test ───► list_tasks_of_project() ──► [task, task, ...] ─┐                        │
                                                                     ├─► dedupe ─► test ──────┘
task_ids_test ───────────────────────────────────────────────────────┘
```

Three rules fall out of that diagram, and all three are covered by `tests/data/test_task_scope.py`:

1. **Additive.** `task_ids_train=[900]` plus `project_ids_train=[46]` gives `[900, 1070]` — the explicit ids first, then whatever the projects expanded to.
2. **Deduped.** A task named twice downloads once. Order is preserved, first occurrence wins.
3. **Test always wins.** `train = [id for id in train_all if id not in set(test)]`. However a task got into the training list — typed by hand or expanded out of a project — if it is also in the test list it is removed from training and named in the log. This is the entire reason the module exists: the batch you set aside for testing normally lives *inside* the training project, so expanding the project without this would train on it, and every metric reported on the test split would be measured against images the model had already seen.

### What the resolver does on its own

- **Empty tasks are skipped.** A task is dropped when CVAT reports `size == 0`, because it exports an empty archive and then fails inside the converter with a much less obvious message. A task whose response has **no** `size` field at all is *kept* — unknown is not zero, and a CVAT version that stops returning the field must not silently empty every project.
- **Unfinished tasks are counted, not dropped.** Anything whose `status` is not `completed` is included but named in the summary line, because a partially annotated image teaches the model that a real object is background.
- **Project order is stable.** `_list_tasks_paged()` (`src/data/downloader/method/cvat.py:51`) sorts the tasks of a project by ascending task id, and pages with an explicit `page`/`page_size=100` rather than following CVAT's `next` URL (CVAT builds that URL from its own configured hostname, which is not necessarily reachable). It gives up after 200 pages with `cvat project 53: stopped after 200 pages, list may be truncated`.

### What it logs

One INFO line, always, whatever the number of tasks — asserted by `test_scope_is_summarised_in_one_line`:

```
2026-07-31 09:14:02 | INFO | src.data.task_scope | cvat scope: train 7 task(s), test 1 task(s) (project 53: 8 task(s), 1 not yet 'completed'); 1 excluded from train as test: [1181]
```

Two warnings can precede it:

```
2026-07-31 09:14:02 | WARNING | src.data.task_scope | cvat project 99 (train) has no tasks
2026-07-31 09:14:02 | WARNING | src.data.task_scope | cvat project 46 (train): all 2 task(s) are empty
```

Then one line per task as it is downloaded, from `src.data.downloader.method.cvat`:

```
2026-07-31 09:14:29 | INFO | src.data.downloader.method.cvat | cvat task 741 "batch-12": ready after 3 polls / 6.1s, 184.42 MB
```

### Who decides download order, and what it does *not* decide

Download order is: every training task first, in the order the resolver produced (explicit `task_ids_train` in the order they were typed, then project tasks by ascending id), then every test task the same way. `DataHandler._handle_cvat` deliberately downloads **all** of it — train and test — before converting anything, because the class map has to see every source before the first label file is written, and because a broken test task then fails in the first minutes instead of after an hour of conversion.

**Download order does not decide class indices.** With `unify_class_order=True` (the default) the map is the *sorted* union of every source's category names, so reordering `task_ids_train` cannot renumber a class. `test_derived_order_ignores_source_order` asserts exactly that. The one case where order does matter is `unify_class_order=False`: there, `data.yaml` is named by whichever task converted last, which is a genuine reason the flag exists only for reproducing old runs.

### What goes wrong

| Mistake | Symptom |
|---|---|
| Every training task is also listed as a test task | `ValueError: no training tasks: task_ids_train and project_ids_train resolved to [1070], and all of it is in task_ids_test ([1070])` |
| Nothing configured at all | Same `ValueError`, with `resolved to nothing` |
| A project id that does not exist, or is empty | `cvat project 99 (train) has no tasks`, then the `ValueError` above if nothing else was configured |
| A typo'd task id | `ignoring non-numeric id 'l70'` from `_as_int_list`, and that task is silently not in the run — check the `cvat scope:` count |
| Two CVAT tasks in the same project share a name | The extract directory is `{CVAT_OUTPUT_DIR}/{project_name}/{task_name}` and is `rmtree`d before extraction, so the second download **overwrites** the first. Project expansion makes this more likely than hand-listing did. Keep task names unique |

---

## `label_studio.*` and `s3.*` — not implemented

State this plainly: neither branch does anything. `DataHandler.export()` (`src/yolov8/data.py:368`) is:

```python
if self.source_type == "s3":
    logger.warning("S3 source not implemented yet")
elif self.source_type == "cvat":
    self._handle_cvat()
elif self.source_type == "label_studio":
    logger.warning("Label Studio source not implemented yet")
```

It then returns `self.dataset_dir` regardless, and that directory was never created — so the run does not stop at the warning, it fails later and further away, when ultralytics is handed a `data.yaml` path that does not exist. The keys exist because `_check_source()` uses them, not because there is an implementation behind them.

### `_check_source` requires exactly one source

`_check_source()` (`src/yolov8/data.py:149`) walks `args_data`, skips the seven non-source keys (`params`, `class_exclude`, `attributes_exclude`, `area_segment_min`, `unify_class_order`, `class_names`, `on_unknown_class`) and anything that is not a dict, and collects the name of every remaining group that has at least one value which is not `None`, `""` or `[]`. Exactly one group must survive:

```
cvat: any of the four lists non-empty   ─┐
label_studio: either id set              ├─► set of source names ─► must have len() == 1
s3: either uri set                      ─┘
```

- **Zero sources** → `ValueError: source must be just 1`. Same message as for two, which is unhelpful but is what the code says.
- **Two sources** → the same error. So you cannot leave an old `s3_uri_dir_train` filled in "for reference" while training from CVAT.
- The `isinstance(d, dict)` guard is what keeps a scalar setting from crashing this walk; `test_settings_are_not_mistaken_for_data_sources` covers it.

The resolved source name is also used as a ClearML tag — `task.add_tags(handler.source_type.upper())` in `src/train.py:51` — so a run is tagged `CVAT`.

---

## `params.*` — the train / valid / test split

### What it does

`split_folder_yolo()` (`src/data/setup.py:120`) collects every image that has a matching `.txt` label, shuffles with `random.seed(42)` (fixed, so the split is reproducible across runs on the same file set), and partitions:

```python
num_train = int(total_files * train_ratio)
if test_ratio:
    num_valid = int(total_files * valid_ratio)
    num_test  = total_files - num_train - num_valid   # remainder lands here
else:
    num_valid = total_files - num_train               # remainder lands here
    num_test  = 0

ls_train = ls_images_labels[:num_train]
ls_valid = ls_images_labels[num_train : num_train + num_valid]
if test_ratio:
    ls_test  = ls_images_labels[num_train + num_valid :]
```

**The sizes are a partition: every pair lands in exactly one split, and the three counts sum to the input.** `tests/data/test_split_partition.py` asserts that at several sizes and ratios. This matters because the splits are built by *moving* files and the staging `images/` and `labels/` directories are `rmtree`d afterwards — anything that falls outside every slice is not left lying around, it is deleted.

### Two things about this that surprise people

**1. `val_ratio` is ignored unless `test_ratio` is set.** With the defaults (`test_ratio = None`), `valid/` takes everything that is not training — `val_ratio` is not read as a size at all. Setting `train_ratio=0.8, val_ratio=0.1` and leaving `test_ratio` empty still gives you a 20% valid split, and now logs a `WARNING` saying so:

```
WARNING | src.data.setup | valid_ratio=0.1 ignored: without a test split valid/ takes the remaining 0.20 of the pairs. Set test_ratio to size valid/ explicitly.
```

This is deliberate — "the rest goes to valid" is what a two-way split means — but it is worth knowing that the number you typed was not used.

**2. With `test_ratio` set, the three ratios must sum to 1.** `train_ratio=0.8, val_ratio=0.2, test_ratio=0.1` sums to 1.1 and is rejected before any file moves:

```
[Splitting] train_ratio=0.8, valid_ratio=0.2 and test_ratio=0.1 sum to 1.1, expected 1.0. Each ratio is the fraction of the dataset that goes to that split.
```

What you meant is `train_ratio=0.7, val_ratio=0.2, test_ratio=0.1` → 700 / 200 / 100. Note that `test_ratio=0.0` is falsy and therefore means "no test split", not "an empty test split".

Both of these used to be silent. Until [#19](https://github.com/agfianf/template-yolov8-clearml/issues/19), `test_ratio` was a truthiness *switch* whose numeric value was never read — the test split was whatever was left over — so `0.8/0.1/0.9` and `0.8/0.1/0.1` produced an identical 80/10/10 split, and `0.8/0.2/0.1` produced an empty `test/` that was quietly dropped from `data.yaml`. The splits were also three independent `int()` slices rather than a partition, so `int(n * (1 - 0.8))` — `1 - 0.8` is `0.19999999999999996` in IEEE 754 — discarded the tail pair of every dataset whose size ended in a `0`.

### A ratio test split is not the same thing as `task_ids_test`

Two different mechanisms produce a `test/` directory, and they are not interchangeable:

| | `params.test_ratio` | `cvat.task_ids_test` / `project_ids_test` |
|---|---|---|
| Where the images come from | A random slice of the same pool the training images came from | Separate CVAT tasks, downloaded into `dataset-yolov8-test` and moved in wholesale |
| Same scenes/sessions as training? | Yes — same tasks, same day, same camera. Optimistic | No, if the tasks were chosen to be different |
| Removed from training? | Yes, by the slicing | Yes, by `resolve_task_scope` subtracting test from train |
| Converted through the same class map? | Trivially | Yes — `_build_class_map(train_dirs + test_dirs)` sees both |

**They are mutually exclusive, and `setup_dataset()` refuses the combination** before the split runs:

```
[Splitting] test_ratio=0.1 and a dedicated test set (dataset-yolov8-test) are mutually exclusive -- configure one or the other. Carve the test split out of the training images with test_ratio, or point task_ids_test/project_ids_test at a test task and leave test_ratio empty.
```

The refusal is issue [#19](https://github.com/agfianf/template-yolov8-clearml/issues/19). Both mechanisms produce a `test/` directory in the same place, and the dedicated one used to win by `rmtree`ing it: the ratio held images back from training, the dedicated set then overwrote the directory holding them, and those images ended up in no split at all. With `0.8 / 0.1 / 0.1` that was 10% of every image downloaded from the training tasks, deleted with no warning — the only trace was that the `split:` and `dataset ready:` log lines reported different test counts. Neither answer could be picked automatically, because either one silently discards what the other configuration meant. Pick one yourself.

Either way, once `data.yaml` has a `test:` entry, `src/train.py:317` sets `args_val["split"] = "test"` for the final validation pass, and every final report is titled `Test ...` instead of `Final ...`.

### What it logs

```
2026-07-31 09:31:44 | INFO | src.data.setup | split: 10,890 pairs -> train 8,712 / valid 2,177
2026-07-31 09:31:52 | INFO | src.data.setup | data.yaml: 4 classes, splits train/valid/test
2026-07-31 09:31:52 | INFO | src.yolov8.data | dataset ready: train 8,712 img / 8,712 lbl, valid 2,177 img / 2,177 lbl, test 1,153 img / 1,153 lbl
```

### What goes wrong

| Mistake | Symptom |
|---|---|
| `train_ratio` so high the valid split rounds to 0 | `[Creating YAML] valid folder is not valid. images=0 labels=0 ...`, preceded by an ERROR line naming the counts and a few filenames |
| Ratios that do not sum to 1 with `test_ratio` set | `[Splitting] ... sum to 1.1, expected 1.0`, raised before any file is moved |
| `test_ratio` so small it rounds to 0 pairs | `[Splitting] test_ratio=... leaves no images for test/`, naming how many of the pairs `train_ratio` and `valid_ratio` already claim |
| `test_ratio` **and** `task_ids_test` both set | `[Splitting] ... are mutually exclusive`, raised before the split runs |
| A dedicated test task whose images are not annotated yet | `WARNING ... split "test" invalid (images=0 labels=0): omitted from data.yaml, the run will not evaluate on a test set`. Not fatal — training proceeds and the final `val()` falls back to the valid split |
| No image/label pairs at all | `[Splitting] Error. No images found in the source directory!` |

---

## `class_exclude` — dropping a class entirely

### What it does

`class_exclude` is an **annotation-level** filter applied in `Coco.get_imageid_to_annotations()` (`src/schema/coco.py:109`), and simultaneously a **class-map** filter applied in `build_class_map()` (`src/data/class_map.py:126`). Both matter:

- Annotations of an excluded class never reach the converter, so no label line is written for them.
- The name is removed from the class map, so it **reserves no class index**. `test_excluded_class_reserves_no_index` and `test_excluded_class_leaves_no_gap` pin this: excluding `speed_limit` from `["car", "speed_limit"]` gives `nc: 1` and `names: ['car']`, not `nc: 2` with a dead slot.

Before and after, on the plate dataset:

```
without class_exclude          class_exclude = "speed_limit"
  0: back                        0: back
  1: front                       1: front
  2: plate                       2: plate
  3: speed_limit                 (gone — and `plate` is still 2, `back` still 0)
  nc: 4                          nc: 3
```

Excluding a name that sorts *before* a kept one does shift the kept ones, exactly as adding a class would: excluding `back` renumbers `front` to 0.

An image left with no annotations at all keeps an **empty** label file and trains as a background image. That is a deliberate choice (`test_dropping_every_class_leaves_background_images`), not an oversight, but it is a choice — excluding your only class turns the whole dataset into negatives.

### Valid values

A comma-separated string (`"stalk, foreign_object"`, the default) or a list. `config_clearml()` splits the string, and `_as_name_list` splits it again and drops empties, so `""` and `None` are both safely "exclude nothing".

### Why the default is what it is

`"stalk, foreign_object"` is left over from a different dataset. On the plate projects neither name exists, so it is a no-op — the filter only ever matches names that are actually present. It is harmless but it is not a considered default; treat it as an example.

### What goes wrong

**Case sensitivity is inconsistent between the two filters, and the mismatch fails the run.** The annotation filter lowercases the *configured* names but compares them against the *raw* category name:

```python
exclude_class = [lbl.lower() for lbl in exclude_class]      # src/schema/coco.py:125
...
if id2label[ann.category_id] in exclude_class:              # src/schema/coco.py:168
```

The class map, by contrast, matches through `_key()`, which lowercases both sides. So with a CVAT label spelled `Stalk` and `class_exclude = "stalk"`:

- the class map drops `Stalk` — no index for it;
- the annotation filter does **not** drop its annotations;
- the converter then looks up `Stalk`, finds nothing, and raises `UnknownClassError: .../instances_default.json: class 'Stalk' is not in the class map (0:back, 1:front, 2:plate). Add it to class_names, add it to class_exclude, or set on_unknown_class=drop.`

The message is actively misleading here, since `Stalk` *is* in `class_exclude`. Spell `class_exclude` entries exactly as CVAT spells them.

Excluding every class is caught up front: `ValueError: class map is empty: no categories found in 3 annotation file(s) after excluding ['stalk']`.

---

## `attributes_exclude` — dropping annotations by CVAT attribute

### What it actually does

```python
if attributes_excluded:
    for attr_name, attr_value in attributes_excluded.items():
        raw = ann.attributes.get(attr_name)
        if raw is None:
            continue

        report.attr_keys_seen.add(attr_name)
        intersection = _attribute_values(attr_value) & _attribute_values(raw)
        if intersection:
            ...
            skip = True
            break  # OR semantics: one matching rule is enough
```

Behaviour, item by item:

- **The semantics.** `{"maturity_truth": "background"}` means: read the annotation's `maturity_truth` attribute, split both it and the configured value on commas, and drop the annotation if the two sets intersect. Multi-value attributes work — `{"tags": "blurred, occluded"}` drops anything tagged either way.
- **Every key is evaluated, and they are OR'd.** `{"maturity": "background", "quality": "bad"}` drops an annotation that matches *either* rule. Each entry stands alone as one exclusion, so adding a rule tightens the filter rather than loosening it, and the order the keys are written in cannot change the result. This is worth stating because OR and AND are indistinguishable on a single-key config, which is what a first attempt always uses.
- **Comparison folds case and surrounding space**, on whole comma-separated tokens. `"Background"`, `"BACKGROUND"` and `" background "` all match `"background"`; `"back"` does not. This matches how `class_exclude` compares names.
- **An annotation that does not carry the attribute is not a match.** CVAT writes an attribute only onto annotations of the labels that declare it, so in any project with more than one label most annotations are missing most attributes. Such an annotation is left for the remaining rules to judge, and kept if none of them match.
- **Non-string attribute values are coerced with `str()`** before comparison, because `Annotation.attributes` is an untyped `dict` and a checkbox attribute arrives as a JSON boolean. Write the config value as text: `{"is_blurry": "true"}`.
- **An empty configured value matches nothing.** `{"maturity_truth": ""}` drops no annotations rather than matching every annotation with an empty attribute.
- **The filter runs after the area filter and before the class filter**, and a match sets `skip = True` rather than `continue`, so the class filter still evaluates. That only affects which reason is recorded, not the outcome.

All of the above is asserted by `tests/data/test_attributes_exclude.py`. Until [issue #20](https://github.com/agfianf/template-yolov8-clearml/issues/20) was fixed, only the first key was evaluated and a missing attribute raised `AttributeError`; a config written before that fix may have been silently filtering on one rule out of several.

### Valid values

`None` (default) or a `dict[str, str]`. There is no string parsing anywhere in the path — `DataHandler` stores it verbatim, and `config_clearml()` only does `args_data.get("attributes_exclude", {})`, which is a no-op — so a value typed into the ClearML UI as text will arrive as a string and fail on `.items()`. Set it by editing `src/params.py`, where the intended shape is already there as a comment:

```python
# "attributes_exclude": {"maturity_truth": "background"},
"attributes_exclude": None,
```

### What it logs

Nothing on its own at INFO. The count folds into the per-source annotation summary, with the *attribute name* as the tally key (not the value):

```
2026-07-31 09:20:11 | INFO | src.schema.coco | annotations: 12,430 total -> 11,908 kept, 522 dropped (attr maturity_truth 522)
```

At `LOG_LEVEL=debug` you additionally get one line per annotation: `ann 88431 dropped: attribute maturity_truth in ['background']`.

One WARNING is possible, emitted once at the end of the data stage rather than per source:

```
2026-07-31 09:20:44 | WARNING | src.yolov8.dataset_report | attributes_exclude: 'occluded' not carried by any annotation in any source, so nothing was dropped for it -- check the attribute name
```

It fires only when *no* source in the run carried the attribute, which in practice means the name is misspelt. A rule aimed at one CVAT task and absent from another is normal and stays silent. This exists because treating a missing attribute as "no match" is what makes a typo silent — before the fix a bad name announced itself by crashing.

---

## `area_segment_min` — dropping tiny annotations

### What it does

One comparison, at `src/schema/coco.py:133`:

```python
if area_segment_min is not None and ann.area < area_segment_min:
```

- The field compared is **`ann.area`** — the COCO `area` field as CVAT exported it, not a recomputed box area and not a fraction of the image. For a polygon annotation CVAT writes the polygon's area; for a box, the box's area. Units are **square pixels at the original image resolution**, so the same threshold means different things on 4K footage and on 640×480 footage, and it is applied *before* any resize to `imgsz`.
- The comparison is strict `<`, so `area_segment_min = 100` keeps an annotation of exactly 100 px².
- It runs **first**, before the attribute and class filters, and uses `continue` — so an annotation dropped for area is counted only under `area<...` and never appears in the per-class drop tally, even if it was also an excluded class.

### Why the default is what it is

`0` is a no-op: `ann.area < 0` is never true for a real annotation. The key is on by presence, not by value — leaving it at `0` means "no area filter", which is the right default because the useful threshold depends entirely on image resolution and on what the model is expected to detect.

### What goes wrong

Set it too high and you do not get an error, you get a quieter dataset: small objects become unlabelled background, the model learns to suppress them, and recall on small instances drops with nothing in the metrics naming the cause. The one place it is visible is the drop count:

```
2026-07-31 09:20:11 | INFO | src.schema.coco | annotations: 12,430 total -> 9,004 kept, 3,426 dropped (area<400 = 3,426)
```

Check that against the box-area distribution in the Dataset report (`src/yolov8/dataset_report.py`) before committing to a value.

---

## `unify_class_order`, `class_names`, `on_unknown_class` — the class order

This is the most important group in the file, because getting it wrong produces no error at all — just a model trained against scrambled targets and metric charts that look merely disappointing.

### The problem

A YOLO label file stores a class **index**. COCO stores a category **id**. CVAT hands out category ids per project, in the order the labels were created in that project's label schema — and a task that never saw a label simply omits it. The converter merges every task into one directory. So the legacy `category_id - 1` writes the same integer for different labels in different tasks.

This project's own CVAT projects label the same four classes in four different orders:

| CVAT project(s) | Category order as exported |
|---|---|
| 60 | `plate, front, back, speed_limit` |
| 59, 58, 57, 54, 53, 49, 43 | `front, back, plate, speed_limit` |
| 48 | `front, plate, speed_limit, back` |
| 46 | `back, plate, front` *(no `speed_limit` at all)* |

Under the legacy `category_id - 1`, merged into one dataset directory:

| Label | proj 60 | projs 59/58/57/54/53/49/43 | proj 48 | proj 46 |
|---|---|---|---|---|
| `plate` | **0** | **2** | **1** | **1** |
| `front` | **1** | **0** | **0** | **2** |
| `back` | **2** | **1** | **3** | **0** |
| `speed_limit` | **3** | **3** | **2** | — |

Index `0` means `plate` in one set of label files, `front` in another and `back` in a third — in the *same* `train/labels` directory. And `data.yaml` gets whichever names the last-converted task happened to use, so the legend is wrong for everything else. Nothing errors.

With `unify_class_order = True`, one map is built from the union of all sources before any label file is written, and every task is converted by **name**:

| Label | derived index, every project |
|---|---|
| `back` | **0** |
| `front` | **1** |
| `plate` | **2** |
| `speed_limit` | **3** |

```
2026-07-31 09:18:03 | INFO | src.data.class_map | class map (derived): 4 classes -> 0:back, 1:front, 2:plate, 3:speed_limit
```

Project 46, which has no `speed_limit`, is not a problem: it simply contributes no annotations at index 3, and index 3 still exists because some other source has it.

### `unify_class_order`

`True` (default) builds the shared map. `False` restores per-source `category_id - 1` and exists **only** to reproduce a run made before the fix; `test_flag_off_restores_the_legacy_numbering` asserts the old behaviour so that changing it has to be deliberate. With the flag off you get one warning naming up to three of the disagreeing sources:

```
2026-07-31 09:18:03 | WARNING | src.data.class_map | unify_class_order is off and 4 source(s) disagree on category order, so the same class index means different things in different tasks: /tmp/cvat/Plate/batch-12/annotations/instances_default.json: plate, front, back, speed_limit | ...
```

Accepts the ClearML UI's strings (`"false"`, `"0"`, `"off"`, …); anything unrecognised keeps the default and warns `unify_class_order='maybe' is not a boolean, using True`.

### `class_names` — derived versus pinned

**Empty (default) = derived.** The map is the union of every source's category names, **sorted**, with matching done on `name.strip().lower()`. Sorting is the whole point: the order must not depend on which task id was typed first, which one CVAT returned first, or which one is missing a label — otherwise adding a task to the config would renumber the classes.

Where several sources spell the same label differently, one spelling has to reach `data.yaml`. Taking whichever appeared first would reintroduce the order dependence, so **most common wins, ties broken alphabetically** (`_preferred_spelling`), and the disagreement is named:

```
2026-07-31 09:18:03 | WARNING | src.data.class_map | label spelled 2 ways across sources: speed_limit x7, Speed_Limit x2 -- CVAT treats these as different labels, this pipeline does not
```

**A derived order is reproducible across runs, but not across a change to the class list.** A new label sorts into the middle:

| | before | after adding `arrow` |
|---|---|---|
| 0 | `back` | `arrow` |
| 1 | `front` | `back` |
| 2 | `plate` | `front` |
| 3 | `speed_limit` | `plate` |
| 4 | — | `speed_limit` |

Every index moved. If you fine-tune a checkpoint trained on the "before" map against a dataset that produces the "after" map, the model's head is now pointed at the wrong classes, and nothing in the run says so.

**Non-empty = pinned.** The listed order is used verbatim, whitespace-stripped, deduped (`class_names lists 'plate' more than once, keeping the first`), minus anything in `class_exclude`. A pinned name that **no source has keeps its slot** — that is precisely what makes pinning survive a task that is missing a class, and it is what you want when the checkpoint you are fine-tuning knows a class your current data does not (`test_pinned_order_survives_a_missing_class`). Match it to the checkpoint's own `names`, in the checkpoint's own order.

Pinning also produces a build-time warning for anything a source has that the pinned list does not, once, naming the label and up to three example files:

```
2026-07-31 09:18:03 | WARNING | src.data.class_map | not in the pinned class_names, and not excluded: speed_limit 9
```

### `on_unknown_class`

Only reachable with a pinned `class_names` — a derived map is the union of every source and cannot be missing one of their classes by construction.

- `"error"` (default): the converter raises on the first annotation of an unlisted class.

  ```
  UnknownClassError: /tmp/cvat/Plate/batch-12/annotations/instances_default.json: class 'speed_limit' is not in the class map (0:back, 1:front, 2:plate). Add it to class_names, add it to class_exclude, or set on_unknown_class=drop.
  ```

- `"drop"`: the annotation is skipped, tallied, and reported once per label at the end of the source — one warning however many thousand instances there were:

  ```
  2026-07-31 09:21:40 | WARNING | src.data.converter.coco2yolo | batch-12: 1,204 annotation(s) dropped, class not in the class map: speed_limit 1,204
  ```

  An image emptied by this keeps an empty label file and trains as a **background image**. That is a real choice, and it is why the count is a warning rather than silence.

- Anything else is rejected in `DataHandler.__init__`, before a single byte is downloaded: `ValueError: on_unknown_class must be 'error' or 'drop', got 'ignore'`. The value is stripped and lowercased first, so `"Error"` is accepted.

---

## Scenarios

### (a) Train on whole projects, hold one task out for test

The common case, and the reason `project_ids_*` exists: a task added in CVAT next week is picked up automatically instead of being silently missing because nobody updated the id list.

```python
"cvat": {
    "task_ids_train": [],
    "task_ids_test": [1181],       # one task, which lives inside project 53
    "project_ids_train": [53],     # 8 tasks
    "project_ids_test": [],
},
```

Resolved:

```
project 53 -> [1169, 1174, 1177, 1181, 1183, 1186, 1190, 1195]
test       =  [1181]
train      =  [1169, 1174, 1177, 1183, 1186, 1190, 1195]        (7, not 8)
```

```
2026-07-31 09:14:02 | INFO | src.data.task_scope | cvat scope: train 7 task(s), test 1 task(s) (project 53: 8 task(s), 2 not yet 'completed'); 1 excluded from train as test: [1181]
```

The 7 training tasks are then ratio-split 80/20 into `train/` and `valid/`, task 1181 becomes `test/` wholesale, and `data.yaml` gains a `test:` entry — which makes `src/train.py` run the final `val()` on `split="test"` and title every final report `Test ...`.

### (b) Mixing project ids and task ids, with an overlap

```python
"cvat": {
    "task_ids_train": [1070, 1070, 900],   # 1070 is also in project 46
    "task_ids_test": [1069],
    "project_ids_train": [46],             # tasks 1069 (empty) and 1070
    "project_ids_test": [],
},
```

Step by step:

```
explicit train : [1070, 1070, 900]
project 46     : [1069 (size 0 -> skipped), 1070]
train_all      : dedupe([1070, 1070, 900] + [1070]) = [1070, 900]
test           : [1069]
train          : [1070, 900]          # 1069 was never in train_all, so nothing is excluded
```

Note what the three mechanisms each did: the duplicate `1070` downloads once, the project contributed nothing new because its only usable task was already listed, and the empty task 1069 was skipped by the size check — so listing it as a test task produces a `test` list containing a task that will export an empty archive. The log:

```
2026-07-31 09:14:02 | INFO | src.data.task_scope | cvat scope: train 2 task(s), test 1 task(s) (project 46: 1 task(s), 1 empty skipped)
```

The `1 empty skipped` note only covers project *expansion*. `task_ids_test` is not size-checked, so an explicitly listed empty task goes through to the downloader. Prefer `project_ids_test` if you want the same protection on the test side.

### (c) Fine-tuning on top of an existing checkpoint

The checkpoint was trained when the dataset had no `arrow` class, so its head is ordered `back, front, plate, speed_limit`. The new data adds `arrow`, which a derived map would sort to index 0 and shift everything else by one. Pin instead:

```python
"class_names": "back, front, plate, speed_limit, arrow",
"unify_class_order": True,
"on_unknown_class": "error",
```

```
2026-07-31 09:18:03 | INFO | src.data.class_map | class map (pinned): 5 classes -> 0:back, 1:front, 2:plate, 3:speed_limit, 4:arrow
```

The four original classes keep the indices the checkpoint knows, and `arrow` is appended at 4 where the fine-tune can learn it. Leave `on_unknown_class` at `error` here: the run *should* stop if the data contains a fifth label you forgot, because silently dropping it would train those objects as background.

Set `args_task["model_latest_id"]` in [`1_task.md`](1_task.md) to actually load the checkpoint; this key only controls the label indices it is fine-tuned against.

### (d) Excluding a class

`speed_limit` is annotated inconsistently across the older projects and is hurting more than it helps.

```python
"class_exclude": "speed_limit",
```

Before and after, with no other change:

```
class map (derived): 4 classes -> 0:back, 1:front, 2:plate, 3:speed_limit
class map (derived): 3 classes -> 0:back, 1:front, 2:plate
```

```
2026-07-31 09:20:11 | INFO | src.schema.coco | annotations: 12,430 total -> 11,714 kept, 716 dropped (class 716 [speed_limit 716])
```

`nc` drops from 4 to 3, `back/front/plate` keep their indices because `speed_limit` sorted last, and images that contained *only* speed-limit signs become background images. Had you excluded `back` instead, `front` would have moved from 1 to 0 and `plate` from 2 to 1 — an exclusion is a class-list change like any other.

### (e) A project that is missing a label the others have

Project 46 has only `back, plate, front` — no `speed_limit`. No configuration is needed; this is the case the derived map handles by construction:

```
proj 46 categories        : back(1), plate(2), front(3)
derived map (all sources) : 0:back, 1:front, 2:plate, 3:speed_limit
proj 46 conversion        : back -> 0, plate -> 2, front -> 1     (index 3 simply unused here)
```

Compare with the legacy path, where `category_id - 1` would have written `back → 0, plate → 1, front → 2` into the same directory as project 60's `plate → 0, front → 1, back → 2`.

The one thing to watch is the **pinned** variant of this scenario. If project 46 is your *only* source and you pin all four names, `speed_limit` keeps index 3 with zero instances — deliberate, and correct for a fine-tune. But zero-instance classes are invisible in a lot of ultralytics output: per-class arrays are indexed by `ap_class_index`, which only contains classes with ground truth in the split, so `speed_limit` will not appear in the per-class metric table at all. It is not missing; it has no data.

### (f) A label spelled two ways

Projects 59 and 60 spell it `speed_limit`; someone recreated the label in project 48 as `Speed_Limit`. Nothing needs configuring — `_key()` folds case and surrounding whitespace, so both map to the same index — but the run tells you, because CVAT does *not* treat them as the same label and one of the two spellings is about to disappear from `data.yaml`:

```
2026-07-31 09:18:03 | WARNING | src.data.class_map | label spelled 2 ways across sources: speed_limit x2, Speed_Limit x1 -- CVAT treats these as different labels, this pipeline does not
2026-07-31 09:18:03 | INFO | src.data.class_map | class map (derived): 4 classes -> 0:back, 1:front, 2:plate, 3:speed_limit
```

Most common wins, so `speed_limit` is the displayed name. Fix it in CVAT anyway — the tie-break is alphabetical, so adding one more project with the capitalised spelling would flip the name in `data.yaml` without changing any index.

---

## Common mistakes

1. **Leaving a test task in `task_ids_train` as well.** Harmless — the resolver subtracts it and says so — but if you see `0 excluded` where you expected 1, the id you typed is not the one in the project.
2. **Expecting `val_ratio` to be honoured without `test_ratio`.** It is not read as a size in that branch — the valid split is everything that is not training, `1 - train_ratio`. A `val_ratio` that disagrees is warned about, not applied.
3. **Ratios that do not sum to 1 with `test_ratio` set.** `0.8 / 0.2 / 0.1` sums to 1.1 and is rejected. Each ratio is that split's share of the dataset; write `0.7 / 0.2 / 0.1`.
4. **Configuring `test_ratio` *and* `task_ids_test`.** Refused — they are two ways of producing the same `test/` directory. Pick one.
5. **Spelling `class_exclude` with different case from CVAT.** The class map drops the name case-insensitively but the annotation filter matches case-sensitively, so the run dies with `UnknownClassError` naming a class you *did* exclude.
6. **Misspelling an attribute name in `attributes_exclude`.** Nothing is dropped, and the run completes — the only signal is one WARNING at the end of the data stage naming the attribute.
7. **Reading `area_segment_min` as a fraction.** It is square pixels at source resolution.
8. **Fine-tuning with a derived class order.** Adding or excluding one class renumbers everything after it alphabetically. Pin `class_names` to the checkpoint's `names`.
9. **Turning `unify_class_order` off to "make the old numbers come back".** It restores per-source `category_id - 1`, which is the bug, not a numbering scheme. Use it only to reproduce a specific historical run.
10. **Leaving a stale `s3_uri_dir_train` filled in.** Two sources is the same error as zero: `ValueError: source must be just 1`.
11. **Two CVAT tasks with the same name in the same project.** The extraction directory is keyed on project name plus task name, so one silently overwrites the other.
12. **Expecting the ClearML UI to parse a dict.** `attributes_exclude` has no string tolerance; the int lists, name lists and booleans do.

---

## See also

- [`0_console.md`](0_console.md) — `log_level` and `progress`. Set `log_level=debug` to get the per-annotation filter decisions referenced throughout this page.
- [`1_task.md`](1_task.md) — model name, `model_latest_id` (the checkpoint that makes pinning `class_names` necessary), project and task naming.
- [`3_augment.md`](3_augment.md) — augmentation applied to the images this stage produced.
- [`4_training.md`](4_training.md) — `epochs`, `batch`, `fraction`, and the rest of `model.train()`.
- [`5_testing.md`](5_testing.md) — validation thresholds, and the `split` that a dedicated test set switches to `test`.
- [`6_predict.md`](6_predict.md) — the prediction grids logged after training.
- [`7_export.md`](7_export.md) — export formats and parameters.
- [`8_visualization.md`](8_visualization.md) — including the Dataset report, which is where the class balance and box-area distribution produced by this stage are charted.
