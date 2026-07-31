# `3_Augment` — `args_augment`

This group holds the thirteen data-augmentation hyperparameters that shape what the **training** dataloader hands the model: colour jitter, the random affine warp, the two flips, and the multi-image mixes (mosaic, mixup, copy-paste). It is defined as `args_augment` in `src/params.py`, published to the ClearML UI by `curr_task.connect(args_augment, name="3_Augment")` in `src/utils/clearml_settings.py`, and appears in the experiment's **Configuration → Hyperparameters → 3_Augment** section. Nothing in this group affects validation, testing, prediction or export: ultralytics builds the val/test datasets with augmentation off, so these keys only ever change the images the optimiser sees.

## How these values reach the model

`config_clearml()` connects the group, and then, on the line after the CVAT/class handling, does:

```python
args_train.update(args_logging)
args_train.update(args_augment)   # src/utils/clearml_settings.py
```

`src/train.py` splats the merged dict straight into the trainer:

```python
model_yolo.train(data=data_yaml_file, **args_train)
```

So every key documented here is delivered to ultralytics as an ordinary *training* argument; the split into a separate `3_Augment` UI group is purely presentational. No key in `args_augment` collides with a key in `args_train`, so the merge is additive and nothing is silently overwritten — but note the direction: **`args_augment` is applied last, so if a key were ever added to both dicts, the augmentation value would win.**

Two things that look like master switches but are not:

- **`args_train["augment"]` does not turn this group on or off.** In ultralytics' config `augment` means "apply test-time augmentation during prediction" (`cfg/default.yaml:68`), and the validator reads it as `augment = self.args.augment and (not self.training)` (`engine/validator.py:156`), so during a training run it is inert in both directions. Training-time augmentation is switched on by the dataloader being built in `train` mode, not by this flag. `src/train.py::_generate_data_yaml()` additionally forces `args_train["augment"] = False` whenever the YOLO task is `segment` — which, with the default `model_name: yolo11n-seg`, is every default run. That line changes TTA, not the augmentations below.
- **`args_train["rect"]` does silently disable three of them.** `data/dataset.py:310-312` sets `mosaic`, `mixup` and `cutmix` to `0.0` when `rect` is true, because rectangular batching and multi-image canvases are incompatible. It is `False` by default here; see [`4_training.md`](4_training.md).

## Quick reference

| Key | Default | Type / range | One-line meaning |
|---|---|---|---|
| `hsv_h` | `0.015` | float, 0–1 (fraction of the hue circle) | Random hue shift, ±`hsv_h`×180 on OpenCV's 0–179 hue scale. |
| `hsv_s` | `0.7` | float, 0–1 (gain) | Saturation multiplied by a random factor in `[1-hsv_s, 1+hsv_s]`. |
| `hsv_v` | `0.4` | float, 0–1 (gain) | Brightness (HSV value) multiplied by a random factor in `[1-hsv_v, 1+hsv_v]`. |
| `degrees` | `25.0` | float, degrees | Random in-plane rotation, uniform in ±`degrees`. Upstream default is `0.0`. |
| `translate` | `0.1` | float, fraction of image size | Random shift, uniform in ±`translate`×`imgsz` pixels on each axis. |
| `scale` | `0.5` | float, gain | Random zoom, uniform in `[1-scale, 1+scale]`. |
| `shear` | `0.0` | float, degrees | Random shear on both axes, uniform in ±`shear`. |
| `perspective` | `0.0` | float, 0–0.001 typical | Random perspective warp coefficient. |
| `flipud` | `0.2` | float, probability 0–1 | Chance of a vertical (top-bottom) mirror. Upstream default is `0.0`. |
| `fliplr` | `0.5` | float, probability 0–1 | Chance of a horizontal (left-right) mirror. |
| `mosaic` | `1.0` | float, probability 0–1 | Chance of composing 4 images into one canvas before the affine warp. |
| `mixup` | `0.0` | float, probability 0–1 | Chance of alpha-blending two composed samples and unioning their labels. |
| `copy_paste` | `0.0` | float, probability 0–1 | Segmentation-only: paste instance masks between images. Inert on a detection model. |

Only `degrees` and `flipud` differ from ultralytics' own defaults (`cfg/default.yaml:117-134`); everything else in this group is the upstream value. Both deviations are argued about in [Recommendations](#recommendations-for-the-license-plate--vehicle-frontback-dataset) below.

## The pipeline, in the order it actually runs

`ultralytics/data/augment.py::v8_transforms()` composes the training transform in this order, and the order matters when you reason about what a knob does:

1. `Mosaic(p=hyp.mosaic)` — four images tiled into a 2×`imgsz` canvas.
2. `CopyPaste(p=hyp.copy_paste)` — inserted between mosaic and the affine when `copy_paste_mode` is `flip` (the default).
3. `RandomPerspective(degrees, translate, scale, shear, perspective)` — one affine/perspective matrix, applied to the mosaic canvas and cropping it back down to `imgsz`.
4. `MixUp(p=hyp.mixup)`.
5. `CutMix(p=hyp.cutmix)` — not exposed in this group; stays at the upstream default `0.0`.
6. `Albumentations(p=1.0)` — see below.
7. `RandomHSV(hsv_h, hsv_s, hsv_v)`.
8. `RandomFlip(direction="vertical", p=flipud)`.
9. `RandomFlip(direction="horizontal", p=fliplr)`.

Consequences of that ordering that are easy to get wrong:

- **The affine is applied to the mosaic canvas, not to the original image.** With `mosaic=1.0`, the four tiles are already at roughly half scale before `scale` is sampled, so `scale=0.5` on a mosaic sample is not the same apparent zoom range as `scale=0.5` on a plain image.
- **The colour jitter happens after the mixes**, so a mosaic tile and its neighbours in the same canvas share one hue/saturation/brightness sample rather than being jittered independently.
- **The two flips are independent draws.** With `flipud=0.2` and `fliplr=0.5`, about 10% of samples get both, which is a 180° rotation.
- **Albumentations is active in this image.** `albumentations==2.0.8` is a pinned dependency (`pyproject.toml`), and when no custom `augmentations` list is passed ultralytics installs `Blur(p=0.01)`, `MedianBlur(p=0.01)`, `ToGray(p=0.01)` and `CLAHE(p=0.01)`. That is where the occasional grayscale or blurred tile in the `Mosaic` debug gallery comes from; it is not configurable from this group.

## Photometric: `hsv_h`, `hsv_s`, `hsv_v`

`RandomHSV` draws one vector `r = uniform(-1, 1, 3) * [hsv_h, hsv_s, hsv_v]` per image and builds three 256-entry lookup tables from it. Hue is **additive** on OpenCV's 0–179 scale (`lut_hue = (x + r[0]*180) % 180`), while saturation and value are **multiplicative and clipped** (`clip(x * (1 + r), 0, 255)`).

- `hsv_h=0.015` shifts the hue by at most ±2.7 units on the 0–179 scale, i.e. ±5.4° around the colour wheel. That is deliberately tiny: hue is the one channel that carries class-discriminative colour (plate background colour, tail-lights red versus head-lights white), and a large value turns a red rear lamp cluster into an amber one. Raising this above ~0.05 is how you teach a model that colour means nothing.
- `hsv_s=0.7` scales saturation between 0.3× and 1.7×, so the pipeline routinely produces near-grayscale and over-saturated versions of every image. This is the cheapest defence against camera white-balance differences between sites and against night-time IR footage, which is effectively monochrome.
- `hsv_v=0.4` scales brightness between 0.6× and 1.4×. This is the knob for exposure variation — dusk, overcast, headlight glare, an auto-exposure camera hunting. Lowering it towards 0 on a dataset collected at one time of day is how a model ends up failing at night; raising it past ~0.6 starts producing clipped-white or crushed-black images whose small objects have no gradient left.

What goes wrong if set wrong: all three are cheap (a LUT apply) so they never cost throughput, and their failure mode is silent — the model simply does not generalise to a lighting condition that was never simulated, or learns to ignore a colour cue that actually mattered.

## Geometric: `degrees`, `translate`, `scale`, `shear`, `perspective`

All five feed one `RandomPerspective` transform, which composes a single 3×3 matrix per image (`data/augment.py:1131-1152`):

- `degrees` → `a = uniform(-degrees, degrees)`, an in-plane rotation about the image centre.
- `translate` → each axis is offset by `uniform(0.5-translate, 0.5+translate) * size`, i.e. a shift of up to ±`translate` of the output size.
- `scale` → `s = uniform(1-scale, 1+scale)`, so `0.5` means anything from half size to 1.5×.
- `shear` → `tan(uniform(-shear, shear))` on both axes.
- `perspective` → the two bottom-row coefficients, each `uniform(-perspective, perspective)`. The useful range really is 0–0.001; values an order of magnitude larger fold the image onto itself.

The one non-obvious cost is **rotation and box tightness**. YOLO labels are axis-aligned. When `RandomPerspective` rotates an image it recomputes each box as the axis-aligned bounding box of the rotated corners, so a rotated object's label grows: an elongated, near-horizontal object such as a licence plate rotated by 25° gets a label noticeably larger than the object, filled with background. Train on enough of those and the model learns to predict loose boxes at inference on unrotated images too, which shows up as a mAP50-95 that lags mAP50 and as crops that include the bumper next to the plate. Rotation of a thin, wide object is the worst case for this effect; rotation of a roughly square object is nearly free.

`translate` and `scale` have no equivalent penalty — they are label-exact — which is why they are the two geometric knobs to reach for first.

## Flips: `flipud`, `fliplr`

Both are probabilities, applied as the last two steps, and both simply mirror the image and its labels. They cost nothing and they are the augmentations most likely to be *wrong for a domain rather than merely aggressive*, because a mirror is only free when the object class is genuinely mirror-symmetric.

- `fliplr=0.5` is the upstream default and is right for most detection datasets.
- `flipud=0.2` is **not** an upstream default (`cfg/default.yaml:125` is `0.0`). Upside-down images are correct augmentation for overhead/satellite imagery and for anything with no gravity cue; they are wrong for ground-level camera footage, where nothing is ever upside down.

For pose models, ultralytics disables both flips unless `flip_idx` is present in `data.yaml`, and warns when it does so. This template trains detection and segmentation models, so that path does not apply.

## Multi-image mixes: `mosaic`, `mixup`, `copy_paste`

**`mosaic` (default `1.0`)** composes four images into one 2×`imgsz` canvas, which the affine step then warps and crops back to `imgsz`. It is the single most powerful augmentation in the YOLO recipe: it multiplies the effective object count per batch, creates unnatural context, and — because each tile ends up around half scale — it is largely responsible for YOLO's small-object performance. It is also the augmentation that most changes the training distribution: at `1.0`, *every* training image the model ever sees is a four-way collage with cropped objects at its seams, which is nothing like a validation image. That is what `close_mosaic` exists to fix; see the next section.

**`mixup` (default `0.0`)** alpha-blends two fully composed samples and takes the union of their labels, so the model sees ghosted objects at partial opacity. It is a strong regulariser for large classification-heavy datasets and generally hurts small-object detection, where a half-transparent object is not a thing that exists. `0.0` is the right default; if a run is clearly overfitting and everything else is exhausted, `0.05-0.15` is the range worth trying.

**`copy_paste` (default `0.0`) is segmentation-only, and inert otherwise.** This is not a style preference, it is a hard guard in the code: `CopyPaste.__call__` starts with `if len(labels["instances"].segments) == 0 or self.p == 0: return labels` (`data/augment.py`). A detection dataset has no segments, so any non-zero value you set here is silently ignored on a detect model. On a genuine segmentation run it pastes instance masks from a horizontally flipped copy of the same image (`copy_paste_mode` defaults to `flip`, and is not exposed in this group), keeping only instances whose box-IoA with existing instances is below 0.30.

## `mosaic` and `close_mosaic`: one mechanism, two UI groups

`close_mosaic` lives in `args_train` and therefore appears under [`4_training.md`](4_training.md), but it is meaningless without `mosaic` and belongs in the same mental model.

The mechanism, from `ultralytics/engine/trainer.py`:

```python
if self.args.close_mosaic:                                  # :418  falsy 0 == never
    base_idx = (self.epochs - self.args.close_mosaic) * nb
...
if epoch == (self.epochs - self.args.close_mosaic):         # :436
    self._close_dataloader_mosaic()
    self.train_loader.reset()
```

and `_close_dataloader_mosaic()` calls `dataset.close_mosaic(hyp)`, which sets `mosaic`, `copy_paste`, `mixup` **and** `cutmix` to `0.0` and rebuilds the transform pipeline (`data/dataset.py:360`). So `close_mosaic` is not just about mosaic — it is the switch that turns off *every multi-image mix* for the final N epochs, letting the model finish on images that look like the ones it will be validated and deployed on.

Three facts about the pair:

- **This repository sets `close_mosaic: 0`, whereas ultralytics defaults to `10`.** With `0`, the `if self.args.close_mosaic:` guard is falsy and the branch never fires: mosaic stays at `1.0` through the last epoch, and the final weights are the weights of a model that has literally never trained on a single un-collaged image. The `best.pt` selected against the un-augmented validation split is the only thing keeping this honest.
- `close_mosaic` counts *epochs from the end*, so it must be smaller than `epochs`. With `epochs: 20` and `close_mosaic: 10`, the mixes stop at epoch 10.
- On resume, the trainer re-checks `if start_epoch > (self.epochs - self.args.close_mosaic): self._close_dataloader_mosaic()`, so resuming into the tail of a run correctly starts with the mixes already off.

**Recommendation:** if `mosaic` is above ~0.5, set `close_mosaic` to roughly 10-20% of `epochs` (and at least 3, since the trainer also uses that window to write its final `train_batch*.jpg` plots). Leaving `close_mosaic: 0` while `mosaic: 1.0` is the combination with the worst train/deploy mismatch.

## Keys ultralytics has that this group does not expose

These stay at the upstream default for every run and cannot be changed from the ClearML UI without editing `src/params.py`: `cutmix` (`0.0`), `bgr` (`0.0`, RGB↔BGR channel swap), `copy_paste_mode` (`flip`), `multi_scale` (`0.0`), `auto_augment` (`randaugment`, classification only), `erasing` (`0.4`, classification only), and `augmentations` (the custom Albumentations list). If you need one of them, add it to `args_augment` — it will be connected to the UI and merged into the train call automatically, with no other code change.

## Recommendations for the licence-plate / vehicle-front/back dataset

> **This section is a recommendation, not repository fact.** Everything above is traceable to `src/params.py` or to the ultralytics source in the pinned image. What follows is reasoning about a specific domain — detecting licence plates and vehicle front/back regions in footage from *fixed* cameras — and should be tested with a run, not adopted on faith. The defaults currently in `src/params.py` are the ones described in the table above; none of the changes below have been applied.

The domain has three properties that matter for augmentation choices: the camera does not move, so the roll angle of the scene is fixed and the perspective is fixed; the objects of interest are small, thin and wide; and one of the classes (the plate) is a *text-bearing* region whose whole downstream purpose is to be read.

**`flipud: 0.2` — recommend `0.0`.** A vertical flip turns every scene upside down. There is no fixed camera on earth that produces that image, so 20% of training compute is spent on a pose that has zero probability at inference. Worse for this dataset specifically: it makes plate glyphs upside-down-mirrored, and it destroys the up/down cue that separates a vehicle front from a vehicle back (lamp cluster geometry, grille below versus above the bumper line). This is the single clearest change to make, and it costs nothing: dropping it strictly increases the fraction of realistic training samples. Note that upstream's own default is `0.0`; the `0.2` here is a local deviation.

**`fliplr: 0.5` — recommend keeping `0.5` for a pure detector, but `0.0` if the same crops feed a character recogniser.** A horizontal mirror is label-preserving for this task (a mirrored car front is still a car front, a mirrored plate is still a plate at the same location), so as pure box-localisation augmentation it is free variety and worth keeping. The caveat is that it produces mirrored text, and if the plate crops produced by this model are ever used to train or fine-tune an OCR stage, or if the plate *class* is split by regional plate layout, mirroring becomes actively harmful. For plate localisation alone, keep it.

**`degrees: 25.0` — recommend `5.0-10.0`.** This is the change with the largest expected effect. A fixed camera has a fixed roll; real in-image rotation comes from vehicles on a cambered road, a slightly untrue camera mount, and motorcycles leaning — a few degrees, not 25. Setting ±25° buys robustness to poses that never occur, and pays for it twice: it spends a large share of the augmentation budget outside the deployment distribution, and, because YOLO labels are axis-aligned, it systematically inflates the label boxes of exactly the thin wide objects this dataset is made of. A 520×110 plate rotated 25° has an axis-aligned bounding box roughly 1.3× wider and 3× taller than the plate itself, most of it bumper. Train on that and the model's plate boxes get loose, which costs mAP50-95 and produces crops with neighbouring clutter. `5.0-10.0` covers the real variation with a fraction of the label damage. Upstream's default is `0.0`.

**`scale: 0.5` — keep, or raise to `0.5-0.7`.** This one is well matched to the domain and is the *most* useful geometric knob here. Vehicles approach and recede, so apparent object size varies by an order of magnitude between the far end of the field of view and the near end, and `scale` is the label-exact way to simulate that. If the deployment cameras vary in mounting height or focal length between sites, raise it.

**`translate: 0.1` — keep, or raise to `0.15-0.2`.** Also label-exact and also well matched: vehicles appear anywhere across the lane. There is no reason to be shy with this one.

**`perspective: 0.0` — consider `0.0-0.0005`, low priority.** Each fixed camera has one fixed perspective, so within a site this augmentation simulates nothing. Across sites — different mounting angles and heights — a small value is a cheap way to make the model less dependent on the exact geometry it was trained on. Worth trying only after the rotation and flip changes, and only if the model is going to new camera installations.

**`shear: 0.0` — keep.** Shear is not a transformation any camera geometry produces here, and like rotation it loosens axis-aligned labels.

**`hsv_v: 0.4` — keep, or raise to `0.5` if the deployment runs at night.** Fixed-camera traffic footage spans full daylight, dusk, sodium/LED street lighting, headlight wash and IR illumination. Brightness variation is the most valuable photometric augmentation for this domain, and it is not the one to trim.

**`hsv_s: 0.7` — keep.** IR night footage is effectively monochrome, and a saturation range that reaches 0.3× teaches the model not to depend on colour, which is exactly right if the same weights must serve day and night cameras.

**`hsv_h: 0.015` — keep small.** Plate background colour can be class- or region-informative, and lamp colour separates front from back. Do not raise this.

**`mosaic: 1.0` — keep `1.0` but pair it with `close_mosaic`.** Mosaic is genuinely useful here: plates are small objects, and mosaic's implicit downscaling plus increased object density is the standard reason YOLO does well on small objects. The problem is not mosaic, it is mosaic *for every epoch to the very end*, which is what `close_mosaic: 0` produces. Set `close_mosaic` to about 10-20% of `epochs`. If plates in your footage are already near the minimum detectable size, consider `mosaic: 0.7-1.0` so that some fraction of epochs sees full-resolution plates throughout.

**`mixup: 0.0` — keep.** Ghosted, half-transparent vehicles are not a failure mode this dataset needs to cover, and blending is particularly bad for small text-bearing regions.

**`copy_paste: 0.0` — keep, and be aware it does nothing on a detection model.** If the run is a segmentation run (`yolo11n-seg`, the default `model_name`), `0.1-0.3` is a legitimate way to increase instance density for rare classes.

A consolidated starting point for this domain, expressed as the values to type into the `3_Augment` group (plus one key from `4_Training`):

```
# 3_Augment
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.5
degrees: 8.0
translate: 0.15
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0

# 4_Training
close_mosaic: 4     # with epochs: 20
```

## Scenarios

### Scenario 1 — Fast smoke run: prove the pipeline, not the model

You want to confirm that CVAT export, class mapping, training, validation, export and prediction all run end to end, in the shortest wall-clock time. Augmentation is pure cost here: it adds CPU work per image and adds variance to metrics that you are not going to read anyway.

```
# 3_Augment
hsv_h: 0.0
hsv_s: 0.0
hsv_v: 0.0
degrees: 0.0
translate: 0.0
scale: 0.0
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.0
mosaic: 0.0
mixup: 0.0
copy_paste: 0.0

# 4_Training (see 4_training.md)
epochs: 2
fraction: 0.05
```

Expected consequences: the `Mosaic` image gallery reported at epoch 0/1 by `on_train_epoch_end` will show plain letterboxed images rather than four-way collages — which is itself the fastest way to confirm the settings took effect. Metrics will be poor and must not be compared against a real run. `close_mosaic` is irrelevant because `mosaic` is already 0.

### Scenario 2 — Small dataset (a few hundred images), model overfits within 10 epochs

Training mAP climbs while validation mAP peaks early and falls. With a small dataset the answer is more augmentation, but chosen so it does not distort labels.

```
# 3_Augment
hsv_s: 0.8
hsv_v: 0.5
degrees: 8.0
translate: 0.2
scale: 0.6
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.1

# 4_Training
epochs: 100
close_mosaic: 15
patience: 25
```

Expected consequences: each epoch produces a more varied batch, so the train/val loss gap narrows and the validation peak moves later. `mixup: 0.1` blends 10% of samples — watch the `Losses/Balance` scalars reported by `_report_loss_ratios`; if the classification loss ratio climbs without validation mAP improving, mixup is confusing the classifier and should go back to `0.0`. `patience: 25` (see [`4_training.md`](4_training.md)) is what turns the longer schedule into a safe one.

### Scenario 3 — Long production run on fixed-camera traffic footage

The domain-tuned configuration from the recommendations section, over a real schedule.

```
# 3_Augment
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.5
degrees: 8.0
translate: 0.15
scale: 0.5
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0

# 4_Training
epochs: 150
close_mosaic: 20
patience: 40
```

Expected consequences: from epoch 130 onwards the mixes switch off and the training distribution becomes the deployment distribution; expect a visible step improvement in validation mAP at that epoch on the `Metrics/mAP` scalar, and expect box tightness (mAP50-95 relative to mAP50) to improve more than mAP50 does. Because `flipud` is off and `degrees` is moderate, the `Error Analysis` worst-image gallery reported once per validation pass should show genuine hard cases (occlusion, motion blur, extreme angles) rather than augmentation artefacts.

### Scenario 4 — "The training images look wrong"

Any time you suspect the augmentation is doing something unintended, do not reason about it — look. `on_train_epoch_end` in `src/yolov8/callbacks.py` uploads `train_batch*.jpg` under the ClearML title **Mosaic** for epochs 0 and 1, which are exactly the ultralytics-rendered training batches with labels drawn on. Upside-down cars, plates rotated far past anything the camera could see, or labels visibly larger than the objects they belong to are all diagnosable from that one gallery. If `close_mosaic` is non-zero, the trainer also queues three more plot indices at the epoch where the mixes close (`trainer.py:418-420`), so a second set of batch images appears there showing the post-mosaic distribution.

## See also

- [`0_console.md`](0_console.md) — log level and progress bars; at `LOG_LEVEL=debug` ultralytics' own per-item output comes back.
- [`1_task.md`](1_task.md) — `model_name` decides whether the run is `detect` or `segment`, which decides whether `copy_paste` does anything at all.
- [`2_data.md`](2_data.md) — the dataset these augmentations are applied to, and the class list they must not distort.
- [`4_training.md`](4_training.md) — `close_mosaic`, `rect`, `augment` and `epochs`, all of which change what this group does.
- [`5_testing.md`](5_testing.md) — validation runs with augmentation off; that is why val metrics and train losses are not directly comparable.
- [`6_predict.md`](6_predict.md) — the post-training prediction gallery, also un-augmented.
- [`7_export.md`](7_export.md) — export sees the trained weights only; nothing here affects it.
- [`8_visualization.md`](8_visualization.md) — which reports are produced, including the `Mosaic` debug gallery referenced above.
