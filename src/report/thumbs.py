"""Thumbnail production for the report galleries.

One tier only: a 192x192 JPEG at quality 80, 4:2:0, `optimize=True`, roughly 9 KB as
base64. A second, larger tier for the lightbox was measured at ~28 KB each, which at the
cap would be 5.6 MB on its own -- the entire size budget spent on a hover interaction.
The lightbox enlarges the same bytes and says so in its caption.

Two shapes, for two different jobs, and **neither carries a drawn box**:

* **Instance crops** (`focus` set) crop to the object, expanded and squared. The crop
  *is* the object, so a CSS ring in the outcome colour and a text badge carry the
  meaning, and stay crisp at any zoom.
* **Whole images** (`focus` unset) letterbox the frame and stop there. The overlay is an
  SVG layer drawn over these pixels from `grids[].items[].boxes`, which `blob.py::
  _norm_boxes` emits as thumbnail-normalised 0..1 floats.

An outline used to be baked into the whole-image JPEGs. It is gone because the gallery
now has one global annotations switch and three overlay tabs (outcome, prediction,
ground truth) over the *same* thumbnail: baked pixels can be neither hidden nor
re-coloured, so a switch over them is impossible. Vector boxes also stay sharp when the
lightbox enlarges the same bytes, which the 2px outline did not.

No pixels are held during validation: this runs afterwards, re-reading at most the cap's
worth of files from the dataset directory, which is still on disk (`cleanup_cache`
deletes only `labels.cache`).
"""

from __future__ import annotations
import base64
import hashlib
import io

from typing import Any

from src.utils.logging import Tally, get_logger


logger = get_logger(__name__)

# Expansion around an instance crop, and the floor below which a crop is unreadable.
CROP_SCALE = 1.6
MIN_CROP_PX = 64
JPEG_QUALITY = 80


class ThumbnailStore:
    """Base64 thumbnails, deduplicated by (file, crop) so shared items cost once."""

    def __init__(self, size: int = 192, max_thumbs: int = 200) -> None:
        self.size = int(size)
        self.max_thumbs = int(max_thumbs)
        self.thumbs: dict[str, str] = {}
        self.skipped = Tally()
        self._cache: dict[tuple, str | None] = {}

    @property
    def full(self) -> bool:
        return len(self.thumbs) >= self.max_thumbs

    def add(self, item: dict[str, Any]) -> str | None:
        """Encode one gallery item, returning its blob key (or None if unusable).

        Called once per gallery item; never logs. The caller emits one summary.
        """
        path = item.get("im_file")
        if not path:
            self.skipped.add("no image path")
            return None
        focus = item.get("focus")
        crop = self._crop_box(item, focus)
        cache_key = (path, crop, bool(focus))
        if cache_key in self._cache:
            return self._cache[cache_key]
        if self.full:
            self.skipped.add("thumbnail cap reached")
            self._cache[cache_key] = None
            return None

        key = hashlib.sha1(
            f"{path}|{crop}|{bool(focus)}".encode(), usedforsecurity=False
        ).hexdigest()[:12]
        if key in self.thumbs:
            self._cache[cache_key] = key
            return key

        try:
            data = self._encode(item, crop)
        except Exception as e:
            self.skipped.add(type(e).__name__, e)
            self._cache[cache_key] = None
            return None
        if data is None:
            self._cache[cache_key] = None
            return None
        self.thumbs[key] = data
        self._cache[cache_key] = key
        return key

    # --- internals ------------------------------------------------------------------
    def _crop_box(self, item: dict, focus: Any) -> tuple[int, int, int, int]:
        h, w = (int(v) for v in item.get("ori_shape") or (0, 0))
        if not focus or w <= 0 or h <= 0:
            return (0, 0, w, h)
        x0, y0, x1, y1 = (float(v) for v in focus[:4])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = max((x1 - x0), (y1 - y0)) * CROP_SCALE
        side = max(side, MIN_CROP_PX)
        side = min(side, float(max(w, h)))
        half = side / 2
        cx = min(max(cx, half), max(w - half, half))
        cy = min(max(cy, half), max(h - half, half))
        return (
            int(max(cx - half, 0)),
            int(max(cy - half, 0)),
            int(min(cx + half, w)),
            int(min(cy + half, h)),
        )

    def _encode(self, item: dict, crop: tuple[int, int, int, int]) -> str | None:
        from PIL import Image

        with Image.open(item["im_file"]) as src:
            img = src.convert("RGB")
        cx0, cy0, cx1, cy1 = crop
        if cx1 > cx0 and cy1 > cy0:
            img = img.crop((cx0, cy0, cx1, cy1))
        size = self.size
        canvas = Image.new("RGB", (size, size), (24, 24, 24))
        scale = min(size / max(img.width, 1), size / max(img.height, 1))
        new = (max(int(img.width * scale), 1), max(int(img.height * scale), 1))
        img = img.resize(new, Image.Resampling.BILINEAR)
        off = ((size - new[0]) // 2, (size - new[1]) // 2)
        canvas.paste(img, off)

        buf = io.BytesIO()
        canvas.save(
            buf,
            format="JPEG",
            quality=JPEG_QUALITY,
            optimize=True,
            subsampling=2,
        )
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def log_summary(self) -> None:
        """One line for the whole gallery build."""
        logger.info(
            "report thumbnails: %d encoded%s",
            len(self.thumbs),
            f", skipped {self.skipped.summary(with_examples=True)}"
            if self.skipped
            else "",
        )
