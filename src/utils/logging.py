"""Logging configuration for the training pipeline.

Deliberately free of project imports. `src/config.py` builds a pydantic
`Settings` with six required fields and raises at import time when any of them
is missing, so reading the log level through it would make logging unusable in
exactly the runs that need it most. The level is read straight from
`os.environ`.

One handler is attached to the `src` logger and `propagate` is turned off.
ClearML calls `logging.basicConfig()` on the root logger from
`clearml/backend_interface/task/log.py`, which would make every line of ours
appear twice if we propagated.
"""

import logging
import os
import sys
import time

from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


ROOT_LOGGER_NAME = "src"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL = logging.INFO
ENV_VAR = "LOG_LEVEL"
_HANDLER_NAME = "src-console"
PROGRESS_MODES = ("auto", "on", "off")

_configured = False
_progress_mode = "auto"
# Values already complained about. get_logger() runs setup_logging() on every
# call, so without this a single unreadable LOG_LEVEL warns once per module.
_warned_values: set[str] = set()


def _parse_level(value: Any) -> int | None:
    """Turn a level name, a numeric string or an int into a logging level.

    Returns None when `value` carries no level -- either because it is empty
    (meaning "not set", not "invalid") or because it is unreadable. Callers
    distinguish the two.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value

    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return logging.getLevelNamesMapping().get(text.upper())


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def resolve_level(explicit: Any = None) -> int:
    """Resolve the effective log level: explicit > $LOG_LEVEL > INFO.

    An unreadable value falls back to INFO *and* warns. A silent fallback means
    debugging a run that quietly ignored the flag you set.
    """
    for source, raw in (
        ("explicit value", explicit),
        (f"${ENV_VAR}", os.environ.get(ENV_VAR)),
    ):
        if _is_blank(raw):
            continue
        level = _parse_level(raw)
        if level is not None:
            return level
        if f"{source}={raw!r}" not in _warned_values:
            _warned_values.add(f"{source}={raw!r}")
            logging.getLogger(f"{ROOT_LOGGER_NAME}.utils.logging").warning(
                "unreadable log level from %s: %r -- falling back to INFO", source, raw
            )
        return DEFAULT_LEVEL
    return DEFAULT_LEVEL


def setup_logging(level: Any = None) -> None:
    """Install the single `src` handler. Idempotent; safe to call from anywhere.

    Only the first call (or one that names a level) decides the level, so the
    `get_logger()` in every module does not keep re-reading the environment and
    undoing a later `set_log_level()`.
    """
    global _configured

    root = logging.getLogger(ROOT_LOGGER_NAME)
    already_configured = _configured
    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        root.handlers = [h for h in root.handlers if h.get_name() != _HANDLER_NAME]
        root.addHandler(handler)
        # Not negotiable: ClearML's basicConfig() on the root logger would
        # otherwise duplicate every line we emit.
        root.propagate = False
        root.setLevel(DEFAULT_LEVEL)
        _configured = True

    if not already_configured or not _is_blank(level):
        root.setLevel(resolve_level(level))


def set_log_level(level: Any) -> None:
    """Re-point the level after startup, e.g. once ClearML params are connected."""
    setup_logging()
    parsed = _parse_level(level)
    logging.getLogger(ROOT_LOGGER_NAME).setLevel(
        DEFAULT_LEVEL if parsed is None else parsed
    )


def set_progress_mode(mode: str | None) -> None:
    """Choose whether `progress()` draws bars: auto (TTY only), on, or off."""
    global _progress_mode

    normalized = (mode or "auto").strip().lower()
    if normalized not in PROGRESS_MODES:
        get_logger(__name__).warning(
            "unknown progress mode %r -- using 'auto' (one of %s)",
            mode,
            ", ".join(PROGRESS_MODES),
        )
        normalized = "auto"
    _progress_mode = normalized


def _main_module_name() -> str:
    """Best-effort module name for a script started as `__main__`."""
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    return Path(path).stem if path else "main"


def _normalize(name: str) -> str:
    """Map any logger name into the `src.` tree.

    `make run` executes src/train.py as a script, so `__name__` there is
    `__main__` -- without this its sixteen log lines would sit outside the tree
    the handler is attached to and stay unconfigured.
    """
    if name == ROOT_LOGGER_NAME or name.startswith(f"{ROOT_LOGGER_NAME}."):
        return name
    if name in {"__main__", "__mp_main__"}:
        name = _main_module_name()
    return f"{ROOT_LOGGER_NAME}.{name}"


def get_logger(name: str) -> logging.Logger:
    """Get a logger inside the configured `src` tree.

    Args:
        name: The logger name, typically __name__.

    Returns:
        Configured logger instance.

    """
    setup_logging()
    return logging.getLogger(_normalize(name))


def is_tty() -> bool:
    """Whether stdout is a terminal. Evaluated per call, never cached at import."""
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


class Tally:
    """Count occurrences per key, keeping the first few examples of each.

    The shape every per-item loop in the data stage collapses into: instead of
    one line per annotation or per file, the loop tallies and the caller emits a
    single summary. Line count then stops tracking dataset size.
    """

    def __init__(self, max_examples: int = 3) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: dict[str, list[str]] = {}
        self.max_examples = max_examples

    def add(self, key: str, example: Any = None) -> None:
        self.counts[key] += 1
        if example is not None:
            seen = self.examples.setdefault(key, [])
            if len(seen) < self.max_examples:
                seen.append(str(example))

    def total(self) -> int:
        return sum(self.counts.values())

    def __bool__(self) -> bool:
        return bool(self.counts)

    def __len__(self) -> int:
        return len(self.counts)

    def __getitem__(self, key: str) -> int:
        return self.counts[key]

    def summary(self, with_examples: bool = False) -> str:
        """`stalk 300, foreign_object 212`, biggest key first."""
        parts = []
        for key, count in self.counts.most_common():
            part = f"{key} {count:,}"
            if with_examples and self.examples.get(key):
                part += f" (e.g. {', '.join(self.examples[key])})"
            parts.append(part)
        return ", ".join(parts)


def progress[T](
    iterable: Iterable[T],
    desc: str,
    total: int | None = None,
    logger: logging.Logger | None = None,
) -> Iterator[T]:
    """Iterate with a tqdm bar on a TTY, and one summary line either way.

    `disable=None` is tqdm's own "only draw on a TTY" mode, so a redirected
    agent console gets the summary line and nothing else.
    """
    from tqdm import tqdm

    log = logger or get_logger(__name__)
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None

    disable = {"auto": None, "on": False, "off": True}[_progress_mode]
    count = 0
    started = time.monotonic()
    for item in tqdm(iterable, desc=desc, total=total, disable=disable):
        count += 1
        yield item
    log.info("%s: %d items in %.1fs", desc, count, time.monotonic() - started)
