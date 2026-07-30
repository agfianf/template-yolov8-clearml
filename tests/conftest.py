"""Shared fixtures.

The one that matters is `capture_src_logs`. pytest's own `caplog` captures
through a handler on the *root* logger, and `src/utils/logging.py` sets
`propagate = False` on the `src` logger on purpose -- so `caplog.records` is
always empty here and any "we emit N lines" assertion built on it passes
vacuously. Capture on the `src` logger directly instead.
"""

import logging

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest

from src.utils import logging as src_logging


class _ListHandler(logging.Handler):
    """Collects records instead of writing them anywhere."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _reset_logging_state() -> None:
    root = logging.getLogger(src_logging.ROOT_LOGGER_NAME)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.NOTSET)
    root.propagate = True
    src_logging._configured = False
    src_logging._progress_mode = "auto"
    src_logging._warned_values.clear()


@pytest.fixture
def reset_src_logging() -> Iterator[None]:
    """Drop the logging module's global handler/`_configured` state.

    Tests that flip `LOG_LEVEL` need the next `setup_logging()` to actually
    reconfigure rather than short-circuit on a leftover handler.
    """
    _reset_logging_state()
    yield
    _reset_logging_state()


@pytest.fixture
def capture_src_logs() -> Iterator[Callable[..., object]]:
    """Context manager yielding the records emitted under the `src` logger."""

    @contextmanager
    def _capture(
        level: int = logging.INFO,
        name: str = src_logging.ROOT_LOGGER_NAME,
    ) -> Iterator[list[logging.LogRecord]]:
        # Deliberately does not call setup_logging(): that would emit the
        # first-configuration output (e.g. the unreadable-LOG_LEVEL warning)
        # before the handler is attached, so tests would never see it.
        logger = logging.getLogger(name)
        handler = _ListHandler()
        handler.setLevel(level)
        previous_level = logger.level
        logger.setLevel(level)
        logger.addHandler(handler)
        try:
            yield handler.records
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

    return _capture
